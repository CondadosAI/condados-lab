"""QVHighlights through the same comparison: one query per video, scored inside its own video.

The basketball and MEVA videos are one long timeline with many event classes. QVHighlights is
the other shape: many 150 s videos, each with its own query and the spans that answer it. So
every number here is per query, averaged over queries, and chance is measured per video.

Steps (the sample is qvh_sample.json, from fetch_qvh.py):

- `upload`: every sampled video into one TwelveLabs index, `egbm-qvh` (Marengo, visual + audio).
- `embed`: Marengo's segment vectors per video, and its text vectors for the queries:
  index/qvh_marengo_visual.npz.
- `gemma`: EmbeddingGemma 2 on the same segments (8 frames each, as embed_windows.py
  --segments does): index/qvh_gemma_visual.npz. Needs a GPU; run it on the desktop.
- `search`: the hosted search, filtered to the query's own video, top result only.
- `evaluate`: for each index file, per query:
  - AP: the video's segments ranked by the query; a segment is positive when it overlaps a
    relevant window. Chance = the share of positive segments.
  - Hit@1: the top segment overlaps a relevant window. Chance = the same share.
  - R1@0.5: one predicted span, scored by IoU >= 0.5 with any relevant window. The span is the
    top segment grown over its neighbours while their score stays above the midpoint between
    the video's best and median score; the same rule for both models. Chance = the rule run on
    shuffled scores. The paper's zero-shot CLIP baseline gets 16.9 on the test split.

    uv run python qvh.py upload
    uv run python qvh.py embed
    uv run python qvh.py gemma --device cuda      # on the desktop
    uv run python qvh.py search
    uv run python qvh.py evaluate
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA, INDEX, OUT = HERE / "data" / "qvh", HERE / "index", HERE / "output"
MODEL, TAG = "marengo3.5", "qvh"
RNG = np.random.default_rng(0)


def sample() -> dict:
    return json.loads((HERE / "qvh_sample.json").read_text())


def client():
    from twelvelabs import TwelveLabs

    return TwelveLabs(api_key=os.environ["TWELVE_API_KEY"])


def assets_file() -> Path:
    return DATA / "assets.json"


def upload(c) -> None:
    idx = next((i for i in c.indexes.list() if i.index_name == f"egbm-{TAG}"), None)
    if idx is None:
        idx = c.indexes.create(
            index_name=f"egbm-{TAG}",
            models=[{"model_name": MODEL, "model_options": ["visual", "audio"]}],
        )
    done = json.loads(assets_file().read_text()) if assets_file().exists() else {}
    t0 = time.time()
    for vid in sample()["videos"]:
        if vid in done:
            continue
        asset = c.multipart_upload.upload_file(str(DATA / f"{vid}.mp4")).asset_id
        while (a := c.assets.retrieve(asset)).status != "ready":
            if a.status == "failed":
                raise RuntimeError(f"{vid}: asset failed")
            time.sleep(5)
        done[vid] = c.indexes.indexed_assets.create(index_id=idx.id, asset_id=asset).id
        assets_file().write_text(json.dumps(done, indent=1))
    for vid, ia in done.items():
        while (s := c.indexes.indexed_assets.retrieve(idx.id, ia).status) != "ready":
            if s == "failed":
                raise RuntimeError(f"{vid}: indexing failed")
            time.sleep(10)
    print(f"{len(done)} videos indexed in egbm-{TAG}, {time.time() - t0:.0f} s")


def index_id(c) -> str:
    return next(i for i in c.indexes.list() if i.index_name == f"egbm-{TAG}").id


def texts() -> list[str]:
    return sorted({q["query"] for q in sample()["queries"]})


def embed(c) -> None:
    from twelvelabs import MultiInputRequest

    ids, done = index_id(c), json.loads(assets_file().read_text())
    tl = texts()
    tv = []
    for t in tl:
        r = c.embed.v_2.create(
            input_type="multi_input", model_name=MODEL, multi_input=MultiInputRequest(input_text=t)
        )
        tv.append(r.data[0].embedding)
    tv = np.array(tv, np.float32)
    tv /= np.linalg.norm(tv, axis=-1, keepdims=True)
    for option in ("visual", "audio"):
        vids, start, end, emb = [], [], [], []
        for vid in sample()["videos"]:
            v = c.indexes.indexed_assets.retrieve(ids, done[vid], embedding_option=[option])
            segs = [s for s in v.embedding.video_embedding.segments if s.embedding_scope == "clip"]
            segs.sort(key=lambda s: s.start_offset_sec)  # the API returns them unsorted
            vids += [vid] * len(segs)
            start += [s.start_offset_sec for s in segs]
            end += [s.end_offset_sec for s in segs]
            emb += [s.float_ for s in segs]
        e = np.array(emb, np.float32)
        e /= np.linalg.norm(e, axis=-1, keepdims=True)
        out = INDEX / f"{TAG}_marengo_{option}.npz"
        INDEX.mkdir(exist_ok=True)
        np.savez(
            out,
            vid=np.array(vids),
            start=np.array(start),
            end=np.array(end),
            emb=e,
            texts=np.array(tl),
            text_emb=tv,
        )
        mean = np.mean(np.subtract(end, start))
        print(f"{len(e)} {option} segments, {mean:.1f} s mean -> {out.name}")


def gemma(device: str, frames: int = 8, tokens: int = 140) -> None:
    import torch
    from embed_windows import MODEL_ID, pooled, sampled, segment_clips
    from evaluate import embed_texts
    from transformers import AutoModel, AutoProcessor

    d = np.load(INDEX / f"{TAG}_marengo_visual.npz")
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval().to(device)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    out_emb = None
    t0 = time.perf_counter()
    for vid in sample()["videos"]:
        rows = np.where(d["vid"] == vid)[0]
        segs = np.stack([d["start"][rows], d["end"][rows]], axis=1)
        clips = list(segment_clips(sampled(DATA / f"{vid}.mp4", 4, 0.0, None), segs, frames))
        for i in range(0, len(clips), 2):
            batch = clips[i : i + 2]
            x = proc(
                text=["<|video|>"] * len(batch),
                videos=batch,
                do_sample_frames=False,
                max_soft_tokens=tokens,
                padding=True,
                return_tensors="pt",
            )
            x = {k: v.to(device) if hasattr(v, "to") else v for k, v in x.items()}
            h = model(**x).last_hidden_state.float().cpu().numpy()
            v = pooled(h, x["attention_mask"].cpu().numpy())
            if out_emb is None:
                out_emb = np.full((len(d["vid"]), v.shape[1]), np.nan, np.float32)
            out_emb[rows[i : i + len(batch)]] = v
    del model
    tl = texts()
    tv = embed_texts(tl, device)
    out = INDEX / f"{TAG}_gemma_visual.npz"
    np.savez(
        out,
        vid=d["vid"],
        start=d["start"],
        end=d["end"],
        emb=out_emb,
        texts=np.array(tl),
        text_emb=tv,
    )
    print(f"{len(out_emb)} segments in {time.perf_counter() - t0:.0f} s -> {out.name}")


def overlaps(s: float, e: float, wins: list) -> bool:
    return any(s < b and e > a for a, b in wins)


def iou(s: float, e: float, a: float, b: float) -> float:
    inter = max(0.0, min(e, b) - max(s, a))
    return inter / (max(e, b) - min(s, a))


def grow(sc: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    """The top segment, grown over neighbours above the midpoint of the best and median scores."""
    i = int(np.argmax(sc))
    thr = (sc[i] + np.median(sc)) / 2
    lo = hi = i
    while lo > 0 and sc[lo - 1] >= thr:
        lo -= 1
    while hi < len(sc) - 1 and sc[hi + 1] >= thr:
        hi += 1
    return float(start[lo]), float(end[hi])


def ap(sc: np.ndarray, pos: np.ndarray) -> float:
    o = np.argsort(-sc)
    h = pos[o]
    prec = np.cumsum(h) / np.arange(1, len(h) + 1)
    return float((prec * h).sum() / h.sum())


def evaluate(paths: list[str]) -> None:
    qs = sample()["queries"]
    lines = [
        f"# QVHighlights val sample: {len(qs)} queries, {len(sample()['videos'])} videos",
        "",
        "| model | AP | chance AP | Hit@1 | chance Hit@1 | R1@0.5 | chance R1@0.5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for path in paths:
        d = np.load(path)
        tv = dict(zip(d["texts"].tolist(), d["text_emb"], strict=True))
        r = {k: [] for k in ("ap", "cap", "hit", "chit", "r1", "cr1")}
        for q in qs:
            rows = np.where(d["vid"] == q["vid"])[0]
            rows = rows[~np.isnan(d["emb"][rows]).any(axis=1)]
            s, e = d["start"][rows], d["end"][rows]
            sc = d["emb"][rows] @ tv[q["query"]]
            wins = q["relevant_windows"]
            pos = np.array([overlaps(a, b, wins) for a, b in zip(s, e, strict=True)])
            if not pos.any():
                continue
            r["ap"].append(ap(sc, pos))
            r["cap"].append(pos.mean())
            r["hit"].append(float(pos[np.argmax(sc)]))
            r["chit"].append(pos.mean())
            ps, pe = grow(sc, s, e)
            r["r1"].append(float(max(iou(ps, pe, a, b) for a, b in wins) >= 0.5))
            hits = []
            for _ in range(100):
                ps, pe = grow(RNG.permutation(sc), s, e)
                hits.append(max(iou(ps, pe, a, b) for a, b in wins) >= 0.5)
            r["cr1"].append(float(np.mean(hits)))
        m = {k: 100 * np.mean(v) for k, v in r.items()}
        lines.append(
            f"| {Path(path).stem} | {m['ap']:.1f} | {m['cap']:.1f} | {m['hit']:.1f} "
            f"| {m['chit']:.1f} | {m['r1']:.1f} | {m['cr1']:.1f} |"
        )
    hosted = OUT / f"{TAG}_twelvelabs_search.json"
    if hosted.exists():
        h = json.loads(hosted.read_text())
        for opt, res in h.items():
            hit = [
                float(overlaps(x["start"], x["end"], x["wins"]))
                for x in res
                if x["start"] is not None
            ]
            r1 = [
                float(max(iou(x["start"], x["end"], a, b) for a, b in x["wins"]) >= 0.5)
                for x in res
                if x["start"] is not None
            ]
            lines.append(
                f"| hosted search, {opt} | | | {100 * np.mean(hit):.1f} | "
                f"| {100 * np.mean(r1):.1f} | |"
            )
    lines += [
        "",
        "Percent, mean over queries. Hit@1 and R1@0.5 use the top result; chance per video.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{TAG}_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def search(c) -> None:
    ids, done = index_id(c), json.loads(assets_file().read_text())
    res = {}
    for opts in (["visual"], ["visual", "audio"]):
        rows = []
        for q in sample()["queries"]:
            items = c.search.query(
                index_id=ids,
                query_text=q["query"],
                search_options=opts,
                filter=json.dumps({"id": [done[q["vid"]]]}),
                page_limit=5,
            )
            top = next(iter(items), None)
            rows.append(
                {
                    "qid": q["qid"],
                    "start": top.start if top else None,
                    "end": top.end if top else None,
                    "wins": q["relevant_windows"],
                }
            )
        res["+".join(opts)] = rows
    OUT.mkdir(exist_ok=True)
    (OUT / f"{TAG}_twelvelabs_search.json").write_text(json.dumps(res, indent=1))
    print({k: sum(r["start"] is None for r in v) for k, v in res.items()}, "queries with no result")


def main() -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("step", choices=["upload", "embed", "gemma", "search", "evaluate"])
    ap_.add_argument("--device", default="cpu")
    ap_.add_argument("indexes", nargs="*", help="evaluate: index files (default: both models)")
    args = ap_.parse_args()
    if args.step == "gemma":
        gemma(args.device)
    elif args.step == "evaluate":
        evaluate(
            args.indexes or [str(INDEX / f"{TAG}_{m}_visual.npz") for m in ("marengo", "gemma")]
        )
    else:
        {"upload": upload, "embed": embed, "search": search}[args.step](client())


if __name__ == "__main__":
    main()
