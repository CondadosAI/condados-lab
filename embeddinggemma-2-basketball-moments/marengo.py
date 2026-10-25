"""The same game, the same queries, the same ground truth, on TwelveLabs (Marengo).

Two comparisons, because TwelveLabs is both a model and a product:

- `embed`: Marengo's own segment vectors (2-10 s, cut at scene changes) and its text vectors,
  written as an index file that evaluate.py scores exactly like the EmbeddingGemma 2 windows.
  Model against model.
- `search`: the hosted search endpoint, as a user would call it. Each class takes its top-k
  results (k = its number of moments) and counts the moments a result overlaps, widened by
  delta. Chance places k clips of the same lengths at random inside the halves, 200 times.
  Product against product: it also uses the audio (crowd, whistle, commentary) when asked.

The video is uploaded and indexed first (an index named `egbm-<game>[-<variant>]`, Marengo,
visual + audio); the index is found by name, so no account id lives here. Needs TWELVE_API_KEY.
Upload a file with its audio track: yt-dlp's 720p stream is video only, so mux the original
audio in first. `--variant mask` is the same game with the score bug blacked out (ffmpeg
drawbox on games.json bug_mask): the bug writes "3 POINTER" after a three, which a model that
reads text can match.

    uv run python marengo.py upload --game loras --video data/loras_720_av.mp4
    uv run python marengo.py upload --game loras --variant mask --video data/loras_720_av_mask.mp4
    uv run python marengo.py embed --game loras     # -> index/<game>_marengo_<option>.npz
    uv run python marengo.py search --game loras --delta 5
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from evaluate import GROUPS
from twelvelabs import MultiInputRequest, TwelveLabs

HERE = Path(__file__).parent
DATA, INDEX, OUT = HERE / "data", HERE / "index", HERE / "output"
MODEL = "marengo3.5"
RNG = np.random.default_rng(0)


def client() -> TwelveLabs:
    return TwelveLabs(api_key=os.environ["TWELVE_API_KEY"])


def find(c: TwelveLabs, tag: str) -> tuple[str, str]:
    """(index id, indexed asset id) of the game's index, created by `upload`."""
    idx = next(i for i in c.indexes.list() if i.index_name == f"egbm-{tag}")
    asset = next(iter(c.indexes.indexed_assets.list(idx.id)))
    return idx.id, asset.id


def upload(c: TwelveLabs, tag: str, video: Path) -> None:
    idx = next((i for i in c.indexes.list() if i.index_name == f"egbm-{tag}"), None)
    if idx is None:
        idx = c.indexes.create(
            index_name=f"egbm-{tag}",
            models=[{"model_name": MODEL, "model_options": ["visual", "audio"]}],
        )
    # Files over 200 MB need the multipart route.
    asset_id = c.multipart_upload.upload_file(str(video)).asset_id
    while (a := c.assets.retrieve(asset_id)).status != "ready":
        if a.status == "failed":
            raise RuntimeError(f"asset {asset_id} failed")
        time.sleep(10)
    ia = c.indexes.indexed_assets.create(index_id=idx.id, asset_id=asset_id)
    while (s := c.indexes.indexed_assets.retrieve(idx.id, ia.id).status) != "ready":
        if s == "failed":
            raise RuntimeError("indexing failed")
        time.sleep(30)
    print(f"indexed {video.name} into egbm-{tag}")


def queries(game: str) -> tuple[dict, list[str]]:
    games = json.loads((HERE / "games.json").read_text())["games"]
    q = json.loads((HERE / games[game].get("queries", "queries.json")).read_text())
    return q, [q["background"]] + [t for v in q["classes"].values() for t in v]


def embed(c: TwelveLabs, game: str, tag: str) -> None:
    index_id, asset_id = find(c, tag)
    _, texts = queries(game)
    tv = []
    for t in texts:
        r = c.embed.v_2.create(
            input_type="multi_input", model_name=MODEL, multi_input=MultiInputRequest(input_text=t)
        )
        tv.append(r.data[0].embedding)
    tv = np.array(tv, np.float32)
    tv /= np.linalg.norm(tv, axis=-1, keepdims=True)

    for option in ("visual", "audio"):
        v = c.indexes.indexed_assets.retrieve(index_id, asset_id, embedding_option=[option])
        if v.embedding.video_embedding.segments is None:  # e.g. audio of a silent video
            print(f"no {option} segments")
            continue
        segs = [s for s in v.embedding.video_embedding.segments if s.embedding_scope == "clip"]
        start = np.array([s.start_offset_sec for s in segs], np.float64)
        end = np.array([s.end_offset_sec for s in segs], np.float64)
        emb = np.array([s.float_ for s in segs], np.float32)
        emb /= np.linalg.norm(emb, axis=-1, keepdims=True)
        INDEX.mkdir(exist_ok=True)
        out = INDEX / f"{tag}_marengo_{option}.npz"
        np.savez(out, start=start, end=end, emb=emb, length=float((end - start).mean()),
                 texts=np.array(texts), text_emb=tv)  # fmt: skip
        print(
            f"{len(segs)} {option} segments, {float((end - start).mean()):.1f} s mean -> {out.name}"
        )


def search(c: TwelveLabs, game: str, tag: str, delta: float) -> None:
    index_id, _ = find(c, tag)
    q, _ = queries(game)
    moments = json.loads((DATA / f"{game}_moments.json").read_text())
    halves: dict[int, tuple[float, float]] = {}
    for m in moments:
        lo, hi = halves.get(m["half"], (m["t"], m["t"]))
        halves[m["half"]] = (min(lo, m["t"]), max(hi, m["t"]))
    bounds = [(lo - 30, hi + 30) for lo, hi in halves.values()]

    def inside(t: float) -> bool:
        return any(lo <= t <= hi for lo, hi in bounds)

    lines = [
        f"# {tag}, TwelveLabs search ({MODEL}); delta = {delta:g} s",
        "",
        "| class | n | options | spotted in top-n | chance (p95) | mean clip s |",
        "|---|---|---|---|---|---|",
    ]
    for cls, phrasings in q["classes"].items():
        ev = sorted(
            (m for m in moments if m["cls"] in GROUPS.get(cls, {cls})), key=lambda m: m["t"]
        )
        if len(ev) < 3:
            continue
        spans = [(m["lo"], m["hi"]) for m in ev]
        for options in (["visual"], ["visual", "audio"]):
            got, lens = [], []
            for p in phrasings:
                clips = []
                for r in c.search.query(index_id=index_id, query_text=p, search_options=options,
                                        page_limit=50):  # fmt: skip
                    if inside((r.start + r.end) / 2):
                        clips.append((r.start, r.end))
                    if len(clips) == len(ev):
                        break
                got.append(match(clips, spans, delta))
                lens += [e - s for s, e in clips]
            chance = [match(random_clips(lens[: len(ev)] or [6.0], bounds), spans, delta)
                      for _ in range(200)]  # fmt: skip
            g = np.array(got, float)
            lines.append(
                f"| {cls} | {len(ev)} | {'+'.join(options)} "
                f"| {g.mean():.1f} ({g.min():.0f}-{g.max():.0f}) "
                f"| {np.mean(chance):.1f} ({np.percentile(chance, 95):.0f}) "
                f"| {np.mean(lens or [0]):.1f} |"
            )
            print(lines[-1], flush=True)
    lines += ["", "Spotted: mean over the three phrasings (min-max)."]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag}_twelvelabs_search_d{delta:g}.md").write_text("\n".join(lines) + "\n")


def match(clips: list[tuple[float, float]], spans: list[tuple[float, float]], delta: float) -> int:
    """Moments overlapped by a clip widened by delta; each clip claims one moment."""
    used, n = set(), 0
    for lo, hi in spans:
        best = [
            j
            for j, (s, e) in enumerate(clips)
            if j not in used and s - delta <= hi and lo <= e + delta
        ]
        if best:
            used.add(best[0])
            n += 1
    return n


def random_clips(lens: list[float], bounds: list[tuple[float, float]]) -> list[tuple[float, float]]:
    total = sum(hi - lo for lo, hi in bounds)
    out = []
    for ln in lens:
        x = RNG.uniform(0, total)
        for lo, hi in bounds:
            if x <= hi - lo:
                out.append((lo + x, lo + x + ln))
                break
            x -= hi - lo
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["upload", "embed", "search"])
    ap.add_argument("--game", default=None)
    ap.add_argument("--delta", type=float, default=5.0)
    ap.add_argument("--variant", default="", help="a second index of the same game, e.g. mask")
    ap.add_argument("--video", default=None, help="file to upload (default: games.json video)")
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]
    tag = f"{key}-{args.variant}" if args.variant else key
    c = client()
    if args.step == "upload":
        upload(c, tag, Path(args.video) if args.video else DATA / games["games"][key]["video"])
    elif args.step == "embed":
        embed(c, key, tag)
    else:
        search(c, key, tag, args.delta)


if __name__ == "__main__":
    main()
