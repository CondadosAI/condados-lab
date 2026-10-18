"""Where does frame order get lost? Four probes, at 1, 2 and 4 frames per second.

The vision tower turns each frame into 120 tokens on its own, so it cannot see order. Order can
only enter in the encoder, through each token's position in the sequence. So every probe computes
each frame's tokens once and then lays the same tokens out in different orders:

- fwd, rfr (reversed), shf (shuffled): the encoder's per-token output is compared frame by frame
  (block k of fwd against the block holding the same frame in rfr), before the mean pool. If the
  tokens differ much more than the pooled vectors do, order is present and the mean washes it out.
- bag: each frame encoded alone as a 1-frame video, the vectors averaged. If it searches as well
  as the clip, retrieval does not need the frames to see each other.
- ts_fwd, ts_rfr: the processor's optional timestamps ("00:00 <frame> 00:01 <frame> ...", off by
  default) stay in ascending order while the frames are reversed, an explicit clock in the text.
- fps: more frames per second means smaller steps between frames, so a motion has more tokens.

Vision and encoder run on the iGPU at f32 (cosine 1.000000 against PyTorch). Each frame's vision
tokens are cached in cache/tokens/, and each clip only computes what index/ does not hold yet, so
adding a probe reruns the encoder and nothing else. `--check` compares 3 clips against PyTorch at 2
and 4 fps (longer sequences).

    uv run python order_probe.py             # resumable
    uv run python order_probe.py --check
    uv run python order_probe.py --frames-report
"""

import argparse
import json
from pathlib import Path

import numpy as np
from evaluate import ci
from loguru import logger
from ov_model import TOK_PER_FRAME, OVEmbedder, igpu
from smoke_export import sample_frames
from transformers.video_utils import VideoMetadata

HERE = Path(__file__).parent
CLIPS, OUT = HERE / "data" / "clips", HERE / "output"
INDEX = HERE / "index" / "order_probe.npz"
FRAMES = HERE / "index" / "order_frames.npz"
TOKENS = HERE / "cache" / "tokens"  # each frame's vision tokens, per fps and clip
FPS = (1, 2, 4)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def blockwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token cosine between two (n_frames, 120, D) outputs -> (n_frames * 120,)."""
    return (unit(a) * unit(b)).sum(-1).ravel()


def frame_blocks(vis: OVEmbedder, frames: np.ndarray, path: Path) -> np.ndarray:
    """Each frame's 120 vision tokens, (n_frames, 120, 512), cached on disk: the vision tower is
    most of the cost and its output does not depend on order, so every probe reuses it."""
    if path.exists():
        return np.load(path)
    x = vis.prompt(frames)
    blocks = vis.frame_tokens(x["pixel_values_videos"], x["video_position_ids"])
    blocks = blocks.reshape(len(frames), TOK_PER_FRAME, -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, blocks)
    return blocks


def probe_clip(
    vis: OVEmbedder, enc: OVEmbedder, frames: np.ndarray, blocks: np.ndarray, fps: int, rng,
    parts: set[str],
) -> tuple[dict, dict, dict]:  # fmt: skip
    """`parts`: "order" (fwd, rfr, shf, per-token and per-frame readouts), "ts" (timestamps),
    "bag" (each frame alone; needs "order" for its per-token scale)."""
    n, d = len(frames), blocks.shape[-1]
    ids = vis.prompt(frames)["input_ids"][0]
    perm = rng.permutation(n)
    while n > 1 and (perm == np.arange(n)).all():
        perm = rng.permutation(n)
    orders = {"fwd": np.arange(n), "rfr": np.arange(n)[::-1], "shf": perm}
    vecs, stats, per_frame = {}, {}, {}

    seqs = {}
    if "order" in parts:
        seqs |= {k: enc.assemble(ids, blocks[o].reshape(-1, d)) for k, o in orders.items()}
    if "ts" in parts:
        # Timestamps are whole seconds (mm:ss), so at 2 and 4 fps neighbouring frames share one.
        meta = VideoMetadata(
            total_num_frames=n, fps=float(fps), frames_indices=list(range(n)), duration=n / fps
        )
        ts_ids = vis.prompt(frames, add_timestamps=True, video_metadata=[meta])["input_ids"][0]
        for k in ("fwd", "rfr"):
            seqs[f"ts_{k}"] = enc.assemble(ts_ids, blocks[orders[k]].reshape(-1, d))
    hid = dict(zip(seqs, enc.hidden(list(seqs.values())), strict=True))
    vecs |= {k: unit(h.mean(0)) for k, h in hid.items()}

    if "order" in parts:
        # Per-token, frame-aligned: slot j of a version holds frame orders[k][j]; put it back.
        vmask = ids == vis.video_id
        aligned = {}
        for k, o in orders.items():
            aligned[k] = np.empty((n, TOK_PER_FRAME, hid[k].shape[-1]), np.float32)
            aligned[k][o] = hid[k][vmask].reshape(n, TOK_PER_FRAME, -1)
        stats["tok_cos_rfr"] = float(np.median(blockwise_cos(aligned["fwd"], aligned["rfr"])))
        stats["tok_cos_shf"] = float(np.median(blockwise_cos(aligned["fwd"], aligned["shf"])))
        # Per-frame readouts, indexed by frame (not by slot): each frame's token mean.
        per_frame["blk_fwd"] = aligned["fwd"].mean(1)
        per_frame["blk_rfr"] = aligned["rfr"].mean(1)

    if "bag" in parts:
        # Bag of frames: each frame alone through the encoder, then the mean of the unit vectors.
        # Its tokens also give the scale: how much does context change a frame's tokens at all?
        one_ids = vis.prompt(frames[:1])["input_ids"][0]
        singles = enc.hidden([enc.assemble(one_ids, b) for b in blocks])
        per_frame["single"] = np.stack([unit(h.mean(0)) for h in singles])
        vecs["bag"] = unit(per_frame["single"].mean(0))
        if "order" in parts:
            alone = np.stack([h[one_ids == vis.video_id] for h in singles])
            stats["tok_cos_alone"] = float(np.median(blockwise_cos(aligned["fwd"], alone)))
    return vecs, stats, per_frame


def missing(store: dict, frames_store: dict, fps: int, k: str) -> set[str]:
    have = {
        "order": f"{fps}/fwd/{k}" in store and f"{fps}/blk_fwd/{k}" in frames_store,
        "ts": f"{fps}/ts_fwd/{k}" in store,
        "bag": f"{fps}/bag/{k}" in store and f"{fps}/single/{k}" in frames_store,
    }
    need = {p for p, ok in have.items() if not ok}
    if "bag" in need:
        need.add("order")  # the bag's per-token scale is measured against fwd
    return need


def run(args) -> None:
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    store = dict(np.load(INDEX)) if INDEX.exists() else {}
    frames_store = dict(np.load(FRAMES)) if FRAMES.exists() else {}
    gpu = igpu()
    vis = OVEmbedder("fp32", gpu, "f32", vision_batch=8)
    enc = vis if args.enc_device == "iGPU" else OVEmbedder("fp32", "CPU")
    for fps in args.fps:
        for i, c in enumerate(clips):
            k = c["key"]
            parts = missing(store, frames_store, fps, k)
            if not parts:
                continue
            frames = sample_frames(CLIPS / f"{k}_fwd.mp4", fps=fps)
            blocks = frame_blocks(vis, frames, TOKENS / f"{fps}fps_{k}.npy")
            rng = np.random.default_rng([fps, i])  # the same shuffle on a resumed run
            vecs, stats, per_frame = probe_clip(vis, enc, frames, blocks, fps, rng, parts)
            for v, e in vecs.items():
                store[f"{fps}/{v}/{k}"] = e
            for s, val in stats.items():
                store[f"{fps}/{s}/{k}"] = np.array(val)
            store[f"{fps}/frames/{k}"] = np.array(len(frames))
            for v, e in per_frame.items():
                frames_store[f"{fps}/{v}/{k}"] = e
            np.savez(INDEX, **store)
            np.savez(FRAMES, **frames_store)
            logger.info(f"{fps} fps {k}: {len(frames)} frames, computed {sorted(parts)}, {stats}")


def check(enc_device: str) -> None:
    """3 clips at 2 and 4 fps: this pipeline against PyTorch fp32, cosine of the clip vector."""
    import torch
    from smoke_export import MODEL_ID, pooled
    from transformers import AutoModel

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    vis = OVEmbedder("fp32", igpu(), "f32")
    enc = vis if enc_device == "iGPU" else OVEmbedder("fp32", "CPU")
    for fps in (2, 4):
        for c in clips[:3]:
            frames = sample_frames(CLIPS / f"{c['key']}_fwd.mp4", fps=fps)
            x = vis.proc(text=["<|video|>"], videos=[frames], do_sample_frames=False,
                         return_tensors="pt")  # fmt: skip
            ref = pooled(model(**x).last_hidden_state.numpy(), x["attention_mask"].numpy())[0]
            v = enc.encode([vis.video_inputs(frames)])[0]
            n = len(frames)
            logger.info(f"{fps} fps {c['key']} ({n} frames): cos vs PyTorch {v @ ref:.6f}")


def search(V: np.ndarray, Q: np.ndarray, ids: list[str], queries: list[dict]) -> str:
    ranks = np.array([
        next(r for r, j in enumerate(np.argsort(-(V @ Q[i])), 1) if ids[j] in q["targets"])
        for i, q in enumerate(queries)
    ])  # fmt: skip
    n = len(queries)
    return f"top-1 {(ranks == 1).sum()}/{n}, top-3 {(ranks <= 3).sum()}/{n}, " \
           f"MRR {np.mean(1 / ranks):.3f}"  # fmt: skip


def paired(F: np.ndarray, R: np.ndarray, Tf: np.ndarray, Tr: np.ndarray) -> str:
    """Paired direction: sign of (clip - reversed) . (caption fwd - caption rev), per clip."""
    ok = (((F - R) * (Tf - Tr)).sum(1) > 0).tolist()
    return ci(sum(ok), len(ok))


def report() -> None:
    E = dict(np.load(INDEX))
    ref = dict(np.load(HERE / "index" / "pytorch_fp32_fps1.npz"))
    spec = json.loads((HERE / "clips.json").read_text())
    clips, queries = spec["clips"], spec["queries"]
    keys, ids = [c["key"] for c in clips], [c["id"] for c in clips]
    pk = [c["key"] for c in clips if c["kind"] == "paired"]
    Q = np.stack([ref[f"query/{i}"] for i in range(len(queries))])
    Tf, Tr = (np.stack([ref[f"{t}/{k}"] for k in pk]) for t in ("cap_fwd", "cap_rev"))
    lines = ["# Where frame order gets lost", ""]
    for fps in FPS:
        if any(f"{fps}/fwd/{k}" not in E for k in keys):
            continue
        V = {v: np.stack([E[f"{fps}/{v}/{k}"] for k in keys])
             for v in ("fwd", "rfr", "shf", "bag", "ts_fwd", "ts_rfr")}  # fmt: skip
        P = {
            v: np.stack([E[f"{fps}/{v}/{k}"] for k in pk])
            for v in ("fwd", "rfr", "ts_fwd", "ts_rfr")
        }
        pairs = (("fwd", "rfr"), ("fwd", "shf"), ("fwd", "bag"), ("ts_fwd", "ts_rfr"))
        cos = {f"{a} vs {b}": float(np.median((V[a] * V[b]).sum(1))) for a, b in pairs}
        tok = {
            s: float(np.median([E[f"{fps}/tok_cos_{s}/{k}"] for k in keys]))
            for s in ("rfr", "shf", "alone")
        }
        frames = [int(E[f"{fps}/frames/{k}"]) for k in keys]
        lines += [
            f"## {fps} fps ({min(frames)}–{max(frames)} frames per clip)",
            "",
            f"- Search, clip: {search(V['fwd'], Q, ids, queries)}",
            f"- Search, bag of frames: {search(V['bag'], Q, ids, queries)}",
            f"- Search, clip with timestamps: {search(V['ts_fwd'], Q, ids, queries)}",
            f"- Direction, paired, same frames reversed: {paired(P['fwd'], P['rfr'], Tf, Tr)}",
            "- Direction, paired, reversed with timestamps kept ascending: "
            f"{paired(P['ts_fwd'], P['ts_rfr'], Tf, Tr)}",
            "- Cosine of the pooled vectors, median: "
            + ", ".join(f"{k} {v:.4f}" for k, v in cos.items()),
            "- Cosine per token (the same frame's tokens, before pooling), median over clips: "
            f"fwd vs rfr {tok['rfr']:.4f}, fwd vs shf {tok['shf']:.4f}; for scale, the frame in "
            f"the clip vs the frame alone {tok['alone']:.4f}",
            "",
        ]
    OUT.mkdir(exist_ok=True)
    (OUT / "order_probe.md").write_text("\n".join(lines))
    print("\n".join(lines))


def frames_report() -> None:
    """Two readings of the per-frame data (index/order_frames.npz, 1 and 4 fps).

    - Lean: does the clip vector sit closer to the frames at the start of the sequence? cos(clip,
      frame encoded alone) by the slot the frame occupies, in quarters, for fwd and rfr.
    - Decodable direction: a linear classifier (logistic regression, one clip held out at a time)
      tells a frame's token mean in the forward sequence from the same frame's in the reversed one.
      Same frame, same pixels, only the order around it differs. The same test on the pooled clip
      vectors shows what the mean keeps.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    F, E = dict(np.load(FRAMES)), dict(np.load(INDEX))
    keys = [c["key"] for c in json.loads((HERE / "clips.json").read_text())["clips"]]
    lines = ["# Per-frame readings", ""]
    for fps in sorted({int(k.split("/")[0]) for k in F}):
        lean = {"fwd": [[] for _ in range(4)], "rfr": [[] for _ in range(4)]}
        X, y, g, Xp, yp, gp = [], [], [], [], [], []
        for ci_, k in enumerate(keys):
            single = F[f"{fps}/single/{k}"]
            n = len(single)
            for v in ("fwd", "rfr"):
                cos = single @ E[f"{fps}/{v}/{k}"]  # by frame (time order)
                slot = np.arange(n) if v == "fwd" else np.arange(n)[::-1]
                for i in range(n):
                    lean[v][min(3, 4 * slot[i] // n)].append(cos[i])
                X.append(F[f"{fps}/blk_{v}/{k}"])
                y += [v == "fwd"] * n
                g += [ci_] * n
                Xp.append(E[f"{fps}/{v}/{k}"])
                yp.append(v == "fwd")
                gp.append(ci_)

        def acc(X, y, g) -> str:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            a = cross_val_score(clf, X, y, groups=g, cv=LeaveOneGroupOut())
            return f"{np.mean(a):.0%}"

        lines += [
            f"## {fps} fps",
            "",
            "Clip vector vs each frame encoded alone, mean cosine by the quarter of the sequence "
            "the frame sits in (1st → 4th):",
            "",
            *[f"- {v}: " + " · ".join(f"{np.mean(q):.4f}" for q in lean[v]) for v in lean],
            "",
            "Forward vs reversed, linear classifier, accuracy with each clip held out "
            "(chance 50%):",
            "",
            f"- from a frame's token mean (same frame in both): "
            f"{acc(np.concatenate(X), np.array(y), np.array(g))}",
            f"- from the pooled clip vector: {acc(np.stack(Xp), np.array(yp), np.array(gp))}",
            "",
        ]
    (OUT / "order_frames.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild output/order_probe.md")
    ap.add_argument("--fps", type=int, nargs="*", default=list(FPS))
    ap.add_argument(
        "--enc-device", choices=("iGPU", "CPU"), default="iGPU",
        help="encoder device at f32; the iGPU is ~2.5x faster and matched PyTorch at 1.000000",
    )  # fmt: skip
    ap.add_argument("--frames-report", action="store_true", help="only order_frames.md")
    args = ap.parse_args()
    if args.frames_report:
        frames_report()
    elif args.check:
        check(args.enc_device)
    else:
        if not args.report:
            run(args)
        report()
