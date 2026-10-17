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

Vision runs on the iGPU at f32 and the encoder on the CPU at f32, both cosine 1.000000 against
PyTorch at 1 fps. `--check` compares 3 clips against PyTorch at 2 and 4 fps (longer sequences).

    uv run python order_probe.py             # resumable: keeps index/order_probe.npz
    uv run python order_probe.py --check
"""

import argparse
import json
from pathlib import Path

import numpy as np
from benchmark import igpu
from evaluate import ci
from loguru import logger
from ov_model import TOK_PER_FRAME, OVEmbedder
from smoke_export import sample_frames
from transformers.video_utils import VideoMetadata

HERE = Path(__file__).parent
CLIPS, OUT = HERE / "data" / "clips", HERE / "output"
INDEX = HERE / "index" / "order_probe.npz"
FPS = (1, 2, 4)


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def blockwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-token cosine between two (n_frames, 120, D) outputs -> (n_frames * 120,)."""
    return (unit(a) * unit(b)).sum(-1).ravel()


def probe_clip(
    vis: OVEmbedder, enc: OVEmbedder, frames: np.ndarray, fps: int, rng
) -> tuple[dict, dict]:
    n = len(frames)
    x = vis.prompt(frames)
    ids = x["input_ids"][0]
    blocks = vis.frame_tokens(x["pixel_values_videos"], x["video_position_ids"])
    blocks = blocks.reshape(n, TOK_PER_FRAME, -1)
    perm = rng.permutation(n)
    while n > 1 and (perm == np.arange(n)).all():
        perm = rng.permutation(n)
    orders = {"fwd": np.arange(n), "rfr": np.arange(n)[::-1], "shf": perm}

    d = blocks.shape[-1]
    seqs = {k: enc.assemble(ids, blocks[o].reshape(-1, d)) for k, o in orders.items()}
    # Timestamps are whole seconds (mm:ss), so at 2 and 4 fps neighbouring frames share one.
    meta = VideoMetadata(
        total_num_frames=n, fps=float(fps), frames_indices=list(range(n)), duration=n / fps
    )
    ts_ids = vis.prompt(frames, add_timestamps=True, video_metadata=[meta])["input_ids"][0]
    for k in ("fwd", "rfr"):
        seqs[f"ts_{k}"] = enc.assemble(ts_ids, blocks[orders[k]].reshape(-1, d))

    hid = dict(zip(seqs, enc.hidden(list(seqs.values())), strict=True))
    vecs = {k: unit(h.mean(0)) for k, h in hid.items()}

    # Per-token, frame-aligned: block j of a version holds frame orders[k][j]; put it back at k.
    vmask = ids == vis.video_id
    out = {k: hid[k][vmask].reshape(n, TOK_PER_FRAME, -1) for k in orders}
    aligned = {k: np.empty_like(out[k]) for k in orders}
    for k, o in orders.items():
        aligned[k][o] = out[k]
    stats = {
        "tok_cos_rfr": float(np.median(blockwise_cos(aligned["fwd"], aligned["rfr"]))),
        "tok_cos_shf": float(np.median(blockwise_cos(aligned["fwd"], aligned["shf"]))),
    }

    # Bag of frames: each frame alone through the encoder, then the mean of the unit vectors.
    # Its tokens also give the scale: how much does context change a frame's tokens at all?
    one_ids = vis.prompt(frames[:1])["input_ids"][0]
    singles = enc.hidden([enc.assemble(one_ids, b) for b in blocks])
    vecs["bag"] = unit(np.mean([unit(h.mean(0)) for h in singles], 0))
    alone = np.stack([h[one_ids == vis.video_id] for h in singles])
    stats["tok_cos_alone"] = float(np.median(blockwise_cos(aligned["fwd"], alone)))
    return vecs, stats


def run(args) -> None:
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    store = dict(np.load(INDEX)) if INDEX.exists() else {}
    gpu = igpu()
    vis = OVEmbedder("fp32", gpu, "f32", vision_batch=8)
    enc = OVEmbedder("fp32", "CPU")
    for fps in FPS:
        for i, c in enumerate(clips):
            k = c["key"]
            if f"{fps}/fwd/{k}" in store:
                continue
            frames = sample_frames(CLIPS / f"{k}_fwd.mp4", fps=fps)
            rng = np.random.default_rng([fps, i])  # the same shuffle on a resumed run
            vecs, stats = probe_clip(vis, enc, frames, fps, rng)
            for v, e in vecs.items():
                store[f"{fps}/{v}/{k}"] = e
            for s, val in stats.items():
                store[f"{fps}/{s}/{k}"] = np.array(val)
            store[f"{fps}/frames/{k}"] = np.array(len(frames))
            np.savez(INDEX, **store)
            logger.info(f"{fps} fps {k}: {len(frames)} frames, {stats}")


def check() -> None:
    """3 clips at 2 and 4 fps: this pipeline against PyTorch fp32, cosine of the clip vector."""
    import torch
    from smoke_export import MODEL_ID, pooled
    from transformers import AutoModel

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    vis = OVEmbedder("fp32", igpu(), "f32")
    enc = OVEmbedder("fp32", "CPU")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--report", action="store_true", help="only rebuild output/order_probe.md")
    args = ap.parse_args()
    if args.check:
        check()
    else:
        if not args.report:
            run(args)
        report()
