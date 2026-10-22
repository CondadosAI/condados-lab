"""Cut the game into overlapping windows and embed each one: index/<game>_<tag>.npz.

A window is `--length` seconds of video sampled at `--fps`, starting every `--stride` seconds,
fed to the model as one video, so a 4 s window at 2 fps is 8 frames. Frames are downscaled
to 640x360 first, as in the first post; the processor then cuts each frame into at most
`--tokens` soft tokens. Length, stride, fps and tokens are the knobs this post measures, not
defaults: averaging over a long window dilutes a 2 s event.

The reference run is PyTorch float32 (float16 gives NaN, per the model card). It runs on
whatever device is given; full games go to a desktop GPU, not the laptop.

    uv run python embed_windows.py --device cuda --start 3300 --end 4500
    uv run python embed_windows.py --device cuda --length 8 --stride 2
    uv run python embed_windows.py --device cuda --game loras --mask-bug \
        --segments index/loras_marengo_visual.npz   # another model's segments, 8 frames each

`--segments` embeds the spans of another index (start/end arrays) instead of sliding windows:
the control for comparing against TwelveLabs, whose segments are cut at scene changes.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import av
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).parent
DATA, INDEX = HERE / "data", HERE / "index"
MODEL_ID = "google/embeddinggemma-2"


def pooled(last_hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean over real tokens, then L2: what SentenceTransformers does after the encoder."""
    m = mask[..., None].astype(np.float32)
    v = (last_hidden * m).sum(axis=1) / m.sum(axis=1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sampled(path: Path, fps: float, start: float, end: float | None, mask: list | None = None):
    """Yield (time, 640x360 RGB frame) at `fps`, from `start` to `end` seconds.

    `mask` = [x0, y0, x1, y1] in source pixels, blacked out first: the score bug. Its clock
    freezes at every stoppage and its shot clock resets at every change of possession, so a
    window could match "timeout" by reading the graphic instead of the court.
    """
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        if start > 0:
            c.seek(int(max(start - 2, 0) / s.time_base), stream=s)
        nxt = start
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if end is not None and t > end:
                return
            if t + 1e-6 >= nxt:
                im = f.to_ndarray(format="rgb24", width=640, height=360)
                if mask:
                    sx, sy = 640 / f.width, 360 / f.height
                    x0, y0, x1, y1 = mask
                    im[int(y0 * sy) : int(y1 * sy) + 1, int(x0 * sx) : int(x1 * sx) + 1] = 0
                yield t, im
                nxt += 1 / fps


def segment_clips(frames, segs: np.ndarray, n: int):
    """Yield n frames spread evenly over each segment, from a time-ordered frame stream."""
    buf, i = [], 0
    for t, im in frames:
        buf.append((t, im))
        while i < len(segs) and t >= segs[i][1]:
            s, e = segs[i]
            want = s + (np.arange(n) + 0.5) * (e - s) / n
            ts = np.array([b[0] for b in buf])
            yield np.stack([buf[int(np.abs(ts - w).argmin())][1] for w in want])
            i += 1
            keep_from = segs[i][0] - 1 if i < len(segs) else t
            buf = [b for b in buf if b[0] >= keep_from]


def tag_of(a) -> str:
    m = "_mask" if a.mask_bug else ""
    if a.segments:
        return f"seg-{Path(a.segments).stem}_N{a.frames}_T{a.tokens}{m}"
    return f"L{a.length:g}_S{a.stride:g}_F{a.fps:g}_T{a.tokens}{m}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None)
    ap.add_argument("--length", type=float, default=4.0)
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--tokens", type=int, default=140)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mask-bug", action="store_true", help="black out games.json bug_mask")
    ap.add_argument("--segments", default=None, help="an index .npz whose start/end to embed")
    ap.add_argument("--frames", type=int, default=8, help="frames per segment with --segments")
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]

    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval().to(args.device)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    n_frames = round(args.length * args.fps)
    every = round(args.stride * args.fps)

    def embed(batch: list[np.ndarray]) -> np.ndarray:
        x = proc(text=["<|video|>"] * len(batch), videos=batch, do_sample_frames=False,
                 max_soft_tokens=args.tokens, padding=True, return_tensors="pt")  # fmt: skip
        x = {k: v.to(args.device) if hasattr(v, "to") else v for k, v in x.items()}
        h = model(**x).last_hidden_state.float().cpu().numpy()
        return pooled(h, x["attention_mask"].cpu().numpy())

    buf: deque = deque(maxlen=n_frames)
    starts, vecs, pending, pend_t = [], [], [], []
    t0 = time.perf_counter()
    mask = games["games"][key]["bug_mask"] if args.mask_bug else None
    frames = sampled(DATA / games["games"][key]["video"], args.fps, args.start, args.end, mask)
    if args.segments:
        d = np.load(args.segments)
        segs = np.stack([d["start"], d["end"]], axis=1)
        segs = segs[np.argsort(segs[:, 0], kind="stable")]  # TwelveLabs returns them unsorted
        for clip in segment_clips(frames, segs, args.frames):
            pending.append(clip)
            if len(pending) == args.batch:
                vecs.append(embed(pending))
                pending = []
        if pending:
            vecs.append(embed(pending))
        emb = np.concatenate(vecs).astype(np.float32)
        INDEX.mkdir(exist_ok=True)
        out = INDEX / f"{key}_{tag_of(args)}.npz"
        np.savez(out, start=segs[: len(emb), 0], end=segs[: len(emb), 1], emb=emb,
                 length=float((segs[:, 1] - segs[:, 0]).mean()), tokens=args.tokens)  # fmt: skip
        print(f"{len(emb)} segments in {time.perf_counter() - t0:.0f} s -> {out.name}")
        return
    for k, (t, im) in enumerate(frames, start=1):
        buf.append((t, im))
        if len(buf) == n_frames and (k - n_frames) % every == 0:
            pend_t.append(buf[0][0])
            pending.append(np.stack([f for _, f in buf]))
            if len(pending) == args.batch:
                vecs.append(embed(pending))
                starts += pend_t
                pending, pend_t = [], []
                if len(starts) % 200 == 0:
                    rate = len(starts) / (time.perf_counter() - t0)
                    print(f"{starts[-1]:.0f} s: {len(starts)} windows, {rate:.1f}/s", flush=True)
    if pending:
        vecs.append(embed(pending))
        starts += pend_t

    INDEX.mkdir(exist_ok=True)
    out = INDEX / f"{key}_{tag_of(args)}_{args.start:g}-{args.end or 'end'}.npz"
    np.savez(out, start=np.array(starts), emb=np.concatenate(vecs).astype(np.float32),
             length=args.length, stride=args.stride, fps=args.fps, tokens=args.tokens)  # fmt: skip
    dt = time.perf_counter() - t0
    print(f"{len(starts)} windows in {dt:.0f} s ({dt / len(starts):.2f} s each) -> {out.name}")


if __name__ == "__main__":
    main()
