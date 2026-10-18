"""Text-to-video search on a laptop: EmbeddingGemma 2, INT8, OpenVINO, on the Intel iGPU.

The first run embeds every clip in data/clips/ (1 frame per second) and saves the vectors in
index/; after that a search only embeds the query, a few tens of milliseconds, and takes one dot
product against the index. `--dim 256` keeps the first 256 of 768 dimensions (the model is trained
for it, Matryoshka-style) and re-normalises, a third of the storage.

    uv run python search.py "a man puts on a hat"
    uv run python search.py "a candle is blown out" --device CPU --dim 256
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from ov_model import OVEmbedder, igpu
from smoke_export import sample_frames

HERE = Path(__file__).parent
CLIPS, INDEX = HERE / "data" / "clips", HERE / "index"


def truncate(v: np.ndarray, dim: int) -> np.ndarray:
    v = v[..., :dim]
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--variant", default="int8", choices=["fp32", "int8w", "int8"])
    ap.add_argument("--device", default="iGPU", choices=["iGPU", "CPU"])
    ap.add_argument("--dim", type=int, default=768, choices=[768, 512, 256, 128])
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    device = igpu() if args.device == "iGPU" else "CPU"
    model = OVEmbedder(args.variant, device)
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    path = INDEX / f"search_{args.variant}_{args.device}.npz"
    if not path.exists():
        t = time.perf_counter()
        V = np.stack([model.embed_videos([sample_frames(CLIPS / f"{c['key']}_fwd.mp4")])[0]
                      for c in clips])  # fmt: skip
        INDEX.mkdir(exist_ok=True)
        np.savez(path, V=V)
        print(f"indexed {len(V)} clips in {time.perf_counter() - t:.0f} s -> {path.name}")
    V = truncate(np.load(path)["V"], args.dim)

    t = time.perf_counter()
    q = truncate(model.embed_text(args.query), args.dim)
    scores = V @ q
    ms = (time.perf_counter() - t) * 1000
    print(f'"{args.query}"  ({args.variant}, {device}, {args.dim}-d, {ms:.0f} ms)')
    for i in np.argsort(-scores)[: args.k]:
        c = clips[i]
        print(f"  {scores[i]:.3f}  {c['action']:<40} https://www.pexels.com/video/{c['id']}/")


if __name__ == "__main__":
    main()
