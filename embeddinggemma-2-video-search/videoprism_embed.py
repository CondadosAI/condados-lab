"""The control model: VideoPrism-LvT-B, a video-text encoder with temporal attention layers.

EmbeddingGemma 2 encodes each frame on its own and lets a text transformer read them in a row.
VideoPrism (arXiv 2402.13217) has four temporal-attention layers between the frames, so if any
open model should tell a clip from its reverse, it is this one. Same clips, same captions, same
index keys as embed.py, so `evaluate.py` scores both the same way:

    uv run python videoprism_embed.py
    uv run python evaluate.py index/videoprism_lvt_b_16f.npz

It reads 16 frames spread evenly over the clip, the count the model was trained on, resized to
288x288 by its own processor. The weights are the transformers conversion in the model repo's
open pull request #2, pinned to its commit; the `main` branch holds only the Flax checkpoint.
Checked against Google's JAX code (github.com/google-deepmind/videoprism @ c6eba0e, jax 0.11.2,
flax 0.12.10) on the same pixels and token ids: cosine 1.0 for video and text. The transformers
tokenizer skips the text clean-up the JAX code applies, so `canonical` does it here; without it
the token ids differ and a caption lands at cosine ~0.99 of the official one.
"""

import argparse
import json
import string
import time
from pathlib import Path

import av
import numpy as np
import torch
from loguru import logger
from transformers import VideoPrismClipModel, VideoPrismProcessor

HERE = Path(__file__).parent
CLIPS, INDEX = HERE / "data" / "clips", HERE / "index"
MODEL_ID = "google/videoprism-lvt-base-f16r288"
REVISION = "fb6de9f0eb7bc285be86bdca1cf7daa3e3ef51ff"  # refs/pr/2, the transformers weights
N_FRAMES = 16


def even_frames(path: Path, n: int = N_FRAMES) -> np.ndarray:
    """n frames spread evenly from the first to the last: (n, H, W, 3) uint8."""
    with av.open(str(path)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    idx = np.linspace(0, len(frames) - 1, n).round().astype(int)
    return np.stack([frames[i] for i in idx])


def canonical(text: str) -> str:
    """The JAX code's text clean-up: punctuation to spaces, lower case, one space, final period."""
    p = string.punctuation
    return " ".join(text.translate(str.maketrans(p, " " * len(p))).lower().split()) + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    spec = json.loads((HERE / "clips.json").read_text())
    clips = spec["clips"][: args.limit]
    model = VideoPrismClipModel.from_pretrained(MODEL_ID, revision=REVISION).eval()
    proc = VideoPrismProcessor.from_pretrained(MODEL_ID, revision=REVISION)

    def embed_video(frames: np.ndarray) -> np.ndarray:
        x = proc(videos=[frames], do_sample_frames=False, return_tensors="pt")
        return model.get_video_features(**x).pooler_output.reshape(-1).numpy()  # (1, 1, 768)

    def embed_text(texts: list[str]) -> np.ndarray:
        x = proc(text=[canonical(t) for t in texts], return_tensors="pt")
        return model.get_text_features(**x).pooler_output.numpy()

    rng = np.random.default_rng(0)
    path = INDEX / f"videoprism_lvt_b_{N_FRAMES}f.npz"
    out: dict[str, np.ndarray] = dict(np.load(path)) if path.exists() and args.limit is None else {}
    t_all = time.perf_counter()
    for c in clips:
        fwd = even_frames(CLIPS / f"{c['key']}_fwd.mp4")
        rev = even_frames(CLIPS / f"{c['key']}_rev.mp4")
        perm = rng.permutation(N_FRAMES)
        versions = {"fwd": fwd, "rev": rev, "shf": fwd[perm], "rfr": fwd[::-1].copy()}
        t = time.perf_counter()
        todo = [v for v in versions if f"{v}/{c['key']}" not in out]
        for v in todo:
            out[f"{v}/{c['key']}"] = embed_video(versions[v])
        dt = (time.perf_counter() - t) / max(len(todo), 1)
        logger.info(f"{c['key']}: {todo}, {dt:.1f} s per version")

    for i, e in enumerate(embed_text([q["query"] for q in spec["queries"]])):
        out[f"query/{i}"] = e
    for c in clips:
        out[f"cap_fwd/{c['key']}"] = embed_text([c["caption_fwd"]])[0]
        if c["kind"] == "paired":
            out[f"cap_rev/{c['key']}"] = embed_text([c["caption_rev"]])[0]

    logger.info(f"{len(clips)} clips + texts in {time.perf_counter() - t_all:.0f} s")
    if args.limit is None:
        INDEX.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)


if __name__ == "__main__":
    main()
