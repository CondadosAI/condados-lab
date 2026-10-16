"""Reference embeddings: PyTorch, float32, CPU. Everything else is compared against these.

For every clip it embeds four versions:

- fwd: the clip as filmed, sampled at `--fps`.
- rev: the reversed file from prepare.py, sampled the same way.
- rfr: the forward frames in reverse order. Same pixels as fwd, only the order differs. `rev`
  samples the reversed file at 1 fps and lands on other instants, so it mixes order with
  different frames; rfr isolates order.
- shf: the forward frames in a fixed random order. If shf scores like fwd, order is not used.

An existing index is extended, not recomputed: only keys it lacks are embedded.

It also embeds the texts: the search queries and, for every clip, its forward caption (and the
reversed caption for paired clips). Texts get the model card's retrieval prefix; videos get none.

    uv run python embed.py              # all clips, 1 fps
    uv run python embed.py --limit 3    # timed pilot
    uv run python embed.py --fps 2
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from smoke_export import MODEL_ID, pooled, sample_frames
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).parent
CLIPS, INDEX = HERE / "data" / "clips", HERE / "index"
QUERY_PREFIX = "task: search result | query: "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    spec = json.loads((HERE / "clips.json").read_text())
    clips = spec["clips"][: args.limit]
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)

    def embed_video(frames: np.ndarray) -> np.ndarray:
        x = proc(text=["<|video|>"], videos=[frames], do_sample_frames=False, return_tensors="pt")
        return pooled(model(**x).last_hidden_state.numpy(), x["attention_mask"].numpy())[0]

    def embed_text(texts: list[str]) -> np.ndarray:
        x = proc(text=[QUERY_PREFIX + t for t in texts], padding=True, return_tensors="pt")
        return pooled(model(**x).last_hidden_state.numpy(), x["attention_mask"].numpy())

    rng = np.random.default_rng(0)
    path = INDEX / f"pytorch_fp32_fps{args.fps:g}.npz"
    out: dict[str, np.ndarray] = dict(np.load(path)) if path.exists() and args.limit is None else {}
    t_all = time.perf_counter()
    for c in clips:
        fwd = sample_frames(CLIPS / f"{c['key']}_fwd.mp4", args.fps)
        rev = sample_frames(CLIPS / f"{c['key']}_rev.mp4", args.fps)
        perm = rng.permutation(len(fwd))
        while len(fwd) > 1 and (perm == np.arange(len(fwd))).all():
            perm = rng.permutation(len(fwd))
        versions = {"fwd": fwd, "rev": rev, "shf": fwd[perm], "rfr": fwd[::-1].copy()}
        t = time.perf_counter()
        todo = [v for v in versions if f"{v}/{c['key']}" not in out]
        for v in todo:
            out[f"{v}/{c['key']}"] = embed_video(versions[v])
        dt = (time.perf_counter() - t) / max(len(todo), 1)
        logger.info(f"{c['key']}: {len(fwd)} frames, {todo}, {dt:.1f} s per version")

    queries = [q["query"] for q in spec["queries"]]
    for i, e in enumerate(embed_text(queries)):
        out[f"query/{i}"] = e
    for c in clips:
        out[f"cap_fwd/{c['key']}"] = embed_text([c["caption_fwd"]])[0]
        if c["kind"] == "paired":
            out[f"cap_rev/{c['key']}"] = embed_text([c["caption_rev"]])[0]

    total = time.perf_counter() - t_all
    logger.info(f"{len(clips)} clips + texts in {total:.0f} s")
    if args.limit is None:
        INDEX.mkdir(parents=True, exist_ok=True)
        np.savez(path, **out)


if __name__ == "__main__":
    main()
