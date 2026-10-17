"""Speed and quality of every OpenVINO variant on every device, against the PyTorch reference.

Configurations: {fp32, int8w, int8} x {CPU, iGPU at f16 (the plugin default), iGPU at f32}.

- quality: embeds the 43 forward clips and 23 queries, then reports cosine against PyTorch fp32
  (median and worst clip), NaN count, and the search scores (top-1, top-3, MRR).
- latency: one 8-frame clip, 3 warm-up runs then 10 timed runs, for each stage:
  vision (8 frames per call, and 1 per call), encoder (1 clip per call, and 4), a text query, and
  the whole clip from frames to vector. Throughput for indexing is clips per second at encoder
  batch 4.

    uv run python benchmark.py --session 1      # repeat with 2, 3 on other days / after reboots
"""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import openvino as ov
from loguru import logger
from ov_model import OVEmbedder
from smoke_export import sample_frames

HERE = Path(__file__).parent
CLIPS, OUT = HERE / "data" / "clips", HERE / "output"
WARMUP, RUNS = 3, 10


def igpu() -> str:
    core = ov.Core()
    for d in core.available_devices:
        if d.startswith("GPU") and "iGPU" in core.get_property(d, "FULL_DEVICE_NAME"):
            return d
    raise RuntimeError("no Intel iGPU found")


def timed(fn) -> np.ndarray:
    for _ in range(WARMUP):
        fn()
    t = []
    for _ in range(RUNS):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    return np.array(t)


def quality(e: OVEmbedder, spec: dict, ref: dict) -> dict:
    clips, queries = spec["clips"], spec["queries"]
    V = np.stack([e.embed_videos([sample_frames(CLIPS / f"{c['key']}_fwd.mp4")])[0] for c in clips])
    Q = np.stack([e.embed_text(q["query"]) for q in queries])
    cos = np.array([V[i] @ ref[f"fwd/{c['key']}"] for i, c in enumerate(clips)])
    qcos = np.array([Q[i] @ ref[f"query/{i}"] for i in range(len(queries))])
    ids = [c["id"] for c in clips]
    ranks = []
    for i, q in enumerate(queries):
        order = np.argsort(-(V @ Q[i]))
        ranks.append(next(r for r, j in enumerate(order, 1) if ids[j] in q["targets"]))
    ranks = np.array(ranks)
    return {
        "nan": int(np.isnan(V).any(1).sum() + np.isnan(Q).any(1).sum()),
        "cos_clip_median": float(np.nanmedian(cos)),
        "cos_clip_min": float(np.nanmin(cos)),
        "cos_query_min": float(np.nanmin(qcos)),
        "top1": int((ranks == 1).sum()),
        "top3": int((ranks <= 3).sum()),
        "mrr": float(np.mean(1 / ranks)),
    }


def latency(variant: str, device: str, precision: str | None, frames: np.ndarray) -> dict:
    e1 = OVEmbedder(variant, device, precision, vision_batch=8, enc_batch=1)
    x = e1.prompt(frames)
    pv, pos = x["pixel_values_videos"], x["video_position_ids"]
    emb = e1.video_inputs(frames)
    r = {
        "vision_b8_ms": timed(lambda: e1.frame_tokens(pv, pos)),
        "encoder_b1_ms": timed(lambda: e1.encode([emb])),
        "query_ms": timed(lambda: e1.embed_text("a man puts on a hat")),
        "clip_end_to_end_ms": timed(lambda: e1.embed_videos([frames])),
    }
    e = OVEmbedder(variant, device, precision, vision_batch=1, enc_batch=4)
    r["vision_b1_ms"] = timed(lambda: e.frame_tokens(pv, pos))
    r["encoder_b4_per_clip_ms"] = timed(lambda: e.encode([emb] * 4)) / 4
    return {k: float(np.median(v)) for k, v in r.items()} | {
        f"{k}_p90": float(np.percentile(v, 90)) for k, v in r.items()
    }


def pytorch_latency(frames: np.ndarray) -> dict:
    import torch
    from ov_model import MODEL_ID, QUERY_PREFIX, pooled
    from transformers import AutoModel, AutoProcessor

    torch.set_grad_enabled(False)
    m = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    p = AutoProcessor.from_pretrained(MODEL_ID)

    def clip():
        x = p(text=["<|video|>"], videos=[frames], do_sample_frames=False, return_tensors="pt")
        return pooled(m(**x).last_hidden_state.numpy(), x["attention_mask"].numpy())

    def query():
        x = p(text=[QUERY_PREFIX + "a man puts on a hat"], return_tensors="pt")
        return pooled(m(**x).last_hidden_state.numpy(), x["attention_mask"].numpy())

    return {
        "clip_end_to_end_ms": float(np.median(timed(clip))),
        "query_ms": float(np.median(timed(query))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--skip-quality", action="store_true")
    args = ap.parse_args()
    spec = json.loads((HERE / "clips.json").read_text())
    ref = dict(np.load(HERE / "index" / "pytorch_fp32_fps1.npz"))
    key = next(
        c["key"] for c in spec["clips"] if len(sample_frames(CLIPS / f"{c['key']}_fwd.mp4")) == 8
    )
    frames = sample_frames(CLIPS / f"{key}_fwd.mp4")
    gpu = igpu()
    core = ov.Core()
    env = {
        "session": args.session,
        "cpu": platform.processor() or platform.machine(),
        "cpu_name": core.get_property("CPU", "FULL_DEVICE_NAME"),
        "gpu_name": core.get_property(gpu, "FULL_DEVICE_NAME"),
        "openvino": ov.get_version(),
        "latency_clip": f"{key} (8 frames, 640x360)",
    }
    logger.info(env)

    configs = [(v, d, p) for v in ("fp32", "int8w", "int8")
               for d, p in (("CPU", None), (gpu, "f16"), (gpu, "f32"))]  # fmt: skip
    rows = [{"variant": "pytorch fp32", "device": "CPU", **pytorch_latency(frames)}]
    for variant, device, precision in configs:
        name = f"{device}{'' if precision is None else ' ' + precision}"
        row = {"variant": variant, "device": name, **latency(variant, device, precision, frames)}
        if not args.skip_quality:
            row |= quality(OVEmbedder(variant, device, precision), spec, ref)
        logger.info(row)
        rows.append(row)

    OUT.mkdir(exist_ok=True)
    path = OUT / f"benchmark_session{args.session}.csv"
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (OUT / f"benchmark_session{args.session}_env.json").write_text(json.dumps(env, indent=1))
    logger.info(f"saved {path}")


if __name__ == "__main__":
    main()
