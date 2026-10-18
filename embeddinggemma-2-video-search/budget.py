"""The vision token budget: how much picture each frame keeps, and what that costs.

The processor's `max_soft_tokens` sets how many tokens a frame is cut into: 70, 140 (the video
default, 120 tokens for a 640x360 frame) or 280. Fewer tokens means less work for the vision tower,
which takes ~90% of the time, and shorter sequences for the encoder. This script exports and
quantizes the vision graph at 70 and 280 (the encoder is shared), then measures, per budget:

- latency on the iGPU, int8 at f16: the laptop clip (8 frames), frames -> vector, 3 warm-ups and
  the median of 10, all budgets in the same session so the ratios are fair;
- search over the 43 clips (top-1, top-3, MRR), int8 on the iGPU and fp32 at f32 as the reference;
- how far int8 sits from fp32 at the same budget, and how far each budget sits from 140;
- the paired direction test (same frames reversed), fp32.

    uv run python budget.py --export     # once: vision graphs at 70 and 280, fp32 and int8
    uv run python budget.py              # -> output/budget.md, output/budget.csv
"""

import argparse
import csv
import json
import time
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch
from evaluate import ci
from loguru import logger
from ov_model import BUDGET_TOKENS, MODEL_ID, MODELS, VISION_BATCH, OVEmbedder, igpu
from smoke_export import VisionPath, sample_frames
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).parent
CLIPS, OUT = HERE / "data" / "clips", HERE / "output"
BUDGETS = (70, 140, 280)
LATENCY_KEY, WARMUP, RUNS, N_CALIB = "7431681", 3, 10, 16


def export(budget: int) -> None:
    """fp32 and int8 vision graphs for one budget, traced and calibrated like the 140 ones."""
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    frames = sample_frames(CLIPS / f"{clips[0]['key']}_fwd.mp4")
    frames = np.concatenate([frames, frames])[:VISION_BATCH]
    x = proc(text=["<|video|>"], videos=[frames], do_sample_frames=False, return_tensors="pt",
             max_soft_tokens=budget)  # fmt: skip
    vis = ov.convert_model(
        VisionPath(model).eval(), example_input=(x["pixel_values_videos"], x["video_position_ids"])
    )
    (MODELS / f"fp32_b{budget}").mkdir(parents=True, exist_ok=True)
    ov.save_model(vis, MODELS / f"fp32_b{budget}" / "vision.xml", compress_to_fp16=False)

    # Same calibration as quantize.py: 16 clips, padded to a full vision batch.
    ref = OVEmbedder("fp32", "CPU", budget=budget)
    items = []
    for c in clips[:: max(1, len(clips) // N_CALIB)][:N_CALIB]:
        p = ref.prompt(sample_frames(CLIPS / f"{c['key']}_fwd.mp4"))
        pv, pos = p["pixel_values_videos"], p["video_position_ids"]
        k = VISION_BATCH - len(pv)
        if k > 0:
            pv = np.concatenate([pv, np.repeat(pv[-1:], k, 0)])
            pos = np.concatenate([pos, np.repeat(pos[-1:], k, 0)])
        items.append((pv[:VISION_BATCH], pos[:VISION_BATCH]))
    q = nncf.quantize(ov.Core().read_model(MODELS / f"fp32_b{budget}" / "vision.xml"),
                      nncf.Dataset(items, lambda it: list(it)),
                      model_type=nncf.ModelType.TRANSFORMER, subset_size=len(items))  # fmt: skip
    (MODELS / f"int8_b{budget}").mkdir(exist_ok=True)
    ov.save_model(q, MODELS / f"int8_b{budget}" / "vision.xml", compress_to_fp16=False)
    logger.info(f"budget {budget}: fp32 and int8 vision saved")


def latency_ms(e: OVEmbedder, frames: np.ndarray) -> float:
    for _ in range(WARMUP):
        e.embed_videos([frames])
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        e.embed_videos([frames])
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def embed_all(e: OVEmbedder, clips: list[dict], spec: dict) -> dict[str, np.ndarray]:
    """Forward clips, the same frames reversed (from the same frame tokens), queries, captions."""
    out: dict[str, list] = {"fwd": [], "rfr": []}
    for c in clips:
        x = e.prompt(sample_frames(CLIPS / f"{c['key']}_fwd.mp4"))
        tok = e.frame_tokens(x["pixel_values_videos"], x["video_position_ids"])
        blocks = tok.reshape(-1, e.tok_per_frame, tok.shape[-1])
        ids = x["input_ids"][0]
        fwd, rfr = e.encode(
            [e.assemble(ids, tok), e.assemble(ids, blocks[::-1].reshape(tok.shape))]
        )
        out["fwd"].append(fwd)
        out["rfr"].append(rfr)
    res = {k: np.stack(v) for k, v in out.items()}
    res["query"] = np.stack([e.embed_text(q["query"]) for q in spec["queries"]])
    pk = [c for c in clips if c["kind"] == "paired"]
    res["tf"] = np.stack([e.embed_text(c["caption_fwd"]) for c in pk])
    res["tr"] = np.stack([e.embed_text(c["caption_rev"]) for c in pk])
    res["paired_idx"] = np.array([i for i, c in enumerate(clips) if c["kind"] == "paired"])
    return res


def search(R: dict, clips: list[dict], spec: dict) -> tuple[int, int, float]:
    ids = [c["id"] for c in clips]
    ranks = []
    for i, q in enumerate(spec["queries"]):
        order = np.argsort(-(R["fwd"] @ R["query"][i]))
        ranks.append(next(r for r, j in enumerate(order, 1) if ids[j] in q["targets"]))
    ranks = np.array(ranks)
    return int((ranks == 1).sum()), int((ranks <= 3).sum()), float(np.mean(1 / ranks))


def measure() -> None:
    spec = json.loads((HERE / "clips.json").read_text())
    clips = spec["clips"]
    gpu = igpu()
    lat_frames = sample_frames(CLIPS / f"{LATENCY_KEY}_fwd.mp4")
    rows, refs = [], {}
    for b in BUDGETS:
        ref = OVEmbedder("fp32", gpu, "f32", budget=b)
        refs[b] = embed_all(ref, clips, spec)
        del ref
        e = OVEmbedder("int8", gpu, budget=b)
        ms = latency_ms(e, lat_frames)
        x = e.prompt(lat_frames)
        vis_ms = latency_ms_vision(e, x)
        R = embed_all(e, clips, spec)
        del e
        P = refs[b]
        pi = P["paired_idx"]
        ok = (((P["fwd"][pi] - P["rfr"][pi]) * (P["tf"] - P["tr"])).sum(1) > 0).tolist()
        t1, t3, mrr = search(R, clips, spec)
        r1, r3, rmrr = search(P, clips, spec)
        rows.append({
            "budget": b, "tokens_per_frame": BUDGET_TOKENS[b],
            "clip_ms_int8_igpu_f16": round(ms), "vision_ms_int8_igpu_f16": round(vis_ms),
            "sequence_tokens": len(x["input_ids"][0]),
            "top1_int8": t1, "top3_int8": t3, "mrr_int8": round(mrr, 3),
            "top1_fp32": r1, "top3_fp32": r3, "mrr_fp32": round(rmrr, 3),
            "cos_int8_vs_fp32_median": float(np.median((R["fwd"] * P["fwd"]).sum(1))),
            "cos_int8_vs_fp32_min": float((R["fwd"] * P["fwd"]).sum(1).min()),
            "paired_right": sum(ok), "paired_n": len(ok),
            "cos_fwd_rfr_median": float(np.median((P["fwd"] * P["rfr"]).sum(1))),
        })  # fmt: skip
        logger.info(rows[-1])
    for r in rows:
        same = (refs[r["budget"]]["fwd"] * refs[140]["fwd"]).sum(1)
        r["cos_vs_budget140_median"] = float(np.median(same))
    write(rows)


def latency_ms_vision(e: OVEmbedder, x: dict) -> float:
    pv, pos = x["pixel_values_videos"], x["video_position_ids"]
    for _ in range(WARMUP):
        e.frame_tokens(pv, pos)
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        e.frame_tokens(pv, pos)
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def write(rows: list[dict]) -> None:
    OUT.mkdir(exist_ok=True)
    with open(OUT / "budget.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    lines = [
        "# Vision token budget per frame",
        "",
        f"The laptop clip ({LATENCY_KEY}, 8 frames at 640x360), int8 on the iGPU at f16: {WARMUP} "
        f"warm-ups, median of {RUNS}, all budgets in one session. Search over 43 clips and 23 "
        "queries; fp32 is the iGPU at f32 (matches PyTorch at cosine 1.0000 at budget 140).",
        "",
        "| Budget | Tokens per frame | Sequence | Clip, frames → vector (ms) | Vision (ms) | "
        "Top-1 int8 | Top-1 fp32 | MRR int8 | MRR fp32 | int8 vs fp32 cos (median, worst) | "
        "vs budget 140 cos (median) | Paired, fp32 | fwd vs reversed cos (median) |",
        "|" + "---|" * 13,
    ]
    for r in rows:
        lines.append(
            f"| {r['budget']} | {r['tokens_per_frame']} | {r['sequence_tokens']} | "
            f"{r['clip_ms_int8_igpu_f16']:,} | {r['vision_ms_int8_igpu_f16']:,} | "
            f"{r['top1_int8']}/23 | {r['top1_fp32']}/23 | "
            f"{r['mrr_int8']:.3f} | {r['mrr_fp32']:.3f} | "
            f"{r['cos_int8_vs_fp32_median']:.4f}, {r['cos_int8_vs_fp32_min']:.4f} | "
            f"{r['cos_vs_budget140_median']:.4f} | {ci(r['paired_right'], r['paired_n'])} | "
            f"{r['cos_fwd_rfr_median']:.4f} |"
        )
    (OUT / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()
    if args.export:
        for b in BUDGETS:
            if b != 140 and not (MODELS / f"int8_b{b}" / "vision.xml").exists():
                export(b)
    else:
        measure()


if __name__ == "__main__":
    main()
