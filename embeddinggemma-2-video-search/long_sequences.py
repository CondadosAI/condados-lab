"""Does int8 hold on longer clips? The int8 encoder on the iGPU, sequence by sequence.

Every other number here uses clips of at most 8 frames (~980 tokens). Longer clips make longer
sequences: at the default budget, 2 fps gives 1,344-1,710 tokens on the first three clips and
4 fps gives 2,686-3,418. Each configuration below embeds those three clips at 2 and 4 fps and is
compared with fp32 on the CPU (cosine 1.0 to PyTorch):

- int8 on the iGPU at f16 (the fastest configuration in benchmark.md) and at f32;
- int8 on the CPU;
- int8 vision with the fp32 encoder on the iGPU at f16.

Then the price of the safe pair: the laptop clip (8 frames), frames -> vector, 3 warm-ups and the
median of 10, int8 alone vs int8 vision with the fp32 encoder, at token budgets 140 and 70.

Each configuration runs in its own process: one model at a time keeps memory to a few GB (a single
fp32 model at 4 fps peaks near 7 GB; several at once do not fit a 32 GB laptop).

    uv run python long_sequences.py            # every configuration, one process each
    uv run python long_sequences.py --report   # -> output/long_sequences.md
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
CACHE, OUT = HERE / "cache" / "long", HERE / "output"
FPS = (2, 4)
LATENCY = [(name, b) for b in (140, 70)
           for name in ("int8_igpu_f16", "int8_vision_fp32_encoder_igpu_f16")]  # fmt: skip
N_CLIPS = 3
# name -> (vision and encoder variant, device, precision hint, encoder variant override)
CONFIGS = {
    "fp32_cpu": ("fp32", "CPU", None, None),
    "int8_igpu_f16": ("int8", "iGPU", None, None),
    "int8_igpu_f32": ("int8", "iGPU", "f32", None),
    "int8_cpu": ("int8", "CPU", None, None),
    "int8_vision_fp32_encoder_igpu_f16": ("int8", "iGPU", None, "fp32"),
}


def run(name: str, fps: int) -> None:
    from ov_model import OVEmbedder, igpu
    from smoke_export import sample_frames

    variant, device, precision, enc = CONFIGS[name]
    e = OVEmbedder(variant, igpu() if device == "iGPU" else "CPU", precision, enc_variant=enc)
    res = {}
    for c in json.loads((HERE / "clips.json").read_text())["clips"][:N_CLIPS]:
        emb = e.video_inputs(sample_frames(HERE / "data" / "clips" / f"{c['key']}_fwd.mp4", fps))
        res[c["key"]], res[f"len/{c['key']}"] = e.encode([emb])[0], np.array(len(emb))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / f"{name}_{fps}fps.npz", **res)
    print(f"{name} {fps} fps: saved")


def latency(name: str, budget: int) -> None:
    import time

    from ov_model import OVEmbedder, igpu
    from smoke_export import sample_frames

    variant, _, precision, enc = CONFIGS[name]
    e = OVEmbedder(variant, igpu(), precision, budget=budget, enc_variant=enc)
    frames = sample_frames(HERE / "data" / "clips" / "7431681_fwd.mp4")
    for _ in range(3):
        e.embed_videos([frames])
    t = []
    for _ in range(10):
        t0 = time.perf_counter()
        e.embed_videos([frames])
        t.append(time.perf_counter() - t0)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"lat_{name}_b{budget}.json").write_text(json.dumps(float(np.median(t) * 1000)))
    print(f"{name} budget {budget}: {np.median(t) * 1000:.0f} ms")


def report() -> None:
    lines = [
        "# int8 on longer sequences",
        "",
        f"First {N_CLIPS} clips at 2 and 4 fps, default token budget (120 per frame). Cosine of "
        "each clip vector to fp32 on the CPU.",
        "",
        "| Configuration | fps | Tokens per clip | Cosine to fp32, per clip |",
        "|---|---|---|---|",
    ]
    for fps in FPS:
        ref = dict(np.load(CACHE / f"fp32_cpu_{fps}fps.npz"))
        keys = [k for k in ref if not k.startswith("len/")]
        tokens = ", ".join(str(int(ref[f"len/{k}"])) for k in keys)
        for name in CONFIGS:
            if name == "fp32_cpu" or not (CACHE / f"{name}_{fps}fps.npz").exists():
                continue
            x = dict(np.load(CACHE / f"{name}_{fps}fps.npz"))
            cos = ", ".join(f"{float(x[k] @ ref[k]):.4f}" for k in keys)
            lines.append(f"| {name} | {fps} | {tokens} | {cos} |")
    lines += [
        "",
        "Laptop clip, 8 frames, iGPU at f16, one session:",
        "",
        "| Token budget | int8 alone (ms) | int8 vision + fp32 encoder (ms) |",
        "|---|---|---|",
    ]
    for b in (140, 70):
        ms = [
            json.loads((CACHE / f"lat_{n}_b{b}.json").read_text()) for n, bb in LATENCY if bb == b
        ]
        lines.append(f"| {b} | {ms[0]:,.0f} | {ms[1]:,.0f} |")
    OUT.mkdir(exist_ok=True)
    (OUT / "long_sequences.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--one", nargs=2, metavar=("CONFIG", "FPS"), help="run one, in this process")
    ap.add_argument("--latency", nargs=2, metavar=("CONFIG", "BUDGET"), help="time one")
    args = ap.parse_args()
    if args.report:
        report()
    elif args.one:
        run(args.one[0], int(args.one[1]))
    elif args.latency:
        latency(args.latency[0], int(args.latency[1]))
    else:
        for name in CONFIGS:
            for fps in FPS:
                if not (CACHE / f"{name}_{fps}fps.npz").exists():
                    subprocess.run([sys.executable, __file__, "--one", name, str(fps)], check=True)
        for name, b in LATENCY:
            if not (CACHE / f"lat_{name}_b{b}.json").exists():
                subprocess.run([sys.executable, __file__, "--latency", name, str(b)], check=True)
        report()


if __name__ == "__main__":
    main()
