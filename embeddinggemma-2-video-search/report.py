"""Tables and chart from every saved benchmark session: output/benchmark.md and latency.png.

Latency cells are the median of each session's median, with the min–max across sessions when
there is more than one (laptop runs drift between sessions, so one session is not a number).

    uv run python report.py
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "output"
LAT = ["vision_b8_ms", "vision_b1_ms", "encoder_b1_ms", "encoder_b4_per_clip_ms", "query_ms",
       "clip_end_to_end_ms"]  # fmt: skip
QUAL = ["nan", "cos_clip_median", "cos_clip_min", "cos_query_min", "top1", "top3", "mrr"]
DEVICE_COLOR = {"CPU": "#2a78d6", "iGPU f16": "#eb6834", "iGPU f32": "#1baf7a"}


def load() -> tuple[list[tuple[str, str]], dict, int]:
    sessions = sorted(OUT.glob("benchmark_session*.csv"))
    data: dict[tuple[str, str], list[dict]] = {}
    for path in sessions:
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            dev = r["device"].replace("GPU.0", "iGPU")
            data.setdefault((r["variant"], dev), []).append(r)
    return list(data), data, len(sessions)


def cell(rows: list[dict], k: str) -> str:
    v = np.array([float(r[k]) for r in rows if r.get(k)])
    if not len(v):
        return "–"
    if len(v) == 1:
        return f"{v[0]:,.0f}"
    return f"{np.median(v):,.0f} ({v.min():,.0f}–{v.max():,.0f})"


def main() -> None:
    keys, data, n = load()
    head = ["Model", "Device", "Vision, 8 frames/call", "Vision, 1 frame/call",
            "Encoder, 1 clip/call", "Encoder, 4 clips/call (per clip)", "Text query",
            "Clip, frames → vector"]  # fmt: skip
    lines = [
        "# EmbeddingGemma 2 on a laptop: latency and quality",
        "",
        f"{n} session(s). Latency in ms, median of 10 timed runs after 3 warm-ups, on one "
        "8-frame 640x360 clip (1 fps); with several sessions, the median across sessions and "
        "the min–max.",
        "",
        "| " + " | ".join(head) + " |",
        "|" + "---|" * len(head),
    ]
    for v, d in keys:
        lines.append(f"| {v} | {d} | " + " | ".join(cell(data[v, d], k) for k in LAT) + " |")
    qhead = ["Model", "Device", "NaN vectors", "Cosine vs PyTorch, median clip", "worst clip",
             "worst query", "Top-1 /23", "Top-3 /23", "MRR"]  # fmt: skip
    lines += [
        "",
        "Quality, latest session: 43 clips, 23 queries. (Session 1 ran before the padding fix,"
        " see ov_model.py.)",
        "",
        "| " + " | ".join(qhead) + " |",
        "|" + "---|" * len(qhead),
    ]
    for v, d in keys:
        r = data[v, d][-1]
        if not r.get("top1"):
            continue
        vals = [r["nan"], *(f"{float(r[k]):.4f}" for k in QUAL[1:4]), r["top1"], r["top3"],
                f"{float(r['mrr']):.3f}"]  # fmt: skip
        lines.append(f"| {v} | {d} | " + " | ".join(vals) + " |")
    (OUT / "benchmark.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # Chart: one clip, frames to vector, per configuration; whiskers = min–max across sessions.
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    labels = []
    for y, (v, d) in enumerate(reversed(keys)):
        t = np.array([float(r["clip_end_to_end_ms"]) for r in data[v, d]]) / 1000
        color = DEVICE_COLOR.get(d, "#8a8986")
        med = np.median(t)
        xerr = [[med - t.min()], [t.max() - med]] if len(t) > 1 else None
        ax.barh(y, med, height=0.7, color=color, xerr=xerr, ecolor="#52514e", capsize=3)
        ax.text(med + 0.12, y, f"{med:.1f} s", va="center", fontsize=8, color="#0b0b0b")
        labels.append(f"{v} · {d}")
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("seconds per 8-frame clip, frames → vector (lower is faster)", fontsize=9)
    ax.set_title("EmbeddingGemma 2, one 8-second clip on an i7-12700H laptop", fontsize=10,
                 loc="left")  # fmt: skip
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in DEVICE_COLOR.values()]
    ax.legend(handles, ["CPU", "iGPU (Iris Xe), f16", "iGPU (Iris Xe), f32"], fontsize=8,
              frameon=False, loc="lower right")  # fmt: skip
    fig.tight_layout()
    fig.savefig(OUT / "latency.png")


if __name__ == "__main__":
    main()
