"""Do you need a vector database yet? Exact search with numpy, from 43 clips to a million.

Two measurements, written to output/scale.md:

- speed: one query against N random unit vectors (float32), a matrix-vector product plus a top-10
  partial sort, at N = 43 ... 1,000,000 and 768 or 256 dimensions. The timing does not depend on
  what the vectors contain, so random ones stand in for a large corpus. Median of 20 runs after 3
  warm-ups, on one CPU thread: with all 20 threads of this hybrid laptop CPU, OpenBLAS adds
  stalls of ~3 ms at random sizes, and one query per core is also how a server would run it.
- quality: the real index (PyTorch reference) searched with the vectors cut to 768, 512, 256 and
  128 dimensions and re-normalised, the Matryoshka sizes the model card lists.

    uv run python scale.py
"""

import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).parent
OUT = HERE / "output"
SIZES = (43, 1_000, 10_000, 100_000, 1_000_000)
DIMS = (768, 256)
K, WARMUP, RUNS = 10, 3, 20


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def search_ms(V: np.ndarray, q: np.ndarray) -> float:
    def once() -> np.ndarray:
        s = V @ q
        top = np.argpartition(-s, K)[:K] if len(s) > K else np.arange(len(s))
        return top[np.argsort(-s[top])]

    for _ in range(WARMUP):
        once()
    t = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        once()
        t.append(time.perf_counter() - t0)
    return float(np.median(t) * 1000)


def main() -> None:
    rng = np.random.default_rng(0)
    lines = [
        "# Exact search, no index",
        "",
        f"One query, top-{K}, float32 numpy, one CPU thread (i7-12700H); median of {RUNS} runs "
        f"after {WARMUP} warm-ups.",
        "",
        "| Vectors | " + " | ".join(f"{d}-d: ms per query | {d}-d: memory" for d in DIMS) + " |",
        "|---|" + "---|---|" * len(DIMS),
    ]
    for n in SIZES:
        cells = []
        with threadpool_limits(1):
            for d in DIMS:
                V = unit(rng.standard_normal((n, d), dtype=np.float32))
                q = unit(rng.standard_normal(d, dtype=np.float32))
                cells += [f"{search_ms(V, q):.3f}", f"{V.nbytes / 2**20:,.2f} MB"]
                del V
        lines.append(f"| {n:,} | " + " | ".join(cells) + " |")
        print(lines[-1])

    spec = json.loads((HERE / "clips.json").read_text())
    E = np.load(HERE / "index" / "pytorch_fp32_fps1.npz")
    ids = [c["id"] for c in spec["clips"]]
    V = np.stack([E[f"fwd/{c['key']}"] for c in spec["clips"]])
    Q = np.stack([E[f"query/{i}"] for i in range(len(spec["queries"]))])
    lines += [
        "",
        f"Search quality with shorter vectors: {len(Q)} queries over {len(V)} clips.",
        "",
        "| Dimensions | Top-1 | Top-3 | MRR |",
        "|---|---|---|---|",
    ]
    for d in (768, 512, 256, 128):
        S = unit(Q[:, :d]) @ unit(V[:, :d]).T
        ranks = np.array([
            next(r for r, j in enumerate(np.argsort(-s), 1) if ids[j] in q["targets"])
            for s, q in zip(S, spec["queries"], strict=True)
        ])  # fmt: skip
        lines.append(f"| {d} | {(ranks == 1).sum()}/{len(Q)} | {(ranks <= 3).sum()}/{len(Q)} | "
                     f"{np.mean(1 / ranks):.3f} |")  # fmt: skip
    (OUT / "scale.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
