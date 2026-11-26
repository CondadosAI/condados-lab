"""Exact chance AP: the average precision a random ranking scores, in expectation.

The tables from evaluate.py and qvh.py print the share of positive windows as "chance AP".
That is the expected precision of a random pick (and the exact chance of Hit@1), but a random
*ranking* scores higher AP than that, more so when there are few windows and few positives.
For N windows of which R are positive, a uniformly random ranking has

    E[AP] = (R - 1) / (N - 1) + (N - R) / (N (N - 1)) * H_N,    H_N = 1 + 1/2 + ... + 1/N

(R = N gives 1; R = 1 gives H_N / N, the mean of 1/rank). This script recomputes chance with
that formula on the same windows and positives the two evaluators use, and checks it against
shuffled rankings.

    uv run python chance_ap.py          # -> output/chance_ap.md

It reads only Marengo index files: Gemma ran on Marengo's segments, so the windows, the
positives and therefore chance are the same for both models on a game.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA, INDEX, OUT = HERE / "data", HERE / "index", HERE / "output"
RNG = np.random.default_rng(0)
GROUPS = {  # as evaluate.py
    "shot_attempt": {"three", "made_two", "dunk", "miss"},
    "vehicle_turns": {"vehicle_turns_left", "vehicle_turns_right"},
}


def expected_ap(n: int, r: int) -> float:
    h = np.sum(1.0 / np.arange(1, n + 1))
    if n == 1:
        return 1.0
    return (r - 1) / (n - 1) + (n - r) / (n * (n - 1)) * h


def shuffled_ap(pos: np.ndarray, draws: int = 2000) -> float:
    out = []
    for _ in range(draws):
        h = pos[RNG.permutation(len(pos))]
        prec = np.cumsum(h) / np.arange(1, len(h) + 1)
        out.append((prec * h).sum() / h.sum())
    return float(np.mean(out))


def game_positives(index: Path, key: str, delta: float, min_moments: int = 3) -> dict:
    """The windows and per-class positives exactly as evaluate.py builds them."""
    games = json.loads((HERE / "games.json").read_text())
    d = np.load(index)
    start, length = d["start"], float(d["length"])
    end = d["end"] if "end" in d else start + length
    centers = (start + end) / 2
    moments = json.loads((DATA / f"{key}_moments.json").read_text())
    spans = {}
    for m in moments:
        lo, hi = spans.get(m["half"], (m["t"], m["t"]))
        spans[m["half"]] = (min(lo, m["t"]), max(hi, m["t"]))
    keep = np.zeros(len(start), bool)
    for lo, hi in spans.values():
        keep |= (centers >= lo - 30) & (centers <= hi + 30)
    start, end, centers = start[keep], end[keep], centers[keep]
    moments = [m for m in moments if centers.min() <= m["t"] <= centers.max()]
    q = json.loads((HERE / games["games"][key].get("queries", "queries.json")).read_text())
    out = {}
    for cls in q["classes"]:
        ev = [m for m in moments if m["cls"] in GROUPS.get(cls, {cls})]
        if len(ev) < min_moments:
            continue
        sp = np.array([(m["lo"], m["hi"]) for m in ev])
        lo, hi = sp[:, 0] - delta, sp[:, 1] + delta
        out[cls] = (
            len(ev),
            np.array([((hi >= s) & (lo <= e)).any() for s, e in zip(start, end, strict=True)]),
        )
    return out


def qvh_rows() -> tuple[list[float], list[float], list[float]]:
    sample = json.loads((HERE / "qvh_sample.json").read_text())
    d = np.load(INDEX / "qvh_marengo_visual.npz")
    share, exact, shuf = [], [], []
    for q in sample["queries"]:
        rows = np.where(d["vid"] == q["vid"])[0]
        rows = rows[~np.isnan(d["emb"][rows]).any(axis=1)]
        s, e = d["start"][rows], d["end"][rows]
        pos = np.array(
            [any(a < we and b > ws for ws, we in q["relevant_windows"]) for a, b in zip(s, e)]
        )
        if not pos.any():
            continue
        share.append(pos.mean())
        exact.append(expected_ap(len(pos), int(pos.sum())))
        shuf.append(shuffled_ap(pos, 500))
    return share, exact, shuf


def main() -> None:
    lines = [
        "# Chance AP: share of positives vs the expected AP of a random ranking",
        "",
        "Exact = closed form (see the docstring); shuffled = mean AP over random rankings, a check.",
        "",
        "## QVHighlights (per query, mean over queries, percent)",
        "",
        "| queries | share of positive segments | exact random AP | shuffled (500 per query) |",
        "|---|---|---|---|",
    ]
    share, exact, shuf = qvh_rows()
    lines.append(
        f"| {len(share)} | {100 * np.mean(share):.1f} | {100 * np.mean(exact):.1f} "
        f"| {100 * np.mean(shuf):.1f} |"
    )
    runs = [
        ("loras", "loras-mask_marengo_visual.npz", 5.0),
        ("misericordia", "misericordia-mask_marengo_visual.npz", 5.0),
        ("meva", "meva-full_marengo_visual.npz", 2.0),
    ]
    for key, idx, delta in runs:
        lines += [
            "",
            f"## {key}, windows = {idx}, delta = {delta:g} s",
            "",
            "| class | n | windows | positive | share | exact random AP | shuffled (2000) |",
            "|---|---|---|---|---|---|---|",
        ]
        for cls, (n, pos) in game_positives(INDEX / idx, key, delta).items():
            lines.append(
                f"| {cls} | {n} | {len(pos)} | {int(pos.sum())} | {pos.mean():.2f} "
                f"| {expected_ap(len(pos), int(pos.sum())):.2f} | {shuffled_ap(pos):.2f} |"
            )
    OUT.mkdir(exist_ok=True)
    (OUT / "chance_ap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
