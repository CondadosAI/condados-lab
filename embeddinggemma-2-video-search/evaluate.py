"""Score an embedding index: text-to-video search, then the direction-of-time test.

Search: each query ranks the 43 forward clips; a hit is any clip in its `targets`.

Direction, three readings, from strict to lenient:

- paired: for a reversible clip, does the difference between the clip and its reverse point the
  same way as the difference between the two captions? sign(dot(v_fwd - v_rev, t_fwd - t_rev)).
  It cancels everything the two versions share (scene, objects, people) and keeps only what
  changed. Chance is 50%.
- ranking: does the forward caption score the forward clip above the reversed one? Weaker,
  because the two clips are nearly the same vector; reported for comparison with other posts.
- one-way clips (a reverse that cannot happen): the ranking reading only.

Plus how far apart the versions are at all: cosine fwd-rev and fwd-shuffled, against the cosine
between two different clips of the corpus for scale.

    uv run python evaluate.py index/pytorch_fp32_fps1.npz
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).parent
OUT = HERE / "output"


def ci(k: int, n: int) -> str:
    r = binomtest(k, n, 0.5)
    lo, hi = r.proportion_ci(0.95, method="exact")
    return f"{k}/{n} = {k / n:.0%} (95% CI {lo:.0%}–{hi:.0%}, p = {r.pvalue:.2f} vs chance)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index", type=Path)
    args = ap.parse_args()
    E = dict(np.load(args.index))
    spec = json.loads((HERE / "clips.json").read_text())
    clips = spec["clips"]
    tag = args.index.stem
    keys = [c["key"] for c in clips]
    ids = [c["id"] for c in clips]
    V = np.stack([E[f"fwd/{k}"] for k in keys])
    lines = [f"# Results for `{tag}`", ""]

    # --- search -------------------------------------------------------------------------
    rows, rr = [], []
    for i, q in enumerate(spec["queries"]):
        s = V @ E[f"query/{i}"]
        order = np.argsort(-s)
        rank = next(r for r, j in enumerate(order, 1) if ids[j] in q["targets"])
        rr.append(1 / rank)
        top = clips[order[0]]
        rows.append((q["query"], rank, top["action"], f"{s[order[0]]:.3f}"))
    ranks = np.array([r[1] for r in rows])
    lines += [
        "## Search",
        "",
        f"{len(rows)} queries over {len(V)} clips: top-1 {(ranks == 1).sum()}/{len(rows)}, "
        f"top-3 {(ranks <= 3).sum()}/{len(rows)}, MRR {np.mean(rr):.3f}",
        "",
        "| Query | Rank of the right clip | Top result | Score |",
        "|---|---|---|---|",
        *[f"| {q} | {r} | {a} | {s} |" for q, r, a, s in rows],
        "",
    ]

    # --- direction ----------------------------------------------------------------------
    off = V @ V.T
    between = off[~np.eye(len(V), dtype=bool)]
    table = {c["key"]: {"key": c["key"], "kind": c["kind"], "action": c["action"]} for c in clips}
    lines += ["## Direction of time", ""]
    spreads = []
    for rv, label in (("rfr", "same frames, reversed order"), ("rev", "reversed file, resampled")):
        if f"{rv}/{keys[0]}" not in E:
            continue
        paired_ok, rank_ok, oneway_ok, cosines = [], [], [], []
        for c in clips:
            k = c["key"]
            f, r, tf = E[f"fwd/{k}"], E[f"{rv}/{k}"], E[f"cap_fwd/{k}"]
            row = table[k]
            row[f"cos_fwd_{rv}"] = float(f @ r)
            row[f"capdiff_{rv}"] = float(tf @ f - tf @ r)
            cosines.append(float(f @ r))
            if c["kind"] == "paired":
                row[f"paired_dot_{rv}"] = float((f - r) @ (tf - E[f"cap_rev/{k}"]))
                paired_ok.append(row[f"paired_dot_{rv}"] > 0)
                rank_ok.append(row[f"capdiff_{rv}"] > 0)
            else:
                oneway_ok.append(row[f"capdiff_{rv}"] > 0)
        spreads.append((f"clip vs {label}", np.array(cosines)))
        lines += [
            f"### Reverse = {label} (`{rv}`)",
            "",
            f"- Paired (reversible clips): {ci(sum(paired_ok), len(paired_ok))}",
            f"- Ranking, reversible clips: {ci(sum(rank_ok), len(rank_ok))}",
            f"- Ranking, one-way clips: {ci(sum(oneway_ok), len(oneway_ok))}",
            "",
        ]
    for c in clips:
        table[c["key"]]["cos_fwd_shf"] = float(E[f"fwd/{c['key']}"] @ E[f"shf/{c['key']}"])
    spreads.append(
        ("clip vs its frames shuffled", np.array([t["cos_fwd_shf"] for t in table.values()]))
    )
    spreads.append(("two different clips", between))
    lines += [
        "How far apart the versions are (cosine):",
        "",
        "| Pair | Median | Min | Max |",
        "|---|---|---|---|",
        *[f"| {n} | {np.median(x):.4f} | {x.min():.4f} | {x.max():.4f} |" for n, x in spreads],
        "",
    ]
    table = list(table.values())
    mirrors = [("7431681", "7431863", "laptop opened vs closed"), ("39173954a", "39173954b", "mug")]
    lines += ["Filmed mirrors (the same set, the action really done both ways):", ""]
    for a, b, what in mirrors:
        if f"fwd/{a}" in E and f"fwd/{b}" in E:
            lines.append(
                f"- {what}: cos(rev {a}, real {b}) = {E[f'rev/{a}'] @ E[f'fwd/{b}']:.4f}, "
                f"cos(fwd {a}, real {b}) = {E[f'fwd/{a}'] @ E[f'fwd/{b}']:.4f}"
            )
    lines.append("")

    OUT.mkdir(exist_ok=True)
    (OUT / f"results_{tag}.md").write_text("\n".join(lines))
    with open(OUT / f"direction_{tag}.csv", "w", newline="") as fh:
        fields = list(dict.fromkeys(k for row in table for k in row))
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(table)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
