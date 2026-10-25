"""Score text queries against the windows, per class, with chance measured on the same game.

Two questions per class:

- Window AP: rank every window of the game by its score; a window is positive when one of the
  class's moments falls inside it (widened by --delta on each side). Chance AP is the share of
  positive windows, which is high when events are dense: that is why it is printed next to it.
- Spotting: keep score peaks at least 2*delta apart, take the top k (k = the number of moments
  of the class) and count the moments that have a peak within delta of their time. Chance is
  the same count with the scores shuffled across windows, 200 times.

Scorers: `raw` = cosine with the query; `contrast` = cosine with the query minus cosine with a
generic "basketball game" query, which removes what every window shares with the game.
Each class has three phrasings; the table reports the mean and the spread.

    uv run python evaluate.py index/<file>.npz          # -> output/<file>.md

An index may also carry its own per-window `end` (segments of varying length) and its own text
vectors (`texts`, `text_emb`), as marengo.py writes for TwelveLabs Marengo: then no Gemma model
is loaded.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

HERE = Path(__file__).parent
DATA, OUT = HERE / "data", HERE / "output"
MODEL_ID = "google/embeddinggemma-2"
QUERY_PREFIX = "task: search result | query: "
RNG = np.random.default_rng(0)
# A query class can stand for several logged classes: any shot, made or missed.
GROUPS = {
    "shot_attempt": {"three", "made_two", "dunk", "miss"},
    "vehicle_turns": {"vehicle_turns_left", "vehicle_turns_right"},
}


def embed_texts(texts: list[str], device: str) -> np.ndarray:
    torch.set_grad_enabled(False)
    model = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32).eval().to(device)
    proc = AutoProcessor.from_pretrained(MODEL_ID)
    x = proc(text=[QUERY_PREFIX + t for t in texts], padding=True, return_tensors="pt")
    x = {k: v.to(device) for k, v in x.items()}
    h = model(**x).last_hidden_state.float().cpu().numpy()
    m = x["attention_mask"].cpu().numpy()[..., None].astype(np.float32)
    v = (h * m).sum(1) / m.sum(1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def average_precision(scores: np.ndarray, positive: np.ndarray) -> float:
    order = np.argsort(-scores)
    hits = positive[order]
    if hits.sum() == 0:
        return float("nan")
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float((prec * hits).sum() / hits.sum())


def peaks(scores: np.ndarray, centers: np.ndarray, k: int, gap: float) -> list[float]:
    """Greedy non-maximum suppression: the k best windows at least `gap` seconds apart."""
    taken: list[float] = []
    for i in np.argsort(-scores):
        if all(abs(centers[i] - c) >= gap for c in taken):
            taken.append(float(centers[i]))
            if len(taken) == k:
                break
    return taken


def spotted(pk: list[float], spans: np.ndarray, delta: float) -> int:
    """Moments with a peak inside their span widened by delta; each peak claims one moment."""
    used, n = set(), 0
    for lo, hi in spans:
        best = [j for j, c in enumerate(pk) if j not in used and lo - delta <= c <= hi + delta]
        if best:
            used.add(min(best, key=lambda j: abs(pk[j] - lo)))
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index")
    ap.add_argument("--game", default=None)
    ap.add_argument("--delta", type=float, default=3.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-moments", type=int, default=3)
    ap.add_argument("--half", type=int, default=None, help="score one half only")
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]

    d = np.load(args.index)
    start, emb, length = d["start"], d["emb"], float(d["length"])
    end = d["end"] if "end" in d else start + length
    centers = (start + end) / 2
    moments = json.loads((DATA / f"{key}_moments.json").read_text())
    # Only windows inside a half: no warm-up, half-time show or post-game.
    spans = {}
    for m in moments:
        if args.half and m["half"] != args.half:
            continue
        lo, hi = spans.get(m["half"], (m["t"], m["t"]))
        spans[m["half"]] = (min(lo, m["t"]), max(hi, m["t"]))
    keep = np.zeros(len(start), bool)
    for lo, hi in spans.values():
        keep |= (centers >= lo - 30) & (centers <= hi + 30)
    start, end, centers, emb = start[keep], end[keep], centers[keep], emb[keep]
    moments = [m for m in moments if centers.min() <= m["t"] <= centers.max()]

    q = json.loads((HERE / games["games"][key].get("queries", "queries.json")).read_text())
    texts = [q["background"]] + [t for v in q["classes"].values() for t in v]
    if "text_emb" in d:
        tv = dict(zip(d["texts"].tolist(), d["text_emb"], strict=True))
    else:
        tv = dict(zip(texts, embed_texts(texts, args.device), strict=True))
    bg = emb @ tv[q["background"]]

    lines = [
        f"# {Path(args.index).stem}",
        "",
        f"{len(start)} windows of {length:g} s (mean) inside the halves; delta = {args.delta:g} s.",
        "",
        "| class | n | positive windows | AP raw | AP contrast | chance AP "
        "| spotted raw | spotted contrast | chance spotted (p95) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for cls, phrasings in q["classes"].items():
        # Each moment's accepted span comes from align.py: a point for live play, the whole
        # stoppage for a timeout or substitution, the score tick for a made free throw.
        ev = sorted(
            (m for m in moments if m["cls"] in GROUPS.get(cls, {cls})), key=lambda m: m["t"]
        )
        spans_c = np.array([(m["lo"], m["hi"]) for m in ev])
        if len(ev) < args.min_moments:
            continue
        lo, hi = spans_c[:, 0] - args.delta, spans_c[:, 1] + args.delta
        pos = np.array([((hi >= s) & (lo <= e)).any() for s, e in zip(start, end, strict=True)])
        res = {"raw": [], "contrast": []}
        for p in phrasings:
            raw = emb @ tv[p]
            for name, sc in (("raw", raw), ("contrast", raw - bg)):
                pk = peaks(sc, centers, len(ev), 2 * args.delta)
                res[name].append((average_precision(sc, pos), spotted(pk, spans_c, args.delta)))
        chance = []
        for _ in range(200):
            pk = peaks(RNG.permutation(len(start)).astype(float), centers, len(ev), 2 * args.delta)
            chance.append(spotted(pk, spans_c, args.delta))

        def fmt(vals, i, pct=False):
            v = np.array([r[i] for r in vals], float)
            return (
                f"{v.mean():.2f} ({v.min():.2f}-{v.max():.2f})"
                if not pct
                else (f"{v.mean():.1f} ({v.min():.0f}-{v.max():.0f})")
            )

        lines.append(
            f"| {cls} | {len(ev)} | {pos.mean():.0%} | {fmt(res['raw'], 0)} "
            f"| {fmt(res['contrast'], 0)} | {pos.mean():.2f} | {fmt(res['raw'], 1, True)} "
            f"| {fmt(res['contrast'], 1, True)} "
            f"| {np.mean(chance):.1f} ({np.percentile(chance, 95):.0f}) |"
        )
    lines += [
        "",
        "AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a "
        "peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.",
    ]
    OUT.mkdir(exist_ok=True)
    out = OUT / f"{Path(args.index).stem}_d{args.delta:g}{f'_h{args.half}' if args.half else ''}.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
