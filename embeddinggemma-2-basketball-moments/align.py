"""Play-by-play (game clock) + score-bug readings (video time) -> moments.json in video time.

Rows that share one half and clock can be several things on screen. A live-play moment (a shot,
a turnover, with its rebound, assist, block or steal) is one moment, classed by its most
visible row (dunk > three > made two > turnover > miss ...). A stoppage row (foul, timeout,
substitution, each free throw) is its own moment, because it happens while the clock is
stopped, seconds to a minute after the play.

Placing a moment in the video (measured on the primary game, 2026-10-21):

- Live play: the bug's clock is in step with the game clock, and the log's clock is right at
  made baskets (the bug's score ticks up 4-6 s after the logged time, the operator's typing
  lag). So `t` = the first frame whose clock shows the logged value.
- Stoppages: the log is a few seconds off (a timeout logged at 18:45 starts when the bug clock
  stops at 18:41). So they snap to the first stoppage on the bug (the same clock on screen for
  3 s or more) within 10 s of game clock after the logged value. A foul is the start of the
  stoppage; a timeout or substitution spans it; a made free throw is the moment the score ticks
  up on the bug (minus 3 s of lag); a missed one spans the stoppage.

Every moment carries [`lo`, `hi`] (the span the evaluation accepts) and `t` (its reference
time). Readings are cleaned first: the pre-game countdown and the post-game reset also read
"1st", so a half starts at the last 20:00 shown before the next half, and a reading that runs
the clock backwards is dropped as an OCR error.

    uv run python align.py                       # -> data/<game>_moments.json
"""

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"


def readings(key: str) -> list[dict]:
    rows = []
    with (DATA / f"{key}_clock.csv").open() as fh:
        for r in csv.DictReader(fh):
            if r["half"] and r["clock"]:
                rows.append(
                    {
                        "t": float(r["t"]),
                        "half": int(r["half"]),
                        "clock": float(r["clock"]),
                        "a": int(r["score_a"]) if r["score_a"] else None,
                        "b": int(r["score_b"]) if r["score_b"] else None,
                    }
                )
    return rows


def half_runs(rows: list[dict], half: int) -> list[dict]:
    """Readings of one half, from its tip-off on, with clock jumps removed."""
    # The bug also reads "1st" during the pre-game countdown and after the final buzzer (reset to
    # 20:00), so a half's readings end where the next half's begin, and its tip-off is the last
    # 20:00 shown before that: the clock never returns to 20:00 once a half has started.
    nxt = min((r["t"] for r in rows if r["half"] > half), default=float("inf"))
    hr = [r for r in rows if r["half"] == half and r["t"] < nxt]
    tips = [i for i, r in enumerate(hr) if r["clock"] >= 1199]
    if not tips:
        return []
    tip = tips[-1]
    out, last = [], None
    for r in hr[tip:]:
        # The game clock only goes down within a half; allow 2 s of OCR wobble.
        if last is not None and r["clock"] > last["clock"] + 2:
            continue
        out.append(r)
        last = r
    return out


PLAY_ORDER = [
    "dunk",
    "three",
    "made_two",
    "turnover",
    "miss",
    "steal",
    "block",
    "rebound",
    "assist",
]
STOPPAGE = {"foul", "timeout", "substitution", "free_throw"}
FT_LAG = 3.0  # seconds from a made free throw to the score ticking up on the bug


def collapse(events: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for e in events:
        if e["clock"] is not None:
            groups.setdefault((e["half"], e["clock"]), []).append(e)
    moments = []
    for (half, clock), rows in groups.items():
        play = [r for r in rows if r["cls"] in PLAY_ORDER]
        if play:
            top = min(play, key=lambda r: PLAY_ORDER.index(r["cls"]))
            moments.append({"half": half, "clock": clock, "cls": top["cls"], "side": top["side"],
                            "rows": [r["text"] for r in play], "score": top["score"]})  # fmt: skip
        stop = [r for r in rows if r["cls"] in STOPPAGE]
        for cls in ("foul", "timeout", "substitution"):
            these = [r for r in stop if r["cls"] == cls]
            if these:
                moments.append({"half": half, "clock": clock, "cls": cls, "side": these[0]["side"],
                                "rows": [r["text"] for r in these]})  # fmt: skip
        for r in (r for r in stop if r["cls"] == "free_throw"):
            moments.append({"half": half, "clock": clock, "cls": "free_throw", "side": r["side"],
                            "rows": [r["text"]], "score": r["score"],
                            "made": r["text"].startswith("GOOD")})  # fmt: skip
    return moments


def stoppages(run: list[dict], min_s: float = 3.0) -> list[dict]:
    """Stretches where the bug shows one clock value for at least `min_s` seconds."""
    out, i = [], 0
    while i < len(run):
        j = i
        while j + 1 < len(run) and run[j + 1]["clock"] == run[i]["clock"]:
            j += 1
        if run[j]["t"] - run[i]["t"] >= min_s:
            out.append({"clock": run[i]["clock"], "t0": run[i]["t"], "t1": run[j]["t"]})
        i = j + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None)
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]
    events = json.loads((DATA / f"{key}_events.json").read_text())
    rows = readings(key)
    runs = {h: half_runs(rows, h) for h in (1, 2, 3)}
    moments = collapse(events)

    stops = {h: stoppages(r) for h, r in runs.items()}
    placed, missing = [], 0
    for m in moments:
        hr = runs.get(m["half"], [])
        if m["cls"] not in STOPPAGE:
            # Under a minute the bug shows tenths (45.3) and the log whole seconds (0:45).
            on = [r for r in hr if 0 <= r["clock"] - m["clock"] < 1]
            if not on:
                missing += 1
                continue
            m["t"] = m["lo"] = m["hi"] = on[0]["t"]
            placed.append(m)
            continue
        cand = [
            s for s in stops.get(m["half"], []) if m["clock"] - 10 <= s["clock"] <= m["clock"] + 1
        ]
        if not cand:
            missing += 1
            continue
        s = max(cand, key=lambda s: s["clock"])
        m["stop"] = [s["t0"], s["t1"]]
        if m["cls"] == "foul":
            m["t"] = m["lo"] = m["hi"] = s["t0"]
        elif m["cls"] == "free_throw" and m["made"]:
            # The first frame in the stoppage (or just after) showing this row's score.
            a, b = m["score"]
            hit = [r["t"] for r in hr if s["t0"] - 1 <= r["t"] <= s["t1"] + 5
                   and (r["a"], r["b"]) == (a, b)]  # fmt: skip
            if hit:
                m["t"] = hit[0] - FT_LAG
                m["lo"], m["hi"] = m["t"] - 1, hit[0]
            else:
                m["t"], m["lo"], m["hi"] = (s["t0"] + s["t1"]) / 2, s["t0"], s["t1"]
        else:
            m["t"], m["lo"], m["hi"] = s["t0"], s["t0"], s["t1"]
        placed.append(m)

    placed.sort(key=lambda m: m["t"])
    out = DATA / f"{key}_moments.json"
    out.write_text(json.dumps(placed, indent=1))
    counts: dict[str, int] = {}
    for m in placed:
        counts[m["cls"]] = counts.get(m["cls"], 0) + 1
    halves = {h: (r[0]["t"], r[-1]["t"]) if r else None for h, r in runs.items()}
    print(f"halves (video s): {halves}")
    print(f"{len(placed)} moments placed, {missing} not found on the bug -> {out.name}")
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
