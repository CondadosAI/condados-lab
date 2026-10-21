"""Official play-by-play → events.json: one row per logged action, with its half and game clock.

NCAA athletics sites run Sidearm, whose box score page carries the whole log as HTML tables,
one per half, with `Time Remaining` per row. A `--` time means "same moment as the row above".
The log is the ground truth for every class in this post; nobody labels events by hand.

The raw page is not committed (it is the school's); this script fetches it again.

    uv run python fetch_log.py                  # the primary game (games.json)
    uv run python fetch_log.py --game loras
"""

import argparse
import html
import json
import re
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
# Cloudflare-fronted sites refuse Python-urllib's default agent.
UA = "Mozilla/5.0 (condados-lab; +https://condados.ai)"

# First match wins: a row's class is what a viewer would call the moment.
CLASSES = [
    ("dunk", r"GOOD DUNK"),
    ("three", r"GOOD 3PTR"),
    ("free_throw", r"(GOOD|MISS) FT"),
    ("timeout", r"TIMEOUT"),
    ("substitution", r"SUB (IN|OUT)"),
    ("turnover", r"TURNOVER"),
    ("foul", r"FOUL"),
    ("made_two", r"GOOD (LAYUP|JUMPER|TIPIN)"),
    ("miss", r"MISS (LAYUP|JUMPER|3PTR|DUNK|TIPIN)"),
    ("rebound", r"REBOUND"),
    ("steal", r"STEAL"),
    ("block", r"BLOCK"),
    ("assist", r"ASSIST"),
]

ROW = re.compile(r"<tr>\s*<td>([^<]*)</td>(.*?)</tr>", re.S)
CELL = re.compile(r"<td[^>]*>(.*?)</td>", re.S)


def classify(text: str) -> str:
    for name, rx in CLASSES:
        if re.search(rx, text):
            return name
    return "other"


def parse(page: str) -> list[dict]:
    events = []
    for half in (1, 2, 3, 4, 5):  # periods 3+ are overtimes
        m = re.search(rf'<div id="period-{half}".*?</table>', page, re.S)
        if not m:
            break
        last = None
        for t, rest in ROW.findall(m.group(0)):
            cells = [html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in CELL.findall(rest)]
            text = " ".join(cells[-1].split())
            if not text:
                continue
            t = t.strip()
            if re.fullmatch(r"\d+:\d\d", t):
                mm, ss = t.split(":")
                last = int(mm) * 60 + int(ss)
            # Which side acted: the mobile team cell carries "team" or "opponent" in its class.
            side = "opponent" if 'play opponent"></td>' in rest else "team"
            # Scoring rows fill the two bold score cells (left team, right team) of the wide layout.
            score = None
            if (
                len(cells) >= 4
                and (sl := re.match(r"\d+", cells[1]))
                and (sr := re.match(r"\d+", cells[3]))
            ):
                score = [int(sl[0]), int(sr[0])]
            events.append(
                {
                    "half": half,
                    "clock": last,
                    "side": side,
                    "cls": classify(text),
                    "text": text,
                    "score": score,
                }
            )
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None)
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]
    g = games["games"][key]
    req = urllib.request.Request(g["log_url"], headers={"User-Agent": UA})
    page = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
    events = parse(page)
    DATA.mkdir(exist_ok=True)
    out = DATA / f"{key}_events.json"
    out.write_text(json.dumps(events, indent=1))
    counts: dict[str, int] = {}
    for e in events:
        counts[e["cls"]] = counts.get(e["cls"], 0) + 1
    print(f"{len(events)} rows -> {out.name}")
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
