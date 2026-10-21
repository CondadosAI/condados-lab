"""Read the broadcast score bug every half second: video time -> (half, game clock, score).

The play-by-play is timed in game clock, and the game clock stops for every foul, free throw
and timeout while the video keeps running, so no straight line maps one onto the other. The
score bug prints the clock on screen; reading it is the alignment.

Each field sits at a fixed place in the bug (pixel boxes per game in games.json), so only the
recogniser runs, on five small crops, with no text detection. Frames without a readable clock
(replays, ads, the half-time show) come back empty and are left out of the map.

    uv run python clock_ocr.py                 # primary game -> data/<game>_clock.csv
    uv run python clock_ocr.py --every 1.0
"""

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
_ocr = None


def _init() -> None:
    global _ocr
    from rapidocr import RapidOCR

    _ocr = RapidOCR(params={"Global.log_level": "error"})


def _read(job: tuple[float, dict[str, np.ndarray]]) -> dict:
    t, crops = job
    row = {"t": round(t, 2)}
    for name, im in crops.items():
        r = _ocr(im, use_det=False, use_cls=False)
        row[name] = (r.txts[0] if r.txts else "").strip()
    return row


def parse_clock(s: str) -> float | None:
    """'6:12' -> 372; '45.3' (last minute, tenths) -> 45.3; anything else -> None."""
    s = s.replace(" ", "").replace(";", ":")
    if m := re.fullmatch(r"(\d{1,2}):(\d\d)", s):
        mm, ss = int(m[1]), int(m[2])
        return mm * 60 + ss if mm <= 20 and ss < 60 else None
    if m := re.fullmatch(r"(\d{1,2})\.(\d)", s):
        return int(m[1]) + int(m[2]) / 10 if int(m[1]) < 60 else None
    return None


def parse_half(s: str) -> int | None:
    s = s.lower()
    if s.startswith("1"):
        return 1
    if s.startswith("2"):
        return 2
    if s.startswith("ot") or s.startswith("0t"):
        return 3
    return None


def frames(path: Path, every: float):
    """Yield (video time, frame) for the frame nearest each `every`-second tick."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        nxt = 0.0
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t + 1e-6 >= nxt:
                yield t, f.to_ndarray(format="rgb24")
                nxt += every


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default=None)
    ap.add_argument("--every", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=float, default=None, help="stop after this many seconds")
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())
    key = args.game or games["primary"]
    boxes = games["games"][key]["bug_boxes"]  # name -> [x0, y0, x1, y1] in source pixels

    def jobs():
        for t, im in frames(DATA / games["games"][key]["video"], args.every):
            if args.limit and t > args.limit:
                return
            # BGR for the recogniser, like cv2.imread would give it.
            yield t, {n: im[y0:y1, x0:x1, ::-1].copy() for n, (x0, y0, x1, y1) in boxes.items()}

    out = DATA / f"{key}_clock.csv"
    n = ok = 0
    with ProcessPoolExecutor(args.workers, initializer=_init) as ex, out.open("w") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "half", "clock", "score_a", "score_b", "shot", "raw_clock"])
        for row in ex.map(_read, jobs(), chunksize=8):
            clock, half = parse_clock(row["clock"]), parse_half(row["period"])
            a, b = row["score_a"], row["score_b"]
            w.writerow(
                [
                    row["t"],
                    half if half else "",
                    "" if clock is None else clock,
                    a if a.isdigit() else "",
                    b if b.isdigit() else "",
                    row.get("shot", "") if row.get("shot", "").isdigit() else "",
                    row["clock"],
                ]
            )
            n += 1
            ok += clock is not None and half is not None
            if n % 2000 == 0:
                print(f"{row['t']:.0f} s: {ok}/{n} frames with half and clock", flush=True)
    print(f"{ok}/{n} frames with half and clock -> {out.name}")


if __name__ == "__main__":
    main()
