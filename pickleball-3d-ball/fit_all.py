"""Fit every groundstroke with gravity known: the arcs the post draws.

``check_gravity.py`` leaves g free to test the method; this is the product.
Gravity is 9.81, drag is held at 0.05 per metre, and the paddle time may move
within 0.15 s of the detected hit. Writes ``output/fits.json``, which
``check_net.py`` and ``render.py`` read. About 2 minutes on 20 cores.

    uv run python pickleball-3d-ball/fit_all.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

from fit import GRAVITY, HERE, fit_flight, load

OPTIONS = {"gravity": GRAVITY, "fit_drag": False, "drag": 0.05, "hit_window_s": 0.15}
CAMERA, FLIGHTS, _ = load()


def _job(i: int) -> dict:
    return fit_flight(CAMERA, FLIGHTS[i], **OPTIONS)


def main() -> None:
    workers = int(os.environ.get("WORKERS", "0")) or None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        fits = list(pool.map(_job, range(len(FLIGHTS))))
    (HERE / "output").mkdir(exist_ok=True)
    (HERE / "output" / "fits.json").write_text(json.dumps(fits, indent=1))
    print(f"{len(fits)} flights fitted -> output/fits.json")


if __name__ == "__main__":
    main()
