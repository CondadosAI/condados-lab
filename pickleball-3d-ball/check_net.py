"""Does every fitted flight clear the net?

A second check that the fit is not told about, after gravity: each of these
groundstrokes crossed the net and came down in the other court, so its 3D arc
has to be above the net where it crosses: its centre at least one ball radius
over the tape. A fit that passes under the net is wrong, however small its
pixel residual.

The net is 0.864 m at the centre and 0.914 m at the posts, 0.3048 m outside each
sideline (USA Pickleball rules 2.B). Between the two the height is interpolated
linearly, which slightly overstates the net near the sidelines (a real net sags
in a curve), so the check errs on the strict side. The ball is 7.29-7.54 cm
across (rule 2.D); the radius used is half the smaller.

A miss is split by size. At the 1.6 px median residual the fit's height is good
to a few centimetres, so a miss under 5 cm is within its noise, and a miss of
tens of centimetres is a fit that went wrong.

Reads ``output/fits.json`` from ``fit_all.py`` (or the file named on the command
line, e.g. ``output/fits_radial.json`` for the g-free fits); writes
``output/net.json`` (or ``net_<stem>.json``).

    uv run python pickleball-3d-ball/check_net.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fit import HERE, load, trajectory

OUT = HERE / "output"
POST_OUTSIDE_M = 0.3048
BALL_RADIUS_M = 0.0729 / 2
NOISE_M = 0.05


def net_height(y: float, court: dict) -> float:
    centre, half = court["width_m"] / 2, court["width_m"] / 2 + POST_OUTSIDE_M
    frac = min(abs(y - centre) / half, 1.0)
    return float(
        court["net_height_centre_m"]
        + frac * (court["net_height_post_m"] - court["net_height_centre_m"])
    )


def crossing(fit: dict, net_x: float) -> tuple[float, np.ndarray]:
    """The time and 3D point where the flight after the paddle crosses the net plane."""
    times = np.linspace(fit["hit_t"], fit["bounce2_t"], 400)
    path = trajectory(fit, times)
    side = np.sign(path[:, 0] - net_x)
    i = np.flatnonzero(side[1:] != side[:-1])
    if len(i) == 0:
        return float("nan"), np.full(3, np.nan)
    i = i[0]
    s = (net_x - path[i, 0]) / (path[i + 1, 0] - path[i, 0])
    return float(times[i] + s * (times[i + 1] - times[i])), path[i] + s * (path[i + 1] - path[i])


def main() -> None:
    _, _, data = load()
    court = data["court"]
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "fits.json"
    fits = json.loads(source.read_text())
    rows = []
    for n, fit in enumerate(fits):
        t, point = crossing(fit, court["net_x_m"])
        net = net_height(point[1], court) + BALL_RADIUS_M if np.isfinite(t) else float("nan")
        rows.append(
            {
                "flight": n,
                "t_cross": round(t, 4),
                "y_m": round(float(point[1]), 3),
                "height_m": round(float(point[2]), 3),
                "net_m": round(net, 3),
                "clearance_m": round(float(point[2] - net), 3),
                "median_px": round(fit["median_px"], 2),
            }
        )
    clearance = np.array([r["clearance_m"] for r in rows])
    crossed = np.isfinite(clearance)
    missed = clearance[crossed]
    result = {
        "source": source.name,
        "flights": len(rows),
        "cross_the_net": int(crossed.sum()),
        "clear_it": int((missed >= 0).sum()),
        "miss_within_noise": int(((missed < 0) & (missed >= -NOISE_M)).sum()),
        "miss_beyond_noise": int((missed < -NOISE_M).sum()),
        "clearance_m": {
            "min": round(float(clearance[crossed].min()), 3),
            "p10": round(float(np.percentile(clearance[crossed], 10)), 3),
            "median": round(float(np.median(clearance[crossed])), 3),
        },
        "apex_m": {
            "median": round(float(np.median([f["apex_m"] for f in fits])), 2),
            "max": round(float(max(f["apex_m"] for f in fits)), 2),
        },
        "rows": rows,
    }
    name = "net.json" if source.name == "fits.json" else f"net_{source.stem}.json"
    (OUT / name).write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    for r in rows:
        if not r["clearance_m"] >= 0:
            print("does not clear:", r)


if __name__ == "__main__":
    main()
