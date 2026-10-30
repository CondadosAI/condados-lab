"""Why do some clean fits pass under the net? Refit them under each suspect.

``check_net.py`` found flights with a pixel residual near 1 px whose fitted arc
crosses the net plane below the tape. A ball that went under the net would not
have landed in the other court, so the fit is wrong while matching every
pixel: it put the ball at the wrong depth along each viewing ray. These are
the suspects, each refitted on the same flights:

- **drag:** held at 0.05 per metre in production; here 0, the two ends of the
  rule-derived range (0.038, 0.061), and free.
- **the paddle time:** the event's own time (no window), and every candidate
  in the window with its cost and clearance, to see whether a near-equal
  minimum clears the net.
- **the second bounce's time:** free within the frames around the detected one.
- **topspin:** an extra vertical acceleration on the leg after the paddle, left
  free (a topspin drive dips faster than gravity alone).
- **the lens:** the pinhole camera instead of the radial one.

Plus one check without any fit: where each detection's ray meets the net plane.
The ball crossed the plane at some instant, on that instant's ray, so if the
rays meet the plane above the tape before the fitted crossing, the camera and
the detections allow a ball that clears it.

Writes ``output/diagnose_net.json``. A few minutes on 20 cores.

    uv run python pickleball-3d-ball/diagnose_net.py
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from check_net import BALL_RADIUS_M, crossing, net_height
from fit import GRAVITY, HERE, STEP_S, fit_flight, load, propagate
from scipy.optimize import least_squares

OUT = HERE / "output"
BASE = {"gravity": GRAVITY, "fit_drag": False, "drag": 0.05, "hit_window_s": 0.15}
# The 8 flights with a median residual of 3 px or less that miss the net by more
# than 5 cm in output/net.json.
FLIGHTS_UNDER = [4, 50, 61, 65, 67, 69, 71, 76]

CAMERA, FLIGHTS, DATA = load()
PINHOLE = load(camera="pinhole")[0]
COURT = DATA["court"]


def clearance(fit: dict) -> dict:
    t, point = crossing(fit, COURT["net_x_m"])
    if not np.isfinite(t):
        return {"t_cross": None, "clearance_m": None}
    net = net_height(point[1], COURT) + BALL_RADIUS_M
    return {"t_cross": round(t, 4), "clearance_m": round(float(point[2] - net), 3)}


def summarise(fit: dict) -> dict:
    keep = ("g_m_s2", "k_per_m", "median_px", "cost", "hit_t", "bounce2_t", "apex_m")
    return {k: round(fit[k], 4) for k in keep} | clearance(fit)


# --- topspin: a second gravity on the leg after the paddle -------------------


def _fit_spin(flight, k: float) -> dict:
    """Like ``fit._fit_once`` with g fixed on the first leg and free on the second."""
    from fit import FLOOR_PX_PER_M, _initial

    x0 = np.r_[_initial(CAMERA, flight, GRAVITY), GRAVITY]
    leg1 = flight.times < flight.hit_t
    times1, times2 = flight.times[leg1], flight.times[~leg1]

    def path(p):
        start = np.r_[p[0], p[1], 0.0]
        first, end1 = propagate(
            start, p[2:5], flight.bounce1_t, np.r_[times1, flight.hit_t], GRAVITY, k
        )
        second, _ = propagate(
            end1[:3], p[5:8], flight.hit_t, np.r_[times2, flight.bounce2_t], p[8], k
        )
        return start, first[:-1], first[-1], second[:-1], second[-1]

    def residuals(p):
        with np.errstate(all="ignore"):
            start, first, paddle, second, landing = path(p)
            seen = CAMERA.to_image(np.vstack([start, first, second, landing]))
        observed = np.vstack([flight.bounce1_px, flight.pixels, flight.bounce2_px])
        below = np.minimum(0.0, np.r_[first[:, 2], paddle[2], second[:, 2]])
        out = np.r_[(seen - observed).ravel(), FLOOR_PX_PER_M * landing[2], FLOOR_PX_PER_M * below]
        return np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)

    lower = np.r_[[-np.inf] * 8, 0.0]
    upper = np.r_[[np.inf] * 8, 40.0]
    result = least_squares(
        residuals,
        np.clip(x0, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=2.0,
        x_scale="jac",
    )
    start, first, paddle, second, _ = path(result.x)
    error = np.linalg.norm(CAMERA.to_image(np.vstack([first, second])) - flight.pixels, axis=1)
    return {
        "g2_m_s2": float(result.x[8]),
        "median_px": float(np.median(error)),
        "cost": float(result.cost),
        "hit_t": float(flight.hit_t),
        "bounce2_t": float(flight.bounce2_t),
        "start": start,
        "v1": result.x[2:5],
        "paddle": paddle,
        "v2": result.x[5:8],
        "k": k,
    }


def spin_clearance(fit: dict) -> dict:
    times = np.linspace(fit["hit_t"], fit["bounce2_t"], 400)
    path, _ = propagate(fit["paddle"], fit["v2"], fit["hit_t"], times, fit["g2_m_s2"], fit["k"])
    side = np.sign(path[:, 0] - COURT["net_x_m"])
    i = np.flatnonzero(side[1:] != side[:-1])
    if len(i) == 0:
        return {"t_cross": None, "clearance_m": None}
    i = i[0]
    s = (COURT["net_x_m"] - path[i, 0]) / (path[i + 1, 0] - path[i, 0])
    point = path[i] + s * (path[i + 1] - path[i])
    net = net_height(point[1], COURT) + BALL_RADIUS_M
    return {"t_cross": round(float(times[i]), 4), "clearance_m": round(float(point[2] - net), 3)}


def fit_spin(flight) -> dict:
    best = None
    for shift in np.arange(-BASE["hit_window_s"], BASE["hit_window_s"] + 1e-9, STEP_S):
        hit_t = flight.hit_t + shift
        if (flight.times < hit_t).sum() < 2 or (flight.times > hit_t).sum() < 2:
            continue
        fit = _fit_spin(replace(flight, hit_t=hit_t), BASE["drag"])
        if best is None or fit["cost"] < best["cost"]:
            best = fit
    return {
        "g2_m_s2": round(best["g2_m_s2"], 3),
        "median_px": round(best["median_px"], 3),
        "cost": round(best["cost"], 4),
        "hit_t": round(best["hit_t"], 4),
    } | spin_clearance(best)


# --- the variants ------------------------------------------------------------


def variant(job: tuple[int, str]) -> dict:
    n, name = job
    flight = FLIGHTS[n]
    if name == "base":
        out = summarise(fit_flight(CAMERA, flight, **BASE))
    elif name.startswith("k="):
        out = summarise(fit_flight(CAMERA, flight, **BASE | {"drag": float(name[2:])}))
    elif name == "k free":
        out = summarise(fit_flight(CAMERA, flight, **BASE | {"fit_drag": True}))
    elif name == "g free":
        out = summarise(fit_flight(CAMERA, flight, **BASE | {"fit_gravity": True}))
    elif name == "event hit":
        out = summarise(fit_flight(CAMERA, flight, **BASE | {"hit_window_s": 0.0}))
    elif name == "pinhole":
        out = summarise(fit_flight(PINHOLE, flight, **BASE))
    elif name == "topspin":
        out = fit_spin(flight)
    elif name == "hit landscape":
        # Every paddle time in the window on its own, to see the minima side by side.
        out = []
        for shift in np.arange(-0.15, 0.15 + 1e-9, STEP_S):
            hit_t = flight.hit_t + shift
            if (flight.times < hit_t).sum() < 2 or (flight.times > hit_t).sum() < 2:
                continue
            out.append(
                summarise(
                    fit_flight(CAMERA, replace(flight, hit_t=hit_t), **BASE | {"hit_window_s": 0.0})
                )
            )
    elif name == "bounce2 free":
        # The second bounce no earlier than the last detection before it, up to 4 frames late.
        last = flight.times.max()
        best = None
        for shift in np.arange(-0.1, 0.134, 1 / DATA["fps"]):
            t = flight.bounce2_t + shift
            if t <= last:
                continue
            fit = fit_flight(CAMERA, replace(flight, bounce2_t=t), **BASE)
            if best is None or fit["cost"] < best["cost"]:
                best = fit
        out = summarise(best) | {"bounce2_shift_s": round(best["bounce2_t"] - flight.bounce2_t, 4)}
    else:
        raise ValueError(name)
    return {"flight": n, "variant": name, "result": out}


VARIANTS = [
    "base",
    "k=0.0",
    "k=0.038",
    "k=0.061",
    "k free",
    "g free",
    "event hit",
    "hit landscape",
    "bounce2 free",
    "topspin",
    "pinhole",
]


def rays_at_net(n: int) -> list[dict]:
    """Model-free: the height where each detection's ray meets the net plane."""
    flight = FLIGHTS[n]
    origin, direction = CAMERA.ray(flight.pixels)
    s = (COURT["net_x_m"] - origin[0]) / direction[:, 0]
    meet = origin + s[:, None] * direction
    return [
        {
            "t": round(float(t), 4),
            "y_m": round(float(p[1]), 3),
            "z_m": round(float(p[2]), 3),
            "net_m": round(net_height(p[1], COURT) + BALL_RADIUS_M, 3),
        }
        for t, p in zip(flight.times, meet, strict=True)
        if t > flight.hit_t
    ]


def main() -> None:
    flights = [int(a) for a in sys.argv[1:]] or FLIGHTS_UNDER
    jobs = [(n, v) for n in flights for v in VARIANTS]
    workers = int(os.environ.get("WORKERS", "0")) or None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(variant, jobs))
    table = {n: {"rays_at_net": rays_at_net(n)} for n in flights}
    for r in results:
        table[r["flight"]][r["variant"]] = r["result"]
    OUT.mkdir(exist_ok=True)
    (OUT / "diagnose_net.json").write_text(json.dumps(table, indent=1))
    print(f"{'flight':>6} " + " ".join(f"{v[:12]:>12}" for v in VARIANTS if v != "hit landscape"))
    for n in flights:
        cells = []
        for v in VARIANTS:
            if v == "hit landscape":
                continue
            r = table[n][v]
            c = r["clearance_m"]
            cells.append(f"{'-' if c is None else f'{c:+.2f}'}/{r['median_px']:.1f}px".rjust(12))
        print(f"{n:>6} " + " ".join(cells))


if __name__ == "__main__":
    main()
