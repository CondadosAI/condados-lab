"""Leave gravity free in every flight fit, and see whether the flights give 9.81 back.

Nothing in the fit is told the value of g. If the camera, the bounce anchoring
and the drag model are right, the median over many flights should land near
9.81 m/s^2; if the camera's lens model is wrong, it lands somewhere else, and a
synthetic run says where.

1. **Synthetic control:** groundstrokes simulated at g = 9.81, k = 0.05 through
   one camera, with heavy-tailed pixel noise and half a frame of jitter on each
   contact time, then fitted through each camera.
2. **The 80 real groundstrokes**, fitted through each camera with drag held at
   0.05 per metre and the paddle time free within 0.15 s of the detected hit.

Writes ``output/gravity.json`` (the summary the post quotes) and
``output/fits_<camera>.json`` (one fit per flight, which ``check_net.py`` and
``render.py`` read). About 20 minutes on 20 cores; set WORKERS to cap the pool.

    uv run python pickleball-3d-ball/check_gravity.py
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from fit import HERE, Groundstroke, fit_flight, load, propagate

OUT = HERE / "output"
OPTIONS = {"fit_gravity": True, "fit_drag": False, "drag": 0.05, "hit_window_s": 0.15}
SYNTHETIC_PER_CAMERA = 91

_, FLIGHTS, DATA = load()
CAMERAS = {name: load(camera=name)[0] for name in DATA["cameras"]}
FPS = DATA["fps"]
COURT = DATA["court"]


def synthetic(seed: int, camera, g: float = 9.81, k: float = 0.05) -> Groundstroke:
    """A bounce, a paddle and a bounce across the net, seen through ``camera``.

    Half of them fly far to near, as the real ones do.
    """
    rng = np.random.default_rng(seed)
    start = np.array([rng.uniform(1.0, 4.0), rng.uniform(0.8, 5.3), 0.0])
    v1 = np.array([rng.uniform(-6, -2), rng.uniform(-1, 1), rng.uniform(2.5, 5)])
    target = np.array([rng.uniform(8.5, 12.5), rng.uniform(0.8, 5.3)])
    if rng.uniform() < 0.5:
        length = 2 * COURT["net_x_m"]
        start[0], v1[0], target[0] = length - start[0], -v1[0], length - target[0]
    t0 = rng.uniform(0, 1 / FPS)
    t_hit = t0 + rng.uniform(0.25, 0.6)
    _, state = propagate(start, v1, t0, np.array([t_hit]), g, k)
    paddle = state[:3]
    v2 = np.r_[(target - paddle[:2]) * rng.uniform(1.0, 1.4), rng.uniform(1.5, 4.5)]
    probe = t_hit + np.arange(1, 4000) / 1000
    path, _ = propagate(paddle, v2, t_hit, probe, g, k)
    t_land = probe[np.argmax(path[:, 2] < 0)]
    t = np.arange(np.ceil(t0 * FPS) + 1, np.floor(t_land * FPS)) / FPS
    t = t[np.abs(t - t_hit) > 0.5 / FPS]
    leg1, _ = propagate(start, v1, t0, t[t < t_hit], g, k)
    leg2, _ = propagate(paddle, v2, t_hit, t[t > t_hit], g, k)
    _, landing = propagate(paddle, v2, t_hit, np.array([t_land]), g, k)
    # Heavy tails, scaled so the median error is near the measured 1.7 px.
    pixels = camera.to_image(np.vstack([leg1, leg2])) + 0.7 * rng.standard_t(3, (len(t), 2))
    jitter = lambda: rng.uniform(-0.5, 0.5) / FPS  # noqa: E731
    return Groundstroke(
        t0 + jitter(),
        camera.to_image(start[None])[0],
        t_hit + jitter(),
        t_land + jitter(),
        camera.to_image(np.r_[landing[:2], 0][None])[0],
        t,
        pixels,
    )


def _synthetic_job(job):
    seed, truth = job
    flight = synthetic(seed, CAMERAS[truth])
    fits = {name: fit_flight(cam, flight, **OPTIONS) for name, cam in CAMERAS.items()}
    return truth, {name: fit["g_m_s2"] for name, fit in fits.items()}


def _real_job(i):
    return {name: fit_flight(cam, FLIGHTS[i], **OPTIONS) for name, cam in CAMERAS.items()}


def summary(g, rng) -> dict:
    """Median g, its bootstrap standard error, and the interquartile range."""
    g = np.asarray(g)
    boot = np.std([np.median(rng.choice(g, len(g))) for _ in range(2000)])
    q1, q3 = np.percentile(g, [25, 75])
    return {
        "median": round(float(np.median(g)), 3),
        "se": round(float(boot), 3),
        "iqr": [round(float(q1), 2), round(float(q3), 2)],
        "n": int(len(g)),
    }


def main() -> None:
    rng = np.random.default_rng(0)
    OUT.mkdir(exist_ok=True)
    workers = int(os.environ.get("WORKERS", "0")) or None
    result = {"options": OPTIONS, "true_g": 9.81, "synthetic": {}, "real": {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [(seed, truth) for truth in CAMERAS for seed in range(SYNTHETIC_PER_CAMERA)]
        done = list(pool.map(_synthetic_job, jobs))
        for truth in CAMERAS:
            for name in CAMERAS:
                g = [fits[name] for t, fits in done if t == truth]
                result["synthetic"][f"true {truth}, fitted {name}"] = summary(g, rng)

        real = list(pool.map(_real_job, range(len(FLIGHTS))))
    for name in CAMERAS:
        fits = [r[name] for r in real]
        (OUT / f"fits_{name}.json").write_text(json.dumps(fits, indent=1))
        result["real"][name] = {
            **summary([f["g_m_s2"] for f in fits], rng),
            "median_residual_px": round(float(np.median([f["median_px"] for f in fits])), 2),
        }
    (OUT / "gravity.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
