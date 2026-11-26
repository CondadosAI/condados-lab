"""A pickleball's 3D flight from one calibrated camera, with gravity and drag.

Each detection puts the ball somewhere on one viewing ray. Between a bounce and
the next bounce the ball is a projectile, so one trajectory has to pass through
all the rays of that flight, and least squares finds it.

One camera cannot see scale: scale a trajectory about the camera centre and
every pixel stays where it was, and the scaled path is still a projectile under
a gravity scaled by the same factor. A bounce fixes that, because the ball is on
the floor there and a scaled path lifts it off. So every flight is a whole
groundstroke, bounce - paddle - bounce, anchored on both bounces.

Drag is quadratic, a = -g z - k |v| v. A pickleball is perforated, so it is not
negligible, and one leg alone cannot tell k from g; the two legs of a
groundstroke share both. Spin is not modelled.

Court coordinates: x along the court (net at x = 6.705 m), y across, z up, in
metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

GRAVITY = 9.81
#: RK4 step: half a frame at 29.97 fps. Against a 1/960 s reference over 1.5 s
#: flights up to 25 m/s, even a whole frame stays within a micrometre.
STEP_S = 1.0 / 60.0
#: Weight of "the second bounce is on the floor" and "nothing goes under the
#: floor", in pixels per metre, so they sit in the same units as the pixels.
FLOOR_PX_PER_M = 200.0

HERE = Path(__file__).parent
DATA = HERE / "flights.json"


# --- the camera -------------------------------------------------------------


def rodrigues(rvec) -> np.ndarray:
    """Rotation matrix from an axis-angle vector (what OpenCV calls Rodrigues)."""
    rvec = np.asarray(rvec, float)
    angle = np.linalg.norm(rvec)
    if angle < 1e-12:
        return np.eye(3)
    kx, ky, kz = rvec / angle
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


@dataclass(frozen=True)
class Camera:
    """A pinhole camera with one radial distortion term, fitted to the court lines.

    ``rvec``/``tvec`` map court points (with height flipped to point down) into
    the camera frame, the OpenCV convention.
    """

    focal_px: float
    principal_point: tuple[float, float]
    k1: float
    rvec: tuple[float, float, float]
    tvec: tuple[float, float, float]

    @classmethod
    def from_dict(cls, d: dict) -> Camera:
        return cls(
            d["focal_px"], tuple(d["principal_point"]), d["k1"], tuple(d["rvec"]), tuple(d["tvec"])
        )

    @property
    def R(self) -> np.ndarray:
        return rodrigues(self.rvec)

    @property
    def position(self) -> np.ndarray:
        """The camera centre in court coordinates, height up."""
        c = -self.R.T @ np.asarray(self.tvec, float)
        return np.array([c[0], c[1], -c[2]])

    def to_image(self, xyz: np.ndarray) -> np.ndarray:
        """Court points (N, 3) to pixels (N, 2)."""
        p = np.asarray(xyz, float).reshape(-1, 3) * [1, 1, -1]
        cam = p @ self.R.T + np.asarray(self.tvec, float)
        n = cam[:, :2] / cam[:, 2:3]
        n = n * (1 + self.k1 * np.sum(n**2, axis=1, keepdims=True))
        return self.focal_px * n + np.asarray(self.principal_point)

    def ray(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The viewing ray of each pixel: the camera centre and unit directions.

        A ball seen at a pixel is somewhere on this ray. Which point is what the
        flight fit decides.
        """
        d = (np.asarray(pixels, float).reshape(-1, 2) - self.principal_point) / self.focal_px
        n = d.copy()
        for _ in range(20):  # invert the radial term by fixed-point iteration
            n = d / (1 + self.k1 * np.sum(n**2, axis=1, keepdims=True))
        rays = np.c_[n, np.ones(len(n))] @ self.R
        rays[:, 2] = -rays[:, 2]
        return self.position, rays / np.linalg.norm(rays, axis=1, keepdims=True)

    def on_floor(self, pixel) -> np.ndarray:
        """Where a pixel's ray meets the floor (z = 0)."""
        origin, direction = self.ray(pixel)
        return origin + (-origin[2] / direction[0, 2]) * direction[0]


# --- the physics ------------------------------------------------------------


def _derivative(state: np.ndarray, g: float, k: float) -> np.ndarray:
    v = state[3:]
    a = -k * np.linalg.norm(v) * v
    a[2] -= g
    return np.r_[v, a]


def propagate(position, velocity, t0: float, times, g: float, k: float):
    """Positions at ``times`` (sorted, all >= t0) and the final state, by RK4.

    The step is shortened so that it lands exactly on each requested time.
    """
    state = np.r_[np.asarray(position, float), np.asarray(velocity, float)]
    out = np.empty((len(times), 3))
    now = t0
    for i, target in enumerate(times):
        span = target - now
        if span > 0:
            steps = int(np.ceil(span / STEP_S))
            h = span / steps
            for _ in range(steps):
                k1 = _derivative(state, g, k)
                k2 = _derivative(state + h / 2 * k1, g, k)
                k3 = _derivative(state + h / 2 * k2, g, k)
                k4 = _derivative(state + h * k3, g, k)
                state = state + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            now = target
        out[i] = state[:3]
    return out, state


# --- one groundstroke -------------------------------------------------------


@dataclass
class Groundstroke:
    """What the camera saw of one bounce - paddle - bounce flight.

    Times in seconds from the start of the video; pixels in the 1920x1080
    frame. The contact frames are left out of ``times``: on a contact the ball
    is in a paddle or on the floor and belongs to neither leg.
    """

    bounce1_t: float
    bounce1_px: np.ndarray
    hit_t: float
    bounce2_t: float
    bounce2_px: np.ndarray
    times: np.ndarray
    pixels: np.ndarray


def load(path: Path = DATA, camera: str = "radial") -> tuple[Camera, list[Groundstroke], dict]:
    """A camera, the 80 groundstrokes and the rest of the file (court, fps, credit).

    Two cameras were fitted to the court lines: ``radial`` (one distortion term,
    the one every published number uses) and ``pinhole`` (none).
    """
    data = json.loads(Path(path).read_text())
    flights = [
        Groundstroke(
            f["bounce1_t"],
            np.array(f["bounce1_px"]),
            f["hit_t"],
            f["bounce2_t"],
            np.array(f["bounce2_px"]),
            np.array(f["times"]),
            np.array(f["pixels"]),
        )
        for f in data["flights"]
    ]
    return Camera.from_dict(data["cameras"][camera]), flights, data


def _initial(camera: Camera, flight: Groundstroke, g: float) -> np.ndarray:
    """A starting guess: bounces on the floor, the paddle on its ray above the first bounce.

    The player hits the ball that just bounced beside her, so the point of the
    paddle-time ray closest to the vertical over the first bounce is a sane start.
    """
    b1, b2 = camera.on_floor(flight.bounce1_px), camera.on_floor(flight.bounce2_px)
    nearest = flight.pixels[np.argmin(np.abs(flight.times - flight.hit_t))]
    origin, direction = camera.ray(nearest)
    across = direction[0, :2]
    hit = origin + max(0.0, (b1[:2] - origin[:2]) @ across / (across @ across)) * direction[0]
    t1, t2 = flight.hit_t - flight.bounce1_t, flight.bounce2_t - flight.hit_t
    v1 = (hit - b1) / t1 + np.array([0, 0, g * t1 / 2])
    v2 = (b2 - hit) / t2 + np.array([0, 0, g * t2 / 2])
    return np.r_[b1[:2], v1, v2]


def fit_flight(
    camera: Camera,
    flight: Groundstroke,
    fit_gravity: bool = False,
    gravity: float = GRAVITY,
    drag: float = 0.05,
    fit_drag: bool = True,
    hit_window_s: float = 0.0,
) -> dict:
    """The 3D groundstroke that best explains the pixels.

    Unknowns: the first bounce's floor position (2), the velocity leaving it
    (3), the velocity leaving the paddle (3), k unless ``fit_drag`` is off, and
    g when ``fit_gravity`` is on. The paddle is wherever the first leg arrives,
    so the path is continuous.

    ``hit_window_s`` lets the paddle time move that far either side of the
    detected hit, one step at a time, keeping the time that explains the pixels
    best. The bounces are anchored by the floor; the hit is not, and the hit
    detector tends to fire a few frames early.
    """
    if hit_window_s <= 0:
        return _fit_once(camera, flight, fit_gravity, gravity, drag, fit_drag)
    # Each candidate starts afresh: these fits have more than one minimum, and
    # warm-starting from the neighbour's answer explores less.
    best = None
    for shift in np.arange(-hit_window_s, hit_window_s + 1e-9, STEP_S):
        hit_t = flight.hit_t + shift
        if (flight.times < hit_t).sum() < 2 or (flight.times > hit_t).sum() < 2:
            continue
        fit = _fit_once(camera, replace(flight, hit_t=hit_t), fit_gravity, gravity, drag, fit_drag)
        if best is None or fit["cost"] < best["cost"]:
            best = fit
    return best


def _fit_once(camera, flight, fit_gravity, gravity, drag, fit_drag) -> dict:
    x0 = np.r_[_initial(camera, flight, gravity), drag, [gravity] if fit_gravity else []]
    leg1 = flight.times < flight.hit_t
    times1, times2 = flight.times[leg1], flight.times[~leg1]

    def unpack(p):
        g = p[9] if fit_gravity else gravity
        return np.r_[p[0], p[1], 0.0], p[2:5], p[5:8], p[8], g

    def path(p):
        start, v1, v2, k, g = unpack(p)
        first, end1 = propagate(start, v1, flight.bounce1_t, np.r_[times1, flight.hit_t], g, k)
        second, _ = propagate(end1[:3], v2, flight.hit_t, np.r_[times2, flight.bounce2_t], g, k)
        return start, first[:-1], first[-1], second[:-1], second[-1]

    def residuals(p):
        # A trial step can launch the ball absurdly fast and overflow; a large
        # finite residual tells the solver to step back.
        with np.errstate(all="ignore"):
            start, first, paddle, second, landing = path(p)
            seen = camera.to_image(np.vstack([start, first, second, landing]))
        observed = np.vstack([flight.bounce1_px, flight.pixels, flight.bounce2_px])
        # The floor is solid: nothing between the contacts may be under it, the
        # paddle included; and the second bounce must be on it.
        below = np.minimum(0.0, np.r_[first[:, 2], paddle[2], second[:, 2]])
        out = np.r_[(seen - observed).ravel(), FLOOR_PX_PER_M * landing[2], FLOOR_PX_PER_M * below]
        return np.nan_to_num(out, nan=1e6, posinf=1e6, neginf=-1e6)

    k_low, k_high = (0.0, 0.5) if fit_drag else (drag - 1e-6, drag + 1e-6)
    lower = np.r_[[-np.inf] * 8, k_low, [0.0] if fit_gravity else []]
    upper = np.r_[[np.inf] * 8, k_high, [40.0] if fit_gravity else []]
    result = least_squares(
        residuals,
        np.clip(x0, lower + 1e-9, upper - 1e-9),
        bounds=(lower, upper),
        loss="soft_l1",  # a few bad detections should not drag the arc
        f_scale=2.0,
        x_scale="jac",
    )
    start, first, paddle, second, _ = path(result.x)
    _, v1, v2, k, g = unpack(result.x)
    heights = np.r_[first[:, 2], second[:, 2]]
    error = np.linalg.norm(camera.to_image(np.vstack([first, second])) - flight.pixels, axis=1)
    return {
        "g_m_s2": float(g),
        "k_per_m": float(k),
        "apex_m": float(heights.max()),
        "paddle_height_m": float(paddle[2]),
        "paddle_speed_m_s": float(np.linalg.norm(v2)),
        "median_px": float(np.median(error)),
        "observations": int(len(flight.times)),
        "converged": bool(result.success),
        "cost": float(result.cost),
        "hit_t": float(flight.hit_t),
        # Enough to redraw the flight: propagate from the bounce to the paddle
        # with the first velocity, then on with the second.
        "bounce1_t": float(flight.bounce1_t),
        "bounce2_t": float(flight.bounce2_t),
        "bounce1_xyz": start.tolist(),
        "velocity_from_bounce": v1.tolist(),
        "paddle_xyz": paddle.tolist(),
        "velocity_from_paddle": v2.tolist(),
    }


def trajectory(fit: dict, times: np.ndarray) -> np.ndarray:
    """The fitted flight's 3D positions at ``times`` (inside the two bounces)."""
    g, k = fit["g_m_s2"], fit["k_per_m"]
    times = np.asarray(times, float)
    first = times <= fit["hit_t"]
    a, _ = propagate(
        fit["bounce1_xyz"], fit["velocity_from_bounce"], fit["bounce1_t"], times[first], g, k
    )
    b, _ = propagate(
        fit["paddle_xyz"], fit["velocity_from_paddle"], fit["hit_t"], times[~first], g, k
    )
    return np.vstack([a, b])


if __name__ == "__main__":
    camera, flights, data = load()
    print(f"{len(flights)} groundstrokes, camera at {np.round(camera.position, 2)} m")
    fit = fit_flight(camera, flights[0], fit_gravity=True, fit_drag=False, hit_window_s=0.15)
    print({key: round(v, 3) for key, v in fit.items() if isinstance(v, float)})
