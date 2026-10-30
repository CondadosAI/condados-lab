"""Draw the fitted 3D flights back onto the broadcast video.

For each frame: the arc so far, its shadow on the floor straight below the ball,
the ball's height above the floor, the speed off the paddle once it is hit, and
a side view of the court with the net, drawn from the same fit. Everything comes
from ``output/fits.json`` (``fit_all.py``) and the radial camera; the video is
the pickleball4you match (CC BY 3.0), downloaded separately.

    uv run python pickleball-3d-ball/render.py VIDEO OUT.mp4 51 52 53 54
    uv run python pickleball-3d-ball/render.py --tall VIDEO OUT.mp4 51 52 53 54

``--tall`` makes a 1080x1350 video for a phone feed: the broadcast on top and,
below it, the same flights in a 3D court seen by a virtual camera that swings
from behind the broadcast camera round to the side, so the arc reads as 3D.

The flights should be consecutive (each one's second bounce is the next one's
first). The clip starts a second before the first bounce and ends just after
the last.
"""

from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
from fit import HERE, load, trajectory

CYAN = (212, 182, 6)  # BGR of the site's accent #06b6d4
INDIGO = (248, 140, 129)  # #818cf8
INK = (248, 245, 244)
MUTED = (173, 161, 154)
SCRIM = (15, 10, 8)
MPH = 2.23694
LEAD_S, TAIL_S = 1.0, 0.6
FONT = cv2.FONT_HERSHEY_DUPLEX


def text(img, s, org, scale, colour=INK, thick=2):
    # A shadow by offsets: a thicker stroke widens the glyph advance and drifts.
    for dx, dy in ((2, 2), (-2, 2), (2, -2), (-2, -2), (0, 3)):
        cv2.putText(img, s, (org[0] + dx, org[1] + dy), FONT, scale, SCRIM, thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def panel(img, x0, y0, x1, y1, alpha=0.72):
    roi = img[y0:y1, x0:x1]
    roi[:] = (roi * (1 - alpha) + np.array(SCRIM) * alpha).astype(np.uint8)


def side_view(img, fits, t, court, box):
    """The court from the side, x along the court and height up, with the arc so far."""
    x0, y0, x1, y1 = box
    panel(img, x0, y0, x1, y1)
    pad, top_m = 30, 2.5
    sx = (x1 - x0 - 2 * pad) / court["length_m"]
    sz = (y1 - y0 - 2 * pad - 30) / top_m

    def to_px(x, z):
        return int(x0 + pad + x * sx), int(y1 - pad - z * sz)

    cv2.line(img, to_px(0, 0), to_px(court["length_m"], 0), MUTED, 2, cv2.LINE_AA)
    net = court["net_x_m"]
    cv2.line(img, to_px(net, 0), to_px(net, court["net_height_centre_m"]), INK, 3, cv2.LINE_AA)
    text(img, "side view, from the fit", (x0 + pad, y0 + 34), 0.7, MUTED, 1)
    for fit in fits:
        if not fit["bounce1_t"] <= t <= fit["bounce2_t"] + 1.5:
            continue
        times = np.linspace(fit["bounce1_t"], min(t, fit["bounce2_t"]), 120)
        path = trajectory(fit, times)
        pts = np.array([to_px(p[0], max(p[2], 0)) for p in path], np.int32)
        live = fit["bounce1_t"] <= t <= fit["bounce2_t"]
        cv2.polylines(img, [pts], False, CYAN if live else MUTED, 3 if live else 2, cv2.LINE_AA)
        if live:
            cv2.circle(img, tuple(pts[-1]), 7, INK, -1, cv2.LINE_AA)


COURT_BLUE = (150, 92, 38)
SURROUND = (70, 52, 30)
KITCHEN_M = 2.134  # the non-volley zone reaches 7 ft either side of the net


class Orbit:
    """A virtual pinhole camera looking at the court from an azimuth and elevation."""

    def __init__(self, target, azimuth_deg, elevation_deg, distance_m, focal_px, centre):
        az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
        self.target = np.asarray(target, float)
        self.eye = self.target + distance_m * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        f = self.target - self.eye
        f /= np.linalg.norm(f)
        r = np.cross(f, [0, 0, 1.0])
        r /= np.linalg.norm(r)
        self.axes = np.vstack([r, np.cross(r, f), f])
        self.focal, self.centre = focal_px, np.asarray(centre, float)

    def depth(self, xyz):
        return (np.asarray(xyz, float).reshape(-1, 3) - self.eye) @ self.axes[2]

    def __call__(self, xyz):
        p = (np.asarray(xyz, float).reshape(-1, 3) - self.eye) @ self.axes.T
        uv = self.focal * p[:, :2] / p[:, 2:3]
        return (self.centre + uv * [1, -1]).astype(np.int32)


def _ring(view, centre, radius=0.18):
    a = np.linspace(0, 2 * np.pi, 40)
    return view(np.c_[centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a), 0 * a])


def view3d(img, fits, t, court, view, scale=1.0):
    """The court in 3D from ``view``, with every flight of the rally so far."""
    L, W, net_x = court["length_m"], court["width_m"], court["net_x_m"]
    lw = max(1, int(round(2 * scale)))
    margin = 3.0
    surround = np.array(
        [
            [-margin, -margin, 0],
            [L + margin, -margin, 0],
            [L + margin, W + margin, 0],
            [-margin, W + margin, 0],
        ]
    )
    cv2.fillPoly(img, [view(surround)], SURROUND, cv2.LINE_AA)
    floor = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0]])
    cv2.fillPoly(img, [view(floor)], COURT_BLUE, cv2.LINE_AA)
    lines = [
        [(0, 0), (L, 0)],
        [(L, 0), (L, W)],
        [(L, W), (0, W)],
        [(0, W), (0, 0)],
        [(net_x - KITCHEN_M, 0), (net_x - KITCHEN_M, W)],
        [(net_x + KITCHEN_M, 0), (net_x + KITCHEN_M, W)],
        [(0, W / 2), (net_x - KITCHEN_M, W / 2)],
        [(net_x + KITCHEN_M, W / 2), (L, W / 2)],
    ]
    for a, b in lines:
        cv2.line(img, *[tuple(p) for p in view([[*a, 0], [*b, 0]])], INK, lw, cv2.LINE_AA)

    # The net: posts 0.3 m outside the sidelines, the tape sagging to the centre.
    post = 0.3048
    ys = np.linspace(-post, W + post, 30)
    sag = court["net_height_post_m"] - (
        court["net_height_post_m"] - court["net_height_centre_m"]
    ) * (1 - np.abs(ys - W / 2) / (W / 2 + post))
    tape = np.c_[np.full_like(ys, net_x), ys, sag]
    base = tape * [1, 1, 0]
    mesh = img.copy()
    cv2.fillPoly(mesh, [np.vstack([view(base), view(tape)[::-1]])], (40, 40, 40), cv2.LINE_AA)
    cv2.addWeighted(mesh, 0.55, img, 0.45, 0, img)
    cv2.polylines(img, [view(tape)], False, INK, lw + 1, cv2.LINE_AA)
    for y in (-post, W + post):
        cv2.line(
            img,
            *[tuple(p) for p in view([[net_x, y, 0], [net_x, y, court["net_height_post_m"]]])],
            INK,
            lw + 1,
            cv2.LINE_AA,
        )

    live_ball = None
    for fit in fits:
        # the live flight and the two before it: enough rally to read, not a tangle
        if t < fit["bounce1_t"] or t > fit["bounce2_t"] + 2.6:
            continue
        live = t <= fit["bounce2_t"]
        end = min(t, fit["bounce2_t"])
        times = np.linspace(fit["bounce1_t"], end, max(2, int((end - fit["bounce1_t"]) * 120)))
        path = trajectory(fit, times)
        colour = CYAN if live else MUTED
        cv2.polylines(img, [view(path * [1, 1, 0])], False, (95, 58, 22), lw + 1, cv2.LINE_AA)
        cv2.polylines(
            img, [view(path)], False, colour, (3 if live else 2) * lw // 2 + 1, cv2.LINE_AA
        )
        cv2.polylines(img, [_ring(view, fit["bounce1_xyz"])], True, INK, lw, cv2.LINE_AA)
        if not live:
            cv2.polylines(img, [_ring(view, path[-1])], True, INK, lw, cv2.LINE_AA)
        else:
            live_ball = path[-1]
    if live_ball is not None:
        ball, below = view(live_ball)[0], view(live_ball * [1, 1, 0])[0]
        cv2.line(img, tuple(ball), tuple(below), INK, lw, cv2.LINE_AA)
        cv2.circle(img, tuple(below), int(4 * scale), INK, -1, cv2.LINE_AA)
        r = max(4, int(view.focal * 0.0375 / view.depth(live_ball)[0] * 2.2))
        cv2.circle(img, tuple(ball), r, (90, 255, 220), -1, cv2.LINE_AA)
        cv2.circle(img, tuple(ball), r, SCRIM, 1, cv2.LINE_AA)
        text(
            img,
            f"{live_ball[2]:.2f} m",
            (int(ball[0]) + 16, int(ball[1]) - 10),
            0.8 * scale,
            INK,
            max(1, int(scale)),
        )
    return live_ball


def orbit_at(t, start, stop, w, h):
    """Swing from just behind the broadcast camera round to a diagonal, easing in and out."""
    u = np.clip((t - start) / (stop - start), 0, 1)
    u = u * u * (3 - 2 * u)
    az, el = 168 - 50 * u, 27 - 10 * u
    # Frame the court (and 2 m of air over it) in the area under the numbers: project
    # with a unit focal, then scale and centre so it fills that box.
    probe = Orbit((6.7, 3.05, 0.5), az, el, 16.0, 1.0, (0, 0))
    corners = np.array([[x, y, z] for x in (0, 13.41) for y in (0, 6.1) for z in (0, 2.0)])
    p = (corners - probe.eye) @ probe.axes.T
    uv = p[:, :2] / p[:, 2:3] * [1, -1]
    lo, hi = uv.min(0), uv.max(0)
    box_top, box_w, box_h = 130, w * 0.94, h - 130 - 70
    focal = min(box_w / (hi[0] - lo[0]), box_h / (hi[1] - lo[1]))
    centre = np.array([w / 2, box_top + box_h / 2]) - focal * (lo + hi) / 2
    return Orbit((6.7, 3.05, 0.5), az, el, 16.0, focal, centre)


def draw(img, camera, fits, t, court, hud=True):
    for fit in fits:
        if not fit["bounce1_t"] <= t <= fit["bounce2_t"] + TAIL_S:
            continue
        live = t <= fit["bounce2_t"]
        end = min(t, fit["bounce2_t"])
        times = np.linspace(fit["bounce1_t"], end, max(2, int((end - fit["bounce1_t"]) * 120)))
        path = trajectory(fit, times)
        arc = camera.to_image(path).astype(np.int32)
        shadow = camera.to_image(path * [1, 1, 0]).astype(np.int32)
        fade = 1.0 if live else max(0.0, 1 - (t - fit["bounce2_t"]) / TAIL_S)
        layer = img.copy()
        cv2.polylines(layer, [shadow], False, (40, 40, 40), 3, cv2.LINE_AA)
        cv2.polylines(layer, [arc], False, CYAN, 4, cv2.LINE_AA)
        bounces = [fit["bounce1_xyz"]] + ([] if live else [path[-1] * [1, 1, 0]])
        for b in bounces:
            c = tuple(camera.to_image(np.asarray(b))[0].astype(int))
            cv2.ellipse(layer, c, (16, 7), 0, 0, 360, INK, 2, cv2.LINE_AA)
        cv2.addWeighted(layer, fade, img, 1 - fade, 0, img)
        if not live:
            continue
        ball, below = arc[-1], shadow[-1]
        height = path[-1][2]
        cv2.line(img, tuple(ball), tuple(below), INK, 1, cv2.LINE_AA)
        cv2.circle(img, tuple(below), 5, INK, -1, cv2.LINE_AA)
        if not hud:
            continue
        text(img, f"{height:.2f} m", (int(ball[0]) + 18, int(ball[1]) - 12), 0.9)
        if t >= fit["hit_t"]:
            mph = fit["paddle_speed_m_s"] * MPH
            panel(img, 20, 20, 520, 150)
            text(img, "off the paddle", (44, 62), 0.75, MUTED, 1)
            text(img, f"{mph:.0f} mph", (44, 128), 2.0, CYAN, 3)


def compose_tall(frame, camera, fits, t, court, start, stop, src):
    """1080x1350: the broadcast with the arc on top, the 3D court below, the numbers between."""
    W, H, top_h = 1080, 1350, 608
    draw(frame, camera, fits, t, court, hud=False)
    out = np.empty((H, W, 3), np.uint8)
    out[:top_h] = cv2.resize(frame, (W, top_h), interpolation=cv2.INTER_AREA)
    below = np.full((H - top_h, W, 3), SCRIM, np.uint8)
    ball = view3d(below, fits, t, court, orbit_at(t, start, stop, W, H - top_h), scale=1.3)
    out[top_h:] = below
    live = [f for f in fits if f["bounce1_t"] <= t <= f["bounce2_t"]]
    panel(out, 0, top_h, W, top_h + 118, 0.55)
    if ball is not None:
        text(out, "height above the floor", (40, top_h + 40), 0.8, MUTED, 1)
        text(out, f"{ball[2]:.2f} m", (40, top_h + 100), 1.8, INK, 3)
    if live and t >= live[0]["hit_t"]:
        mph = live[0]["paddle_speed_m_s"] * MPH
        text(out, "speed off the paddle", (600, top_h + 40), 0.8, MUTED, 1)
        text(out, f"{mph:.0f} mph", (600, top_h + 100), 1.8, CYAN, 3)
    panel(out, 0, H - 48, W, H, 0.7)
    text(
        out, "One fixed camera, no depth sensor. 3D from a physics fit.", (24, H - 16), 0.7, INK, 1
    )
    text(out, "Video: pickleball4you, CC BY 3.0", (W - 380, H - 16), 0.6, MUTED, 1)
    cv2.line(out, (0, top_h), (W, top_h), MUTED, 1)
    return out


def main() -> None:
    args = sys.argv[1:]
    tall = "--tall" in args
    video, out, *ids = [a for a in args if a != "--tall"]
    camera, _, data = load()
    court = data["court"]
    all_fits = json.loads((HERE / "output" / "fits.json").read_text())
    fits = [all_fits[int(i)] for i in ids]
    start = fits[0]["bounce1_t"] - LEAD_S
    stop = fits[-1]["bounce2_t"] + TAIL_S
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first = int(round(start * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_w, src_h = w, h
    if tall:
        w, h = 1080, 1350
    enc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{w}x{h}",
            "-r",
            f"{fps}",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            out,
        ],
        stdin=subprocess.PIPE,
    )
    n = first
    while n / fps <= stop:
        ok, img = cap.read()
        if not ok:
            break
        t = n / fps
        if tall:
            img = compose_tall(img, camera, fits, t, court, start, stop, (src_w, src_h))
            enc.stdin.write(img.tobytes())
            n += 1
            continue
        draw(img, camera, fits, t, court)
        side_view(img, fits, t, court, (w - 640, 20, w - 20, 300))
        panel(img, 0, h - 52, w, h, 0.6)
        text(
            img,
            "One fixed camera, no depth sensor: height and speed from a physics fit to the 2D track",
            (24, h - 18),
            0.75,
            INK,
            1,
        )
        text(img, "Video: pickleball4you, CC BY 3.0", (w - 430, h - 18), 0.65, MUTED, 1)
        enc.stdin.write(img.tobytes())
        n += 1
    enc.stdin.close()
    enc.wait()
    print(f"{n - first} frames -> {out}")


if __name__ == "__main__":
    main()
