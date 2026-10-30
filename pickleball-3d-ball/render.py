"""Draw the fitted 3D flights back onto the broadcast video.

For each frame: the arc so far, its shadow on the floor straight below the ball,
the ball's height above the floor, the speed off the paddle once it is hit, and
a side view of the court with the net, drawn from the same fit. Everything comes
from ``output/fits.json`` (``fit_all.py``) and the radial camera; the video is
the pickleball4you match (CC BY 3.0), downloaded separately.

    uv run python pickleball-3d-ball/render.py VIDEO OUT.mp4 51 52 53 54

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


def draw(img, camera, fits, t, court):
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
        text(img, f"{height:.2f} m", (int(ball[0]) + 18, int(ball[1]) - 12), 0.9)
        if t >= fit["hit_t"]:
            mph = fit["paddle_speed_m_s"] * MPH
            panel(img, 20, 20, 520, 150)
            text(img, "off the paddle", (44, 62), 0.75, MUTED, 1)
            text(img, f"{mph:.0f} mph", (44, 128), 2.0, CYAN, 3)


def main() -> None:
    video, out, *ids = sys.argv[1:]
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
