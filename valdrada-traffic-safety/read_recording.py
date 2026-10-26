"""Read any frame source and write a preview video. It never learns where the frames came from.

python read_recording.py <recording folder | video file | rtsp://... | webcam> [--out x.mp4]
"""

import argparse

import cv2
from sources import open_source

ap = argparse.ArgumentParser()
ap.add_argument("source")
ap.add_argument("--out", default="preview.mp4")
a = ap.parse_args()

src = open_source(a.source)
cam = src.camera
print(f"{cam.width}x{cam.height}, K known: {cam.K is not None}, pose known: {cam.pose is not None}")
writer, n, t = None, 0, 0.0
for f in src:
    if writer is None:
        writer = cv2.VideoWriter(
            a.out, cv2.VideoWriter_fourcc(*"mp4v"), 10, (cam.width, cam.height)
        )
    label = f"frame {f.index}  t = {f.t:.2f} s"
    cv2.putText(f.image, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    writer.write(f.image)
    n, t = n + 1, f.t
if writer:
    writer.release()
print(f"{n} frames, {t:.2f} s of video -> {a.out}")
