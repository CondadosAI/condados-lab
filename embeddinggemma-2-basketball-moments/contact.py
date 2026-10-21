"""A contact sheet of frames at given video times, each labelled: for checking labels by eye.

    uv run python contact.py out.jpg 1431.5:dunk 1454.5:three 2299:ft
"""

import sys
from pathlib import Path

import av
import cv2
import numpy as np

HERE = Path(__file__).parent


def grab(path: Path, t: float) -> np.ndarray:
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 2, 0) / s.time_base), stream=s)
        for f in c.decode(s):
            if float(f.pts * s.time_base) >= t:
                return f.to_ndarray(format="bgr24", width=480, height=270)
    return np.zeros((270, 480, 3), np.uint8)


def main() -> None:
    out, specs = sys.argv[1], sys.argv[2:]
    video = HERE / "data" / "misericordia_1080.mp4"
    tiles = []
    for spec in specs:
        t, _, label = spec.partition(":")
        im = grab(video, float(t)).copy()
        cv2.rectangle(im, (0, 0), (480, 22), (0, 0, 0), -1)
        cv2.putText(im, f"{float(t):.1f} {label}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)  # fmt: skip
        tiles.append(im)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(out, np.vstack(rows))


if __name__ == "__main__":
    main()
