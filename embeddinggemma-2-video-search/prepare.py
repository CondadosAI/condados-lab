"""Cut, scale and strip every clip the same way, and write its reversed twin.

Two things here protect the direction test from measuring something other than frame order:

- No audio (`-an`). The model has an audio encoder, and reversed audio is trivially separable.
- The forward and reversed copies come out of the same filter chain and the same encoder settings,
  each encoded once from the source. A difference between them cannot come from compression.

Every clip also ends up at 640x360, because the exported vision graph has a fixed token count
per frame (see smoke_export.py).

    uv run python prepare.py
"""

import json
import subprocess
from pathlib import Path

from loguru import logger

HERE = Path(__file__).parent
RAW, CLIPS = HERE / "data" / "raw", HERE / "data" / "clips"
SIZE = "scale=640:360:force_original_aspect_ratio=increase,crop=640:360,setsar=1"


def encode(src: Path, dst: Path, start: float, end: float, reverse: bool) -> None:
    vf = SIZE + (",reverse" if reverse else "")
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-to", str(end), "-i", str(src),
         "-vf", vf, "-an", "-r", "25", "-c:v", "libx264", "-crf", "18", "-preset", "medium",
         "-pix_fmt", "yuv420p", str(dst)],
        check=True,
    )  # fmt: skip


def main() -> None:
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    CLIPS.mkdir(parents=True, exist_ok=True)
    for c in clips:
        src = RAW / f"{c['id']}.mp4"
        start, end = c["trim"]
        for tag, rev in (("fwd", False), ("rev", True)):
            dst = CLIPS / f"{c['key']}_{tag}.mp4"
            if not dst.exists():
                encode(src, dst, start, end, rev)
        logger.info(f"{c['key']} {c['action']}: {end - start:.1f} s")


if __name__ == "__main__":
    main()
