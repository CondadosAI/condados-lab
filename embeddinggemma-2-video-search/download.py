"""Download the clips listed in clips.json from Pexels into data/raw/.

The clips are under the Pexels License: free to use and modify, but not to redistribute, so this
repository ships their ids and the script that fetches them, never the files.

    PEXELS_API_KEY=... uv run python download.py
"""

import http.client
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

HERE = Path(__file__).parent
RAW = HERE / "data" / "raw"
UA = {"User-Agent": "condados-lab/embeddinggemma-2-video-search"}


def pick_file(files: list[dict]) -> dict:
    """The smallest rendition that is still at least 960 px wide (we scale to 640x360 later)."""
    usable = [
        f for f in files if f.get("width") and f["width"] >= 960 and f["file_type"] == "video/mp4"
    ]
    return min(usable or files, key=lambda f: f["width"] * f["height"])


def main() -> None:
    key = os.environ["PEXELS_API_KEY"]
    clips = json.loads((HERE / "clips.json").read_text())["clips"]
    RAW.mkdir(parents=True, exist_ok=True)
    for c in clips:
        out = RAW / f"{c['id']}.mp4"
        if out.exists():
            continue
        req = urllib.request.Request(
            f"https://api.pexels.com/videos/videos/{c['id']}", headers={"Authorization": key, **UA}
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            meta = json.load(r)
        f = pick_file(meta["video_files"])
        for attempt in range(4):  # the CDN sometimes drops a connection mid-file
            try:
                req = urllib.request.Request(f["link"], headers=UA)
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                break
            except (http.client.IncompleteRead, urllib.error.URLError) as e:
                logger.warning(f"{c['id']}: {type(e).__name__}, retry {attempt + 1}")
        else:
            raise RuntimeError(f"{c['id']}: download failed 4 times")
        out.write_bytes(data)
        logger.info(f"{c['id']}: {f['width']}x{f['height']} {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
