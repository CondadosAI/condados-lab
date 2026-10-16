"""Search Pexels for candidate clips and lay out their preview frames, before downloading anything.

Pexels returns ~15 preview pictures per video, so a contact sheet per search term shows what each
candidate actually contains. The clips that make it into the corpus are then written by hand into
`clips.json`, together with the queries that should find them, before any embedding is computed.

    PEXELS_API_KEY=... uv run python find_candidates.py
"""

import json
import os
import urllib.parse
import urllib.request
from io import BytesIO
from pathlib import Path

from loguru import logger
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
OUT = HERE / "cache" / "candidates"
API = "https://api.pexels.com/videos/search"
UA = {"User-Agent": "condados-lab/embeddinggemma-2-video-search"}

# Reversible pairs: the action has a natural description in both directions.
PAIRED = [
    "person sitting down on chair",
    "person standing up from chair",
    "opening door",
    "closing door",
    "putting on jacket",
    "taking off jacket",
    "putting on glasses",
    "opening laptop",
    "closing laptop",
    "opening book",
    "closing book",
    "opening box",
    "walking towards camera",
    "walking away from camera",
    "walking up stairs",
    "walking down stairs",
    "picking up cup from table",
    "putting down cup on table",
    "camera zoom in",
    "plane taking off",
    "plane landing",
    "folding towel",
    "unfolding map",
    "stacking blocks",
    "train arriving station",
    "boat leaving dock",
    "escalator going up",
    "opening curtains",
    "lying down on bed",
    "getting out of bed",
    "sunrise timelapse",
    "sunset timelapse",
    "inflating balloon",
    "putting on hat",
    "tying shoelaces",
    "packing suitcase",
    "car parking",
    "car driving away",
]
# Second round (2026-10-16): concepts the first round found nothing usable for, and new ones so
# the pairs are not a handful of concepts filmed several times.
PAIRED += [
    "closing a book on table",
    "opening a book",
    "wrapping a gift",
    "unwrapping gift",
    "opening drawer",
    "opening refrigerator door",
    "putting on shoes",
    "taking off shoes",
    "getting into car",
    "getting out of car",
    "elevator doors opening",
    "zipping up jacket",
    "putting on gloves",
    "folding clothes",
    "person entering room",
    "raising flag",
    "building lego",
    "dolly zoom in city",
    "putting on headphones",
    "pouring water into glass",
]
# One-directional: the reversed clip shows something that does not happen.
ONE_WAY = [
    "diving into pool",
    "glass breaking",
    "cracking egg",
    "pouring coffee into cup",
    "blowing out candles",
    "long jump",
    "waves crashing",
    "smoke rising",
    "dominoes falling",
    "ice melting timelapse",
    "paper burning",
    "leaves falling",
    "candle blown out",
    "ice cube melting time lapse",
    "water balloon popping",
    "egg falling",
    "breaking pencil",
    "tree falling",
    "building demolition",
    "paint splash in water",
    "balloon popping slow motion",
    "match lighting",
]


def search(term: str, key: str, n: int) -> list[dict]:
    q = urllib.parse.urlencode({"query": term, "per_page": n, "orientation": "landscape"})
    req = urllib.request.Request(f"{API}?{q}", headers={"Authorization": key, **UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["videos"]


def sheet(term: str, videos: list[dict], path: Path, per_row: int = 6) -> None:
    """One row per candidate: its id and duration, then six preview frames across the clip."""
    tw, th = 192, 108
    img = Image.new("RGB", (tw * per_row, (th + 18) * len(videos)), "white")
    d = ImageDraw.Draw(img)
    for r, v in enumerate(videos):
        pics = v["video_pictures"]
        picks = [pics[round(i * (len(pics) - 1) / (per_row - 1))] for i in range(per_row)]
        y = r * (th + 18)
        d.text((4, y + 2), f"{v['id']}  {v['duration']}s  {v['width']}x{v['height']}", fill="black")
        for c, p in enumerate(picks):
            req = urllib.request.Request(p["picture"], headers=UA)
            with urllib.request.urlopen(req, timeout=30) as resp:
                im = Image.open(BytesIO(resp.read())).convert("RGB")
            img.paste(im.resize((tw, th)), (c * tw, y + 18))
    img.save(path, quality=80)
    logger.info(f"{term}: {len(videos)} candidates -> {path.name}")


def main(n: int = 5) -> None:
    key = os.environ["PEXELS_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    index = OUT / "candidates.json"
    found = json.loads(index.read_text()) if index.exists() else {}
    for kind, terms in (("paired", PAIRED), ("one_way", ONE_WAY)):
        for term in terms:
            path = OUT / f"{kind}__{term.replace(' ', '_')}.jpg"
            if path.exists():  # rerunnable: only new terms hit the API
                continue
            vids = [v for v in search(term, key, n) if 3 <= v["duration"] <= 30]
            found[term] = {
                "kind": kind,
                "candidates": [
                    {
                        "id": v["id"],
                        "url": v["url"],
                        "duration": v["duration"],
                        "author": v["user"]["name"],
                    }
                    for v in vids
                ],
            }
            if vids:
                sheet(term, vids, path)
    index.write_text(json.dumps(found, indent=1))


if __name__ == "__main__":
    main()
