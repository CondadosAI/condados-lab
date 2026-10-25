"""MEVA traffic clips -> one video + timed vehicle events (data/meva_720.mp4, meva_moments.json).

MEVA (Kitware / IARPA, CC BY 4.0, https://mevadata.org) films a test site's roads and parking lots
from fixed cameras, in 5-minute clips, with every activity annotated in frames: a car stops,
starts, turns, reverses, makes a U-turn; a person opens a door, gets in or out. Most events last
2-5 s inside long quiet stretches, which is the case this post tests: a short action in a scene
where little else changes.

The clips (one per camera, the busiest of each, listed in games.json) are downloaded from the
public bucket, re-encoded to 720p and joined end to end, so the joined video is evaluated exactly
like a basketball game: one timeline, many classes. Each clip's events are shifted by its offset.

    uv run python fetch_meva.py               # all clips in games.json
    uv run python fetch_meva.py --first 2     # a pilot
"""

import argparse
import json
import subprocess
import urllib.request
from pathlib import Path

import av
import imageio_ffmpeg
import yaml

HERE = Path(__file__).parent
DATA = HERE / "data"
VIDEOS = "https://mevadata-public-01.s3.amazonaws.com/drops-123-r13"
REPO = "https://gitlab.kitware.com/meva/meva-data-repo/-/raw/master"
ANNOTATIONS = f"{REPO}/annotation/DIVA-phase-2/MEVA/kitware"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def folder(clip: str) -> str:
    """Both the bucket and the annotation repo file a clip under its date and its end hour."""
    date, _, end = clip.split(".")[:3]
    return f"{date}/{end[:2]}"


def fetch(url: str, dst: Path) -> Path:
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)
    return dst


def events(clip: str, fps: float, offset: float) -> list[dict]:
    """Vehicle activities of one clip, in seconds of the joined video."""
    path = fetch(f"{ANNOTATIONS}/{folder(clip)}/{clip}.activities.yml",
                 DATA / "meva" / f"{clip}.activities.yml")  # fmt: skip
    out = []
    for d in yaml.safe_load(path.read_text()) or []:
        act = d.get("act")
        if not act:
            continue
        cls = max(act["act2"], key=act["act2"].get)
        if "vehicle" not in cls:
            continue
        s, e = act["timespan"][0]["tsr0"]
        lo, hi = offset + s / fps, offset + e / fps
        out.append({"half": 1, "cls": cls, "clip": clip, "t": lo, "lo": lo, "hi": hi})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=None, help="only the first n clips")
    args = ap.parse_args()
    game = json.loads((HERE / "games.json").read_text())["games"]["meva"]
    clips = game["clips"][: args.first]
    (DATA / "meva").mkdir(parents=True, exist_ok=True)

    parts, moments, offset = [], [], 0.0
    for clip in clips:
        raw = fetch(f"{VIDEOS}/{folder(clip)}/{clip}.r13.avi", DATA / "meva" / f"{clip}.avi")
        with av.open(str(raw)) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate)
        mp4 = DATA / "meva" / f"{clip}.mp4"
        if not mp4.exists():
            # Same codec settings for every part, so the concat demuxer can join them as is.
            subprocess.run([FFMPEG, "-v", "error", "-y", "-i", str(raw), "-vf", "scale=-2:720",
                            "-r", "30", "-c:v", "libx264", "-crf", "23", "-preset", "medium",
                            "-an", str(mp4)], check=True)  # fmt: skip
        with av.open(str(mp4)) as c:
            dur = c.duration / 1e6
        moments += events(clip, fps, offset)
        parts.append(mp4)
        print(f"{clip}: {fps:g} fps, {dur:.1f} s, offset {offset:.1f} s")
        offset += dur

    lst = DATA / "meva" / "concat.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    out = DATA / game["video"]
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(out)], check=True)  # fmt: skip
    moments.sort(key=lambda m: m["t"])
    (DATA / "meva_moments.json").write_text(json.dumps(moments, indent=1))
    counts: dict[str, int] = {}
    for m in moments:
        counts[m["cls"]] = counts.get(m["cls"], 0) + 1
    print(f"{out.name}: {offset:.0f} s, {len(moments)} vehicle events")
    print(", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))


if __name__ == "__main__":
    main()
