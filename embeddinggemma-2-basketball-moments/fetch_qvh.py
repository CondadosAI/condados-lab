"""A seeded sample of QVHighlights val: data/qvh/<vid>.mp4 and the manifest qvh_sample.json.

QVHighlights (Lei et al., NeurIPS 2021, https://github.com/jayleicn/moment_detr) pairs 150 s cuts
of YouTube vlogs and news with natural-language queries ("a woman cooks in her kitchen") and the
spans that answer them. The annotations are CC BY-NC-SA 4.0; the videos are the uploaders' own,
used by the authors under fair use for research, so they are fetched here from YouTube, cut to the
annotated window (`vid` = <youtube id>_<start>_<end>), and never redistributed: the manifest of
sampled ids is what the repo keeps.

Videos get taken down, so more are tried than kept: the first `--keep` that download, in a fixed
seeded order, are the sample.

    uv run python fetch_qvh.py --keep 100
"""

import argparse
import json
import random
import subprocess
import urllib.request
from pathlib import Path

import av
import imageio_ffmpeg
import yt_dlp

HERE = Path(__file__).parent
DATA = HERE / "data" / "qvh"
VAL = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"


def fetch_clip(vid: str) -> Path | None:
    out = DATA / f"{vid}.mp4"
    if out.exists():
        return out
    ytid, s, e = vid.rsplit("_", 2)
    full = DATA / f"{ytid}.full.mp4"
    opts = {
        # The whole video at 360p, then cut here: asking yt-dlp for just the window drops the
        # video track of DASH formats and silently leaves an audio-only file.
        "format": "18/bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]",
        "merge_output_format": "mp4",
        "outtmpl": str(full),
        "quiet": True,
        "no_warnings": True,
        # YouTube hides most formats from a client that cannot run its JavaScript challenge.
        "js_runtimes": {"deno": {}, "node": {}, "bun": {}},
    }
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            y.download([f"https://www.youtube.com/watch?v={ytid}"])
    except yt_dlp.utils.DownloadError as err:
        print(f"{vid}: {str(err).splitlines()[0][:120]}")
        return None
    if not full.exists():
        return None
    # 360p (the smallest TwelveLabs indexes; EmbeddingGemma downscales to 640x360 anyway) at
    # CRF 28 keeps a 150 s clip near 5 MB, so 100 fit the free plan.
    cut = ["-ss", s, "-to", e, "-i", str(full), "-vf", "scale=-2:360"]
    enc = ["-c:v", "libx264", "-crf", "28", "-preset", "medium", "-c:a", "aac", "-b:a", "64k"]
    try:
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y", *cut, *enc, str(out)],
                       check=True)  # fmt: skip
    except subprocess.CalledProcessError:  # e.g. YouTube served no video track at all
        print(f"{vid}: cut failed")
        out.unlink(missing_ok=True)
        return None
    finally:
        full.unlink()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    val = DATA / "highlight_val_release.jsonl"
    if not val.exists():
        urllib.request.urlretrieve(VAL, val)
    rows = [json.loads(line) for line in val.read_text().splitlines() if line.strip()]
    by_vid: dict[str, list[dict]] = {}
    for r in rows:
        by_vid.setdefault(r["vid"], []).append(r)
    order = sorted(by_vid)
    random.Random(args.seed).shuffle(order)

    kept, failed = [], []
    for vid in order:
        if len(kept) == args.keep:
            break
        path = fetch_clip(vid)
        if path is None:
            failed.append(vid)
            continue
        try:
            with av.open(str(path)) as c:
                got = c.duration / 1e6
                height = c.streams.video[0].height if c.streams.video else 0
        except av.error.FFmpegError:
            got, height = 0.0, 0
        want = by_vid[vid][0]["duration"]
        if abs(got - want) > 3 or height < 360:  # a missed cut would shift every span
            print(f"{vid}: {got:.1f} s ({want} s expected), {height}p; dropped")
            failed.append(vid)
            continue
        kept.append(vid)
        print(f"{len(kept):3d} {vid} {got:.1f} s")

    queries = [
        {k: r[k] for k in ("qid", "query", "vid", "duration", "relevant_windows")}
        for vid in kept
        for r in by_vid[vid]
    ]
    manifest = {"split": "val", "seed": args.seed, "videos": kept, "failed": failed,
                "queries": queries}  # fmt: skip
    (HERE / "qvh_sample.json").write_text(json.dumps(manifest, indent=1) + "\n")
    size = sum((DATA / f"{v}.mp4").stat().st_size for v in kept) / 1e9
    print(f"{len(kept)} videos ({size:.2f} GB), {len(queries)} queries, {len(failed)} failed")


if __name__ == "__main__":
    main()
