"""The post's figures and lab data, from the clips and the PyTorch reference index.

- hero.webp: one clip as filmed next to the same clip played backwards, with the query score of
  each and the cosine between them (the model's 1 fps frames, forward and in reverse order).
  ogimage.webp is one still of it.
- search.webp: four queries, each with its top three clips playing and their scores.
- thumbs/<key>.webp: one small frame per clip, for the lab.
- lab.json: per query, the score of every clip at 768, 512, 256 and 128 dimensions; per
  reversible clip, the four caption-to-clip scores behind the paired test, for EmbeddingGemma 2
  and VideoPrism. The lab draws only these numbers, so it cannot disagree with the tables.

The clips stay out of git (Pexels License), and every clip shown is credited in the post.

    uv run python media.py <site>/public/blog/<slug> [--font Inter.woff --font-bold Inter-600.woff]
"""

import argparse
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).parent
CLIPS, INDEX = HERE / "data" / "clips", HERE / "index"
W, H, FPS = 1200, 675, 12
BG, FG, MUTED, ACCENT, WARN = "#0b0e18", "#e5e7eb", "#8b909c", "#818cf8", "#f59e0b"
HERO_KEY, HERO_QUERY = "7431681", "someone opens a laptop"
FONTS: dict[bool, str] = {}  # bold -> font file, from --font / --font-bold
SEARCHES = [
    "a man inflating a purple balloon",
    "striking a match",
    "espresso dripping into a white cup",
    "a swimmer dives off the starting block",
]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for name in (FONTS.get(bold), "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        if not name:
            continue
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def frames_at(path: Path, fps: float = FPS) -> list[Image.Image]:
    """Every frame nearest each 1/fps tick, as PIL images."""
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        frames = [f.to_image() for f in c.decode(s)]
        native = float(s.average_rate)
    step = native / fps
    return [frames[min(round(k * step), len(frames) - 1)] for k in range(int(len(frames) / step))]


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def save_anim(frames: list[Image.Image], path: Path, fps: int = FPS, quality: int = 50) -> None:
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=round(1000 / fps),
                   loop=0, quality=quality, method=6)  # fmt: skip
    print(f"{path.name}: {len(frames)} frames, {path.stat().st_size / 1e6:.2f} MB")


def hero(E: dict, out: Path) -> None:
    k = HERO_KEY
    fwd, rev = frames_at(CLIPS / f"{k}_fwd.mp4"), frames_at(CLIPS / f"{k}_rev.mp4")
    q = E["cap_fwd/" + k]  # the query is the clip's forward caption
    # The scores are the model's 8 frames at 1 fps, in order and reversed (`rfr`); the panels
    # play the same clip at 12 fps so the motion reads.
    s_fwd, s_rev = float(q @ E["fwd/" + k]), float(q @ E["rfr/" + k])
    cos = float(E["fwd/" + k] @ E["rfr/" + k])
    pw, ph, top = 564, 317, 150
    xs = (24, W - 24 - pw)
    out_frames = []
    for a, b in zip(fwd, rev, strict=False):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        d.text((24, 26), "Query", font=font(22), fill=MUTED)
        d.text((24, 56), f"“{HERO_QUERY}”", font=font(40, True), fill=FG)
        for x, frame, label, s in ((xs[0], a, "as filmed", s_fwd),
                                   (xs[1], b, "played backwards", s_rev)):  # fmt: skip
            im.paste(frame.resize((pw, ph), Image.LANCZOS), (x, top))
            d.text((x, top - 34), label, font=font(24, True), fill=FG)
            d.text((x, top + ph + 14), f"score {s:.3f}", font=font(30, True), fill=ACCENT)
        d.text((24, H - 92), f"Cosine between the two videos: {cos:.3f}", font=font(30, True),
               fill=WARN)  # fmt: skip
        d.text((24, H - 50), "EmbeddingGemma 2 finds the clip, then scores it the same both ways.",
               font=font(22), fill=MUTED)  # fmt: skip
        out_frames.append(im)
    save_anim(out_frames, out / "hero.webp")
    out_frames[len(out_frames) // 2].save(out / "ogimage.webp", quality=82)


def search_anim(E: dict, spec: dict, out: Path) -> None:
    clips = spec["clips"]
    V = np.stack([E["fwd/" + c["key"]] for c in clips])
    index = {q["query"]: i for i, q in enumerate(spec["queries"])}
    tw, th, top = 376, 212, 190
    out_frames = []
    for text in SEARCHES:
        s = V @ E[f"query/{index[text]}"]
        best = np.argsort(-s)[:3]
        reels = [frames_at(CLIPS / f"{clips[i]['key']}_fwd.mp4") for i in best]
        for t in range(int(3.5 * FPS)):
            im = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(im)
            d.text((24, 26), "Query", font=font(22), fill=MUTED)
            d.text((24, 56), f"“{text}”", font=font(36, True), fill=FG)
            d.text((24, 118), "Top 3 of 43 clips", font=font(22), fill=MUTED)
            for r, (i, reel) in enumerate(zip(best, reels, strict=True)):
                x = 24 + r * (tw + 12)
                im.paste(reel[t % len(reel)].resize((tw, th), Image.LANCZOS), (x, top))
                d.text((x, top + th + 12), f"{r + 1}.  score {s[i]:.3f}", font=font(24, True),
                       fill=ACCENT if r == 0 else FG)  # fmt: skip
                d.text((x, top + th + 46), clips[i]["action"], font=font(20), fill=MUTED)
            d.text((24, H - 50), "EmbeddingGemma 2, one dot product per clip", font=font(22),
                   fill=MUTED)  # fmt: skip
            out_frames.append(im)
    save_anim(out_frames, out / "search.webp")


def lab_data(spec: dict, out: Path) -> None:
    E = dict(np.load(INDEX / "pytorch_fp32_fps1.npz"))
    P = dict(np.load(INDEX / "videoprism_lvt_b_16f.npz"))
    clips = spec["clips"]
    V = np.stack([E["fwd/" + c["key"]] for c in clips])
    Q = np.stack([E[f"query/{i}"] for i in range(len(spec["queries"]))])
    data = {
        "clips": [{"key": c["key"], "action": c["action"], "kind": c["kind"]} for c in clips],
        "queries": [{"text": q["query"], "targets": q["targets"]} for q in spec["queries"]],
        "scores": {
            str(d): np.round(unit(Q[:, :d]) @ unit(V[:, :d]).T, 4).tolist()
            for d in (768, 512, 256, 128)
        },  # fmt: skip
        "pairs": [],
    }
    for c in clips:
        if c["kind"] != "paired":
            continue
        k, row = c["key"], {"key": c["key"], "fwd": c["caption_fwd"], "rev": c["caption_rev"]}
        for name, X in (("gemma", E), ("prism", P)):
            vf, vr, tf, tr = X["fwd/" + k], X["rfr/" + k], X["cap_fwd/" + k], X["cap_rev/" + k]
            row[name] = [round(float(x), 6) for x in (tf @ vf, tf @ vr, tr @ vf, tr @ vr, vf @ vr)]
        data["pairs"].append(row)
    (out / "lab.json").write_text(json.dumps(data, separators=(",", ":")))
    print(f"lab.json: {(out / 'lab.json').stat().st_size / 1e3:.0f} kB")
    (out / "thumbs").mkdir(exist_ok=True)
    for c in clips:
        f = frames_at(CLIPS / f"{c['key']}_fwd.mp4", 1)
        f[len(f) // 2].resize((192, 108), Image.LANCZOS).save(out / "thumbs" / f"{c['key']}.webp",
                                                              quality=72)  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--font", help="regular font file (TTF/OTF/WOFF); DejaVu Sans otherwise")
    ap.add_argument("--font-bold")
    args = ap.parse_args()
    FONTS.update({False: args.font, True: args.font_bold or args.font})
    args.out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((HERE / "clips.json").read_text())
    E = dict(np.load(INDEX / "pytorch_fp32_fps1.npz"))
    hero(E, args.out)
    search_anim(E, spec, args.out)
    lab_data(spec, args.out)


if __name__ == "__main__":
    main()
