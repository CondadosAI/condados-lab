"""Side-by-side reel: one text query, top-k segments from Marengo (left) and Gemma (right).

Both rank the same segments (Marengo's, cut at scene changes), so the only difference is the
model. Each clip is marked against the play-by-play: a tick when a moment of the class falls
inside it (widened by --delta), a cross when not.

    uv run python demo_reel.py --game misericordia --cls three --k 5 --device cuda
"""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from evaluate import GROUPS, embed_texts, peaks

HERE = Path(__file__).parent
DATA, INDEX, OUT = HERE / "data", HERE / "index", HERE / "output"
W, H = 640, 360


def frames(path: Path, t0: float, t1: float, fps: float = 15):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t0 - 2, 0) / s.time_base), stream=s)
        nxt = t0
        for f in c.decode(s):
            t = float(f.pts * s.time_base)
            if t > t1:
                return
            if t >= nxt:
                yield f.to_ndarray(format="bgr24", width=W, height=H)
                nxt += 1 / fps


def label(im: np.ndarray, text: str, ok: bool) -> np.ndarray:
    im = im.copy()
    cv2.rectangle(im, (0, 0), (W, 34), (0, 0, 0), -1)
    cv2.putText(im, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    col = (60, 200, 60) if ok else (60, 60, 230)
    cv2.rectangle(im, (0, 0), (W - 1, H - 1), col, 6)
    cv2.putText(im, "HIT" if ok else "MISS", (W - 90, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="misericordia")
    ap.add_argument("--variant", default="mask")
    ap.add_argument("--cls", default="three")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--delta", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    games = json.loads((HERE / "games.json").read_text())["games"]
    game = games[args.game]
    q = json.loads((HERE / game.get("queries", "queries.json")).read_text())
    query = q["classes"][args.cls][0]
    tag = f"{args.game}-{args.variant}" if args.variant else args.game
    mar = np.load(INDEX / f"{tag}_marengo_visual.npz")
    gem = np.load(INDEX / f"{args.game}_seg-{tag}_marengo_visual_N8_T140_mask.npz")
    mtv = dict(zip(mar["texts"].tolist(), mar["text_emb"], strict=True))[query]
    gtv = embed_texts([query], args.device)[0]

    moments = json.loads((DATA / f"{args.game}_moments.json").read_text())
    spans = [(m["lo"], m["hi"]) for m in moments if m["cls"] in GROUPS.get(args.cls, {args.cls})]

    def top(d, tv):
        s, e = d["start"], d["end"]
        sc = d["emb"] @ tv
        picks = peaks(sc, (s + e) / 2, args.k, 2 * args.delta)
        c = (s + e) / 2
        out = []
        for p in picks:
            i = int(np.argmin(np.abs(c - p)))
            hit = any(lo - args.delta <= e[i] and hi + args.delta >= s[i] for lo, hi in spans)
            out.append((float(s[i]), float(e[i]), hit))
        return out

    left, right = top(mar, mtv), top(gem, gtv)
    video = DATA / game.get("demo_video", game["video"])
    OUT.mkdir(exist_ok=True)
    out = OUT / f"reel_{args.game}_{args.cls}.mp4"
    with av.open(str(out), "w") as o:
        st = o.add_stream("libx264", rate=15)
        st.width, st.height, st.pix_fmt = 2 * W, H + 60, "yuv420p"
        st.options = {"crf": "22"}

        def write(img):
            for p in st.encode(av.VideoFrame.from_ndarray(img, format="bgr24")):
                o.mux(p)

        head = np.zeros((60, 2 * W, 3), np.uint8)
        cv2.putText(head, f'query: "{query}"', (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)  # fmt: skip
        cv2.putText(head, "TwelveLabs Marengo 3.5", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 120), 1)  # fmt: skip
        cv2.putText(head, "EmbeddingGemma 2 (open, local GPU)", (W + 10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 200, 200), 1)  # fmt: skip
        for r, ((a0, a1, ah), (b0, b1, bh)) in enumerate(zip(left, right, strict=True), start=1):
            fa = [label(f, f"#{r}  {a0:.0f}-{a1:.0f} s", ah) for f in frames(video, a0, a1)]
            fb = [label(f, f"#{r}  {b0:.0f}-{b1:.0f} s", bh) for f in frames(video, b0, b1)]
            n = max(len(fa), len(fb))
            fa += [fa[-1]] * (n - len(fa))
            fb += [fb[-1]] * (n - len(fb))
            for x, y in zip(fa, fb, strict=True):
                write(np.vstack([head, np.hstack([x, y])]))
        for p in st.encode():
            o.mux(p)
    print(f"Marengo hits {sum(h for *_, h in left)}/{args.k}, "
          f"Gemma hits {sum(h for *_, h in right)}/{args.k} -> {out.name}")  # fmt: skip


if __name__ == "__main__":
    main()
