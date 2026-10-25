# What a free video model finds in your footage, and what it misses

Companion code for the condados.ai post. The question is a founder's: you have hours of video and want to find
moments in it by typing what you are looking for. Can an open model on your own GPU
([EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2)) do that, or do you need a hosted video search
([TwelveLabs](https://www.twelvelabs.io/) Marengo 3.5)? Both rank **the same segments** of the same videos against the
same queries, so the only difference is the model, on three kinds of video.

## Results

| video | queries / events | EmbeddingGemma 2 | Marengo 3.5 | chance |
|---|---|---|---|---|
| Vlogs and news (QVHighlights val, 90 videos) | 92 queries, Hit@1 | 88.0 | 93.5 | 34.8 |
| same, R1@0.5 | | 48.9 | 53.3 | 4.4 |
| Basketball broadcast, threes (window AP, ±5 s) | Loras 23 / Misericordia 9 | 0.11 / 0.03 | 0.44 / 0.49 | 0.09 / 0.03 |
| same, free throws | 31 / 46 | 0.41 / 0.39 | 0.61 / 0.60 | 0.21 / 0.24 |
| Traffic camera (MEVA, 8 cameras), vehicle stops (window AP, ±2 s) | 76 | 0.50 | 0.50 | 0.37 |
| same, person opens a door | 64 | 0.28 | 0.40 | 0.30 |

- **Scenes, Gemma is close.** On QVHighlights it is about 5 points behind on every measure. Paired by query, Marengo
  finds 5 that Gemma misses and the reverse never happens (exact McNemar p = 0.06). All 5 are a person doing
  something specific ("man in pink top eats fish": Gemma's pick is a man in a pink top *holding* a fish).
- **A sports action in a wide shot, Gemma is at chance.** Marengo finds threes and free throws well above chance;
  Gemma is above chance only on stoppages, where the picture itself changes.
- **A short action on a fixed camera, both miss.** A car stopping lasts 2-5 s, fills a few percent of the frame and
  sits in a ~9 s segment where nothing else changes.

Full tables: `output/qvh_eval.md`; `output/<game>_seg-…_d{2,5}.md` (Gemma on Marengo's segments) and
`output/<game>-mask_marengo_{visual,audio}_d{2,5}.md`, `output/<game>-mask_twelvelabs_search_d{2,5}.md` for the two
games; `output/meva*_d{0,2}.md` for MEVA. `notebook.ipynb` walks all of it.

## Data and licences

| data | licence | what this repo keeps |
|---|---|---|
| Misericordia–Manhattanville, 12 Jan 2019 (YouTube `cjf7Qd3lIl0`, MCN 87) | CC BY | id, score-bug boxes (`games.json`) |
| Loras–Buena Vista, 10 Jan 2026 (YouTube `66NC8NhwUHk`, DUMEDIA) | CC BY | id, score-bug boxes |
| Play-by-play (Sidearm box scores, URLs in `games.json`) | the schools' | nothing: `fetch_log.py` fetches it |
| [MEVA](https://mevadata.org) (Kitware Inc. and IARPA) | CC BY 4.0 | clip names (`games.json`), queries |
| [QVHighlights](https://github.com/jayleicn/moment_detr) annotations | CC BY-NC-SA 4.0 | `qvh_sample.json` (ids, queries, spans of the sample) |
| QVHighlights videos | the YouTube uploaders' | nothing: fetched per id, never redistributed |
| EmbeddingGemma 2 weights | [Gemma terms](https://ai.google.dev/gemma/terms) | nothing: downloaded from Hugging Face |

Videos, frames, vectors and the demo reels stay out of git (`data/`, `index/`, `*.mp4`).

## Run it

From the repository root, `uv sync` once. The TwelveLabs steps need an API key in `TWELVE_API_KEY` (the free plan
holds 3 GB of video: one game and the QVH sample fit, not everything at once; delete an index before the next).
The Gemma steps need a CUDA GPU in practice (`--device cuda`; float32, 8 GB is enough); the rest runs on a CPU.

**Basketball.** The video, its audio, and a copy with the score bug blacked out (the bug writes "3 POINTER" after a
three, which a model that reads text could match):

```bash
uvx yt-dlp -f 137 -o data/misericordia_1080.mp4 https://www.youtube.com/watch?v=cjf7Qd3lIl0
uvx yt-dlp -f "ba[ext=m4a]" -o data/misericordia_audio.m4a https://www.youtube.com/watch?v=cjf7Qd3lIl0
# 720p, bug masked; the frames outside the two halves are blacked too, to fit the free plan
ffmpeg -i data/misericordia_1080.mp4 -i data/misericordia_audio.m4a -map 0:v -map 1:a \
  -vf "scale=1280:720,drawbox=x=200:y=610:w=968:h=60:color=black:t=fill,drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='lt(t,1342)+between(t,3873,4696)+gt(t,7240)'" \
  -c:v libx264 -crf 23 -preset medium -c:a copy -movflags +faststart data/misericordia_720_av_mask.mp4

uvx yt-dlp -f "bv*[height<=720][vcodec^=avc1]" -o data/loras_720.mp4 https://www.youtube.com/watch?v=66NC8NhwUHk
uvx yt-dlp -f "ba[ext=m4a]" -o data/loras_audio.m4a https://www.youtube.com/watch?v=66NC8NhwUHk
ffmpeg -i data/loras_720.mp4 -i data/loras_audio.m4a -map 0:v -map 1:a \
  -vf "drawbox=x=175:y=638:w=930:h=59:color=black:t=fill" -c:v libx264 -crf 23 -c:a copy data/loras_720_av_mask.mp4
```

Then, for each game (`--game misericordia` is the default, add `--game loras` for the other):

```bash
uv run python fetch_log.py                     # play-by-play -> data/<game>_events.json
uv run python clock_ocr.py                     # score-bug clock every 0.5 s -> data/<game>_clock.csv
uv run python align.py                         # events in video time -> data/<game>_moments.json
uv run python contact.py out.jpg 1454.5:three  # check any moment by eye
uv run python marengo.py upload --game misericordia --variant mask --video data/misericordia_720_av_mask.mp4
uv run python marengo.py embed --game misericordia --variant mask       # -> index/<game>-mask_marengo_*.npz
uv run python marengo.py search --game misericordia --variant mask --delta 5
uv run python embed_windows.py --device cuda --game misericordia --mask-bug --fps 4 --batch 2 \
  --segments index/misericordia-mask_marengo_visual.npz                 # Gemma on Marengo's segments
uv run python evaluate.py index/misericordia_seg-misericordia-mask_marengo_visual_N8_T140_mask.npz --delta 5
uv run python evaluate.py index/misericordia-mask_marengo_visual.npz --delta 5
uv run python demo_reel.py --game misericordia --cls three --k 5 --device cuda   # -> output/reel_*.mp4
```

On Loras, Gemma's table used the segments of the first Loras index (the same video without the mask, uploaded as
`--variant ""` and deleted since to free space), hence the file name `loras_seg-loras_marengo_visual_…`.
`embed_windows.py` without `--segments` slides its own windows instead (`--length`, `--stride`, `--fps`,
`--tokens`); on Loras, 4 s windows did no better than Marengo's segments.

**Traffic camera (MEVA).**

```bash
uv run python fetch_meva.py                    # 8 clips -> data/meva_720.mp4 + data/meva_moments.json (460 MB)
uv run python marengo.py upload --game meva --variant full --video data/meva_720.mp4
uv run python marengo.py embed --game meva --variant full
uv run python marengo.py search --game meva --variant full --delta 2
uv run python embed_windows.py --device cuda --game meva --segments index/meva-full_marengo_visual.npz --fps 4 --tokens 140 --batch 2
uv run python evaluate.py index/meva_seg-meva-full_marengo_visual_N8_T140.npz --game meva --delta 2
uv run python evaluate.py index/meva-full_marengo_visual.npz --game meva --delta 2
```

**Vlogs and news (QVHighlights).**

```bash
uv run python fetch_qvh.py --keep 90           # seeded sample of val -> data/qvh/, qvh_sample.json (0.73 GB)
uv run python qvh.py upload                    # one TwelveLabs index, egbm-qvh
uv run python qvh.py embed                     # Marengo segment and text vectors -> index/qvh_marengo_*.npz
uv run python qvh.py gemma --device cuda       # Gemma on the same segments -> index/qvh_gemma_visual.npz
uv run python qvh.py search                    # the hosted search, top result per query
uv run python qvh.py evaluate                  # -> output/qvh_eval.md
uv run python demo_reel.py --qvh 6065 222 5097 7578 6266 5651             # -> output/reel_qvh.mp4
uv run --with jupyter jupyter lab notebook.ipynb
```

YouTube takes videos down, so `fetch_qvh.py` tries more than it keeps, in a fixed order; a later run may end up
with a slightly different sample than `qvh_sample.json`.

Measured on one desktop GPU (8 GB), float32: Gemma embeds 1825 QVH segments (13,500 s of video) in 1,226 s and the
MEVA video (2,400 s) in 208 s, about 11× faster than real time. TwelveLabs uploaded and indexed 80 QVH videos in
764 s.
