# Video search on a laptop with EmbeddingGemma 2, and the arrow of time it cannot see

Companion code for the condados.ai post. [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) is
Google's open multimodal embedding model: it maps a video and a sentence into the same 768-d space, so finding a
clip is one dot product. This folder runs it fully offline on an Intel laptop with OpenVINO, measures what INT8 and
the integrated GPU cost and buy, and then tests whether the vectors know which way time runs.

## Results

43 Pexels clips (4–8 s each), 23 English queries, i7-12700H with Iris Xe graphics.

| | |
|---|---|
| Search, top-1 / top-3 | 22/23 / 23/23 in every configuration, PyTorch and all nine OpenVINO ones |
| One 8-frame clip → vector | **2.3 s INT8 on the iGPU** (2.25–2.39 s over 3 sessions) vs 10.8 s PyTorch on the CPU |
| One text query → vector | 18–59 ms, depending on the configuration |
| INT8 vs PyTorch, cosine | median clip 0.994, worst clip 0.974; search unchanged |
| Forward vs reversed clip | cosine 0.997; direction right in 16/32 caption pairs, 52% for a trained classifier (chance 50%) |
| VideoPrism-LvT-B, same test | search 23/23; direction 19/32 and 56%: also at chance |

What the probes found (`order_probe.py`): each frame's tokens do change with where the frame sits in the sequence,
but no readout, from a caption to a held-out linear classifier, recovers which way the clip runs. Averaging every
frame encoded on its own ("bag of frames") searches as well as the whole clip. VideoPrism, built with attention
across frames, notices when frames are shuffled (its vector moves 9× more than when they are reversed) but not the
direction: reversed motion is still smooth motion.

Full tables: `output/benchmark.md` (and `latency.png`), `output/results_*.md`, `output/order_probe.md`,
`output/order_frames.md`, `output/scale.md`.

## Run it

From the repository root, `uv sync` once. Then, inside this folder, in order:

```bash
PEXELS_API_KEY=... uv run python download.py   # the clips in clips.json, from Pexels (free API key)
uv run python prepare.py                       # 640x360, no audio, plus a reversed copy of each clip
uv run python embed.py                         # PyTorch fp32 reference vectors -> index/
uv run python evaluate.py index/pytorch_fp32_fps1.npz   # search + direction -> output/
uv run python export_openvino.py               # OpenVINO IR -> models/fp32/
uv run python quantize.py                      # NNCF INT8 -> models/int8w/, models/int8/
uv run python search.py "a man puts on a hat"  # INT8 on the iGPU; --device CPU, --dim 256
uv run python benchmark.py --session 1         # repeat as 2 and 3 on other days, then:
uv run python report.py                        # -> output/benchmark.md, latency.png
uv run python order_probe.py && uv run python order_probe.py --frames-report
uv run python videoprism_embed.py && uv run python evaluate.py index/videoprism_lvt_b_16f.npz
uv run python scale.py                         # exact search from 43 to 1M vectors -> output/scale.md
uv run --with jupyter jupyter lab notebook.ipynb
```

`smoke_export.py` and `smoke_gpu.py` are the first feasibility checks (does the model export at all, does it survive
the iGPU's f16), kept because the post refers to them. `find_candidates.py` is how the clips were chosen: it lays out
Pexels preview frames per search term; the picks and their queries were then written by hand into `clips.json`.

Measured on the laptop: `search.py` indexes the 43 clips in under 2 min the first time, `benchmark.py` takes about
1 h per session, `videoprism_embed.py` about 15 min. `embed.py` and `order_probe.py` are the long ones (tens of
minutes on the CPU and iGPU); `order_probe.py` saves after every clip and resumes, `embed.py` only adds the keys an
existing index lacks. Nothing needs a discrete GPU.

## Files

| File | Does |
|---|---|
| `clips.json` | The corpus: Pexels id, trim, action, forward and reversed captions, and the queries with their target clips |
| `ov_model.py` | `OVEmbedder`: video and text embeddings on OpenVINO with static shapes (and why the length buckets) |
| `search.py` | Text → top-k clips, the whole product in one file |
| `evaluate.py` | Search scores and the direction tests for any index |
| `order_probe.py` | Where order gets lost: per-token comparisons, bag of frames, timestamps, 1/2/4 fps |
| `videoprism_embed.py` | The control model on the same clips, captions and index keys |
| `scale.py` | Do you need a vector database yet? |
| `notebook.ipynb` | The walk-through, end to end |

`data/`, `models/`, `index/` and `cache/` are created by the scripts and stay out of git.

## Two things that will bite you

- **f32 on the iGPU can return NaN if you pad too far.** Most encoder layers attend within a 512-token window; pad a
  clip by more than that and some rows see only padding, the softmax divides by zero, and the NaN spreads. f16 hides
  it. `ov_model.py` pads to buckets 512 apart.
- **VideoPrism in transformers needs its text cleaned the official way** (lower case, punctuation to spaces, a final
  period), or the token ids differ from Google's and captions drift to cosine ~0.99. `videoprism_embed.py` does it,
  and its docstring records the check against the JAX code.

## Environment

Python 3.12, OpenVINO 2026.4.1, NNCF 3.4.0, PyTorch 2.14.1 (CPU), transformers 5.20.1, NumPy 2.4.6, PyAV 19.0.1,
scikit-learn 1.9.1; Ubuntu with kernel 6.8, Intel i7-12700H, Iris Xe. Laptop
timings drift by up to ~20% between sessions, so latency is reported as the median and min–max of three.

## Licences and credits

- Code: Apache-2.0 (repository `LICENSE`).
- EmbeddingGemma 2: Apache-2.0. VideoPrism: Apache-2.0; the transformers weights come from the draft pull request #2
  on `google/videoprism-lvt-base-f16r288`, pinned by commit.
- Clips: [Pexels License](https://www.pexels.com/license/), which allows use but not redistribution, so this
  repository ships the ids and a downloader, never the videos. Each clip's page is `https://www.pexels.com/video/<id>/`.
- The direction test was prompted by an OpenCV University post that compared forward and reversed clips.
