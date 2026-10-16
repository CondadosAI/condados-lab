# EmbeddingGemma 2 on a laptop: latency and quality

1 session(s). Latency in ms, median of 10 timed runs after 3 warm-ups, on one 8-frame 640x360 clip (1 fps); with several sessions, the median across sessions and the min–max.

| Model | Device | Vision, 8 frames/call | Vision, 1 frame/call | Encoder, 1 clip/call | Encoder, 4 clips/call (per clip) | Text query | Clip, frames → vector |
|---|---|---|---|---|---|---|---|
| pytorch fp32 | CPU | – | – | – | – | 59 | 10,819 |
| fp32 | CPU | 9,832 | 7,435 | 1,155 | 803 | 54 | 9,691 |
| fp32 | iGPU f16 | 2,735 | 3,137 | 285 | 276 | 29 | 3,161 |
| fp32 | iGPU f32 | 5,060 | 4,736 | 378 | 434 | 36 | 5,393 |
| int8w | CPU | 4,576 | 4,314 | 535 | 551 | 23 | 5,200 |
| int8w | iGPU f16 | 3,834 | 3,647 | 310 | 295 | 36 | 4,102 |
| int8w | iGPU f32 | 4,858 | 4,582 | 439 | 434 | 44 | 5,431 |
| int8 | CPU | 3,223 | 3,095 | 405 | 427 | 18 | 3,775 |
| int8 | iGPU f16 | 2,136 | 2,073 | 238 | 200 | 25 | 2,388 |
| int8 | iGPU f32 | 2,650 | 2,468 | 296 | 260 | 23 | 2,770 |

Quality, latest session: 43 clips, 23 queries. (Session 1 ran before the padding fix, see ov_model.py.)

| Model | Device | NaN vectors | Cosine vs PyTorch, median clip | worst clip | worst query | Top-1 /23 | Top-3 /23 | MRR |
|---|---|---|---|---|---|---|---|---|
| fp32 | CPU | 1 | 1.0000 | 1.0000 | 1.0000 | 22 | 23 | 0.978 |
| fp32 | iGPU f16 | 0 | 1.0000 | 0.9999 | 1.0000 | 22 | 23 | 0.978 |
| fp32 | iGPU f32 | 1 | 1.0000 | 1.0000 | 1.0000 | 22 | 23 | 0.978 |
| int8w | CPU | 1 | 0.9996 | 0.9991 | 0.9996 | 22 | 23 | 0.978 |
| int8w | iGPU f16 | 0 | 0.9998 | 0.9996 | 0.9998 | 22 | 23 | 0.978 |
| int8w | iGPU f32 | 1 | 0.9998 | 0.9997 | 0.9998 | 22 | 23 | 0.978 |
| int8 | CPU | 0 | 0.9943 | 0.9741 | 0.9795 | 22 | 23 | 0.978 |
| int8 | iGPU f16 | 0 | 0.9945 | 0.9745 | 0.9808 | 22 | 23 | 0.978 |
| int8 | iGPU f32 | 0 | 0.9944 | 0.9753 | 0.9808 | 22 | 23 | 0.978 |
