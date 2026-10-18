# int8 on longer sequences

First 3 clips at 2 and 4 fps, default token budget (120 per frame). Cosine of each clip vector to fp32 on the CPU.

| Configuration | fps | Tokens per clip | Cosine to fp32, per clip |
|---|---|---|---|
| int8_igpu_f16 | 2 | 1710, 1344, 1466 | 0.6158, 0.9797, 0.9855 |
| int8_igpu_f32 | 2 | 1710, 1344, 1466 | 0.6579, 0.9809, 0.9861 |
| int8_cpu | 2 | 1710, 1344, 1466 | 0.9970, 0.9957, 0.9959 |
| int8_vision_fp32_encoder_igpu_f16 | 2 | 1710, 1344, 1466 | 0.9985, 0.9975, 0.9969 |
| int8_igpu_f16 | 4 | 3418, 2686, 2930 | 0.5530, 0.6069, 0.5759 |
| int8_igpu_f32 | 4 | 3418, 2686, 2930 | 0.5734, 0.6294, 0.5805 |
| int8_cpu | 4 | 3418, 2686, 2930 | 0.9971, 0.9960, 0.9971 |
| int8_vision_fp32_encoder_igpu_f16 | 4 | 3418, 2686, 2930 | 0.9984, 0.9967, 0.9979 |

Laptop clip, 8 frames, iGPU at f16, one session:

| Token budget | int8 alone (ms) | int8 vision + fp32 encoder (ms) |
|---|---|---|
| 140 | 2,417 | 2,462 |
| 70 | 1,052 | 1,117 |
