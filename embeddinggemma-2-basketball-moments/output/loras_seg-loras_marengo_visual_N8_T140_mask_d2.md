# loras_seg-loras_marengo_visual_N8_T140_mask

465 windows of 8.67731 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 19% | 0.39 (0.30-0.44) | 0.32 (0.26-0.35) | 0.19 | 11.0 (8-14) | 7.7 (7-8) | 4.0 (7) |
| timeout | 4 | 9% | 0.40 (0.07-0.61) | 0.54 (0.38-0.64) | 0.09 | 1.0 (0-2) | 2.7 (2-3) | 0.3 (1) |
| three | 23 | 7% | 0.09 (0.08-0.10) | 0.07 (0.06-0.08) | 0.07 | 0.7 (0-1) | 0.0 (0-0) | 0.5 (2) |
| made_two | 23 | 8% | 0.15 (0.13-0.17) | 0.10 (0.07-0.12) | 0.08 | 0.7 (0-2) | 0.0 (0-0) | 0.5 (2) |
| substitution | 22 | 31% | 0.36 (0.31-0.40) | 0.37 (0.33-0.40) | 0.31 | 5.0 (4-7) | 4.0 (1-6) | 4.9 (8) |
| foul | 25 | 8% | 0.11 (0.11-0.12) | 0.10 (0.09-0.10) | 0.08 | 1.3 (1-2) | 1.3 (1-2) | 0.7 (2) |
| turnover | 21 | 7% | 0.13 (0.10-0.18) | 0.13 (0.11-0.14) | 0.07 | 1.0 (0-2) | 1.0 (0-2) | 0.6 (2) |
| shot_attempt | 102 | 30% | 0.38 (0.34-0.42) | 0.28 (0.24-0.32) | 0.30 | 14.3 (11-19) | 8.0 (6-11) | 10.3 (16) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
