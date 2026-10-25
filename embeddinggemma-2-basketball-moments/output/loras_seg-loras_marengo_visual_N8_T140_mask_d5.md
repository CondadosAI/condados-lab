# loras_seg-loras_marengo_visual_N8_T140_mask

465 windows of 8.67731 s (mean) inside the halves; delta = 5 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 21% | 0.41 (0.31-0.47) | 0.34 (0.28-0.38) | 0.21 | 12.3 (8-15) | 9.7 (8-12) | 5.2 (8) |
| timeout | 4 | 9% | 0.39 (0.07-0.58) | 0.52 (0.37-0.62) | 0.09 | 1.7 (0-3) | 3.3 (3-4) | 0.4 (1) |
| three | 23 | 9% | 0.11 (0.10-0.12) | 0.09 (0.08-0.09) | 0.09 | 2.0 (1-3) | 0.3 (0-1) | 1.4 (3) |
| made_two | 23 | 9% | 0.16 (0.14-0.18) | 0.12 (0.09-0.14) | 0.09 | 2.7 (1-5) | 0.7 (0-1) | 1.4 (3) |
| substitution | 22 | 33% | 0.38 (0.34-0.41) | 0.38 (0.35-0.41) | 0.33 | 7.3 (7-8) | 6.0 (3-8) | 5.7 (9) |
| foul | 25 | 11% | 0.16 (0.16-0.17) | 0.14 (0.13-0.15) | 0.11 | 2.0 (1-3) | 2.0 (2-2) | 1.5 (3) |
| turnover | 21 | 10% | 0.14 (0.11-0.19) | 0.15 (0.13-0.17) | 0.10 | 2.3 (1-4) | 2.0 (2-2) | 1.1 (3) |
| shot_attempt | 102 | 38% | 0.42 (0.39-0.46) | 0.34 (0.30-0.38) | 0.38 | 34.7 (32-40) | 26.7 (20-34) | 24.8 (31) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
