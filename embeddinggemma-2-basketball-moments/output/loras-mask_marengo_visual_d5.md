# loras-mask_marengo_visual

467 windows of 8.64576 s (mean) inside the halves; delta = 5 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 21% | 0.61 (0.58-0.63) | 0.62 (0.62-0.63) | 0.21 | 22.0 (19-24) | 22.0 (19-24) | 5.1 (8) |
| timeout | 4 | 9% | 0.48 (0.22-0.63) | 0.64 (0.64-0.66) | 0.09 | 1.3 (1-2) | 3.0 (2-4) | 0.2 (1) |
| three | 23 | 9% | 0.44 (0.38-0.49) | 0.37 (0.31-0.40) | 0.09 | 11.7 (11-12) | 11.0 (10-12) | 1.5 (4) |
| made_two | 23 | 9% | 0.28 (0.22-0.34) | 0.25 (0.19-0.32) | 0.09 | 8.3 (6-11) | 6.7 (4-9) | 1.3 (3) |
| substitution | 22 | 33% | 0.42 (0.40-0.44) | 0.41 (0.40-0.42) | 0.33 | 6.0 (6-6) | 5.3 (5-6) | 5.7 (9) |
| foul | 25 | 11% | 0.29 (0.21-0.39) | 0.32 (0.23-0.40) | 0.11 | 4.7 (4-5) | 4.3 (2-6) | 1.4 (3) |
| turnover | 21 | 10% | 0.25 (0.20-0.29) | 0.21 (0.18-0.24) | 0.10 | 4.7 (4-5) | 3.0 (2-4) | 1.0 (3) |
| shot_attempt | 102 | 38% | 0.63 (0.60-0.64) | 0.40 (0.30-0.46) | 0.38 | 53.0 (52-55) | 40.0 (27-48) | 24.7 (31) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
