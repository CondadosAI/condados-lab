# loras-mask_marengo_audio

467 windows of 8.64576 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 19% | 0.42 (0.41-0.45) | 0.36 (0.30-0.40) | 0.19 | 10.7 (9-13) | 8.3 (7-9) | 3.9 (6) |
| timeout | 4 | 9% | 0.23 (0.21-0.27) | 0.30 (0.24-0.39) | 0.09 | 2.0 (1-3) | 2.3 (2-3) | 0.2 (1) |
| three | 23 | 7% | 0.27 (0.25-0.30) | 0.30 (0.21-0.35) | 0.07 | 2.3 (2-3) | 2.7 (2-3) | 0.5 (2) |
| made_two | 23 | 8% | 0.13 (0.12-0.14) | 0.21 (0.15-0.29) | 0.08 | 0.7 (0-1) | 1.3 (0-2) | 0.5 (2) |
| substitution | 22 | 31% | 0.33 (0.32-0.34) | 0.33 (0.31-0.34) | 0.31 | 6.7 (6-8) | 4.7 (3-6) | 4.8 (7) |
| foul | 25 | 8% | 0.28 (0.18-0.35) | 0.31 (0.25-0.35) | 0.08 | 4.7 (4-5) | 3.3 (2-4) | 0.6 (2) |
| turnover | 21 | 7% | 0.09 (0.07-0.11) | 0.13 (0.10-0.18) | 0.07 | 1.0 (0-2) | 2.3 (2-3) | 0.5 (2) |
| shot_attempt | 102 | 30% | 0.26 (0.24-0.27) | 0.28 (0.27-0.28) | 0.30 | 9.3 (9-10) | 10.0 (7-12) | 9.8 (14) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
