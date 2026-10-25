# loras-mask_marengo_visual

467 windows of 8.64576 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 19% | 0.61 (0.58-0.62) | 0.62 (0.61-0.63) | 0.19 | 19.0 (15-21) | 19.3 (17-21) | 3.9 (6) |
| timeout | 4 | 9% | 0.49 (0.22-0.65) | 0.67 (0.66-0.68) | 0.09 | 1.3 (1-2) | 2.7 (2-4) | 0.2 (1) |
| three | 23 | 7% | 0.49 (0.43-0.56) | 0.40 (0.34-0.44) | 0.07 | 5.7 (5-6) | 4.7 (4-6) | 0.5 (2) |
| made_two | 23 | 8% | 0.27 (0.18-0.33) | 0.24 (0.17-0.33) | 0.08 | 1.3 (1-2) | 1.3 (1-2) | 0.5 (2) |
| substitution | 22 | 31% | 0.40 (0.38-0.43) | 0.40 (0.39-0.41) | 0.31 | 5.7 (5-6) | 3.3 (2-4) | 4.8 (7) |
| foul | 25 | 8% | 0.22 (0.16-0.26) | 0.25 (0.18-0.30) | 0.08 | 3.0 (2-4) | 2.7 (2-3) | 0.6 (2) |
| turnover | 21 | 7% | 0.25 (0.19-0.29) | 0.21 (0.18-0.23) | 0.07 | 2.7 (2-4) | 1.7 (1-2) | 0.5 (2) |
| shot_attempt | 102 | 30% | 0.56 (0.55-0.57) | 0.34 (0.25-0.40) | 0.30 | 26.7 (23-29) | 14.3 (9-19) | 9.8 (14) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
