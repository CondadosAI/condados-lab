# misericordia-mask_marengo_audio

552 windows of 9.00539 s (mean) inside the halves; delta = 5 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 46 | 24% | 0.30 (0.29-0.32) | 0.34 (0.32-0.37) | 0.24 | 11.7 (9-16) | 9.0 (8-11) | 8.0 (11) |
| timeout | 6 | 13% | 0.32 (0.24-0.38) | 0.54 (0.53-0.55) | 0.13 | 3.7 (2-5) | 3.7 (3-4) | 0.7 (2) |
| three | 9 | 3% | 0.16 (0.13-0.18) | 0.15 (0.10-0.18) | 0.03 | 1.3 (1-2) | 1.7 (0-3) | 0.2 (1) |
| made_two | 46 | 17% | 0.29 (0.26-0.32) | 0.21 (0.16-0.29) | 0.17 | 5.7 (4-7) | 7.3 (3-15) | 4.2 (8) |
| substitution | 32 | 36% | 0.40 (0.35-0.46) | 0.46 (0.44-0.49) | 0.36 | 11.7 (9-15) | 13.0 (11-14) | 8.7 (12) |
| foul | 41 | 14% | 0.31 (0.25-0.38) | 0.24 (0.22-0.27) | 0.14 | 12.7 (9-17) | 4.7 (4-5) | 3.1 (6) |
| turnover | 24 | 9% | 0.13 (0.09-0.17) | 0.08 (0.06-0.09) | 0.09 | 2.7 (2-3) | 0.7 (0-2) | 1.2 (3) |
| shot_attempt | 135 | 38% | 0.51 (0.47-0.55) | 0.32 (0.29-0.36) | 0.38 | 47.7 (41-53) | 30.0 (19-40) | 34.2 (39) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
