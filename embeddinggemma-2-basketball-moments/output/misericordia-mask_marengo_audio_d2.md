# misericordia-mask_marengo_audio

552 windows of 9.00539 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 46 | 22% | 0.27 (0.26-0.30) | 0.32 (0.30-0.36) | 0.22 | 11.0 (8-15) | 9.0 (7-12) | 6.5 (10) |
| timeout | 6 | 12% | 0.33 (0.24-0.40) | 0.56 (0.55-0.57) | 0.12 | 3.7 (2-5) | 3.7 (3-4) | 0.6 (2) |
| three | 9 | 3% | 0.09 (0.07-0.12) | 0.07 (0.03-0.10) | 0.03 | 0.3 (0-1) | 0.7 (0-1) | 0.1 (1) |
| made_two | 46 | 12% | 0.18 (0.17-0.18) | 0.15 (0.11-0.20) | 0.12 | 2.0 (1-3) | 1.3 (0-3) | 1.4 (3) |
| substitution | 32 | 34% | 0.38 (0.33-0.46) | 0.46 (0.44-0.50) | 0.34 | 11.3 (9-15) | 12.0 (9-14) | 7.3 (10) |
| foul | 41 | 9% | 0.30 (0.24-0.36) | 0.16 (0.15-0.17) | 0.09 | 7.3 (5-10) | 2.7 (2-3) | 1.5 (4) |
| turnover | 24 | 7% | 0.12 (0.08-0.16) | 0.06 (0.05-0.08) | 0.07 | 0.3 (0-1) | 0.0 (0-0) | 0.3 (1) |
| shot_attempt | 135 | 31% | 0.42 (0.40-0.44) | 0.26 (0.24-0.30) | 0.31 | 15.0 (13-16) | 4.0 (2-7) | 11.5 (16) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
