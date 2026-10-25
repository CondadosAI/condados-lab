# loras-mask_marengo_audio

467 windows of 8.64576 s (mean) inside the halves; delta = 5 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 21% | 0.45 (0.42-0.48) | 0.39 (0.33-0.44) | 0.21 | 13.0 (11-15) | 9.7 (9-11) | 5.1 (8) |
| timeout | 4 | 9% | 0.24 (0.22-0.28) | 0.31 (0.25-0.38) | 0.09 | 2.3 (2-3) | 2.3 (2-3) | 0.2 (1) |
| three | 23 | 9% | 0.31 (0.27-0.34) | 0.30 (0.21-0.34) | 0.09 | 6.3 (4-8) | 7.7 (5-9) | 1.5 (4) |
| made_two | 23 | 9% | 0.16 (0.16-0.17) | 0.22 (0.18-0.28) | 0.09 | 3.3 (3-4) | 6.0 (3-9) | 1.3 (3) |
| substitution | 22 | 33% | 0.35 (0.34-0.35) | 0.34 (0.33-0.35) | 0.33 | 6.7 (6-8) | 6.3 (5-8) | 5.7 (9) |
| foul | 25 | 11% | 0.32 (0.24-0.38) | 0.33 (0.28-0.36) | 0.11 | 8.0 (6-9) | 6.0 (4-8) | 1.4 (3) |
| turnover | 21 | 10% | 0.12 (0.10-0.14) | 0.20 (0.17-0.24) | 0.10 | 1.7 (1-2) | 2.7 (2-3) | 1.0 (3) |
| shot_attempt | 102 | 38% | 0.36 (0.35-0.37) | 0.37 (0.36-0.38) | 0.38 | 23.0 (21-25) | 26.0 (23-29) | 24.7 (31) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
