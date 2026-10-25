# loras_L4_S2_F2_T140_mask_0-end

2026 windows of 4 s (mean) inside the halves; delta = 5 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 20% | 0.35 (0.25-0.40) | 0.31 (0.25-0.34) | 0.20 | 11.0 (7-14) | 7.3 (6-9) | 5.1 (8) |
| timeout | 4 | 9% | 0.41 (0.08-0.60) | 0.55 (0.41-0.64) | 0.09 | 1.3 (0-2) | 3.0 (3-3) | 0.3 (1) |
| three | 23 | 8% | 0.09 (0.09-0.10) | 0.08 (0.08-0.09) | 0.08 | 1.7 (0-3) | 0.3 (0-1) | 1.4 (3) |
| made_two | 23 | 8% | 0.13 (0.11-0.15) | 0.09 (0.07-0.11) | 0.08 | 2.3 (2-3) | 0.7 (0-2) | 1.3 (3) |
| substitution | 22 | 30% | 0.37 (0.30-0.42) | 0.35 (0.31-0.38) | 0.30 | 7.0 (6-8) | 5.0 (2-9) | 5.5 (9) |
| foul | 25 | 9% | 0.12 (0.11-0.13) | 0.10 (0.09-0.10) | 0.09 | 3.0 (2-5) | 1.7 (1-2) | 1.6 (4) |
| turnover | 21 | 7% | 0.11 (0.09-0.13) | 0.10 (0.09-0.11) | 0.07 | 2.0 (1-3) | 2.0 (1-3) | 1.0 (3) |
| shot_attempt | 102 | 32% | 0.40 (0.37-0.45) | 0.30 (0.26-0.35) | 0.32 | 42.7 (35-50) | 30.0 (21-39) | 24.5 (30) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
