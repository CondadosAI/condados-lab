# loras_L4_S2_F2_T140_mask_0-end

2026 windows of 4 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| free_throw | 31 | 17% | 0.30 (0.22-0.34) | 0.27 (0.21-0.30) | 0.17 | 7.3 (4-10) | 4.7 (3-7) | 4.0 (7) |
| timeout | 4 | 9% | 0.43 (0.07-0.64) | 0.58 (0.43-0.68) | 0.09 | 1.0 (0-2) | 3.0 (3-3) | 0.3 (1) |
| three | 23 | 5% | 0.06 (0.06-0.06) | 0.04 (0.04-0.05) | 0.05 | 1.3 (0-2) | 0.0 (0-0) | 0.6 (2) |
| made_two | 23 | 4% | 0.10 (0.08-0.12) | 0.06 (0.05-0.08) | 0.04 | 0.3 (0-1) | 0.7 (0-2) | 0.6 (2) |
| substitution | 22 | 27% | 0.35 (0.27-0.40) | 0.33 (0.28-0.36) | 0.27 | 6.0 (5-8) | 4.3 (1-8) | 4.8 (8) |
| foul | 25 | 5% | 0.08 (0.06-0.10) | 0.06 (0.06-0.06) | 0.05 | 0.7 (0-2) | 0.7 (0-1) | 0.6 (2) |
| turnover | 21 | 4% | 0.08 (0.06-0.09) | 0.06 (0.06-0.08) | 0.04 | 1.0 (1-1) | 1.7 (1-2) | 0.4 (2) |
| shot_attempt | 102 | 20% | 0.29 (0.25-0.33) | 0.19 (0.16-0.23) | 0.20 | 19.7 (15-24) | 10.3 (7-14) | 10.5 (15) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
