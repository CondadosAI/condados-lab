# Chance AP: share of positives vs the expected AP of a random ranking

Exact = closed form (see the docstring); shuffled = mean AP over random rankings, a check.

## QVHighlights (per query, mean over queries, percent)

| queries | share of positive segments | exact random AP | shuffled (500 per query) |
|---|---|---|---|
| 92 | 34.8 | 43.7 | 43.7 |

## loras, windows = loras-mask_marengo_visual.npz, delta = 5 s

| class | n | windows | positive | share | exact random AP | shuffled (2000) |
|---|---|---|---|---|---|---|
| free_throw | 31 | 467 | 97 | 0.21 | 0.22 | 0.22 |
| timeout | 4 | 467 | 42 | 0.09 | 0.10 | 0.10 |
| three | 23 | 467 | 42 | 0.09 | 0.10 | 0.10 |
| made_two | 23 | 467 | 43 | 0.09 | 0.10 | 0.10 |
| substitution | 22 | 467 | 153 | 0.33 | 0.34 | 0.34 |
| foul | 25 | 467 | 52 | 0.11 | 0.12 | 0.12 |
| turnover | 21 | 467 | 45 | 0.10 | 0.11 | 0.11 |
| shot_attempt | 102 | 467 | 176 | 0.38 | 0.38 | 0.39 |

## misericordia, windows = misericordia-mask_marengo_visual.npz, delta = 5 s

| class | n | windows | positive | share | exact random AP | shuffled (2000) |
|---|---|---|---|---|---|---|
| free_throw | 46 | 552 | 135 | 0.24 | 0.25 | 0.25 |
| timeout | 6 | 552 | 72 | 0.13 | 0.14 | 0.14 |
| three | 9 | 552 | 19 | 0.03 | 0.04 | 0.04 |
| made_two | 46 | 552 | 93 | 0.17 | 0.18 | 0.18 |
| substitution | 32 | 552 | 198 | 0.36 | 0.37 | 0.37 |
| foul | 41 | 552 | 77 | 0.14 | 0.15 | 0.15 |
| turnover | 24 | 552 | 47 | 0.09 | 0.09 | 0.10 |
| shot_attempt | 135 | 552 | 212 | 0.38 | 0.39 | 0.39 |

## meva, windows = meva-full_marengo_visual.npz, delta = 2 s

| class | n | windows | positive | share | exact random AP | shuffled (2000) |
|---|---|---|---|---|---|---|
| vehicle_stops | 76 | 264 | 98 | 0.37 | 0.38 | 0.38 |
| vehicle_starts | 67 | 264 | 86 | 0.33 | 0.34 | 0.34 |
| vehicle_turns | 125 | 264 | 138 | 0.52 | 0.53 | 0.53 |
| vehicle_turns_left | 60 | 264 | 75 | 0.28 | 0.30 | 0.30 |
| vehicle_turns_right | 65 | 264 | 96 | 0.36 | 0.38 | 0.38 |
| vehicle_reverses | 14 | 264 | 37 | 0.14 | 0.16 | 0.16 |
| vehicle_makes_u_turn | 3 | 264 | 7 | 0.03 | 0.05 | 0.05 |
| person_opens_vehicle_door | 65 | 264 | 80 | 0.30 | 0.32 | 0.32 |
| person_closes_vehicle_door | 64 | 264 | 71 | 0.27 | 0.28 | 0.28 |
| person_enters_vehicle | 42 | 264 | 51 | 0.19 | 0.21 | 0.21 |
| person_exits_vehicle | 62 | 264 | 65 | 0.25 | 0.26 | 0.26 |
| vehicle_drops_off_person | 8 | 264 | 34 | 0.13 | 0.15 | 0.14 |
| vehicle_picks_up_person | 4 | 264 | 16 | 0.06 | 0.08 | 0.08 |
