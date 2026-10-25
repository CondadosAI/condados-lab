# meva-full_marengo_visual

264 windows of 9.0928 s (mean) inside the halves; delta = 0 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| vehicle_stops | 76 | 29% | 0.41 (0.39-0.44) | 0.27 (0.26-0.28) | 0.29 | 8.3 (7-10) | 6.3 (4-8) | 6.3 (9) |
| vehicle_starts | 67 | 26% | 0.36 (0.30-0.43) | 0.24 (0.23-0.25) | 0.26 | 11.7 (10-13) | 6.0 (5-8) | 7.0 (11) |
| vehicle_turns | 125 | 49% | 0.55 (0.54-0.57) | 0.52 (0.49-0.53) | 0.49 | 29.0 (28-31) | 30.0 (29-32) | 24.6 (30) |
| vehicle_turns_left | 60 | 25% | 0.31 (0.30-0.33) | 0.28 (0.24-0.32) | 0.25 | 8.0 (6-10) | 8.3 (7-9) | 7.0 (10) |
| vehicle_turns_right | 65 | 33% | 0.37 (0.37-0.38) | 0.37 (0.35-0.40) | 0.33 | 5.7 (5-6) | 7.0 (6-8) | 6.2 (9) |
| vehicle_reverses | 14 | 12% | 0.16 (0.15-0.17) | 0.10 (0.09-0.10) | 0.12 | 1.0 (0-2) | 0.0 (0-0) | 1.0 (3) |
| vehicle_makes_u_turn | 3 | 2% | 0.07 (0.04-0.09) | 0.03 (0.02-0.04) | 0.02 | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0) |
| person_opens_vehicle_door | 65 | 22% | 0.35 (0.33-0.35) | 0.22 (0.22-0.23) | 0.22 | 6.7 (5-8) | 4.0 (4-4) | 4.4 (7) |
| person_closes_vehicle_door | 64 | 20% | 0.28 (0.26-0.29) | 0.20 (0.20-0.21) | 0.20 | 6.3 (6-7) | 1.0 (1-1) | 4.0 (7) |
| person_enters_vehicle | 42 | 15% | 0.21 (0.19-0.22) | 0.29 (0.26-0.31) | 0.15 | 5.7 (5-6) | 6.3 (6-7) | 2.8 (5) |
| person_exits_vehicle | 62 | 21% | 0.26 (0.23-0.29) | 0.23 (0.22-0.23) | 0.21 | 7.7 (7-9) | 3.0 (2-4) | 5.1 (8) |
| vehicle_drops_off_person | 8 | 12% | 0.13 (0.10-0.18) | 0.08 (0.08-0.09) | 0.12 | 0.7 (0-1) | 0.0 (0-0) | 0.8 (2) |
| vehicle_picks_up_person | 4 | 6% | 0.10 (0.04-0.17) | 0.05 (0.04-0.05) | 0.06 | 0.3 (0-1) | 0.0 (0-0) | 0.1 (1) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
