# meva-full_marengo_visual

264 windows of 9.0928 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| vehicle_stops | 76 | 37% | 0.50 (0.46-0.56) | 0.35 (0.33-0.37) | 0.37 | 19.7 (18-22) | 13.0 (10-15) | 14.7 (19) |
| vehicle_starts | 67 | 33% | 0.47 (0.40-0.55) | 0.31 (0.30-0.32) | 0.33 | 20.0 (17-23) | 9.7 (8-12) | 12.7 (17) |
| vehicle_turns | 125 | 52% | 0.58 (0.57-0.59) | 0.56 (0.53-0.57) | 0.52 | 49.3 (48-51) | 49.7 (45-52) | 42.7 (49) |
| vehicle_turns_left | 60 | 28% | 0.34 (0.33-0.36) | 0.32 (0.28-0.36) | 0.28 | 14.3 (12-17) | 10.3 (7-12) | 11.1 (16) |
| vehicle_turns_right | 65 | 36% | 0.41 (0.40-0.42) | 0.41 (0.39-0.44) | 0.36 | 16.7 (15-18) | 17.0 (15-19) | 13.6 (18) |
| vehicle_reverses | 14 | 14% | 0.19 (0.18-0.20) | 0.12 (0.11-0.12) | 0.14 | 1.3 (0-2) | 0.0 (0-0) | 1.6 (4) |
| vehicle_makes_u_turn | 3 | 3% | 0.08 (0.05-0.11) | 0.05 (0.03-0.07) | 0.03 | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0) |
| person_opens_vehicle_door | 65 | 30% | 0.40 (0.39-0.41) | 0.29 (0.28-0.29) | 0.30 | 15.3 (13-17) | 8.3 (8-9) | 8.7 (12) |
| person_closes_vehicle_door | 64 | 27% | 0.36 (0.35-0.38) | 0.27 (0.26-0.28) | 0.27 | 7.7 (6-10) | 2.7 (2-3) | 6.9 (10) |
| person_enters_vehicle | 42 | 19% | 0.24 (0.23-0.26) | 0.39 (0.35-0.41) | 0.19 | 7.3 (7-8) | 7.3 (7-8) | 4.2 (7) |
| person_exits_vehicle | 62 | 25% | 0.27 (0.25-0.31) | 0.26 (0.25-0.26) | 0.25 | 11.3 (9-14) | 10.0 (9-11) | 9.1 (13) |
| vehicle_drops_off_person | 8 | 13% | 0.13 (0.11-0.18) | 0.09 (0.08-0.09) | 0.13 | 0.7 (0-1) | 0.0 (0-0) | 0.9 (2) |
| vehicle_picks_up_person | 4 | 6% | 0.10 (0.04-0.17) | 0.05 (0.04-0.05) | 0.06 | 0.3 (0-1) | 0.0 (0-0) | 0.2 (1) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
