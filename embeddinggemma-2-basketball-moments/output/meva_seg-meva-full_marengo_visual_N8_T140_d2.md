# meva_seg-meva-full_marengo_visual_N8_T140

263 windows of 9.0928 s (mean) inside the halves; delta = 2 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| vehicle_stops | 76 | 37% | 0.50 (0.49-0.51) | 0.33 (0.33-0.34) | 0.37 | 17.0 (16-18) | 11.7 (10-13) | 14.6 (19) |
| vehicle_starts | 67 | 33% | 0.44 (0.43-0.45) | 0.37 (0.32-0.40) | 0.33 | 19.3 (17-21) | 14.3 (11-18) | 13.4 (18) |
| vehicle_turns | 125 | 52% | 0.58 (0.58-0.59) | 0.59 (0.58-0.60) | 0.52 | 49.7 (47-53) | 45.0 (44-46) | 43.6 (50) |
| vehicle_turns_left | 60 | 29% | 0.36 (0.35-0.38) | 0.32 (0.30-0.34) | 0.29 | 16.3 (15-18) | 12.0 (10-14) | 11.3 (16) |
| vehicle_turns_right | 65 | 37% | 0.42 (0.41-0.42) | 0.47 (0.45-0.49) | 0.37 | 17.0 (16-18) | 17.7 (16-19) | 13.6 (18) |
| vehicle_reverses | 14 | 14% | 0.16 (0.15-0.18) | 0.16 (0.13-0.18) | 0.14 | 1.3 (1-2) | 1.3 (1-2) | 1.4 (3) |
| vehicle_makes_u_turn | 3 | 3% | 0.10 (0.08-0.12) | 0.05 (0.03-0.06) | 0.03 | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0) |
| person_opens_vehicle_door | 64 | 30% | 0.28 (0.28-0.28) | 0.25 (0.25-0.25) | 0.30 | 7.7 (7-9) | 6.7 (6-7) | 8.9 (12) |
| person_closes_vehicle_door | 63 | 27% | 0.26 (0.25-0.27) | 0.22 (0.22-0.22) | 0.27 | 5.3 (5-6) | 1.7 (1-2) | 6.2 (10) |
| person_enters_vehicle | 41 | 19% | 0.31 (0.29-0.34) | 0.33 (0.30-0.36) | 0.19 | 8.3 (7-9) | 8.0 (8-8) | 4.2 (7) |
| person_exits_vehicle | 62 | 25% | 0.23 (0.22-0.24) | 0.24 (0.23-0.27) | 0.25 | 10.3 (9-11) | 10.0 (9-11) | 8.9 (13) |
| vehicle_drops_off_person | 8 | 13% | 0.13 (0.12-0.14) | 0.09 (0.08-0.09) | 0.13 | 0.7 (0-1) | 0.0 (0-0) | 0.8 (2) |
| vehicle_picks_up_person | 4 | 6% | 0.06 (0.05-0.08) | 0.04 (0.04-0.05) | 0.06 | 0.0 (0-0) | 0.0 (0-0) | 0.2 (1) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
