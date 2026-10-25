# meva_seg-meva-full_marengo_visual_N8_T140

263 windows of 9.0928 s (mean) inside the halves; delta = 0 s.

| class | n | positive windows | AP raw | AP contrast | chance AP | spotted raw | spotted contrast | chance spotted (p95) |
|---|---|---|---|---|---|---|---|---|
| vehicle_stops | 76 | 29% | 0.41 (0.40-0.42) | 0.27 (0.27-0.27) | 0.29 | 10.0 (10-10) | 4.0 (3-5) | 6.2 (10) |
| vehicle_starts | 67 | 26% | 0.34 (0.33-0.35) | 0.28 (0.24-0.30) | 0.26 | 10.0 (7-12) | 8.0 (6-10) | 7.1 (11) |
| vehicle_turns | 125 | 49% | 0.55 (0.55-0.56) | 0.54 (0.54-0.55) | 0.49 | 28.0 (26-30) | 26.7 (26-28) | 25.2 (32) |
| vehicle_turns_left | 60 | 25% | 0.32 (0.31-0.32) | 0.27 (0.26-0.29) | 0.25 | 8.3 (7-10) | 9.7 (9-10) | 6.9 (10) |
| vehicle_turns_right | 65 | 33% | 0.39 (0.39-0.40) | 0.43 (0.42-0.43) | 0.33 | 7.0 (6-8) | 6.3 (5-7) | 6.1 (9) |
| vehicle_reverses | 14 | 12% | 0.15 (0.13-0.16) | 0.13 (0.11-0.15) | 0.12 | 0.3 (0-1) | 0.3 (0-1) | 0.9 (2) |
| vehicle_makes_u_turn | 3 | 2% | 0.09 (0.06-0.11) | 0.05 (0.03-0.06) | 0.02 | 0.0 (0-0) | 0.0 (0-0) | 0.0 (0) |
| person_opens_vehicle_door | 64 | 22% | 0.22 (0.22-0.23) | 0.20 (0.19-0.20) | 0.22 | 2.3 (1-4) | 2.7 (2-3) | 4.3 (7) |
| person_closes_vehicle_door | 63 | 20% | 0.18 (0.18-0.18) | 0.17 (0.17-0.17) | 0.20 | 1.7 (1-2) | 0.0 (0-0) | 3.6 (6) |
| person_enters_vehicle | 41 | 15% | 0.28 (0.26-0.31) | 0.26 (0.24-0.27) | 0.15 | 7.3 (6-8) | 7.0 (7-7) | 2.8 (6) |
| person_exits_vehicle | 62 | 21% | 0.22 (0.20-0.23) | 0.22 (0.20-0.25) | 0.21 | 4.3 (3-6) | 3.0 (3-3) | 4.9 (8) |
| vehicle_drops_off_person | 8 | 12% | 0.12 (0.11-0.13) | 0.08 (0.08-0.09) | 0.12 | 0.7 (0-1) | 0.0 (0-0) | 0.7 (2) |
| vehicle_picks_up_person | 4 | 6% | 0.06 (0.05-0.08) | 0.04 (0.04-0.05) | 0.06 | 0.0 (0-0) | 0.0 (0-0) | 0.1 (1) |

AP and spotted: mean over the three phrasings (min-max). Spotted = moments with a peak within delta among the top-n peaks; chance = shuffled scores, 200 draws.
