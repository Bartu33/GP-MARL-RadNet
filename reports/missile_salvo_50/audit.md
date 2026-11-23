# Hedef sınıfları pilot dataset denetimi

Dataset: `C:\Users\tahau\Desktop\marl\demo\data\processed\missile_salvo_50`  
Tüm kontrolleri geçen hedef: **125 / 125**

Değerler `min / medyan / max`. Bantlar `docs/target_classification.md` ve `src/core/target_classes.py` ile aynı kaynaktan.

## ep-000000-0edf45f8e919 (train)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 736 m
- Salvo: **converging**, 2 füze, süre 22 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni 173.3°, giriş +0.0 s); T1 barrel_roll_dive → Zone A (tehdit ekseni 116.0°, giriş +3.0 s); T2 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-1375, -428) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 387; T1 Zone A 321
- Radar: 3 — R1 (4125, 72) yön 158°, R2 (-4875, 3322) yön -139°, R3 (3375, -4928) yön 123°
- Planlanan görünürlük oranı: [1.0, 0.956, 1.0]
- Arama barları (°): R1 [6.5, 12.5, 18.5, 24.5, 30.5, 36.5, 42.5]; R2 [7.2, 13.2, 19.2, 25.2, 31.2, 37.2, 43.2, 49.2]; R3 [3.6, 9.6, 15.6, 21.6, 27.6, 33.6]
- Yanlış alarm raporu: 26

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 762 / 884 / 1008 | 619 / 9323 / 10019 | -849.8 / -456.1 / 0.0 | 0.00 / 9.47 / 26.20 | 0.0 / 8.0 / 16.9 | -6.5 / -0.2 / 5.2 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 775 / 807 / 932 | 604 / 8084 / 8538 | -772.4 / -382.3 / 0.0 | 0.00 / 13.28 / 25.74 | 0.0 / 2.7 / 15.5 | -7.4 / -0.6 / 4.3 | 3 {0: 2, 2: 1} | 7% | 7% | 9.5 / 10.5 | 23.4 / 23.5 / 24.2 | 590 / 632 / 761 | tamam |
| T2 | uav/fixed_wing (L0) | CT+CT | 38 / 38 / 38 | 702 / 722 / 743 | 1.1 / 1.1 / 1.1 | 0.62 / 0.62 / 0.68 | 9.2 / 9.2 / 10.1 | -11.5 / -1.8 / 1.4 | 9 {2: 9} | 20% | 80% | 4.0 / 5.0 | 14.5 / 47.1 / 56.8 | 4 / 25 / 38 | tamam |

## ep-000001-604364fbbb56 (train)
- Arazi: valley_sinuous_pass / asymmetric / snowy_highlands
- Arazi: **twisting_gorge**, ölçülen kabartı 2186 m
- Salvo: **single**, 1 füze, süre 26.5 s (hedefe eşzamanlı varış)
- Profiller: T0 cruise_weave → Zone A (tehdit ekseni -97.3°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-1161, 840); Facility 1 [facility] (-1165, 3243) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 735
- Radar: 2 — R1 (-1411, -1160) yön -114°, R2 (2839, 5090) yön -131°
- Planlanan görünürlük oranı: [1.0]
- Arama barları (°): R1 [3.8, 9.8, 15.8]; R2 [1.7]
- Yanlış alarm raporu: 24

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | terrain_follow+weave+pop_up+terminal_descent | 229 / 230 / 230 | 70 / 71 / 175 | -11.8 / 0.0 / 18.0 | 0.00 / 0.09 / 2.94 | 0.0 / 0.0 / 5.5 | -12.8 / -10.4 / -0.5 | 20 {0: 13, 1: 7} | 34% | 96% | 0.5 / 1.5 | 12.3 / 39.2 / 76.4 | 190 / 228 / 230 | tamam |

## ep-000002-0cda2851a9d2 (validation)
- Arazi: basin_highlands / dense / grassland_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1436 m
- Salvo: **diverging**, 3 füze, süre 21.5 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 165.8°, giriş -2.5 s); T1 popup_dive → Zone B (tehdit ekseni 176.2°, giriş +0.0 s); T2 shallow_dive → Zone C (tehdit ekseni 167.6°, giriş -5.0 s)
- Bölgeler: Zone A [zone] (-124, -669); Zone B [zone] (1658, 3300); Zone C [zone] (4331, -766); Facility 1 [facility] (-4284, -2217) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 302, Zone B 4249, Zone C 4677; T1 Zone A 4256, Zone B 631, Zone C 5386; T2 Zone A 1019, Zone B 3241, Zone C 158
- Radar: 1 — R1 (-1795, -4878) yön 118°
- Planlanan görünürlük oranı: [0.659, 0.432, 0.591]
- Arama barları (°): R1 [0.8, 6.8, 12.8, 18.8, 24.8, 30.8, 36.8, 42.8]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 780 / 819 / 819 | 197 / 6451 / 10250 | -691.2 / -691.2 / 0.0 | 0.00 / 0.00 / 16.68 | 0.0 / 0.0 / 0.0 | -5.6 / 1.2 / 3.7 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 503 / 524 / 624 | 47 / 245 / 1815 | -192.2 / 0.7 / 255.1 | 0.03 / 0.97 / 26.27 | 0.0 / 0.0 / 0.0 | 0.3 / 3.7 / 4.8 | 3 {0: 3} | 7% | 12% | 18.5 / 19.5 | 27.5 / 29.4 / 42.7 | 32 / 54 / 76 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+push_over+dive | 628 / 637 / 637 | 270 / 4665 / 6844 | -334.3 / -334.3 / -57.8 | 0.00 / 0.00 / 11.91 | 0.0 / 0.0 / 0.0 | -0.2 / 3.5 / 4.3 | 3 {0: 3} | 7% | 53% | 9.5 / 10.5 | 23.2 / 24.1 / 45.8 | 251 / 277 / 300 | tamam |

## ep-000003-2c9eabf5c670 (train)
- Arazi: ridge_multi_obstacle / asymmetric / snowy_highlands
- Arazi: **knife_ridges**, ölçülen kabartı 1054 m
- Salvo: **single**, 1 füze, süre 14 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni -94.4°, giriş +0.0 s); T1 traffic/multirotor → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (798, -726) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 49
- Radar: 3 — R1 (798, 4024) yön -122°, R2 (5298, 1274) yön -129°, R3 (798, -3976) yön -101°
- Planlanan görünürlük oranı: [1.0, 0.966]
- Arama barları (°): R1 [1.4, 7.4, 13.4, 19.4]; R2 [-3.3, 2.7, 8.7, 14.7, 20.7]; R3 [-6.3, -0.3, 5.7, 11.7, 17.7, 23.7, 29.7, 35.7]
- Yanlış alarm raporu: 17

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 818 / 835 / 940 | 516 / 3000 / 3333 | -685.8 / -58.0 / 0.0 | 0.00 / 2.52 / 26.08 | 0.0 / 1.0 / 17.1 | -7.5 / -3.8 / 2.9 | 5 {0: 2, 1: 3} | 18% | 36% | 7.5 / 9.5 | 14.8 / 25.9 / 40.2 | 456 / 760 / 869 | tamam |
| T1 | uav/multirotor (L0) | hover+hover | 0 / 0 / 20 | 44 / 50 / 50 | -0.0 / -0.0 / 2.2 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -21.0 / -18.9 / -13.9 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000004-1f2dca306803 (test)
- Arazi: mountain_chokepoint / dense / dry_canyon
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1408 m
- Salvo: **diverging**, 3 füze, süre 18.5 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 51.5°, giriş -6.5 s); T1 dogleg_serpentine → Zone B (tehdit ekseni 74.2°, giriş +0.0 s); T2 supersonic_skim → Zone C (tehdit ekseni 45.1°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (1338, 378); Zone B [zone] (-1338, -2398); Zone C [zone] (268, 3718); Facility 1 [facility] (-5710, 1297); Facility 2 [facility] (4027, -2439) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 281, Zone B 3637, Zone C 2726; T1 Zone A 288, Zone B 63, Zone C 1317; T2 Zone A 3837, Zone B 6835, Zone C 562
- Radar: 2 — R1 (-661, -2184) yön 66°, R2 (3589, -5434) yön 90°
- Planlanan görünürlük oranı: [0.737, 0.395, 0.868]
- Arama barları (°): R1 [-17.9, -11.9, -5.9, 0.1, 6.1, 12.1, 18.1, 24.1]; R2 [-4.4, 1.6, 7.6, 13.6, 19.6]
- Yanlış alarm raporu: 14

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 841 / 877 / 877 | 179 / 2260 / 6548 | -620.7 / -506.5 / 0.0 | 0.00 / 0.00 / 17.52 | 0.0 / 0.0 / 0.0 | -7.4 / -0.1 / 3.9 | 5 {0: 4, 1: 1} | 14% | 0% | 5.0 / — | 14.1 / 22.0 / 38.9 | 647 / 713 / 837 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 512 / 636 / 639 | 35 / 416 / 873 | -339.1 / 0.9 / 313.8 | 0.11 / 11.10 / 26.26 | 0.0 / 0.0 / 16.8 | -3.8 / -2.0 / 3.1 | 1 {0: 1} | 3% | 0% | 8.0 / — | 32.7 / 32.7 / 32.7 | 464 / 464 / 464 | tamam |
| T2 | missile/supersonic (L2) | terrain_follow+terminal_weave | 608 / 610 / 611 | 52 / 87 / 105 | 0.0 / 19.2 / 26.4 | 0.02 / 0.31 / 10.66 | 0.0 / 0.0 / 9.8 | -6.5 / -3.0 / 3.2 | 2 {0: 2} | 5% | 0% | 8.0 / — | 13.2 / 21.4 / 29.6 | 582 / 582 / 583 | tamam |

## ep-000005-0510af767329 (train)
- Arazi: valley_linear_canyon / sparse / green_forest_valley
- Arazi: **deep_canyon**, ölçülen kabartı 1251 m
- Salvo: **converging**, 2 füze, süre 23 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 131.8°, giriş -3.5 s); T1 dogleg_serpentine → Zone A (tehdit ekseni 87.3°, giriş +0.0 s); T2 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-844, -866) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 112; T1 Zone A 408
- Radar: 1 — R1 (5156, -116) yön 153°
- Planlanan görünürlük oranı: [0.83, 0.447, 1.0]
- Arama barları (°): R1 [2.7, 8.7, 14.7, 20.7, 26.7, 32.7]
- Yanlış alarm raporu: 7

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 836 / 912 / 912 | 266 / 7253 / 10467 | -668.3 / -668.3 / 0.0 | 0.00 / 9.73 / 13.97 | 0.0 / 0.9 / 12.7 | -7.8 / -2.0 / 3.3 | 3 {0: 3} | 7% | 7% | 21.0 / 22.0 | 25.4 / 34.7 / 47.6 | 446 / 542 / 621 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 481 / 549 / 598 | 46 / 179 / 746 | -337.2 / 20.4 / 259.9 | 0.01 / 3.19 / 26.26 | 0.0 / 0.1 / 16.8 | -5.4 / -2.5 / 5.7 | 7 {0: 7} | 15% | 26% | 16.5 / 17.5 | 12.0 / 24.0 / 54.4 | 32 / 189 / 261 | tamam |
| T2 | aircraft/fighter (L1) | CA | 120 / 168 / 216 | 958 / 1003 / 2060 | 22.4 / 22.4 / 22.4 | 0.04 / 0.06 / 0.08 | 0.0 / 0.0 / 0.0 | 9.5 / 9.9 / 10.5 | 13 {0: 13} | 28% | 93% | 1.0 / 2.0 | 10.1 / 46.0 / 54.2 | 1 / 28 / 45 | tamam |

## ep-000006-75d86ec2fc7b (train)
- Arazi: mountain_chokepoint / sparse / snowy_highlands
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1688 m
- Salvo: **single**, 1 füze, süre 29.5 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 45.6°, giriş +0.0 s); T1 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-550, -251); Facility 1 [facility] (2231, -2552); Facility 2 [facility] (-1184, -4692) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 171
- Radar: 3 — R1 (3450, 4749) yön -168°, R2 (700, 6499) yön 6°, R3 (-2800, -1501) yön 41°
- Planlanan görünürlük oranı: [0.817, 1.0]
- Arama barları (°): R1 [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]; R2 [4.8, 10.8, 16.8, 22.8, 28.8, 34.8, 40.8, 46.8]; R3 [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0]
- Yanlış alarm raporu: 27

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 796 / 834 / 834 | 96 / 4129 / 8074 | -678.6 / -192.0 / 0.0 | 0.00 / 0.00 / 16.12 | 0.0 / 0.0 / 0.0 | -6.4 / -1.6 / 2.9 | 8 {1: 6, 2: 2} | 14% | 32% | 11.0 / 12.0 | 10.3 / 29.6 / 54.9 | 486 / 704 / 775 | tamam |
| T1 | uav/fixed_wing (L0) | CV | 35 / 35 / 35 | 427 / 451 / 454 | 2.0 / 2.0 / 2.0 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -5.4 / -4.2 / -1.6 | 11 {0: 11} | 19% | 88% | 1.0 / 4.0 | 18.4 / 40.9 / 58.7 | 1 / 3 / 6 | tamam |

## ep-000007-1bd5690e9ddc (train)
- Arazi: valley_sinuous_pass / dense / snowy_highlands
- Arazi: **twisting_gorge**, ölçülen kabartı 1398 m
- Salvo: **diverging**, 2 füze, süre 21 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni -153.3°, giriş +6.5 s); T1 shallow_dive → Zone B (tehdit ekseni -139.1°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-309, -690); Zone B [zone] (2810, 1261); Facility 1 [facility] (206, 5944) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 444, Zone B 4115; T1 Zone A 522, Zone B 90
- Radar: 1 — R1 (1751, -1215) yön -167°
- Planlanan görünürlük oranı: [0.93, 0.814]
- Arama barları (°): R1 [8.0, 14.0, 20.0, 26.0, 32.0, 38.0, 44.0, 50.0]
- Yanlış alarm raporu: 4

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 820 / 863 / 953 | 495 / 4267 / 4770 | -791.1 / 0.0 / 0.0 | 0.00 / 0.00 / 26.19 | 0.0 / 0.0 / 16.2 | -6.4 / -5.0 / 2.9 | 3 {0: 3} | 7% | 10% | 18.5 / 19.5 | 26.5 / 30.2 / 60.9 | 497 / 580 / 657 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 711 / 812 / 812 | 314 / 5116 / 5773 | -367.4 / -367.4 / 0.0 | 0.00 / 0.00 / 12.79 | 0.0 / 0.0 / 0.0 | -4.7 / -1.2 / 6.5 | 9 {0: 9} | 21% | 62% | 7.5 / 8.5 | 17.5 / 29.3 / 53.2 | 473 / 754 / 779 | tamam |

## ep-000008-8e1f27fc2caa (validation)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 799 m
- Salvo: **converging**, 3 füze, süre 29.5 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 137.8°, giriş +3.0 s); T1 barrel_roll_dive → Zone A (tehdit ekseni 47.8°, giriş +9.5 s); T2 cruise_weave → Zone A (tehdit ekseni -118.9°, giriş +0.0 s); T3 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-413, 170); Facility 1 [facility] (-1498, -1403); Facility 2 [facility] (4946, -1230) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 93; T1 Zone A 712; T2 Zone A 811
- Radar: 2 — R1 (1087, -5330) yön 152°, R2 (-6663, 1670) yön 7°
- Planlanan görünürlük oranı: [0.683, 0.717, 0.817, 1.0]
- Arama barları (°): R1 [0.7, 6.7, 12.7, 18.7, 24.7, 30.7, 36.7]; R2 [3.2, 9.2, 15.2, 21.2, 27.2, 33.2, 39.2, 45.2]
- Yanlış alarm raporu: 16

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 765 / 801 / 821 | 340 / 9505 / 10891 | -642.7 / -472.0 / 0.0 | 0.00 / 9.35 / 12.48 | 0.0 / 0.0 / 13.8 | -4.6 / 1.9 / 5.7 | 12 {0: 10, 1: 2} | 20% | 64% | 10.0 / 11.0 | 12.7 / 27.1 / 53.6 | 384 / 702 / 777 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 716 / 748 / 872 | 674 / 9028 / 9057 | -731.5 / 0.0 / 0.0 | 0.00 / 0.00 / 25.93 | 0.0 / 0.0 / 14.0 | -1.5 / 0.2 / 4.2 | 8 {1: 8} | 14% | 31% | 9.0 / 10.0 | 16.2 / 26.9 / 32.7 | 331 / 608 / 688 | tamam |
| T2 | missile/subsonic_cruise (L2) | terrain_follow+weave | 238 / 239 / 239 | 50 / 50 / 50 | -0.0 / 0.0 / 0.0 | 0.00 / 0.00 / 3.08 | 0.0 / 0.0 / 7.3 | -3.8 / -0.9 / 3.1 | 1 {1: 1} | 2% | 0% | 29.5 / — | 26.9 / 26.9 / 26.9 | 2 / 2 / 2 | tamam |
| T3 | aircraft/transport (L1) | CA | 116 / 135 / 154 | 2434 / 2548 / 2663 | 7.8 / 7.8 / 7.8 | 0.01 / 0.01 / 0.01 | 0.0 / 0.0 / 0.0 | 15.7 / 16.8 / 18.4 | 12 {0: 3, 1: 9} | 20% | 73% | 7.0 / 8.5 | 32.8 / 60.2 / 75.8 | 17 / 94 / 114 | tamam |

## ep-000009-7e6778038113 (train)
- Arazi: valley_linear_canyon / asymmetric / rocky_mountain
- Arazi: **deep_canyon**, ölçülen kabartı 1936 m
- Salvo: **diverging**, 3 füze, süre 18 s (hedefe eşzamanlı varış)
- Profiller: T0 popup_dive → Zone A (tehdit ekseni -84.9°, giriş -6.0 s); T1 supersonic_skim → Zone B (tehdit ekseni -109.1°, giriş +0.0 s); T2 break_turn_dive → Zone C (tehdit ekseni -93.3°, giriş -4.0 s)
- Bölgeler: Zone A [zone] (266, 840); Zone B [zone] (-3584, 1420); Zone C [zone] (-614, 3939); Facility 1 [facility] (-3455, 2916) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 300, Zone B 3860, Zone C 3476; T1 Zone A 3668, Zone B 1033, Zone C 4488; T2 Zone A 310, Zone B 3362, Zone C 517
- Radar: 1 — R1 (-1311, 6316) yön -105°
- Planlanan görünürlük oranı: [0.946, 0.568, 0.973]
- Arama barları (°): R1 [-2.7, 3.3, 9.3, 15.3, 21.3, 27.3, 33.3, 39.3]
- Yanlış alarm raporu: 2

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 511 / 540 / 640 | 70 / 392 / 1791 | -384.1 / -28.7 / 253.1 | 0.00 / 2.21 / 26.27 | 0.0 / 0.0 / 0.0 | -3.5 / -2.7 / 1.8 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/supersonic (L2) | terrain_follow+dogleg_turn+terminal_weave | 515 / 560 / 573 | 55 / 256 / 432 | -49.0 / -37.7 / 49.0 | 0.01 / 2.54 / 10.90 | 0.0 / 0.0 / 11.1 | -6.1 / -5.0 / -1.5 | 5 {0: 5} | 14% | 50% | 3.0 / 4.5 | 7.7 / 29.6 / 33.4 | 518 / 547 / 552 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 690 / 865 / 888 | 499 / 7269 / 9550 | -720.7 / -703.2 / 0.0 | 0.00 / 2.99 / 26.22 | 0.0 / 2.6 / 16.9 | -8.3 / -3.4 / 3.3 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000010-03465c812972 (train)
- Arazi: basin_highlands / dense / mixed_rock_vegetation
- Arazi: **highland_bowl**, ölçülen kabartı 1425 m
- Salvo: **single**, 1 füze, süre 19 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni -49.7°, giriş +0.0 s); T1 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (486, 965); Facility 1 [facility] (5187, -2599); Facility 2 [facility] (-3193, 4881) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 315
- Radar: 3 — R1 (1236, -1535) yön 89°, R2 (4736, 4715) yön -101°, R3 (-1514, -3035) yön 65°
- Planlanan görünürlük oranı: [0.974, 1.0]
- Arama barları (°): R1 [11.5, 17.5, 23.5, 29.5, 35.5, 41.5, 47.5, 53.5]; R2 [1.6, 7.6, 13.6, 19.6, 25.6, 31.6, 37.6, 43.6]; R3 [10.2, 16.2, 22.2, 28.2, 34.2, 40.2, 46.2, 52.2]
- Yanlış alarm raporu: 25

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 910 / 994 / 994 | 208 / 6780 / 9366 | -816.7 / -816.7 / 0.0 | 0.00 / 0.00 / 19.40 | 0.0 / 0.0 / 0.0 | -6.2 / 0.0 / 4.4 | 1 {2: 1} | 3% | 0% | 18.0 / — | 6.8 / 6.8 / 6.8 | 52 / 52 / 52 | tamam |
| T1 | aircraft/fighter (L1) | CA+CT+CA | 217 / 272 / 298 | 1219 / 1555 / 1683 | 17.5 / 17.5 / 17.5 | 0.04 / 0.04 / 5.87 | 0.0 / 0.0 / 11.1 | 3.3 / 6.0 / 10.8 | 12 {0: 4, 1: 1, 2: 7} | 29% | 95% | 1.0 / 1.5 | 19.3 / 39.7 / 61.5 | 153 / 231 / 287 | tamam |

## ep-000011-6715d240864a (test)
- Arazi: ridge_multi_obstacle / asymmetric / snowy_highlands
- Arazi: **knife_ridges**, ölçülen kabartı 1074 m
- Salvo: **converging**, 2 füze, süre 24 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 120.6°, giriş +0.0 s); T1 steep_dive → Zone A (tehdit ekseni -134.4°, giriş +2.5 s)
- Bölgeler: Zone A [zone] (-455, 1293); Facility 1 [facility] (2193, 3073) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 95; T1 Zone A 81
- Radar: 2 — R1 (-4705, 2543) yön -81°, R2 (-7205, 1543) yön 71°
- Planlanan görünürlük oranı: [0.755, 0.837]
- Arama barları (°): R1 [16.5, 22.5, 28.5, 34.5, 40.5, 46.5, 52.5, 58.5]; R2 [17.0, 23.0, 29.0, 35.0, 41.0, 47.0, 53.0, 59.0]
- Yanlış alarm raporu: 19

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 777 / 848 / 848 | 301 / 6474 / 9531 | -573.2 / -573.2 / 0.0 | 0.00 / 10.68 / 15.91 | 0.0 / 8.0 / 12.3 | -5.0 / 0.9 / 4.6 | 6 {1: 6} | 12% | 56% | 9.0 / 11.0 | 13.6 / 28.6 / 45.4 | 313 / 641 / 731 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 880 / 885 / 886 | 299 / 9730 / 11384 | -796.5 / -710.6 / 0.0 | 0.00 / 0.00 / 16.78 | 0.0 / 0.0 / 0.0 | -1.9 / 3.7 / 5.6 | 4 {0: 4} | 8% | 38% | 13.5 / 15.5 | 13.5 / 22.8 / 40.5 | 489 / 727 / 775 | tamam |

## ep-000012-6cf3f70f9bcf (train)
- Arazi: valley_linear_canyon / sparse / grassland_valley
- Arazi: **deep_canyon**, ölçülen kabartı 1719 m
- Salvo: **diverging**, 2 füze, süre 29 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 127.5°, giriş +1.0 s); T1 cruise_weave → Zone B (tehdit ekseni 121.7°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-324, -909); Zone B [zone] (-476, 2304); Facility 1 [facility] (3884, -2834); Facility 2 [facility] (3211, 2890) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 90, Zone B 1926; T1 Zone A 3710, Zone B 497
- Radar: 3 — R1 (1350, -2552) yön 113°, R2 (3100, -4802) yön 119°, R3 (-4150, 2948) yön 111°
- Planlanan görünürlük oranı: [1.0, 0.407]
- Arama barları (°): R1 [3.6, 9.6, 15.6, 21.6, 27.6]; R2 [3.5, 9.5, 15.5, 21.5, 27.5]; R3 [3.9, 9.9, 15.9, 21.9, 27.9, 33.9, 39.9, 45.9]
- Yanlış alarm raporu: 30

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 812 / 831 / 831 | 266 / 4862 / 9082 | -682.1 / -232.5 / 0.0 | 0.00 / 0.00 / 16.27 | 0.0 / 0.0 / 0.0 | -9.5 / -4.3 / 2.5 | 10 {0: 1, 1: 5, 2: 4} | 17% | 47% | 14.5 / 15.5 | 14.4 / 30.4 / 52.3 | 659 / 814 / 830 | tamam |
| T1 | missile/subsonic_cruise (L2) | terrain_follow+weave+pop_up | 245 / 246 / 246 | 78 / 97 / 148 | -11.5 / -4.5 / 18.0 | 0.00 / 0.10 / 3.17 | 0.0 / 0.0 / 5.5 | -10.0 / -8.3 / 3.1 | 6 {0: 3, 2: 3} | 10% | 43% | 2.0 / 3.0 | 17.0 / 36.1 / 48.3 | 203 / 223 / 245 | tamam |

## ep-000013-09f6cc90fa6e (train)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 458 m
- Salvo: **converging**, 3 füze, süre 19.5 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni 115.5°, giriş +5.5 s); T1 popup_dive → Zone A (tehdit ekseni 60.9°, giriş -0.5 s); T2 supersonic_skim → Zone A (tehdit ekseni 7.9°, giriş +0.0 s); T3 traffic/multirotor → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (37, -1409); Facility 1 [facility] (-2826, 700); Facility 2 [facility] (1985, 776) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 647; T1 Zone A 298; T2 Zone A 1345
- Radar: 2 — R1 (-713, 2091) yön -70°, R2 (5037, 841) yön 137°
- Planlanan görünürlük oranı: [0.975, 0.475, 0.375, 0.75]
- Arama barları (°): R1 [1.9, 7.9, 13.9, 19.9, 25.9, 31.9, 37.9, 43.9]; R2 [-3.7, 2.3, 8.3, 14.3, 20.3, 26.3, 32.3]
- Yanlış alarm raporu: 16

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 851 / 883 / 987 | 477 / 5568 / 5617 | -781.9 / 0.0 / 0.0 | 0.00 / 0.00 / 26.18 | 0.0 / 0.0 / 16.5 | -4.4 / -0.5 / 5.2 | 1 {1: 1} | 3% | 0% | 10.0 / — | 23.9 / 23.9 / 23.9 | 610 / 610 / 610 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 537 / 618 / 706 | 83 / 287 / 1961 | -673.9 / 19.3 / 322.3 | 0.01 / 0.73 / 26.26 | 0.0 / 0.0 / 0.0 | -7.0 / 0.0 / 3.2 | 1 {0: 1} | 3% | 0% | 15.0 / — | 22.0 / 22.0 / 22.0 | 231 / 231 / 231 | tamam |
| T2 | missile/supersonic (L2) | terrain_follow+terminal_weave | 543 / 543 / 544 | 100 / 118 / 146 | -18.9 / 4.3 / 21.0 | 0.02 / 0.71 / 7.23 | 0.0 / 0.0 / 7.5 | -6.0 / -1.5 / 4.7 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T3 | uav/multirotor (L0) | CA+hover+CT | 5 / 15 / 15 | 120 / 131 / 143 | -1.2 / -1.2 / -1.2 | 0.00 / 0.36 / 0.36 | 0.0 / 13.5 / 13.5 | -18.1 / -14.5 / -13.5 | 7 {0: 7} | 18% | 90% | 1.5 / 2.5 | 12.8 / 27.1 / 31.8 | 2 / 10 / 15 | tamam |

## ep-000014-c5df19aa807d (train)
- Arazi: mountain_chokepoint / dense / snowy_highlands
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1550 m
- Salvo: **single**, 1 füze, süre 15 s (hedefe eşzamanlı varış)
- Profiller: T0 barrel_roll_dive → Zone A (tehdit ekseni -64.6°, giriş +0.0 s); T1 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (995, -559) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 341
- Radar: 1 — R1 (-2005, 1941) yön -57°
- Planlanan görünürlük oranı: [0.968, 1.0]
- Arama barları (°): R1 [8.9, 14.9, 20.9, 26.9]
- Yanlış alarm raporu: 4

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 857 / 877 / 998 | 431 / 4839 / 4880 | -801.5 / -119.7 / 0.0 | 0.00 / 10.31 / 26.02 | 0.0 / 1.9 / 15.9 | -5.7 / -3.3 / 0.6 | 6 {0: 6} | 20% | 50% | 7.0 / 8.0 | 19.3 / 22.4 / 34.8 | 753 / 811 / 931 | tamam |
| T1 | aircraft/fighter (L1) | CV+CA+CT | 255 / 263 / 263 | 1359 / 2623 / 2757 | 3.0 / 3.0 / 3.0 | 0.00 / 5.93 / 5.93 | 0.0 / 12.7 / 12.7 | 10.2 / 11.7 / 14.2 | 8 {0: 8} | 27% | 70% | 4.0 / 5.0 | 14.5 / 46.3 / 53.2 | 38 / 169 / 232 | tamam |

## ep-000015-309965279b87 (validation)
- Arazi: ridge_multi_obstacle / asymmetric / green_forest_valley
- Arazi: **knife_ridges**, ölçülen kabartı 1434 m
- Salvo: **single**, 1 füze, süre 17 s (hedefe eşzamanlı varış)
- Profiller: T0 dogleg_serpentine → Zone A (tehdit ekseni -12.9°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (209, 622) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 1100
- Radar: 3 — R1 (-1041, -5128) yön 50°, R2 (-3041, 1872) yön 3°, R3 (1959, -5878) yön 59°
- Planlanan görünürlük oranı: [0.6]
- Arama barları (°): R1 [7.0, 13.0]; R2 [-6.3, -0.3]; R3 [-2.8, 3.2]
- Yanlış alarm raporu: 20

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 517 / 614 / 689 | 63 / 196 / 1213 | -402.1 / 53.4 / 88.3 | 0.11 / 3.86 / 26.24 | 0.0 / 0.2 / 16.7 | -4.1 / 1.3 / 6.3 | 9 {0: 3, 1: 1, 2: 5} | 24% | 44% | 8.0 / 9.0 | 17.0 / 29.0 / 49.7 | 92 / 287 / 506 | tamam |

## ep-000016-ad3738fd6a38 (train)
- Arazi: valley_sinuous_pass / asymmetric / mixed_rock_vegetation
- Arazi: **twisting_gorge**, ölçülen kabartı 2374 m
- Salvo: **converging**, 2 füze, süre 23 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni 153.8°, giriş +0.0 s); T1 shallow_dive → Zone A (tehdit ekseni 80.0°, giriş +1.0 s)
- Bölgeler: Zone A [zone] (-969, -847); Facility 1 [facility] (5737, -260) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 511; T1 Zone A 242
- Radar: 1 — R1 (1031, -5347) yön 97°
- Planlanan görünürlük oranı: [0.872, 0.809]
- Arama barları (°): R1 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 698 / 762 / 819 | 183 / 5989 / 8622 | -681.0 / -571.0 / 0.0 | 0.00 / 1.99 / 26.21 | 0.0 / 1.0 / 16.6 | -6.7 / -3.4 / 2.6 | 3 {0: 3} | 7% | 17% | 18.5 / 19.5 | 26.2 / 34.6 / 40.8 | 352 / 415 / 452 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 624 / 712 / 712 | 266 / 4164 / 5197 | -397.9 / -397.9 / 0.0 | 0.00 / 0.00 / 14.16 | 0.0 / 0.0 / 0.0 | -7.5 / -3.8 / 2.8 | 10 {0: 10} | 22% | 72% | 5.5 / 6.5 | 11.8 / 27.3 / 32.9 | 559 / 600 / 658 | tamam |

## ep-000017-a08adfe3ce4b (test)
- Arazi: basin_highlands / sparse / snowy_highlands
- Arazi: **highland_bowl**, ölçülen kabartı 1410 m
- Salvo: **diverging**, 3 füze, süre 19 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 55.5°, giriş -9.0 s); T1 popup_dive → Zone B (tehdit ekseni 56.8°, giriş -1.5 s); T2 dogleg_serpentine → Zone C (tehdit ekseni 20.7°, giriş +0.0 s); T3 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-158, 342); Zone B [zone] (-3123, 4285); Zone C [zone] (-4713, 1474) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 262, Zone B 4903, Zone C 4951; T1 Zone A 4745, Zone B 252, Zone C 3314; T2 Zone A 3882, Zone B 974, Zone C 616
- Radar: 2 — R1 (2585, 5784) yön -145°, R2 (-1915, -966) yön 69°
- Planlanan görünürlük oranı: [0.974, 0.436, 0.692, 1.0]
- Arama barları (°): R1 [4.4, 10.4, 16.4, 22.4, 28.4, 34.4, 40.4, 46.4]; R2 [3.5, 9.5, 15.5, 21.5, 27.5, 33.5]
- Yanlış alarm raporu: 13

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 771 / 776 / 776 | 80 / 1123 / 6629 | -587.4 / -489.3 / -0.0 | 0.00 / 0.00 / 13.56 | 0.0 / 0.0 / 0.0 | -3.1 / -1.9 / 3.3 | 6 {1: 6} | 16% | 50% | 7.0 / 9.5 | 22.9 / 30.4 / 52.4 | 596 / 700 / 765 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 573 / 631 / 730 | 101 / 430 / 1947 | -552.1 / 68.0 / 352.6 | 0.02 / 0.64 / 26.24 | 0.0 / 0.0 / 0.0 | -2.5 / 1.6 / 4.8 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T2 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 482 / 539 / 614 | 48 / 267 / 802 | -230.7 / 50.7 / 258.9 | 0.10 / 5.81 / 26.27 | 0.0 / 0.0 / 16.8 | -2.4 / 3.5 / 5.9 | 9 {0: 1, 1: 8} | 24% | 71% | 1.5 / 2.5 | 19.6 / 43.2 / 50.8 | 195 / 259 / 450 | tamam |
| T3 | aircraft/fighter (L1) | CT | 218 / 218 / 218 | 1059 / 2215 / 2439 | 27.4 / 27.4 / 27.4 | 3.73 / 3.73 / 3.73 | 9.7 / 9.7 / 9.7 | 5.1 / 12.6 / 17.2 | 10 {0: 6, 1: 4} | 26% | 58% | 7.0 / 8.5 | 12.3 / 35.8 / 58.7 | 62 / 133 / 205 | tamam |

## ep-000018-f86055dfd214 (train)
- Arazi: mountain_chokepoint / asymmetric / rocky_mountain
- Arazi: **alpine_chokepoint**, ölçülen kabartı 2164 m
- Salvo: **diverging**, 2 füze, süre 19 s (hedefe eşzamanlı varış)
- Profiller: T0 supersonic_skim → Zone A (tehdit ekseni 68.8°, giriş +0.0 s); T1 cruise_weave → Zone B (tehdit ekseni 90.3°, giriş -6.5 s)
- Bölgeler: Zone A [zone] (-786, -689); Zone B [zone] (-1974, 3448); Facility 1 [facility] (2462, -1805); Facility 2 [facility] (3038, -4222) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 1274, Zone B 2560; T1 Zone A 4880, Zone B 578
- Radar: 2 — R1 (-3130, 630) yön 38°, R2 (3370, 4380) yön 164°
- Planlanan görünürlük oranı: [0.385, 0.949]
- Arama barları (°): R1 [-11.7, -5.7, 0.3]; R2 [3.3, 9.3]
- Yanlış alarm raporu: 11

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/supersonic (L2) | terrain_follow+terminal_weave | 576 / 577 / 578 | 50 / 50 / 62 | -3.0 / 0.0 / 8.8 | 0.00 / 0.00 / 9.90 | 0.0 / 0.0 / 9.6 | -7.7 / -3.6 / 5.9 | 7 {1: 7} | 18% | 29% | 10.0 / 14.0 | 7.9 / 44.4 / 65.7 | 66 / 299 / 443 | tamam |
| T1 | missile/subsonic_cruise (L2) | terrain_follow+weave | 257 / 258 / 258 | 76 / 78 / 78 | 5.8 / 6.9 / 7.1 | 0.00 / 0.08 / 3.06 | 0.0 / 0.1 / 6.7 | -7.4 / -1.1 / 4.2 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000019-159cf38cb5a4 (train)
- Arazi: ridge_multi_obstacle / asymmetric / grassland_valley
- Arazi: **knife_ridges**, ölçülen kabartı 1020 m
- Salvo: **converging**, 3 füze, süre 20.5 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 12.9°, giriş +0.0 s); T1 shallow_dive → Zone A (tehdit ekseni 72.3°, giriş +0.0 s); T2 random_jink_dive → Zone A (tehdit ekseni 127.8°, giriş +6.0 s); T3 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (497, -174); Facility 1 [facility] (3585, 610); Facility 2 [facility] (199, 1358) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 202; T1 Zone A 274; T2 Zone A 332
- Radar: 1 — R1 (-3, -4174) yön 74°
- Planlanan görünürlük oranı: [0.929, 0.833, 0.857, 0.881]
- Arama barları (°): R1 [8.9, 14.9, 20.9, 26.9, 32.9, 38.9, 44.9, 50.9]
- Yanlış alarm raporu: 6

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 829 / 840 / 842 | 320 / 7379 / 9966 | -785.9 / -727.8 / 0.0 | 0.00 / 0.00 / 17.16 | 0.0 / 0.0 / 0.0 | -4.0 / 0.8 / 3.8 | 4 {0: 4} | 10% | 22% | 15.5 / 16.5 | 17.5 / 45.2 / 55.9 | 331 / 589 / 630 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 748 / 763 / 763 | 215 / 4177 / 5803 | -412.4 / -412.4 / 0.0 | 0.00 / 0.00 / 14.74 | 0.0 / 0.0 / 0.0 | -3.9 / -2.0 / 0.0 | 7 {0: 7} | 17% | 71% | 5.5 / 6.5 | 16.3 / 31.4 / 45.2 | 694 / 737 / 751 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+random_jinks | 838 / 870 / 980 | 478 / 4031 / 4170 | -864.1 / 0.0 / 0.0 | 0.00 / 0.00 / 26.19 | 0.0 / 0.0 / 15.3 | -3.6 / -2.6 / 2.4 | 0 {} | 0% | 2% | — / 20.5 | — | — | tamam |
| T3 | uav/fixed_wing (L0) | CV+CT+CV | 22 / 22 / 22 | 633 / 646 / 649 | 0.2 / 0.2 / 0.2 | 0.00 / 0.00 / 0.77 | 0.0 / 0.0 / 19.1 | 0.4 / 0.8 / 2.7 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000020-78fdd371105d (train)
- Arazi: valley_sinuous_pass / dense / green_forest_valley
- Arazi: **twisting_gorge**, ölçülen kabartı 1233 m
- Salvo: **single**, 1 füze, süre 21.5 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 147.3°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (479, -982); Facility 1 [facility] (1796, 4724); Facility 2 [facility] (-2464, -1403) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 73
- Radar: 3 — R1 (-1271, 2768) yön -168°, R2 (-771, -2982) yön 125°, R3 (2729, -2982) yön 141°
- Planlanan görünürlük oranı: [0.955]
- Arama barları (°): R1 [20.0, 26.0, 32.0, 38.0, 44.0, 50.0, 56.0, 62.0]; R2 [5.0, 11.0, 17.0, 23.0, 29.0, 35.0, 41.0, 47.0]; R3 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5, 37.5, 43.5]
- Yanlış alarm raporu: 25

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 889 / 983 / 998 | 249 / 8701 / 10566 | -801.2 / -731.7 / 0.0 | 0.00 / 10.12 / 17.99 | 0.0 / 0.0 / 13.7 | -4.6 / -0.5 / 3.1 | 10 {0: 2, 1: 6, 2: 2} | 21% | 44% | 11.5 / 12.5 | 12.8 / 24.8 / 55.2 | 628 / 916 / 970 | tamam |

## ep-000021-6c92219964d3 (train)
- Arazi: valley_linear_canyon / sparse / rocky_mountain
- Arazi: **deep_canyon**, ölçülen kabartı 1336 m
- Salvo: **converging**, 2 füze, süre 17 s (hedefe eşzamanlı varış)
- Profiller: T0 barrel_roll_dive → Zone A (tehdit ekseni -80.6°, giriş +3.0 s); T1 break_turn_dive → Zone A (tehdit ekseni 141.5°, giriş +0.0 s); T2 traffic/multirotor → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-794, 412); Facility 1 [facility] (-4277, -4072) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 244; T1 Zone A 618
- Radar: 1 — R1 (3456, 412) yön -160°
- Planlanan görünürlük oranı: [0.543, 0.686, 1.0]
- Arama barları (°): R1 [-7.0, -1.0, 5.0, 11.0, 17.0, 23.0]
- Yanlış alarm raporu: 1

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 826 / 840 / 958 | 433 / 3107 / 3972 | -718.4 / 0.0 / 0.0 | 0.00 / 0.00 / 25.83 | 0.0 / 0.0 / 9.8 | -1.8 / 1.3 / 5.1 | 3 {0: 3} | 9% | 15% | 10.5 / 11.5 | 24.1 / 31.7 / 48.7 | 542 / 562 / 578 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 809 / 898 / 1004 | 463 / 4970 / 5543 | -788.9 / -56.8 / 0.0 | 0.00 / 12.43 / 26.19 | 0.0 / 7.9 / 16.5 | -1.9 / -0.4 / 5.4 | 3 {0: 3} | 9% | 12% | 8.5 / 9.5 | 12.7 / 31.0 / 36.9 | 611 / 646 / 680 | tamam |
| T2 | uav/multirotor (L0) | CA+CA+hover | 1 / 5 / 20 | 122 / 134 / 147 | 1.5 / 1.5 / 1.5 | 0.00 / 0.03 / 0.29 | 0.0 / 0.0 / 0.0 | -15.8 / -12.9 / -9.7 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000022-4c7bbc321222 (validation)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 628 m
- Salvo: **single**, 1 füze, süre 14 s (hedefe eşzamanlı varış)
- Profiller: T0 barrel_roll_dive → Zone A (tehdit ekseni -19.4°, giriş +0.0 s); T1 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (906, -1189); Facility 1 [facility] (1525, 692) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 375
- Radar: 2 — R1 (-594, 2061) yön -31°, R2 (2656, 4811) yön -145°
- Planlanan görünürlük oranı: [0.966, 1.0]
- Arama barları (°): R1 [3.6, 9.6, 15.6, 21.6, 27.6, 33.6]; R2 [6.3, 12.3, 18.3, 24.3, 30.3]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 862 / 905 / 1006 | 358 / 5181 / 5645 | -786.0 / -430.8 / 0.0 | 0.00 / 11.57 / 26.13 | 0.0 / 3.0 / 10.4 | -3.8 / 1.2 / 6.1 | 6 {0: 6} | 21% | 32% | 9.0 / 10.0 | 24.8 / 39.2 / 46.7 | 514 / 870 / 896 | tamam |
| T1 | aircraft/transport (L1) | CT+CA | 99 / 110 / 110 | 1902 / 1908 / 1913 | 5.7 / 5.7 / 5.7 | 0.01 / 0.51 / 0.51 | 0.0 / 2.6 / 2.6 | 17.5 / 18.8 / 19.6 | 7 {1: 7} | 25% | 75% | 3.0 / 4.0 | 36.7 / 59.0 / 66.4 | 59 / 62 / 71 | tamam |

## ep-000023-b82d72e9333e (train)
- Arazi: basin_highlands / dense / grassland_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1427 m
- Salvo: **diverging**, 3 füze, süre 27 s (hedefe eşzamanlı varış)
- Profiller: T0 cruise_weave → Zone A (tehdit ekseni -164.6°, giriş +0.0 s); T1 shallow_dive → Zone B (tehdit ekseni -145.3°, giriş +4.5 s); T2 jink_dive → Zone C (tehdit ekseni -159.2°, giriş +5.0 s)
- Bölgeler: Zone A [zone] (828, 638); Zone B [zone] (4758, 350); Zone C [zone] (4086, -2604); Facility 1 [facility] (-4452, -3865); Facility 2 [facility] (-78, 3785) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 854, Zone B 4730, Zone C 5012; T1 Zone A 2633, Zone B 152, Zone C 1893; T2 Zone A 3895, Zone B 2791, Zone C 267
- Radar: 3 — R1 (-3026, -2039) yön -108°, R2 (8974, -2039) yön -177°, R3 (2474, 961) yön -156°
- Planlanan görünürlük oranı: [0.909, 1.0, 1.0]
- Arama barları (°): R1 [25.1, 31.1, 37.1, 43.1, 49.1, 55.1, 61.1, 67.1]; R2 [-2.5, 3.5, 9.5, 15.5, 21.5, 27.5, 33.5]; R3 [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, 39.0, 45.0]
- Yanlış alarm raporu: 25

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | terrain_follow+weave | 253 / 254 / 254 | 50 / 55 / 77 | -11.5 / 1.0 / 15.6 | 0.00 / 0.10 / 3.11 | 0.0 / 0.0 / 6.9 | -17.1 / -11.7 / 1.9 | 6 {2: 6} | 11% | 31% | 18.0 / 19.0 | 21.9 / 35.7 / 43.9 | 253 / 253 / 254 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 707 / 723 / 723 | 312 / 4558 / 6415 | -320.6 / -320.6 / 0.0 | 0.00 / 0.00 / 11.25 | 0.0 / 0.0 / 0.0 | -3.9 / 0.0 / 4.9 | 16 {0: 5, 1: 6, 2: 5} | 26% | 74% | 6.5 / 7.5 | 13.3 / 31.5 / 45.0 | 430 / 598 / 699 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+push_over+jinks | 901 / 904 / 904 | 259 / 9184 / 11181 | -673.0 / -478.5 / 0.0 | 0.00 / 9.07 / 11.63 | 0.0 / 0.0 / 10.8 | -6.3 / 0.4 / 5.5 | 1 {1: 1} | 2% | 0% | 13.5 / — | 26.1 / 26.1 / 26.1 | 895 / 895 / 895 | tamam |

## ep-000024-63a486045fcc (test)
- Arazi: basin_highlands / dense / green_forest_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1486 m
- Salvo: **single**, 1 füze, süre 27.5 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 128.4°, giriş +0.0 s); T1 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (624, -304); Facility 1 [facility] (79, 4738) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 265
- Radar: 3 — R1 (-5376, -3554) yön 15°, R2 (3874, 2446) yön -126°, R3 (3374, 4446) yön 162°
- Planlanan görünürlük oranı: [0.804, 0.482]
- Arama barları (°): R1 [-6.0, -0.0, 6.0, 12.0, 18.0, 24.0]; R2 [-10.8, -4.8, 1.2, 7.2, 13.2, 19.2, 25.2]; R3 [-6.9, -0.9, 5.1, 11.1, 17.1, 23.1, 29.1]
- Yanlış alarm raporu: 30

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 780 / 791 / 791 | 106 / 3907 / 8155 | -607.5 / -270.5 / 0.0 | 0.00 / 0.00 / 14.09 | 0.0 / 0.0 / 0.0 | -3.6 / 0.8 / 4.0 | 6 {2: 6} | 11% | 38% | 10.0 / 11.0 | 15.0 / 37.7 / 48.1 | 443 / 629 / 671 | tamam |
| T1 | uav/fixed_wing (L0) | CV | 27 / 27 / 27 | 420 / 430 / 440 | 0.8 / 0.8 / 0.8 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -8.9 / 0.3 / 0.9 | 5 {0: 1, 1: 4} | 9% | 44% | 1.0 / 2.0 | 14.5 / 40.6 / 53.7 | 17 / 17 / 26 | tamam |

## ep-000025-73adee0e3f57 (train)
- Arazi: ridge_multi_obstacle / dense / dry_canyon
- Arazi: **knife_ridges**, ölçülen kabartı 986 m
- Salvo: **diverging**, 3 füze, süre 19 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 22.9°, giriş +0.5 s); T1 random_jink_dive → Zone B (tehdit ekseni 37.9°, giriş +0.0 s); T2 supersonic_skim → Zone C (tehdit ekseni 13.7°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-664, 1264); Zone B [zone] (-4699, 1992); Zone C [zone] (-2482, -1260); Facility 1 [facility] (-4597, -756) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 82, Zone B 4143, Zone C 3065; T1 Zone A 3205, Zone B 547, Zone C 3895; T2 Zone A 465, Zone B 3826, Zone C 1466
- Radar: 1 — R1 (-865, -3585) yön 87°
- Planlanan görünürlük oranı: [0.821, 1.0, 0.615]
- Arama barları (°): R1 [-2.2, 3.8, 9.8, 15.8, 21.8, 27.8, 33.8, 39.8]
- Yanlış alarm raporu: 8

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 874 / 915 / 936 | 176 / 7713 / 9122 | -831.5 / -609.7 / 0.0 | 0.00 / 0.00 / 17.30 | 0.0 / 0.0 / 0.0 | -4.8 / 0.9 / 6.1 | 4 {0: 4} | 11% | 21% | 14.5 / 15.5 | 18.4 / 48.9 / 56.1 | 267 / 600 / 656 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+random_jinks | 689 / 717 / 823 | 339 / 5677 / 5863 | -682.0 / -182.1 / 0.0 | 0.00 / 4.95 / 26.22 | 0.0 / 0.8 / 17.0 | -0.1 / 1.1 / 4.9 | 6 {0: 6} | 16% | 39% | 11.0 / 12.0 | 16.5 / 33.0 / 47.2 | 196 / 459 / 523 | tamam |
| T2 | missile/supersonic (L2) | terrain_follow+dogleg_turn+terminal_weave | 510 / 554 / 556 | 89 / 229 / 439 | -49.0 / -4.5 / 48.2 | 0.01 / 4.35 / 10.77 | 0.0 / 1.2 / 11.0 | 0.2 / 2.5 / 5.8 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000026-948266231c23 (train)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 842 m
- Salvo: **diverging**, 2 füze, süre 23 s (hedefe eşzamanlı varış)
- Profiller: T0 popup_dive → Zone A (tehdit ekseni 21.0°, giriş +0.0 s); T1 break_turn_dive → Zone B (tehdit ekseni 24.7°, giriş +0.5 s); T2 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-863, -1071); Zone B [zone] (-2634, 1397); Facility 1 [facility] (2853, 1810) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 67, Zone B 3079; T1 Zone A 3104, Zone B 118
- Radar: 2 — R1 (-748, 6663) yön -49°, R2 (-2498, -6087) yön 46°
- Planlanan görünürlük oranı: [0.851, 0.979, 1.0]
- Arama barları (°): R1 [3.2, 9.2, 15.2, 21.2, 27.2, 33.2, 39.2, 45.2]; R2 [2.1, 8.1, 14.1, 20.1, 26.1, 32.1, 38.1]
- Yanlış alarm raporu: 15

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 493 / 534 / 634 | 66 / 322 / 2034 | -446.0 / 36.2 / 272.6 | 0.00 / 0.69 / 26.28 | 0.0 / 0.0 / 0.0 | -6.5 / 1.7 / 4.8 | 6 {0: 3, 1: 3} | 13% | 26% | 16.5 / 17.5 | 14.4 / 39.2 / 49.4 | 13 / 190 / 440 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 646 / 721 / 830 | 244 / 8073 / 8780 | -701.7 / -503.4 / 0.0 | 0.00 / 0.90 / 26.23 | 0.0 / 0.7 / 16.7 | -1.5 / 3.0 / 5.2 | 7 {0: 1, 1: 6} | 15% | 48% | 11.5 / 12.5 | 16.1 / 30.6 / 44.4 | 388 / 576 / 643 | tamam |
| T2 | aircraft/transport (L1) | CV | 157 / 157 / 157 | 1976 / 1989 / 2002 | 1.9 / 1.9 / 1.9 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 13.5 / 16.2 / 18.6 | 5 {0: 2, 1: 3} | 11% | 22% | 17.5 / 18.5 | 35.7 / 60.1 / 69.1 | 122 / 123 / 140 | tamam |

## ep-000027-7115b79c999c (train)
- Arazi: valley_linear_canyon / asymmetric / rocky_mountain
- Arazi: **deep_canyon**, ölçülen kabartı 3038 m
- Salvo: **converging**, 3 füze, süre 22.5 s (hedefe eşzamanlı varış)
- Profiller: T0 dogleg_serpentine → Zone A (tehdit ekseni 179.1°, giriş +0.0 s); T1 shallow_dive → Zone A (tehdit ekseni 42.4°, giriş -4.5 s); T2 barrel_roll_dive → Zone A (tehdit ekseni 121.0°, giriş +4.5 s)
- Bölgeler: Zone A [zone] (-1000, -761); Facility 1 [facility] (3604, 288) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 270; T1 Zone A 301; T2 Zone A 230
- Radar: 2 — R1 (-7750, -2011) yön 60°, R2 (500, -3011) yön 114°
- Planlanan görünürlük oranı: [0.674, 0.674, 0.978]
- Arama barları (°): R1 [4.5, 10.5, 16.5, 22.5, 28.5, 34.5]; R2 [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0]
- Yanlış alarm raporu: 14

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 486 / 536 / 590 | 42 / 245 / 682 | -322.9 / 24.2 / 254.6 | 0.08 / 4.31 / 26.26 | 0.0 / 0.1 / 16.8 | -3.9 / 0.5 / 4.8 | 8 {0: 7, 1: 1} | 18% | 38% | 11.0 / 12.5 | 12.9 / 40.6 / 55.0 | 80 / 203 / 393 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 671 / 788 / 788 | 245 / 3171 / 6595 | -345.9 / -345.9 / 0.0 | 0.00 / 0.00 / 11.99 | 0.0 / 0.0 / 0.0 | -6.0 / -3.1 / 5.3 | 0 {} | 0% | 2% | — / 22.5 | — | — | tamam |
| T2 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 744 / 751 / 875 | 199 / 5273 / 5379 | -628.3 / -29.7 / 0.0 | 0.00 / 5.15 / 25.77 | 0.0 / 0.0 / 12.6 | -8.4 / -0.1 / 5.6 | 9 {0: 6, 1: 3} | 18% | 62% | 8.0 / 8.5 | 15.7 / 26.7 / 45.1 | 36 / 481 / 760 | tamam |

## ep-000028-801c04e4ddb7 (validation)
- Arazi: mountain_chokepoint / asymmetric / green_forest_valley
- Arazi: **alpine_chokepoint**, ölçülen kabartı 2230 m
- Salvo: **single**, 1 füze, süre 25 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 158.1°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (437, -269) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 244
- Radar: 3 — R1 (-313, -3519) yön 127°, R2 (687, -7019) yön 120°, R3 (-5063, 1231) yön -5°
- Planlanan görünürlük oranı: [0.882]
- Arama barları (°): R1 [11.0, 17.0, 23.0, 29.0, 35.0, 41.0, 47.0, 53.0]; R2 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5, 37.5, 43.5]; R3 [25.0, 31.0, 37.0, 43.0, 49.0, 55.0, 61.0, 67.0]
- Yanlış alarm raporu: 24

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 764 / 797 / 804 | 181 / 8551 / 10344 | -748.8 / -709.5 / 0.0 | 0.00 / 0.00 / 16.28 | 0.0 / 0.0 / 0.0 | -5.6 / 2.9 / 6.7 | 7 {0: 1, 1: 2, 2: 4} | 14% | 26% | 12.0 / 19.0 | 16.0 / 24.3 / 60.4 | 41 / 236 / 724 | tamam |

## ep-000029-bc0ddc86524f (train)
- Arazi: valley_sinuous_pass / asymmetric / mixed_rock_vegetation
- Arazi: **twisting_gorge**, ölçülen kabartı 2392 m
- Salvo: **converging**, 2 füze, süre 21 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni -175.8°, giriş +0.0 s); T1 jink_dive → Zone A (tehdit ekseni 126.9°, giriş +1.5 s); T2 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (1454, -28); Facility 1 [facility] (1877, 1716) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 168; T1 Zone A 122
- Radar: 1 — R1 (4954, -278) yön 175°
- Planlanan görünürlük oranı: [1.0, 0.907, 0.744]
- Arama barları (°): R1 [-2.3, 3.7, 9.7, 15.7, 21.7, 27.7, 33.7, 39.7]
- Yanlış alarm raporu: 7

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 758 / 859 / 997 | 244 / 7673 / 8180 | -828.8 / -387.6 / 0.0 | 0.00 / 13.29 / 26.20 | 0.0 / 10.5 / 16.9 | -4.3 / -0.7 / 3.7 | 6 {0: 6} | 14% | 33% | 13.0 / 14.5 | 9.5 / 28.4 / 53.2 | 530 / 882 / 919 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+jinks | 919 / 980 / 1024 | 244 / 7082 / 7848 | -729.2 / -467.0 / 0.0 | 0.00 / 9.81 / 20.49 | 0.0 / 0.0 / 10.6 | -4.6 / -0.6 / 5.5 | 5 {0: 5} | 12% | 26% | 15.0 / 16.0 | 20.0 / 39.9 / 55.1 | 311 / 940 / 947 | tamam |
| T2 | aircraft/fighter (L1) | CV+CV | 258 / 258 / 258 | 774 / 1361 / 1411 | 0.2 / 0.2 / 0.2 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 5.5 / 6.5 / 10.1 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000030-15c136150f49 (train)
- Arazi: basin_highlands / sparse / rocky_mountain
- Arazi: **highland_bowl**, ölçülen kabartı 1064 m
- Salvo: **converging**, 2 füze, süre 18.5 s (hedefe eşzamanlı varış)
- Profiller: T0 popup_dive → Zone A (tehdit ekseni -6.5°, giriş -1.0 s); T1 supersonic_skim → Zone A (tehdit ekseni 77.7°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (134, -1009); Facility 1 [facility] (1457, 5748) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 55; T1 Zone A 1232
- Radar: 2 — R1 (134, 4991) yön 9°, R2 (4384, -3509) yön 75°
- Planlanan görünürlük oranı: [0.395, 0.368]
- Arama barları (°): R1 [-3.5, 2.5, 8.5]; R2 [-5.1, 0.9, 6.9, 12.9, 18.9, 24.9]
- Yanlış alarm raporu: 14

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 555 / 597 / 697 | 69 / 232 / 1878 | -428.6 / 22.1 / 315.2 | 0.19 / 1.05 / 26.26 | 0.0 / 0.0 / 0.0 | -6.3 / 1.6 / 3.4 | 1 {1: 1} | 3% | 0% | 11.5 / — | 56.6 / 56.6 / 56.6 | 263 / 263 / 263 | tamam |
| T1 | missile/supersonic (L2) | terrain_follow+terminal_weave | 522 / 571 / 572 | 60 / 99 / 795 | -49.0 / 0.0 / 49.0 | 0.05 / 1.14 / 8.20 | 0.0 / 0.0 / 7.9 | -2.7 / -1.0 / 6.5 | 4 {0: 4} | 11% | 62% | 6.5 / 7.5 | 10.4 / 44.8 / 58.6 | 97 / 411 / 457 | tamam |

## ep-000031-aae8884023cf (test)
- Arazi: ridge_multi_obstacle / dense / green_forest_valley
- Arazi: **knife_ridges**, ölçülen kabartı 1026 m
- Salvo: **single**, 1 füze, süre 18.5 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni 47.0°, giriş +0.0 s); T1 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (128, -372) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 573
- Radar: 3 — R1 (-3872, -372) yön -37°, R2 (-6622, -622) yön 28°, R3 (-4872, 2878) yön 8°
- Planlanan görünürlük oranı: [0.974, 1.0]
- Arama barları (°): R1 [2.7, 8.7, 14.7, 20.7, 26.7, 32.7]; R2 [4.9, 10.9, 16.9, 22.9, 28.9]; R3 [1.3, 7.3, 13.3, 19.3, 25.3, 31.3]
- Yanlış alarm raporu: 21

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 714 / 791 / 850 | 248 / 4687 / 5763 | -613.1 / -505.9 / 0.0 | 0.00 / 5.89 / 25.68 | 0.0 / 0.2 / 16.0 | -4.6 / -0.1 / 6.8 | 9 {1: 1, 2: 8} | 24% | 65% | 6.0 / 7.0 | 19.3 / 35.9 / 52.9 | 303 / 542 / 631 | tamam |
| T1 | aircraft/transport (L1) | CT+CA | 142 / 153 / 156 | 1556 / 1642 / 2274 | -7.7 / -7.7 / -7.7 | 0.01 / 0.01 / 0.48 | 0.0 / 0.0 / 1.7 | 17.0 / 21.2 / 23.4 | 8 {0: 7, 1: 1} | 22% | 73% | 4.5 / 5.5 | 31.2 / 57.2 / 76.3 | 88 / 123 / 127 | tamam |

## ep-000032-32f5a38cf98e (train)
- Arazi: mountain_chokepoint / sparse / mixed_rock_vegetation
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1327 m
- Salvo: **converging**, 3 füze, süre 24 s (hedefe eşzamanlı varış)
- Profiller: T0 dogleg_serpentine → Zone A (tehdit ekseni 135.3°, giriş +0.0 s); T1 dive_pullout → Zone A (tehdit ekseni -149.9°, giriş -1.0 s); T2 cruise_weave → Zone A (tehdit ekseni 31.1°, giriş -5.0 s); T3 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (574, 422) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 179; T1 Zone A 292; T2 Zone A 593
- Radar: 1 — R1 (5324, 2172) yön 178°
- Planlanan görünürlük oranı: [0.367, 0.816, 0.612, 1.0]
- Arama barları (°): R1 [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0]
- Yanlış alarm raporu: 5

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 461 / 532 / 606 | 66 / 150 / 1424 | -540.5 / 33.8 / 182.6 | 0.01 / 7.14 / 26.28 | 0.0 / 0.9 / 16.8 | -5.3 / 1.3 / 5.3 | 2 {0: 2} | 4% | 2% | 23.5 / 24.0 | 20.1 / 28.3 / 36.4 | 2 / 19 / 37 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 840 / 847 / 847 | 107 / 4314 / 7641 | -702.2 / -322.8 / 0.0 | 0.00 / 0.00 / 16.83 | 0.0 / 0.0 / 0.0 | -6.9 / -3.8 / -0.3 | 10 {0: 10} | 21% | 50% | 10.5 / 12.0 | 18.8 / 30.5 / 36.3 | 581 / 774 / 834 | tamam |
| T2 | missile/subsonic_cruise (L2) | terrain_follow+weave | 247 / 248 / 248 | 54 / 62 / 80 | -13.9 / -2.9 / 3.6 | 0.01 / 0.15 / 2.35 | 0.0 / 0.0 / 5.3 | -9.1 / -3.6 / -1.0 | 4 {0: 4} | 8% | 4% | 22.5 / 23.5 | 13.8 / 42.5 / 51.1 | 240 / 243 / 244 | tamam |
| T3 | aircraft/transport (L1) | CV+CT+CA | 129 / 129 / 136 | 2422 / 2623 / 2839 | 12.7 / 12.7 / 12.7 | 0.00 / 0.01 / 0.21 | 0.0 / 0.0 / 0.9 | 24.2 / 24.4 / 25.5 | 6 {0: 6} | 12% | 42% | 13.5 / 14.5 | 48.7 / 65.7 / 69.7 | 3 / 7 / 21 | tamam |

## ep-000033-c8b3137b0fef (train)
- Arazi: valley_sinuous_pass / dense / rocky_mountain
- Arazi: **twisting_gorge**, ölçülen kabartı 1454 m
- Salvo: **single**, 1 füze, süre 26.5 s (hedefe eşzamanlı varış)
- Profiller: T0 popup_dive → Zone A (tehdit ekseni -37.3°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-900, -632) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 199
- Radar: 1 — R1 (100, 2118) yön -67°
- Planlanan görünürlük oranı: [0.333]
- Arama barları (°): R1 [5.3, 11.3, 17.3, 23.3, 29.3, 35.3, 41.3, 47.3]
- Yanlış alarm raporu: 14

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 464 / 543 / 643 | 60 / 887 / 3464 | -641.4 / 33.5 / 278.6 | 0.00 / 0.53 / 26.28 | 0.0 / 0.0 / 0.0 | -4.7 / 1.7 / 4.9 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000034-dae80aeac3c9 (train)
- Arazi: valley_linear_canyon / asymmetric / snowy_highlands
- Arazi: **deep_canyon**, ölçülen kabartı 2262 m
- Salvo: **diverging**, 2 füze, süre 18 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 137.1°, giriş -9.5 s); T1 supersonic_skim → Zone B (tehdit ekseni 142.6°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (954, -1110); Zone B [zone] (-3059, 217) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 151, Zone B 2127; T1 Zone A 5231, Zone B 1075
- Radar: 3 — R1 (-5053, 1804) yön 138°, R2 (447, 304) yön 140°, R3 (3197, 554) yön 156°
- Planlanan görünürlük oranı: [0.919, 1.0]
- Arama barları (°): R1 [3.4, 9.4, 15.4, 21.4, 27.4, 33.4, 39.4, 45.4]; R2 [3.2, 9.2, 15.2, 21.2, 27.2, 33.2, 39.2, 45.2]; R3 [0.5, 6.5, 12.5, 18.5, 24.5, 30.5, 36.5, 42.5]
- Yanlış alarm raporu: 18

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 859 / 874 / 874 | 154 / 5440 / 9211 | -600.1 / -600.1 / -214.4 | 0.22 / 11.06 / 13.61 | 0.0 / 7.5 / 12.0 | -4.1 / -0.5 / 4.2 | 3 {1: 2, 2: 1} | 8% | 28% | 10.0 / 11.0 | 27.9 / 33.6 / 43.7 | 825 / 855 / 859 | tamam |
| T1 | missile/supersonic (L2) | terrain_follow+terminal_weave | 581 / 581 / 582 | 52 / 53 / 82 | -12.8 / 0.0 / 18.0 | 0.00 / 0.69 / 8.48 | 0.0 / 0.0 / 8.1 | -9.1 / -7.3 / 1.8 | 5 {1: 4, 2: 1} | 14% | 6% | 13.5 / 17.5 | 12.5 / 35.7 / 50.9 | 509 / 518 / 558 | tamam |

## ep-000035-42fac8efdd7a (validation)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 494 m
- Salvo: **converging**, 3 füze, süre 18.5 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni -60.7°, giriş -2.0 s); T1 barrel_roll_dive → Zone A (tehdit ekseni 19.7°, giriş +2.5 s); T2 dogleg_serpentine → Zone A (tehdit ekseni 167.8°, giriş +0.0 s); T3 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (284, -1336); Facility 1 [facility] (-5696, -284); Facility 2 [facility] (5015, -2922) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 547; T1 Zone A 162; T2 Zone A 579
- Radar: 2 — R1 (5534, 1914) yön 180°, R2 (-5716, 1914) yön -16°
- Planlanan görünürlük oranı: [1.0, 1.0, 0.711, 1.0]
- Arama barları (°): R1 [2.6, 8.6, 14.6, 20.6, 26.6, 32.6, 38.6, 44.6]; R2 [2.3, 8.3, 14.3, 20.3, 26.3, 32.3, 38.3]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 676 / 793 / 843 | 422 / 5647 / 6650 | -597.5 / -540.8 / 0.0 | 0.00 / 9.77 / 26.22 | 0.0 / 8.2 / 16.6 | -6.4 / -0.5 / 4.3 | 1 {1: 1} | 3% | 0% | 10.0 / — | 23.8 / 23.8 / 23.8 | 728 / 728 / 728 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 784 / 817 / 939 | 435 / 5138 / 5232 | -858.0 / 0.0 / 0.0 | 0.00 / 0.00 / 25.93 | 0.0 / 0.0 / 16.6 | -3.5 / 2.3 / 7.5 | 4 {1: 4} | 11% | 19% | 7.5 / 8.5 | 19.2 / 23.8 / 32.8 | 666 / 686 / 697 | tamam |
| T2 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 472 / 574 / 632 | 42 / 168 / 782 | -418.5 / 23.5 / 270.0 | 0.16 / 8.40 / 26.26 | 0.0 / 0.2 / 16.8 | -9.6 / -1.0 / 3.5 | 3 {0: 2, 1: 1} | 8% | 14% | 7.5 / 8.5 | 23.5 / 33.5 / 35.8 | 528 / 530 / 564 | tamam |
| T3 | uav/fixed_wing (L0) | CT | 26 / 26 / 26 | 1140 / 1177 / 1207 | -3.0 / -3.0 / -3.0 | 0.43 / 0.43 / 0.43 | 9.3 / 9.3 / 9.3 | -8.6 / -0.7 / 4.6 | 7 {0: 5, 1: 2} | 19% | 76% | 3.5 / 5.0 | 21.3 / 43.0 / 75.1 | 3 / 15 / 22 | tamam |

## ep-000036-f82989027810 (train)
- Arazi: valley_linear_canyon / asymmetric / dry_canyon
- Arazi: **deep_canyon**, ölçülen kabartı 2766 m
- Salvo: **single**, 1 füze, süre 29 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni 51.9°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-664, -400) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 266
- Radar: 3 — R1 (5336, -400) yön 89°, R2 (-414, -3400) yön 69°, R3 (-3414, -4900) yön 63°
- Planlanan görünürlük oranı: [0.695]
- Arama barları (°): R1 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5]; R2 [1.5, 7.5, 13.5, 19.5]; R3 [1.5, 7.5, 13.5, 19.5]
- Yanlış alarm raporu: 28

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 791 / 831 / 835 | 157 / 4973 / 8183 | -679.4 / -157.7 / 0.0 | 0.00 / 0.00 / 16.13 | 0.0 / 0.0 / 0.0 | -7.3 / 0.6 / 6.4 | 11 {0: 4, 1: 6, 2: 1} | 17% | 45% | 8.0 / 9.0 | 10.3 / 26.5 / 44.9 | 454 / 638 / 779 | tamam |

## ep-000037-a643cedff356 (test)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 527 m
- Salvo: **converging**, 3 füze, süre 23 s (hedefe eşzamanlı varış)
- Profiller: T0 cruise_weave → Zone A (tehdit ekseni 153.0°, giriş +0.0 s); T1 random_jink_dive → Zone A (tehdit ekseni 88.1°, giriş +6.0 s); T2 shallow_dive → Zone A (tehdit ekseni 30.0°, giriş +1.0 s); T3 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (543, -856); Facility 1 [facility] (-1580, 3226); Facility 2 [facility] (2406, 4600) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 957; T1 Zone A 374; T2 Zone A 106
- Radar: 2 — R1 (543, 2144) yön -32°, R2 (-4457, 1144) yön 9°
- Planlanan görünürlük oranı: [0.723, 0.574, 1.0, 1.0]
- Arama barları (°): R1 [-2.7, 3.3, 9.3, 15.3, 21.3, 27.3, 33.3, 39.3]; R2 [3.8, 9.8, 15.8, 21.8, 27.8, 33.8, 39.8]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | terrain_follow+weave | 256 / 257 / 257 | 50 / 52 / 55 | -0.7 / 1.2 / 2.6 | 0.01 / 0.04 / 2.78 | 0.0 / 0.0 / 6.1 | -12.9 / -6.9 / 2.5 | 6 {1: 6} | 13% | 37% | 14.0 / 15.0 | 30.4 / 44.7 / 56.0 | 240 / 250 / 254 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+random_jinks | 726 / 742 / 865 | 481 / 5611 / 5617 | -640.4 / 0.0 / 0.0 | 0.00 / 0.00 / 26.21 | 0.0 / 0.0 / 15.7 | -3.8 / 1.5 / 5.6 | 2 {1: 2} | 4% | 2% | 20.5 / 23.0 | 27.1 / 27.9 / 28.8 | 45 / 152 / 259 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+push_over+dive | 676 / 686 / 686 | 272 / 4637 / 5179 | -386.4 / -386.4 / 0.0 | 0.00 / 0.00 / 13.90 | 0.0 / 0.0 / 0.0 | -5.6 / -2.8 / 1.7 | 12 {0: 8, 1: 4} | 26% | 70% | 6.5 / 7.5 | 16.7 / 31.3 / 62.2 | 77 / 553 / 624 | tamam |
| T3 | aircraft/transport (L1) | CV+CV | 134 / 134 / 134 | 601 / 753 / 802 | -10.8 / -10.8 / -10.8 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 16.6 / 19.5 / 23.0 | 17 {0: 10, 1: 7} | 35% | 93% | 1.0 / 2.0 | 19.6 / 54.1 / 78.1 | 82 / 111 / 127 | tamam |

## ep-000038-739a2c3048b0 (train)
- Arazi: basin_highlands / dense / green_forest_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1395 m
- Salvo: **converging**, 2 füze, süre 17 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 54.0°, giriş -5.5 s); T1 supersonic_skim → Zone A (tehdit ekseni 113.6°, giriş +0.0 s); T2 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (118, 1407) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 301; T1 Zone A 325
- Radar: 1 — R1 (-1382, 157) yön 61°
- Planlanan görünürlük oranı: [0.571, 0.429, 1.0]
- Arama barları (°): R1 [3.0, 9.0, 15.0, 21.0, 27.0, 33.0, 39.0, 45.0]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 821 / 890 / 897 | 179 / 5820 / 9524 | -793.9 / -761.1 / 0.0 | 0.00 / 10.12 / 10.97 | 0.0 / 0.0 / 0.0 | -4.3 / -1.1 / 3.8 | 3 {0: 3} | 9% | 15% | 14.0 / 15.0 | 30.6 / 50.1 / 57.8 | 750 / 781 / 804 | tamam |
| T1 | missile/supersonic (L2) | terrain_follow+dogleg_turn+terminal_weave | 567 / 636 / 647 | 60 / 174 / 310 | -49.0 / -2.5 / 49.0 | 0.05 / 4.25 / 12.27 | 0.0 / 3.0 / 11.0 | -4.4 / -3.6 / 3.0 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T2 | uav/fixed_wing (L0) | CV | 42 / 42 / 42 | 540 / 573 / 612 | 2.2 / 2.2 / 2.2 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 3.2 / 3.3 / 3.4 | 8 {0: 8} | 24% | 74% | 3.5 / 5.0 | 13.6 / 45.8 / 56.9 | 15 / 18 / 20 | tamam |

## ep-000039-8a4392ea635e (train)
- Arazi: ridge_multi_obstacle / dense / mixed_rock_vegetation
- Arazi: **knife_ridges**, ölçülen kabartı 954 m
- Salvo: **diverging**, 3 füze, süre 22 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni -129.8°, giriş +0.0 s); T1 steep_dive → Zone B (tehdit ekseni -135.0°, giriş +0.5 s); T2 barrel_roll_dive → Zone C (tehdit ekseni -128.9°, giriş +4.5 s)
- Bölgeler: Zone A [zone] (471, 200); Zone B [zone] (5046, 536); Zone C [zone] (650, 4350); Facility 1 [facility] (-499, 1720) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 223, Zone B 4521, Zone C 4356; T1 Zone A 3130, Zone B 136, Zone C 5938; T2 Zone A 2049, Zone B 5426, Zone C 602
- Radar: 1 — R1 (-444, 3696) yön -85°
- Planlanan görünürlük oranı: [0.889, 0.867, 0.933]
- Arama barları (°): R1 [17.5, 23.5, 29.5, 35.5, 41.5, 47.5, 53.5, 59.5]
- Yanlış alarm raporu: 12

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 684 / 818 / 866 | 448 / 6969 / 8648 | -682.6 / -628.7 / 0.0 | 0.00 / 1.52 / 26.22 | 0.0 / 1.0 / 16.3 | -8.1 / -1.4 / 5.1 | 1 {0: 1} | 2% | 0% | 19.5 / — | 31.8 / 31.8 / 31.8 | 375 / 375 / 375 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 857 / 932 / 951 | 347 / 8235 / 10041 | -795.0 / -663.6 / 0.0 | 0.00 / 0.00 / 19.05 | 0.0 / 0.0 / 0.0 | -2.8 / 0.3 / 2.5 | 6 {0: 6} | 14% | 45% | 11.5 / 12.5 | 14.9 / 29.6 / 48.1 | 408 / 727 / 753 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 692 / 700 / 824 | 487 / 4407 / 5116 | -628.8 / 0.0 / 0.0 | 0.00 / 0.00 / 25.71 | 0.0 / 0.0 / 14.2 | -1.8 / -0.4 / 3.7 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000040-1dceba8a9b22 (train)
- Arazi: mountain_chokepoint / sparse / dry_canyon
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1401 m
- Salvo: **single**, 1 füze, süre 25.5 s (hedefe eşzamanlı varış)
- Profiller: T0 popup_dive → Zone A (tehdit ekseni -2.6°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (118, -305); Facility 1 [facility] (-455, 4249) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 143
- Radar: 3 — R1 (-5632, -2055) yön 1°, R2 (2368, 5695) yön -54°, R3 (1618, 1945) yön -32°
- Planlanan görünürlük oranı: [0.481]
- Arama barları (°): R1 [3.3, 9.3, 15.3, 21.3]; R2 [5.9, 11.9, 17.9, 23.9, 29.9]; R3 [6.4, 12.4, 18.4, 24.4, 30.4, 36.4, 42.4, 48.4]
- Yanlış alarm raporu: 29

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 508 / 601 / 700 | 35 / 138 / 2853 | -678.8 / 46.2 / 322.8 | 0.02 / 0.98 / 26.27 | 0.0 / 0.0 / 0.0 | -3.7 / -0.1 / 2.6 | 11 {0: 8, 1: 2, 2: 1} | 20% | 27% | 14.5 / 15.5 | 13.3 / 24.5 / 49.1 | 27 / 281 / 469 | tamam |

## ep-000041-454970c2124b (train)
- Arazi: valley_sinuous_pass / asymmetric / mixed_rock_vegetation
- Arazi: **twisting_gorge**, ölçülen kabartı 1983 m
- Salvo: **diverging**, 2 füze, süre 24.5 s (hedefe eşzamanlı varış)
- Profiller: T0 dive_pullout → Zone A (tehdit ekseni -44.7°, giriş +0.0 s); T1 dogleg_serpentine → Zone B (tehdit ekseni -36.5°, giriş +0.0 s); T2 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (647, 894); Zone B [zone] (4595, 1751) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 30, Zone B 3397; T1 Zone A 4321, Zone B 339
- Radar: 2 — R1 (6871, 1322) yön -100°, R2 (-1129, 2072) yön 19°
- Planlanan görünürlük oranı: [0.56, 0.38, 1.0]
- Arama barları (°): R1 [-6.2, -0.2, 5.8, 11.8, 17.8, 23.8, 29.8, 35.8]; R2 [3.1, 9.1, 15.1, 21.1, 27.1, 33.1]
- Yanlış alarm raporu: 28

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 822 / 849 / 849 | 275 / 4821 / 8785 | -693.1 / -385.6 / 0.0 | 0.00 / 0.00 / 16.49 | 0.0 / 0.0 / 0.0 | -6.6 / -0.1 / 2.9 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 461 / 510 / 512 | 41 / 204 / 848 | -440.4 / 42.1 / 253.2 | 0.02 / 6.63 / 26.28 | 0.0 / 0.0 / 16.8 | -5.3 / 1.5 / 6.3 | 3 {1: 3} | 6% | 6% | 22.5 / 23.5 | 36.4 / 41.9 / 50.0 | 233 / 237 / 244 | tamam |
| T2 | aircraft/transport (L1) | CT+CV+CV | 132 / 132 / 132 | 1058 / 1641 / 1757 | 3.5 / 3.5 / 3.5 | 0.00 / 0.00 / 0.38 | 0.0 / 0.0 / 1.6 | 18.0 / 20.3 / 22.6 | 14 {1: 14} | 29% | 78% | 5.0 / 6.0 | 21.5 / 54.0 / 74.5 | 45 / 75 / 97 | tamam |

## ep-000042-b1435a0f2057 (validation)
- Arazi: ridge_multi_obstacle / dense / dry_canyon
- Arazi: **knife_ridges**, ölçülen kabartı 959 m
- Salvo: **single**, 1 füze, süre 21 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 88.5°, giriş +0.0 s); T1 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (169, -291) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 228
- Radar: 2 — R1 (2169, -291) yön -160°, R2 (4669, -1291) yön 107°
- Planlanan görünürlük oranı: [0.953, 1.0]
- Arama barları (°): R1 [4.5, 10.5, 16.5, 22.5, 28.5, 34.5, 40.5, 46.5]; R2 [-7.1, -1.1, 4.9, 10.9, 16.9, 22.9, 28.9, 34.9]
- Yanlış alarm raporu: 13

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 783 / 818 / 818 | 169 / 5282 / 7368 | -573.1 / -573.1 / 0.0 | 0.00 / 11.90 / 16.09 | 0.0 / 8.3 / 14.0 | -3.6 / 0.7 / 5.3 | 6 {0: 1, 1: 5} | 14% | 40% | 12.0 / 13.0 | 23.8 / 40.1 / 55.1 | 36 / 638 / 726 | tamam |
| T1 | aircraft/fighter (L1) | CT | 268 / 268 / 268 | 1482 / 1604 / 1763 | 7.5 / 7.5 / 7.5 | 5.42 / 5.42 / 5.42 | 11.4 / 11.4 / 11.4 | 7.3 / 14.3 / 15.9 | 8 {0: 8} | 19% | 62% | 5.5 / 8.5 | 29.3 / 43.9 / 52.0 | 32 / 210 / 251 | tamam |

## ep-000043-65f6d0fa7b5f (train)
- Arazi: basin_highlands / sparse / green_forest_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1052 m
- Salvo: **diverging**, 2 füze, süre 25 s (hedefe eşzamanlı varış)
- Profiller: T0 cruise_weave → Zone A (tehdit ekseni 119.4°, giriş +0.0 s); T1 shallow_dive → Zone B (tehdit ekseni 143.9°, giriş +1.0 s)
- Bölgeler: Zone A [zone] (-777, 382); Zone B [zone] (848, 4229); Facility 1 [facility] (-2071, -3015) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 846, Zone B 2332; T1 Zone A 4135, Zone B 96
- Radar: 3 — R1 (-4215, -3194) yön 92°, R2 (35, -3444) yön 115°, R3 (-1215, 4056) yön 175°
- Planlanan görünürlük oranı: [0.922, 1.0]
- Arama barları (°): R1 [-6.1, -0.1, 5.9, 11.9, 17.9, 23.9]; R2 [0.5, 6.5, 12.5, 18.5, 24.5]; R3 [-0.3, 5.7, 11.7, 17.7, 23.7, 29.7, 35.7, 41.7]
- Yanlış alarm raporu: 26

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | terrain_follow+dogleg_turn+weave+pop_up+terminal_descent | 251 / 261 / 261 | 51 / 77 / 179 | -19.6 / -1.9 / 19.6 | 0.00 / 0.37 / 3.47 | 0.0 / 0.0 / 7.6 | -14.5 / -7.0 / -1.3 | 10 {1: 7, 2: 3} | 20% | 54% | 10.5 / 12.0 | 11.3 / 27.3 / 96.5 | 39 / 258 / 261 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 729 / 796 / 796 | 265 / 6638 / 7667 | -440.4 / -440.4 / 0.0 | 0.00 / 0.00 / 15.70 | 0.0 / 0.0 / 0.0 | -6.0 / -1.9 / 3.5 | 10 {0: 4, 1: 4, 2: 2} | 20% | 64% | 8.5 / 9.5 | 16.0 / 32.1 / 45.0 | 224 / 583 / 715 | tamam |

## ep-000044-85e5cb0e6073 (test)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 1318 m
- Salvo: **diverging**, 3 füze, süre 29 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni 116.8°, giriş +10.5 s); T1 dive_pullout → Zone B (tehdit ekseni 123.6°, giriş +0.0 s); T2 break_turn_dive → Zone C (tehdit ekseni 109.2°, giriş +5.5 s); T3 traffic/fixed_wing → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-341, -1448); Zone B [zone] (-1895, -5735); Zone C [zone] (-3974, -793); Facility 1 [facility] (-249, 2597) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 183, Zone B 4566, Zone C 3046; T1 Zone A 3765, Zone B 118, Zone C 1121; T2 Zone A 3866, Zone B 5831, Zone C 485
- Radar: 1 — R1 (680, -6409) yön 111°
- Planlanan görünürlük oranı: [0.746, 0.678, 0.712, 1.0]
- Arama barları (°): R1 [4.4, 10.4, 16.4, 22.4, 28.4]
- Yanlış alarm raporu: 7

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 689 / 708 / 837 | 250 / 5682 / 5772 | -667.9 / 0.0 / 0.0 | 0.00 / 0.00 / 26.22 | 0.0 / 0.0 / 16.2 | -3.3 / -0.5 / 2.5 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 847 / 930 / 930 | 162 / 5472 / 10045 | -737.2 / -315.1 / 0.0 | 0.00 / 0.00 / 17.31 | 0.0 / 0.0 / 0.0 | -7.6 / -2.0 / 2.1 | 4 {0: 4} | 7% | 19% | 22.5 / 24.0 | 20.3 / 29.8 / 41.0 | 832 / 884 / 894 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 652 / 723 / 846 | 430 / 7972 / 7982 | -674.4 / 0.0 / 0.0 | 0.00 / 0.75 / 26.22 | 0.0 / 0.8 / 17.1 | -3.5 / -0.7 / 4.9 | 5 {0: 5} | 9% | 31% | 19.5 / 20.5 | 10.0 / 30.7 / 48.9 | 560 / 699 / 727 | tamam |
| T3 | uav/fixed_wing (L0) | CV+CV+CT | 30 / 30 / 30 | 537 / 557 / 576 | 1.4 / 1.4 / 1.4 | 0.00 / 0.00 / 0.94 | 0.0 / 0.0 / 17.8 | -6.6 / 0.3 / 0.7 | 17 {0: 17} | 29% | 95% | 1.0 / 2.0 | 14.4 / 47.5 / 54.7 | 6 / 8 / 29 | tamam |

## ep-000045-5fc6ddadcbc9 (train)
- Arazi: mountain_chokepoint / sparse / rocky_mountain
- Arazi: **alpine_chokepoint**, ölçülen kabartı 1605 m
- Salvo: **converging**, 2 füze, süre 25 s (hedefe eşzamanlı varış)
- Profiller: T0 dogleg_serpentine → Zone A (tehdit ekseni -24.2°, giriş +0.0 s); T1 popup_dive → Zone A (tehdit ekseni -67.3°, giriş -4.5 s)
- Bölgeler: Zone A [zone] (611, 1258) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 151; T1 Zone A 230
- Radar: 3 — R1 (611, -742) yön 14°, R2 (-639, -4242) yön 47°, R3 (-3139, 758) yön -4°
- Planlanan görünürlük oranı: [0.549, 0.549]
- Arama barları (°): R1 [-2.0, 4.0, 10.0, 16.0, 22.0, 28.0, 34.0, 40.0]; R2 [-8.7, -2.7, 3.3, 9.3, 15.3, 21.3, 27.3]; R3 [-14.0, -8.0, -2.0, 4.0, 10.0, 16.0, 22.0]
- Yanlış alarm raporu: 26

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 461 / 509 / 558 | 50 / 174 / 925 | -403.3 / 0.0 / 252.1 | 0.08 / 6.87 / 26.28 | 0.0 / 0.1 / 17.0 | -7.2 / 2.5 / 5.1 | 2 {0: 1, 2: 1} | 4% | 6% | 8.0 / 24.0 | 12.8 / 27.1 / 41.4 | 323 / 375 / 427 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 464 / 512 / 612 | 110 / 696 / 3322 | -575.7 / 66.3 / 240.9 | 0.00 / 0.59 / 26.28 | 0.0 / 0.0 / 0.0 | -5.5 / 2.4 / 4.7 | 3 {1: 1, 2: 2} | 6% | 18% | 20.0 / 21.0 | 35.9 / 45.7 / 59.0 | 59 / 237 / 265 | tamam |

## ep-000046-afa6a196e1c4 (train)
- Arazi: valley_sinuous_pass / dense / dry_canyon
- Arazi: **twisting_gorge**, ölçülen kabartı 1212 m
- Salvo: **single**, 1 füze, süre 19.5 s (hedefe eşzamanlı varış)
- Profiller: T0 shallow_dive → Zone A (tehdit ekseni 178.0°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-890, 697); Facility 1 [facility] (-2174, -4973); Facility 2 [facility] (-2093, 3110) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 19
- Radar: 1 — R1 (-3390, -1553) yön 137°
- Planlanan görünürlük oranı: [0.725]
- Arama barları (°): R1 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5, 37.5]
- Yanlış alarm raporu: 7

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 615 / 627 / 627 | 193 / 3541 / 5223 | -346.1 / -346.1 / 0.0 | 0.00 / 0.00 / 12.41 | 0.0 / 0.0 / 0.0 | -5.0 / -0.1 / 3.3 | 1 {0: 1} | 3% | 0% | 11.0 / — | 30.1 / 30.1 / 30.1 | 450 / 450 / 450 | tamam |

## ep-000047-0e37513976a3 (train)
- Arazi: valley_linear_canyon / asymmetric / rocky_mountain
- Arazi: **deep_canyon**, ölçülen kabartı 1855 m
- Salvo: **converging**, 3 füze, süre 19.5 s (hedefe eşzamanlı varış)
- Profiller: T0 supersonic_skim → Zone A (tehdit ekseni 46.9°, giriş +0.0 s); T1 cruise_weave → Zone A (tehdit ekseni 1.2°, giriş -3.5 s); T2 random_jink_dive → Zone A (tehdit ekseni 113.1°, giriş +0.0 s); T3 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-1235, 769); Facility 1 [facility] (991, -1285) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 842; T1 Zone A 587; T2 Zone A 467
- Radar: 2 — R1 (-7235, -1481) yön 26°, R2 (5015, 3519) yön 163°
- Planlanan görünürlük oranı: [1.0, 0.975, 1.0, 1.0]
- Arama barları (°): R1 [-7.3, -1.3, 4.7, 10.7, 16.7, 22.7, 28.7]; R2 [-4.7, 1.3, 7.3, 13.3, 19.3, 25.3, 31.3]
- Yanlış alarm raporu: 15

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/supersonic (L2) | terrain_follow+terminal_weave | 531 / 551 / 552 | 103 / 141 / 558 | -49.0 / 5.6 / 35.6 | 0.01 / 2.28 / 9.57 | 0.0 / 1.0 / 9.8 | -2.6 / -0.6 / 5.2 | 5 {0: 5} | 13% | 18% | 4.0 / 16.5 | 13.6 / 31.1 / 36.8 | 502 / 521 / 540 | tamam |
| T1 | missile/subsonic_cruise (L2) | terrain_follow+dogleg_turn+weave | 244 / 251 / 251 | 74 / 74 / 101 | -14.5 / -0.0 / 0.2 | 0.00 / 0.75 / 3.47 | 0.0 / 1.7 / 8.0 | -13.7 / -6.4 / -1.9 | 1 {0: 1} | 3% | 0% | 18.5 / — | 24.1 / 24.1 / 24.1 | 218 / 218 / 218 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+random_jinks | 725 / 779 / 886 | 198 / 5608 / 6488 | -746.0 / -444.2 / 0.0 | 0.00 / 7.59 / 26.21 | 0.0 / 3.4 / 16.8 | -2.0 / -0.0 / 2.0 | 8 {0: 5, 1: 3} | 18% | 33% | 12.5 / 13.5 | 17.5 / 24.0 / 42.1 | 46 / 302 / 611 | tamam |
| T3 | aircraft/fighter (L1) | CT+CT | 196 / 196 / 196 | 968 / 1019 / 1166 | -10.2 / -10.2 / -10.2 | 3.79 / 4.39 / 4.39 | 10.9 / 12.6 / 12.6 | 4.8 / 11.1 / 13.1 | 12 {0: 10, 1: 2} | 31% | 92% | 1.0 / 2.0 | 15.9 / 48.8 / 62.3 | 20 / 139 / 191 | tamam |

## ep-000048-27307e1f727f (validation)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 455 m
- Salvo: **single**, 1 füze, süre 22.5 s (hedefe eşzamanlı varış)
- Profiller: T0 steep_dive → Zone A (tehdit ekseni 60.8°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-1311, -416) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 224
- Radar: 1 — R1 (-1811, 2834) yön 36°
- Planlanan görünürlük oranı: [0.5]
- Arama barları (°): R1 [25.1, 31.1, 37.1, 43.1, 49.1, 55.1, 61.1, 67.1]
- Yanlış alarm raporu: 12

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 927 / 1008 / 1044 | 249 / 10330 / 11896 | -915.9 / -692.7 / 0.0 | 0.00 / 0.00 / 18.85 | 0.0 / 0.0 / 0.0 | -1.2 / 2.7 / 5.0 | 0 {} | 0% | 0% | — / — | — | — | tamam |

## ep-000049-56f5933d301c (train)
- Arazi: mountain_chokepoint / asymmetric / grassland_valley
- Arazi: **alpine_chokepoint**, ölçülen kabartı 2214 m
- Salvo: **diverging**, 3 füze, süre 28 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 142.9°, giriş +4.5 s); T1 barrel_roll_dive → Zone B (tehdit ekseni 140.8°, giriş +9.0 s); T2 cruise_weave → Zone C (tehdit ekseni 148.6°, giriş +0.0 s); T3 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (540, -177); Zone B [zone] (1515, -4986); Zone C [zone] (-3017, -3614); Facility 1 [facility] (-1210, 4931) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 196, Zone B 4873, Zone C 5090; T1 Zone A 3194, Zone B 148, Zone C 1710; T2 Zone A 4937, Zone B 5396, Zone C 779
- Radar: 2 — R1 (-4071, -6175) yön 99°, R2 (-6571, 75) yön -9°
- Planlanan görünürlük oranı: [0.807, 0.772, 0.86, 1.0]
- Arama barları (°): R1 [6.6, 12.6, 18.6, 24.6, 30.6, 36.6, 42.6, 48.6]; R2 [-0.7, 5.3, 11.3, 17.3, 23.3, 29.3, 35.3, 41.3]
- Yanlış alarm raporu: 23

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 758 / 814 / 852 | 198 / 9170 / 9787 | -658.0 / -411.5 / 0.0 | 0.00 / 5.19 / 15.21 | 0.0 / 0.0 / 13.2 | -2.0 / 1.1 / 3.1 | 11 {0: 11} | 20% | 71% | 7.0 / 8.0 | 10.0 / 29.2 / 39.5 | 393 / 522 / 616 | tamam |
| T1 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 710 / 739 / 866 | 421 / 7605 / 9348 | -738.4 / 0.0 / 0.0 | 0.00 / 0.00 / 25.23 | 0.0 / 0.0 / 15.8 | -1.2 / 3.7 / 5.6 | 2 {0: 1, 1: 1} | 4% | 0% | 15.0 / — | 12.8 / 15.1 / 17.5 | 31 / 163 / 296 | tamam |
| T2 | missile/subsonic_cruise (L2) | terrain_follow+weave+pop_up | 249 / 250 / 250 | 50 / 61 / 179 | -3.1 / 6.4 / 18.0 | 0.00 / 0.10 / 2.65 | 0.0 / 0.0 / 4.4 | -11.1 / -4.2 / 2.3 | 6 {0: 6} | 11% | 34% | 17.0 / 19.0 | 31.5 / 42.5 / 64.4 | 84 / 171 / 208 | tamam |
| T3 | aircraft/transport (L1) | CT | 120 / 120 / 120 | 2618 / 2730 / 2737 | 0.6 / 0.6 / 0.6 | 0.37 / 0.37 / 0.37 | 1.7 / 1.7 / 1.7 | 18.5 / 24.0 / 24.8 | 13 {1: 13} | 23% | 93% | 1.0 / 2.5 | 13.8 / 60.0 / 67.6 | 65 / 82 / 108 | tamam |

## Varyant bantları (referans)

| Varyant | Hız m/s (sert) | AGL m (sert) | max \|v_z\| | max a_n g | max dönüş °/s |
|---|---|---|---|---|---|
| uav/multirotor | 0–20 | 30–400 | 8 | 0.8 | 57.3 |
| uav/fixed_wing | 18–62 | 100–1500 | 5 | 1.0 | 20.1 |
| aircraft/transport | 80–200 | 300–4000 | 15 | 0.6 | 3.4 |
| aircraft/fighter | 120–320 | 100–4000 | 60 | 8.2 | 20.1 |
| missile/subsonic_cruise | 200–280 | 30–200 | 20 | 4.1 | 8.6 |
| missile/supersonic | 450–950 | 30–800 | 50 | 15.3 | 11.5 |
| missile/terminal_dive | 450–1100 | 30–15000 | 1000 | 30.6 | 17.2 |
