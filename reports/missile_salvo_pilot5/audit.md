# Hedef sınıfları pilot dataset denetimi

Dataset: `C:\Users\tahau\Desktop\marl\demo\data\processed\missile_salvo_pilot5`  
Tüm kontrolleri geçen hedef: **12 / 12**

Değerler `min / medyan / max`. Bantlar `docs/target_classification.md` ve `src/core/target_classes.py` ile aynı kaynaktan.

## ep-000000-4b1eb8e1ba86 (train)
- Arazi: rolling_hills / open / None
- Arazi: **rolling_hills**, ölçülen kabartı 996 m
- Salvo: **converging**, 2 füze, süre 21 s (hedefe eşzamanlı varış)
- Profiller: T0 random_jink_dive → Zone A (tehdit ekseni -23.2°, giriş -0.5 s); T1 dogleg_serpentine → Zone A (tehdit ekseni -169.2°, giriş +0.0 s); T2 traffic/multirotor → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-578, 1161); Facility 1 [facility] (-4441, 3078) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 554; T1 Zone A 473
- Radar: 2 — R1 (-2578, -1339) yön 174°, R2 (1422, 3161) yön -57°
- Planlanan görünürlük oranı: [0.558, 0.442, 1.0]
- Arama barları (°): R1 [2.6, 8.6, 14.6, 20.6, 26.6, 32.6, 38.6, 44.6]; R2 [-0.3, 5.7, 11.7, 17.7, 23.7, 29.7, 35.7, 41.7]
- Yanlış alarm raporu: 15

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 690 / 801 / 828 | 392 / 6702 / 8497 | -646.2 / -566.2 / 0.0 | 0.00 / 6.67 / 26.22 | 0.0 / 4.4 / 16.4 | -4.6 / 0.6 / 5.9 | 3 {1: 3} | 7% | 19% | 16.0 / 17.5 | 23.5 / 28.8 / 46.7 | 664 / 702 / 713 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 534 / 609 / 679 | 55 / 77 / 636 | -349.7 / 5.9 / 277.0 | 0.05 / 7.09 / 26.25 | 0.0 / 0.1 / 16.7 | -5.3 / -1.6 / 6.6 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T2 | uav/multirotor (L0) | CA | 7 / 13 / 20 | 53 / 56 / 58 | -0.1 / -0.1 / -0.1 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -16.2 / -15.3 / -14.7 | 10 {1: 10} | 24% | 90% | 1.0 / 2.5 | 17.0 / 37.7 / 40.9 | 1 / 2 / 2 | tamam |

## ep-000001-e8975a2f2d44 (train)
- Arazi: valley_linear_canyon / dense / rocky_mountain
- Arazi: **deep_canyon**, ölçülen kabartı 1614 m
- Salvo: **diverging**, 3 füze, süre 23.5 s (hedefe eşzamanlı varış)
- Profiller: T0 cruise_weave → Zone A (tehdit ekseni 51.7°, giriş +0.0 s); T1 popup_dive → Zone B (tehdit ekseni 73.0°, giriş -6.0 s); T2 shallow_dive → Zone C (tehdit ekseni 64.4°, giriş +2.0 s)
- Bölgeler: Zone A [zone] (-217, 497); Zone B [zone] (2351, -2565); Zone C [zone] (-3688, -1399); Facility 1 [facility] (-1525, -2441); Facility 2 [facility] (-1020, 3068) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 1012, Zone B 4328, Zone C 4899; T1 Zone A 3639, Zone B 291, Zone C 6403; T2 Zone A 2357, Zone B 5992, Zone C 54
- Radar: 3 — R1 (-6268, -2906) yön 31°, R2 (-4018, -2406) yön 38°, R3 (732, 5594) yön -61°
- Planlanan görünürlük oranı: [1.0, 0.812, 1.0]
- Arama barları (°): R1 [0.7, 6.7, 12.7, 18.7, 24.7]; R2 [3.2, 9.2, 15.2, 21.2, 27.2, 33.2]; R3 [-18.0, -12.0, -6.0, 0.0, 6.0, 12.0, 18.0, 24.0]
- Yanlış alarm raporu: 29

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | terrain_follow+weave+pop_up+terminal_descent | 243 / 244 / 244 | 75 / 121 / 177 | -19.6 / -13.9 / 18.0 | 0.00 / 0.54 / 3.47 | 0.0 / 0.0 / 4.8 | -10.2 / -8.9 / -0.9 | 4 {0: 2, 1: 2} | 9% | 0% | 1.0 / — | 11.6 / 15.0 / 23.2 | 235 / 240 / 243 | tamam |
| T1 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 464 / 566 / 567 | 61 / 642 / 1801 | -464.0 / 147.1 / 388.9 | 0.00 / 4.83 / 26.28 | 0.0 / 0.0 / 0.0 | -2.9 / 4.3 / 7.2 | 8 {0: 6, 1: 1, 2: 1} | 15% | 40% | 1.0 / 11.0 | 19.7 / 26.7 / 49.4 | 46 / 261 / 437 | tamam |
| T2 | missile/terminal_dive (L2) | cruise+push_over+dive | 649 / 666 / 666 | 184 / 3236 / 4579 | -369.1 / -369.1 / 0.0 | 0.00 / 0.00 / 13.23 | 0.0 / 0.0 / 0.0 | -5.8 / -1.1 / 7.2 | 14 {0: 5, 1: 9} | 26% | 66% | 7.5 / 8.5 | 13.0 / 27.5 / 49.5 | 533 / 643 / 665 | tamam |

## ep-000002-f69d4fdeb72c (validation)
- Arazi: ridge_multi_obstacle / asymmetric / dry_canyon
- Arazi: **knife_ridges**, ölçülen kabartı 1417 m
- Salvo: **single**, 1 füze, süre 15 s (hedefe eşzamanlı varış)
- Profiller: T0 barrel_roll_dive → Zone A (tehdit ekseni -42.2°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (1105, 731); Facility 1 [facility] (-3257, 1843); Facility 2 [facility] (4250, 3678) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 615
- Radar: 1 — R1 (605, -5519) yön 25°
- Planlanan görünürlük oranı: [0.871]
- Arama barları (°): R1 [1.5, 7.5, 13.5, 19.5, 25.5, 31.5]
- Yanlış alarm raporu: 5

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 835 / 852 / 971 | 544 / 4265 / 4361 | -759.2 / 0.0 / 0.0 | 0.00 / 5.17 / 25.84 | 0.0 / 0.0 / 14.3 | -2.8 / 2.0 / 2.8 | 1 {0: 1} | 3% | 0% | 8.5 / — | 25.2 / 25.2 / 25.2 | 254 / 254 / 254 | tamam |

## ep-000003-1be2ec0bac32 (train)
- Arazi: basin_highlands / sparse / grassland_valley
- Arazi: **highland_bowl**, ölçülen kabartı 1392 m
- Salvo: **converging**, 2 füze, süre 24.5 s (hedefe eşzamanlı varış)
- Profiller: T0 jink_dive → Zone A (tehdit ekseni 41.2°, giriş +4.0 s); T1 steep_dive → Zone A (tehdit ekseni -9.6°, giriş +0.0 s); T2 traffic/fighter → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (-1191, -22); Facility 1 [facility] (2732, -741); Facility 2 [facility] (-1928, -5367) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 254; T1 Zone A 115
- Radar: 1 — R1 (-5941, 728) yön -4°
- Planlanan görünürlük oranı: [0.7, 0.84, 1.0]
- Arama barları (°): R1 [-8.1, -2.1, 3.9, 9.9, 15.9, 21.9, 27.9, 33.9]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 879 / 934 / 981 | 277 / 7863 / 8655 | -708.4 / -409.9 / 0.0 | 0.00 / 9.80 / 19.98 | 0.0 / 0.0 / 11.9 | -7.2 / -3.5 / 1.9 | 0 {} | 0% | 2% | — / 24.5 | — | — | tamam |
| T1 | missile/terminal_dive (L2) | cruise+push_over+dive | 765 / 784 / 804 | 316 / 7828 / 8783 | -718.3 / -432.0 / 0.0 | 0.00 / 0.00 / 10.02 | 0.0 / 0.0 / 0.0 | -4.0 / -0.2 / 4.1 | 4 {0: 4} | 8% | 27% | 17.5 / 18.5 | 13.9 / 39.9 / 50.9 | 530 / 645 / 667 | tamam |
| T2 | aircraft/fighter (L1) | CT+CT | 243 / 243 / 243 | 2579 / 2906 / 3423 | -31.7 / -31.7 / -31.7 | 6.54 / 7.64 / 7.64 | 15.3 / 17.9 / 17.9 | 8.5 / 14.2 / 17.6 | 9 {0: 9} | 18% | 67% | 7.5 / 8.5 | 19.2 / 52.0 / 61.0 | 14 / 149 / 235 | tamam |

## ep-000004-0afe92e2a3bb (test)
- Arazi: mountain_chokepoint / asymmetric / green_forest_valley
- Arazi: **alpine_chokepoint**, ölçülen kabartı 2153 m
- Salvo: **single**, 1 füze, süre 23 s (hedefe eşzamanlı varış)
- Profiller: T0 break_turn_dive → Zone A (tehdit ekseni 121.2°, giriş +0.0 s); T1 traffic/transport → — (tehdit ekseni —°, giriş +0.0 s)
- Bölgeler: Zone A [zone] (787, -964); Facility 1 [facility] (-536, 3846) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 300
- Radar: 3 — R1 (-2213, -964) yön 42°, R2 (-713, -4464) yön 95°, R3 (1787, -3964) yön 111°
- Planlanan görünürlük oranı: [0.979, 1.0]
- Arama barları (°): R1 [-1.5, 4.5, 10.5, 16.5, 22.5, 28.5, 34.5, 40.5]; R2 [-7.1, -1.1, 4.9, 10.9, 16.9, 22.9, 28.9, 34.9]; R3 [13.7, 19.7, 25.7, 31.7, 37.7, 43.7]
- Yanlış alarm raporu: 23

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 676 / 775 / 897 | 651 / 7649 / 8506 | -764.4 / -210.6 / 0.0 | 0.00 / 11.01 / 26.22 | 0.0 / 10.2 / 17.1 | -7.5 / -1.4 / 3.2 | 10 {0: 5, 1: 4, 2: 1} | 22% | 48% | 11.5 / 12.5 | 15.4 / 29.8 / 51.0 | 216 / 541 / 682 | tamam |
| T1 | aircraft/transport (L1) | CV+CV+CV | 120 / 120 / 120 | 1071 / 1093 / 1282 | -1.8 / -1.8 / -1.8 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | 14.9 / 16.0 / 17.3 | 13 {2: 13} | 28% | 91% | 1.5 / 2.5 | 19.8 / 53.2 / 63.6 | 100 / 108 / 111 | tamam |

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
