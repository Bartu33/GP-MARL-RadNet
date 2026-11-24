# Hedef sınıfları pilot dataset denetimi

Dataset: `data`  
Tüm kontrolleri geçen hedef: **13 / 13**

Değerler `min / medyan / max`. Bantlar `docs/target_classification.md` ve `src/core/target_classes.py` ile aynı kaynaktan.

## ep-000000-d35912e0daf7 (train)
- Arazi: valley_linear_canyon / asymmetric / dry_canyon
- Arazi profili: **deep_canyon**, kabartı 2472 m (ayarlanan relief 1256 m, vadi 3233 m)
- Rota deseni: **converging** (2 füze, fazlar ['in', 'in'], yönler [-141.0, 37.2]°)
- Bölgeler: Zone A [zone] (85, -1164); Facility 1 [facility] (3653, 1498); Facility 2 [facility] (1995, -5041) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 204; T1 Zone A 179
- Radar: 2 — R1 (-4551, 2742) yön -57°, R2 (-1708, -2286) yön 13°
- Planlanan görünürlük oranı: [0.903, 0.71]
- Arama barları (°): R1 [2.0, 7.0, 15.0, 25.0]; R2 [2.0, 7.0, 15.0, 25.0]
- Yanlış alarm raporu: 11

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | CA+CV | 256 / 274 / 274 | 50 / 62 / 73 | 0.8 / 0.8 / 0.8 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -10.5 / -2.9 / 2.1 | 3 {1: 3} | 5% | 0% | 1.0 / — | 20.1 / 21.6 / 59.9 | 256 / 274 / 277 | tamam |
| T1 | missile/subsonic_cruise (L2) | CV+CV+CV | 252 / 252 / 252 | 46 / 61 / 73 | 0.9 / 0.9 / 0.9 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -17.4 / -1.5 / 1.7 | 2 {1: 2} | 3% | 0% | 23.0 / — | 42.0 / 47.0 / 52.1 | 252 / 252 / 253 | tamam |

## ep-000001-5bd9a94dd1c7 (train)
- Arazi: ridge_multi_obstacle / asymmetric / rocky_mountain
- Arazi profili: **knife_ridges**, kabartı 1100 m (ayarlanan relief 1100 m, vadi 2912 m)
- Rota deseni: **diverging** (3 füze, fazlar ['out', 'out', 'out'], yönler [-22.9, 3.8, -177.4]°)
- Bölgeler: Zone A [zone] (357, 983); Zone B [zone] (5367, -876); Zone C [zone] (5847, 1363); Zone D [zone] (-5150, 666); Facility 1 [facility] (2225, 4553); Facility 2 [facility] (2807, 562) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 98, Zone B 0, Zone C 2290, Zone D 4329; T1 Zone A 33, Zone B 2203, Zone C 25, Zone D 5047; T2 Zone A 70, Zone B 2267, Zone C 1634, Zone D 0
- Radar: 1 — R1 (-2826, -1474) yön 36°
- Planlanan görünürlük oranı: [0.387, 0.935, 0.387]
- Arama barları (°): R1 [2.0, 7.0, 15.0, 25.0]
- Yanlış alarm raporu: 6

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | CV+CT | 226 / 226 / 226 | 37 / 108 / 187 | -3.6 / -3.6 / -3.6 | 0.00 / 0.00 / 1.01 | 0.0 / 0.0 / 2.5 | -6.2 / -3.7 / -1.9 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T1 | missile/subsonic_cruise (L2) | CV | 250 / 250 / 250 | 43 / 133 / 177 | 13.1 / 13.1 / 13.1 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -7.2 / -6.6 / -3.1 | 0 {} | 0% | 0% | — / — | — | — | tamam |
| T2 | missile/supersonic (L2) | CV | 648 / 648 / 648 | 117 / 401 / 424 | -1.1 / -1.1 / -1.1 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -3.7 / -3.3 / 3.2 | 5 {0: 5} | 8% | 62% | 2.0 / 7.5 | 28.4 / 40.5 / 48.9 | 104 / 464 / 605 | tamam |

## ep-000002-a5562e3d4bcb (validation)
- Arazi: basin_highlands / dense / dry_canyon
- Arazi profili: **highland_bowl**, kabartı 1307 m (ayarlanan relief 1086 m, vadi 4163 m)
- Rota deseni: **hub** (3 füze, fazlar ['in', 'out', 'out'], yönler [13.2, -166.4, 156.1]°)
- Bölgeler: Zone A [zone] (-1620, 80); Zone B [zone] (-7017, -1150); Zone C [zone] (-6726, 2187); Facility 1 [facility] (-4566, 1817); Facility 2 [facility] (1789, -638) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 186, Zone B 173, Zone C 3331; T1 Zone A 93, Zone B 90, Zone C 3175; T2 Zone A 186, Zone B 3168, Zone C 57
- Radar: 3 — R1 (-5458, -1385) yön 26°, R2 (955, -2567) yön 152°, R3 (270, 4072) yön -134°
- Planlanan görünürlük oranı: [0.419, 0.484, 0.742]
- Arama barları (°): R1 [2.0, 7.0, 15.0, 25.0]; R2 [2.0, 7.0, 15.0, 25.0]; R3 [2.0, 7.0, 15.0, 25.0]
- Yanlış alarm raporu: 18

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/supersonic (L2) | CV | 577 / 577 / 577 | 39 / 93 / 405 | 2.9 / 2.9 / 2.9 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -8.8 / -4.8 / 4.5 | 7 {0: 2, 1: 5} | 12% | 35% | 18.5 / 20.0 | 11.9 / 44.4 / 64.6 | 59 / 448 / 576 | tamam |
| T1 | missile/supersonic (L2) | CA+CV+CV | 450 / 450 / 722 | 153 / 203 / 413 | 1.5 / 1.5 / 1.5 | 0.00 / 0.00 / 0.01 | 0.0 / 0.0 / 0.0 | -5.7 / -2.4 / 1.7 | 2 {0: 1, 1: 1} | 3% | 0% | 1.5 / — | 18.9 / 27.0 / 35.0 | 384 / 535 / 685 | tamam |
| T2 | missile/supersonic (L2) | CV | 575 / 575 / 575 | 47 / 169 / 306 | 12.1 / 12.1 / 12.1 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -3.1 / -1.1 / 4.4 | 4 {1: 4} | 7% | 65% | 10.0 / 11.0 | 28.8 / 37.3 / 49.2 | 562 / 564 / 571 | tamam |

## ep-000003-2ad4bc8fcf50 (train)
- Arazi: valley_sinuous_pass / dense / grassland_valley
- Arazi profili: **twisting_gorge**, kabartı 1509 m (ayarlanan relief 1258 m, vadi 3044 m)
- Rota deseni: **diverging** (2 füze, fazlar ['out', 'out'], yönler [152.2, 144.2]°)
- Bölgeler: Zone A [zone] (-1028, 766); Zone B [zone] (-5898, 3359); Zone C [zone] (-5422, 4089); Facility 1 [facility] (4140, -2524) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T0 Zone A 25, Zone B 45, Zone C 868; T1 Zone A 132, Zone B 870, Zone C 40
- Radar: 1 — R1 (-959, 2112) yön -99°
- Planlanan görünürlük oranı: [0.355, 0.323]
- Arama barları (°): R1 [2.0, 7.0, 15.0, 25.0]
- Yanlış alarm raporu: 8

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/subsonic_cruise (L2) | CV | 224 / 224 / 224 | 72 / 150 / 198 | -3.3 / -3.3 / -3.3 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -6.1 / -5.5 / 0.0 | 2 {0: 2} | 3% | 0% | 1.0 / — | 43.5 / 52.0 / 60.4 | 57 / 105 / 153 | tamam |
| T1 | missile/subsonic_cruise (L2) | CA | 200 / 217 / 233 | 43 / 144 / 174 | -0.6 / -0.6 / -0.6 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -5.3 / -3.7 / 3.1 | 1 {0: 1} | 2% | 0% | 1.0 / — | 48.5 / 48.5 / 48.5 | 179 / 179 / 179 | tamam |

## ep-000004-1ef44b816c84 (test)
- Arazi: mountain_chokepoint / dense / rocky_mountain
- Arazi profili: **alpine_chokepoint**, kabartı 1231 m (ayarlanan relief 911 m, vadi 3397 m)
- Rota deseni: **hub** (2 füze, fazlar ['in', 'out'], yönler [-26.9, 129.4]°)
- Bölgeler: Zone A [zone] (-910, 121); Zone B [zone] (-4576, 4294) — arazi içinde: True
- Füze–bölge en yakın geçiş (m): T1 Zone A 44, Zone B 2105; T2 Zone A 236, Zone B 0
- Radar: 2 — R1 (-4748, 2239) yön -10°, R2 (5366, -2362) yön 151°
- Planlanan görünürlük oranı: [1.0, 0.806, 0.387]
- Arama barları (°): R1 [2.0, 7.0, 15.0, 25.0]; R2 [2.0, 7.0, 15.0, 25.0]
- Yanlış alarm raporu: 10

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | aircraft/transport (L1) | CT+CV | 127 / 127 / 127 | 1111 / 2304 / 2399 | 7.8 / 7.8 / 7.8 | 0.00 / 0.00 / 0.60 | 0.0 / 0.0 / 2.7 | 16.7 / 21.8 / 25.4 | 9 {0: 6, 1: 3} | 13% | 50% | 13.5 / 15.5 | 13.6 / 61.8 / 79.3 | 32 / 51 / 87 | tamam |
| T1 | missile/subsonic_cruise (L2) | CV | 259 / 259 / 259 | 49 / 100 / 149 | 1.0 / 1.0 / 1.0 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -9.7 / -9.3 / -2.3 | 8 {0: 8} | 13% | 53% | 13.0 / 14.5 | 26.9 / 47.7 / 58.6 | 257 / 258 / 259 | tamam |
| T2 | missile/supersonic (L2) | CV+CV | 634 / 634 / 634 | 115 / 234 / 360 | -8.5 / -8.5 / -8.5 | 0.00 / 0.00 / 0.00 | 0.0 / 0.0 / 0.0 | -4.2 / -2.1 / 1.8 | 2 {0: 1, 1: 1} | 3% | 0% | 1.0 / — | 35.5 / 38.4 / 41.2 | 430 / 509 / 587 | tamam |

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
