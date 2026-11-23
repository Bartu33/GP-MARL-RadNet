# Hedef sınıfları pilot dataset denetimi

Dataset: `C:\Users\tahau\Desktop\marl\demo\data\processed\missile_dive_pilot10`  
Tüm kontrolleri geçen hedef: **10 / 10**

Değerler `min / medyan / max`. Bantlar `docs/target_classification.md` ve `src/core/target_classes.py` ile aynı kaynaktan.

## ep-000000-cc731c21506b (train)
- Arazi: rolling_hills / open / None
- Radar: 1 — R1 (-1225, -3566) yön 84°
- Planlanan görünürlük oranı: [0.955] (radar başına [0.955])
- Dalış profili: **steep_dive**, yaklaşma yönü -92°, başlangıç AGL 10704 m, süre 21.5 s, korunan nokta [-475.3, 184.0]
- Arama barları (°): R1 [11.7, 17.7, 23.7, 29.7, 35.7, 41.7]
- Yanlış alarm raporu: 8

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 780 / 874 / 891 | 253 / 8725 / 10944 | -778.5 / -692.8 / 0.0 | 0.00 / 0.00 / 15.30 | 0.0 / 0.0 / 0.0 | -5.0 / -1.3 / 1.3 | 6 {0: 6} | 14% | 21% | 16.5 / 17.5 | 12.5 / 32.9 / 51.9 | 528 / 752 / 787 | tamam |

## ep-000001-613927b7332a (train)
- Arazi: rolling_hills / open / None
- Radar: 3 — R1 (4020, -845) yön 141°, R2 (2270, 4405) yön 169°, R3 (-1980, -3345) yön 101°
- Planlanan görünürlük oranı: [1.0] (radar başına [0.975, 0.825, 1.0])
- Dalış profili: **shallow_dive**, yaklaşma yönü -47°, başlangıç AGL 5620 m, süre 19.5 s, korunan nokta [20.2, 904.7]
- Arama barları (°): R1 [3.0, 9.0, 15.0, 21.0]; R2 [3.7, 9.7, 15.7, 21.7, 27.7, 33.7]; R3 [4.3, 10.3, 16.3, 22.3, 28.3]
- Yanlış alarm raporu: 22

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive | 669 / 688 / 688 | 337 / 4207 / 5620 | -384.1 / -384.1 / 0.0 | 0.00 / 0.00 / 13.78 | 0.0 / 0.0 / 0.0 | -4.9 / -0.1 / 5.0 | 10 {0: 4, 1: 3, 2: 3} | 23% | 62% | 7.0 / 8.0 | 10.8 / 25.5 / 40.7 | 478 / 606 / 673 | tamam |

## ep-000002-13615c5a0596 (validation)
- Arazi: rolling_hills / open / None
- Radar: 2 — R1 (3574, -2280) yön 116°, R2 (2574, -6280) yön 120°
- Planlanan görünürlük oranı: [0.938] (radar başına [0.938, 0.729])
- Dalış profili: **jink_dive**, yaklaşma yönü -80°, başlangıç AGL 8619 m, süre 23.5 s, korunan nokta [74.5, 219.8]
- Arama barları (°): R1 [2.2, 8.2, 14.2, 20.2, 26.2, 32.2]; R2 [1.5, 7.5, 13.5, 19.5, 25.5]
- Yanlış alarm raporu: 20

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+jinks | 807 / 825 / 825 | 251 / 6992 / 8619 | -573.6 / -573.6 / 0.0 | 0.00 / 9.71 / 16.10 | 0.0 / 8.6 / 13.5 | -4.2 / -0.8 / 3.9 | 11 {0: 7, 1: 4} | 23% | 64% | 8.0 / 9.0 | 16.3 / 29.0 / 54.0 | 562 / 768 / 805 | tamam |

## ep-000003-709ca7e123be (train)
- Arazi: rolling_hills / open / None
- Radar: 2 — R1 (4285, -4264) yön 84°, R2 (3285, 6486) yön 43°
- Planlanan görünürlük oranı: [0.825] (radar başına [0.684, 0.404])
- Dalış profili: **dive_pullout**, yaklaşma yönü -115°, başlangıç AGL 9406 m, süre 28 s, korunan nokta [1285.3, -14.0]
- Arama barları (°): R1 [5.2, 11.2, 17.2, 23.2, 29.2]; R2 [37.4, 43.4, 49.4, 55.4, 61.4, 67.4]
- Yanlış alarm raporu: 17

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+push_over+dive+pull_out+low_dash | 753 / 776 / 780 | 156 / 5737 / 9406 | -668.4 / -305.7 / 0.0 | 0.00 / 0.00 / 16.31 | 0.0 / 0.0 / 0.0 | -4.3 / 1.2 / 7.2 | 11 {0: 5, 1: 6} | 20% | 71% | 0.5 / 2.0 | 17.7 / 29.5 / 41.9 | 457 / 567 / 643 | tamam |

## ep-000004-244d4aaba132 (test)
- Arazi: rolling_hills / open / None
- Radar: 3 — R1 (-1858, -5212) yön 114°, R2 (4642, -5212) yön 144°, R3 (-6858, 1288) yön 90°
- Planlanan görünürlük oranı: [0.933] (radar başına [0.75, 0.583, 0.6])
- Dalış profili: **weave_dive**, yaklaşma yönü -53°, başlangıç AGL 7467 m, süre 29.5 s, korunan nokta [-357.8, -712.2]
- Arama barları (°): R1 [2.5, 8.5, 14.5, 20.5, 26.5, 32.5]; R2 [2.2, 8.2, 14.2, 20.2, 26.2]; R3 [29.8, 35.8, 41.8, 47.8, 53.8, 59.8]
- Yanlış alarm raporu: 28

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | weave+push_over+dive+jink | 783 / 783 / 884 | 185 / 7397 / 7467 | -657.7 / 0.0 / 0.0 | 0.00 / 11.80 / 15.00 | 0.0 / 8.5 / 11.0 | -3.2 / 1.2 / 7.6 | 9 {0: 4, 1: 4, 2: 1} | 14% | 41% | 16.5 / 17.5 | 19.0 / 31.1 / 48.0 | 323 / 729 / 820 | tamam |

## ep-000005-9c9cd8eaea1f (train)
- Arazi: rolling_hills / open / None
- Radar: 1 — R1 (-2605, -6378) yön 80°
- Planlanan görünürlük oranı: [0.915] (radar başına [0.915])
- Dalış profili: **popup_dive**, yaklaşma yönü -93°, başlangıç AGL 49 m, süre 23 s, korunan nokta [-1105.0, -628.4]
- Arama barları (°): R1 [1.1, 7.1]
- Yanlış alarm raporu: 10

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+pop_up+dive | 532 / 557 / 657 | 49 / 82 / 1549 | -253.0 / 3.3 / 317.3 | 0.02 / 0.21 / 26.26 | 0.0 / 0.0 / 0.0 | -3.7 / -2.8 / 3.6 | 8 {0: 8} | 17% | 43% | 12.5 / 13.5 | 18.5 / 28.3 / 44.6 | 434 / 536 / 622 | tamam |

## ep-000006-e11ca94b6d66 (train)
- Arazi: rolling_hills / open / None
- Radar: 2 — R1 (572, -2233) yön -152°, R2 (2322, 1767) yön -143°
- Planlanan görünürlük oranı: [0.974] (radar başına [0.789, 0.974])
- Dalış profili: **barrel_roll_dive**, yaklaşma yönü 51°, başlangıç AGL 8932 m, süre 18.5 s, korunan nokta [-1178.4, 517.2]
- Arama barları (°): R1 [22.5, 28.5, 34.5, 40.5, 46.5, 52.5]; R2 [9.9, 15.9, 21.9, 27.9, 33.9, 39.9]
- Yanlış alarm raporu: 15

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+barrel_roll_dive | 786 / 888 / 947 | 550 / 7626 / 8932 | -812.2 / -657.3 / 0.0 | 0.00 / 14.33 / 26.01 | 0.0 / 5.4 / 15.4 | -3.6 / -0.1 / 3.4 | 8 {0: 2, 1: 6} | 22% | 43% | 9.5 / 11.0 | 25.7 / 29.5 / 47.4 | 534 / 836 / 899 | tamam |

## ep-000007-09a6f8abb15b (train)
- Arazi: rolling_hills / open / None
- Radar: 1 — R1 (1894, -3429) yön -134°
- Planlanan görünürlük oranı: [0.592] (radar başına [0.592])
- Dalış profili: **dogleg_serpentine**, yaklaşma yönü 75°, başlangıç AGL 53 m, süre 24 s, korunan nokta [-605.8, 821.2]
- Arama barları (°): R1 [2.8, 8.8]
- Yanlış alarm raporu: 9

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | terrain_follow+dogleg_turn+serpentine+pop_up+dive | 463 / 509 / 545 | 52 / 78 / 696 | -425.1 / -7.6 / 247.2 | 0.02 / 0.86 / 26.28 | 0.0 / 0.0 / 16.8 | -5.3 / -4.0 / 6.8 | 9 {0: 9} | 19% | 58% | 0.5 / 1.5 | 16.4 / 43.3 / 58.0 | 391 / 508 / 509 | tamam |

## ep-000008-bc4b9306e07f (validation)
- Arazi: rolling_hills / open / None
- Radar: 3 — R1 (2406, -4301) yön 70°, R2 (-7344, -801) yön 4°, R3 (-844, -2051) yön 31°
- Planlanan görünürlük oranı: [0.979] (radar başına [0.83, 0.979, 0.957])
- Dalış profili: **break_turn_dive**, yaklaşma yönü -163°, başlangıç AGL 7602 m, süre 23 s, korunan nokta [-844.4, -51.0]
- Arama barları (°): R1 [32.2, 38.2, 44.2, 50.2, 56.2, 62.2]; R2 [6.6, 12.6, 18.6, 24.6, 30.6]; R3 [25.9, 31.9, 37.9, 43.9, 49.9, 55.9]
- Yanlış alarm raporu: 25

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+break_turn+boost_reacquire+dive+dive_break+terminal_dive | 634 / 726 / 812 | 351 / 7464 / 7694 | -668.0 / -202.8 / 0.0 | 0.00 / 7.46 / 26.22 | 0.0 / 8.2 / 17.3 | -5.2 / -0.2 / 3.2 | 17 {0: 8, 1: 2, 2: 7} | 35% | 70% | 1.5 / 2.5 | 14.2 / 25.2 / 52.3 | 390 / 475 / 750 | tamam |

## ep-000009-8199dc6bdc22 (train)
- Arazi: rolling_hills / open / None
- Radar: 1 — R1 (1853, 57) yön -47°
- Planlanan görünürlük oranı: [0.871] (radar başına [0.871])
- Dalış profili: **random_jink_dive**, yaklaşma yönü 139°, başlangıç AGL 4239 m, süre 15 s, korunan nokta [353.0, -193.0]
- Arama barları (°): R1 [22.6, 28.6, 34.6, 40.6]
- Yanlış alarm raporu: 8

| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | SNR dB | \|ṙ\| m/s | Kontroller |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | missile/terminal_dive (L2) | cruise+random_jinks | 756 / 757 / 865 | 497 / 4042 / 4239 | -599.8 / -122.9 / 0.0 | 0.00 / 4.52 / 26.09 | 0.0 / 0.8 / 10.2 | -5.1 / -3.3 / 3.0 | 10 {0: 10} | 33% | 90% | 1.0 / 2.0 | 19.6 / 34.8 / 39.7 | 646 / 698 / 817 | tamam |

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
