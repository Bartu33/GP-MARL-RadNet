# Hedef Sınıflandırması: İHA, Uçak, Füze

> Bu dosya `scripts/characterize_target_classes.py` tarafından üretilir. Parametreleri değiştirmek için `src/core/target_classes.py` dosyasını düzenleyip scripti yeniden çalıştırın. (örnek sayısı / varyant = 4000, seed = 0)

## 1. Radarın ölçtüğü büyüklükler

Simülatördeki radar (`src/core/radar.py`) her tespit için şu değerleri üretir; sınıflandırma bu büyüklükler ve EKF'nin bunlardan türettiği durum üzerine kuruludur.

| Büyüklük | Kaynak / denklem | Sınıf bilgisi taşır mı? |
|---|---|---|
| Menzil r | `geometry.measure`, σ_r = max(5 m, (c/2B)/√(2·SNR)) | Dolaylı: tespit menzili RCS'e bağlı |
| Azimut α | `geometry.measure`, σ_α = max(0.2 mrad, θ_az/√(2·SNR)) | Hayır (yalnızca konum) |
| Yükseliş θ | `geometry.measure` | Evet → irtifa = h_radar + r·sin θ |
| Radyal hız ṙ | `geometry.measure`, σ_ṙ = max(0.2, λ/(2·T_dwell)/√(2·SNR)) | Evet → hız alt sınırı |
| Doppler f_D | f_D = −2ṙ/λ | ṙ ile aynı bilgi, Hz cinsinden |
| SNR | radar denklemi χ = P_t G² λ² σ n / ((4π)³ R⁴ L k T₀ B F) | Evet → R biliniyorsa σ̂ (RCS) geri hesaplanır |
| P_D, sinyal kalitesi | Marcum Q (P_FA = 1e-6), (SNR−10)/(25−10) | SNR'ın fonksiyonu |
| Kör hız | katlanmış f_D o look'un PRF'inde çentik içindeyse tespit yok (`doppler_blind`) | Evet: ṙ ≈ 0 (havada asılı İHA, teğetsel uçuş) |
| Türetilen (iz) | EKF durumu x = [p, v] | \|v\|, v_z, ivme, dönüş hızı, irtifa |

Varsayılan radar: f_c = 10 GHz (λ = 3.0 cm), P_t = 1000 W, G = 36 dBi, B = 1 MHz (Δr = 150 m), F = 4 dB, L = 5 dB, r_max = 14 km, arama 32 darbe / takip 384 darbe (dwell 2 / 24 ms), nominal PRF 16 kHz (MPRF), look başına PRF stagger 14, 16, 18 kHz, clutter çentiği ±1 m/s.

## 2. Sınıflar ve varyantlar (fiziksel zarf)

Her sınıfın gerçek platformlardaki farkı yansıtan iki varyantı var. Değerler belirli bir platformun değil, platform kategorisinin açık literatürdeki genel zarfıdır. Aralıklar: `sert sınır` (tipik bant). Örneklerin %85'i tipik banttan çekilir.

| Sınıf / varyant | Hız (m/s) | İrtifa AGL (m) | max \|v_z\| (m/s) | max boyuna ivme (m/s²) | max yanal ivme (m/s²) | max dönüş (rad/s) | Swerling |
|---|---|---|---|---|---|---|---|
| **UAV** / multirotor | 0 – 20 (2 – 12) | 30 – 400 (40 – 150) | 8 | 5 | 8 (≈0.8 g) | 1 (57°/s) | 1 |
| **UAV** / fixed_wing | 18 – 62 (22 – 45) | 100 – 1500 (200 – 800) | 5 | 3 | 10 (≈1.0 g) | 0.35 (20°/s) | 1 |
| **Aircraft** / transport | 80 – 200 (100 – 160) | 300 – 4000 (800 – 3000) | 15 | 2 | 6 (≈0.6 g) | 0.06 (3°/s) | 1 |
| **Aircraft** / fighter | 120 – 320 (180 – 280) | 100 – 4000 (150 – 2000) | 60 | 10 | 80 (≈8.2 g) | 0.35 (20°/s) | 1 |
| **Missile** / subsonic_cruise | 200 – 280 (220 – 260) | 30 – 200 (30 – 100) | 20 | 5 | 40 (≈4.1 g) | 0.15 (9°/s) | 3 |
| **Missile** / supersonic | 450 – 950 (550 – 800) | 30 – 800 (30 – 200) | 50 | 30 | 150 (≈15.3 g) | 0.2 (11°/s) | 3 |
| **Missile** / terminal_dive | 450 – 1100 (600 – 950) | 30 – 15000 (3000 – 12000) | 1000 | 60 | 300 (≈30.6 g) | 0.3 (17°/s) | 3 |

### Görünüşe bağlı ortalama RCS (X-bandı, dBsm, örnekleme aralığı)

| Sınıf / varyant | Burun | Yan | Kuyruk | Üst/alt | Açıklama |
|---|---|---|---|---|---|
| UAV / multirotor | -22 – -12 | -20 – -10 | -22 – -12 | -15 – -5 | Small multirotor (quadcopter / hexacopter, < 25 kg) |
| UAV / fixed_wing | -15 – -5 | -5 – 3 | -15 – -5 | 0 – 8 | Small/tactical fixed-wing UAV (2-6 m span, piston/electric) |
| Aircraft / transport | 10 – 18 | 15 – 25 | 10 – 18 | 18 – 28 | Transport / airliner / large turboprop |
| Aircraft / fighter | 0 – 7 | 7 – 15 | 3 – 10 | 10 – 18 | Fighter / jet trainer / attack aircraft |
| Missile / subsonic_cruise | -20 – -10 | -3 – 3 | -12 – -5 | -2 – 5 | Subsonic cruise missile, terrain following |
| Missile / supersonic | -12 – -5 | 0 – 6 | -8 – -2 | 2 – 8 | Supersonic missile (ramjet / rocket), low-altitude dash |
| Missile / terminal_dive | -12 – -5 | 0 – 6 | -8 – -2 | 2 – 8 | Supersonic manoeuvring missile ending in a steep terminal dive (high-altitude dive or terrain-following pop-up attack) |

## 3. Ölçüm uzayındaki sınıf bantları (Monte Carlo)

Hedefler tek bir radarın (yükseklik 15 m) ±60° sektörüne, 0.5–14 km yer menziline alan-üniform yerleştirildi; yön rastgele, yanal ivme ±%30·a_max. Görünmeyen (r > r_max, θ > 60°) örnekler atıldı. SNR hüzme merkezinde (en iyi durum) hesaplandı; her örnek stagger setinden rastgele bir PRF görür ve kör hıza düşerse P_D = 0 sayılır. Hücreler: **p5 / medyan / p95**; oranlar için ortalama.

### 3.1 Sınıf düzeyi

| Ölçüm | Birim | UAV | Aircraft | Missile |
|---|---|---|---|---|
| Yatay hız \|v\| | m/s | 3 / **19** / 44 | 105 / **168** / 273 | 223 / **366** / 877 |
| İrtifa (AGL) | m | 52 / **235** / 788 | 336 / **1471** / 2936 | 36 / **91** / 7893 |
| Dikey hız v_z | m/s | -4.5 / **+0.0** / +4.5 | -33.1 / **-0.0** / +32.9 | -287.1 / **+0.4** / +273.4 |
| Menzil r | km | 3.3 / **10.0** / 13.6 | 3.7 / **10.0** / 13.7 | 3.6 / **10.2** / 13.7 |
| Yükseliş açısı θ | ° | 0.2 / **1.4** / 8.0 | 2.0 / **9.0** / 28.1 | 0.1 / **0.5** / 47.0 |
| Radyal hız \|ṙ\| | m/s | 1 / **8** / 37 | 13 / **104** / 237 | 26 / **217** / 709 |
| Doppler \|f_D\| | kHz | 0.04 / **0.55** / 2.49 | 0.84 / **6.94** / 15.78 | 1.74 / **14.50** / 47.30 |
| Ortalama RCS (görünüş) | dBsm | -17.5 / **-9.8** / 0.9 | 6.9 / **15.0** / 22.7 | -9.0 / **-0.2** / 5.4 |
| Anlık RCS (Swerling) | dBsm | -25.5 / **-11.4** / 1.4 | -0.1 / **12.6** / 23.7 | -12.1 / **-1.5** / 6.3 |
| SNR – arama | dB | 2.4 / **18.9** / 40.3 | 27.1 / **42.6** / 61.9 | 15.3 / **27.4** / 45.6 |
| SNR – takip | dB | 13.2 / **29.7** / 51.1 | 37.9 / **53.4** / 72.7 | 26.1 / **38.2** / 56.4 |
| P_D – arama | - | 0.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 0.99 / **1.00** / 1.00 |
| P_D – takip | - | 0.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 |
| Kör hız (çentik) oranı | - | **0.071** | **0.009** | **0.011** |
| σ_r – takip | m | 5.0 / **5.0** / 23.2 | 5.0 / **5.0** / 5.0 | 5.0 / **5.0** / 5.3 |
| σ_ṙ – takip | m/s | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 |
| σ_ṙ – arama | m/s | 0.20 / **0.60** / 4.01 | 0.20 / **0.20** / 0.23 | 0.20 / **0.23** / 0.91 |

### 3.2 Varyant düzeyi

| Ölçüm | Birim | UAV/multirotor | UAV/fixed_wing | Aircraft/transport | Aircraft/fighter | Missile/subsonic_cruise | Missile/supersonic | Missile/terminal_dive |
|---|---|---|---|---|---|---|---|---|
| Yatay hız \|v\| | m/s | 2 / **7** / 13 | 23 / **34** / 46 | 101 / **131** / 160 | 181 / **230** / 278 | 221 / **239** / 259 | 555 / **677** / 798 | 605 / **769** / 943 |
| İrtifa (AGL) | m | 46 / **100** / 247 | 228 / **522** / 1055 | 873 / **1891** / 2975 | 247 / **1141** / 2486 | 34 / **69** / 146 | 39 / **127** / 545 | 2781 / **5750** / 10378 |
| Dikey hız v_z | m/s | -5.6 / **+0.1** / +5.5 | -3.4 / **-0.0** / +3.5 | -9.9 / **-0.1** / +9.9 | -40.7 / **-0.3** / +41.7 | -13.5 / **+0.0** / +14.0 | -34.1 / **-0.1** / +33.6 | -418.2 / **+9.5** / +424.5 |
| Menzil r | km | 3.1 / **10.2** / 13.6 | 3.3 / **9.8** / 13.6 | 3.8 / **10.1** / 13.7 | 3.5 / **10.0** / 13.7 | 3.0 / **9.8** / 13.6 | 3.3 / **9.9** / 13.7 | 6.5 / **11.5** / 13.8 |
| Yükseliş açısı θ | ° | 0.2 / **0.5** / 2.0 | 1.1 / **3.1** / 10.1 | 4.6 / **11.3** / 32.5 | 1.4 / **6.9** / 23.5 | 0.1 / **0.3** / 1.2 | 0.1 / **0.7** / 3.6 | 14.0 / **34.4** / 56.7 |
| Radyal hız \|ṙ\| | m/s | 0 / **4** / 11 | 3 / **23** / 42 | 10 / **84** / 143 | 16 / **153** / 252 | 18 / **167** / 247 | 48 / **458** / 741 | 43 / **399** / 826 |
| Doppler \|f_D\| | kHz | 0.03 / **0.27** / 0.72 | 0.17 / **1.50** / 2.77 | 0.66 / **5.59** / 9.51 | 1.09 / **10.22** / 16.84 | 1.21 / **11.16** / 16.46 | 3.20 / **30.54** / 49.45 | 2.86 / **26.65** / 55.11 |
| Ortalama RCS (görünüş) | dBsm | -18.4 / **-14.2** / -10.3 | -9.2 / **-3.4** / 1.8 | 14.9 / **18.8** / 23.6 | 6.0 / **10.6** / 15.1 | -11.4 / **-2.5** / 2.0 | -5.0 / **1.7** / 6.1 | -3.1 / **2.6** / 6.1 |
| Anlık RCS (Swerling) | dBsm | -27.9 / **-16.1** / -8.1 | -17.9 / **-5.5** / 3.2 | 5.4 / **17.2** / 25.3 | -2.7 / **8.9** / 16.8 | -14.3 / **-4.0** / 3.4 | -8.6 / **0.4** / 7.4 | -6.6 / **1.5** / 7.7 |
| SNR – arama | dB | -0.7 / **13.1** / 33.0 | 10.1 / **24.1** / 43.7 | 32.5 / **46.1** / 64.6 | 24.3 / **38.4** / 57.9 | 13.2 / **25.7** / 46.3 | 18.0 / **30.0** / 49.3 | 18.8 / **28.2** / 39.5 |
| SNR – takip | dB | 10.1 / **23.8** / 43.8 | 20.9 / **34.9** / 54.5 | 43.3 / **56.9** / 75.4 | 35.1 / **49.2** / 68.7 | 24.0 / **36.5** / 57.1 | 28.8 / **40.8** / 60.1 | 29.6 / **39.0** / 50.3 |
| P_D – arama | - | 0.00 / **0.64** / 1.00 | 0.06 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 0.77 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 |
| P_D – takip | - | 0.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 | 1.00 / **1.00** / 1.00 |
| Kör hız (çentik) oranı | - | **0.115** | **0.020** | **0.005** | **0.010** | **0.012** | **0.009** | **0.009** |
| σ_r – takip | m | 5.0 / **6.8** / 33.2 | 5.0 / **5.0** / 9.6 | 5.0 / **5.0** / 5.0 | 5.0 / **5.0** / 5.0 | 5.0 / **5.0** / 6.7 | 5.0 / **5.0** / 5.0 | 5.0 / **5.0** / 5.0 |
| σ_ṙ – takip | m/s | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.20 |
| σ_ṙ – arama | m/s | 0.20 / **1.18** / 5.76 | 0.20 / **0.33** / 1.66 | 0.20 / **0.20** / 0.20 | 0.20 / **0.20** / 0.32 | 0.20 / **0.27** / 1.16 | 0.20 / **0.20** / 0.67 | 0.20 / **0.21** / 0.61 |
| Sektörde görünür oran | - | 1.00 | 1.00 | 0.97 | 0.99 | 1.00 | 1.00 | 0.56 |

## 4. Tespit menzili (P_D = 0.9, dalgalanmasız, hüzme merkezi)

Varyantın burun ve yan RCS aralıklarının orta noktası kullanıldı. r_max = 14 km'nin üstündeki değerler radar menziliyle sınırlıdır.

Gerekli SNR (P_D = 0.9, P_FA = 1e-06): 13.2 dB.

| Varyant | Burun: σ → R_arama / R_takip (km) | Yan: σ → R_arama / R_takip (km) |
|---|---|---|
| UAV/multirotor | -17.0 dBsm → 9.1 / 17.0 | -15.0 dBsm → 10.2 / 19.0 |
| UAV/fixed_wing | -10.0 dBsm → 13.6 / 25.4 | -1.0 dBsm → 22.9 / 42.6 |
| Aircraft/transport | +14.0 dBsm → 54.3 / 101.1 | +20.0 dBsm → 76.8 / 142.8 |
| Aircraft/fighter | +3.5 dBsm → 29.7 / 55.3 | +11.0 dBsm → 45.7 / 85.1 |
| Missile/subsonic_cruise | -15.0 dBsm → 10.2 / 19.0 | +0.0 dBsm → 24.3 / 45.2 |
| Missile/supersonic | -8.5 dBsm → 14.9 / 27.7 | +3.0 dBsm → 28.8 / 53.7 |
| Missile/terminal_dive | -8.5 dBsm → 14.9 / 27.7 | +3.0 dBsm → 28.8 / 53.7 |

## 5. Ölçüm başına ayırt edici kurallar

Yukarıdaki bantlardan okunan, sınıflandırıcının kullanması beklenen fiziksel ipuçları:

- **Hız / Doppler:** |ṙ| ≤ |v| olduğundan ṙ hızın *alt sınırıdır*. |v| ≤ 50 m/s yalnızca İHA, |v| > 320 m/s yalnızca (süpersonik) füze. 200–280 m/s bandında uçak (avcı) ile seyir füzesi örtüşür.
- **RCS (SNR + menzilden):** Uçak ≥ ~0 dBsm, seyir füzesi burundan −20…−10 dBsm, çok rotorlu İHA −22…−5 dBsm. Aynı hız bandındaki avcı ile füzeyi ayıran ana büyüklük RCS'tir (≈15–20 dB fark).
- **İrtifa (θ ve r'den):** Seyir füzesi 30–200 m AGL'de arazi takibi yapar (θ < ~2°); avcı alçak uçuşta 100 m'ye inebilir ama tipik bandı 150–2000 m. Çok rotorlu İHA ≤ 400 m, sabit kanatlı İHA ≤ 1.5 km, uçaklar ≤ 4 km.
- **Manevra (iz):** Yanal ivme > 10 g yalnızca füze; ≈0.6 g (30° yatış) üstü yanal ivme nakliye uçağını dışlar. Havada asılı kalma (|v| ≈ 0) yalnızca çok rotorlu İHA.
- **Dalgalanma:** Uçak/İHA Swerling 1 (çok saçıcı), füze Swerling 3 (baskın saçıcı + küçükler); füzenin SNR'ı look'tan look'a daha az dalgalanır.

## 6. Varsayımlar ve bilinen sınırlamalar

- **Doppler / menzil belirsizliği:** nominal PRF 16 kHz tek başına ±λ·PRF/4 = ±120 m/s hız ve c/(2·PRF) = 9.4 km menzil belirsizliği verir. Radar look'tan look'a PRF değiştirir (stagger) ve belirsizlikleri bu set üzerinden (Çin kalan teoremi) çözer; bu yüzden raporlar katlanmamış r ve ṙ taşır. Çözümleme hataları (hayalet hedef) modellenmez.
- **Kör hızlar modelleniyor:** katlanmış Doppler o look'un PRF'inde clutter çentiğine düşerse hedef o look'ta görülmez. Stagger sayesinde bir sonraki look'ta görünür; yalnızca ṙ ≈ 0 olan hedefler (havada asılı İHA, radara teğet uçan hedef) her PRF'te kördür.
- **Menzil kör bölgeleri (eclipsing)** ve clutter genliği (CNR) modellenmez; çentik ikili bir maskedir.
- **Mikro-Doppler yok.** Rotor/pervane modülasyonu (İHA'nın en güçlü gerçek ayırıcısı) bu algılama seviyesindeki modelde bulunmaz.
- RCS tablosu `LookupTableRCS.synthetic` ile burun/yan/kuyruk/üst değerleri arasında yumuşak geçişli, ±1.5 dB dalgacıklı sentetik bir diyagramdır; gerçek ölçülmüş bir diyagram değildir.
- Monte Carlo'da arazi düz, tek radar ve hüzme merkezinde SNR (arama hüzmesi kayıpları dahil değil).

## 7. Kaynaklar

- M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001 — tipik mikrodalga RCS değerleri (küçük uçak ~1 m², avcı 2–6 m², yolcu jeti 20–100 m², kanatlı füze ~0.5 m²); Swerling modelleri.
- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., 2014 — radar denklemi, Doppler, PRF belirsizliği.
- [PN] Proje içi not: *Physics-Informed Simulation of Three-Dimensional Radar Detections* (denklem numaraları kodda).
- M. N. Alpdemir, *Tactical UAV path optimization under radar threat using deep RL*, NCA 2022.
- Küçük İHA X-bandı RCS ölçümleri (açık literatür, ör. çok rotorlu İHA'lar için −25…−10 dBsm aralığı).
