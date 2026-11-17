# Öğrenen ajanın gözlem gereksinimleri

Kişi 2 haftalık teslimi (İP2–İP3 → İP5 bağlantısı). Bu belge, öğrenen ajana
verilecek bilgilerin (iz belirsizliği, radar kaynak durumu, tespit kalitesi
vb.) **ne olması gerektiğini**, **kodda nereden geldiğini** ve **doğru ve
tutarlı üretildiğinin nasıl sınandığını** tanımlar.

> **Not (MAPPO / merkezi RL):** Görev listesinde "MAPPO ajanlarının
> gözlemleri" geçiyor. 2209-B önerisinin son hâlinde (danışmanın isteğiyle) RL,
> füzyon merkezinde **tek bir görev yöneticisi**dir; radarların kendi
> görevlerini öğrendiği çok etmenli yapı gelecek çalışmadır. Bu nedenle
> belge iki gözlemi birlikte ele alır: (1) mevcut ajan gözlemi
> (`RadarNetEnv._observe`, MAPPO/IPPO/central_ppo bunu kullanıyor) ve
> (2) görev yöneticisi gözlemi (`src/core/observation.py`, önerinin
> Bölüm 2.6'sındaki o_t). Doğrulama testleri ikisini de kapsar.

## 1. Temel ilkeler

1. **Gerçek durum kullanılmaz.** Gözlem yalnızca füzyon merkezinin gerçekten
   sahip olduğu bilgilerden üretilir: bütünleşik iz tablosu (kestirim ve
   kovaryans), radarların yayımlanmış parametreleri ve kaynak raporları,
   yerel zamanlayıcıların cevapları. Test: hedeflerin gerçek konumu
   değiştirildiğinde gözlem değişmez (`test_observations_never_use_truth`).
2. **Sabit boyut.** İz sayısı değişse de gözlem boyutu sabittir: en fazla K
   iz (senaryodaki `max_track_slots`), boş satırlar sıfır ve maskeli.
3. **Aynı sıra.** "k. satır", her radar ve her adımda aynı izi gösterir
   (onaylı izler önce, sonra en belirsiz olanlar; `env.slots`).
4. **Ölçekleme.** Her özellik yaklaşık [0, 1] (yönler için [−1, 1])
   aralığına ölçeklenir; ölçek sabitleri aşağıdaki tablolarda.
5. **Eksik bilgi açıkça işaretlenir.** Bir değer yoksa varsayılan değerle
   doldurulur ve bir geçerlilik maskesiyle işaretlenir (Bölüm 5).

## 2. Mevcut ajan gözlemi (`RadarNetEnv._observe`)

Her radar için: K iz satırı × 12 özellik + 12 kendi-durum özelliği
(nominal senaryoda K = 6 → 84 boyut).

| # | İz satırı özelliği | Kaynak |
|---|---|---|
| 0 | satır dolu | `env.slots` |
| 1 | menzil / r_max | kestirimden, `geometry.measure` |
| 2–3 | sin, cos(azimut) | aynı |
| 4 | elevasyon (rad) | aynı |
| 5 | radyal hız / 300 | aynı |
| 6 | log10(1 + σ_konum) / 3 | `ekf.pos_std` (iz belirsizliği) |
| 7 | onaylı | `track.confirmed` |
| 8 | son güncellemeden geçen süre / 10 s | `last_update_t` (bilgi yaşı) |
| 9 | radarın görüş alanında | azimut ve menzil |
| 10 | tahmini SNR / 30 dB (0,1 m² referans hedef) | `snr_linear` |
| 11 | en olası sınıfın olasılığı | `class_probs` |

Kendi durum: enerji, ısı payı, kalan zaman, boş takip kanalı, gönderim
durumu, son bakış süresi (Eq. 57), mod (4'lü one-hot), çevrimiçi, bölüm
ilerlemesi.

**Doğrulandı:** iz satırlarının geometrisi, belirsizlik ve bilgi yaşı iz
tablosuyla birebir; kaynak özellikleri radar durumuyla birebir ve [0, 1]
içinde; tüm radarlar aynı satır sırasını görüyor; gerçek durum kullanılmıyor
(`tests/test_observation.py`).

**Eksikler (görev yöneticisi gözleminde giderildi):**

* Belirsizlik tek bir sayı (σ_konum); hız belirsizliği ve kovaryansın
  büyüklüğü (log det P) yok.
* İzin tutarlılığı (NIS), yaşı, kaç radarın güncellediği ve sınıf
  belirsizliği (entropi) yok.
* "Görüş alanında" bayrağı elevasyonu ve en kısa menzili hesaba katmıyor.
* Tahmini SNR sabit bir referans hedef içindir (merkez gerçek RCS'yi
  bilmez); bu bilerek böyle, ama bir güncellemenin **ne kadar bilgi
  kazandıracağı** gözlemde yok.
* Radarların talep cevapları (kabul/ret) ve kuyruk durumu yok — talep
  tabanlı yapı için gerekli.

## 3. Görev yöneticisi gözlemi (`src/core/observation.py`)

o_t = [ {τ_j}, {ρ_i}, Q_t, maskeler ]. Boyut: K·17 + N·16 + N·K·5 + K + N·K + 1
(nominal K = 6, N = 4: **317**; multi_target K = 12, N = 4: **569**).
Bir gözlem ~5 ms'de üretiliyor (K = 12).

### 3.1 İz özeti τ_j (17 özellik)

| Özellik | Ölçek | Kaynak |
|---|---|---|
| satır dolu | 0/1 | `env.slots` |
| konum x, y, z | / alan yarıçapı, / 5000 m | `track.x` |
| hız vx, vy, vz | / 300 m/s | `track.x` |
| log10(1 + σ_konum) | / 3 | `ekf.pos_std` |
| log det P_konum | / 50 | `ekf.P` |
| onaylı | 0/1 | M-of-N onayı |
| iz yaşı | / 60 s, kırpılmış | `born_t` |
| bilgi yaşı (AoI) | / 10 s, kırpılmış | `last_update_t` |
| son güncellemenin NIS'i | / 13,28 (χ²₄ %99), 2'de kırpılmış | `ekf.last_nis` (maske: `nis_valid`) |
| isabet sayısı | / 20, kırpılmış | `track.hits` |
| son 3 s'de güncelleyen radar sayısı | / N | `updated_by` |
| sınıf entropisi | / log C (1 = bilinmiyor) | `class_probs` |
| bekleyen talep var | 0/1 | zamanlayıcı kuyrukları |

### 3.2 Radar özeti ρ_i (16 özellik)

| Özellik | Ölçek | Kaynak |
|---|---|---|
| çevrimiçi | 0/1 | `RadarState.online` |
| konum x, y; sin/cos(yön) | / alan yarıçapı | `RadarParams` |
| enerji, ısı payı, boş takip kanalı | 0–1 | `RadarState` (Eq. 54–57) |
| zamanlayıcı kuyruğu | / c_max | `LocalScheduler.queue` |
| son adımdaki cevaplar: kabul, gecikmeli, kısmi, ret | oran (toplam 1; talep yoksa 0) | `env.last_answers` |
| merkeze ulaşan rapor sayısı | / 20, kırpılmış | `env.last_report_counts` (iletişim durumu) |
| şu an takip bakışında | 0/1 | `RadarState.mode` |
| takip bakışına ayrılan adım oranı | 0–1 | `env.track_load` |

### 3.3 Hizmet kalitesi Q_ij (5 özellik, radar × iz)

"Radar i, iz j'yi şimdi tam takip bakışıyla güncellerse ne olur?"

| Özellik | Ölçek | Hesap |
|---|---|---|
| kapsanabilir | 0/1 (maske) | menzil, azimut, elevasyon, çevrimiçi |
| tahmini SNR (0,1 m² referans) | / 30 dB | radar denklemi (Eq. 27) |
| tahmini çapraz menzil σ | / 1000 m | r · σ_az (Eq. 42) |
| beklenen bilgi kazancı ½(log det P − log det P⁺) | / 10 nat | P⁺ = (P⁻¹ + Hᵀ R⁻¹ H)⁻¹ |
| bir takip bakışının enerji maliyeti | / E_max | kaynak modeli |

Bilgi kazancı, önerideki ödülün ana terimi (β_I) ile aynı büyüklüktür;
ajan böylece hangi talebin ödülü en çok artıracağını gözlemden
çıkarabilir.

## 4. Doğrulama (`tests/test_observation.py`)

| Test | Ne gösteriyor |
|---|---|
| `test_agent_observation_slots_match_the_track_table` | menzil, açı, belirsizlik, onay ve bilgi yaşı iz tablosuyla birebir |
| `test_agent_resource_features_follow_the_radar_state` | enerji ve kanal özellikleri radar durumuyla birebir, [0, 1] içinde |
| `test_every_agent_sees_the_same_slot_order` | k. satır her radar için aynı iz |
| `test_observations_never_use_truth` | gerçek hedefler değişince iki gözlem de değişmiyor |
| `test_central_observation_shapes_masks_and_ranges` | boyutlar, maskeler, aralıklar; boş satırlar sıfır |
| `test_quality_mask_matches_coverage` | kalite maskesi = kapsama |
| `test_information_gain_larger_for_uncertain_tracks` | belirsiz iz → daha büyük beklenen bilgi kazancı (> 0) |
| `test_last_answers_and_nis_are_reported` | cevap oranları ve NIS doğru aktarılıyor |
| `test_track_features_follow_the_tracker` | iz özetinin özellikleri izleyiciyle tutarlı |

Ayrıca gözlemin dayandığı büyüklükler kendi testleriyle doğrulandı:
ölçüm kovaryansı ve P_D (`radar_physics_report`, 21/21), izleyici
tutarlılığı (NEES, `test_tracker.py`), zamanlayıcı cevapları
(`test_scheduler.py`).

## 5. Bir radar güven bilgisi göndermezse

Simülasyonda her rapor kovaryans ve SNR içeriyor; gerçek bir ağda bu bilgi
eksik olabilir. Kural: **varsayılanla doldur, maskeyle işaretle, uydurma.**

| Eksik bilgi | Varsayılan | İşaret |
|---|---|---|
| raporun ölçüm kovaryansı R | radarın yayımlanmış doğruluğu: tahmini SNR'de `measurement_sigmas` | rapor bayrağı `cov_reported = 0` (öneri) |
| raporun SNR / kalite skoru | 0 (en düşük kalite) | aynı bayrak |
| sınıf olasılıkları | entropi = 1 (bilinmiyor) | — |
| NIS (iz yeni başlatıldı) | 0 | `nis_valid = 0` |
| radarın kaynak raporu (enerji, ısı) | son bilinen değer | ayrıca "raporun yaşı" özelliği eklenmeli (öneri) |

Kodda şu an yalnızca son üç satır uygulanıyor; ilk ikisi ve kaynak
raporunun yaşı, gerçek bir ağ arayüzü tanımlandığında eklenecek.

## 6. Açık konular / sonraki adım

* **Aksiyon:** önerideki gibi önce yalnızca güncelleme talepleri (U_t);
  her radar için tek talep veya radar–iz öncelik puanı + Macar ataması.
  Ortam tarafı hazır: `env.step_requests` ve `task_greedy` baseline'ı.
* **Ödül:** Q_ij'deki bilgi kazancı, τ_j'deki bilgi yaşı ve kesinti /
  iz silme olayları önerideki ödülün terimleriyle birebir; ödül bu
  büyüklüklerden hesaplanmalı.
* **K seçimi:** Gözlem boyutu K ile doğrusal büyüyor (N·K·5 terimi);
  12 iz için 569. Daha büyük sahnelerde iz başına ortak ağırlıklı bir
  kodlayıcı (ör. attention / DeepSets) düşünülebilir.
* **Normalizasyon:** Ölçekler bu senaryolara göre seçildi; farklı alan
  boyutlarında yeniden kontrol edilmeli.
