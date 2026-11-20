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

o_t = [ {τ_j}, {ρ_i}, Q_t, maskeler ]. Boyut: K·22 + N·16 + N·K·8 + K + N·K + 1
(nominal K = 6, N = 4: **419**; multi_target K = 12, N = 4: **773**).
Bütün konumlar **ortak dünya koordinatında** (ENU, kartezyen); radarlar
kendi kutupsal koordinatlarında ölçer, dönüşümü füzyon merkezindeki EKF yapar.

### 3.1 İz özeti τ_j (22 özellik)

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
| anlık SNR (son rapor) | / 30 dB | `track.snr` |
| kayan ortalama SNR (son 10 rapor, **doğrusal** ortalama, dB'ye çevrilmiş) | / 30 dB | `track.snr` |
| SNR standart sapması (dB) | / 10 dB | `track.snr` |
| istenen bakışlarda tespit oranı | 0–1 (geçmiş yoksa 0,5) | `track.dwells` |
| bakış geçmişi var | 0/1 | `track.dwells` |

**SNR neden üç değerle ve tespit oranıyla veriliyor:** Dalgalanan
(Swerling I) hedefte anlık SNR çok değişkendir (doğrusal güçte standart
sapma ortalamaya eşit), bu yüzden tek başına yanıltıcıdır; kayan ortalama
hedefin gerçek sinyal seviyesini, std ise ne kadar dalgalandığını verir.
Ortalama dB üzerinden değil doğrusal güç üzerinden alınır (dB ortalaması
düşük çıkar). SNR yalnızca **tespit edilen** bakışlarda ölçülür; kaçırılan
bakışlar düşük SNR'lıdır ama ortalamaya girmez, yani ortalama olduğundan
yüksek görünür (seçim yanlılığı). Bunu düzeltmek için istenen bakışların
kaçında hedefin bulunduğu (tespit oranı) ayrıca verilir. Medyan SNR
gözlemde değil, sonuç analizinde kullanılır (aykırı değerlere dayanıklı).
Swerling (1960); Richards (2022); genlik bilgisinin takipte kullanımı için
Lerro & Bar-Shalom (1993).
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

### 3.3 Hizmet kalitesi ve geçmiş Q_ij (8 özellik, radar × iz)

"Radar i, iz j'yi şimdi tam takip bakışıyla güncellerse ne olur?"

| Özellik | Ölçek | Hesap |
|---|---|---|
| kapsanabilir | 0/1 (maske) | menzil, azimut, elevasyon, çevrimiçi |
| tahmini SNR (0,1 m² referans) | / 30 dB | radar denklemi (Eq. 27) |
| tahmini çapraz menzil σ | / 1000 m | r · σ_az (Eq. 42) |
| beklenen bilgi kazancı ½(log det P − log det P⁺) | / 10 nat | P⁺ = (P⁻¹ + Hᵀ R⁻¹ H)⁻¹ |
| bir takip bakışının enerji maliyeti | / E_max | kaynak modeli |
| bu radarın bu izdeki son SNR'ı | / 30 dB (0 = hiç) | `track.snr[i]` |
| bu radarın bu izi en son güncellemesinden beri geçen süre | / 10 s (1 = hiç) | `track.updated_by[i]` |
| bu radarın bu izdeki istenen bakışlarda tespit oranı | 0–1 (geçmiş yoksa 0,5) | `track.dwells[i]` |

Son üç özellik "hangi radarı ne zaman tercih etmeliyiz" sorusunun
cevabıdır: hangi radar bu hedefi iyi görüyor, en son kim güncelledi,
kimin bakışları boşa gidiyor.

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

## 6. Öğrenen görev yöneticisi (`src/core/task_env.py`, `src/models/rl/ppo.py`)

Tek ajan, füzyon merkezinde, düz PPO (hiyerarşi ve çok ajan yok).

* **Karar periyodu:** 1 s (2 benzetim adımı).
* **Gözlem:** Bölüm 3'teki vektör (nominal'de 419 boyut).
* **Aksiyon:** her radar için "iz 1 … iz K" ya da "talep yok"
  (MultiDiscrete, N × (K+1)); radarın kapsamadığı izler maskelenir.
  Talepler radarların yerel zamanlayıcısına gider; radar yine kabul,
  erteleme, kısmi hizmet ya da ret kararını kendisi verir.
* **Ödül (gerçek durum kullanmaz):** + bilgi kazancı (log det P azalması)
  − bilgi yaşı − onaylı iz silinmesi + yeni onaylanan iz − reddedilen
  talep − aynı izin birden fazla radardan istenmesi − gereksiz güncelleme
  (yakın zamanda güncellenmiş ve belirsizliği düşük bir izi istemek).
  Ağırlıklar (`DEFAULT_REWARD`): bilgi kazancı 0,05; bilgi yaşı 2;
  iz silme 2; yeni iz 1; ret 0,05; çift talep 0,2; gereksiz güncelleme 0,2.
* **Neden bu ağırlıklar:** İlk denemede bilgi kazancı ağırlığı yüksekti
  (0,2) ve gereksiz güncelleme cezası yoktu. Ajan zaten iyi takip edilen
  izleri tekrar tekrar güncelledi, aramayı ihmal etti; bu ödülde rastgele
  politika ile kural tabanlı yöntem neredeyse aynı puanı alıyordu. Yeni
  ağırlıklarla kural tabanlı yöntem (+17) rastgeleden (−7) ve hiç talep
  göndermemekten (−6) açıkça ayrışıyor (nominal, 60 s, 5 bölüm).
* **Taklitle başlatma (davranış klonlama):** Ağ önce kural tabanlı görev
  yöneticisinin (`task_greedy`) kararlarını taklit eder (aktör çapraz
  entropiyle, kritik o bölümlerin getirisiyle), sonra PPO ile ince ayar
  yapılır. Algoritma yine düz PPO; yalnızca başlangıç noktası değişir.
* **Eğitim:** `python -m src.services.training_service --task --decisions 30000`
  (varsayılan 200 öğretmen bölümüyle taklitten başlar; `--bc-episodes 0`
  sıfırdan başlatır) → `models/checkpoints/ppo_task.pt`; arayüzde
  `ppo_task` olarak görünür.
* MAPPO, IPPO ve central_ppo kodda duruyor ama arayüzde artık önerilmiyor
  (radar modunu doğrudan seçtikleri için).

## 7. Açık konular / sonraki adım

* **Aksiyonun genişletilmesi:** öncelik ve son tarih seçimi, füzyonda
  radar güven katsayısı (bozulan radar senaryosuyla), PRF profili.
* **K seçimi:** Gözlem boyutu K ile doğrusal büyüyor (N·K·8 terimi);
  12 iz için 773. Daha büyük sahnelerde iz başına ortak ağırlıklı bir
  kodlayıcı (ör. attention / DeepSets) düşünülebilir.
* **Normalizasyon:** Ölçekler bu senaryolara göre seçildi; farklı alan
  boyutlarında yeniden kontrol edilmeli.
