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

o_t = [ {τ_j}, {ρ_i}, Q_t, maskeler ]. Boyut: K·22 + N·16 + N·K·9 + K + N·K + 1
(nominal K = 6, N = 4: **443**; multi_target K = 12, N = 4: **821**).
Bütün konumlar **ortak dünya koordinatında** (ENU, kartezyen); radarlar
kendi kutupsal koordinatlarında ölçer, dönüşümü füzyon merkezindeki EKF yapar.

### 3.1 İz özeti τ_j (22 özellik)

| Özellik | Ölçek | Kaynak |
|---|---|---|
| satır dolu | 0/1 | `env.slots` |
| konum x, y, z | / (2 · alan yarıçapı), / 5000 m | `track.x` |
| hız vx, vy; vz | / 1000 m/s; / 300 m/s | `track.x` |
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
| anlık SNR (son rapor) | / 80 dB | `track.snr` |
| kayan ortalama SNR (son 10 rapor, **doğrusal** ortalama, dB'ye çevrilmiş) | / 80 dB | `track.snr` |
| SNR standart sapması (dB) | / 20 dB, 1'de kırpılmış | `track.snr` |
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
| konum x, y; sin/cos(yön) | / (2 · alan yarıçapı) | `RadarParams` |
| enerji, ısı payı, boş takip kanalı | 0–1 | `RadarState` (Eq. 54–57) |
| zamanlayıcı kuyruğu | / c_max | `LocalScheduler.queue` |
| son taleplerine verdiği cevaplar: kabul, gecikmeli, kısmi, ret | oran (toplam 1; hiç talep almadıysa 0) | `env.last_answers` (yeni talep gelene kadar korunur) |
| merkeze ulaşan rapor sayısı | / 20, kırpılmış | `env.last_report_counts` (iletişim durumu) |
| şu an takip bakışında | 0/1 | `RadarState.mode` |
| takip bakışına ayrılan adım oranı | 0–1 | `env.track_load` |

### 3.3 Hizmet kalitesi ve geçmiş Q_ij (9 özellik, radar × iz)

"Radar i, iz j'yi şimdi tam takip bakışıyla güncellerse ne olur?"

| Özellik | Ölçek | Hesap |
|---|---|---|
| kapsanabilir | 0/1 (maske) | menzil, azimut, elevasyon, çevrimiçi |
| tahmini SNR (0,1 m² referans) | / 80 dB | radar denklemi (Eq. 27) |
| tahmini çapraz menzil σ | / 100 m, kırpılmış | r · σ_az (Eq. 42) |
| beklenen bilgi kazancı ½(log det P − log det P⁺) | / 15 nat | P⁺ = (P⁻¹ + Hᵀ R⁻¹ H)⁻¹ |
| bir takip bakışının enerji maliyeti | / E_max | kaynak modeli |
| bu radarın bu izdeki son SNR'ı | / 80 dB (0 = hiç) | `track.snr[i]` |
| bu radarın bu izi en son güncellemesinden beri geçen süre | / 10 s (1 = hiç) | `track.updated_by[i]` |
| bu radarın bu izdeki istenen bakışlarda tespit oranı | 0–1 (geçmiş yoksa 0,5) | `track.dwells[i]` |
| bu radarın PRF'lerinden izin kör hızına düşen beklenen oran | 0–1 | `blind_share`: izin radyal hızı ve EKF belirsizliği, her PRF'nin ±1 m/s kör bantları |

Kör hız özelliği orta PRF dalga biçiminin sonucudur: bir bakış 3 PRF alt
aralığından en az 2'sinde tespit ister, dolayısıyla izin radyal hızı bir
radarın iki PRF'sinin kör bandına düşüyorsa o radar hedefi ne kadar güçlü
olursa olsun göremez; başka bir radar seçilmelidir. Hız belirsizken (yeni
doğmuş iz) olasılık küçük kalır; tek bir hız tahminiyle "kör" demek yeni
izlerin %47'sini yanlışlıkla kör gösteriyordu.

Son dört özellik "hangi radarı ne zaman tercih etmeliyiz" sorusunun
cevabıdır: hangi radar bu hedefi iyi görüyor, en son kim güncelledi,
kimin bakışları boşa gidiyor.

Bilgi kazancı, önerideki ödülün ana terimi (β_I) ile aynı büyüklüktür;
ajan böylece hangi talebin ödülü en çok artıracağını gözlemden
çıkarabilir.

### 3.4 Ölçeklerin seçimi

Ölçekler, orta PRF dalga biçimi ve İHA / uçak / füze hedefleriyle nominal ve
multi_target senaryolarında (6'şar tohum, kural tabanlı görev yöneticisi)
ölçülen değer aralıklarından seçildi. Eski ölçeklerde SNR özelliklerinin
%20–60'ı 1'in üstündeydi (raporların SNR'ı 10–80 dB: uçak RCS'si +25 dBsm'e
kadar, takip bakışı 384 darbe), iz konumları alan dışına taşıyordu (izler
14 km menzile kadar tutulur) ve füze hızları 950 m/s'e çıkıyordu. Yeni
ölçeklerle bütün özelliklerde ±1 dışına taşan oran %0,3'ün altında.

Aynı ölçümde bir hata da bulundu ve düzeltildi: 1 saniyelik karar iki 0,5 s
adımı kapsar, ikinci adımda talep olmadığı için zamanlayıcı cevapları
siliniyordu; gözlemdeki "son cevaplar" özellikleri hep 0 görünüyordu.
Artık her radarın son taleplerine verdiği cevaplar yeni talebe kadar korunur;
ödüldeki ret sayımı ise yalnızca o adımın cevaplarını (`step_answers`) kullanır.

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
ölçüm kovaryansı, P_D, PRF kademesi ve kör hızlar (`radar_physics_report`, 25/25), izleyici
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
* **Gözlem:** Bölüm 3'teki vektör (nominal'de 443 boyut).
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
* **Öğretmen kuralı (`policies.task_order`):** önce her yeni onaysız ize tek
  bir doğrulama bakışı (en yeni önce; tek noktadan başlatılan izin tahmini
  konumu hızlı hedefte birkaç saniyede kayar), bu bakışı kaçıran aday arama
  onu tekrar bulana kadar istenmez; sonra zamanı gelen onaylı izler (en
  belirsiz önce). `task_greedy` ve PPO'nun öğretmeni aynı fonksiyonu kullanır;
  kural `central_greedy_confirm` politikasınınkiyle aynı. 10 eşleşmiş
  tohumda eski kurala göre kapsama nominal'de 0,673 → 0,736, multi_target'ta
  0,656 → 0,725; en uzun iz ~4 s uzun; talep sayısı yarıya iniyor. Onaylı ama
  hedefle eşleşmeyen iz sayısı ise iki katına çıkıyor (0,48 → 0,96): bunların
  %57'si gerçek bir uçak / füzenin 1,5 km yakınında, yani hızlı hedeflerin
  kopya ya da eskiyen izleri (15 s zaman aşımına kadar sürüyorlar); izleyicide
  ele alınması gereken bir konu.

## 7. Açık konular / sonraki adım

* **Aksiyonun genişletilmesi:** öncelik ve son tarih seçimi, füzyonda
  radar güven katsayısı (bozulan radar senaryosuyla), PRF profili.
* **K seçimi:** Gözlem boyutu K ile doğrusal büyüyor (N·K·9 terimi);
  12 iz için 821. Daha büyük sahnelerde iz başına ortak ağırlıklı bir
  kodlayıcı (ör. attention / DeepSets) düşünülebilir.
* **Normalizasyon:** Ölçekler bu senaryolara göre seçildi; farklı alan
  boyutlarında yeniden kontrol edilmeli.
