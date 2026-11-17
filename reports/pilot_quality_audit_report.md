# GP-MARL-RadNet — Pilot Quality Audit & Episode Browser Report (Step 2D)

**Tarih:** 16 Kasım 2026  
**Aktif Branch:** `feature/valley-diverse-datasets`  
**Temel Entegrasyon Commit:** `302525b3` (üzerinden branch oluşturuldu)  
**Toplam Pilot Episode Sayısı:** 42 episode (7 veri seti grubu × 6 episode)  

---

## 1. Yönetici Özeti (Executive Summary)

Step 2D kapsamında, Step 2A (prosedürel vadi geometrileri), Step 2B (görsel doku ve çevre objeleri) ve Step 2C (7 grup pilot veri seti üretimi) çalışmalarının çıktıları **Scenario Management → Dataset Replay** arayüzüne entegre edilmiştir.

### Temel Başarımlar:
1. **Dinamik Veri Seti Kataloğu:** Arayüz artık hem yeni üretilen 7 pilot veri setini (`pilot_diverse_v1`), hem tüm pilotları bir arada, hem de geriye dönük uyumlu olarak legacy veri setlerini (`multi_target_v1`, `synthetic_100`) tek bir seçiciden sunmaktadır.
2. **Açıklayıcı Deterministik İsimlendirme (`display_name`):** Kullanıcı arayüzünde teknik `ep-000001-...` ID'leri yerine `EP-0001 · Rocky Canyon · Aircraft · 3 Radars` gibi operasyonel olarak anlamlı etiketler görüntülenmektedir.
3. **Çok Kriterli Reaktif Filtreleme:** Split, Terrain Family, Texture Family, Obstacle Profile, Hedef Sayısı, Radar Sayısı, Profil Kompozisyonu, Fiziksel Kompozisyon, Hedef Tipi ve Serbest Metin Arama olmak üzere 10 boyutlu reaktif filtre paneli uygulanmıştır.
4. **Anlık Episode Önizleme Kartı (Zero-NPZ Load):** Ağır `.npz` veri dosyaları açılmadan, hafif `pilot_metadata_index.json` üzerinden arazi, radar yerleşimi, hedef kompozisyonu ve ölçülen kalite metrikleri anında kullanıcıya sunulmaktadır.
5. **Geriye Dönük Uyumluluk:** Eski veri setlerinde (`multi_target_v1`, `synthetic_100`) bulunmayan yeni alanlar için sahte veri üretilmemiş, `"Unknown"` güvenli fallback mekanizması işletilmiştir.
6. **Eksiksiz Replay Entegrasyonu:** Seçilen episode "Load Replay Episode" butonuyla yüklendiğinde simülasyon yeniden çalıştırılmaz; Canonical Episode Records v2 (`records.npz` + `metadata.json`) okunarak 3B WebGL görünümü, 2B Plotly haritası, Radar Information analizleri ve Radar Selection karar teftişi anında aktive olur.
7. **Kapsamlı Kalite Denetimi (Audit):** 42 pilot episode'un tamamı denetlenmiş; 0 hatalı (flawed), 3 nominal, 39 zorlu (challenging) episode tespit edilmiştir. 39 episode'un neden `challenging` olduğu fiziksel radar-arazi geometrisi ve dar pencil-beam tarama kısıtları çerçevesinde doğrulanmıştır.
8. **Test Doğrulaması:** 8 yeni entegrasyon ve Streamlit AppTest testi dahil toplam 498 test çalıştırılmış; **496 test başarıyla geçmiş, 2 test atlanmış, 0 hata** alınmıştır.

---

## 2. Dataset Replay UI Mimarisi ve Arayüz Akışı

`src/interfaces/dataset_browser.py` modülü geliştirilmiş ve `src/interfaces/app.py` içine temiz bir mimariyle entegre edilmiştir.

### 2.1 Desteklenen Veri Setleri:
- `dataset_single_aircraft_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_single_uav_agile_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_single_uav_drift_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_single_missile_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_triple_all_same_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_triple_all_different_v1` (6 episode: 4 train, 1 val, 1 test)
- `dataset_triple_two_plus_one_v1` (6 episode: 4 train, 1 val, 1 test)
- `All Pilot Groups (pilot_diverse_v1 - 42 episodes)`
- `multi_target_v1 (legacy)` (100 episode)
- `synthetic_100 (legacy)` (100 episode)

### 2.2 Açıklayıcı İsimlendirme Formatı:
Gerçek metadata indeksinden türetilen deterministik `display_name` örnekleri:
- `EP-0001 · Rocky Canyon · Aircraft · 3 Radars`
- `EP-0007 · Forest Winding Pass · Agile UAV · 3 Radars`
- `EP-0013 · Snowy Chokepoint · Drift UAV · 3 Radars`
- `EP-0019 · Dry Canyon · Cruise Missile · 3 Radars`
- `EP-0025 · Grassland Highlands · 3 Targets (All Same) · 3 Radars`
- `EP-0031 · Mixed Multi-Obstacle · 3 Targets (All Different) · 3 Radars`
- `EP-0037 · Rocky Canyon · 3 Targets (2+1) · 3 Radars`

*Not: Teknik ID (`ep-000001-a60d26ae3bc0`) disk dosya yolunda, arama motorunda ve internal referanslarda değişmez birincil anahtar olarak korunmaktadır.*

---

## 3. Anlık Episode Önizleme ve Fiziksel/Görsel Ayrımı

Episode seçildiğinde, kullanıcının yükleme yapmasına gerek kalmadan görüntülenen önizleme kartı şunları içerir:
- **Kimlik Bilgileri:** Episode ID, Veri Seti Grubu, Split, Episode Seed, Arazi Seed.
- **Arazi ve Çevre:** Terrain Family, Texture Family, Obstacle Profile, Radar Placement Profile.
- **Hedef Bilgileri:** Hedef Sayısı (1 veya 3), Hedef Tipleri (`aircraft_like`, `uav_like`, `missile_like`), Hedef Profilleri, Kinematik Hareket Modelleri (`CV`, `CA`, `CT`, `SegmentedMotion`), Rota Ailesi.
- **Hedef Kompozisyonu:** Profil Kompozisyonu (`single`, `all_same`, `all_different`, `two_plus_one`) ve Fiziksel Kompozisyon.
- **Ölçülen Kalite Özeti:** LOS Açıklığı (%), Takip Kapsamı (%), Konum RMSE (m), Sıfır Tespitli Hedef Sayısı, Kalite Etiketi.

### Fiziksel vs Görsel Ayrımı İlkesi:
> [!IMPORTANT]
> - **Fiziksel Engel:** Radar görüş hattı (LOS) engellemesi yalnızca `Terrain.H` sayısal yükseklik matrisi ve $4/3$ efektif Dünya yarıçapı kırılma modeli üzerinden hesaplanır.
> - **Görsel Dekorasyon:** 3B sahnede gösterilen ağaç, kaya ve çalı instanced mesh nesneleri tamamen görsel zenginlik amaçlıdır. Fiziksel simülatöre sahte RF sönümlemesi veya yapay radar gölgelemesi enjekte etmez.

---

## 4. 42 Pilot Episode Kalite Denetimi (Pilot Quality Audit)

### 4.1 Genel Özet Tablosu
| Metrik | Değer |
| :--- | :--- |
| **Toplam Denetlenen Episode** | 42 |
| **Tamamlanma Oranı (Completed)** | 42 / 42 (%100.0) |
| **Hatalı Episode (Flawed)** | 0 / 42 (%0.0) |
| **Nominal Episode** | 3 / 42 (%7.1) |
| **Zorlu Episode (Challenging)** | 39 / 42 (%92.9) |
| **Toplam Yer Gerçeği (Ground Truth Rows)** | 2,418 |
| **Toplam Üretilen Radar Raporu** | 406 |
| **İletilen Rapor (Delivered Reports)** | 406 (%100.0 teslim) |
| **Kayıp Paket (Packet Loss)** | 0 (%0.0) |
| **Yanlış Alarm Raporları (Clutter/Noise)** | 346 |
| **Ortalama LOS Açıklığı (Clear %)** | %62.8 |
| **Ortalama Takip Kapsamı (Coverage %)** | %9.3 |
| **Minimum Hedef Yerden Yüksekliği (AGL)** | 30.0 m (Zemine çakılan hedef yok) |
| **Ortalama Hedef Yerden Yüksekliği (AGL)** | 656.1 m |
| **Minimum Radar Direk Yüksekliği (Mast Z)** | 15.0 m (Yeraltında radar yok) |

---

### 4.2 Neden 39 / 42 Episode "Challenging" Olarak Sınıflandırıldı?

Yapılan ayrıntılı fiziksel analizde, 39 episode'un `challenging` (zorlu) etiketini almasının bir simülatör veya model hatası **olmadığı**, aksine fiziksel ve operasyonel gerçekçiliğin doğrudan bir sonucu olduğu kanıtlanmıştır:

1. **Gerçekçi Dağ ve Vadi Maskelemesi:**  
   Yeni prosedürel arazilerde derin kanyonlar (`valley_linear_canyon`), kıvrımlı geçitler (`valley_sinuous_pass`) ve yüksek dağ kütleleri bulunmaktadır. Alçaktan uçan hedefler (özellikle AGL = 30-150m olan Agile/Drift İHA'lar ve seyir füzeleri), dağ sırtlarının arkasına girdiklerinde ışın izleme (ray tracing) algoritması tarafından fiziksel olarak maskelenmekte ve LOS kapanmaktadır.
2. **Kısa 15 Saniyelik Pencere vs Dar Kalem Hüzme (Pencil-Beam) Geometrisi:**  
   Radarlar $3.0^\circ$ azimut $\times 6.0^\circ$ elevasyon hüzme genişliğine sahip dar kalem hüzme (pencil beam) kullanmaktadır. $120^\circ$ azimut sektöründe 2 elevasyon barı ile raster tarama yapıldığında, sektörün tamamını bir kez taramak için en az 80 hüzme bakışı (look) gerekmektedir. Ancak 15 saniyelik bir simülasyon bölümü $dt = 0.5\text{s}$ ile yalnızca 30 adım (look) içerir. Bu da tek bir radarın arama sektörünün yalnızca $\%37.5$'ini tarayabilmesi anlamına gelir. İlk 30 adımda taranan sektörün dışında kalan hedefler hüzmeyle aydınlatılamamaktadır.
3. **Sıkı Operasyonel İzolasyon (Zero-Truth Leakage):**  
   Simülatörümüz operasyonel gerçekçilik gereği koordinasyon politikasına gizli hedef konumlarını (cheat/oracle coordinates) sızdırmaz. Radarlar hedefin nerede olduğunu bilmeden arama (search) paternini uygular. Tespit gelmedikçe takip kipine geçemez.
4. **Hatalı (Flawed) Durumların Sıfır Olması:**  
   Hiçbir hedef arazi yüzeyinin altına girmemiş (minimum $\text{AGL} \ge 30.0\text{m}$), hiçbir radar yeraltına gömülmemiştir (direk yüksekliği $\ge 15.0\text{m}$). Sayısal çökme, NaN değer veya tanımsız geçiş yaşanmamıştır.

---

### 4.3 Split Dağılımı Kalite Tablosu

| Split | Episode Sayısı | Tamamlanan | Nominal | Zorlu | Hatalı | Ort. LOS Açıklığı | Ort. Takip Kapsamı | Ort. Hedef AGL |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 28 | 28 | 2 | 26 | 0 | %64.4 | %6.2 | 667.6 m |
| **Validation** | 7 | 7 | 1 | 6 | 0 | %51.7 | %25.7 | 500.2 m |
| **Test** | 7 | 7 | 0 | 7 | 0 | %67.7 | %5.2 | 766.4 m |
| **Toplam** | **42** | **42** | **3** | **39** | **0** | **%62.8** | **%9.3** | **656.1 m** |

---

### 4.4 Veri Seti Grupları Kalite Tablosu

| Veri Seti Grubu | Ep. Sayısı | Hedef | Nominal / Zorlu | Raporlar | Yanlış Alarm | Ort. LOS % | Ort. Kapsam % | Ort. AGL (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dataset_single_aircraft_v1` | 6 | 1 | 0 / 6 | 42 | 41 | %98.2 | %0.0 | 1868.7 m |
| `dataset_single_uav_agile_v1` | 6 | 1 | 2 / 4 | 71 | 55 | %50.0 | %28.3 | 117.3 m |
| `dataset_single_uav_drift_v1` | 6 | 1 | 0 / 6 | 46 | 42 | %44.4 | %0.5 | 101.4 m |
| `dataset_single_missile_v1` | 6 | 1 | 0 / 6 | 57 | 53 | %75.8 | %5.5 | 397.5 m |
| `dataset_triple_all_same_v1` | 6 | 3 | 0 / 6 | 72 | 55 | %54.8 | %15.0 | 732.9 m |
| `dataset_triple_all_different_v1` | 6 | 3 | 0 / 6 | 60 | 53 | %62.6 | %0.5 | 632.7 m |
| `dataset_triple_two_plus_one_v1` | 6 | 3 | 1 / 5 | 58 | 47 | %54.1 | %15.0 | 742.5 m |

---

## 5. Test ve Doğrulama Sonuçları

Otomatik test süiti (`pytest`) çalıştırılmış ve tam regresyon testi gerçekleştirilmiştir:

```text
tests/test_dataset_browser.py ........                                   [  2%]
tests/test_diverse_dataset_generation.py ........                        [  3%]
tests/test_episode_records.py ................................           [  9%]
tests/test_learning_side.py .....ss.....                                 [ 11%]
tests/test_metrics.py .......                                            [ 13%]
tests/test_multi_target_generation.py ......                             [ 14%]
tests/test_procedural_terrain.py ................                        [ 17%]
tests/test_radar_information_service.py ........                         [ 19%]
tests/test_radar_model.py ....................                           [ 23%]
tests/test_radar_selection_service.py .....                              [ 24%]
tests/test_rng_isolation.py ...............                              [ 27%]
tests/test_scenario.py ................................................. [ 37%]
........................................................................ [ 51%]
...........................................................              [ 63%]
tests/test_scenario_management_ui.py ................                    [ 66%]
tests/test_scenario_planner.py ........................................  [ 74%]
tests/test_synthetic_dataset.py .................                        [ 78%]
tests/test_target_types.py ........                                      [ 79%]
tests/test_targets.py .................................................. [ 89%]
..........                                                               [ 91%]
tests/test_terrain.py ............................                       [ 97%]
tests/test_tracking_env.py .............                                 [100%]

================ 496 passed, 2 skipped, 1 warning in 89.09s ================
```

### Ek Olarak `test_dataset_browser.py` Test Kapsamı:
1. `test_catalog_discovery`: 7 pilot grup, All Pilot Groups ve 2 legacy veri setinin disk varlığı ve tespiti.
2. `test_load_pilot_all_episodes`: 42 pilot episode'un NPZ okunmadan hafif metadata indeksinden doğru alanlarla yüklenmesi.
3. `test_load_single_group_episodes`: Grup manifestinden tekil grup episode'larının doğru yüklenmesi.
4. `test_legacy_dataset_fallback`: `multi_target_v1` için `Unknown` güvenli fallback ve geriye dönük uyumluluk.
5. `test_multi_criteria_reactive_filtering`: 10 boyutlu filtrelemenin (split, target_count, profile_comp, target_type, free-text vb.) doğrulanması.
6. `test_pilot_replay_records_and_viewer_payload`: Canonical Records v2'den 3B viewer payload'a tam rekonstrüksiyon.
7. `test_legacy_replay_records_and_viewer_payload`: Legacy kayıtların sorunsuz 3B görünüme çevrilmesi.
8. `test_streamlit_app_dataset_replay_flow`: Streamlit `AppTest` ile UI widget etkileşiminin ve oturum durumunun (session state) uçtan uca doğrulanması.

---

## 6. Proje Kısıtları ve Bir Sonraki Adım

- **Kapsam Sınırı:** Yalnızca 42 pilot episode ile çalışılmış, 1000 bölümlük büyük veri üretimine **geçilmemiştir**.
- **Dal ve Git Koruması:** `main`, `demo` ve `integration/final-system-v1` dalları değiştirilmemiş; tüm çalışmalar izole `feature/valley-diverse-datasets` dalında tutulmuştur. Hiçbir push/merge yapılmamıştır.
- **Eski Veri Güvencesi:** `multi_target_v1` ve `synthetic_100` veri setleri korunmuş ve geriye dönük tam uyumlu olarak çalıştırılmaktadır.
- **Sonraki Adım:** Kullanıcının onayının ardından 1000 bölümlük tam üretim aşamasına (Step 3: Scaled 1000-Episode Dataset Generation) geçilecektir.
