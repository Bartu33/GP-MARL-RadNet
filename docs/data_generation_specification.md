# GP-MARL-RadNet: Sentetik Veri Üretim Spesifikasyonu (Data Generation Specification)

**Sürüm:** 2.1 (Class-Conditioned Profile Generation & Bounded Trajectory Dynamics)  
**Tarih:** 15 Kasım 2026  
**İlgili Reçeteler:** `configs/datasets/planner_profiles_100.yaml`  
**Referans Branch:** `feature/synthetic-data-v2`  

---

## 1. Genel Kapsam ve Amaç

GP-MARL-RadNet simülatörü, radar koordinasyonu, çoklu sensör veri birleştirme (EKF) ve kinematik hedef sınıflandırması (GBM) araştırmaları için sentetik hava hedefleri üretir.

Bu doküman; **Aşama 1B** kapsamında geliştirilen **Sınıfa Bağlı Sentetik Veri Üretim (Class-Conditioned Synthetic Data Generation)** mimarisini, her parametrenin fiziksel kökenini, birimlerini, örnekleme dağılımlarını, koordinat sistemi ayrımını ve kabul edilen varsayımları tanımlar.

---

## 2. Hedef Sınıfları ve Parametre Spesifikasyonu

Hedef profilleri, operasyonel gerçek platformların birebir kopyası değildir; radar araştırmalarında literatürde sıkça kullanılan non-operational sentetik davranış arketipleridir. 

Sınıflandırma problemini yapay olarak aşırı kolaylaştırmamak adına sınıflar arasında **kısmen örtüşen hız ve irtifa bantları** seçilmiştir.

### Hedef Sınıfları Parametre Tablosu

| Parametre | Birim | Sınıf 0: `A_fast_cruise` | Sınıf 1: `B_agile_hover` | Sınıf 2: `C_erratic_slow` | Sınıf 3: `D_drift` |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Referans Arketipi** | - | Yüksek hızlı sabit kanat seyir | Döner kanatlı / taktik İHA (dur-kalk) | Yavaş, düzensiz yönlü, dikey salınımlı | Pasif rüzgar sürüklenmesi (balon/clutter) |
| **Başlangıç Yatay Hızı ($v_{h0}$)**| m/s | $130.0 - 156.0$ ($468 - 562$ km/h) | $0.0 - 17.0$ ($0 - 61$ km/h) | $0.0 - 14.1$ ($0 - 51$ km/h) | $0.0 - 8.5$ ($0 - 31$ km/h) |
| **Azami Yörünge Hızı ($v_{\max}$)**| m/s | $\le 175.0$ ($630$ km/h) | $\le 20.0$ ($72$ km/h, sınırlı sprint) | $\le 16.0$ ($58$ km/h) | $\le 8.0$ ($29$ km/h) |
| **Dikey Hız Aralığı ($v_z$)**| m/s | $0.0$ (sabit irtifa) | $[-1.0, 1.0]$ | $[-1.0, 1.0]$ | $[-0.2, 1.2]$ (tırmanma eğilimli) |
| **Dünya ENU Z İrtifası ($p_z$)**| m | $1000.0 - 2000.0$ | $60.0 - 300.0$ | $60.0 - 650.0$ | $400.0 - 1400.0$ |
| **Asgari Yer Açıklığı (AGL)**| m | $\ge 30.0$ (Simülasyonda $\ge 880$ m) | $\ge 30.0$ (Simülasyonda $\ge 61$ m) | $\ge 30.0$ (Simülasyonda $\ge 105$ m) | $\ge 30.0$ (Simülasyonda $\ge 438$ m) |
| **İzin Verilen Hareketler** | - | CV, CT ($\pm 0.02$), CA ($[0.8, 0, 0]$)| CV, CT ($\pm 0.08$), Segmented CA+CV | CV, CT ($\pm 0.04$), Segmented CA+CV | CV, CA ($[0.05, 0.05, 0]$) |
| **Manevra İvmesi / Dönüşü**| - | $\omega = 0.02$ rad/s ($a_{\text{lat}} \approx 3$ m/s² ~ $0.3g$) | $\omega = 0.08$ rad/s, $a = 1.2$ m/s² (6s sprint)| $\omega = 0.04$ rad/s, $a = 0.35$ m/s² (8s ivme)| $\omega = 0.0$, $a = 0.05$ m/s² |
| **RCS Modeli** | - | `EllipsoidRCS` (büyük üç eksenli) | `EllipsoidRCS` (yassı disk elipsoit) | `ConstantRCS` (izotropik saçıcı) | `LookupTableRCS` (2B sentetik diyagram) |
| **RCS Parametreleri** | m | $a \in [3, 5], b \in [0.8, 1.3], c \in [0.5, 0.8]$ | $a,b \in [0.25, 0.4], c \in [0.08, 0.15]$ | $\sigma \in [0.008, 0.05]$ m² | Burun: $-8$, Yan: $-2$, Üst: $0$ dBsm |
| **Swerling Durumu** | - | Swerling 1 (Look başına üstel) | Swerling 1 (Look başına üstel) | Swerling 1 (Look başına üstel) | Swerling 0 (Deterministik) |

---

## 3. Parametrelerin Kaynağı ve Dağılım Varsayımları

### 3.1. Koordinat Sistemi: Dünya ENU Z ve AGL Ayrımı
- **Dünya Koordinat Sistemi (ENU):** Hedef konum vektörü $p = [p_x, p_y, p_z]$ Kartezyen East-North-Up (Doğu-Kuzey-Yukarı) koordinatlarındadır. $p_z$, deniz seviyesi veya arazi gridinin $z=0$ referans düzlemine göre mutlak yüksekliktir.
- **AGL (Above Ground Level):** Yerden yerel yüksekliktir ve şu bağıntıyla hesaplanır:
  $$\text{AGL}(t) = p_z(t) - \text{ground.height}(p_x(t), p_y(t))$$
- **Clearance Kuralı:** 
  1. Başlangıçta (`scenario_planner.py:253`): $\text{AGL}_0 \ge 30\text{ m}$ doğrulanır; aksi halde episode reddedilir ve konum yeniden çekilir.
  2. Trajectory boyunca (`targets.py:keep_above_ground`): Hedef $p_z(t) < \text{ground.height} + 30\text{ m}$ altına inerse, irtifa otomatik olarak arazi $+ 30\text{ m}$ seviyesine kenetlenir ve $v_z \ge 0$ yapılır.
- **Gerçekleşen Değerler:** 100 episode'luk simülasyonda tepe noktaları spawn bölgesinin dışında kaldığı için 30 saniyelik uçuş boyunca AGL asgari $61.5\text{ m}$ seviyesinde kalmış, araziye temas veya müdahale yaşanmamıştır.

### 3.2. Başlangıç Hızı ve İvmelenme Dinamikleri
- **Sınıf 0:** 12 farklı rotaya dağıtılmış $130 - 156$ m/s ($468 - 562$ km/h) seyir hızları (`choice`).
- **Sınıf 1:** $[-12, 12]$ m/s üniform yatay hız ($v_{h0} \le 17$ m/s). İvmelenme modelinde 30 saniyelik sürekli ivmelenme yerine **SegmentedMotion** kullanılarak 6 saniyelik sprint ($a = 1.2$ m/s², $\Delta v = 7.2$ m/s) ve ardından 24 saniyelik CV seyir/hover uygulanır. Böylece azami yörünge hızı $19.93$ m/s ($71.7$ km/h) ile sınırlandırılmıştır.
- **Sınıf 2:** $[-10, 10]$ m/s üniform yatay hız ($v_{h0} \le 14.1$ m/s). 8 saniyelik hafif ivmelenme ($a = [0.25, 0.25, 0]$ m/s²) ve 22 saniyelik CV uygulanır. Azami yörünge hızı $15.36$ m/s ile sınırlandırılmıştır.
- **Sınıf 3:** $[-6, 6]$ m/s üniform yatay hız ($v_{h0} \le 8.5$ m/s). Çok hafif sürüklenme ivmesi ($a = [0.05, 0.05, 0]$ m/s²) ile azami yörünge hızı $8.0$ m/s ile sınırlandırılmıştır.

### 3.3. Manevralar ve Hareket Modelleri
- `CV`: Doğrusal sabit hızlı hareket.
- `CT`: Yatay koordineli dönüş ($\omega = 0.02$ rad/s geniş, $0.04$ rad/s orta, $0.08$ rad/s dar çevik dönüş).
- `SegmentedMotion`: Sınırlı süreli manevra fazları (sprint/frenleme) sonrası kararlı uçuş.

---

## 4. Split İzolasyonu ve Katmanlı Örnekleme

1. **Katmanlı / Dengeli Sınıf Tahsisi (`balanced: true`):**
   - 100 episode'luk planda ($70$ train, $15$ val, $15$ test):
     - Train: $17$ veya $18$ episode / sınıf (mükemmel denge).
     - Validation: Her sınıf en az $3$ veya $4$ kez temsil edilir.
     - Test: Her sınıf en az $3$ veya $4$ kez temsil edilir.
   - Küçük alt kümeler ($N < 27$): Güvercin Yuvası Prensibi gereği validation ($N=2$) ve test ($N=1$) split'lerinde tüm 4 sınıf aynı anda bulunamaz. Tam temsil için en az $N \ge 27$ gereklidir.
2. **Terrain İzolasyonu:**
   - Train, validation ve test terrain havuzları tamamen ayrıktır; `terrain_group_id` split sınırlarını aşamaz.
3. **RNG Determinizmi:**
   - `SeedSequence` hiyerarşisi (`master_seed = 2026`):
     - $600$: Alan örnekleme akışı `(600, 1, index, field_id, attempt)`
     - $601$: Episode seed akışı `(601, 1, index)`
     - $602$: Terrain havuzu seçim akışı `(602, 1, index)`
   - Aynı master seed her zaman bit-for-bit aynı planı üretir.

---

## 5. Çıktı ve Kayıt Formatı

- **Biçim:** Canonical Episode Records v2
- **Dosyalar:** `metadata.json` (JSON) ve `records.npz` (NumPy compressed, `allow_pickle=False`)
- **İçerik:** `states`, `truth`, `reports`, `annotations`, `tracks`, `transitions`, `communication`, `deliveries`
- **Sızıntı Önleme:** Radarlar ve takipçiler truth etiketlerini (`truth_id`, `is_false_alarm`) görmez; yalnızca çevrimdışı değerlendirmede kullanılır.
