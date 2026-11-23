"""Characterise the UAV / aircraft / missile classes through the radar model.

Monte-Carlo: draw targets from ``src.core.target_classes``, place them at
random geometry in front of one default radar (``RadarParams()``) and push
them through exactly the simulator's measurement chain -- geometry.measure,
LookupTableRCS + Swerling draw, radar-equation SNR, Marcum P_D and the
SNR-dependent measurement sigmas.  The per-class measurement envelopes are
written to ``docs/target_classification.md``.

    python scripts/characterize_target_classes.py [--samples 4000] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core import geometry as geo  # noqa: E402
from src.core.detection import pd_nonfluctuating, probability_of_detection  # noqa: E402
from src.core.modes import DEFAULT_PROFILES, Mode  # noqa: E402
from src.core.radar import (RadarParams, doppler_blind, measurement_sigmas,  # noqa: E402
                            noise_power, snr_linear, visible)
from src.core.rcs import incidence_in_body, m2_to_dbsm, sample_swerling  # noqa: E402
from src.core.target_classes import (CLASS_ORDER, RCS_ASPECTS, TARGET_CLASSES,  # noqa: E402
                                     rcs_model, sample_instance, variant_spec)

OUT = ROOT / "docs" / "target_classification.md"
RADAR_HEIGHT_M = 15.0
SECTOR_HALF_DEG = 60.0
GROUND_RANGE_M = (500.0, 14000.0)

METRICS = [  # key, Turkish label, unit, format
    ("speed", "Yatay hız \|v\|", "m/s", "{:.0f}"),
    ("alt", "İrtifa (AGL)", "m", "{:.0f}"),
    ("climb", "Dikey hız v_z", "m/s", "{:+.1f}"),
    ("rng", "Menzil r", "km", "{:.1f}"),
    ("el", "Yükseliş açısı θ", "°", "{:.1f}"),
    ("rdot", "Radyal hız \|ṙ\|", "m/s", "{:.0f}"),
    ("fd", "Doppler \|f_D\|", "kHz", "{:.2f}"),
    ("rcs_mean", "Ortalama RCS (görünüş)", "dBsm", "{:.1f}"),
    ("rcs_inst", "Anlık RCS (Swerling)", "dBsm", "{:.1f}"),
    ("snr_s", "SNR – arama", "dB", "{:.1f}"),
    ("snr_t", "SNR – takip", "dB", "{:.1f}"),
    ("pd_s", "P_D – arama", "-", "{:.2f}"),
    ("pd_t", "P_D – takip", "-", "{:.2f}"),
    ("blind", "Kör hız (çentik) oranı", "-", "{:.2f}"),
    ("sr_t", "σ_r – takip", "m", "{:.1f}"),
    ("srd_t", "σ_ṙ – takip", "m/s", "{:.2f}"),
    ("srd_s", "σ_ṙ – arama", "m/s", "{:.2f}"),
]


def simulate_variant(class_name, variant, n, rng, p: RadarParams):
    g0 = 10 ** (p.gain_dbi / 10)
    radar_pos = np.array([0.0, 0.0, RADAR_HEIGHT_M])
    spec = variant_spec(class_name, variant)
    rows, drawn = [], 0
    while len(rows) < n:
        drawn += 1
        inst = sample_instance(class_name, rng, variant)
        rho = np.sqrt(rng.uniform(GROUND_RANGE_M[0] ** 2, GROUND_RANGE_M[1] ** 2))  # uniform in area
        beta = np.deg2rad(rng.uniform(-SECTOR_HALF_DEG, SECTOR_HALF_DEG))
        pt = np.array([rho * np.cos(beta), rho * np.sin(beta), inst["altitude_agl_m"]])
        psi = rng.uniform(-np.pi, np.pi)
        v = np.array([inst["speed_mps"] * np.cos(psi), inst["speed_mps"] * np.sin(psi),
                      inst["climb_rate_mps"]])
        meas = geo.measure(pt, v, radar_pos, 0.0)
        if not visible(meas, p):
            continue
        a_lat = rng.uniform(-0.3, 0.3) * spec.max_lateral_accel_mps2
        u_b = incidence_in_body(radar_pos, pt, geo.body_rotation(v, a_lat))
        sig_mean = rcs_model(inst).mean_rcs(u_b)
        sig = sample_swerling(sig_mean, inst["swerling"], rng)
        r = meas[0]
        snr_s = float(snr_linear(r, sig, g0, p, DEFAULT_PROFILES[Mode.SEARCH].n_pulses))
        snr_t = float(snr_linear(r, sig, g0, p, DEFAULT_PROFILES[Mode.TRACK].n_pulses))
        sg_s = measurement_sigmas(snr_s, p, DEFAULT_PROFILES[Mode.SEARCH])
        sg_t = measurement_sigmas(snr_t, p, DEFAULT_PROFILES[Mode.TRACK])
        blind = doppler_blind(meas[3], p.prf_stagger_hz[int(rng.integers(len(p.prf_stagger_hz)))], p)
        rows.append({
            "speed": inst["speed_mps"], "alt": inst["altitude_agl_m"], "climb": inst["climb_rate_mps"],
            "rng": r / 1000, "el": np.rad2deg(meas[2]), "rdot": abs(meas[3]),
            "fd": abs(geo.doppler_shift(meas[3], p.wavelength)) / 1000,
            "rcs_mean": float(m2_to_dbsm(sig_mean)), "rcs_inst": float(m2_to_dbsm(sig)),
            "snr_s": 10 * np.log10(max(snr_s, 1e-30)), "snr_t": 10 * np.log10(max(snr_t, 1e-30)),
            "pd_s": 0.0 if blind else float(probability_of_detection(snr_s, p.pfa, p.pd_model)),
            "pd_t": 0.0 if blind else float(probability_of_detection(snr_t, p.pfa, p.pd_model)),
            "blind": float(blind),
            "sr_t": sg_t[0], "srd_t": sg_t[3], "srd_s": sg_s[3],
        })
    data = {k: np.array([row[k] for row in rows]) for k in rows[0]}
    return data, n / drawn


def detection_range_km(sigma_dbsm, n_pulses, p: RadarParams, pd_req=0.9):
    """Range where a non-fluctuating target reaches ``pd_req`` on boresight."""
    chi_req = brentq(lambda x: pd_nonfluctuating(x, p.pfa) - pd_req, 1e-3, 1e4)
    g = 10 ** (p.gain_dbi / 10)
    num = p.pt_w * g * g * p.wavelength ** 2 * 10 ** (sigma_dbsm / 10) * n_pulses
    den = (4 * np.pi) ** 3 * 10 ** (p.losses_db / 10) * noise_power(p) * chi_req
    return (num / den) ** 0.25 / 1000, 10 * np.log10(chi_req)


def _band(x, fmt):
    if set(np.unique(x)) <= {0.0, 1.0}:   # indicator -> mean rate
        return f"**{np.mean(x):.3f}**"
    q5, q50, q95 = np.percentile(x, [5, 50, 95])
    return f"{fmt.format(q5)} / **{fmt.format(q50)}** / {fmt.format(q95)}"


def _rng_txt(lo_hi, fmt="{:g}"):
    return f"{fmt.format(lo_hi[0])} – {fmt.format(lo_hi[1])}"


def build_markdown(results, p: RadarParams, n, seed):
    lam = p.wavelength
    search, track = DEFAULT_PROFILES[Mode.SEARCH], DEFAULT_PROFILES[Mode.TRACK]
    prf = 1 / search.pri_s
    out = []
    w = out.append
    w("# Hedef Sınıflandırması: İHA, Uçak, Füze\n")
    w("> Bu dosya `scripts/characterize_target_classes.py` tarafından üretilir. "
      "Parametreleri değiştirmek için `src/core/target_classes.py` dosyasını düzenleyip scripti "
      f"yeniden çalıştırın. (örnek sayısı / varyant = {n}, seed = {seed})\n")

    w("## 1. Radarın ölçtüğü büyüklükler\n")
    w("Simülatördeki radar (`src/core/radar.py`) her tespit için şu değerleri üretir; sınıflandırma "
      "bu büyüklükler ve EKF'nin bunlardan türettiği durum üzerine kuruludur.\n")
    w("| Büyüklük | Kaynak / denklem | Sınıf bilgisi taşır mı? |")
    w("|---|---|---|")
    w("| Menzil r | `geometry.measure`, σ_r = max(5 m, (c/2B)/√(2·SNR)) | Dolaylı: tespit menzili RCS'e bağlı |")
    w("| Azimut α | `geometry.measure`, σ_α = max(0.2 mrad, θ_az/√(2·SNR)) | Hayır (yalnızca konum) |")
    w("| Yükseliş θ | `geometry.measure` | Evet → irtifa = h_radar + r·sin θ |")
    w("| Radyal hız ṙ | `geometry.measure`, σ_ṙ = max(0.2, λ/(2·T_dwell)/√(2·SNR)) | Evet → hız alt sınırı |")
    w("| Doppler f_D | f_D = −2ṙ/λ | ṙ ile aynı bilgi, Hz cinsinden |")
    w("| SNR | radar denklemi χ = P_t G² λ² σ n / ((4π)³ R⁴ L k T₀ B F) | Evet → R biliniyorsa σ̂ (RCS) geri hesaplanır |")
    w("| P_D, sinyal kalitesi | Marcum Q (P_FA = 1e-6), (SNR−10)/(25−10) | SNR'ın fonksiyonu |")
    w("| Kör hız | katlanmış f_D o look'un PRF'inde çentik içindeyse tespit yok (`doppler_blind`) | Evet: ṙ ≈ 0 (havada asılı İHA, teğetsel uçuş) |")
    w("| Türetilen (iz) | EKF durumu x = [p, v] | \|v\|, v_z, ivme, dönüş hızı, irtifa |")
    w("")
    w(f"Varsayılan radar: f_c = {p.fc_hz / 1e9:g} GHz (λ = {lam * 100:.1f} cm), P_t = {p.pt_w:g} W, "
      f"G = {p.gain_dbi:g} dBi, B = {p.bandwidth_hz / 1e6:g} MHz (Δr = {geo.C_LIGHT / (2 * p.bandwidth_hz):.0f} m), "
      f"F = {p.noise_figure_db:g} dB, L = {p.losses_db:g} dB, r_max = {p.r_max_m / 1000:g} km, "
      f"arama {search.n_pulses} darbe / takip {track.n_pulses} darbe (dwell {search.dwell_s * 1e3:g} / "
      f"{track.dwell_s * 1e3:g} ms), nominal PRF {prf / 1000:g} kHz (MPRF), look başına PRF stagger "
      f"{', '.join(f'{x / 1000:g}' for x in p.prf_stagger_hz)} kHz, clutter çentiği ±{p.clutter_notch_mps:g} m/s.\n")

    w("## 2. Sınıflar ve varyantlar (fiziksel zarf)\n")
    w("Her sınıfın gerçek platformlardaki farkı yansıtan iki varyantı var. Değerler belirli bir "
      "platformun değil, platform kategorisinin açık literatürdeki genel zarfıdır. Aralıklar: "
      "`sert sınır` (tipik bant). Örneklerin %85'i tipik banttan çekilir.\n")
    w("| Sınıf / varyant | Hız (m/s) | İrtifa AGL (m) | max \|v_z\| (m/s) | max boyuna ivme (m/s²) | "
      "max yanal ivme (m/s²) | max dönüş (rad/s) | Swerling |")
    w("|---|---|---|---|---|---|---|---|")
    for cname in CLASS_ORDER:
        for vname, s in TARGET_CLASSES[cname]["variants"].items():
            w(f"| **{TARGET_CLASSES[cname]['name']}** / {vname} | {_rng_txt(s.speed_range_mps)} "
              f"({_rng_txt(s.speed_typical_mps)}) | {_rng_txt(s.altitude_agl_range_m)} "
              f"({_rng_txt(s.altitude_agl_typical_m)}) | {s.max_climb_rate_mps:g} | {s.max_long_accel_mps2:g} | "
              f"{s.max_lateral_accel_mps2:g} (≈{s.max_lateral_accel_mps2 / 9.81:.1f} g) | "
              f"{s.max_turn_rate_rad_s:g} ({np.rad2deg(s.max_turn_rate_rad_s):.0f}°/s) | {s.swerling} |")
    w("")
    w("### Görünüşe bağlı ortalama RCS (X-bandı, dBsm, örnekleme aralığı)\n")
    w("| Sınıf / varyant | Burun | Yan | Kuyruk | Üst/alt | Açıklama |")
    w("|---|---|---|---|---|---|")
    for cname in CLASS_ORDER:
        for vname, s in TARGET_CLASSES[cname]["variants"].items():
            cells = " | ".join(_rng_txt(s.rcs_dbsm[a]) for a in RCS_ASPECTS)
            w(f"| {TARGET_CLASSES[cname]['name']} / {vname} | {cells} | {s.description} |")
    w("")

    w("## 3. Ölçüm uzayındaki sınıf bantları (Monte Carlo)\n")
    w(f"Hedefler tek bir radarın (yükseklik {RADAR_HEIGHT_M:g} m) ±{SECTOR_HALF_DEG:g}° sektörüne, "
      f"{GROUND_RANGE_M[0] / 1000:g}–{GROUND_RANGE_M[1] / 1000:g} km yer menziline alan-üniform yerleştirildi; "
      "yön rastgele, yanal ivme ±%30·a_max. Görünmeyen (r > r_max, θ > 60°) örnekler atıldı. "
      "SNR hüzme merkezinde (en iyi durum) hesaplandı; her örnek stagger setinden rastgele bir PRF görür ve "
      "kör hıza düşerse P_D = 0 sayılır. Hücreler: **p5 / medyan / p95**; oranlar için ortalama.\n")
    w("### 3.1 Sınıf düzeyi\n")
    w("| Ölçüm | Birim | " + " | ".join(TARGET_CLASSES[c]["name"] for c in CLASS_ORDER) + " |")
    w("|---|---|" + "---|" * len(CLASS_ORDER))
    merged = {}
    for cname in CLASS_ORDER:
        variants = TARGET_CLASSES[cname]["variants"]
        parts = []
        for vname, spec in variants.items():
            d = results[(cname, vname)][0]
            k = int(round(spec.weight * n))
            parts.append({key: arr[:k] for key, arr in d.items()})
        merged[cname] = {key: np.concatenate([pt[key] for pt in parts]) for key in parts[0]}
    for key, label, unit, fmt in METRICS:
        w(f"| {label} | {unit} | " + " | ".join(_band(merged[c][key], fmt) for c in CLASS_ORDER) + " |")
    w("")
    w("### 3.2 Varyant düzeyi\n")
    cols = [(c, v) for c in CLASS_ORDER for v in TARGET_CLASSES[c]["variants"]]
    w("| Ölçüm | Birim | " + " | ".join(f"{TARGET_CLASSES[c]['name']}/{v}" for c, v in cols) + " |")
    w("|---|---|" + "---|" * len(cols))
    for key, label, unit, fmt in METRICS:
        w(f"| {label} | {unit} | " + " | ".join(_band(results[cv][0][key], fmt) for cv in cols) + " |")
    w("| Sektörde görünür oran | - | " + " | ".join(f"{results[cv][1]:.2f}" for cv in cols) + " |")
    w("")

    w("## 4. Tespit menzili (P_D = 0.9, dalgalanmasız, hüzme merkezi)\n")
    w("Varyantın burun ve yan RCS aralıklarının orta noktası kullanıldı. "
      f"r_max = {p.r_max_m / 1000:g} km'nin üstündeki değerler radar menziliyle sınırlıdır.\n")
    rows = []
    for c, v in cols:
        s = variant_spec(c, v)
        cells = []
        for aspect in ("nose", "side"):
            sig = float(np.mean(s.rcs_dbsm[aspect]))
            rs, chi_db = detection_range_km(sig, search.n_pulses, p)
            rt, _ = detection_range_km(sig, track.n_pulses, p)
            cells.append(f"{sig:+.1f} dBsm → {rs:.1f} / {rt:.1f}")
        rows.append(f"| {TARGET_CLASSES[c]['name']}/{v} | " + " | ".join(cells) + " |")
    w(f"Gerekli SNR (P_D = 0.9, P_FA = {p.pfa:g}): {chi_db:.1f} dB.\n")
    w("| Varyant | Burun: σ → R_arama / R_takip (km) | Yan: σ → R_arama / R_takip (km) |")
    w("|---|---|---|")
    out.extend(rows)
    w("")

    w("## 5. Ölçüm başına ayırt edici kurallar\n")
    w("Yukarıdaki bantlardan okunan, sınıflandırıcının kullanması beklenen fiziksel ipuçları:\n")
    w("- **Hız / Doppler:** |ṙ| ≤ |v| olduğundan ṙ hızın *alt sınırıdır*. |v| ≤ 50 m/s yalnızca İHA, "
      "|v| > 320 m/s yalnızca (süpersonik) füze. 200–280 m/s bandında uçak (avcı) ile seyir füzesi örtüşür.")
    w("- **RCS (SNR + menzilden):** Uçak ≥ ~0 dBsm, seyir füzesi burundan −20…−10 dBsm, çok rotorlu İHA "
      "−22…−5 dBsm. Aynı hız bandındaki avcı ile füzeyi ayıran ana büyüklük RCS'tir (≈15–20 dB fark).")
    w("- **İrtifa (θ ve r'den):** Seyir füzesi 30–200 m AGL'de arazi takibi yapar (θ < ~2°); avcı alçak uçuşta 100 m'ye inebilir "
      "ama tipik bandı 150–2000 m. Çok rotorlu İHA ≤ 400 m, sabit kanatlı İHA ≤ 1.5 km, uçaklar ≤ 4 km.")
    w("- **Manevra (iz):** Yanal ivme > 10 g yalnızca füze; ≈0.6 g (30° yatış) üstü yanal ivme nakliye uçağını dışlar. "
      "Havada asılı kalma (|v| ≈ 0) yalnızca çok rotorlu İHA.")
    w("- **Dalgalanma:** Uçak/İHA Swerling 1 (çok saçıcı), füze Swerling 3 (baskın saçıcı + küçükler); "
      "füzenin SNR'ı look'tan look'a daha az dalgalanır.")
    w("")

    w("## 6. Varsayımlar ve bilinen sınırlamalar\n")
    w(f"- **Doppler / menzil belirsizliği:** nominal PRF {prf / 1000:g} kHz tek başına ±λ·PRF/4 = "
      f"±{lam * prf / 4:.0f} m/s hız ve c/(2·PRF) = {geo.C_LIGHT / (2 * prf) / 1000:.1f} km menzil belirsizliği verir. "
      "Radar look'tan look'a PRF değiştirir (stagger) ve belirsizlikleri bu set üzerinden (Çin kalan teoremi) "
      "çözer; bu yüzden raporlar katlanmamış r ve ṙ taşır. Çözümleme hataları (hayalet hedef) modellenmez.")
    w("- **Kör hızlar modelleniyor:** katlanmış Doppler o look'un PRF'inde clutter çentiğine düşerse hedef "
      "o look'ta görülmez. Stagger sayesinde bir sonraki look'ta görünür; yalnızca ṙ ≈ 0 olan hedefler "
      "(havada asılı İHA, radara teğet uçan hedef) her PRF'te kördür.")
    w("- **Menzil kör bölgeleri (eclipsing)** ve clutter genliği (CNR) modellenmez; çentik ikili bir maskedir.")
    w("- **Mikro-Doppler yok.** Rotor/pervane modülasyonu (İHA'nın en güçlü gerçek ayırıcısı) bu "
      "algılama seviyesindeki modelde bulunmaz.")
    w("- RCS tablosu `LookupTableRCS.synthetic` ile burun/yan/kuyruk/üst değerleri arasında yumuşak geçişli, "
      "±1.5 dB dalgacıklı sentetik bir diyagramdır; gerçek ölçülmüş bir diyagram değildir.")
    w("- Monte Carlo'da arazi düz, tek radar ve hüzme merkezinde SNR (arama hüzmesi kayıpları dahil değil).")
    w("")
    w("## 7. Kaynaklar\n")
    w("- M. I. Skolnik, *Introduction to Radar Systems*, 3rd ed., McGraw-Hill, 2001 — tipik mikrodalga "
      "RCS değerleri (küçük uçak ~1 m², avcı 2–6 m², yolcu jeti 20–100 m², kanatlı füze ~0.5 m²); Swerling modelleri.")
    w("- M. A. Richards, *Fundamentals of Radar Signal Processing*, 2nd ed., 2014 — radar denklemi, Doppler, "
      "PRF belirsizliği.")
    w("- [PN] Proje içi not: *Physics-Informed Simulation of Three-Dimensional Radar Detections* (denklem numaraları kodda).")
    w("- M. N. Alpdemir, *Tactical UAV path optimization under radar threat using deep RL*, NCA 2022.")
    w("- Küçük İHA X-bandı RCS ölçümleri (açık literatür, ör. çok rotorlu İHA'lar için −25…−10 dBsm aralığı).")
    w("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples", type=int, default=4000, help="visible samples per variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    p = RadarParams()
    results = {}
    for cname in CLASS_ORDER:
        for vname in TARGET_CLASSES[cname]["variants"]:
            results[(cname, vname)] = simulate_variant(cname, vname, args.samples, rng, p)
            print(f"{cname}/{vname}: done (visible fraction {results[(cname, vname)][1]:.2f})")
    Path(args.out).write_text(build_markdown(results, RadarParams(), args.samples, args.seed),
                              encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
