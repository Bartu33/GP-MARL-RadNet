"""Radar physics verification table (SNR, P_D, P_FA, RCS, measurement covariance).

Runs the deterministic and Monte Carlo checks of Section 16 of the physics
note (the same checks as ``tests/test_radar_model.py``) and writes them as a
table with expected value, measured value, tolerance and result, plus two
figures:

* ``pd_vs_snr.png``     -- empirical detection rate of the simulated radar
                          against the Marcum (non-fluctuating) and Swerling I
                          curves over a range of SNR values;
* ``sigma_vs_snr.png``  -- measurement standard deviations of the model
                          (Eqs. 39-46) and the empirical spread of simulated
                          reports.

Run:
    python -m src.services.radar_physics_report
Outputs go to ``data/processed/radar_physics/``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core import geometry as geo
from src.core.detection import pd_nonfluctuating, pd_swerling1
from src.core.modes import DEFAULT_PROFILES, Mode
from src.core.radar import (RadarAction, RadarParams, RadarState, antenna_gain, doppler_blind,
                            look_detection_probability, measurement_sigmas,
                            radar_step, snr_linear)
from src.core.rcs import ConstantRCS, EllipsoidRCS, sample_swerling
from src.core.targets import TargetState

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "radar_physics"
TRACK = DEFAULT_PROFILES[Mode.TRACK]


def _row(group, check, quantity, expected, measured, tol, ok, test, ref):
    return {"group": group, "check": check, "quantity": quantity, "expected": expected,
            "measured": measured, "tolerance": tol, "result": "PASS" if ok else "FAIL",
            "test": test, "reference": ref}


def _f(x, d=3):
    return f"{x:.{d}f}"


def _target(r, sigma=1.0, swerling=0):
    t = TargetState(id=0, label=0, p=np.array([r, 0.0, 300.0]), v=np.array([-50.0, 0, 0]),
                    rcs=ConstantRCS(sigma), swerling=swerling)
    t.update_attitude()
    return t


def _track_look(tgt, p):
    return RadarAction(Mode.TRACK, beam=tuple(geo.measure(tgt.p, tgt.v, p.pos, p.heading)[1:3]))


def empirical_pd(snr_db, n, rng, swerling=0):
    """Detection rate of radar_step for a target placed at the given mean SNR (track dwell)."""
    p = RadarParams(position=(0, 0, 0), heading_deg=0.0, false_alarms_per_beam=0.0)
    r = np.hypot(9000.0, 300.0)
    sigma = 10 ** (snr_db / 10) / snr_linear(r, 1.0, antenna_gain(0, 0, p), p, TRACK.n_pulses)
    tgt = _target(9000.0, sigma, swerling)
    act = _track_look(tgt, p)
    hits = 0
    for _ in range(n):
        dets, _ = radar_step([tgt], act, RadarState.initial(p), p, 0.5, rng)
        hits += len(dets)
    chi = snr_linear(np.linalg.norm(tgt.p), sigma, antenna_gain(0, 0, p), p, TRACK.n_pulses)
    return hits / n, chi, p.pfa


def blind_speed_rates(n, rng, snr_db=17.0):
    """Detection rate of a track dwell on a closing target at the first blind
    speed of the first PRF of the stagger, and on the same target at a speed
    no PRF is blind to.  Returns (rate blind, rate clear, look SNR)."""
    p = RadarParams(position=(0, 0, 0), heading_deg=0.0, false_alarms_per_beam=0.0)
    r = np.hypot(9000.0, 300.0)
    sigma = 10 ** (snr_db / 10) / snr_linear(r, 1.0, antenna_gain(0, 0, p), p, TRACK.n_pulses)
    tgt = _target(9000.0, sigma)
    u = tgt.p / np.linalg.norm(tgt.p)
    act = _track_look(tgt, p)
    out = []
    for speed in (p.wavelength * p.prf_stagger_hz[0] / 2, 50.0):    # blind at PRF 1 / clear at all
        tgt.v = -speed * u
        hits = sum(len(radar_step([tgt], act, RadarState.initial(p), p, 0.5, rng)[0]) for _ in range(n))
        out.append(hits / n)
    chi = snr_linear(np.linalg.norm(tgt.p), sigma, antenna_gain(0, 0, p), p, TRACK.n_pulses)
    return out[0], out[1], chi


def measurement_errors(r, n, rng, snr_db=None):
    """Errors of n simulated looks at a target at range r (RCS 1 m², or scaled so that
    the mean SNR is ``snr_db``); only looks that detect it give a report."""
    p = RadarParams(position=(0, 0, 0), heading_deg=0.0, false_alarms_per_beam=0.0, bias_std=(0, 0, 0, 0))
    sigma = 1.0
    if snr_db is not None:
        sigma = 10 ** (snr_db / 10) / snr_linear(np.hypot(r, 300.0), 1.0, antenna_gain(0, 0, p), p, TRACK.n_pulses)
    tgt = _target(r, sigma)
    truth = geo.measure(tgt.p, tgt.v, p.pos, p.heading)
    act = _track_look(tgt, p)
    errs, sigs, snr = [], [], None
    for _ in range(n):
        dets, _ = radar_step([tgt], act, RadarState.initial(p), p, 0.5, rng)
        if dets:
            e = dets[0]["z"] - truth
            e[1] = geo.wrap_angle(e[1])
            errs.append(e)
            sigs.append(np.sqrt(np.diag(dets[0]["measurement_covariance"])))
            snr = dets[0]["snr_db"] if "snr_db" in dets[0] else np.nan
    return np.array(errs), np.array(sigs), snr


def run_checks(n_mc: int = 3000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    p = RadarParams()
    g0 = antenna_gain(0, 0, p)

    # ------------------------------------------------------------ deterministic
    bad = 0
    for hd in (0.0, 0.7, np.pi / 2):
        R = geo.sensor_basis(hd)
        r, a, e, _ = geo.measure(1000 * R[:, 0], np.zeros(3), np.zeros(3), hd)
        _, a_r, _, _ = geo.measure(1000 * R[:, 1], np.zeros(3), np.zeros(3), hd)
        _, _, e_u, _ = geo.measure(1000 * R[:, 2], np.zeros(3), np.zeros(3), hd)
        bad += not (np.isclose(r, 1000) and np.isclose(a, 0) and np.isclose(e, 0)
                    and np.isclose(a_r, np.pi / 2) and np.isclose(e_u, np.pi / 2))
    rows.append(_row("Geometry", "Axis convention (x boresight, y right, z up)",
                     "azimuth/elevation of unit points, 3 headings", "boresight 0, right +90°, up +90°",
                     f"{3 - bad}/3 headings correct", "exact", bad == 0, "test_axes_convention", "PN Eqs. (8)-(12)"))

    pt = np.array([3000.0, 4000.0, 0.0])
    rd_away = geo.measure(pt, pt / np.linalg.norm(pt) * 50, np.zeros(3), 0.3)[3]
    rd_tan = geo.measure(pt, np.array([-40.0, 30.0, 0.0]), np.zeros(3), 0.3)[3]
    rows.append(_row("Geometry", "Radial velocity sign", "r_dot for 50 m/s receding / tangential motion",
                     "+50 / 0 m/s", f"{rd_away:+.3f} / {rd_tan:+.2e}", "1e-6",
                     abs(rd_away - 50) < 1e-6 and abs(rd_tan) < 1e-6, "test_radial_velocity", "PN Eq. (13)"))
    fd = geo.doppler_shift(50.0, p.wavelength)
    rows.append(_row("Geometry", "Doppler sign", "f_D for a receding target (r_dot = +50 m/s)",
                     f"-2 r_dot / λ = {-2 * 50 / p.wavelength:.1f} Hz", f"{fd:.1f} Hz", "exact",
                     np.isclose(fd, -2 * 50 / p.wavelength), "-", "Richards (2022)"))

    x = np.array([4000.0, -1500.0, 800.0, -60.0, 30.0, 5.0])
    radar, hd = np.array([100.0, 50.0, 10.0]), 0.4
    H = geo.measurement_jacobian(x[:3], x[3:], radar, hd)
    num = np.zeros_like(H)
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = 1e-3
        num[:, j] = (geo.measure((x + dx)[:3], (x + dx)[3:], radar, hd)
                     - geo.measure((x - dx)[:3], (x - dx)[3:], radar, hd)) / 2e-3
    rel = np.max(np.abs(num - H) / (np.abs(H) + 1e-8))
    rows.append(_row("Geometry", "Measurement Jacobian (EKF)", "max relative difference to finite differences",
                     "0", f"{rel:.1e}", "1e-4", rel < 1e-4, "test_jacobian_matches_finite_difference",
                     "PN Eq. (14)"))

    d12 = 10 * np.log10(snr_linear(5000, 1.0, g0, p) / snr_linear(10000, 1.0, g0, p))
    rows.append(_row("SNR", "Range doubled", "SNR change", "-12.04 dB (R^-4)", f"{-d12:.3f} dB", "±0.01 dB",
                     abs(d12 - 12.04) < 0.01, "test_double_range_minus_12db", "Skolnik (2001); PN Eq. (27)"))
    d3 = 10 * np.log10(snr_linear(5000, 2.0, g0, p) / snr_linear(5000, 1.0, g0, p))
    rows.append(_row("SNR", "RCS doubled", "SNR change", "+3.01 dB (∝ σ)", f"{d3:+.3f} dB", "±0.01 dB",
                     abs(d3 - 3.01) < 0.01, "test_double_rcs_plus_3db", "Skolnik (2001); PN Eq. (27)"))
    half = np.deg2rad(p.beamwidth_az_deg) / 2
    g_half = 10 * np.log10(antenna_gain(half, 0, p) / g0)
    sym = abs(antenna_gain(half, 0, p) - antenna_gain(-half, 0, p)) / g0
    rows.append(_row("SNR", "Beam pattern at half beamwidth", "gain relative to boresight; left/right",
                     "-3.01 dB, symmetric", f"{g_half:.3f} dB, asym. {sym:.1e}", "±0.01 dB",
                     abs(g_half + 3.01) < 0.01 and sym < 1e-12, "test_beam_symmetric_and_3db", "PN Eq. (24)"))
    lo, hi = measurement_sigmas(10 ** 0.8, p, TRACK), measurement_sigmas(10 ** 2.5, p, TRACK)
    rows.append(_row("Covariance", "Accuracy improves with SNR", "σ_r, σ_az, σ_el, σ_ṙ at 8 dB vs 25 dB",
                     "each σ(8 dB) ≥ σ(25 dB)", " / ".join(f"{a:.3g}≥{b:.3g}" for a, b in zip(lo, hi)), "-",
                     bool(np.all(lo >= hi)), "test_accuracy_decreases_with_snr", "PN Eqs. (39)-(46); MathWorks"))

    ell, sph = EllipsoidRCS(2.0, 1.0, 0.5), EllipsoidRCS(1.0, 1.0, 1.0)
    nose, side = ell.mean_rcs([1, 0, 0]), ell.mean_rcs([0, 1, 0])
    rows.append(_row("RCS", "Ellipsoid RCS (a=2, b=1, c=0.5 m)", "nose-on σ; side > nose",
                     f"π b²c²/a² = {np.pi * 0.25 / 4:.4f} m²", f"{nose:.4f} m², side {side:.3f} m²", "1e-9",
                     np.isclose(nose, np.pi * 0.25 / 4) and side > nose, "test_rcs_models",
                     "Alpdemir (2022); Kabamba et al. (2006)"))
    rows.append(_row("RCS", "Sphere limit (a=b=c=1 m)", "σ at an oblique aspect", f"π a² = {np.pi:.4f} m²",
                     f"{sph.mean_rcs([0.6, 0.0, 0.8]):.4f} m²", "1e-9",
                     np.isclose(sph.mean_rcs([0.6, 0.0, 0.8]), np.pi), "test_rcs_models", "Skolnik (2001)"))

    s = RadarState.initial(p)
    e_ok = h_ok = True
    for _ in range(400):
        radar_step([], RadarAction(Mode.TRACK, beam=(0.0, 0.05)), s, p, 0.5, rng)
        e_ok &= 0 <= s.energy_j <= p.e_max_j
        h_ok &= s.heat_k <= p.h_max_k + 1e-9
    rows.append(_row("Resources", "Energy and heat stay in limits", "400 continuous track dwells",
                     "0 ≤ E ≤ E_max, H ≤ H_max", f"E ok: {e_ok}, H ok: {h_ok}", "-", e_ok and h_ok,
                     "test_resources_stay_bounded", "PN Eqs. (52)-(56)"))

    # waveform: medium-PRF stagger and blind speeds
    from src.core.target_classes import TARGET_CLASSES
    from src.services.radar_info import stagger_unambiguous
    r_u, v_u = stagger_unambiguous(p)
    v_fast = max(v.speed_range_mps[1] for c in TARGET_CLASSES.values() for v in c["variants"].values())
    rows.append(_row("Waveform", "PRF stagger resolves range",
                     "unambiguous range over the stagger, c / 2·gcd(PRF)",
                     f"≥ instrumented range {p.r_max_m / 1000:.1f} km", f"{r_u / 1000:.1f} km", "-",
                     r_u >= p.r_max_m, "test_parameter_table_and_derived_values", "Richards (2022), MPRF"))
    rows.append(_row("Waveform", "PRF stagger resolves radial velocity",
                     "unambiguous radial velocity over the stagger, ±λ·lcm(PRF) / 4",
                     f"≥ fastest target class {v_fast:.0f} m/s", f"±{v_u:.0f} m/s", "-",
                     v_u >= v_fast, "test_parameter_table_and_derived_values", "Richards (2022), MPRF"))
    zero = all(doppler_blind(0.5 * p.clutter_notch_mps, f, p) for f in p.prf_stagger_hz)
    vv = np.arange(1.01, 1.1 * v_fast, 0.01) * p.clutter_notch_mps
    worst = max(sum(doppler_blind(x, f, p) for f in p.prf_stagger_hz) for x in vv[::7])
    rows.append(_row("Waveform", "Clutter notch and blind speeds",
                     f"PRFs blind at once: zero Doppler / any |ṙ| in ({p.clutter_notch_mps:g}, {1.1 * v_fast:.0f}] m/s",
                     f"{len(p.prf_stagger_hz)} / ≤ 1", f"{len(p.prf_stagger_hz) if zero else '<' + str(len(p.prf_stagger_hz))} / {worst}",
                     "exact", zero and worst <= 1, "test_doppler_blind_speeds_follow_prf_stagger",
                     "Skolnik (2001), MTI blind speeds"))

    # ------------------------------------------------------------ Monte Carlo
    pd_emp, chi, pfa = empirical_pd(16.0, n_mc, rng)
    pd_mod = float(look_detection_probability(chi, p))
    tol = 3 * np.sqrt(pd_mod * (1 - pd_mod) / n_mc)
    rows.append(_row("P_D", f"Detection rate vs {p.prf_m_of_n}-of-{len(p.prf_stagger_hz)} PRF model (non-fluctuating)",
                     f"empirical P_D at {10 * np.log10(chi):.1f} dB look SNR, {n_mc} looks", _f(pd_mod),
                     _f(pd_emp), f"±{tol:.3f} (3σ binomial)", abs(pd_emp - pd_mod) <= tol,
                     "test_empirical_pd_matches_model", "Marcum (1960); PN Eq. (31); binomial M-of-N"))

    n_b = n_mc // 2
    r_bl, r_cl, chi_b = blind_speed_rates(n_b, rng)
    m_bl, m_cl = (float(look_detection_probability(chi_b, p, k)) for k in (1, 0))
    t_bl, t_cl = (3 * np.sqrt(max(m * (1 - m), 1e-4) / n_b) for m in (m_bl, m_cl))
    rows.append(_row("Waveform", "Blind speed in the simulator",
                     f"P_D at {10 * np.log10(chi_b):.1f} dB: closing at the first blind speed of "
                     f"{p.prf_stagger_hz[0] / 1e3:g} kHz / at 50 m/s, {n_b} looks each",
                     f"{_f(m_bl)} / {_f(m_cl)}", f"{_f(r_bl)} / {_f(r_cl)}", f"±{t_bl:.3f} / ±{t_cl:.3f} (3σ)",
                     abs(r_bl - m_bl) <= t_bl and abs(r_cl - m_cl) <= t_cl and m_bl < m_cl,
                     "test_doppler_blind_speeds_follow_prf_stagger", "Skolnik (2001), MTI blind speeds"))

    chi_sw, pfa_sw = 20.0, 1e-6
    n_sw = max(n_mc * 5, 10000)
    pd_mc = np.mean(rng.random(n_sw) < pd_nonfluctuating(chi_sw * rng.exponential(size=n_sw), pfa_sw))
    pd_cf = float(pd_swerling1(chi_sw, pfa_sw))
    tol = 3 * np.sqrt(pd_cf * (1 - pd_cf) / n_sw)
    rows.append(_row("P_D", "Swerling I closed form vs Monte Carlo",
                     f"P_D at χ̄ = 13.0 dB, P_FA = 1e-6, {n_sw} draws", _f(pd_cf), _f(pd_mc),
                     f"±{tol:.3f} (3σ)", abs(pd_mc - pd_cf) <= tol, "test_swerling1_closed_form_vs_monte_carlo",
                     "Swerling (1960)"))

    for case in (1, 3):
        xs = np.array([sample_swerling(2.0, case, rng) for _ in range(n_sw)])
        tol = 3 * xs.std() / np.sqrt(len(xs))
        rows.append(_row("RCS", f"Swerling {'I' if case == 1 else 'III'} fluctuation keeps the mean",
                         f"mean of {len(xs)} samples, σ̄ = 2 m²", "2.000 m²", f"{xs.mean():.3f} m²",
                         f"±{tol:.3f} (3σ)", abs(xs.mean() - 2.0) <= tol, "test_swerling_mean_is_preserved",
                         "Swerling (1960)"))

    pf = RadarParams(false_alarms_per_beam=0.2)
    s = RadarState.initial(pf)
    n_fa = n_beams = 0
    for _ in range(max(n_mc // 2, 500)):
        dets, s = radar_step([], RadarAction(Mode.SEARCH), s, pf, 0.5, rng)
        n_fa += sum(d["is_false_alarm"] for d in dets)
        n_beams += len(s.last_beams)
        s.energy_j, s.heat_k = pf.e_max_j, pf.h_ambient_k
    rate = n_fa / n_beams
    tol = 3 * np.sqrt(0.2 / n_beams)
    rows.append(_row("P_FA", "False alarms per beam (Poisson)", f"empirical rate over {n_beams} search beams",
                     "0.200", _f(rate), f"±{tol:.3f} (3σ Poisson)", abs(rate - 0.2) <= tol,
                     "test_false_alarm_rate", "PN Eq. (50)"))

    # constant-RCS target at 18 dB: above the detection threshold, below the accuracy floors
    # each report carries its own sigma (2 or 3 PRF sub-dwells detected), so the
    # errors are compared with it: error / reported sigma should have mean 0, std 1
    errs, sigs, _ = measurement_errors(9000.0, n_mc, rng, snr_db=18.0)
    zs = errs / sigs
    names = ["range [m]", "azimuth [rad]", "elevation [rad]", "radial velocity [m/s]"]
    for k, nm in enumerate(names):
        m, sd = zs[:, k].mean(), zs[:, k].std()
        rows.append(_row("Covariance", f"Measurement error at 18 dB SNR, {nm}",
                         f"error / reported σ over {len(errs)} reports (median σ {np.median(sigs[:, k]):.3g})",
                         "mean 0, std 1", f"{m:+.3f} / {sd:.3f}", "mean ±3/√n, std ±8 %",
                         abs(m) <= 3 / np.sqrt(len(errs)) and abs(sd - 1) <= 0.08,
                         "test_measurement_error_statistics", "PN Eqs. (35)-(37)"))
    corr = np.corrcoef(errs.T)
    off = np.max(np.abs(corr - np.eye(4)))
    tol = 4 / np.sqrt(len(errs))
    rows.append(_row("Covariance", "Measurement errors uncorrelated (diagonal R)",
                     "largest correlation between r, az, el, ṙ errors", "0", f"{off:.3f}", f"≤ {tol:.3f}",
                     off <= tol, "-", "PN Eq. (37)"))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def make_figures(out: Path, n_mc: int = 1500, seed: int = 1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    snr_db = np.arange(8.0, 27.0, 1.0)
    emp = np.array([empirical_pd(s, n_mc, rng)[0] for s in snr_db])
    emp_sw = np.array([empirical_pd(s, n_mc, rng, swerling=1)[0] for s in snr_db])
    p = RadarParams()
    pfa = p.pfa
    fine = np.linspace(snr_db[0], snr_db[-1], 200)
    chi = 10 ** (fine / 10)
    rule = f"{p.prf_m_of_n} of {len(p.prf_stagger_hz)} PRFs"
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(fine, pd_nonfluctuating(chi, pfa), color="0.6", lw=1, label="one coherent dwell, Marcum (reference)")
    ax.plot(fine, look_detection_probability(chi, p), "k-", label=f"{rule}, non-fluctuating")
    ax.plot(fine, look_detection_probability(chi, p, swerling=1), "k--", label=f"{rule}, Swerling I")
    e = np.sqrt(emp * (1 - emp) / n_mc)
    ax.errorbar(snr_db, emp, 2 * e, fmt="o", ms=4, capsize=2, color="#c0392b", label="simulated, constant RCS")
    e = np.sqrt(emp_sw * (1 - emp_sw) / n_mc)
    ax.errorbar(snr_db, emp_sw, 2 * e, fmt="s", ms=4, capsize=2, color="#3182bd", label="simulated, Swerling I")
    ax.set_xlabel("mean SNR of the whole track look [dB]")
    ax.set_ylabel("probability of detection")
    ax.set_title(f"P_D vs SNR (P_FA = {pfa:g}, {n_mc} looks per point, ±2σ)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "pd_vs_snr.png", dpi=150)
    plt.close(fig)

    p = RadarParams()
    snr = np.linspace(10, 34, 100)
    sig = np.array([measurement_sigmas(10 ** (s / 10), p, TRACK) for s in snr])
    pts = []
    for s_db in (13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 31.0):
        errs, _, _ = measurement_errors(9000.0, 800, rng, snr_db=s_db)
        if len(errs) > 100:
            pts.append((s_db, errs.std(0)))
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
    labels = ["σ range [m]", "σ azimuth [mrad]", "σ elevation [mrad]", "σ radial velocity [m/s]"]
    scale = [1, 1e3, 1e3, 1]
    for k, ax in enumerate(axes):
        ax.plot(snr, sig[:, k] * scale[k], "k-", label="model (Eqs. 39-46)")
        if pts:
            ax.plot([q[0] for q in pts], [q[1][k] * scale[k] for q in pts], "o", color="#c0392b",
                    label="simulated reports")
        ax.set_xlabel("SNR [dB]")
        ax.set_title(labels[k], fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Measurement accuracy vs SNR (track dwell): model and empirical spread of detected reports; "
                 "flat parts are the accuracy floors", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "sigma_vs_snr.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000, help="Monte Carlo looks per check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    table = run_checks(a.n, a.seed)
    table.to_csv(a.out / "radar_physics_checks.csv", index=False)
    try:
        table.to_excel(a.out / "radar_physics_checks.xlsx", index=False)
    except ImportError:
        pass                                   # openpyxl not installed: CSV only
    if not a.no_figures:
        make_figures(a.out)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60):
        print(table[["group", "check", "expected", "measured", "tolerance", "result"]].to_string(index=False))
    print(f"\n{(table.result == 'PASS').sum()}/{len(table)} checks passed; written to {a.out}")


if __name__ == "__main__":
    main()
