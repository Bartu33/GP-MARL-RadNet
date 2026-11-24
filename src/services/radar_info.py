"""Human-readable radar and target parameters and the radar-equation budget.

Used by the dashboard (parameter tables, "how was the SNR computed") and by
the 3-D viewer's context menu.  Every number comes from the same objects and
functions the simulator uses (``RadarParams``, ``snr_linear``,
``measurement_sigmas``, the P_D models), so what is shown is what is computed.
"""
from __future__ import annotations

import numpy as np

from src.core import geometry as geo
from src.core.modes import INTERLEAVE_SEARCH_BEAMS, Mode
from src.core.radar import (K_BOLTZMANN, RadarParams, antenna_gain, doppler_blind, look_detection_probability,
                            measurement_sigmas, noise_power, snr_linear, unambiguous_limits)
from src.core.rcs import ConstantRCS, EllipsoidRCS, LookupTableRCS, ScatteringCentreRCS
from src.core.targets import class_short

C = geo.C_LIGHT
db = lambda x: 10 * np.log10(x)


def stagger_unambiguous(p: RadarParams) -> tuple[float, float]:
    """Unambiguous range and radial velocity after resolving across the PRF stagger.

    Range folds with c / 2 PRF at each PRF; the folded values agree again only
    every c / 2 gcd(PRF).  Doppler folds with PRF; the first speed blind at
    every PRF at once is lambda lcm(PRF) / 2, so +-lambda lcm(PRF) / 4 is
    measured unambiguously.  With a single PRF these are the usual limits.
    """
    return unambiguous_limits(p.prf_stagger_hz, p)


def radar_parameters(p: RadarParams) -> list[tuple[str, str, str]]:
    """(group, parameter, value) rows for one radar, incl. derived quantities."""
    lam = p.wavelength
    tr, se = p.profiles[Mode.TRACK], p.profiles[Mode.SEARCH]
    prf = 1.0 / tr.pri_s
    rows = [
        ("Location", "position (E, N, U)", f"{p.position[0]:.0f}, {p.position[1]:.0f}, {p.position[2]:.0f} m"),
        ("Location", "boresight azimuth (CCW from east)", f"{p.heading_deg:.0f}°"),
        ("Location", "field of view (azimuth)", f"±{p.fov_az_deg:.0f}°"),
        ("Location", "elevation limits", f"{p.el_min_deg:.0f}° … {p.el_max_deg:.0f}°"),
        ("Location", "instrumented range", f"{p.r_min_m / 1000:.1f} – {p.r_max_m / 1000:.1f} km"),
        ("Transmitter", "carrier frequency f_c", f"{p.fc_hz / 1e9:.2f} GHz"),
        ("Transmitter", "wavelength λ = c / f_c", f"{lam * 100:.2f} cm"),
        ("Transmitter", "peak power P_t", f"{p.pt_w:.0f} W ({db(p.pt_w):.1f} dBW)"),
        ("Antenna", "gain G_0 (transmit = receive)", f"{p.gain_dbi:.1f} dBi"),
        ("Antenna", "3-dB beamwidth az × el", f"{p.beamwidth_az_deg:.1f}° × {p.beamwidth_el_deg:.1f}°"),
        ("Receiver", "bandwidth B", f"{p.bandwidth_hz / 1e6:.2f} MHz"),
        ("Receiver", "noise figure F", f"{p.noise_figure_db:.1f} dB"),
        ("Receiver", "system losses L", f"{p.losses_db:.1f} dB"),
        ("Receiver", "noise power k T_0 B F", f"{db(noise_power(p)):.1f} dBW"),
        ("Detection", "false-alarm probability P_FA", f"{p.pfa:g}"),
        ("Detection", "P_D model", p.pd_model),
        ("Detection", "false alarms per beam position", f"{p.false_alarms_per_beam:g}"),
        ("Waveform", "pulse repetition frequency (PRF)", f"{prf / 1e3:.1f} kHz (PRI {tr.pri_s * 1e6:.0f} µs)"),
        ("Waveform", "track dwell", f"{tr.n_pulses} pulses = {tr.dwell_s * 1e3:.1f} ms"),
        ("Waveform", "search dwell", f"{se.n_pulses} pulses × {se.beams_per_look} beams = "
                                     f"{se.total_dwell_s * 1e3:.1f} ms per step"),
        ("Waveform", "interleaved look", f"{INTERLEAVE_SEARCH_BEAMS} search beams + "
                                         f"{p.profiles[Mode.INTERLEAVED].n_pulses}-pulse track dwell"),
        ("Derived", "range resolution c / 2B", f"{C / (2 * p.bandwidth_hz):.0f} m"),
        ("Derived", "radial-velocity resolution λ / 2T_CPI (track)", f"{lam / (2 * tr.dwell_s):.2f} m/s"),
        ("Derived", "unambiguous range c / 2·PRF", f"{C / (2 * prf) / 1000:.1f} km"),
        ("Derived", "unambiguous radial velocity ±λ·PRF / 4", f"±{lam * prf / 4:.0f} m/s"),
        ("Waveform", "PRF stagger (every look: one coherent sub-dwell per PRF)",
         ", ".join(f"{f / 1e3:g}" for f in p.prf_stagger_hz) + " kHz"),
        ("Waveform", "track sub-dwell per PRF", f"{tr.n_pulses // len(p.prf_stagger_hz)} pulses, "
                                                f"{tr.dwell_s / len(p.prf_stagger_hz) * 1e3:.1f} ms"),
        ("Detection", "detection rule (M of N PRF sub-dwells)", f"{p.prf_m_of_n} of {len(p.prf_stagger_hz)}"),
        ("Derived", "unambiguous range over the stagger c / 2·gcd(PRF)",
         f"{stagger_unambiguous(p)[0] / 1000:.1f} km"),
        ("Derived", "unambiguous radial velocity over the stagger ±λ·lcm(PRF) / 4",
         f"±{stagger_unambiguous(p)[1]:.0f} m/s"),
        ("Derived", "clutter notch (blind if folded |ṙ| inside)", f"±{p.clutter_notch_mps:g} m/s"),
        ("Accuracy", "floors σ_r, σ_angle, σ_ṙ", f"{p.floor_r_m:g} m, {p.floor_ang_rad * 1e3:g} mrad, "
                                                 f"{p.floor_rdot_mps:g} m/s"),
        ("Resources", "energy budget E_max", f"{p.e_max_j:.0f} J (recovers {p.e_recovery_w:.0f} W)"),
        ("Resources", "thermal limit", f"{p.h_max_k - p.h_ambient_k:.0f} K above ambient"),
        ("Resources", "tracking channels c_max", f"{p.c_max}"),
    ]
    return rows


def ambiguity_notes(p: RadarParams, max_speed_mps: float) -> list[str]:
    """Plain warnings where the scenario exceeds what this waveform can measure unambiguously.

    Every look uses all PRFs of the stagger (``RadarParams.prf_stagger_hz``) and
    resolves range and radial velocity over the PRFs that detected the target:
    over the full set the limits are those of :func:`stagger_unambiguous`, over
    M of them they are smaller.
    """
    r_u, v_u = stagger_unambiguous(p)
    notes = []
    if max_speed_mps > v_u:
        notes.append(f"{p.name}: targets up to {max_speed_mps:.0f} m/s, but the PRF set measures radial velocity "
                     f"unambiguously only within ±{v_u:.0f} m/s; faster targets are reported with folded "
                     "(aliased) radial velocity.")
    m = p.prf_m_of_n
    if m < len(p.prf_stagger_hz):
        from itertools import combinations
        v_pair = min(unambiguous_limits(c, p)[1] for c in combinations(p.prf_stagger_hz, m))
        if max_speed_mps > v_pair:
            notes.append(f"{p.name}: targets up to {max_speed_mps:.0f} m/s. When one PRF is blind or misses, the "
                         f"remaining {m} PRFs resolve radial velocity only within ±{v_pair:.0f} m/s; such looks "
                         "report folded radial velocity.")
    if r_u < p.r_max_m:
        notes.append(f"{p.name}: unambiguous range {r_u / 1000:.1f} km is shorter than the "
                     f"instrumented range {p.r_max_m / 1000:.1f} km.")
    return notes


def describe_rcs(model) -> str:
    if isinstance(model, EllipsoidRCS):
        return f"ellipsoid a×b×c = {model.a:.2f} × {model.b:.2f} × {model.c:.2f} m"
    if isinstance(model, ConstantRCS):
        return f"constant {model.sigma_m2:.3f} m²"
    if isinstance(model, LookupTableRCS):
        return "aspect table"
    if isinstance(model, ScatteringCentreRCS):
        return "scattering centres"
    return type(model).__name__


def rcs_range(model, n: int = 200, seed: int = 0) -> tuple[float, float, float]:
    """(min, median, max) mean RCS over random aspects."""
    u = np.random.default_rng(seed).normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    v = np.array([model.mean_rcs(x) for x in u])
    return float(v.min()), float(np.median(v)), float(v.max())


def target_parameters(targets) -> list[dict]:
    rows = []
    for t in targets:
        lo, med, hi = rcs_range(t.rcs)
        cls = (f"{class_short(t.label)} ({t.variant.replace('_', ' ')})" if getattr(t, "target_class", None)
               else class_short(t.label, legacy=True))
        rows.append({"target": f"T{t.id}", "class": cls, "RCS model": describe_rcs(t.rcs),
                     "RCS over aspects (min / median / max)": f"{lo:.3g} / {med:.3g} / {hi:.3g} m²",
                     "fluctuation": f"Swerling {t.swerling}" if t.swerling else "none (Swerling 0)"})
    return rows


def link_budget(p: RadarParams, r_m: float, sigma_m2: float, mode: Mode = Mode.TRACK,
                off_az_rad: float = 0.0, off_el_rad: float = 0.0, swerling: int = 0,
                r_dot_mps: float | None = None) -> dict:
    """Radar equation (Eq. 27) term by term in dB, then P_D and accuracy.

    SNR = P_t G_t G_r λ² σ N / ((4π)³ R⁴ L k T_0 B F)   with N pulses integrated
    over the whole look.  The look is split into one coherent sub-dwell per PRF
    of the stagger (SNR / n each) and declared a detection on M of them;
    ``r_dot_mps`` removes the sub-dwells whose PRF makes the target blind."""
    prof = p.profiles[mode]
    g = antenna_gain(off_az_rad, off_el_rad, p)
    lam = p.wavelength
    terms = [
        ("+ P_t  peak power", db(p.pt_w)),
        ("+ G_t  transmit gain (beam pattern)", db(g)),
        ("+ G_r  receive gain", db(g)),
        ("+ λ²  wavelength squared", db(lam ** 2)),
        ("+ σ  target RCS (this aspect)", db(max(sigma_m2, 1e-12))),
        (f"+ N  coherent integration ({prof.n_pulses} pulses)", db(prof.n_pulses)),
        ("− (4π)³", -db((4 * np.pi) ** 3)),
        ("− R⁴  range to the fourth power", -db(r_m ** 4)),
        ("− L  system losses", -p.losses_db),
        ("− k T_0  thermal noise density", -db(K_BOLTZMANN * p.t0_k)),
        ("− B  bandwidth", -db(p.bandwidth_hz)),
        ("− F  noise figure", -p.noise_figure_db),
    ]
    chi = float(snr_linear(r_m, sigma_m2, g, p, prof.n_pulses))
    sig = measurement_sigmas(chi, p, prof)
    n_prf = len(p.prf_stagger_hz)
    n_blind = 0 if r_dot_mps is None else sum(doppler_blind(r_dot_mps, f, p) for f in p.prf_stagger_hz)
    out = {
        "terms": terms,
        "snr_db": db(max(chi, 1e-30)),
        "sub_dwell_snr_db": db(max(chi / n_prf, 1e-30)),
        "n_blind_prfs": int(n_blind),
        "pd_constant_rcs": float(look_detection_probability(chi, p, n_blind)),
        "pd_swerling1": float(look_detection_probability(chi, p, n_blind, swerling=1)),
        "pd_model": (f"{p.prf_m_of_n}-of-{n_prf} PRF sub-dwells, "
                     + ("Swerling I (RCS fluctuates each look)" if swerling == 1 else "non-fluctuating (Marcum)")),
        "sigma": {"range": f"{sig[0]:.1f} m", "azimuth": f"{sig[1] * 1e3:.2f} mrad",
                  "elevation": f"{sig[2] * 1e3:.2f} mrad", "radial velocity": f"{sig[3]:.2f} m/s"},
        "max_range_pd50_km": None,
    }
    # range at which this target would give P_D = 0.5 (bisection on the same equation)
    lo, hi = 100.0, 200_000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if look_detection_probability(snr_linear(mid, sigma_m2, g, p, prof.n_pulses), p, n_blind,
                                      swerling=swerling) > 0.5:
            lo = mid
        else:
            hi = mid
    out["max_range_pd50_km"] = lo / 1000
    return out

