"""Probability of detection and probability of track.

References
----------
[PN]  Physics-informed radar detection note, Section 7, Eqs. (29)-(34).
[AL]  Alpdemir (2022), Eqs. (1)-(4).

The central rule of [PN] Eq. (29) is kept everywhere in this code base:
P_D decides *whether* a report exists, it never replaces the report values.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import chndtr, erfc


def marcum_q1(a, b):
    """First-order Marcum Q function Q_1(a, b) via the non-central chi-square sf."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return 1.0 - chndtr(b * b, 2, a * a)      # ufunc: much faster than ncx2.sf


def pd_nonfluctuating(snr_linear, pfa: float):
    """Square-law detector, non-fluctuating target, Eq. (31):
    P_D = Q_1( sqrt(2 chi), sqrt(-2 ln P_FA) )."""
    snr = np.maximum(np.asarray(snr_linear, float), 0.0)
    return marcum_q1(np.sqrt(2.0 * snr), np.sqrt(-2.0 * np.log(pfa)))


def pd_swerling1(snr_linear, pfa: float):
    """Closed-form single-look Swerling I result P_D = P_FA^(1 / (1 + chi)).

    Use this only when the RCS fluctuation is *not* already sampled per look,
    otherwise the fluctuation would be counted twice.
    """
    snr = np.maximum(np.asarray(snr_linear, float), 0.0)
    return pfa ** (1.0 / (1.0 + snr))


def pd_logistic(snr_db, snr50_db: float = 13.0, slope: float = 0.8):
    """Calibrated logistic curve for early development, Eq. (34)."""
    return 1.0 / (1.0 + np.exp(-slope * (np.asarray(snr_db, float) - snr50_db)))


def pd_erfc_approx(snr_linear, pfa: float):
    """Single-pulse, non-fluctuating approximation quoted in [AL] Eq. (1):
    P_D ~ 0.5 erfc( sqrt(-ln P_FA) - sqrt(chi + 0.5) )."""
    snr = np.maximum(np.asarray(snr_linear, float), 0.0)
    return 0.5 * erfc(np.sqrt(-np.log(pfa)) - np.sqrt(snr + 0.5))


PD_MODELS = {
    "marcum": lambda snr, pfa, **_: pd_nonfluctuating(snr, pfa),
    "swerling1": lambda snr, pfa, **_: pd_swerling1(snr, pfa),
    "erfc": lambda snr, pfa, **_: pd_erfc_approx(snr, pfa),
    "logistic": lambda snr, pfa, **kw: pd_logistic(10 * np.log10(np.maximum(snr, 1e-30)), **kw),
}


# ---------------------------------------------------------------------------
# Medium-PRF dwell: n coherent sub-dwells, M-of-N detection
# ---------------------------------------------------------------------------

def binomial_at_least(p, n: int, m: int):
    """P(at least m successes in n independent trials of probability p)."""
    from math import comb
    p = np.asarray(p, float)
    return sum(comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(m, n + 1)) if m <= n else np.zeros_like(p)


_LAGUERRE = np.polynomial.laguerre.laggauss(64)


def pd_m_of_n(snr_total, pfa: float, n: int, m: int, n_blind: int = 0, model: str = "marcum",
              swerling: int = 0):
    """P_D of one look split into ``n`` coherent sub-dwells (one per PRF of the
    stagger), declared a detection when at least ``m`` of them cross the
    threshold.  ``snr_total`` is the SNR of the whole look; each sub-dwell has
    1/n of the pulses and so ``snr_total / n``.  Sub-dwells whose PRF puts the
    target in the clutter notch (``n_blind``) cannot detect.  Noise is
    independent between sub-dwells; the RCS is the same within one look
    (``swerling=1``: exponential from look to look, averaged here; with the RCS
    already sampled per look use ``swerling=0``).  With n = m = 1 this is the
    single-dwell model."""
    usable = n - n_blind
    if usable < m:
        return np.zeros_like(np.asarray(snr_total, float))
    sub = np.asarray(snr_total, float) / n
    if swerling != 1:
        return binomial_at_least(PD_MODELS[model](sub, pfa), usable, m)
    x, w = _LAGUERRE                                   # E over sigma ~ Exp(1): int f(x) e^-x dx
    vals = [binomial_at_least(PD_MODELS[model](sub * xi, pfa), usable, m) for xi in x]
    return np.clip(sum(wi * v for wi, v in zip(w, vals)), 0.0, 1.0)


def probability_of_detection(snr_linear, pfa: float, model: str = "marcum", **kw):
    """Dispatch to one of the P_D models above (Eq. (30))."""
    return PD_MODELS[model](snr_linear, pfa, **kw)


# ---------------------------------------------------------------------------
# Probability of track ([AL] Eqs. (2)-(4))
# ---------------------------------------------------------------------------

def run_of_misses_probability(pd: float, k: int, l: int) -> float:
    """Probability that at least one run of ``l`` consecutive misses occurs
    in ``k`` independent looks with per-look detection probability ``pd``.

    Standard recursion with P_m = 1 - P_D:
        q_n = 0                       for n < l
        q_l = P_m^l
        q_{n+1} = q_n + (1 - q_{n-l}) * P_D * P_m^l      for n >= l
    A run that first ends at look n+1 needs a detection at look n-l+1, then
    l misses, and no earlier run.  [AL] Eq. (2) prints the last factor as
    (1 - P_m)^l; the run-of-misses form above is the one that matches a
    Monte-Carlo check (see tests/test_detection.py).
    """
    pm = 1.0 - pd
    q = np.zeros(k + 1)
    if l <= k:
        q[l] = pm ** l
    for n in range(l, k):
        q[n + 1] = q[n] + (1.0 - q[n - l]) * pd * pm ** l
    return float(q[k])


def probability_of_track(pd: float, k: int = 30, l: int = 4) -> float:
    """P_track = 1 - z[k]: the track survives k looks without l missed in a row."""
    return 1.0 - run_of_misses_probability(pd, k, l)


def track_probability_curve(R, sigma, c1: float, c2: float):
    """Fitted range/RCS relation of [AL] Eq. (3): P = 1 / (1 + (c2 R^4 / sigma)^c1).
    R in km, sigma in m^2 (the units the constants are fitted in)."""
    return 1.0 / (1.0 + (c2 * np.asarray(R, float) ** 4 / sigma) ** c1)


def fit_track_probability_curve(R_km, p_track, sigma: float):
    """Fit (c1, c2) of Eq. (3) to a sampled P_track(R) curve."""
    def f(r, c1, log_c2):
        return track_probability_curve(r, sigma, c1, np.exp(log_c2))
    (c1, log_c2), _ = curve_fit(f, R_km, p_track, p0=(5.0, np.log(1e-8)), maxfev=20000)
    return c1, float(np.exp(log_c2))


def aggregate_track_probability(p_tracks) -> float:
    """Probability that at least one radar holds the track, [AL] Eq. (4)."""
    p = np.clip(np.asarray(p_tracks, float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - p))
