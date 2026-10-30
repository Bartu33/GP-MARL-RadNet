"""Kinematic ("flight-profile") features computed from a window of fused
track estimates.  The classifier only ever sees tracker output -- never the
true target state -- so it learns from the same noisy data it will face
during a simulation.

Feature families follow the track-behaviour literature (speed stability,
acceleration / jerk, turn rate and curvature, straightness, periodicity,
altitude, hover fraction) plus a track-quality term.
"""
from __future__ import annotations

import numpy as np

FEATURE_NAMES = [
    "speed_mean", "speed_std", "speed_min", "speed_max", "speed_cv",
    "vz_mean", "vz_std", "vz_absmean",
    "acc_mean", "acc_std", "acc_max", "jerk_mean",
    "turn_rate_absmean", "turn_rate_std", "curvature_mean",
    "straightness", "alt_mean", "alt_std", "hover_frac",
    "vz_dom_freq", "vz_dom_power", "speed_dom_power",
    "pos_std",
]


def _dominant(signal: np.ndarray, dt: float) -> tuple[float, float]:
    """Dominant non-DC frequency and its share of the spectral power."""
    x = signal - signal.mean()
    if len(x) < 4 or np.allclose(x, 0):
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    freqs = np.fft.rfftfreq(len(x), dt)
    spec[0] = 0.0
    k = int(np.argmax(spec))
    total = spec.sum()
    return float(freqs[k]), float(spec[k] / total) if total > 0 else 0.0


def window_features(times: np.ndarray, states: np.ndarray, pos_std: float = 0.0) -> np.ndarray:
    """Features of one window.

    times  : (N,) seconds, roughly uniform spacing
    states : (N, 6) [px, py, pz, vx, vy, vz] fused estimates
    """
    t = np.asarray(times, float)
    X = np.asarray(states, float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    p, v = X[:, :3], X[:, 3:]
    speed = np.linalg.norm(v, axis=1)
    hspeed = np.linalg.norm(v[:, :2], axis=1)
    acc = np.gradient(v, dt, axis=0)
    acc_mag = np.linalg.norm(acc, axis=1)
    jerk = np.linalg.norm(np.gradient(acc, dt, axis=0), axis=1)
    heading = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.gradient(heading, dt)
    turn[hspeed < 1.0] = 0.0                    # heading is meaningless while hovering
    curvature = np.abs(turn) / np.maximum(hspeed, 1.0)
    path = np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))
    chord = np.linalg.norm(p[-1] - p[0])
    vz_f, vz_pw = _dominant(v[:, 2], dt)
    _, sp_pw = _dominant(speed, dt)
    return np.array([
        speed.mean(), speed.std(), speed.min(), speed.max(), speed.std() / max(speed.mean(), 1e-3),
        v[:, 2].mean(), v[:, 2].std(), np.abs(v[:, 2]).mean(),
        acc_mag.mean(), acc_mag.std(), acc_mag.max(), jerk.mean(),
        np.abs(turn).mean(), turn.std(), curvature.mean(),
        chord / max(path, 1e-3), p[:, 2].mean(), p[:, 2].std(), float(np.mean(speed < 3.0)),
        vz_f, vz_pw, sp_pw,
        pos_std,
    ])


def sliding_windows(times, states, window_s: float = 10.0, stride_s: float = 2.5, pos_stds=None):
    """Cut a track history into overlapping windows; returns (F, end_times)."""
    t = np.asarray(times, float)
    feats, ends = [], []
    start = t[0]
    while start + window_s <= t[-1] + 1e-9:
        m = (t >= start) & (t <= start + window_s)
        if m.sum() >= 6:
            ps = float(np.mean(np.asarray(pos_stds)[m])) if pos_stds is not None else 0.0
            feats.append(window_features(t[m], np.asarray(states)[m], ps))
            ends.append(start + window_s)
        start += stride_s
    return np.array(feats), np.array(ends)
