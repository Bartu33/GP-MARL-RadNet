"""Observation of the fusion-centre task manager (prototype of Section 2.6 of the proposal).

o_t = [ {tau_j}_{j=1..K}, {rho_i}_{i=1..N}, Q_t, last answers ]

* ``tracks``  (K x len(TRACK_FEATURES)): summary tau_j of the fused tracks, in
  the same order as ``env.slots`` (confirmed first, then the most uncertain);
* ``radars``  (N x len(RADAR_FEATURES)): summary rho_i of every radar, incl.
  its local scheduler's queue and its answers to the last requests;
* ``quality`` (N x K x len(QUALITY_FEATURES)): expected service quality of
  radar i for track j if it were asked to update it now;
* masks: ``track_mask`` (slot filled), ``quality_mask`` (radar i can cover
  track j), ``nis_valid`` (the track has been updated at least once since its
  initialisation, so its NIS exists).

Everything is computed from what the centre really has: the fused track table
(estimates and covariances), the radars' published parameters and resource
reports, and the schedulers' answers.  Simulation truth is never used.  Each
feature is scaled to roughly [0, 1] (or [-1, 1] for directions); the scale
constants are listed next to the feature names and in
doc/research/gozlem_gereksinimleri.md.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from .modes import Mode
from .radar import antenna_gain, measurement_sigmas, snr_linear
from .targets import CLASS_NAMES

REF_RCS_M2 = 0.1            # reference target for the predicted SNR (the centre does not know the true RCS)
CHI2_GATE = 13.28           # 99 % gate of chi-square(4): NIS scale

TRACK_FEATURES = [
    "valid",                 # 1 = slot holds a track
    "x / area_radius", "y / area_radius", "z / 5000 m",
    "vx / 300 m/s", "vy / 300 m/s", "vz / 300 m/s",
    "log10(1 + pos_std) / 3",
    "logdet(P_pos) / 50",    # log determinant of the 3x3 position covariance (m^6, natural log)
    "confirmed",
    "age / 60 s (clipped)",
    "AoI: time since last update / 10 s (clipped)",
    "NIS of last update / 13.28 (clipped at 2)",
    "hits / 20 (clipped)",
    "radars that updated it in the last 3 s / N",
    "class entropy / log(C) (1 = unknown)",
    "pending request at some radar",
    # SNR of the reports associated with the track (all radars, last SNR_HISTORY reports)
    "instantaneous SNR: last report / 30 dB",
    "rolling mean SNR (linear average, in dB) / 30 dB",
    "SNR standard deviation in dB / 10 dB",
    "detection rate of requested dwells",       # corrects the SNR selection bias (misses carry no SNR)
    "dwell history available",
]
RADAR_FEATURES = [
    "online",
    "x / area_radius", "y / area_radius",
    "sin(heading)", "cos(heading)",
    "energy / E_max", "heat margin", "free tracking channels / c_max",
    "scheduler queue length / c_max",
    "last answers: accepted", "delayed", "partial", "rejected",   # fractions of last step's requests
    "reports delivered last step / 20 (clipped)",
    "tracking now (track or interleaved dwell)",
    "share of steps spent on track dwells (cumulative)",
]
QUALITY_FEATURES = [
    "coverable (range, field of view, elevation)",
    "predicted SNR for a 0.1 m^2 target / 30 dB",
    "predicted cross-range sigma / 1000 m",
    "expected information gain 0.5*(logdet P - logdet P+) / 10 nats",
    "energy cost of a track dwell / E_max",
    "last SNR measured by this radar on this track / 30 dB (0 = never)",
    "time since this radar last updated the track / 10 s (1 = never)",
    "detection rate of this radar's requested dwells on the track (0.5 = no history)",
]


def snr_statistics(tr, radar: int | None = None):
    """(last, rolling mean, std) of the SNR in dB of the reports associated with a
    track, from one radar or from all radars; None when there is no report.

    The mean is taken over linear power and converted back to dB (a mean of dB
    values would be biased); the spread is the standard deviation of the dB
    values.  Only detected looks carry an SNR, so these numbers describe the
    detections, not every look -- see ``dwell_detection_rate``."""
    hist = tr.snr.get(radar, []) if radar is not None else sorted(
        (x for h in tr.snr.values() for x in h), key=lambda x: x[0])[-10:]
    if not hist:
        return None
    db = np.array([v for _, v in hist])
    mean_db = 10 * np.log10(np.mean(10 ** (db / 10)))
    return float(db[-1]), float(mean_db), float(db.std())


def dwell_detection_rate(tr, radar: int | None = None):
    """Share of requested dwells on this track that produced a detection (None if none)."""
    hist = tr.dwells.get(radar, []) if radar is not None else [x for h in tr.dwells.values() for x in h]
    return float(np.mean([d for _, d in hist])) if hist else None


def _coverable(p, tr) -> bool:
    r, a, e, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
    return (abs(a) <= np.deg2rad(p.fov_az_deg) and p.r_min_m <= r <= p.r_max_m
            and np.deg2rad(p.el_min_deg) <= e <= np.deg2rad(p.el_max_deg))


def track_summary(env, tr) -> tuple[np.ndarray, bool]:
    area = float(env.sc["area_radius_m"])
    P = tr.ekf.P
    _, logdet = np.linalg.slogdet(P[:3, :3])
    nis = tr.ekf.last_nis
    if tr.class_probs is None:
        ent = 1.0
    else:
        q = np.clip(np.asarray(tr.class_probs, float), 1e-12, 1)
        ent = float(-(q * np.log(q)).sum() / np.log(len(CLASS_NAMES)))
    recent = sum(1 for t in tr.updated_by.values() if env.t - t < 3.0)
    pending = any(q.req.track_id == tr.id for sch in env.schedulers for q in sch.queue)
    f = np.array([
        1.0,
        tr.x[0] / area, tr.x[1] / area, tr.x[2] / 5000.0,
        tr.x[3] / 300.0, tr.x[4] / 300.0, tr.x[5] / 300.0,
        np.log10(1 + tr.ekf.pos_std) / 3.0,
        logdet / 50.0,
        float(tr.confirmed),
        min(1.0, (env.t - tr.born_t) / 60.0),
        min(1.0, (env.t - tr.last_update_t) / 10.0),
        min(2.0, nis / CHI2_GATE) if nis is not None else 0.0,
        min(1.0, tr.hits / 20.0),
        recent / env.n_radars,
        ent,
        float(pending),
        *_snr_features(tr),
    ])
    return f, nis is not None


def _snr_features(tr) -> list:
    st = snr_statistics(tr)
    last, mean, std = (st[0] / 30.0, st[1] / 30.0, min(2.0, st[2] / 10.0)) if st else (0.0, 0.0, 0.0)
    rate = dwell_detection_rate(tr)
    return [last, mean, std, rate if rate is not None else 0.5, float(rate is not None)]


def radar_summary(env, i: int) -> np.ndarray:
    p, st, sch = env.radar_params[i], env.radar_states[i], env.schedulers[i]
    area = float(env.sc["area_radius_m"])
    answers = env.last_answers.get(i, [])
    frac = np.zeros(4)
    if answers:
        for a in answers:
            frac[["accepted", "delayed", "partial", "rejected"].index(a.response.value)] += 1
        frac /= len(answers)
    h_margin = 1 - np.clip((st.heat_k - p.h_ambient_k) / (p.h_max_k - p.h_ambient_k), 0, 1)
    return np.array([
        float(st.online),
        p.pos[0] / area, p.pos[1] / area,
        np.sin(p.heading), np.cos(p.heading),
        st.energy_j / p.e_max_j, h_margin, st.channels_free / p.c_max,
        len(sch.queue) / p.c_max,
        *frac,
        min(1.0, env.last_report_counts.get(i, 0) / 20.0),
        float(st.mode in (Mode.TRACK, Mode.INTERLEAVED)),
        min(1.0, env.track_load[i] / max(env.step_count * p.profiles[Mode.TRACK].dwell_s, 1e-9)),
    ])


def service_quality(env, i: int, tr) -> tuple[np.ndarray, bool]:
    """Expected result if radar i updated track tr with a full track dwell now."""
    p, st = env.radar_params[i], env.radar_states[i]
    if not (st.online and _coverable(p, tr)):
        return np.zeros(len(QUALITY_FEATURES)), False
    prof = p.profiles[Mode.TRACK]
    r = float(np.linalg.norm(tr.x[:3] - p.pos))
    chi = float(snr_linear(r, REF_RCS_M2, antenna_gain(0, 0, p), p, prof.n_pulses))
    sig = measurement_sigmas(chi, p, prof)
    R = np.diag(sig ** 2)
    H = geo.measurement_jacobian(tr.x[:3], tr.x[3:], p.pos, p.heading)
    P = tr.ekf.P
    info = np.linalg.inv(P) + H.T @ np.linalg.solve(R, H)
    _, ld_prior = np.linalg.slogdet(P)
    _, ld_post = np.linalg.slogdet(np.linalg.inv(info))
    gain = 0.5 * (ld_prior - ld_post)
    cost = (p.pt_w * prof.dwell_s / p.efficiency + p.e_processing_j) / p.e_max_j
    return np.array([
        1.0,
        10 * np.log10(max(chi, 1e-12)) / 30.0,
        min(1.0, r * sig[1] / 1000.0),
        min(1.0, gain / 10.0),
        cost,
        *_pair_history(env, i, tr),
    ]), True


def _pair_history(env, i: int, tr) -> list:
    """What the centre knows about radar i on this track: its last SNR, how long
    ago it updated the track, and how often its requested dwells found it."""
    st = snr_statistics(tr, i)
    last_t = tr.updated_by.get(i)
    rate = dwell_detection_rate(tr, i)
    return [st[0] / 30.0 if st else 0.0,
            min(1.0, (env.t - last_t) / 10.0) if last_t is not None else 1.0,
            rate if rate is not None else 0.5]


def central_observation(env) -> dict:
    """Structured observation of the task manager; ``vector`` is the flat
    concatenation used by a policy network (fixed size for a given scenario)."""
    K, N = env.K, env.n_radars
    tracks = np.zeros((K, len(TRACK_FEATURES)))
    track_mask = np.zeros(K, bool)
    nis_valid = np.zeros(K, bool)
    quality = np.zeros((N, K, len(QUALITY_FEATURES)))
    quality_mask = np.zeros((N, K), bool)
    for k, tr in enumerate(env.slots):
        if tr is None:
            continue
        tracks[k], nis_valid[k] = track_summary(env, tr)
        track_mask[k] = True
        for i in range(N):
            quality[i, k], quality_mask[i, k] = service_quality(env, i, tr)
    radars = np.array([radar_summary(env, i) for i in range(N)])
    vector = np.concatenate([tracks.ravel(), radars.ravel(), quality.ravel(),
                             track_mask.astype(float), quality_mask.astype(float).ravel(),
                             [env.step_count / env.max_steps]]).astype(np.float32)
    return {"tracks": tracks, "track_mask": track_mask, "nis_valid": nis_valid, "radars": radars,
            "quality": quality, "quality_mask": quality_mask, "vector": vector,
            "track_ids": [tr.id if tr is not None else None for tr in env.slots]}


def observation_size(K: int, N: int) -> int:
    return (K * len(TRACK_FEATURES) + N * len(RADAR_FEATURES) + N * K * len(QUALITY_FEATURES)
            + K + N * K + 1)
