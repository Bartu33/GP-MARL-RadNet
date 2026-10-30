"""Detection-level, physics-informed radar model.

Implements the end-to-end algorithm of Section 13 of
[PN] "Physics-Informed Simulation of Three-Dimensional Radar Detections":
geometry -> visibility -> aspect-dependent RCS -> antenna gain -> radar
equation SNR -> P_D gate -> SNR-dependent noisy report -> false alarms ->
resource update.  Equation numbers in comments refer to that note.

All default parameter values are generic textbook-style numbers chosen so
that the scenario plays out over ~10 km.  They are *not* the parameters of
any real system.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geometry as geo
from .detection import probability_of_detection
from .modes import DEFAULT_PROFILES, INTERLEAVE_SEARCH_BEAMS, Mode, ModeProfile, SearchRaster
from .rcs import incidence_in_body, sample_swerling

K_BOLTZMANN = 1.380649e-23


@dataclass
class RadarParams:
    name: str = "R1"
    position: tuple = (0.0, 0.0, 10.0)        # m, ENU
    heading_deg: float = 90.0                 # boresight azimuth, CCW from east
    fc_hz: float = 10e9
    pt_w: float = 1000.0                      # peak transmit power
    gain_dbi: float = 36.0                    # G0 of Eq. (24)
    beamwidth_az_deg: float = 3.0             # full 3-dB beamwidths
    beamwidth_el_deg: float = 6.0
    noise_figure_db: float = 4.0              # F
    losses_db: float = 5.0                    # L
    bandwidth_hz: float = 1e6                 # B_n and waveform bandwidth B
    t0_k: float = 290.0
    pfa: float = 1e-6
    pd_model: str = "marcum"
    r_min_m: float = 300.0
    r_max_m: float = 14000.0                  # instrumented / unambiguous range
    fov_az_deg: float = 180.0                 # half-width of the azimuth field of view
    el_min_deg: float = -2.0
    el_max_deg: float = 60.0
    search_sector_deg: float = 360.0
    search_bars_deg: tuple = (3.0, 10.0)
    # accuracy model, Eqs. (39)-(46)
    k_r: float = 1.0
    k_ang: float = 1.0
    k_rdot: float = 1.0
    floor_r_m: float = 5.0
    floor_ang_rad: float = 2e-4
    floor_rdot_mps: float = 0.2
    quantize: bool = False
    # signal quality, Eq. (49)
    snr_threshold_db: float = 10.0
    snr_good_db: float = 25.0
    # slowly varying bias, Eq. (38)
    bias_rho: float = 0.995
    bias_std: tuple = (2.0, 1e-4, 1e-4, 0.05)
    # false alarms, Eq. (58): expected number per beam position = P_FA * N_cells
    false_alarms_per_beam: float = 0.03
    # resource model, Eqs. (51)-(57)
    e_max_j: float = 1000.0
    efficiency: float = 0.5                    # eta
    e_recovery_w: float = 60.0                 # energy recovered per second
    e_processing_j: float = 1.0
    thermal_a: float = 0.04                    # K per joule radiated
    thermal_b: float = 0.03                    # 1/s cooling rate
    h_ambient_k: float = 300.0
    h_max_k: float = 360.0
    c_max: int = 4                             # simultaneous tracking channels
    profiles: dict = field(default_factory=lambda: dict(DEFAULT_PROFILES))

    @property
    def wavelength(self) -> float:
        return geo.C_LIGHT / self.fc_hz

    @property
    def heading(self) -> float:
        return np.deg2rad(self.heading_deg)

    @property
    def pos(self) -> np.ndarray:
        return np.asarray(self.position, float)


@dataclass
class RadarState:
    energy_j: float
    heat_k: float
    channels_free: int
    mode: Mode = Mode.SEARCH
    online: bool = True
    bias: np.ndarray = field(default_factory=lambda: np.zeros(4))
    raster: SearchRaster | None = None
    last_beams: list = field(default_factory=list)   # [(az, el), ...] of the last look
    last_dwell_s: float = 0.0
    last_energy_used_j: float = 0.0
    time_remaining_frac: float = 1.0

    @classmethod
    def initial(cls, p: RadarParams) -> "RadarState":
        return cls(energy_j=p.e_max_j, heat_k=p.h_ambient_k, channels_free=p.c_max,
                   raster=SearchRaster(p.search_sector_deg, p.beamwidth_az_deg, p.search_bars_deg))


@dataclass
class RadarAction:
    """What the agent asks the radar to do in one simulator step.

    mode : SEARCH, TRACK or IDLE
    beam : (az, el) in the sensor frame for TRACK; ignored for SEARCH/IDLE
    """
    mode: Mode = Mode.SEARCH
    beam: tuple | None = None


# ---------------------------------------------------------------------------
# Physics blocks
# ---------------------------------------------------------------------------

def antenna_gain(d_alpha, d_theta, p: RadarParams):
    """Gaussian two-dimensional pattern, Eq. (24), linear power gain."""
    g0 = 10 ** (p.gain_dbi / 10)
    ba, be = np.deg2rad(p.beamwidth_az_deg), np.deg2rad(p.beamwidth_el_deg)
    return g0 * np.exp(-4 * np.log(2) * (d_alpha ** 2 / ba ** 2 + d_theta ** 2 / be ** 2))


def noise_power(p: RadarParams) -> float:
    """P_n = k_B T_0 B_n F, Eq. (26)."""
    return K_BOLTZMANN * p.t0_k * p.bandwidth_hz * 10 ** (p.noise_figure_db / 10)


def snr_linear(r_m, sigma_m2, gain_lin, p: RadarParams, n_pulses: int = 1):
    """Effective SNR chi of Eq. (27) with G_t = G_r = gain and coherent
    integration gain G_int = n_pulses."""
    lam = p.wavelength
    num = p.pt_w * gain_lin * gain_lin * lam ** 2 * sigma_m2 * n_pulses
    den = (4 * np.pi) ** 3 * np.asarray(r_m, float) ** 4 * 10 ** (p.losses_db / 10) * noise_power(p)
    return num / den


def measurement_sigmas(snr_lin: float, p: RadarParams, profile: ModeProfile) -> np.ndarray:
    """SNR-dependent standard deviations, Eqs. (39)-(46)."""
    s = np.sqrt(2.0 * max(snr_lin, 1e-6))
    d_r = geo.C_LIGHT / (2 * p.bandwidth_hz)                      # Eq. (40)
    d_a = np.deg2rad(p.beamwidth_az_deg)
    d_e = np.deg2rad(p.beamwidth_el_deg)
    d_rdot = p.wavelength / (2 * profile.dwell_s)                  # Eqs. (44)-(45)
    return np.array([
        max(p.floor_r_m, p.k_r * d_r / s),                         # Eq. (41)
        max(p.floor_ang_rad, p.k_ang * d_a / s),                   # Eq. (42)
        max(p.floor_ang_rad, p.k_ang * d_e / s),                   # Eq. (43)
        max(p.floor_rdot_mps, p.k_rdot * d_rdot / s),              # Eq. (46)
    ])


def signal_quality(snr_db: float, p: RadarParams) -> float:
    """Normalised quality score, Eq. (49)."""
    return float(np.clip((snr_db - p.snr_threshold_db) / (p.snr_good_db - p.snr_threshold_db), 0, 1))


def visible(meas, p: RadarParams) -> bool:
    """Range / field-of-view / elevation checks of Section 5 (no terrain occlusion yet)."""
    r, a, e, _ = meas
    return (p.r_min_m <= r <= p.r_max_m
            and abs(a) <= np.deg2rad(p.fov_az_deg)
            and np.deg2rad(p.el_min_deg) <= e <= np.deg2rad(p.el_max_deg))


def resource_vector(s: RadarState, p: RadarParams, profile_dwell_max: float) -> np.ndarray:
    """Normalised resource observation s_resource of Eq. (57)."""
    h_margin = 1 - np.clip((s.heat_k - p.h_ambient_k) / (p.h_max_k - p.h_ambient_k), 0, 1)  # Eq. (54)
    return np.array([
        s.energy_j / p.e_max_j,
        h_margin,
        s.time_remaining_frac,
        s.channels_free / p.c_max,                                  # Eq. (56)
        1.0 if s.last_dwell_s > 0 else 0.0,                          # P_t / P_t,max
        min(1.0, s.last_dwell_s / profile_dwell_max),
    ])


# ---------------------------------------------------------------------------
# One radar step
# ---------------------------------------------------------------------------

def los_status(tgt, p: RadarParams, terrain=None) -> int:
    """0 = outside range / field of view, 1 = visible, 2 = hidden by terrain."""
    if not visible(geo.measure(tgt.p, tgt.v, p.pos, p.heading), p):
        return 0
    if terrain is not None and not terrain.line_of_sight(p.pos, tgt.p):
        return 2
    return 1


def radar_step(targets, action: RadarAction, state: RadarState, p: RadarParams,
               dt: float, rng: np.random.Generator, radar_index: int = 0, terrain=None):
    """Run one look of one radar, following the 13 steps of [PN] Section 13.

    ``targets`` is a sequence of objects with attributes ``id``, ``p``, ``v``,
    ``R_WT`` (body->world rotation), ``rcs`` (an RCSModel) and ``swerling``.

    Returns (detections, state).  Each detection is a dict whose numeric
    report follows the output contract of Section 14; ``truth_id`` and
    ``is_false_alarm`` are simulation truth, to be used only for scoring.
    """
    detections: list[dict] = []
    mode = action.mode if state.online else Mode.OFFLINE

    # --- planned looks: list of (beam centre, profile) ----------------------
    pointer = state.raster.pointer
    search = p.profiles[Mode.SEARCH]
    if mode == Mode.SEARCH:
        looks = [(b, search) for b in state.raster.next_positions(search.beams_per_look)]
    elif mode == Mode.TRACK and action.beam is not None:
        looks = [(tuple(action.beam), p.profiles[Mode.TRACK])]
    elif mode == Mode.INTERLEAVED and action.beam is not None:
        looks = ([(b, search) for b in state.raster.next_positions(INTERLEAVE_SEARCH_BEAMS)]
                 + [(tuple(action.beam), p.profiles[Mode.INTERLEAVED])])
    else:
        looks = []
        if mode in (Mode.TRACK, Mode.INTERLEAVED):
            mode = Mode.IDLE

    # --- resource feasibility (Section 10) --------------------------------
    if looks:
        dwell_plan = sum(pr.dwell_s for _, pr in looks)
        cost = p.pt_w * dwell_plan / p.efficiency + p.e_processing_j
        h_next = state.heat_k + p.thermal_a * p.pt_w * dwell_plan
        if cost > state.energy_j or h_next > p.h_max_k:
            mode, looks = Mode.IDLE, []              # forced rest
            state.raster.pointer = pointer           # the raster did not advance
    beams = [b for b, _ in looks]

    # --- target-originated reports ----------------------------------------
    if beams:
        for tgt in targets:
            truth = geo.measure(tgt.p, tgt.v, p.pos, p.heading)          # steps 1-2
            if not visible(truth, p):                                     # step 3
                continue
            if terrain is not None and not terrain.line_of_sight(p.pos, tgt.p):
                continue                                                  # terrain occlusion
            u_b = incidence_in_body(p.pos, tgt.p, tgt.R_WT)               # step 4
            sigma = sample_swerling(tgt.rcs.mean_rcs(u_b), tgt.swerling, rng)  # step 5
            # the look (beam position and pulse count) giving the highest SNR
            best = max(((float(snr_linear(truth[0], sigma,
                                          antenna_gain(geo.wrap_angle(truth[1] - b[0]), truth[2] - b[1], p),
                                          p, pr.n_pulses)), pr) for b, pr in looks),   # steps 6-7
                       key=lambda c: c[0])
            chi, profile = best
            pd = float(probability_of_detection(chi, p.pfa, p.pd_model))  # step 8
            if rng.random() >= pd:
                continue
            snr_db = 10 * np.log10(max(chi, 1e-30))
            sig = measurement_sigmas(chi, p, profile)                     # step 9
            z = truth + state.bias + rng.normal(0.0, sig)                 # Eq. (35)
            z[1] = geo.wrap_angle(z[1])
            if p.quantize:                                                # Eq. (47)
                d_r = geo.C_LIGHT / (2 * p.bandwidth_hz)
                z[0] = d_r * np.round(z[0] / d_r)
            detections.append(_make_report(z, sig, snr_db, pd, p, radar_index,
                                           truth_id=tgt.id, false_alarm=False))

        # --- false alarms (Section 11) --------------------------------------
        n_fa = rng.poisson(p.false_alarms_per_beam * len(beams))
        for _ in range(n_fa):
            b, profile = looks[rng.integers(len(looks))]
            ba, be = np.deg2rad(p.beamwidth_az_deg), np.deg2rad(p.beamwidth_el_deg)
            z = np.array([rng.uniform(p.r_min_m, p.r_max_m),
                          geo.wrap_angle(b[0] + rng.uniform(-ba / 2, ba / 2)),
                          b[1] + rng.uniform(-be / 2, be / 2),
                          rng.normal(0.0, 1.5)])                          # clutter-like Doppler
            snr_db = p.snr_threshold_db + rng.exponential(2.0)
            sig = measurement_sigmas(10 ** (snr_db / 10), p, profile)
            detections.append(_make_report(z, sig, snr_db, p.pfa, p, radar_index,
                                           truth_id=None, false_alarm=True))

    # --- resource update, Eqs. (52)-(56) ------------------------------------
    dwell = sum(pr.dwell_s for _, pr in looks)
    used = p.pt_w * dwell / p.efficiency + (p.e_processing_j if dwell else 0.0)
    state.energy_j = float(np.clip(state.energy_j - used + p.e_recovery_w * dt, 0, p.e_max_j))
    state.heat_k = float(state.heat_k + p.thermal_a * p.pt_w * dwell
                         - p.thermal_b * (state.heat_k - p.h_ambient_k) * dt)
    state.time_remaining_frac = float(max(0.0, 1.0 - dwell / dt))
    state.bias = p.bias_rho * state.bias + rng.normal(0.0, np.asarray(p.bias_std) * np.sqrt(1 - p.bias_rho ** 2))
    state.mode = mode
    state.last_beams = beams
    state.last_dwell_s = dwell
    state.last_energy_used_j = used
    return detections, state


def _make_report(z, sig, snr_db, pd, p: RadarParams, radar_index, truth_id, false_alarm):
    world = geo.sensor_point_to_world(z[0], z[1], z[2], p.pos, p.heading)
    return {
        "radar": radar_index,
        "detection_valid": 1,
        "z": z,                                          # [r, alpha, theta, r_dot]
        "range_m": float(z[0]),
        "azimuth_rad": float(z[1]),
        "elevation_rad": float(z[2]),
        "radial_velocity_mps": float(z[3]),
        "doppler_hz": float(geo.doppler_shift(z[3], p.wavelength)),
        "snr_db": float(snr_db),
        "detection_probability": float(pd),
        "signal_quality": signal_quality(snr_db, p),
        "measurement_covariance": np.diag(sig ** 2),     # Eq. (37)
        "world_xyz": world,
        "truth_id": truth_id,
        "is_false_alarm": false_alarm,
    }
