"""Verification checks from Section 16 of the physics-informed detection note."""
import numpy as np
import pytest

from src.core import geometry as geo
from src.core.detection import (aggregate_track_probability, pd_nonfluctuating, pd_swerling1,
                                probability_of_track, run_of_misses_probability)
from src.core.modes import DEFAULT_PROFILES, Mode
from src.core.radar import (RadarAction, RadarParams, RadarState, antenna_gain, measurement_sigmas,
                            radar_step, snr_linear)
from src.core.rcs import ConstantRCS, EllipsoidRCS, LookupTableRCS, ScatteringCentreRCS, sample_swerling
from src.core.targets import TargetState


# ---------------------------------------------------------------- 16.1 deterministic
@pytest.mark.parametrize("heading", [0.0, 0.7, np.pi / 2])
def test_axes_convention(heading):
    R = geo.sensor_basis(heading)
    radar = np.zeros(3)
    fwd, right, up = R[:, 0], R[:, 1], R[:, 2]
    r, a, e, _ = geo.measure(1000 * fwd, np.zeros(3), radar, heading)
    assert r == pytest.approx(1000) and a == pytest.approx(0) and e == pytest.approx(0)
    _, a, _, _ = geo.measure(1000 * right, np.zeros(3), radar, heading)
    assert a == pytest.approx(np.pi / 2)          # azimuth positive to the right
    _, _, e, _ = geo.measure(1000 * up, np.zeros(3), radar, heading)
    assert e == pytest.approx(np.pi / 2)


def test_radial_velocity():
    p = np.array([3000.0, 4000.0, 0.0])
    v_away = p / np.linalg.norm(p) * 50
    assert geo.measure(p, v_away, np.zeros(3), 0.3)[3] == pytest.approx(50)
    v_tangent = np.array([-4.0, 3.0, 0.0]) * 10
    assert geo.measure(p, v_tangent, np.zeros(3), 0.3)[3] == pytest.approx(0, abs=1e-9)


def test_jacobian_matches_finite_difference():
    x = np.array([4000.0, -1500.0, 800.0, -60.0, 30.0, 5.0])
    radar, hd = np.array([100.0, 50.0, 10.0]), 0.4
    H = geo.measurement_jacobian(x[:3], x[3:], radar, hd)
    eps = 1e-3
    for j in range(6):
        dx = np.zeros(6)
        dx[j] = eps
        hp = geo.measure((x + dx)[:3], (x + dx)[3:], radar, hd)
        hm = geo.measure((x - dx)[:3], (x - dx)[3:], radar, hd)
        assert np.allclose((hp - hm) / (2 * eps), H[:, j], rtol=1e-4, atol=1e-8)


def test_double_range_minus_12db():
    p = RadarParams()
    g = antenna_gain(0, 0, p)
    d = 10 * np.log10(snr_linear(5000, 1.0, g, p) / snr_linear(10000, 1.0, g, p))
    assert d == pytest.approx(12.04, abs=0.01)


def test_double_rcs_plus_3db():
    p = RadarParams()
    g = antenna_gain(0, 0, p)
    d = 10 * np.log10(snr_linear(5000, 2.0, g, p) / snr_linear(5000, 1.0, g, p))
    assert d == pytest.approx(3.01, abs=0.01)


def test_beam_symmetric_and_3db():
    p = RadarParams()
    g0 = antenna_gain(0, 0, p)
    half = np.deg2rad(p.beamwidth_az_deg) / 2
    assert antenna_gain(half, 0, p) == pytest.approx(antenna_gain(-half, 0, p))
    assert 10 * np.log10(antenna_gain(half, 0, p) / g0) == pytest.approx(-3.01, abs=0.01)


def test_accuracy_decreases_with_snr():
    p = RadarParams()
    prof = DEFAULT_PROFILES[Mode.TRACK]
    lo, hi = measurement_sigmas(10 ** 0.8, p, prof), measurement_sigmas(10 ** 2.5, p, prof)
    assert np.all(lo >= hi)


def test_rcs_models():
    ell = EllipsoidRCS(2.0, 1.0, 0.5)
    # optical-region ellipsoid: nose-on sigma = pi b^2 c^2 / a^2
    assert ell.mean_rcs([1, 0, 0]) == pytest.approx(np.pi * 1.0 * 0.25 / 4.0)
    assert ell.mean_rcs([0, 1, 0]) > ell.mean_rcs([1, 0, 0])
    sphere = EllipsoidRCS(1.0, 1.0, 1.0)
    assert sphere.mean_rcs([0.6, 0.0, 0.8]) == pytest.approx(np.pi)       # sphere: pi a^2
    lut = LookupTableRCS.synthetic()
    assert lut.mean_rcs([0, 1, 0]) > lut.mean_rcs([1, 0, 0])              # side > nose
    single = ScatteringCentreRCS(np.zeros((1, 3)), np.array([2.0]))
    assert single.mean_rcs([1, 0, 0]) == pytest.approx(2.0)
    assert ConstantRCS(0.3).mean_rcs([0, 0, 1]) == 0.3


def test_resources_stay_bounded():
    p = RadarParams()
    s = RadarState.initial(p)
    rng = np.random.default_rng(0)
    for _ in range(400):
        radar_step([], RadarAction(Mode.TRACK, beam=(0.0, 0.05)), s, p, 0.5, rng)
        assert 0 <= s.energy_j <= p.e_max_j
        assert s.heat_k <= p.h_max_k + 1e-9
    assert s.mode in (Mode.TRACK, Mode.IDLE)


# ---------------------------------------------------------------- 16.2 Monte Carlo
def _target_at(r, label=0, sigma=1.0):
    t = TargetState(id=0, label=label, p=np.array([r, 0.0, 300.0]), v=np.array([-50.0, 0, 0]),
                    rcs=ConstantRCS(sigma), swerling=0)
    t.update_attitude()
    return t


def test_empirical_pd_matches_model():
    p = RadarParams(position=(0, 0, 0), heading_deg=0.0, false_alarms_per_beam=0.0)
    n_p = DEFAULT_PROFILES[Mode.TRACK].n_pulses
    r = np.hypot(9000.0, 300.0)
    sigma = 10 ** 1.15 / snr_linear(r, 1.0, antenna_gain(0, 0, p), p, n_p)   # ~11.5 dB SNR
    tgt = _target_at(9000.0, sigma=sigma)
    beam = tuple(geo.measure(tgt.p, tgt.v, p.pos, p.heading)[1:3])
    rng = np.random.default_rng(1)
    hits, pds = 0, []
    n = 3000
    for _ in range(n):
        s = RadarState.initial(p)
        dets, _ = radar_step([tgt], RadarAction(Mode.TRACK, beam=beam), s, p, 0.5, rng)
        hits += len(dets)
        if dets:
            pds.append(dets[0]["detection_probability"])
    chi = snr_linear(np.linalg.norm(tgt.p), sigma, antenna_gain(0, 0, p), p, n_p)
    pd = float(pd_nonfluctuating(chi, p.pfa))
    assert 0.05 < pd < 0.95                       # the test is informative
    assert hits / n == pytest.approx(pd, abs=0.03)


def test_measurement_error_statistics():
    p = RadarParams(position=(0, 0, 0), heading_deg=0.0, false_alarms_per_beam=0.0,
                    bias_std=(0, 0, 0, 0))
    tgt = _target_at(4000.0, sigma=1.0)
    truth = geo.measure(tgt.p, tgt.v, p.pos, p.heading)
    rng = np.random.default_rng(2)
    errs, sig = [], None
    for _ in range(3000):
        s = RadarState.initial(p)
        dets, _ = radar_step([tgt], RadarAction(Mode.TRACK, beam=tuple(truth[1:3])), s, p, 0.5, rng)
        if dets:
            errs.append(dets[0]["z"] - truth)
            sig = np.sqrt(np.diag(dets[0]["measurement_covariance"]))
    errs = np.array(errs)
    assert np.allclose(errs.mean(0), 0, atol=3 * sig / np.sqrt(len(errs)) + 1e-12)
    assert np.allclose(errs.std(0), sig, rtol=0.08)


def test_false_alarm_rate():
    p = RadarParams(false_alarms_per_beam=0.2)
    rng = np.random.default_rng(3)
    s = RadarState.initial(p)
    n_fa, n_beams = 0, 0
    for _ in range(2000):
        dets, s = radar_step([], RadarAction(Mode.SEARCH), s, p, 0.5, rng)
        n_fa += sum(d["is_false_alarm"] for d in dets)
        n_beams += len(s.last_beams)
        s.energy_j, s.heat_k = p.e_max_j, p.h_ambient_k
    assert n_fa / n_beams == pytest.approx(0.2, rel=0.05)


def test_swerling_mean_is_preserved():
    rng = np.random.default_rng(4)
    for case in (1, 3):
        x = [sample_swerling(2.0, case, rng) for _ in range(40000)]
        assert np.mean(x) == pytest.approx(2.0, rel=0.03)


def test_swerling1_closed_form_vs_monte_carlo():
    rng = np.random.default_rng(5)
    chi, pfa = 20.0, 1e-6
    pd_mc = np.mean([rng.random() < pd_nonfluctuating(chi * rng.exponential(), pfa) for _ in range(20000)])
    assert pd_mc == pytest.approx(float(pd_swerling1(chi, pfa)), abs=0.015)


def test_run_of_misses_recursion_monte_carlo():
    rng = np.random.default_rng(6)
    pd, k, l = 0.6, 30, 4
    miss = rng.random((40000, k)) >= pd
    run = np.zeros(len(miss), bool)
    cnt = np.zeros(len(miss), int)
    for j in range(k):
        cnt = np.where(miss[:, j], cnt + 1, 0)
        run |= cnt >= l
    assert run_of_misses_probability(pd, k, l) == pytest.approx(run.mean(), abs=0.01)
    assert probability_of_track(0.99) > probability_of_track(0.5)


def test_aggregate_track_probability():
    assert aggregate_track_probability([0.5, 0.5]) == pytest.approx(0.75)
    assert aggregate_track_probability([]) == 0.0


def test_interleaved_mode_searches_and_tracks_in_one_look():
    from src.core.modes import INTERLEAVE_SEARCH_BEAMS
    p = RadarParams(false_alarms_per_beam=0.0)
    s = RadarState.initial(p)
    rng = np.random.default_rng(7)
    radar_step([], RadarAction(Mode.INTERLEAVED, beam=(0.1, 0.05)), s, p, 0.5, rng)
    assert s.mode == Mode.INTERLEAVED
    assert len(s.last_beams) == INTERLEAVE_SEARCH_BEAMS + 1
    assert tuple(s.last_beams[-1]) == (0.1, 0.05)
    expected = (INTERLEAVE_SEARCH_BEAMS * DEFAULT_PROFILES[Mode.SEARCH].dwell_s
                + DEFAULT_PROFILES[Mode.INTERLEAVED].dwell_s)
    assert s.last_dwell_s == pytest.approx(expected)


def test_forced_idle_does_not_advance_search_raster():
    p = RadarParams()
    s = RadarState.initial(p)
    s.energy_j = 0.0
    before = s.raster.pointer
    radar_step([], RadarAction(Mode.SEARCH), s, p, 0.5, np.random.default_rng(0))
    assert s.mode == Mode.IDLE and s.raster.pointer == before
