"""Track management: gating / outlier rejection, M-of-N confirmation, coasting,
deletion and filter consistency (NEES)."""
import numpy as np
import pytest

from src.core import geometry as geo
from src.core.evaluation import NEES_95, TrackEvaluator, nees
from src.core.radar import RadarParams
from src.models.tracking.ekf import EKF
from src.models.tracking.tracker import GATE_CHI2_4DOF_99, TrackManager

DT = 0.5
R = np.diag([20.0 ** 2, 0.003 ** 2, 0.003 ** 2, 0.5 ** 2])
RADAR = RadarParams(position=(0.0, 0.0, 10.0), heading_deg=0.0)


class Target:
    """Constant-velocity truth used to generate reports."""

    def __init__(self, p=(6000.0, 1000.0, 800.0), v=(-80.0, 20.0, 0.0)):
        self.p = np.array(p, float)
        self.v = np.array(v, float)

    def move(self, dt=DT):
        self.p = self.p + self.v * dt


def report(tgt, rng=None, radar=0, offset=None):
    z = geo.measure(tgt.p, tgt.v, RADAR.pos, RADAR.heading)
    if rng is not None:
        z = z + rng.normal(0, np.sqrt(np.diag(R)))
    if offset is not None:
        z = z + np.asarray(offset, float)
    return {"z": z, "measurement_covariance": R, "radar": radar, "signal_quality": 1.0,
            "truth_id": 0, "is_false_alarm": False}


def step(tm, t, dets, radars=(RADAR,)):
    tm.predict(DT)
    tm.process(t, dets, list(radars))


# ---------------------------------------------------------------- gating
def test_gate_accepts_matching_report_and_rejects_outlier():
    tgt = Target()
    tm = TrackManager()
    tm.process(0.0, [report(tgt)], [RADAR])
    tgt.move()
    tm.predict(DT)
    tr = tm.tracks[0]
    good = report(tgt)
    far = report(tgt, offset=(3000.0, 0.0, 0.0, 0.0))             # 3 km off in range
    assert tr.ekf.mahalanobis2(good["z"], R, RADAR.pos, RADAR.heading) < GATE_CHI2_4DOF_99
    assert tr.ekf.mahalanobis2(far["z"], R, RADAR.pos, RADAR.heading) > GATE_CHI2_4DOF_99
    x_before = tr.x.copy()
    tm.process(DT, [far], [RADAR])
    assert np.allclose(tr.x, x_before)                              # outlier did not update the track
    assert tr.hits == 1
    assert len(tm.tracks) == 2                                      # it started its own tentative track


def test_one_report_updates_at_most_one_track():
    tgt = Target()
    tm = TrackManager()
    tm.process(0.0, [report(tgt), report(tgt, offset=(5.0, 0, 0, 0))], [RADAR])   # two close tracks
    tgt.move()
    step(tm, DT, [report(tgt)])
    assert sum(tr.hits for tr in tm.tracks) == 3                    # 2 births + one update, not two


# ---------------------------------------------------------------- M-of-N confirmation
def test_m_of_n_confirms_after_m_hits_in_window():
    tgt = Target()
    tm = TrackManager(confirm_hits=3, confirm_window=5)
    tm.process(0.0, [report(tgt)], [RADAR])
    for k in (1, 2):
        tgt.move()
        step(tm, k * DT, [report(tgt)])
    assert tm.tracks[0].confirmed


def test_m_of_n_needs_hits_inside_the_window():
    tgt = Target()
    tm = TrackManager(confirm_hits=3, confirm_window=3, tentative_timeout_s=10.0)
    pattern = [1, 0, 0, 1, 0, 0, 1]                                 # hits 3 steps apart: never 3 in 3
    tm.process(0.0, [report(tgt)], [RADAR])
    for k, hit in enumerate(pattern[1:], start=1):
        tgt.move()
        step(tm, k * DT, [report(tgt)] if hit else [])
    assert len(tm.tracks) == 1 and tm.tracks[0].hits == 3
    assert not tm.tracks[0].confirmed


def test_reports_of_two_radars_in_one_step_count_separately():
    radars = [RADAR, RadarParams(name="R2", position=(0.0, 0.0, 10.0), heading_deg=0.0)]
    tgt = Target()
    tm = TrackManager(confirm_hits=3, confirm_window=5)
    tm.process(0.0, [report(tgt, radar=0)], radars)
    tgt.move()
    step(tm, DT, [report(tgt, radar=0), report(tgt, radar=1)], radars)
    tr = tm.tracks[0]
    assert tr.hits == 3 and tr.confirmed                            # 3 reports within 2 steps


def test_old_hits_fall_out_of_the_window():
    tgt = Target()
    tm = TrackManager(confirm_hits=3, confirm_window=4, tentative_timeout_s=10.0)
    tm.process(0.0, [report(tgt)], [RADAR])
    for k, hit in enumerate([1, 0, 0, 0, 1], start=1):              # hits at steps 0, 1, 5
        tgt.move()
        step(tm, k * DT, [report(tgt)] if hit else [])
    assert tm.tracks[0].hits == 3 and not tm.tracks[0].confirmed


def test_window_must_cover_m():
    with pytest.raises(ValueError):
        TrackManager(confirm_hits=3, confirm_window=2)


# ---------------------------------------------------------------- coasting and deletion
def _confirmed_track():
    tgt = Target()
    tm = TrackManager()
    tm.process(0.0, [report(tgt)], [RADAR])
    for k in (1, 2, 3):
        tgt.move()
        step(tm, k * DT, [report(tgt)])
    assert tm.tracks[0].confirmed
    return tm, tgt, 3 * DT


def test_confirmed_track_coasts_through_missed_detections():
    tm, tgt, t = _confirmed_track()
    tr = tm.tracks[0]
    std0 = tr.ekf.pos_std
    for k in range(1, 6):                                           # 2.5 s without reports
        tgt.move()
        step(tm, t + k * DT, [])
    assert tm.tracks == [tr] and tr.coasting(t + 5 * DT)
    assert tr.ekf.pos_std > std0                                    # uncertainty grows while coasting
    assert np.linalg.norm(tr.x[:3] - tgt.p) < 100.0                 # prediction still follows the target


def test_confirmed_track_deleted_after_timeout():
    tm, tgt, t = _confirmed_track()
    n = int(tm.confirmed_timeout_s / DT)
    for k in range(1, n + 2):
        step(tm, t + k * DT, [])
    assert tm.tracks == []


def test_tentative_track_deleted_sooner_than_confirmed():
    tgt = Target()
    tm = TrackManager()
    tm.process(0.0, [report(tgt)], [RADAR])
    n = int(tm.tentative_timeout_s / DT)
    for k in range(1, n + 2):
        step(tm, k * DT, [])
    assert tm.tracks == []
    assert tm.tentative_timeout_s < tm.confirmed_timeout_s


def test_track_deleted_when_uncertainty_explodes():
    tgt = Target()
    tm = TrackManager(max_pos_std_m=200.0, tentative_timeout_s=100.0)
    tm.process(0.0, [report(tgt)], [RADAR])
    for k in range(1, 200):
        step(tm, k * DT, [])
        if not tm.tracks:
            break
    assert tm.tracks == [] and k * DT < 100.0                       # removed by the covariance limit


# ---------------------------------------------------------------- consistency (NEES)
def test_nees_is_zero_for_exact_estimate_and_scales_with_error():
    P = np.diag([100.0, 100.0, 100.0, 4.0, 4.0, 4.0])
    x = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    assert nees(x, P, x[:3], x[3:]) == pytest.approx(0.0)
    assert nees(x + np.array([10.0, 0, 0, 0, 0, 0]), P, x[:3], x[3:]) == pytest.approx(1.0)


def test_consistent_filter_has_nees_near_its_dof():
    """Monte Carlo: an EKF with the true noise levels gives mean NEES about 6
    and about 95 % of the values inside the chi-square(6) interval."""
    rng = np.random.default_rng(1)
    vals = []
    for _ in range(40):
        tgt = Target()
        z0 = report(tgt, rng)
        ekf = EKF.from_detection(z0, RADAR.pos, RADAR.heading, q=1e-6)
        for k in range(60):
            tgt.move()
            ekf.predict(DT)
            ekf.update(report(tgt, rng)["z"], R, RADAR.pos, RADAR.heading)
            if k >= 20:                                             # after the initial transient
                vals.append(nees(ekf.x, ekf.P, tgt.p, tgt.v))
    vals = np.array(vals)
    assert 4.5 < vals.mean() < 7.5
    assert np.mean((vals >= NEES_95[0]) & (vals <= NEES_95[1])) > 0.85


def test_overconfident_filter_has_large_nees():
    rng = np.random.default_rng(2)
    vals = []
    for _ in range(20):
        tgt = Target()
        ekf = EKF.from_detection(report(tgt, rng), RADAR.pos, RADAR.heading, q=1e-6)
        for k in range(60):
            tgt.move()
            ekf.predict(DT)
            ekf.update(report(tgt, rng)["z"], R / 25.0, RADAR.pos, RADAR.heading)   # claims 5x better sensors
            if k >= 20:
                vals.append(nees(ekf.x, ekf.P, tgt.p, tgt.v))
    assert np.mean(vals) > 30.0


def test_evaluator_reports_nees():
    from types import SimpleNamespace
    tgt = SimpleNamespace(id=0, label=0, p=np.zeros(3), v=np.zeros(3))
    tr = SimpleNamespace(id=1, class_probs=None, x=np.array([10.0, 0, 0, 0, 0, 0]),
                         P=np.diag([100.0, 100.0, 100.0, 1.0, 1.0, 1.0]))
    ev = TrackEvaluator(1, dt=DT)
    ev.tracks([tgt], [], [], {0: tr}, {1}, None)
    s = ev.summary()
    assert s["nees_mean"] == pytest.approx(1.0)
    assert s["nees_in_95"] == pytest.approx(0.0)                    # 1.0 < 1.24: too small for chi-square(6)


def test_tracking_report_rows_and_table():
    """Structure of the local-vs-fused study on one short episode."""
    from src.core.scenario import load_scenario
    from src.services import tracking_report as tr
    from src.services.simulation_service import run_episode
    sc = load_scenario("multi_target")
    sc["terrain"] = {"source": "procedural"}
    sc["duration_s"] = 20
    env, _ = run_episode(sc, "central_greedy", seed=0, classifier=None, record=False)
    rows = tr.track_rows(env, "multi_target", "central_greedy", 0)
    assert [r["track"] for r in rows] == [f"R{i} (local)" for i in range(1, 5)] + ["Fused"]
    assert all(0.0 <= r["coverage_net"] <= 1.0 for r in rows)
    import pandas as pd
    table = tr.summary_table(pd.DataFrame(rows))
    assert set(table.track) == {"Fused", "Local (mean of radars)", "Local (best per metric)"}
