"""Evaluation metrics: continuity, track loss vs. missed detection, coverage, outages."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.evaluation import METRIC_INFO, TrackEvaluator, in_beam, runs
from src.core.radar import RadarParams
from src.core.scenario import load_scenario
from src.services.simulation_service import run_episode


def test_runs_counts_interruptions_and_reacquisition():
    s = runs([0, 0, 1, 1, 0, 1, 1, 1, 0])
    assert s["longest"] == 3
    assert s["interruptions"] == 2             # one reacquired gap + the open gap at the end
    assert s["interrupted"] == 2
    assert s["reacquired"] == [1]


def test_runs_never_tracked_is_not_an_interruption():
    s = runs([0, 0, 0])
    assert s == {"longest": 0, "interruptions": 0, "interrupted": 0, "reacquired": []}


def test_in_beam_uses_half_power_width():
    p = RadarParams(beamwidth_az_deg=3.0, beamwidth_el_deg=6.0)
    beams = [(0.0, np.deg2rad(5.0))]
    assert in_beam((1000, np.deg2rad(1.4), np.deg2rad(5.0), 0), beams, p)
    assert not in_beam((1000, np.deg2rad(1.6), np.deg2rad(5.0), 0), beams, p)
    assert not in_beam((1000, 0.0, np.deg2rad(8.5), 0), beams, p)


def _tr(i, label=0):
    probs = np.eye(3)[label]                         # UAV / aircraft / missile
    return SimpleNamespace(id=i, class_probs=probs, x=np.zeros(6))


def test_missed_update_is_not_a_track_loss_but_deletion_is():
    tgt = [SimpleNamespace(id=0, label=0, target_class="uav", p=np.zeros(3), v=np.zeros(3))]
    ev = TrackEvaluator(1, dt=0.5)
    ev.tracks(tgt, [], [], {0: _tr(5)}, {5}, None)      # tracked by track 5
    ev.tracks(tgt, [], [], {}, {5}, None)               # not matched, track 5 still exists (coasting)
    assert ev.track_losses == 0
    ev.tracks(tgt, [], [], {}, set(), None)             # track 5 deleted by the tracker
    assert ev.track_losses == 1
    ev.tracks(tgt, [], [], {0: _tr(9)}, {9}, None)      # reacquired by a new track: not an id switch
    s = ev.summary()
    assert ev.id_switches == 0
    assert s["interruptions"] == 1 and s["reacquire_s"] == pytest.approx(1.0)
    assert s["correct_class_continuity"] == pytest.approx(0.5)


def test_switch_to_another_existing_track_is_an_id_switch():
    tgt = [SimpleNamespace(id=0, label=1, target_class="aircraft", p=np.zeros(3), v=np.zeros(3))]
    ev = TrackEvaluator(1, dt=0.5)
    ev.tracks(tgt, [], [], {0: _tr(3, 1)}, {3, 4}, None)
    ev.tracks(tgt, [], [], {0: _tr(4, 1)}, {3, 4}, None)
    assert ev.id_switches == 1 and ev.track_losses == 0


def _episode(name):
    sc = load_scenario(name)
    sc["terrain"] = {"source": "procedural"}             # offline
    _, s = run_episode(sc, "central_greedy", seed=0, classifier=None, record=False)
    return s


def test_outage_scenario_forces_track_loss_and_reacquisition():
    nominal, outage = _episode("nominal"), _episode("track_loss")
    assert outage["track_losses"] >= 1
    assert outage["track_losses"] > nominal["track_losses"]
    assert outage["detection_loss"] > nominal["detection_loss"]
    assert np.isfinite(outage["reacquire_s"])


def test_summary_has_documented_metrics_in_range():
    s = _episode("nominal")
    assert "track_loss_rate" not in s                  # old name was only 1 - coverage
    for k in ("track_coverage", "coverage_in_range", "detection_loss", "correct_class_continuity"):
        assert 0.0 <= s[k] <= 1.0, k
    assert s["longest_track_s"] <= 120.0
    for k in s:
        assert k in METRIC_INFO, f"{k} has no definition for the dashboard"
