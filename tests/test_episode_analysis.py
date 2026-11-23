"""Episode analysis: track-state shares, signal physics, EKF gain, resources, decisions, ground truth."""
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.core.scenario import load_scenario
from src.services import episode_analysis_service as A
from src.services.simulation_service import load_episode_records, run_episode

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data" / "processed" / "missile_routes_pilot5"


def _finite(x):
    if isinstance(x, dict):
        return all(_finite(v) for v in x.values())
    if isinstance(x, list):
        return all(_finite(v) for v in x)
    return not (isinstance(x, float) and not math.isfinite(x))


@pytest.fixture(scope="module")
def live():
    sc = load_scenario("nominal")
    sc["duration_s"] = 12.0
    env, _ = run_episode(sc, "central_greedy", seed=3, classifier=None, record=False, record_episode=True)
    rec = env.episode_records
    rec["metadata"]["dataset_provenance"] = {"policy_name": "central_greedy"}
    return rec


def _as_v2(rec):
    v2 = deepcopy(rec)
    v2["metadata"]["record_schema_version"] = 2
    v2["metadata"]["episode_id"] = rec["metadata"]["episode_id"] + "-v2"
    v2.pop("radar_resources")
    v2.pop("reward_terms")
    return v2


def test_track_state_shares_use_one_denominator_and_count_coasting():
    P = np.eye(6)
    rows = [{"source": "fused", "t_s": 20.0, "last_update_t_s": lu, "confirmed": c, "status": st, "P": P}
            for lu, c, st in [(20.0, True, "confirmed"), (10.0, True, "confirmed_coasting"),
                              (9.5, False, "tentative_coasting"), (20.0, False, "tentative")]]
    b = A.track_state_breakdown({"tracks": rows})
    assert b["total_track_steps"] == 4 and b["confirmed_steps"] == 2 and b["coasting_steps"] == 2
    assert b["confirmed_pct"] == 50.0 and b["coasting_pct"] == 50.0
    assert sum(b["counts"].values()) == b["total_track_steps"]
    assert b["coasting_age_s"]["max"] == 10.5 and b["coasting_over_5s"] == 2


def test_measurement_physics_relations(live):
    m = A.measurement_physics(live)
    assert m["total_reports"] == len(live["reports"]) > 0
    rep = live["reports"][0]
    cross = rep["z"][0] * math.sqrt(rep["R"][1, 1])
    assert m["noise"]["cross_range_az_m"]["min"] <= cross + 0.1 <= m["noise"]["cross_range_az_m"]["max"] + 0.2
    assert m["rcs_hat_dbsm"] is not None and _finite(m)


def test_ekf_information_gain_positive_on_updates(live):
    g = A.ekf_information_gain(live)
    assert g["updates"] > 0
    assert g["information_gain_bits"]["p50"] > 0 and g["delta_sigma_pos_m"]["p50"] > 0


def test_replay_of_v2_records_matches_recorded_v3_resources(live):
    rec_v3 = A.resource_metrics(live)
    assert rec_v3["source"].startswith("recorded")
    rec_v2 = A.resource_metrics(_as_v2(live))
    assert "verified" in rec_v2["source"]
    assert rec_v2["radars"] == rec_v3["radars"]
    for r in rec_v3["radars"]:
        assert 0 <= r["duty_cycle_pct"] <= 100 and 0 <= r["thermal_margin_min_pct"] <= 100
    assert 0 < rec_v3["jain_fairness_energy"] <= 1


def test_reward_terms_rebuild_the_reward(live):
    rb = A.reward_breakdown(live)
    assert rb["max_reconstruction_error"] < 1e-9
    assert rb["reward_sum"] == pytest.approx(sum(float(t["rewards"][0]) for t in live["transitions"]), abs=1e-3)


def test_decision_context_is_numeric_and_reproduces_the_policy(live):
    d = A.decision_context(live, 5, 0)
    assert d["replay_verified"] and d["policy_reproduced"] == 1.0
    assert d["reasons"] and all(isinstance(r, str) for r in d["reasons"])
    for c in d["candidates"]:
        assert {"sigma_pos_m", "age_s", "in_fov", "in_range", "coverable_by"} <= set(c)
    assert "error" in A.decision_context(live, 5, 9)
    assert "error" in A.decision_context(live, 10_000, 0)


def test_ground_truth_metrics_are_consistent(live):
    gt = A.ground_truth_metrics(live)
    d, t = gt["detection"], gt["tracking"]
    assert d["true_detections"] + d["false_alarms"] == d["reports_total"]
    assert d["los_clear_pct"] + d["terrain_blocked_pct"] + d["out_of_coverage_pct"] == pytest.approx(100, abs=0.05)
    for v in (d["pd_overall"], d["pd_given_los"]):
        assert v is None or 0 <= v <= 1
    assert 0 <= t["track_completeness_pct"] <= 100 and _finite(gt)


@pytest.mark.skipif(not PILOT.exists(), reason="missile_routes_pilot5 not generated")
def test_pilot_dataset_has_no_nan_or_bad_percentages():
    for d in sorted(PILOT.glob("*/ep-*")):
        rec = load_episode_records(d)
        out = [A.track_state_breakdown(rec), A.measurement_physics(rec), A.ekf_information_gain(rec),
               A.resource_metrics(rec), A.ground_truth_metrics(rec), A.decision_context(rec, 3, 0)]
        assert _finite(out)
        assert A.replay_episode(rec)["verified"]


@pytest.mark.skipif(not PILOT.exists(), reason="missile_routes_pilot5 not generated")
def test_replay_sections_render_without_exceptions():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"))
    at.run(timeout=60)
    at.radio(key="scen_op_mode").set_value("Dataset Replay").run(timeout=60)
    name = next(o for o in at.selectbox(key="rep_dataset_sel").options if o.startswith("missile_routes_pilot5"))
    at.selectbox(key="rep_dataset_sel").set_value(name).run(timeout=60)
    at.button(key="rep_load_btn").click().run(timeout=120)
    assert len(at.exception) == 0
    at.slider(key="scen_insp_step").set_value(20).run(timeout=120)
    assert len(at.exception) == 0
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    labels = {m.label for m in at.metric}
    assert "Numeric reason" in text and "Ground-Truth Evaluation" in text
    assert {"Coasting", "Azimuth Cross-Range", "Estimated RCS", "Info Gain / Update", "Duty Cycle",
            "Requested Action", "Position NEES", "Episode Reward"} <= labels
    assert all(m.help for m in at.metric if m.label in labels - {"Simulation Steps", "Ground Truth Rows",
               "Reports Generated", "Reports Delivered", "Packet Loss"})
