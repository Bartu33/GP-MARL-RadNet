"""Missile terminal-dive planner, tracker settings and the confirmation-dwell policy."""
import numpy as np
import pytest

from src.core.env import _cached_terrain
from src.core.scenario import validate_scenario
from src.core.target_classes import variant_spec
from src.models.baselines.policies import POLICIES
from src.services.class_dataset_service import audit_class_dataset, check_trajectory, propagate
from src.services.missile_dive_service import (
    DIVE_PROFILES, EVASIVE_PROFILES, LOW_LEVEL_PROFILES, plan_missile_dive_episodes,
)
from src.services.synthetic_dataset_service import _validate_plan, generate_dataset


@pytest.fixture(scope="module")
def plan():
    return plan_missile_dive_episodes({"episodes": 5, "master_seed": 9})


def test_plan_deterministic_and_accepted(plan):
    again = plan_missile_dive_episodes({"episodes": 5, "master_seed": 9})
    assert [e["resolved_configuration_hash"] for e in plan["episodes"]] == \
           [e["resolved_configuration_hash"] for e in again["episodes"]]
    _validate_plan(plan)
    assert [e["dive_profile"] for e in plan["episodes"]] == list(DIVE_PROFILES)
    assert {e["radar_count"] for e in plan["episodes"]} == {1, 2, 3}


def test_dives_respect_envelope_and_descend(plan):
    spec = variant_spec("missile", "terminal_dive")
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        terrain = _cached_terrain(sc)
        (t,) = sc["targets"]
        assert (t["target_class"], t["variant"], t["label"]) == ("missile", "terminal_dive", 2)
        P, V = propagate(t["position"], t["velocity"], t["motion"], sc["duration_s"])
        assert check_trajectory(spec, P, V, terrain) is None
        agl = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
        assert agl[0] >= 2000 and agl[-1] <= 450                  # high start, low end
        assert np.hypot(*(P[-1, :2] - np.array(e["defended_point"]))) < 1000
        if e["dive_profile"] != "dive_pullout":
            assert V[-1, 2] < -0.3 * np.linalg.norm(V[-1])        # still diving at the end


def test_radars_on_local_high_ground(plan):
    off = np.linspace(-1500.0, 1500.0, 9)
    OX, OY = np.meshgrid(off, off)
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        terrain = _cached_terrain(sc)
        assert sc["tracker"]["q"] > 0
        for r in sc["radars"]:
            x, y = r["position"][:2]
            assert np.mean(terrain.height(x + OX, y + OY) <= terrain.height(x, y)) >= 0.8
            bars = r["search_bars_deg"]
            assert np.allclose(np.diff(bars), 6.0, atol=0.11)    # contiguous bars, one beamwidth apart


def test_tracker_block_validation():
    base = {"radars": [{"name": "R1"}], "targets": []}
    validate_scenario({**base, "tracker": {"q": 100.0, "confirm_hits": 3}})
    for bad in ({"unknown": 1}, {"q": 0}, {"confirm_hits": 2.5}):
        with pytest.raises(ValueError):
            validate_scenario({**base, "tracker": bad})
    assert "tracker" not in validate_scenario(base)               # absent stays absent (snapshots unchanged)


def test_generated_dive_is_tracked(tmp_path):
    assert "central_greedy_confirm" in POLICIES
    small = plan_missile_dive_episodes({"episodes": 2, "master_seed": 9})
    manifest = generate_dataset(small, tmp_path / "ds", policy_name="central_greedy_confirm")
    assert manifest["generation_status"] == "completed"
    audit = audit_class_dataset(tmp_path / "ds")
    assert audit["targets_passing_all_checks"] == audit["targets_checked"] == 2
    assert max(t["tracked_step_share"] for e in audit["episodes"] for t in e["targets"]) > 0.2


EXPECTED_PHASES = {
    "popup_dive": ["terrain_follow", "pop_up", "dive"],
    "barrel_roll_dive": ["cruise", "barrel_roll_dive"],
    "dogleg_serpentine": ["terrain_follow", "dogleg_turn", "serpentine", "pop_up", "dive"],
    "break_turn_dive": ["cruise", "break_turn", "boost_reacquire", "dive", "dive_break", "terminal_dive"],
    "random_jink_dive": ["cruise", "random_jinks"],
}


def test_evasive_profiles_fly_their_manoeuvres():
    spec = variant_spec("missile", "terminal_dive")
    plan = plan_missile_dive_episodes({"episodes": 5, "master_seed": 21, "profiles": list(EVASIVE_PROFILES)})
    _validate_plan(plan)
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        terrain = _cached_terrain(sc)
        (t,) = sc["targets"]
        assert e["maneuvers"][0] == EXPECTED_PHASES[e["dive_profile"]]
        assert all(seg["model"] == "CA" for seg in t["motion"]["segments"])   # autopilot pieces
        P, V = propagate(t["position"], t["velocity"], t["motion"], sc["duration_s"])
        assert check_trajectory(spec, P, V, terrain) is None
        agl = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
        if e["dive_profile"] in LOW_LEVEL_PROFILES:
            assert agl[0] < 120 and agl.max() > 350                 # hidden low, then the pop-up
            assert all(r.get("el_min_deg") == -5.0 for r in sc["radars"])
        else:
            assert agl[0] >= 2000
        assert 100 <= agl[-1] <= 800                                # stopped just before impact
        assert np.hypot(*(P[-1, :2] - np.array(e["defended_point"]))) < 1500
        A = np.diff(V, axis=0) / 0.1
        U = V[:-1] / np.linalg.norm(V[:-1], axis=1)[:, None]
        a_n = np.linalg.norm(A - (A * U).sum(1)[:, None] * U, axis=1)
        assert a_n.max() > 8 * 9.81                                 # a real high-g manoeuvre happens
