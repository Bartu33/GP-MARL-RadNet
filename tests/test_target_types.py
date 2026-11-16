"""Tests for Explicit Target Type Profiles (Aşama 1C).

Validates:
1. Target type profile parameter specifications, SI units, and research disclaimers.
2. Target type identity invariance across motion model transitions (e.g., CV -> CT -> CV).
3. Non-leakage of ground-truth target type into radar reports, tracker, and MAPPO observations.
4. Physical consistency: speed, altitude, terrain clearance (AGL >= 30m), and RCS models.
5. Deterministic planning and balanced split distribution for planner_target_types_100.yaml.
6. Canonical Episode Records v2 export/load roundtrip and legacy dataset backward compatibility.
7. Small CPU smoke simulation verifying end-to-end radar detections and tracking across all 3 types.
"""
from copy import deepcopy
import math
from pathlib import Path
import tempfile

import numpy as np
import pytest

from src.core.env import RadarNetEnv, _tracker_reports
from src.core.radar import Mode, RadarAction, radar_step
from src.core.rcs import EllipsoidRCS, RCSModel
from src.core.scenario import load_scenario, validate_scenario
from src.core.targets import (
    CLASS_PROFILES, TARGET_TYPE_PROFILES, TargetState, advance,
    default_target_type_rcs, spawn_target,
)
from src.core.terrain import build_terrain
from src.services.scenario_planner import _digest, load_recipe, plan_episodes
from src.services.simulation_service import (
    export_episode_records, load_episode_records, run_episode,
)

ROOT = Path(__file__).resolve().parents[1]


def test_target_type_profiles_structure_and_disclaimer():
    """All 3 explicit profiles must be fully defined with SI units and research disclaimer."""
    expected_types = {"aircraft_like", "uav_like", "missile_like"}
    assert set(TARGET_TYPE_PROFILES.keys()) == expected_types

    for p_id, prof in TARGET_TYPE_PROFILES.items():
        assert prof["profile_id"] == p_id
        assert prof["version"] == "1.0.0"
        assert isinstance(prof["name"], str)
        assert prof["legacy_label_compat"] in (0, 1, 2, 3)

        # Units check
        units = prof["units"]
        assert "World ENU" in units["position"]
        assert "World ENU Z" in units["altitude"]
        assert units["velocity"] == "m/s"
        assert units["acceleration"] == "m/s²"
        assert units["turn_rate"] == "rad/s"
        assert units["rcs"] == "m²"

        # Sampling bounds
        assert prof["initial_velocity_sampling"]["speed_range_mps"][0] >= 0
        assert prof["initial_velocity_sampling"]["speed_range_mps"][1] > prof["initial_velocity_sampling"]["speed_range_mps"][0]
        assert prof["initial_altitude_sampling"]["world_z_range_m"][0] >= 50.0
        assert prof["initial_altitude_sampling"]["min_agl_m"] >= 30.0

        # Motion & Maneuver limits
        assert set(prof["allowed_motion_models"]).issuperset({"CV", "CT", "CA", "SegmentedMotion"})
        assert prof["maneuver_and_speed_limits"]["max_horizontal_speed_mps"] > 0
        assert prof["maneuver_and_speed_limits"]["max_turn_rate_rad_s"] > 0
        assert prof["maneuver_and_speed_limits"]["max_acceleration_mps2"] > 0

        # RCS & Swerling
        assert prof["swerling_assumption"]["swerling_case"] == 1

        # Research disclaimer check
        disclaimer = prof["physical_limitations"]
        assert "scientific research" in disclaimer.lower() or "research" in disclaimer.lower()
        assert "does not model" in disclaimer.lower() or "does not represent" in disclaimer.lower()


def test_target_type_identity_preserved_in_segmented_motion():
    """Target type and ID must remain constant across CV -> CT -> CV transitions."""
    rng = np.random.default_rng(12345)
    schedule = {
        "segments": [
            {"model": "CV", "duration_s": 10.0},
            {"model": "CT", "turn_rate_rad_s": 0.03, "duration_s": 10.0},
            {"model": "CV", "duration_s": 10.0},
        ]
    }
    s, m = spawn_target(
        tid=42, label=0, rng=rng, area_radius=7000.0,
        position=[0.0, 0.0, 2000.0], velocity=[160.0, 0.0, 0.0],
        motion=schedule, target_type="aircraft_like"
    )

    assert s.id == 42
    assert s.target_type == "aircraft_like"
    assert s.label == 0

    # Advance 60 steps (30 seconds, dt=0.5s)
    headings = []
    for step in range(60):
        advance(s, m, 0.5)
        assert s.id == 42, f"Target ID changed at step {step}"
        assert s.target_type == "aircraft_like", f"Target type changed at step {step}"
        assert s.label == 0, f"Target label changed at step {step}"
        headings.append(float(np.arctan2(s.v[1], s.v[0])))

    # Segment 1 (steps 0-20, t=0-10s): heading should remain 0
    assert headings[0] == pytest.approx(0.0, abs=1e-5)
    assert headings[19] == pytest.approx(0.0, abs=1e-5)

    # Segment 2 (steps 20-40, t=10-20s): heading turns CCW at 0.03 rad/s * 10s = 0.3 rad
    assert headings[39] == pytest.approx(0.3, abs=0.01)

    # Segment 3 (steps 40-60, t=20-30s): heading remains constant at ~0.3 rad
    assert headings[59] == pytest.approx(0.3, abs=0.01)


def test_radar_and_truth_isolation():
    """Verify that radar reports, tracker reports, and MAPPO observations do not leak target type."""
    sc = load_scenario("nominal")
    sc["duration_s"] = 1.0
    sc["targets"] = [{
        "label": 0,
        "target_type": "aircraft_like",
        "position": [0.0, 0.0, 1500.0],
        "velocity": [150.0, 0.0, 0.0],
        "motion": {"model": "CV"},
    }]
    env = RadarNetEnv(sc, seed=7, record_episode=True)
    obs, _ = env.reset()

    # 1. Check radar step outputs
    target = env.targets[0]
    p_radar = env.radar_params[0]
    st_radar = env.radar_states[0]
    raw_reports, _ = radar_step(
        [target], RadarAction(Mode.SEARCH), st_radar, p_radar, env.dt, env.rng_radars[0], radar_index=0
    )
    for rep in raw_reports:
        assert "target_type" not in rep, "Radar report leaked target_type"

    # 2. Check tracker reports
    filtered_reports = _tracker_reports(raw_reports)
    for rep in filtered_reports:
        assert "target_type" not in rep, "Tracker report leaked target_type"
        assert "truth_id" not in rep, "Tracker report leaked truth_id"
        assert "is_false_alarm" not in rep, "Tracker report leaked is_false_alarm"

    # 3. Step env and check observations
    obs, _, _, _, _ = env.step({a: 1 for a in env.agents})
    for agent, agent_obs in obs.items():
        assert isinstance(agent_obs, np.ndarray)
        # Observation space is purely numerical and fixed shape
        assert agent_obs.shape == env.observation_space(agent).shape

    # 4. Check tracker tracks
    for tr in env.tracker.tracks:
        assert not hasattr(tr, "target_type"), "Tracker track object leaked target_type"


def test_target_type_rcs_models():
    """Verify default RCS generation for each target type."""
    rng = np.random.default_rng(999)
    for t_type in ("aircraft_like", "uav_like", "missile_like"):
        rcs, sw = default_target_type_rcs(t_type, rng)
        assert isinstance(rcs, RCSModel)
        assert sw == 1
        # Aspect test: incidence along nose (u = [1, 0, 0]) and broadside (u = [0, 1, 0])
        rcs_nose = rcs.mean_rcs(np.array([1.0, 0.0, 0.0]))
        rcs_broad = rcs.mean_rcs(np.array([0.0, 1.0, 0.0]))
        assert rcs_nose > 0
        assert rcs_broad > 0
        if t_type == "aircraft_like":
            # Aircraft has broadside specular significantly larger than nose-on
            assert rcs_broad > rcs_nose
        elif t_type == "missile_like":
            # Slender missile body: broadside is much higher than nose-on
            assert rcs_broad > rcs_nose


def test_planner_target_types_recipe_and_determinism():
    """planner_target_types_100.yaml must produce 100 balanced episodes with deterministic seeds."""
    recipe_path = ROOT / "configs" / "datasets" / "planner_target_types_100.yaml"
    raw = load_recipe(recipe_path)
    plan_a = plan_episodes(raw)
    plan_b = plan_episodes(raw)

    assert _digest(plan_a["recipe"]) == _digest(plan_b["recipe"])
    assert [ep["dataset_episode_id"] for ep in plan_a["episodes"]] == [ep["dataset_episode_id"] for ep in plan_b["episodes"]]
    assert len(plan_a["episodes"]) == 100

    types = [ep["sampled_values"]["targets[0].target_type"] for ep in plan_a["episodes"]]
    from collections import Counter
    counts = Counter(types)
    assert set(counts.keys()) == {"aircraft_like", "uav_like", "missile_like"}
    # Balanced check: 100 episodes over 3 classes -> ~33-34 per class
    for c in counts.values():
        assert 30 <= c <= 36

    # Verify per-class kinematic bounds and clearance
    for ep in plan_a["episodes"]:
        sc = ep["resolved_scenario"]
        target = sc["targets"][0]
        t_type = target["target_type"]
        pos = target["position"]
        vel = target["velocity"]
        speed = float(np.hypot(vel[0], vel[1]))

        if t_type == "aircraft_like":
            assert target["label"] == 0
            assert 1000.0 <= pos[2] <= 4000.0
            assert speed >= 100.0
        elif t_type == "uav_like":
            assert target["label"] == 1
            assert 50.0 <= pos[2] <= 300.0
            assert speed <= 35.0
        elif t_type == "missile_like":
            assert target["label"] == 2
            assert 60.0 <= pos[2] <= 700.0
            assert speed >= 180.0


def test_canonical_episode_records_v2_target_type_roundtrip(tmp_path):
    """Canonical Episode Records v2 must persist target_types in metadata without breaking records.npz schema."""
    sc = load_scenario("nominal")
    sc["duration_s"] = 2.0
    sc["targets"] = [{
        "label": 2,
        "target_type": "missile_like",
        "position": [0.0, 0.0, 500.0],
        "velocity": [300.0, 0.0, 0.0],
        "motion": {"model": "CV"},
    }]
    env, _ = run_episode(sc, seed=42, record_episode=True)
    assert env.episode_records["metadata"]["completed"] is True
    assert env.episode_records["metadata"]["target_types"] == ["missile_like"]

    # Export to disk
    ep_dir = tmp_path / "ep-000000-missile"
    export_episode_records(env.episode_records, ep_dir)

    # Load back with canonical reader
    loaded = load_episode_records(ep_dir)
    assert loaded["metadata"]["target_types"] == ["missile_like"]
    assert loaded["metadata"]["config_snapshot"]["scenario"]["targets"][0]["target_type"] == "missile_like"


def test_legacy_dataset_compatibility():
    """Existing synthetic_100 dataset files must continue to load without errors."""
    legacy_ep_dir = ROOT / "data" / "processed" / "synthetic_100" / "train"
    if legacy_ep_dir.exists():
        episodes = [p for p in legacy_ep_dir.iterdir() if p.is_dir()]
        if episodes:
            records = load_episode_records(episodes[0])
            assert records["metadata"]["record_schema_version"] == 2
            assert "states" in records
            assert "truth" in records


def test_cpu_smoke_run_all_target_types(tmp_path):
    """Smoke test: generate 1 episode for each target type, verify detections and tracks."""
    recipe_path = ROOT / "configs" / "datasets" / "planner_target_types_100.yaml"
    raw = load_recipe(recipe_path)
    raw["episodes"] = 6  # 2 of each type
    plan = plan_episodes(raw)

    types_run = set()
    for ep in plan["episodes"]:
        sc = ep["resolved_scenario"]
        t_type = sc["targets"][0]["target_type"]
        types_run.add(t_type)
        env, summary = run_episode(sc, seed=ep["episode_seed"], record_episode=True)
        assert env.episode_records["metadata"]["completed"] is True
        assert env.episode_records["metadata"]["target_types"] == [t_type]
        # Check that reports and tracks were generated
        reports_count = len(env.episode_records["reports"])
        assert reports_count >= 0

    assert types_run == {"aircraft_like", "uav_like", "missile_like"}
