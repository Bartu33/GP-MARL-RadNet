"""Tests for Multi-Target Synthetic Dataset Generation (Stage 2B)."""
import json
from pathlib import Path
import numpy as np
import pytest

from src.core.env import RadarNetEnv
from src.core.targets import TARGET_TYPE_PROFILES
from src.core.terrain import build_terrain
from src.services.scenario_planner import load_recipe, plan_episodes, _digest
from src.services.simulation_service import load_episode_records, run_episode
from src.services.synthetic_dataset_service import generate_dataset, _validate_plan


def test_multi_target_recipe_validation_and_planning():
    recipe_path = Path("configs/datasets/planner_multi_target_100.yaml")
    assert recipe_path.exists(), "planner_multi_target_100.yaml must exist"
    recipe = load_recipe(recipe_path)
    plan = plan_episodes(recipe)
    
    assert plan["plan_schema_version"] == 1
    assert plan["recipe_version"] == 1
    assert len(plan["episodes"]) == 100
    assert plan["split_counts"] == {"train": 70, "validation": 15, "test": 15}
    
    # Check that each episode has 3 radars and 4 targets
    for ep in plan["episodes"]:
        sc = ep["resolved_scenario"]
        assert len(sc["radars"]) == 3
        assert len(sc["targets"]) == 4
        
        # Verify target identities and profile types
        assert sc["targets"][0]["target_type"] == "aircraft_like"
        assert sc["targets"][0]["label"] == 0
        assert sc["targets"][1]["target_type"] == "uav_like"
        assert sc["targets"][1]["label"] == 1
        assert sc["targets"][2]["target_type"] == "uav_like"
        assert sc["targets"][2]["label"] == 3
        assert sc["targets"][3]["target_type"] == "missile_like"
        assert sc["targets"][3]["label"] == 2
        
        # Verify all targets have valid explicit motion models
        for i, t in enumerate(sc["targets"]):
            assert "position" in t and len(t["position"]) == 3
            assert "velocity" in t and len(t["velocity"]) == 3
            assert "motion" in t
            
        # Verify route_group_id is multi-target route fingerprint
        assert ep["route_group_id"].startswith("route-")


def test_independent_rng_streams_determinism():
    recipe = load_recipe("configs/datasets/planner_multi_target_100.yaml")
    plan1 = plan_episodes(recipe)
    plan2 = plan_episodes(recipe)
    
    # Determinism across identical runs
    assert _digest(plan1["episodes"]) == _digest(plan2["episodes"])
    
    # RNG stream independence: target 0 draws should not depend on target 1..3
    # Compare target 0 draws in a recipe with only target 0 vs with target 0..3
    recipe_t0_only = json.loads(json.dumps(recipe))
    recipe_t0_only["episodes"] = 5
    recipe_t0_only["overrides"]["targets"] = [recipe["overrides"]["targets"][0]]
    recipe_t0_only["randomization"] = {
        k: v for k, v in recipe["randomization"].items() if k.startswith("targets[0]") or k.startswith("stress")
    }
    
    plan_t0 = plan_episodes(recipe_t0_only)
    
    for ep_idx in range(5):
        t0_full = plan1["episodes"][ep_idx]["resolved_scenario"]["targets"][0]
        t0_alone = plan_t0["episodes"][ep_idx]["resolved_scenario"]["targets"][0]
        assert np.allclose(t0_full["position"], t0_alone["position"])
        assert np.allclose(t0_full["velocity"], t0_alone["velocity"])
        assert t0_full["motion"] == t0_alone["motion"]


def test_multi_target_trajectory_speed_and_agl_bounds():
    recipe = load_recipe("configs/datasets/planner_multi_target_100.yaml")
    recipe["episodes"] = 2
    plan = plan_episodes(recipe)
    
    for ep in plan["episodes"]:
        sc = ep["resolved_scenario"]
        terrain = build_terrain(sc["site"], sc["terrain"], seed=sc["seed"])
        env, _ = run_episode(sc, policy_name="central_greedy", seed=ep["episode_seed"],
                             record_episode=True, record=False)
        records = env.episode_records
        
        # Check all truth steps
        for row in records["truth"]:
            tid = row["target_id"]
            p = row["p"]
            v = row["v"]
            h_speed = np.hypot(v[0], v[1])
            ground_z = float(terrain.height(p[0], p[1]))
            agl = p[2] - ground_z
            
            # Clearance check: AGL must be >= 30m throughout
            assert agl >= 29.99, f"Target {tid} AGL {agl:.2f} m < 30 m at step {row['step_index']}"
            
            # Speed bounds check
            target_type = sc["targets"][tid]["target_type"]
            limits = TARGET_TYPE_PROFILES[target_type]["maneuver_and_speed_limits"]
            min_v = limits["min_horizontal_speed_mps"]
            max_v = limits["max_horizontal_speed_mps"]
            assert min_v - 1.0 <= h_speed <= max_v + 1.0, (
                f"Target {tid} ({target_type}) speed {h_speed:.2f} outside [{min_v}, {max_v}]"
            )


def test_multi_target_truth_records_integrity():
    recipe = load_recipe("configs/datasets/planner_multi_target_100.yaml")
    recipe["episodes"] = 1
    plan = plan_episodes(recipe)
    ep = plan["episodes"][0]
    sc = ep["resolved_scenario"]
    
    env, _ = run_episode(sc, policy_name="central_greedy", seed=ep["episode_seed"],
                         record_episode=True, record=False)
    records = env.episode_records
    
    steps = sc["duration_s"] / sc["dt"]
    expected_steps = int(round(steps))
    assert len(records["states"]) == expected_steps + 1
    assert len(records["truth"]) == (expected_steps + 1) * 4
    
    # Check that every step index has unique target IDs [0, 1, 2, 3]
    truth_by_step = {}
    for row in records["truth"]:
        truth_by_step.setdefault(row["step_index"], []).append(row["target_id"])
        
    assert len(truth_by_step) == expected_steps + 1
    for s_idx in range(expected_steps + 1):
        assert sorted(truth_by_step[s_idx]) == [0, 1, 2, 3]


def test_multi_target_dataset_generation_and_inspection(tmp_path):
    recipe = load_recipe("configs/datasets/planner_multi_target_100.yaml")
    recipe["episodes"] = 2
    plan = plan_episodes(recipe)
    
    out_dir = tmp_path / "synthetic_multi_2"
    manifest = generate_dataset(plan, out_dir)
    
    assert len(manifest["episodes"]) == 2
    for ep_entry in manifest["episodes"]:
        assert ep_entry["generation_status"] == "completed"
        ep_dir = out_dir / ep_entry["split"] / ep_entry["dataset_episode_id"]
        rec = load_episode_records(ep_dir)
        assert len(rec["states"]) == 61
        assert len(rec["truth"]) == 61 * 4
        
    # Test resume idempotence
    manifest_resumed = generate_dataset(plan, out_dir, resume=True)
    assert len(manifest_resumed["episodes"]) == 2
    for ep_entry in manifest_resumed["episodes"]:
        assert ep_entry["generation_status"] == "completed"


def test_backward_compatibility_single_target():
    recipe_100 = load_recipe("configs/datasets/planner_100.yaml")
    recipe_100["episodes"] = 3
    plan_100 = plan_episodes(recipe_100)
    
    assert len(plan_100["episodes"]) == 3
    for ep in plan_100["episodes"]:
        assert len(ep["resolved_scenario"]["targets"]) == 1
        assert ep["route_group_id"].startswith("route-")
