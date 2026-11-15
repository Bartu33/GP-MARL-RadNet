"""Config-only planning: stable streams, terrain ownership and bounded geometry retries."""
from copy import deepcopy
import hashlib
import json
import re

import numpy as np
import pytest

from src.core.scenario import ROOT, validate_scenario
from src.services.scenario_planner import export_plan, load_recipe, plan_episodes, terrain_group_id


def recipe():
    return load_recipe(ROOT / "configs" / "datasets" / "planner_demo.yaml")


def test_repeated_plan_is_identical_and_has_expected_quotas():
    raw = recipe()
    before = deepcopy(raw)
    a, b = plan_episodes(raw), plan_episodes(raw)
    assert a == b and raw == before
    assert a["split_counts"] == {"train": 7, "validation": 2, "test": 1}
    assert len({ep["dataset_episode_id"] for ep in a["episodes"]}) == 10
    for ep in a["episodes"]:
        assert ep["generation_status"] == "planned"
        assert validate_scenario(ep["resolved_scenario"]) == ep["resolved_scenario"]
        assert len(ep["resolved_scenario"]["targets"]) == 1
        assert len(ep["resolved_scenario"]["radars"]) == 4


def test_different_master_seed_changes_samples_and_episode_seeds():
    raw = recipe()
    a = plan_episodes(raw)["episodes"]
    raw["master_seed"] += 1
    b = plan_episodes(raw)["episodes"]
    assert [x["episode_seed"] for x in a] != [x["episode_seed"] for x in b]
    assert [x["sampled_values"] for x in a] != [x["sampled_values"] for x in b]


def test_episode_count_extension_preserves_prefix_and_ratios():
    raw = recipe()
    short = plan_episodes(raw)["episodes"]
    raw["episodes"] = 40
    plan = plan_episodes(raw)
    assert plan["episodes"][:10] == short
    assert plan["split_counts"] == {"train": 28, "validation": 6, "test": 6}
    for n in range(1, 41):
        for split, weight in plan["split_ratios"].items():
            count = sum(ep["split"] == split for ep in plan["episodes"][:n])
            assert abs(count - n * weight) <= 1


def test_added_field_and_recipe_key_order_do_not_shift_other_streams():
    raw = recipe()
    distribution = raw["randomization"].pop("stress.noise_scale")
    a = plan_episodes(raw)["episodes"]
    raw["randomization"]["stress.noise_scale"] = distribution
    raw["randomization"] = dict(reversed(list(raw["randomization"].items())))
    b = plan_episodes(raw)["episodes"]
    for left, right in zip(a, b):
        assert left["episode_seed"] == right["episode_seed"]
        assert left["terrain_group_id"] == right["terrain_group_id"]
        assert left["split"] == right["split"]
        for key, value in left["sampled_values"].items():
            assert right["sampled_values"][key] == value


def test_terrain_route_and_scene_groups_are_disjoint_between_splits():
    plan = plan_episodes(recipe())
    for key in ("terrain_group_id", "route_group_id", "geometry_group_id"):
        owners = {}
        for ep in plan["episodes"]:
            assert owners.setdefault(ep[key], ep["split"]) == ep["split"]


def test_same_explicit_surface_different_seed_resolution_or_order_cannot_leak():
    raw = recipe()
    copied = deepcopy(raw["terrain_pools"]["train"][0])
    copied["seed"] = 999
    copied["terrain"]["resolution"] = 256
    hill = copied["terrain"]["hills"][0]
    hill["center_m"] = [float(v) for v in hill["center_m"]]
    hill["height_m"] = float(hill["height_m"])
    raw["terrain_pools"]["test"] = [copied]
    with pytest.raises(ValueError, match="terrain group.*split leakage"):
        plan_episodes(raw)
    terrain = recipe()["terrain_pools"]["train"][0]["terrain"]
    duplicated = deepcopy(terrain)
    duplicated["hills"] *= 2
    combined = deepcopy(terrain)
    combined["hills"][0]["height_m"] *= 2
    assert terrain_group_id(duplicated) == terrain_group_id(combined)
    extra = {"center_m": [100, 200], "height_m": 250, "sigma_m": 500}
    terrain["hills"].append(extra)
    assert terrain_group_id(terrain) == terrain_group_id({**terrain, "hills": terrain["hills"][::-1]})


def test_flat_and_empty_hills_are_the_same_group():
    assert terrain_group_id({"source": "flat"}) == terrain_group_id({"source": "procedural", "hills": []})
    raw = recipe()
    raw["terrain_pools"]["train"] = [{"terrain": {"source": "flat"}}]
    raw["terrain_pools"]["test"] = [{"terrain": {"source": "procedural", "hills": []}, "seed": 99}]
    with pytest.raises(ValueError, match="split leakage"):
        plan_episodes(raw)


def test_fixed_route_replicates_cannot_cross_splits():
    raw = recipe()
    for field in ("targets[0].position", "targets[0].velocity", "targets[0].motion"):
        raw["randomization"].pop(field)
    with pytest.raises(ValueError, match="route group.*split leakage"):
        plan_episodes(raw)


@pytest.mark.parametrize("field,distribution,path", [
    ("dt", {"uniform": {"low": 0.1, "high": 0.5}}, "randomization.dt"),
    ("radars[0].position", {"choice": [[0, 0, 10]]}, "randomization.radars"),
    ("__import__('os').system('bad')", {"choice": [1]}, "randomization"),
    ("targets[0].label", {"uniform": {"low": 0, "high": 3}}, "uniform"),
    ("targets[0].label", {"choice": []}, "choice"),
    ("targets[0].label", {"choice": [True]}, "label"),
    ("targets[0].label", {"normal": [0, 1]}, "normal"),
    ("stress.noise_scale", {"choice": [1], "uniform": {"low": 1, "high": 2}}, "exactly one"),
    ("stress.noise_scale", {"uniform": {"low": 2, "high": 1}}, "high"),
    ("stress.noise_scale", {"uniform": {"low": 0, "high": 1}}, "noise_scale"),
    ("stress.comm_drop_prob", {"uniform": {"low": 0, "high": 2}}, "comm_drop_prob"),
    ("targets[0].position", {"uniform": {"low": [0, 0], "high": [1, 1, 1]}}, "low"),
    ("targets[0].velocity", {"uniform": {"low": [0, True, 0], "high": [1, 1, 1]}}, "low"),
    ("stress.noise_scale", {"uniform": {"low": float("nan"), "high": 1}}, "low"),
    ("targets[0].motion", {"choice": [{"model": "UNKNOWN"}]}, "motion"),
])
def test_invalid_fields_and_distributions_are_rejected(field, distribution, path):
    raw = recipe()
    raw["randomization"][field] = distribution
    with pytest.raises(ValueError, match=re.escape(path)):
        plan_episodes(raw)


@pytest.mark.parametrize("field,value", [("recipe_version", 2), ("master_seed", True),
                                         ("episodes", 0), ("max_attempts", 0),
                                         ("scenario_family", ""), ("split_ratios", {"train": 1})])
def test_invalid_recipe_controls(field, value):
    raw = recipe()
    raw[field] = value
    with pytest.raises(ValueError, match=field):
        plan_episodes(raw)


def test_insufficient_pool_and_unsupported_sources_fail_without_network():
    raw = recipe()
    raw["terrain_pools"]["test"] = []
    with pytest.raises(ValueError, match="nonempty"):
        plan_episodes(raw)
    for source in ("auto", "tiles", "procedural"):
        raw = recipe()
        raw["terrain_pools"]["test"] = [{"terrain": {"source": source}}]
        with pytest.raises(ValueError, match="only flat or explicit"):
            plan_episodes(raw)


def test_invalid_geometry_has_bounded_explainable_retries():
    raw = recipe()
    raw["randomization"]["targets[0].position"] = {"choice": [[0, 0, 0]]}
    raw["max_attempts"] = 3
    with pytest.raises(ValueError, match="episode_index=0.*after 3 attempts.*initial AGL") as error:
        plan_episodes(raw)
    assert str(error.value).count('"attempt":') == 3


def test_geometry_retries_do_not_shift_nonposition_fields():
    raw = recipe()
    raw["randomization"]["targets[0].position"] = {"choice": [[0, 0, 0], [0, 0, 1000]]}
    raw["max_attempts"] = 20
    a = plan_episodes(raw)["episodes"]
    raw["randomization"]["targets[0].position"] = {"choice": [[0, 0, 1000]]}
    b = plan_episodes(raw)["episodes"]
    assert any(ep["geometry_attempts"] > 1 for ep in a)
    for left, right in zip(a, b):
        assert left["episode_seed"] == right["episode_seed"]
        for key in left["sampled_values"]:
            if key != "targets[0].position":
                assert left["sampled_values"][key] == right["sampled_values"][key]


def test_json_roundtrip_hash_and_explicit_export(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = recipe()
    # Named base scenarios resolve independently of the caller working directory.
    plan = plan_episodes(raw)
    assert list(tmp_path.iterdir()) == []
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    export_plan(plan, a)
    export_plan(plan_episodes(raw), b)
    assert a.read_bytes() == b.read_bytes()
    restored = json.loads(a.read_text(encoding="utf-8"))
    assert restored == plan
    for ep in restored["episodes"]:
        encoded = json.dumps(ep["config_snapshot"], sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        assert hashlib.sha256(encoded.encode("utf-8")).hexdigest() == ep["resolved_configuration_hash"]


def test_recipe_duplicate_keys_are_rejected(tmp_path):
    path = tmp_path / "recipe.yaml"
    path.write_text("master_seed: 1\nmaster_seed: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="master_seed.*duplicate"):
        load_recipe(path)


def test_planning_never_constructs_or_steps_environment(monkeypatch):
    from src.core.env import RadarNetEnv
    def forbidden(*args, **kwargs):
        pytest.fail("planning cannot run episodes")
    monkeypatch.setattr(RadarNetEnv, "__init__", forbidden)
    monkeypatch.setattr(RadarNetEnv, "step", forbidden)
    assert len(plan_episodes(recipe())["episodes"]) == 10


def test_class_conditioned_profile_recipe_loads_and_plans_deterministically():
    recipe_path = ROOT / "configs" / "datasets" / "planner_profiles_100.yaml"
    raw = load_recipe(recipe_path)
    before = deepcopy(raw)
    a, b = plan_episodes(raw), plan_episodes(raw)
    assert a == b and raw == before
    assert a["split_counts"] == {"train": 70, "validation": 15, "test": 15}
    assert len(a["episodes"]) == 100

    # Verify balanced class distribution across every split
    by_split = {split: [] for split in ("train", "validation", "test")}
    for ep in a["episodes"]:
        by_split[ep["split"]].append(ep["sampled_values"]["targets[0].label"])

    for split, labels in by_split.items():
        assert len(labels) == a["split_counts"][split]
        # In validation (15) and test (15), every class must appear at least 3 times
        for label in range(4):
            count = labels.count(label)
            assert count >= 3, f"Class {label} has count {count} in {split}"


def test_class_conditioned_sampling_respects_per_class_bounds():
    recipe_path = ROOT / "configs" / "datasets" / "planner_profiles_100.yaml"
    raw = load_recipe(recipe_path)
    plan = plan_episodes(raw)

    for ep in plan["episodes"]:
        label = ep["sampled_values"]["targets[0].label"]
        pos = ep["sampled_values"]["targets[0].position"]
        vel = ep["sampled_values"]["targets[0].velocity"]
        motion = ep["sampled_values"]["targets[0].motion"]

        speed = float(np.hypot(vel[0], vel[1]))

        if label == 0:
            # Class 0: Fast cruise (high altitude, high speed, wide turn / CV / CA)
            assert 1000.0 <= pos[2] <= 2000.0
            assert speed >= 130.0
            if "model" in motion and motion["model"] == "CT":
                assert abs(motion["turn_rate_rad_s"]) == pytest.approx(0.02)
        elif label == 1:
            # Class 1: Agile hover drone (low altitude, agile speed, tight turn / sprint)
            assert 60.0 <= pos[2] <= 300.0
            assert speed <= 20.0
            if "model" in motion and motion["model"] == "CT":
                assert abs(motion["turn_rate_rad_s"]) == pytest.approx(0.08)
        elif label == 2:
            # Class 2: Erratic slow (low-medium altitude, slow speed)
            assert 60.0 <= pos[2] <= 650.0
            assert speed <= 16.0
            if "model" in motion and motion["model"] == "CT":
                assert abs(motion["turn_rate_rad_s"]) == pytest.approx(0.04)
        elif label == 3:
            # Class 3: Drift (medium altitude, low speed passive drift)
            assert 400.0 <= pos[2] <= 1400.0
            assert speed <= 10.0
            assert motion.get("model") in ("CV", "CA")


def test_trajectory_speed_and_agl_bounds_over_30s():
    from src.core.targets import spawn_target, advance
    from src.core.terrain import build_terrain
    recipe_path = ROOT / "configs" / "datasets" / "planner_profiles_100.yaml"
    raw = load_recipe(recipe_path)
    plan = plan_episodes(raw)

    for ep in plan["episodes"]:
        sc = ep["resolved_scenario"]
        c = sc["targets"][0]["label"]
        ground = build_terrain(sc["site"], sc["terrain"], sc["seed"])
        s, m = spawn_target(0, c, np.random.default_rng(ep["episode_seed"]), 6500,
                            position=sc["targets"][0]["position"],
                            velocity=sc["targets"][0]["velocity"],
                            motion=sc["targets"][0]["motion"])
        for _ in range(60):
            advance(s, m, 0.5, terrain=ground)
            vh = float(np.hypot(s.v[0], s.v[1]))
            agl = float(s.p[2] - ground.height(s.p[0], s.p[1]))
            assert agl >= 30.0, f"Terrain clearance violated: AGL={agl}"
            if c == 0:
                assert 110.0 <= vh <= 230.0, f"Class 0 speed out of bounds: {vh}"
            elif c == 1:
                assert vh <= 26.0, f"Class 1 speed out of bounds: {vh}"
            elif c == 2:
                assert vh <= 22.0, f"Class 2 speed out of bounds: {vh}"
            elif c == 3:
                assert vh <= 10.0, f"Class 3 speed out of bounds: {vh}"


def test_by_label_validation_and_error_handling():
    raw = recipe()
    # Invalid: Combining by_label with choice
    raw["randomization"]["targets[0].velocity"] = {
        "choice": [[10, 0, 0]],
        "by_label": {"0": {"choice": [[10, 0, 0]]}}
    }
    with pytest.raises(ValueError, match="by_label cannot be combined"):
        plan_episodes(raw)

    # Invalid: unsupported class label in by_label
    raw["randomization"]["targets[0].velocity"] = {
        "by_label": {"99": {"choice": [[10, 0, 0]]}}
    }
    with pytest.raises(ValueError, match="unsupported target class"):
        plan_episodes(raw)

    # Invalid: non-integer key in by_label
    raw["randomization"]["targets[0].velocity"] = {
        "by_label": {"not_a_number": {"choice": [[10, 0, 0]]}}
    }
    with pytest.raises(ValueError, match="keys must be integer"):
        plan_episodes(raw)

    # Invalid: empty by_label
    raw["randomization"]["targets[0].velocity"] = {"by_label": {}}
    with pytest.raises(ValueError, match="expected a nonempty mapping"):
        plan_episodes(raw)


def test_smoke_cpu_generate_dataset_with_profiles(tmp_path):
    from src.services.synthetic_dataset_service import generate_dataset
    recipe_path = ROOT / "configs" / "datasets" / "planner_profiles_100.yaml"
    raw = load_recipe(recipe_path)
    raw["episodes"] = 4  # small 4-episode smoke run
    plan = plan_episodes(raw)
    manifest = generate_dataset(plan, tmp_path)
    assert manifest["generation_status"] == "completed"
    assert len(manifest["episodes"]) == 4
    for ep in manifest["episodes"]:
        assert ep["generation_status"] == "completed"

