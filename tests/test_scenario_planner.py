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
