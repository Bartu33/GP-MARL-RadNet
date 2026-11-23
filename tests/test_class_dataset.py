"""UAV / aircraft / missile targets in the simulator and the class dataset planner."""
from collections import Counter

import numpy as np
import pytest

from src.core.env import RadarNetEnv, _cached_terrain
from src.core.scenario import validate_scenario
from src.core.target_classes import TARGET_CLASSES, variant_spec
from src.core.targets import spawn_class_target
from src.services.class_dataset_service import (
    audit_class_dataset, check_trajectory, plan_class_episodes, propagate,
)
from src.services.synthetic_dataset_service import _validate_plan, generate_dataset

RADAR = {"name": "R1", "position": [0, -5000, 15]}


def class_target(**over):
    return {"target_class": "missile", "variant": "supersonic", "position": [0, 0, 300],
            "velocity": [600, 0, 0], "motion": {"model": "CV"}, **over}


def scenario(*targets):
    return {"radars": [RADAR], "duration_s": 5, "targets": list(targets)}


def test_class_target_validation():
    validate_scenario(scenario(class_target(label=2, rcs_dbsm={"nose": -8, "side": 3, "tail": -5, "top": 5})))
    bad = [
        class_target(variant="multirotor"),                       # variant of another class
        class_target(label=0),                                    # label must follow the class
        class_target(target_type="missile_like"),                 # two label systems on one target
        {k: v for k, v in class_target().items() if k != "motion"},
        class_target(rcs_dbsm={"nose": -8}),
    ]
    for target in bad:
        with pytest.raises(ValueError):
            validate_scenario(scenario(target))
    with pytest.raises(ValueError, match="requires target_class"):
        validate_scenario(scenario({"label": 0, "variant": "supersonic"}))
    with pytest.raises(ValueError, match="cannot be mixed"):
        validate_scenario(scenario(class_target(), {"label": 0}))


def test_spawn_class_target_uses_variant_rcs_and_swerling():
    rng = np.random.default_rng(0)
    for name, cls in TARGET_CLASSES.items():
        for variant, spec in cls["variants"].items():
            s, _ = spawn_class_target(0, name, variant, rng, 7000.0, position=[0, 0, 500],
                                      velocity=[spec.speed_typical_mps[0], 0, 0], motion={"model": "CV"})
            assert (s.label, s.swerling, s.target_class, s.variant) == (cls["label"], spec.swerling, name, variant)
            side = 10 * np.log10(s.rcs.mean_rcs(np.array([0.0, 1.0, 0.0])))
            # LookupTableRCS.synthetic adds a +-1.5 dB lobe ripple (cos 6 psi) to the aspect means
            assert spec.rcs_dbsm["side"][0] - 1.6 <= side <= spec.rcs_dbsm["side"][1] + 1.6


def test_episode_metadata_carries_class_label_space():
    env = RadarNetEnv(validate_scenario(scenario(class_target())), record_episode=True, seed=1)
    env.reset(seed=1)
    meta = env.episode_records["metadata"]
    assert meta["label_space"] == "target_classes_v1"
    assert meta["target_types"] == ["missile/supersonic"]
    assert env.episode_records["truth"][0]["class_label"] == 2


@pytest.fixture(scope="module")
def plan():
    return plan_class_episodes({"episodes": 6, "master_seed": 3})


def test_plan_is_deterministic_balanced_and_split_disjoint(plan):
    again = plan_class_episodes({"episodes": 6, "master_seed": 3})
    assert [e["resolved_configuration_hash"] for e in plan["episodes"]] == \
           [e["resolved_configuration_hash"] for e in again["episodes"]]
    _validate_plan(plan)                         # schema accepted by generate_dataset
    radars = {e["radar_count"] for e in plan["episodes"]}
    targets = {e["target_count"] for e in plan["episodes"]}
    assert radars == {1, 2, 3} and targets == {1, 2, 3}
    classes = Counter(c for e in plan["episodes"] for c in e["target_classes"])
    assert set(classes) == set(TARGET_CLASSES) and max(classes.values()) - min(classes.values()) <= 1
    owner = {}
    for e in plan["episodes"]:
        for group in (e["terrain_group_id"], e["route_group_id"]):
            assert owner.setdefault(group, e["split"]) == e["split"]


def test_planned_trajectories_respect_envelopes(plan):
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        terrain = _cached_terrain(sc)
        for t in sc["targets"]:
            spec = variant_spec(t["target_class"], t["variant"])
            assert t["label"] == TARGET_CLASSES[t["target_class"]]["label"]
            P, V = propagate(t["position"], t["velocity"], t["motion"], sc["duration_s"])
            assert check_trajectory(spec, P, V, terrain) is None
            for seg in t["motion"]["segments"]:
                if seg["model"] == "CT":
                    speed = np.linalg.norm(t["velocity"])
                    assert abs(seg["turn_rate_rad_s"]) <= spec.max_turn_rate_rad_s + 1e-9
                if seg["model"] == "CA":
                    assert np.linalg.norm(seg["acceleration_mps2"]) <= spec.max_long_accel_mps2 * 1.01 + 0.05 \
                        or t["variant"] == "multirotor"


def test_generate_and_audit(tmp_path):
    small = plan_class_episodes({"episodes": 2, "master_seed": 5})
    manifest = generate_dataset(small, tmp_path / "ds")
    assert manifest["generation_status"] == "completed"
    audit = audit_class_dataset(tmp_path / "ds")
    assert audit["targets_checked"] > 0
    assert audit["targets_passing_all_checks"] == audit["targets_checked"]
