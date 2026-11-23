"""Missile salvo planner: realistic profiles inside the class envelopes, time-on-target, zones."""
import numpy as np
import pytest

from src.core.env import _cached_terrain
from src.core.target_classes import variant_spec
from src.services.class_dataset_service import check_trajectory, propagate
from src.services.missile_salvo_service import (
    LOW_PROFILES, PROFILE_VARIANT, align_to_duration, plan_low_level_target, plan_salvo_episodes, resolve_recipe,
)

RECIPE = {"episodes": 2, "master_seed": 11}


@pytest.fixture(scope="module")
def plan():
    return plan_salvo_episodes(RECIPE)


def test_recipe_checks():
    with pytest.raises(ValueError):
        resolve_recipe({"profiles": ["loop_the_loop"]})
    with pytest.raises(ValueError):
        resolve_recipe({"missile_count": [1, 4]})
    assert "weave_dive" not in resolve_recipe({})["profiles"]


def test_salvo_targets_follow_their_variant_and_end_together(plan):
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        terrain = _cached_terrain(sc)
        missiles = [(t, p) for t, p in zip(sc["targets"], e["salvo_profiles"]) if t["target_class"] == "missile"]
        assert len(missiles) == e["missile_count"] >= 1
        for t, profile in missiles:
            assert t["variant"] == PROFILE_VARIANT[profile]
            total = sum(s["duration_s"] for s in t["motion"]["segments"])
            assert abs(total - sc["duration_s"]) < 1e-6              # time on target: every flight ends at T
            spec = variant_spec("missile", t["variant"])
            P, V = propagate(t["position"], t["velocity"], t["motion"], sc["duration_s"])
            assert check_trajectory(spec, P, V, terrain) is None
        names = [z["name"] for z in sc["zones"]]
        assert names[0] == "Zone A" and len(set(names)) == len(names)
        assert all(f >= 0.25 for f in e["planned_visible_fraction"])


def test_align_to_duration_keeps_the_end_point():
    target = {"position": [0.0, 0.0, 5000.0], "velocity": [800.0, 0.0, 0.0],
              "motion": {"segments": [{"model": "CV", "duration_s": 4.0},
                                      {"model": "CA", "acceleration_mps2": [0.0, 0.0, -50.0], "duration_s": 6.0}]}}
    P, _ = propagate(target["position"], target["velocity"], target["motion"], 10.0)
    for total in (7.0, 13.0):
        t2, _ = align_to_duration(target, 10.0, total, 0.5)
        Q, _ = propagate(t2["position"], t2["velocity"], t2["motion"], total)
        assert np.allclose(Q[-1], P[-1], atol=0.5)


@pytest.mark.parametrize("profile", LOW_PROFILES)
def test_low_level_profiles_manoeuvre_inside_the_envelope(profile):
    terrain = _cached_terrain({"site": {"lat": 40.3, "lon": -111.7},
                               "terrain": {"source": "procedural", "resolution": 128,
                                           "hills": [{"center_m": [3000, 2000], "height_m": 150, "sigma_m": 1500}]},
                               "seed": 0})
    rng = np.random.default_rng(3)
    t, info = plan_low_level_target(profile, rng, terrain, (0.0, 0.0), 0.5, (0.0, 3.14))
    assert "terrain_follow" in info["maneuvers"]
    V = np.diff(info["trajectory"], axis=0) / 0.1
    heading = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
    assert np.ptp(heading) > np.radians(5)                     # not a straight line
