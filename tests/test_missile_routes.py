"""Missile-focused class dataset: multi-missile routes, rugged terrain, reference zones."""
import numpy as np
import pytest

from src.core.scenario import validate_scenario
from src.services.class_dataset_service import RUGGED_PROFILES, plan_class_episodes, resolve_recipe

RECIPE = {"episodes": 3, "master_seed": 7, "target_count": [2, 3],
          "terrain": {"rugged_profiles": list(RUGGED_PROFILES)}, "missile_focus": {"missile_count": [2, 3]}}


def zone(**over):
    return {"name": "Zone A", "kind": "zone", "position": [0, 0], "radius_m": 400, "height_m": 100, **over}


def test_zone_validation():
    base = {"radars": [{"name": "R1", "position": [0, -5000, 15]}]}
    sc = validate_scenario({**base, "zones": [zone(), zone(name="Facility 1", kind="facility")]})
    assert len(sc["zones"]) == 2
    assert "zones" not in validate_scenario(base)                      # absent unless given: old hashes stay valid
    for bad in (zone(kind="bunker"), zone(radius_m=0), zone(position=[1]), zone(extra=1)):
        with pytest.raises(ValueError):
            validate_scenario({**base, "zones": [bad]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_scenario({**base, "zones": [zone(), zone()]})


def test_recipe_checks():
    with pytest.raises(ValueError):
        resolve_recipe({"terrain": {"rugged_profiles": ["nope"]}})
    with pytest.raises(ValueError):
        resolve_recipe({"missile_focus": {"route_patterns": ["spiral"]}})
    assert resolve_recipe({})["missile_focus"] is None                  # default behaviour is unchanged


@pytest.fixture(scope="module")
def plan():
    return plan_class_episodes(RECIPE)


def test_plan_routes_zones_and_terrain(plan):
    assert plan == plan_class_episodes(RECIPE)                          # deterministic
    for e in plan["episodes"]:
        sc = e["resolved_scenario"]
        missiles = [t for t in sc["targets"] if t["target_class"] == "missile"]
        assert len(missiles) == e["missile_count"] >= 2
        assert e["rugged_profile"] in RUGGED_PROFILES
        names = [z["name"] for z in sc["zones"]]
        assert names[0] == "Zone A" and len(set(names)) == len(names)
        anchor = np.asarray(sc["zones"][0]["position"], float)
        assert np.allclose(anchor, e["anchor"])
        assert {"in", "out"} >= set(e["route_phases"])
        if e["route_pattern"] == "converging":
            assert set(e["route_phases"]) == {"in"}
        if e["route_pattern"] == "diverging":
            assert set(e["route_phases"]) == {"out"}


def test_viewer_payload_carries_zones(plan):
    from src.services.simulation_service import _zones_payload
    sc = plan["episodes"][0]["resolved_scenario"]
    z = _zones_payload(sc)
    assert [x["name"] for x in z] == [x["name"] for x in sc["zones"]]
    assert all(set(x) == {"name", "kind", "pos", "radius", "height"} for x in z)
