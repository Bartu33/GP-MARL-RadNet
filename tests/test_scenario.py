"""Scenario validation and reproducible, inspectable configuration snapshots."""
from copy import deepcopy
from dataclasses import asdict
import json
import re

import numpy as np
import pytest

import src.core.env as env_module
from src.core.env import RadarNetEnv
from src.core.scenario import (
    DEFAULTS, _merge, build_radars, list_scenarios, load_scenario,
    save_snapshot, scenario_snapshot, validate_scenario,
)


def minimal():
    return {"terrain": {"source": "flat"}, "radars": [{}], "targets": [], "duration_s": 2}


@pytest.mark.parametrize("hill,path", [
    (None, "terrain.hills[0]"), ({}, "terrain.hills[0].center_m"),
    ({"center_m": [0]}, "terrain.hills[0].center_m"),
    ({"center_m": [0, True]}, "terrain.hills[0].center_m[1]"),
    ({"center_m": [float("nan"), 0]}, "terrain.hills[0].center_m[0]"),
    ({"center_m": [0, float("inf")]}, "terrain.hills[0].center_m[1]"),
    ({"center_m": "0,0"}, "terrain.hills[0].center_m"),
    ({"center_m": [0, 0]}, "terrain.hills[0].height_m"),
    ({"center_m": [0, 0], "height_m": 0}, "terrain.hills[0].height_m"),
    ({"center_m": [0, 0], "height_m": -1}, "terrain.hills[0].height_m"),
    ({"center_m": [0, 0], "height_m": True}, "terrain.hills[0].height_m"),
    ({"center_m": [0, 0], "height_m": float("inf")}, "terrain.hills[0].height_m"),
    ({"center_m": [0, 0], "height_m": 800}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": 0}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": -1}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": False}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": float("nan")}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": "500"}, "terrain.hills[0].sigma_m"),
    ({"center_m": [0, 0], "height_m": 800, "sigma_m": 500, "typo": 1}, "terrain.hills[0].typo"),
])
def test_invalid_explicit_hills_have_full_paths(hill, path):
    sc = minimal()
    sc["terrain"] = {"source": "procedural", "hills": [hill]}
    with pytest.raises(ValueError, match=re.escape(path)):
        validate_scenario(sc)


@pytest.mark.parametrize("source,hills", [("procedural", None), ("procedural", {}),
                                        ("procedural", "hills"), ("flat", []), ("tiles", []), ("auto", [])])
def test_invalid_hills_container_or_source(source, hills):
    sc = minimal()
    sc["terrain"] = {"source": source, "hills": hills}
    with pytest.raises(ValueError, match="terrain.hills"):
        validate_scenario(sc)


@pytest.mark.parametrize("name", list_scenarios())
def test_existing_scenarios_validate_without_terrain_io(name, monkeypatch):
    def no_terrain(*args, **kwargs):
        pytest.fail("validation must not load terrain")

    monkeypatch.setattr(env_module, "_cached_terrain", no_terrain)
    sc = load_scenario(name)
    assert validate_scenario(sc) == sc
    assert len(build_radars(sc)) == len(sc["radars"])


@pytest.mark.parametrize("field,value", [
    ("dt", 0), ("dt", -1), ("dt", True), ("dt", "0.5"), ("dt", None),
    ("dt", float("nan")), ("dt", float("inf")), ("dt", 10 ** 400),
    ("duration_s", 0), ("duration_s", -1), ("duration_s", False),
    ("duration_s", 0.1), ("duration_s", float("inf")),
    ("seed", -1), ("seed", 1.5), ("seed", True), ("seed", None),
    ("seed", float("nan")), ("seed", "1"),
    ("area_radius_m", 0), ("max_track_slots", 0), ("max_track_slots", 1.5),
    ("share_actions", "false"), ("interleaved_actions", 1),
    ("site", None), ("terrain", None), ("stress", None), ("reward", None),
    ("radars", None), ("radars", []), ("targets", None),
    ("name", 1), ("description", None),
])
def test_invalid_top_level_fields(field, value):
    sc = minimal()
    sc[field] = value
    with pytest.raises(ValueError, match=re.escape(field)):
        validate_scenario(sc)


@pytest.mark.parametrize("field,value", [
    ("r_max_m", -1), ("r_max_m", 100), ("r_min_m", -1),
    ("position", [1, 2]), ("position", [1, 2, None]),
    ("position", [1, True, 3]), ("position", [1, 2, float("nan")]),
    ("position", np.array(1)), ("position", "1,2,3"),
    ("pfa", 0), ("pfa", 1), ("pfa", -0.1), ("pfa", float("inf")),
    ("pd_model", "unknown"), ("pd_model", None),
    ("k_r", -1), ("floor_ang_rad", -1), ("false_alarms_per_beam", -1),
    ("bias_std", [0, 0, 0]), ("bias_std", [0, 0, 0, -1]), ("bias_rho", 2),
    ("fc_hz", 0), ("bandwidth_hz", False), ("pt_w", -1),
    ("beamwidth_az_deg", 0), ("beamwidth_el_deg", 181),
    ("fov_az_deg", 181), ("search_sector_deg", 0), ("search_bars_deg", []),
    ("search_bars_deg", [91]), ("el_max_deg", -3), ("el_min_deg", -91),
    ("efficiency", 0), ("efficiency", 2), ("e_max_j", 0),
    ("e_processing_j", -1), ("e_recovery_w", -1), ("thermal_a", -1),
    ("h_max_k", 200), ("snr_good_db", 9), ("c_max", True), ("c_max", 1.5),
    ("quantize", 1), ("mode", "SEARCH"), ("profiles", {"SEARCH": {}}),
])
def test_invalid_radar_fields_have_full_path(field, value):
    sc = minimal()
    sc["radars"] = [{}, {field: value}]
    with pytest.raises(ValueError, match=re.escape(f"radars[1].{field}")):
        validate_scenario(sc)


@pytest.mark.parametrize("target,path", [
    ({}, "label"), ({"label": 4}, "label"), ({"label": -1}, "label"),
    ({"label": True}, "label"), ({"label": 1.5}, "label"),
    ({"label": "1"}, "label"), ({"label": None}, "label"),
    ({"label": 0, "position": None}, "position"),
    ({"label": 0, "position": [0, 0]}, "position"),
    ({"label": 0, "position": [0, 0, float("inf")]}, "position[2]"),
    ({"label": 0, "motion": "CV"}, "motion"),
])
def test_invalid_targets(target, path):
    sc = minimal()
    sc["targets"] = [target]
    with pytest.raises(ValueError, match=re.escape(f"targets[0].{path}")):
        validate_scenario(sc)


@pytest.mark.parametrize("section,field,value", [
    ("stress", "comm_drop_prob", -0.1), ("stress", "comm_drop_prob", 1.1),
    ("stress", "comm_drop_prob", True), ("stress", "noise_scale", 0),
    ("stress", "noise_scale", float("inf")), ("stress", "fail_radar", 1),
    ("stress", "fail_radar", False), ("stress", "max_reports_per_radar", -1),
    ("stress", "max_reports_per_radar", 1.5), ("stress", "fail_time_s", -1),
    ("stress", "outages", None), ("stress", "outages", [["bad", 0, 1]]),
    ("stress", "outages", [["all", 1, 1]]), ("stress", "outages", [[0, 1]]),
    ("site", "lat", 90), ("site", "lon", 181), ("site", "lat", float("nan")),
    ("terrain", "source", "typo"), ("terrain", "resolution", 1),
    ("terrain", "resolution", True), ("terrain", "resolution", None),
    ("reward", "err_scale_m", 0), ("reward", "w_track", float("nan")),
    ("reward", "w_miss", True), ("radar_defaults", "position", None),
    ("radar_defaults", "profiles", {}), ("radar_defaults", "unknown", 1),
])
def test_invalid_nested_fields(section, field, value):
    sc = minimal()
    sc.setdefault(section, {})[field] = value
    with pytest.raises(ValueError, match=re.escape(f"{section}.{field}")):
        validate_scenario(sc)


@pytest.mark.parametrize("text,path", [
    ("", "scenario"), ("null", "scenario"), ("[]", "scenario"), ("12", "scenario"),
    ("dt: 1\ndt: 2", "dt"),
    ("radars:\n  - pfa: 0.1\n    pfa: 0.2", "radars[0].pfa"),
    ("stress:\n  noise_scale: 1\n  noise_scale: 2", "stress.noise_scale"),
    ("radars: [{}]\nunknown: 1", "scenario.unknown"),
    ("radars: [{}]\nsite: {typo: 1}", "site.typo"),
    ("radars: [{}]\nterrain: {source: flat, typo: 1}", "terrain.typo"),
    ("radars: [{}]\nreward: {typo: 1}", "reward.typo"),
    ("radars: [{}]\nstress: {typo: 1}", "stress.typo"),
    ("1: value", "scenario"), ("radars: &r [*r]", "radars[0]"),
    ("radars: [{}]\nsite: {<<: {lat: 40}}", "site"),
])
def test_yaml_errors(tmp_path, text, path):
    file = tmp_path / "scenario.yaml"
    file.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(path)):
        load_scenario(str(file))


def test_merge_resolves_defaults_replaces_lists_and_detaches_inputs():
    raw = minimal()
    raw["targets"] = [{"label": 0}]
    raw["reward"] = {"w_comm": 0.2}
    before, defaults = deepcopy(raw), deepcopy(DEFAULTS)
    sc = validate_scenario(raw)
    assert sc["reward"]["w_track"] == DEFAULTS["reward"]["w_track"]
    assert sc["reward"]["w_comm"] == 0.2
    sc["targets"][0]["label"] = 3
    assert raw == before and DEFAULTS == defaults
    assert _merge({"targets": [1, 2]}, {"targets": []})["targets"] == []


def test_large_integer_seed_and_numpy_coordinates_are_supported():
    sc = minimal()
    sc["seed"] = 1 << 2048
    sc["radars"] = [{"position": np.array([-10.0, -20.0, 15.0])}]
    env = RadarNetEnv(sc)
    env.reset()
    assert json.loads(json.dumps(env.config_snapshot()))["episode_seed"] == 1 << 2048


def test_compatible_edge_cases_and_partial_radar_defaults():
    sc = minimal()
    sc["duration_s"] = 1.1          # retain rounding, not forced divisibility
    sc["radar_defaults"] = {"r_min_m": 15000}
    sc["radars"] = [{"r_max_m": 16000, "position": [-5, -10, -2]}]
    sc["stress"] = {"outages": [["all", 100, 200]], "fail_time_s": 300,
                    "comm_drop_prob": 1, "max_reports_per_radar": 0}
    resolved = validate_scenario(sc)
    assert resolved["targets"] == []
    assert resolved["radars"][0]["position"] == [-5, -10, -2]


@pytest.mark.parametrize("source", ["flat", "procedural", "tiles", "auto"])
def test_terrain_string_form(source):
    sc = minimal()
    sc["terrain"] = source
    assert validate_scenario(sc)["terrain"] == source


def test_tile_resolution_and_scaled_overflow():
    sc = minimal()
    sc["terrain"] = {"source": "auto", "resolution": 300}
    with pytest.raises(ValueError, match="terrain.resolution"):
        validate_scenario(sc)
    sc["terrain"] = "flat"
    sc["stress"] = {"noise_scale": 1e308}
    sc["radars"] = [{"k_r": 1e308}]
    with pytest.raises(ValueError, match=re.escape("radars[0].k_r")):
        validate_scenario(sc)


@pytest.mark.parametrize("override", [None, -1, True, 1.5])
def test_invalid_environment_precedes_terrain(monkeypatch, override):
    def forbidden(*args, **kwargs):
        pytest.fail("terrain must not be reached for invalid input")

    monkeypatch.setattr(env_module, "_cached_terrain", forbidden)
    sc = minimal()
    if override is None:
        sc["radars"] = [{"r_max_m": -1}]
    with pytest.raises(ValueError):
        RadarNetEnv(sc, seed=override)


def test_dict_environment_detaches_and_revalidates_after_load():
    sc = minimal()
    env = RadarNetEnv(sc)
    sc["radars"][0]["position"] = [1, 2]
    assert env.sc["radars"] == [{}]
    env.reset()
    env.step({"radar_0": 0})
    sc = load_scenario("nominal")
    sc["dt"] = 0
    with pytest.raises(ValueError, match="dt"):
        RadarNetEnv(sc)


def test_snapshot_is_deterministic_serializable_and_scaled_once(tmp_path):
    sc = minimal()
    sc["seed"] = 3
    sc["stress"] = {"noise_scale": 2}
    sc["radars"] = [{"floor_r_m": 7, "position": [np.float64(-1), 0, 15]}]
    before = deepcopy(sc)
    snap = scenario_snapshot(sc, episode_seed=11)
    assert snap == scenario_snapshot(sc, episode_seed=11)
    assert sc == before
    assert snap["scenario"]["radars"][0]["floor_r_m"] == 7
    assert snap["effective_radars"][0]["floor_r_m"] == 14
    assert snap["effective_radars"][0]["profiles"]["SEARCH"]["n_pulses"] == 8
    assert snap["episode_seed"] == 11
    assert snap["scenario_seed"] == snap["terrain_seed"] == 3
    assert snap["rng_layout_version"] == 1
    assert snap["simulation_steps"] == 4
    assert snap["effective_duration_s"] == 2
    assert snap["terrain_provenance"]["actual_source"] is None
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    save_snapshot(snap, a)
    save_snapshot(snap, b)
    assert a.read_bytes() == b.read_bytes()
    restored = json.loads(a.read_text(encoding="utf-8"))
    # The snapshot envelope is not scenario input: only replay its scenario.
    env = RadarNetEnv(restored["scenario"], seed=restored["episode_seed"])
    assert env.radar_params[0].floor_r_m == 14
    with pytest.raises(ValueError, match="unsupported configuration field"):
        validate_scenario(restored)


def test_environment_snapshot_uses_actual_radars_and_current_episode_seed():
    sc = minimal()
    sc["terrain"] = {"source": "procedural", "resolution": 32}
    sc["seed"] = 3
    env = RadarNetEnv(sc, seed=11)
    env.reset(seed=21)
    radar_before = asdict(env.radar_params[0])
    rng_before = deepcopy(env.rng_policy.bit_generator.state)
    snap = env.config_snapshot()
    assert snap == env.config_snapshot()
    assert snap["episode_seed"] == 21
    assert snap["terrain_seed"] == 3
    assert snap["radar_position_reference"] == "terrain_adjusted"
    assert snap["effective_radars"][0]["position"] == list(env.radar_params[0].position)
    assert snap["scenario"]["radars"] == [{}]
    assert snap["terrain_provenance"]["actual_source"] == "procedural"
    assert snap["terrain_provenance"]["missing"] == ["data_hash"]
    assert asdict(env.radar_params[0]) == radar_before
    assert env.rng_policy.bit_generator.state == rng_before


def test_snapshot_roundtrip_preserves_trajectories():
    sc = minimal()
    sc["targets"] = [{"label": i} for i in range(4)]
    a = RadarNetEnv(sc, seed=19)
    a.reset()
    snap = json.loads(json.dumps(a.config_snapshot(), allow_nan=False))
    b = RadarNetEnv(snap["scenario"], seed=snap["episode_seed"])
    b.reset()
    while a.agents:
        a.step({ag: 1 for ag in a.agents})
        b.step({ag: 1 for ag in b.agents})
        for left, right in zip(a.targets, b.targets):
            np.testing.assert_array_equal(left.p, right.p)
            np.testing.assert_array_equal(left.v, right.v)
