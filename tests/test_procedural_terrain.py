"""Tests for Procedural Valley Terrain System (Step 2A).

Verifies geomorphic terrain families, obstacle profiles, physical line-of-sight (LOS),
deterministic seed control, and integration with Terrain and RadarNetEnv.
"""
from copy import deepcopy
import numpy as np
import pytest

from src.core.env import RadarNetEnv
from src.core.procedural_terrain import (
    SUPPORTED_TERRAIN_FAMILIES,
    SUPPORTED_OBSTACLE_PROFILES,
    SUPPORTED_TEXTURE_FAMILIES,
    generate_procedural_terrain_matrix,
    generate_environment_objects,
    build_procedural_valley_terrain,
)
from src.core.scenario import load_scenario, validate_scenario
from src.core.terrain import Terrain, build_terrain, _grid_extent
from src.services.scenario_planner import terrain_group_id, plan_episodes


@pytest.fixture
def grid_coordinates():
    lat, lon = 40.30, -111.70
    x_west, y_north, size = _grid_extent(lat, lon)
    resolution = 128
    cell = size / resolution
    axis = (np.arange(resolution) + 0.5) * cell
    X, Y = np.meshgrid(x_west + axis, y_north - axis)
    return X, Y, x_west, y_north, cell, lat, lon


def test_supported_families_and_profiles_exist():
    assert len(SUPPORTED_TERRAIN_FAMILIES) == 5
    assert "valley_linear_canyon" in SUPPORTED_TERRAIN_FAMILIES
    assert "valley_sinuous_pass" in SUPPORTED_TERRAIN_FAMILIES
    assert "mountain_chokepoint" in SUPPORTED_TERRAIN_FAMILIES
    assert "basin_highlands" in SUPPORTED_TERRAIN_FAMILIES
    assert "ridge_multi_obstacle" in SUPPORTED_TERRAIN_FAMILIES

    assert len(SUPPORTED_OBSTACLE_PROFILES) == 4
    for prof in ("clear", "sparse", "dense", "asymmetric"):
        assert prof in SUPPORTED_OBSTACLE_PROFILES


def test_all_families_generate_valid_matrices(grid_coordinates):
    X, Y, _, _, _, _, _ = grid_coordinates
    for family in SUPPORTED_TERRAIN_FAMILIES:
        for profile in SUPPORTED_OBSTACLE_PROFILES:
            H = generate_procedural_terrain_matrix(X, Y, family, seed=42, obstacle_profile=profile)
            assert H.shape == X.shape
            assert not np.any(np.isnan(H)), f"{family}/{profile} generated NaN"
            assert not np.any(np.isinf(H)), f"{family}/{profile} generated Inf"
            assert np.all(H >= 0.0), f"{family}/{profile} generated negative elevation"
            assert float(np.max(H)) > 400.0, f"{family}/{profile} failed to generate relief"


def test_determinism_and_seed_diversity(grid_coordinates):
    X, Y, _, _, _, _, _ = grid_coordinates
    for family in SUPPORTED_TERRAIN_FAMILIES:
        H_a = generate_procedural_terrain_matrix(X, Y, family, seed=123)
        H_b = generate_procedural_terrain_matrix(X, Y, family, seed=123)
        H_c = generate_procedural_terrain_matrix(X, Y, family, seed=124)

        # Same seed must yield identical matrix
        assert np.array_equal(H_a, H_b), f"{family} failed reproducibility"
        # Different seed must yield distinct matrix
        assert not np.array_equal(H_a, H_c), f"{family} failed seed diversity"


def test_family_diversity(grid_coordinates):
    X, Y, _, _, _, _, _ = grid_coordinates
    matrices = {}
    for family in SUPPORTED_TERRAIN_FAMILIES:
        matrices[family] = generate_procedural_terrain_matrix(X, Y, family, seed=50)

    families = list(SUPPORTED_TERRAIN_FAMILIES)
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            f1, f2 = families[i], families[j]
            assert not np.allclose(matrices[f1], matrices[f2], atol=10.0), \
                f"Families {f1} and {f2} are too similar"


def test_obstacle_profiles_alter_geometry(grid_coordinates):
    X, Y, _, _, _, _, _ = grid_coordinates
    H_clear = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=77, obstacle_profile="clear")
    H_sparse = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=77, obstacle_profile="sparse")
    H_dense = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=77, obstacle_profile="dense")
    H_asym = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=77, obstacle_profile="asymmetric")

    assert not np.array_equal(H_clear, H_sparse)
    assert not np.array_equal(H_sparse, H_dense)
    assert not np.array_equal(H_clear, H_asym)

    # Dense obstacle profile should have higher mean elevation than clear
    assert np.mean(H_dense) > np.mean(H_clear)


def test_line_of_sight_physical_occlusion(grid_coordinates):
    X, Y, x_west, y_north, cell, _, _ = grid_coordinates
    # Generate linear canyon running North-South (valley_angle_deg = 0)
    H = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=10,
                                          params={"valley_width_m": 3000.0, "relief_height_m": 900.0},
                                          obstacle_profile="clear")
    terrain = Terrain(H, x_west, y_north, cell, "procedural")

    # 1. Straight line across the valley from West ridge to East ridge:
    # West at (-5000, 0, 100), East at (5000, 0, 100).
    # Since the mountain crest rises to ~900m, this low ray must be occluded by the mountain.
    assert not terrain.line_of_sight([-5000, 0, 100], [5000, 0, 100]), \
        "Low ray across the mountain crest must be occluded"

    # 2. High ray clearing above the mountain crests:
    # West at (-5000, 0, 1500), East at (5000, 0, 1500).
    assert terrain.line_of_sight([-5000, 0, 1500], [5000, 0, 1500]), \
        "High ray above the mountain peak must be clear"

    # 3. Ray traveling straight along the valley corridor:
    # South entrance at (0, -4000, 80), North exit at (0, 4000, 80).
    # Valley floor is clear at 0m, so 80m AGL must have clear line of sight.
    assert terrain.line_of_sight([0, -4000, 80], [0, 4000, 80]), \
        "Ray along the unobstructed valley floor must be clear"


def test_build_terrain_integration():
    site = {"name": "Test Site", "lat": 40.30, "lon": -111.70}
    spec = {
        "source": "procedural",
        "resolution": 128,
        "family": "valley_sinuous_pass",
        "valley_width_m": 2500.0,
        "relief_height_m": 800.0,
        "obstacle_profile": "sparse",
    }
    terrain = build_terrain(site, spec, seed=99)
    assert isinstance(terrain, Terrain)
    assert terrain.source == "procedural"
    assert terrain.H.shape == (128, 128)
    assert terrain.H.min() >= 0.0
    assert terrain.H.max() > 700.0

    # Payload export for 3D viewer
    payload = terrain.to_payload(max_px=128)
    assert payload["source"] == "procedural"
    assert "data" in payload
    assert payload["w"] == 128
    assert payload["h"] == 128


def test_scenario_validation_accepts_procedural_family():
    sc = load_scenario("nominal")
    sc["terrain"] = {
        "source": "procedural",
        "resolution": 128,
        "family": "mountain_chokepoint",
        "valley_width_m": 2000,
        "relief_height_m": 900,
        "obstacle_profile": "dense",
    }
    validated = validate_scenario(sc)
    assert validated["terrain"]["family"] == "mountain_chokepoint"
    assert validated["terrain"]["obstacle_profile"] == "dense"


def test_terrain_group_id_with_family():
    t1 = {
        "source": "procedural",
        "family": "valley_linear_canyon",
        "valley_width_m": 3000,
        "relief_height_m": 800,
        "obstacle_profile": "clear",
        "seed": 10,
    }
    t2 = deepcopy(t1)
    t3 = deepcopy(t1)
    t3["seed"] = 11

    gid1 = terrain_group_id(t1)
    gid2 = terrain_group_id(t2)
    gid3 = terrain_group_id(t3)

    assert gid1.startswith("terrain-")
    assert gid1 == gid2, "Identical parametric specs must yield identical group ID"
    assert gid1 != gid3, "Different seed must yield different group ID"


def test_env_simulation_with_procedural_valley():
    sc = load_scenario("nominal")
    sc["duration_s"] = 5.0
    sc["terrain"] = {
        "source": "procedural",
        "resolution": 128,
        "family": "valley_sinuous_pass",
        "valley_width_m": 2500,
        "relief_height_m": 800,
        "obstacle_profile": "clear",
    }
    env = RadarNetEnv(sc, seed=42)
    obs, info = env.reset(seed=42)

    # Check that radar masts stand on the local procedural ground
    for p in env.radar_params:
        x, y, z = p.position
        ground_h = float(env.terrain.height(x, y))
        assert z >= ground_h, "Radar antenna must be at or above ground"

    # Step simulation
    actions = {agent: 1 for agent in env.agents}
    next_obs, rewards, terms, truncs, infos = env.step(actions)
    assert len(next_obs) == len(env.possible_agents)


def test_supported_texture_families():
    assert len(SUPPORTED_TEXTURE_FAMILIES) == 6
    expected = (
        "rocky_mountain",
        "green_forest_valley",
        "dry_canyon",
        "snowy_highlands",
        "grassland_valley",
        "mixed_rock_vegetation",
    )
    for tf in expected:
        assert tf in SUPPORTED_TEXTURE_FAMILIES


def test_scenario_validation_accepts_texture_families():
    sc = load_scenario("nominal")
    for tf in SUPPORTED_TEXTURE_FAMILIES:
        sc["terrain"] = {
            "source": "procedural",
            "family": "valley_linear_canyon",
            "texture_family": tf,
        }
        val = validate_scenario(sc)
        assert val["terrain"]["texture_family"] == tf

    sc["terrain"]["texture_family"] = "invalid_style"
    with pytest.raises(ValueError, match="terrain.texture_family"):
        validate_scenario(sc)


def test_environment_objects_generation(grid_coordinates):
    X, Y, x_west, y_north, cell, _, _ = grid_coordinates
    H = generate_procedural_terrain_matrix(X, Y, "valley_linear_canyon", seed=42)

    # 1. Clear profile must yield 0 objects
    objs_clear = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="clear", seed=42)
    assert len(objs_clear) == 0

    # 2. Sparse profile must yield 70-120 objects
    objs_sparse = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="sparse", seed=42)
    assert 70 <= len(objs_sparse) <= 120

    # 3. Dense profile must yield 300-500 objects
    objs_dense = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="dense", seed=42)
    assert 300 <= len(objs_dense) <= 500

    # 4. Asymmetric profile must yield 180-300 objects
    objs_asym = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="asymmetric", seed=42)
    assert 180 <= len(objs_asym) <= 300

    # 5. Determinism
    objs_dense_b = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="dense", seed=42)
    assert objs_dense == objs_dense_b
    objs_dense_c = generate_environment_objects(H, x_west, y_north, cell, obstacle_profile="dense", seed=43)
    assert objs_dense != objs_dense_c


def test_environment_objects_decorativeness_and_surface_fit():
    site = {"name": "Test Site", "lat": 40.30, "lon": -111.70}
    spec = {
        "source": "procedural",
        "resolution": 128,
        "family": "valley_linear_canyon",
        "valley_width_m": 2500.0,
        "relief_height_m": 800.0,
        "obstacle_profile": "dense",
        "texture_family": "green_forest_valley",
        "seed": 42,
    }
    terrain = build_terrain(site, spec, seed=42)
    assert hasattr(terrain, "visual_objects")
    assert len(terrain.visual_objects) >= 300

    # Verify that every object sits on the terrain surface (h matches terrain.height within 2m)
    for obj in terrain.visual_objects:
        assert obj["type"] in ("tree", "rock", "shrub")
        calc_h = terrain.height(obj["x"], obj["y"])
        assert abs(obj["h"] - calc_h) < 2.0, f"Object {obj} not seated on terrain surface"

    # Verify that visual objects are purely decorative (LOS is computed strictly by Terrain.H)
    p0 = [-3000, 0, 950]
    p1 = [3000, 0, 950]
    los_val = terrain.line_of_sight(p0, p1)
    # Even if visual objects are emptied, LOS result must be strictly unchanged
    orig_objs = terrain.visual_objects
    terrain.visual_objects = []
    assert terrain.line_of_sight(p0, p1) == los_val
    terrain.visual_objects = orig_objs


def test_terrain_payload_contains_visual_and_physical_metadata():
    site = {"name": "Test Site", "lat": 40.30, "lon": -111.70}
    spec = {
        "source": "procedural",
        "resolution": 128,
        "family": "mountain_chokepoint",
        "valley_width_m": 2200.0,
        "relief_height_m": 850.0,
        "obstacle_profile": "sparse",
        "texture_family": "dry_canyon",
        "seed": 105,
    }
    terrain = build_terrain(site, spec, seed=105)
    payload = terrain.to_payload(max_px=128)

    assert payload["terrain_family"] == "mountain_chokepoint"
    assert payload["texture_family"] == "dry_canyon"
    assert payload["obstacle_profile"] == "sparse"
    assert "visual_objects" in payload
    assert len(payload["visual_objects"]) >= 70
    assert payload["physical_vs_visual"]["physical_radar_occlusion"] == "Terrain.H"
    assert payload["physical_vs_visual"]["visual_rf_attenuation_modelled"] is False


def test_physical_consistency_multi_radar_and_detections():
    # Verify that in a typical valley scenario with 3 radars, radar mast clearance is maintained,
    # target AGL >= 30m, and non-zero detections occur.
    sc = load_scenario("nominal")
    sc["duration_s"] = 10.0
    sc["terrain"] = {
        "source": "procedural",
        "resolution": 128,
        "family": "valley_linear_canyon",
        "valley_width_m": 3500.0,
        "relief_height_m": 700.0,
        "obstacle_profile": "sparse",
        "texture_family": "rocky_mountain",
        "seed": 77,
    }
    # Set 3 radars
    sc["radars"] = sc["radars"][:3]
    assert len(sc["radars"]) == 3

    env = RadarNetEnv(sc, seed=77)
    obs, info = env.reset(seed=77)

    # 1. Check radar mast height and that antennas are strictly above ground
    for p in env.radar_params:
        x, y, z = p.position
        ground_h = float(env.terrain.height(x, y))
        assert z >= ground_h, f"Radar at {p.position} is underground (ground={ground_h})"
        mast_h = z - ground_h
        assert mast_h >= 10.0, f"Radar mast height {mast_h}m too small"

    # 2. Check target clearance AGL >= 30m
    for tgt in env.targets:
        ground_h = float(env.terrain.height(tgt.p[0], tgt.p[1]))
        agl = tgt.p[2] - ground_h
        assert agl >= 29.9, f"Target at {tgt.p} has insufficient clearance AGL={agl}m (ground={ground_h})"

    # 3. Step environment and check that radars produce real tracks / detections
    actions = {agent: 1 for agent in env.agents}
    for _ in range(4):
        next_obs, rewards, terms, truncs, infos = env.step(actions)
    assert len(env.tracker.tracks) > 0, "Radars should not be 100% blocked in open valley corridor"

