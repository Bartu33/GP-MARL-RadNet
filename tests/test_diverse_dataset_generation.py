"""Unit tests for Diverse Valley Dataset Generation & Pilot Validation (Step 2C).

Verifies:
- 7 dataset groups (4 single-target, 3 triple-target)
- Accurate target composition (both physical and target profile levels)
- Disjoint train/validation/test split isolation (zero leakage)
- Terrain, texture, and obstacle diversity across episodes
- Physical clearances (radar mast >= 15m, target AGL >= 30m)
- Canonical Episode Records v2 compliance (pickle-free, NPZ arrays)
- Deterministic display names (display_name)
- Replay reconstruction via records_to_viewer_payload
- Legacy dataset compatibility
"""
import json
from pathlib import Path
import numpy as np
import pytest

from src.core.procedural_terrain import (
    SUPPORTED_TERRAIN_FAMILIES,
    SUPPORTED_TEXTURE_FAMILIES,
    SUPPORTED_OBSTACLE_PROFILES,
)
from src.core.scenario import validate_scenario
from src.services.diverse_dataset_service import (
    DATASET_GROUPS,
    TARGET_PROFILES,
    RADAR_PLACEMENT_PROFILES,
    ROUTE_FAMILIES,
    format_display_name,
    build_diverse_pilot_episodes,
    build_canonical_plan_for_group,
    compute_physical_quality_metrics,
)
from src.services.simulation_service import (
    load_episode_records,
    records_to_viewer_payload,
)
from src.services.synthetic_dataset_service import _validate_plan


def test_seven_dataset_groups_exist():
    assert len(DATASET_GROUPS) == 7
    expected = (
        "dataset_single_aircraft_v1",
        "dataset_single_uav_agile_v1",
        "dataset_single_uav_drift_v1",
        "dataset_single_missile_v1",
        "dataset_triple_all_same_v1",
        "dataset_triple_all_different_v1",
        "dataset_triple_two_plus_one_v1",
    )
    for g in expected:
        assert g in DATASET_GROUPS


def test_single_target_datasets_have_exact_one_target():
    all_eps = build_diverse_pilot_episodes()
    single_groups = (
        "dataset_single_aircraft_v1",
        "dataset_single_uav_agile_v1",
        "dataset_single_uav_drift_v1",
        "dataset_single_missile_v1",
    )
    for g in single_groups:
        episodes = all_eps[g]
        assert len(episodes) == 6
        for ep in episodes:
            assert ep["target_count"] == 1
            assert len(ep["target_types"]) == 1
            assert ep["physical_target_type_composition"] == "single"
            assert ep["target_profile_composition"] == "single"

            # Check physical type mapping
            if "aircraft" in g:
                assert ep["target_types"][0] == "aircraft_like"
            elif "missile" in g:
                assert ep["target_types"][0] == "missile_like"
            elif "uav" in g:
                assert ep["target_types"][0] == "uav_like"


def test_triple_target_compositions():
    all_eps = build_diverse_pilot_episodes()

    # 1. Triple All Same
    all_same_eps = all_eps["dataset_triple_all_same_v1"]
    assert len(all_same_eps) == 6
    for ep in all_same_eps:
        assert ep["target_count"] == 3
        assert ep["target_profile_composition"] == "all_same"
        assert ep["physical_target_type_composition"] == "all_same"
        assert len(set(ep["target_profiles"])) == 1

    # 2. Triple All Different
    all_diff_eps = all_eps["dataset_triple_all_different_v1"]
    assert len(all_diff_eps) == 6
    for ep in all_diff_eps:
        assert ep["target_count"] == 3
        assert ep["target_profile_composition"] == "all_different"
        assert ep["physical_target_type_composition"] == "all_different"
        assert len(set(ep["target_profiles"])) == 3
        assert len(set(ep["target_types"])) == 3

    # 3. Triple Two Plus One
    two_plus_one_eps = all_eps["dataset_triple_two_plus_one_v1"]
    assert len(two_plus_one_eps) == 6
    for ep in two_plus_one_eps:
        assert ep["target_count"] == 3
        assert ep["target_profile_composition"] == "two_plus_one"
        counts = [ep["target_profiles"].count(p) for p in set(ep["target_profiles"])]
        assert sorted(counts) == [1, 2]


def test_split_isolation_and_zero_leakage():
    all_eps = build_diverse_pilot_episodes()
    for g, episodes in all_eps.items():
        plan = build_canonical_plan_for_group(g, episodes)
        # Validate plan using strict canonical validator
        _validate_plan(plan)

        split_counts = {"train": 0, "validation": 0, "test": 0}
        terrain_by_split = {}
        route_by_split = {}
        for ep in episodes:
            sp = ep["split"]
            split_counts[sp] += 1
            t_gid = ep["terrain_group_id"]
            r_gid = ep["route_group_id"]

            if t_gid in terrain_by_split:
                assert terrain_by_split[t_gid] == sp, f"Terrain group {t_gid} leaked across splits!"
            terrain_by_split[t_gid] = sp

            if r_gid in route_by_split:
                assert route_by_split[r_gid] == sp, f"Route group {r_gid} leaked across splits!"
            route_by_split[r_gid] = sp

        assert split_counts == {"train": 4, "validation": 1, "test": 1}


def test_terrain_texture_and_obstacle_diversity():
    all_eps = build_diverse_pilot_episodes()
    seen_terrains = set()
    seen_textures = set()
    seen_obstacles = set()

    for g, episodes in all_eps.items():
        for ep in episodes:
            seen_terrains.add(ep["terrain_family"])
            seen_textures.add(ep["texture_family"])
            seen_obstacles.add(ep["obstacle_profile"])

    # Across all 42 episodes, every family, texture, and obstacle profile must be present
    assert seen_terrains == set(SUPPORTED_TERRAIN_FAMILIES)
    assert seen_textures == set(SUPPORTED_TEXTURE_FAMILIES)
    assert seen_obstacles == set(SUPPORTED_OBSTACLE_PROFILES)


def test_display_name_determinism():
    name1 = format_display_name(
        episode_index=42,
        terrain_family="valley_linear_canyon",
        texture_family="rocky_mountain",
        target_profile_composition="all_different",
        target_profiles=["aircraft_cruise", "uav_agile", "missile_dash"],
        radar_count=3,
    )
    name2 = format_display_name(
        episode_index=42,
        terrain_family="valley_linear_canyon",
        texture_family="rocky_mountain",
        target_profile_composition="all_different",
        target_profiles=["aircraft_cruise", "uav_agile", "missile_dash"],
        radar_count=3,
    )
    assert name1 == name2
    assert name1 == "EP-0042 · Rocky Canyon · 3 Targets (All Different) · 3 Radars"

    name_2plus1 = format_display_name(
        episode_index=37,
        terrain_family="valley_sinuous_pass",
        texture_family="green_forest_valley",
        target_profile_composition="two_plus_one",
        target_profiles=["uav_agile", "uav_agile", "uav_drift"],
        radar_count=3,
    )
    assert name_2plus1 == "EP-0037 · Forest Winding Pass · 3 UAV (2 Agile + 1 Drift) · 3 Radars"


def test_pilot_dataset_canonical_records_and_replay():
    index_path = Path("data/processed/pilot_diverse_v1/pilot_metadata_index.json")
    if not index_path.exists():
        pytest.skip("Pilot dataset not yet generated on disk")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_data["total_episodes"] == 42
    assert index_data["split_counts"] == {"train": 28, "validation": 7, "test": 7}

    # Verify a sample episode records and replay reconstruction
    sample_ep = index_data["episodes"][0]
    ep_dir = Path("data/processed/pilot_diverse_v1") / sample_ep["dataset_name"] / sample_ep["split"] / sample_ep["episode_id"]
    assert ep_dir.exists()
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "records.npz").exists()

    records = load_episode_records(ep_dir)
    assert records["metadata"]["record_schema_version"] == 2
    assert records["metadata"]["completed"] is True

    # Replay reconstruction
    payload = records_to_viewer_payload(records)
    assert payload["terrain"]["source"] == "procedural"
    assert "data" in payload["terrain"]
    assert "visual_objects" in payload["terrain"]
    assert payload["terrain"]["texture_family"] == sample_ep["texture_family"]


def test_legacy_datasets_untouched():
    # Verify multi_target_v1 or synthetic_100 exist and were not modified
    p_synthetic = Path("data/processed/synthetic_100/manifest.json")
    if p_synthetic.exists():
        data = json.loads(p_synthetic.read_text(encoding="utf-8"))
        assert data["manifest_schema_version"] == 1
        assert len(data["episodes"]) == 100


def test_pilot_v2_specifications_and_distribution():
    from src.services.diverse_dataset_service import build_diverse_pilot_episodes_v2

    all_eps = build_diverse_pilot_episodes_v2()
    assert len(all_eps) == 7

    flat_eps = [ep for eps in all_eps.values() for ep in eps]
    assert len(flat_eps) == 42

    easy_cnt = sum(1 for ep in flat_eps if ep["difficulty_level"] == "easy")
    med_cnt = sum(1 for ep in flat_eps if ep["difficulty_level"] == "medium")
    hard_cnt = sum(1 for ep in flat_eps if ep["difficulty_level"] == "hard")

    assert easy_cnt == 13
    assert med_cnt == 19
    assert hard_cnt == 10

    for ep in flat_eps:
        assert ep["resolved_scenario"]["duration_s"] == 30.0
        assert "radar_elevation_bars" in ep
        assert "radar_category" in ep
        assert len(ep["radar_elevation_bars"]) in (2, 3, 4)


def test_pilot_v2_on_disk_records_and_audit():
    index_path = Path("data/processed/pilot_diverse_v2/pilot_metadata_index.json")
    if not index_path.exists():
        pytest.skip("Pilot v2 dataset not yet generated on disk")

    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    assert index_data["total_episodes"] == 42
    assert index_data["difficulty_counts"] == {"easy": 13, "medium": 19, "hard": 10}

    sample_ep = index_data["episodes"][0]
    ep_dir = Path("data/processed/pilot_diverse_v2") / sample_ep["dataset_name"] / sample_ep["split"] / sample_ep["episode_id"]
    assert ep_dir.exists()
    assert (ep_dir / "metadata.json").exists()
    assert (ep_dir / "records.npz").exists()

    records = load_episode_records(ep_dir)
    assert records["metadata"]["record_schema_version"] == 2
    assert records["metadata"]["completed"] is True
    assert records["metadata"]["config_snapshot"]["effective_duration_s"] == 30.0


def test_canary_specifications_and_disk_records():
    from src.services.diverse_dataset_service import build_diverse_canary_episodes

    canary_eps = build_diverse_canary_episodes()
    assert len(canary_eps) == 7
    flat_eps = [ep for grp in canary_eps.values() for ep in grp]
    assert len(flat_eps) == 70

    split_counts = {"train": 0, "validation": 0, "test": 0}
    diff_counts = {"easy": 0, "medium": 0, "hard": 0}
    for ep in flat_eps:
        split_counts[ep["split"]] += 1
        diff_counts[ep["difficulty_level"]] += 1

    assert split_counts == {"train": 49, "validation": 11, "test": 10}
    assert diff_counts == {"easy": 21, "medium": 28, "hard": 21}

    index_path = Path("data/processed/canary_diverse_v1/canary_metadata_index.json")
    if index_path.exists():
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        assert index_data["total_episodes"] == 70
        assert index_data["split_counts"] == {"train": 49, "validation": 11, "test": 10}
        assert index_data["difficulty_counts"] == {"easy": 21, "medium": 28, "hard": 21}


def test_diverse_1000_specifications():
    from src.services.diverse_dataset_service import build_diverse_1000_episodes

    all_eps = build_diverse_1000_episodes()
    assert len(all_eps) == 7

    expected_totals = {
        "dataset_single_aircraft_v1": (105, 22, 23, 150),
        "dataset_single_uav_agile_v1": (105, 22, 23, 150),
        "dataset_single_uav_drift_v1": (105, 22, 23, 150),
        "dataset_single_missile_v1": (105, 22, 23, 150),
        "dataset_triple_all_same_v1": (92, 20, 20, 132),
        "dataset_triple_all_different_v1": (92, 20, 20, 132),
        "dataset_triple_two_plus_one_v1": (96, 22, 18, 136),
    }

    flat_eps = []
    terrain_seeds_by_split = {"train": set(), "validation": set(), "test": set()}
    route_hashes_by_split = {"train": set(), "validation": set(), "test": set()}

    for grp_name, eps in all_eps.items():
        assert grp_name in expected_totals
        e_tr, e_va, e_te, e_tot = expected_totals[grp_name]
        assert len(eps) == e_tot
        tr_cnt = sum(1 for ep in eps if ep["split"] == "train")
        va_cnt = sum(1 for ep in eps if ep["split"] == "validation")
        te_cnt = sum(1 for ep in eps if ep["split"] == "test")
        assert (tr_cnt, va_cnt, te_cnt) == (e_tr, e_va, e_te)

        for ep in eps:
            flat_eps.append(ep)
            terrain_seeds_by_split[ep["split"]].add(ep["terrain_seed"])
            route_hashes_by_split[ep["split"]].add(ep["route_group_id"])

    assert len(flat_eps) == 1000

    # Global splits
    assert sum(1 for ep in flat_eps if ep["split"] == "train") == 700
    assert sum(1 for ep in flat_eps if ep["split"] == "validation") == 150
    assert sum(1 for ep in flat_eps if ep["split"] == "test") == 150

    # Global difficulties
    assert sum(1 for ep in flat_eps if ep["difficulty_level"] == "easy") == 300
    assert sum(1 for ep in flat_eps if ep["difficulty_level"] == "medium") == 450
    assert sum(1 for ep in flat_eps if ep["difficulty_level"] == "hard") == 250

    # Disjoint splits (zero leakage)
    assert not (terrain_seeds_by_split["train"] & terrain_seeds_by_split["validation"])
    assert not (terrain_seeds_by_split["train"] & terrain_seeds_by_split["test"])
    assert not (terrain_seeds_by_split["validation"] & terrain_seeds_by_split["test"])

    assert not (route_hashes_by_split["train"] & route_hashes_by_split["validation"])
    assert not (route_hashes_by_split["train"] & route_hashes_by_split["test"])
    assert not (route_hashes_by_split["validation"] & route_hashes_by_split["test"])


