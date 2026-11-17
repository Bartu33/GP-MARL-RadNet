"""Tests for Dataset Replay Episode Browser, Filtering, Preview & Replay Payload Conversion."""
from __future__ import annotations

from pathlib import Path
import pytest

from src.interfaces.dataset_browser import (
    load_dataset_catalog,
    load_dataset_episodes,
    filter_episodes,
    DATASET_OPTIONS,
)
from src.services import simulation_service as sim

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_discovery():
    """Verify that catalog discovers all 7 pilot groups, all-pilot group, and legacy datasets."""
    catalog = load_dataset_catalog(ROOT)
    
    expected_pilot_groups = [
        "dataset_single_aircraft_v1",
        "dataset_single_uav_agile_v1",
        "dataset_single_uav_drift_v1",
        "dataset_single_missile_v1",
        "dataset_triple_all_same_v1",
        "dataset_triple_all_different_v1",
        "dataset_triple_two_plus_one_v1",
    ]
    for group in expected_pilot_groups:
        assert group in catalog, f"Expected {group} in catalog"

    assert "All Pilot Groups (pilot_diverse_v1 - 42 episodes)" in catalog
    assert "multi_target_v1 (legacy)" in catalog
    assert "synthetic_100 (legacy)" in catalog


def test_load_pilot_all_episodes():
    """Verify lightweight loading of all 42 pilot episodes from metadata index."""
    eps = load_dataset_episodes("All Pilot Groups (pilot_diverse_v1 - 42 episodes)", ROOT)
    assert len(eps) == 42, f"Expected 42 episodes, got {len(eps)}"

    for ep in eps:
        assert "display_name" in ep and len(ep["display_name"]) > 5
        assert "episode_id" in ep and ep["episode_id"].startswith("ep-")
        assert ep["split"] in ("train", "validation", "test")
        assert ep["terrain_family"] in (
            "valley_linear_canyon",
            "valley_sinuous_pass",
            "mountain_chokepoint",
            "basin_highlands",
            "ridge_multi_obstacle",
        )
        assert ep["texture_family"] in (
            "rocky_mountain",
            "green_forest_valley",
            "dry_canyon",
            "snowy_highlands",
            "grassland_valley",
            "mixed_rock_vegetation",
        )
        assert ep["obstacle_profile"] in ("clear", "sparse", "dense", "asymmetric")
        assert ep["target_count"] in (1, 3)
        assert ep["radar_count"] == 3
        assert Path(ep["full_path"]).is_dir()
        assert (Path(ep["full_path"]) / "records.npz").exists()
        assert (Path(ep["full_path"]) / "metadata.json").exists()


def test_load_single_group_episodes():
    """Verify loading a single pilot group from its manifest without opening NPZs."""
    eps = load_dataset_episodes("dataset_single_aircraft_v1", ROOT)
    assert len(eps) == 6
    for ep in eps:
        assert ep["target_count"] == 1
        assert "aircraft_like" in ep["target_types"]
        assert ep["physical_target_type_composition"] == "single"
        assert ep["target_profile_composition"] == "single"


def test_legacy_dataset_fallback():
    """Verify safe fallback to 'Unknown' for legacy datasets without crashing or data fabrication."""
    eps = load_dataset_episodes("multi_target_v1 (legacy)", ROOT)
    assert len(eps) == 100
    for ep in eps:
        assert ep["terrain_family"] == "Unknown"
        assert ep["texture_family"] == "Unknown"
        assert ep["obstacle_profile"] == "Unknown"
        assert ep["radar_count"] == "Unknown"
        assert ep["target_count"] == "Unknown"
        assert "(Legacy)" in ep["display_name"]
        assert Path(ep["full_path"]).is_dir()


def test_multi_criteria_reactive_filtering():
    """Verify multi-criteria reactive filtering across all metadata dimensions."""
    all_eps = load_dataset_episodes("All Pilot Groups (pilot_diverse_v1 - 42 episodes)", ROOT)

    # 1. Split filtering
    train_eps = filter_episodes(all_eps, split="train")
    val_eps = filter_episodes(all_eps, split="validation")
    test_eps = filter_episodes(all_eps, split="test")
    assert len(train_eps) == 28
    assert len(val_eps) == 7
    assert len(test_eps) == 7
    assert len(train_eps) + len(val_eps) + len(test_eps) == 42

    # 2. Target Count filtering
    single_eps = filter_episodes(all_eps, target_count="1")
    triple_eps = filter_episodes(all_eps, target_count="3")
    assert len(single_eps) == 24  # 4 groups * 6 eps
    assert len(triple_eps) == 18  # 3 groups * 6 eps
    assert len(single_eps) + len(triple_eps) == 42

    # 3. Target Profile Composition filtering
    all_diff_eps = filter_episodes(all_eps, target_profile_composition="all_different")
    all_same_eps = filter_episodes(all_eps, target_profile_composition="all_same")
    two_plus_one_eps = filter_episodes(all_eps, target_profile_composition="two_plus_one")
    assert len(all_diff_eps) == 6
    assert len(all_same_eps) == 6
    assert len(two_plus_one_eps) == 6

    # 4. Target Type filtering
    missile_eps = filter_episodes(all_eps, target_type="missile_like")
    assert len(missile_eps) >= 6
    for ep in missile_eps:
        assert "missile_like" in ep["target_types"]

    # 5. Free-text search
    canyon_eps = filter_episodes(all_eps, search_text="Canyon")
    assert len(canyon_eps) > 0
    for ep in canyon_eps:
        assert "canyon" in ep["display_name"].lower() or "canyon" in ep["episode_id"].lower()

    # 6. Specific ID search
    target_id = all_eps[0]["episode_id"]
    id_eps = filter_episodes(all_eps, search_text=target_id)
    assert len(id_eps) == 1
    assert id_eps[0]["episode_id"] == target_id

    # 7. Zero-match condition
    zero_eps = filter_episodes(all_eps, search_text="NonExistentPatternXYZ12345")
    assert len(zero_eps) == 0


def test_pilot_replay_records_and_viewer_payload():
    """Verify Canonical Records v2 loading and conversion to 3-D viewer payload for pilot episode."""
    eps = load_dataset_episodes("dataset_single_uav_agile_v1", ROOT)
    ep = eps[0]
    
    recs = sim.load_episode_records(ep["full_path"])
    assert "reports" in recs
    assert "tracks" in recs
    assert "truth" in recs
    assert "metadata" in recs
    assert recs["metadata"]["record_schema_version"] == 2

    payload = sim.records_to_viewer_payload(recs)
    assert "terrain" in payload
    assert "frames" in payload
    assert len(payload["frames"]) > 0
    assert payload["terrain"]["terrain_family"] == ep["terrain_family"]
    assert payload["terrain"]["texture_family"] == ep["texture_family"]
    assert payload["terrain"]["obstacle_profile"] == ep["obstacle_profile"]
    assert "visual_objects" in payload["terrain"]


def test_legacy_replay_records_and_viewer_payload():
    """Verify backwards-compatible loading and payload conversion for legacy multi_target_v1."""
    eps = load_dataset_episodes("multi_target_v1 (legacy)", ROOT)
    ep = eps[0]
    
    recs = sim.load_episode_records(ep["full_path"])
    assert recs["metadata"]["record_schema_version"] == 2
    
    payload = sim.records_to_viewer_payload(recs)
    assert "terrain" in payload
    assert "frames" in payload
    assert len(payload["frames"]) > 0


def test_pilot_v2_catalog_and_difficulty_filtering():
    """Verify pilot_diverse_v2 catalog loading, 42 episodes, and difficulty filtering."""
    eps = load_dataset_episodes("All Pilot Groups (pilot_diverse_v2 - 42 episodes)", ROOT)
    assert len(eps) == 42

    easy_eps = filter_episodes(eps, difficulty="easy")
    med_eps = filter_episodes(eps, difficulty="medium")
    hard_eps = filter_episodes(eps, difficulty="hard")

    assert len(easy_eps) == 13
    assert len(med_eps) == 19
    assert len(hard_eps) == 10

    for ep in eps:
        assert ep["difficulty_level"] in ("easy", "medium", "hard")
        assert Path(ep["full_path"]).is_dir()
        assert (Path(ep["full_path"]) / "records.npz").exists()
        assert (Path(ep["full_path"]) / "metadata.json").exists()


def test_streamlit_app_dataset_replay_flow():
    """Verify end-to-end UI interaction in Streamlit AppTest for Dataset Replay."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=30).run()
    assert not at.exception

    # Switch to Dataset Replay
    at.radio(key="scen_op_mode").set_value("Dataset Replay").run()
    assert not at.exception

    # Verify dataset selector
    ds_sel = at.selectbox(key="rep_dataset_sel")
    assert len(ds_sel.options) >= 10
    assert "All Pilot Groups (pilot_diverse_v2 - 42 episodes)" in ds_sel.options
    assert "dataset_single_aircraft_v1" in ds_sel.options
    assert "multi_target_v1 (legacy)" in ds_sel.options

    # Verify episode selector has episodes and display_name
    ep_sel = at.selectbox(key=f"rep_ep_sel_{ds_sel.value}")
    assert len(ep_sel.options) in (1000, 70, 42, 6)

    # Load episode into replay
    at.button(key="rep_load_btn").click().run()
    assert not at.exception
    assert at.session_state["scen_mode_active"] == "replay"
    assert "EP-0001" in at.session_state["scen_ep_name"]
    assert len(at.session_state["scen_records"]["reports"]) > 0


def test_canary_catalog_and_difficulty_filtering():
    """Verify canary_diverse_v1 catalog loading, 70 episodes, and difficulty filtering."""
    eps = load_dataset_episodes("All Canary Groups (canary_diverse_v1 - 70 episodes)", ROOT)
    assert len(eps) == 70

    easy_eps = filter_episodes(eps, difficulty="easy")
    med_eps = filter_episodes(eps, difficulty="medium")
    hard_eps = filter_episodes(eps, difficulty="hard")

    assert len(easy_eps) == 21
    assert len(med_eps) == 28
    assert len(hard_eps) == 21

    for ep in eps:
        assert ep["difficulty_level"] in ("easy", "medium", "hard")
        assert Path(ep["full_path"]).is_dir()
        assert (Path(ep["full_path"]) / "records.npz").exists()
        assert (Path(ep["full_path"]) / "metadata.json").exists()


def test_diverse_1000_catalog_and_difficulty_filtering():
    """Verify diverse_1000_v1 catalog loading, 1000 episodes, and difficulty filtering."""
    eps = load_dataset_episodes("All Final Groups (diverse_1000_v1 - 1000 episodes)", ROOT)
    assert len(eps) == 1000

    easy_eps = filter_episodes(eps, difficulty="easy")
    med_eps = filter_episodes(eps, difficulty="medium")
    hard_eps = filter_episodes(eps, difficulty="hard")

    assert len(easy_eps) == 300
    assert len(med_eps) == 450
    assert len(hard_eps) == 250

    sample_ep = eps[0]
    assert sample_ep["difficulty_level"] in ("easy", "medium", "hard")
    assert Path(sample_ep["full_path"]).is_dir()
    assert (Path(sample_ep["full_path"]) / "records.npz").exists()
    assert (Path(sample_ep["full_path"]) / "metadata.json").exists()


def test_all_episodes_option_and_group_counts():
    """Verify All Episodes option and exact counts for each of the 7 final groups."""
    all_alias_eps = load_dataset_episodes("All Episodes (diverse_1000_v1 - 1000 episodes)", ROOT)
    assert len(all_alias_eps) == 1000

    group_counts = {
        "dataset_single_aircraft_v1 (Final 1000)": 150,
        "dataset_single_uav_agile_v1 (Final 1000)": 150,
        "dataset_single_uav_drift_v1 (Final 1000)": 150,
        "dataset_single_missile_v1 (Final 1000)": 150,
        "dataset_triple_all_same_v1 (Final 1000)": 132,
        "dataset_triple_all_different_v1 (Final 1000)": 132,
        "dataset_triple_two_plus_one_v1 (Final 1000)": 136,
    }
    for grp, exp_c in group_counts.items():
        grp_eps = load_dataset_episodes(grp, ROOT)
        assert len(grp_eps) == exp_c, f"{grp} expected {exp_c}, got {len(grp_eps)}"


def test_first_middle_last_episode_replay_records():
    """Verify Canonical Records v2 loading for first (EP-0001), middle (EP-0500), and last (EP-1000)."""
    eps = load_dataset_episodes("All Final Groups (diverse_1000_v1 - 1000 episodes)", ROOT)
    assert len(eps) == 1000

    test_indices = [0, 499, 999]
    for idx in test_indices:
        ep = eps[idx]
        recs = sim.load_episode_records(ep["full_path"])
        assert "reports" in recs
        assert "tracks" in recs
        assert "truth" in recs
        payload = sim.records_to_viewer_payload(recs)
        assert len(payload["frames"]) == 61
        assert "targets" in payload["frames"][0]
        assert "radars" in payload["frames"][0]




