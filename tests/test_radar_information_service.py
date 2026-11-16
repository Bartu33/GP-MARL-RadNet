"""Tests for the Radar Information Analysis Service (Stage 3A)."""
from pathlib import Path
import numpy as np
import pytest

from src.services.radar_information_service import (
    analyze_dataset,
    analyze_episode,
    analyze_radar_target_visibility,
    get_detection_timeline,
    get_track_uncertainty_timeline,
    load_records_or_dir,
    summarize_communication,
    summarize_radar_measurements,
    summarize_tracker_estimates,
)
from src.services.simulation_service import load_episode_records

DATASET_ROOT = Path("data/processed/multi_target_v1")
SAMPLE_EPISODE = next((DATASET_ROOT / "train").glob("ep-*"), None)


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_summarize_radar_measurements_operational_contract():
    records = load_episode_records(SAMPLE_EPISODE)
    summary = summarize_radar_measurements(records)

    assert summary["total_reports"] > 0
    assert summary["snr_db"] is not None
    assert summary["range_m"]["min"] >= 0.0
    assert summary["azimuth_deg"]["min"] >= -180.0
    assert summary["azimuth_deg"]["max"] <= 180.0
    assert summary["radial_velocity_mps"] is not None
    assert summary["measurement_uncertainty"]["sigma_range_m"]["mean"] > 0.0

    # Ensure NO ground truth leakage in operational measurement summary
    assert "truth_id" not in summary
    assert "target_type" not in summary
    assert "class_label" not in summary


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_summarize_communication_schema_v2():
    records = load_episode_records(SAMPLE_EPISODE)
    comm = summarize_communication(records)

    assert comm["total_generated"] == len(records["reports"])
    assert comm["delivered"] + comm["packet_lost"] == comm["total_generated"]
    assert 0.0 <= comm["delivery_rate_pct"] <= 100.0
    assert "delivered" in comm["status_counts"]


def test_summarize_communication_schema_v1_fallback():
    # Construct synthetic v1 records (no communication table)
    v1_records = {
        "metadata": {"record_schema_version": 1},
        "reports": [{"radar_id": 0, "t_s": 0.5}],
    }
    comm = summarize_communication(v1_records)
    assert comm["status"] == "unavailable_in_schema_v1"
    assert comm["total_generated"] == 1
    assert comm["delivered"] is None


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_radar_and_time_filters():
    records = load_episode_records(SAMPLE_EPISODE)

    # Filter by radar
    r0_summary = summarize_radar_measurements(records, radar_id=0)
    r1_summary = summarize_radar_measurements(records, radar_id=1)
    all_summary = summarize_radar_measurements(records)
    assert r0_summary["total_reports"] + r1_summary["total_reports"] <= all_summary["total_reports"]

    # Filter by time
    t_summary = summarize_radar_measurements(records, t_range=(5.0, 15.0))
    assert t_summary["total_reports"] <= all_summary["total_reports"]


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_summarize_tracker_estimates():
    records = load_episode_records(SAMPLE_EPISODE)
    tracker = summarize_tracker_estimates(records, source="fused")

    assert tracker["total_track_steps"] > 0
    assert tracker["unique_track_ids"] > 0
    assert tracker["position_uncertainty_m"]["mean"] > 0.0
    assert tracker["velocity_uncertainty_mps"]["mean"] > 0.0
    assert tracker["coasting_duration_s"]["min"] >= 0.0
    assert 0.0 <= tracker["confirmed_steps_pct"] <= 100.0


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_analyze_radar_target_visibility():
    records = load_episode_records(SAMPLE_EPISODE)
    vis = analyze_radar_target_visibility(records)

    breakdown = vis["overall_sensing_breakdown"]
    total = breakdown["total_looks_evaluated"]
    assert total > 0

    sum_parts = (
        breakdown["out_of_range"]
        + breakdown["out_of_fov"]
        + breakdown["terrain_blocked"]
        + breakdown["los_clear_unilluminated_or_missed"]
        + breakdown["measurement_formed_but_lost"]
        + breakdown["measurement_delivered"]
    )
    assert sum_parts == total
    assert len(vis["per_target_breakdown"]) == 4  # 4 targets in multi-target


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_timeline_extraction():
    records = load_episode_records(SAMPLE_EPISODE)
    det_tl = get_detection_timeline(records, bin_width_s=1.0)
    assert len(det_tl["bin_centers_s"]) > 0
    assert sum(det_tl["generated_counts"]) == len(records["reports"])

    track_tl = get_track_uncertainty_timeline(records, source="fused")
    assert len(track_tl["time_s"]) > 0
    assert len(track_tl["mean_position_uncertainty_m"]) == len(track_tl["time_s"])


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_analyze_dataset_subset():
    res = analyze_dataset(DATASET_ROOT, split="train", max_episodes=2)
    assert res["episodes_analyzed"] == 2
    assert res["radar_measurements"]["total_reports"] > 0
    assert res["communication"]["overall_delivery_rate_pct"] > 80.0
