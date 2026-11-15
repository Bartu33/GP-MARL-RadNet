"""Tests for Radar Selection & Handover Analysis Service (Stage 3B)."""
from pathlib import Path
import pytest

from src.services.radar_selection_service import (
    analyze_radar_actions,
    analyze_radar_resources,
    analyze_track_radar_transitions,
    explain_radar_selection_decision,
    load_records,
    verify_stage3a_metrics,
)
from src.services.simulation_service import load_episode_records

DATASET_ROOT = Path("data/processed/multi_target_v1")
SAMPLE_EPISODE = next((DATASET_ROOT / "train").glob("ep-*"), None)


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_analyze_radar_actions():
    records = load_episode_records(SAMPLE_EPISODE)
    res = analyze_radar_actions(records)

    assert res["total_steps"] == 60
    assert res["n_radars"] == 3
    for r_name in ["R1", "R2", "R3"]:
        assert r_name in res["radars"]
        r_info = res["radars"][r_name]
        total_acts = r_info["idle_steps"] + r_info["search_steps"] + r_info["track_steps"]
        assert total_acts == 60


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_analyze_radar_resources():
    records = load_episode_records(SAMPLE_EPISODE)
    res = analyze_radar_resources(records)

    assert "R1" in res
    assert "R2" in res
    assert "R3" in res
    for r_name, r_info in res.items():
        assert r_info["estimated_radiated_energy_j"] >= 0.0
        assert r_info["max_channels"] == 4


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_analyze_track_radar_transitions_and_handover_status():
    records = load_episode_records(SAMPLE_EPISODE)
    res = analyze_track_radar_transitions(records)

    # Must explicitly document absence of autonomous distributed handover protocol
    assert res["explicit_handover_protocol_supported"] is False
    assert "NO explicit handover" in res["protocol_status_note"]
    assert isinstance(res["transitions"], list)


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_explain_radar_selection_decision():
    records = load_episode_records(SAMPLE_EPISODE)
    
    # Test step 0
    exp0 = explain_radar_selection_decision(records, step_index=0, radar_id=0)
    assert "mode" in exp0
    assert "explanation" in exp0
    assert "justification" in exp0

    # Test step 10
    exp10 = explain_radar_selection_decision(records, step_index=10, radar_id=1)
    assert exp10["step_index"] == 10
    assert exp10["radar_id"] == 1


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="multi_target_v1 dataset not found")
def test_verify_stage3a_metrics_reconciliation():
    rec = verify_stage3a_metrics(DATASET_ROOT)

    pkt = rec["packet_loss_reconciliation"]
    assert pkt["proof_valid"] is True
    assert pkt["dataset_total_packet_lost"] == 533
    assert pkt["false_alarms_lost"] == 117
    assert pkt["target_lost_inside_nominal_fov_los"] == 390
    assert pkt["target_lost_at_edge_of_fov"] == 26

    cov = rec["coasting_covariance_reconciliation"]
    assert "actual_ekf_covariance_pos_std_m" in cov
