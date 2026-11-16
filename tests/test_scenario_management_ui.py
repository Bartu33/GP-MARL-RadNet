"""Tests for Scenario Management UI components, adaptor, plots, and validation (Stages 4B, 4C, 4D)."""
import json
from pathlib import Path
import pytest
import plotly.graph_objects as go

from src.core.scenario import load_scenario, validate_scenario
from src.interfaces.plots import (
    plot_action_distribution,
    plot_detection_timeline,
    plot_uncertainty_vs_aoi,
)
from src.services.radar_information_service import (
    summarize_radar_measurements,
    summarize_communication,
    get_detection_timeline,
    get_track_uncertainty_timeline,
    summarize_tracker_estimates,
)
from src.services.radar_selection_service import (
    analyze_radar_actions,
    analyze_radar_resources,
    analyze_track_radar_transitions,
    explain_radar_selection_decision,
)
from src.services.simulation_service import (
    load_episode_records,
    records_to_viewer_payload,
    run_episode,
)

DATASET_ROOT = Path("data/processed/multi_target_v1")
SAMPLE_EPISODE = next((DATASET_ROOT / "train").glob("ep-*"), None)


def test_scenario_mutation_and_validation():
    """Verify that user-configured scenario adjustments validate properly."""
    base = load_scenario("nominal")
    cfg = dict(base)
    cfg["duration_s"] = 15.0
    cfg["dt"] = 0.5
    cfg["seed"] = 42

    # Validate valid config
    validated = validate_scenario(cfg)
    assert validated["duration_s"] == 15.0
    assert validated["dt"] == 0.5
    assert validated["seed"] == 42

    # Reject invalid duration
    invalid_cfg = dict(cfg)
    invalid_cfg["duration_s"] = -5.0
    with pytest.raises(ValueError):
        validate_scenario(invalid_cfg)


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_records_to_viewer_payload_from_dataset():
    """Verify conversion of Canonical Episode Records v2 to 3D/2D viewer payload."""
    records = load_episode_records(SAMPLE_EPISODE)
    payload = records_to_viewer_payload(records)

    assert "site" in payload
    assert "terrain" in payload
    assert "radars" in payload
    assert len(payload["radars"]) == 3
    assert "frames" in payload
    assert len(payload["frames"]) in (60, 61)

    # Check frame structure
    f0 = payload["frames"][0]
    assert "t" in f0
    assert "radars" in f0
    assert "targets" in f0
    assert "dets" in f0
    assert "tracks" in f0

    # Ensure no beam hallucination during replay
    for r in f0["radars"]:
        assert r["beams"] == []

    # Check replay metadata notice
    assert "replay_metadata" in payload
    assert "unrecorded_notice" in payload["replay_metadata"]


def test_records_to_viewer_payload_from_live_simulation():
    """Verify live simulation output runs through the viewer adaptor."""
    sc = load_scenario("nominal")
    sc_copy = dict(sc)
    sc_copy["duration_s"] = 2.0
    sc_copy["dt"] = 0.5

    env, summ = run_episode(sc_copy, "central_greedy", seed=10, record_episode=True)
    assert env.episode_records is not None
    records = env.episode_records

    payload = records_to_viewer_payload(records)
    assert len(payload["frames"]) in (4, 5)  # 2.0s / 0.5s = 4 steps + t=0
    assert len(payload["radars"]) == len(records["metadata"]["agent_ids"])


def test_scenario_management_plot_action_distribution():
    sample_summary = {
        "R1": {"search_pct": 50.0, "track_pct": 40.0, "idle_pct": 10.0},
        "R2": {"search_pct": 60.0, "track_pct": 30.0, "idle_pct": 10.0},
        "R3": {"search_pct": 70.0, "track_pct": 20.0, "idle_pct": 10.0},
    }
    fig = plot_action_distribution(sample_summary)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Search" in trace_names
    assert "Track" in trace_names
    assert "Idle" in trace_names


def test_scenario_management_plot_detection_timeline():
    timeline_data = {
        "bin_centers_s": [1.0, 2.0, 3.0],
        "generated_counts": [10, 15, 8],
        "delivered_counts": [9, 14, 7],
    }
    fig = plot_detection_timeline(timeline_data)
    assert isinstance(fig, go.Figure)
    trace_names = [t.name for t in fig.data]
    assert "Delivered to Fusion" in trace_names
    assert "Packet Lost" in trace_names


def test_scenario_management_plot_uncertainty_vs_aoi():
    aoi_data = {
        0.5: 25.0,
        1.0: 45.0,
        2.0: 95.0,
        3.0: 160.0,
    }
    fig = plot_uncertainty_vs_aoi(aoi_data)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_radar_information_service_integration():
    """Verify radar information service functions work with loaded records."""
    records = load_episode_records(SAMPLE_EPISODE)
    meas = summarize_radar_measurements(records)
    assert "total_reports" in meas
    assert meas["total_reports"] >= 0

    meas_r0 = summarize_radar_measurements(records, radar_id=0)
    assert meas_r0["radar_id"] == 0

    comm = summarize_communication(records)
    assert "delivery_rate_pct" in comm
    assert "total_generated" in comm
    assert "delivered" in comm

    timeline = get_detection_timeline(records)
    assert "bin_centers_s" in timeline
    assert "generated_counts" in timeline


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_decision_inspector_rule_vs_inferred_badges():
    """Verify decision inspector returns clear rule vs inference attribution."""
    records = load_episode_records(SAMPLE_EPISODE)
    exp = explain_radar_selection_decision(records, step_index=5, radar_id=0)

    assert "step_index" in exp
    assert "radar_id" in exp
    assert "action" in exp
    assert "mode" in exp
    assert "justification" in exp
    assert "explanation" in exp


def test_existing_tabs_preserved():
    """Verify that all 6 tabs exist in the application code."""
    app_path = Path("src/interfaces/app.py")
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")

    # Check tab variable declarations
    assert "tab_scen" in content
    assert "tab_sim" in content
    assert "tab_lab" in content
    assert "tab_cls" in content
    assert "tab_exp" in content
    assert "tab_about" in content

    # Check tab header strings
    assert "Scenario Management" in content
    assert "3-D Simulation" in content
    assert "Radar Model Lab" in content
    assert "Classifier" in content
    assert "Experiments" in content
    assert "About" in content


@pytest.mark.skipif(not DATASET_ROOT.exists(), reason="multi_target_v1 dataset not found")
def test_multi_target_dataset_split_allocation_and_provenance():
    """Verify manifest split ratios, prefix-stable allocation, and load multi-split episodes."""
    manifest_path = DATASET_ROOT / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    train_eps = [e for e in manifest["episodes"] if e["split"] == "train"]
    val_eps = [e for e in manifest["episodes"] if e["split"] == "validation"]
    test_eps = [e for e in manifest["episodes"] if e["split"] == "test"]

    assert len(train_eps) == 70
    assert len(val_eps) == 15
    assert len(test_eps) == 15

    # Check disk existence and load from each split
    ep_samples = [
        ("train", "ep-000000-d3e687bcdcd2"),
        ("train", "ep-000070-f262fb427205"),
        ("validation", "ep-000002-04bb138892cf"),
        ("validation", "ep-000075-58056a9042f8"),
        ("test", "ep-000004-9eecce8e758c"),
        ("test", "ep-000084-e82774dfab37"),
    ]

    for split, ep_name in ep_samples:
        ep_dir = DATASET_ROOT / split / ep_name
        assert ep_dir.is_dir(), f"Expected episode dir {ep_dir} not found"
        recs = load_episode_records(ep_dir)

        # Check provenance
        meta = recs["metadata"]
        assert meta["dataset_provenance"]["split"] == split

        # Check records integrity
        assert len(recs["states"]) == 61
        assert len(recs["truth"]) == 244
        assert len(recs["transitions"]) == 60

        # Check 3D/2D adaptor
        payload = records_to_viewer_payload(recs)
        assert len(payload["frames"]) == 61
        for fr in payload["frames"]:
            for r in fr["radars"]:
                assert r["beams"] == []  # No beam hallucination in replay


def test_simulation_step_vs_state_vs_truth_counts():
    """Verify exact mathematical relations between duration, dt, transitions, states, and truth."""
    # 15.0 s with dt = 0.5 s, 4 targets
    sc = load_scenario("nominal")
    sc15 = dict(sc)
    sc15["duration_s"] = 15.0
    sc15["dt"] = 0.5
    sc15["targets"] = [{"label": 0}, {"label": 1}, {"label": 2}, {"label": 3}]

    env, _ = run_episode(sc15, "central_greedy", seed=42, record_episode=True)
    recs = env.episode_records

    # Transitions (actions taken) = duration / dt = 15.0 / 0.5 = 30
    assert len(recs["transitions"]) == 30
    # States (time points from t=0.0 to t=15.0) = 30 + 1 = 31
    assert len(recs["states"]) == 31
    # Truth rows = 31 states * 4 targets = 124
    assert len(recs["truth"]) == 124


def test_streamlit_apptest_live_and_replay_execution():
    """Programmatic end-to-end verification of Streamlit app without exceptions."""
    from streamlit.testing.v1 import AppTest
    app_file = Path(__file__).resolve().parent.parent / "src" / "interfaces" / "app.py"
    at = AppTest.from_file(app_file)
    at.run(timeout=30)
    assert len(at.exception) == 0

    # 1. Test live simulation configuration and execution
    at.slider(key="scen_dur_slider").set_value(10.0).run(timeout=30)
    assert len(at.exception) == 0
    at.button(key="scen_btn_val").click().run(timeout=30)
    assert len(at.exception) == 0
    at.button(key="scen_btn_run").click().run(timeout=60)
    assert len(at.exception) == 0
    assert at.session_state.get("scen_records") is not None
    assert at.session_state.get("scen_mode_active") == "live"

    # 2. Test switching to replay mode and loading episode
    at.radio(key="scen_op_mode").set_value("Dataset Replay (multi_target_v1)").run(timeout=30)
    assert len(at.exception) == 0
    at.button(key="rep_load_btn").click().run(timeout=30)
    assert len(at.exception) == 0
    assert at.session_state.get("scen_mode_active") == "replay"


def test_format_target_label_explicit_and_fallback():
    """Verify target labeling with explicit archetypes and legacy fallbacks."""
    from src.services.simulation_service import format_target_label

    # Explicit archetypes
    assert "Aircraft-like" in format_target_label(0, 0, "aircraft_like")
    assert "UAV-like (Agile)" in format_target_label(1, 1, "uav_like")
    assert "UAV-like (Drift)" in format_target_label(2, 3, "uav_like")
    assert "Missile-like" in format_target_label(3, 2, "missile_like")
    assert "[Ground Truth]" in format_target_label(0, 0, "aircraft_like", ground_truth_tag=True)
    assert "[Ground Truth]" not in format_target_label(0, 0, "aircraft_like", ground_truth_tag=False)

    # Legacy fallbacks
    assert format_target_label(0, 0, None) == "truth A · fast"
    assert format_target_label(1, 1, None) == "truth B · agile"
    assert format_target_label(2, 2, None) == "truth C · erratic"
    assert format_target_label(3, 3, None) == "truth D · drift"


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_target_labels_in_records_and_viewer_payload():
    """Verify 3D and 2D payloads contain consistent target labels and types."""
    records = load_episode_records(SAMPLE_EPISODE)
    payload = records_to_viewer_payload(records)

    assert "target_labels" in payload
    labels = payload["target_labels"]
    assert 0 in labels and "Aircraft-like" in labels[0]
    assert 1 in labels and "UAV-like (Agile)" in labels[1]
    assert 2 in labels and "UAV-like (Drift)" in labels[2]
    assert 3 in labels and "Missile-like" in labels[3]

    # Verify every frame target has display_name and target_type
    for fr in payload["frames"]:
        for t in fr["targets"]:
            assert "display_name" in t
            assert "target_type" in t
            assert t["display_name"] == labels[t["id"]]


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_map_2d_auto_zoom_and_equal_aspect_ratio_and_slider_stability():
    """Verify 2D map auto-zoom encloses active area, keeps equal aspect ratio and does not jump."""
    from src.interfaces.plots import map_2d

    records = load_episode_records(SAMPLE_EPISODE)
    payload = records_to_viewer_payload(records)

    # Generate figure for frame 0, frame 30, and frame 60
    fig0 = map_2d(payload, 0)
    fig30 = map_2d(payload, 30)
    fig60 = map_2d(payload, 60)

    # 1. Slider stability: range MUST be identical across frames
    assert fig0.layout.xaxis.range == fig30.layout.xaxis.range == fig60.layout.xaxis.range
    assert fig0.layout.yaxis.range == fig30.layout.yaxis.range == fig60.layout.yaxis.range

    # 2. Equal physical scale: yaxis scaleanchor must be 'x' and scaleratio 1
    assert fig0.layout.yaxis.scaleanchor == "x"
    assert fig0.layout.yaxis.scaleratio == 1

    # Equal axis span
    x_range = fig0.layout.xaxis.range
    y_range = fig0.layout.yaxis.range
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]
    assert pytest.approx(x_span, rel=1e-3) == y_span

    # 3. Enclose all radars and targets
    for r in payload["radars"]:
        assert x_range[0] <= r["pos"][0] <= x_range[1]
        assert y_range[0] <= r["pos"][1] <= y_range[1]

    for fr in payload["frames"]:
        for t in fr["targets"]:
            assert x_range[0] <= t["p"][0] <= x_range[1]
            assert y_range[0] <= t["p"][1] <= y_range[1]


@pytest.mark.skipif(SAMPLE_EPISODE is None, reason="multi_target_v1 dataset not found")
def test_fused_tracks_metric_confirmed_vs_cumulative():
    """Verify distinct confirmed vs cumulative track ID counts from records."""
    records = load_episode_records(SAMPLE_EPISODE)
    fused = [tr for tr in records["tracks"] if tr.get("source") == "fused"]
    cumul_ids = set(tr["track_id"] for tr in fused)
    conf_ids = set(tr["track_id"] for tr in fused if tr.get("confirmed") or tr.get("status") == "confirmed")

    # Confirmed tracks matches the 4 physical targets
    assert len(conf_ids) == 4
    # Cumulative track IDs is larger due to tentative clutter / initialization
    assert len(cumul_ids) >= len(conf_ids)
    assert len(cumul_ids) in (17, 18)
