"""Full-precision episode recording is passive and independent of UI replay."""
from copy import deepcopy
import json

import numpy as np
import pytest

import src.core.env as env_module
from src.core.env import RadarNetEnv
from src.models.tracking.tracker import TrackManager
from src.core.radar import _make_report
from src.services.simulation_service import export_episode_records, load_episode_records, run_episode


def scenario():
    return {
        "dt": 0.123456, "duration_s": 0.493824, "terrain": {"source": "flat"},
        "radar_defaults": {"heading_deg": 0, "search_sector_deg": 120,
                           "beamwidth_az_deg": 120, "beamwidth_el_deg": 60,
                           "search_bars_deg": [4], "false_alarms_per_beam": 0},
        "radars": [{"position": [0, 0, 15]}, {"position": [1000, 0, 15]}],
        "targets": [{"label": 3, "position": [4000.123456789, 0, 300.987654321]}],
    }


def make_env(**kwargs):
    env = RadarNetEnv(scenario(), seed=7, **kwargs)
    env.reset()
    return env


def assert_equal(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_equal(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        np.testing.assert_equal(a, b)


def run(env):
    while env.agents:
        env.step({a: 1 for a in env.agents})


def test_t_transitions_have_t_plus_one_states_and_correct_times(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = make_env(record_episode=True)
    assert list(tmp_path.iterdir()) == []
    records = env.episode_records
    assert records["states"] == [{"step_index": 0, "t_s": 0.0}]
    assert not records["metadata"]["completed"]
    run(env)
    assert len(records["states"]) == env.max_steps + 1
    assert len(records["transitions"]) == env.max_steps
    assert len(records["truth"]) == env.max_steps + 1
    assert records["metadata"]["completed"]
    assert records["metadata"]["recorded_steps"] == env.max_steps
    assert records["metadata"]["config_snapshot"] == env.config_snapshot()
    assert records["metadata"]["episode_id"]
    for i, transition in enumerate(records["transitions"], start=1):
        assert transition["step_index"] == i
        assert transition["start_t_s"] == records["states"][i - 1]["t_s"]
        assert transition["end_t_s"] == records["states"][i]["t_s"]
        np.testing.assert_array_equal(transition["requested_actions"], [1, 1])
        np.testing.assert_array_equal(transition["rewards"], [env.log["reward"][i - 1]] * 2)
        assert not transition["terminations"].any()
        assert transition["truncations"].all() == (i == env.max_steps)
    assert records["reports"]
    for report in records["reports"]:
        assert report["t_s"] == records["states"][report["step_index"]]["t_s"]
    assert env.frames == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("local_tracks", [False, True])
def test_recording_does_not_change_simulation_or_rng(local_tracks):
    a = make_env(local_tracks=local_tracks, record=True)
    b = make_env(local_tracks=local_tracks, record=True, record_episode=True)
    assert a.episode_records is None
    while a.agents:
        actions = {ag: 1 for ag in a.agents}
        assert_equal(a.step(actions), b.step(actions))
        assert_equal(a.frames, b.frames)
        assert_equal(a.log, b.log)
        assert_equal(a._last_reports, b._last_reports)
        for ta, tb in zip(a.targets, b.targets):
            assert_equal(ta.p, tb.p)
            assert_equal(ta.v, tb.v)
        for left, right in zip([a.tracker, *a.local_trackers], [b.tracker, *b.local_trackers]):
            assert len(left.tracks) == len(right.tracks)
            for x, y in zip(left.tracks, right.tracks):
                assert_equal(x.x, y.x)
                assert_equal(x.P, y.P)
        for name in ("rng_targets", "rng_radars", "rng_comm"):
            for x, y in zip(getattr(a, name), getattr(b, name)):
                assert_equal(x.bit_generator.state, y.bit_generator.state)
        assert_equal(a.rng_policy.bit_generator.state, b.rng_policy.bit_generator.state)


def test_full_precision_and_detached_arrays(monkeypatch):
    produced = {}
    original = env_module.radar_step

    def capture(*args, **kwargs):
        reports, state = original(*args, **kwargs)
        produced[kwargs["radar_index"]] = reports
        return reports, state

    monkeypatch.setattr(env_module, "radar_step", capture)
    env = make_env(record_episode=True, local_tracks=True)
    env.step({a: 1 for a in env.agents})
    records = env.episode_records
    assert records["reports"] and records["tracks"]
    for row in records["reports"]:
        index = int(row["report_id"].split(":")[-1])
        report = produced[row["radar_id"]][index]
        for field, original_field in (("z", "z"), ("R", "measurement_covariance")):
            assert_equal(row[field], report[original_field])
            assert not np.shares_memory(row[field], report[original_field])
    for row in records["tracks"]:
        tracker = env.tracker if row["source"] == "fused" else env.local_trackers[row["radar_id"]]
        track = next(t for t in tracker.tracks if t.id == row["track_id"])
        for field in ("x", "P"):
            assert_equal(row[field], getattr(track, field))
            assert not np.shares_memory(row[field], getattr(track, field))
    first = records["truth"][0]
    assert first["p"][0] == 4000.123456789
    assert first["p"][0] != round(first["p"][0], 1)
    assert not np.shares_memory(records["truth"][-1]["orientation"], env.targets[0].R_WT)
    preserved = deepcopy(records)
    env.step({a: 0 for a in env.agents})
    for table in ("states", "truth", "reports", "annotations", "tracks", "transitions"):
        assert_equal(records[table][:len(preserved[table])], preserved[table])
    before_reset = deepcopy(records)
    old_id = records["metadata"]["episode_id"]
    env.reset(seed=7)
    assert_equal(records, before_reset)
    assert env.episode_records["metadata"]["episode_id"] != old_id


def test_local_fused_track_keys_and_status():
    env = make_env(record_episode=True, local_tracks=True)
    run(env)
    rows = env.episode_records["tracks"]
    assert {r["source"] for r in rows} == {"fused", "local"}
    keys = [(r["step_index"], r["source"], r["radar_id"], r["track_id"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert all(r["radar_id"] == -1 for r in rows if r["source"] == "fused")
    assert {r["radar_id"] for r in rows if r["source"] == "local"} == {0, 1}
    assert any(r["confirmed"] for r in rows)
    assert {r["status"] for r in rows} <= {"tentative", "confirmed", "tentative_coasting", "confirmed_coasting"}


@pytest.mark.parametrize("complete", [False, True])
def test_export_import_roundtrip_without_pickle(tmp_path, complete):
    env = make_env(record_episode=True, local_tracks=True)
    if complete:
        run(env)
    metadata_path, arrays_path = export_episode_records(env.episode_records, tmp_path)
    assert metadata_path.name == "metadata.json" and arrays_path.name == "records.npz"

    def reject_constant(value):
        pytest.fail(f"non-standard JSON constant: {value}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert metadata["completed"] == complete
    with np.load(arrays_path, allow_pickle=False) as arrays:
        assert all(arrays[name].dtype.kind != "O" for name in arrays.files)
        assert arrays["reports_R"].shape[1:] == (4, 4)
        assert arrays["tracks_P"].shape[1:] == (6, 6)
        assert arrays["transitions_requested_actions"].shape[1:] == (2,)
    restored = load_episode_records(tmp_path)
    assert_equal(restored, env.episode_records)
    export_episode_records(restored, tmp_path / "again")
    assert_equal(load_episode_records(tmp_path / "again"), restored)


def test_empty_targets_no_detections_and_missing_actions(tmp_path):
    sc = scenario()
    sc["targets"] = []
    env = RadarNetEnv(sc, record_episode=True)
    env.reset()
    env.step({})
    rows = env.episode_records
    assert rows["truth"] == rows["reports"] == rows["tracks"] == rows["annotations"] == []
    assert not rows["transitions"][0]["action_present"].any()
    np.testing.assert_array_equal(rows["transitions"][0]["requested_actions"], [1, 1])
    export_episode_records(rows, tmp_path)
    assert_equal(load_episode_records(tmp_path), rows)


def test_precommunication_reports_and_annotations_do_not_leak(monkeypatch):
    original = TrackManager.process
    calls = []

    def inspect(self, t, reports, radars):
        for report in reports:
            assert "truth_id" not in report and "is_false_alarm" not in report
            assert "class_label" not in report and "orientation" not in report
        calls.append(len(reports))
        return original(self, t, reports, radars)

    monkeypatch.setattr(TrackManager, "process", inspect)
    sc = scenario()
    sc["stress"] = {"comm_drop_prob": 1.0}
    env = RadarNetEnv(sc, record_episode=True, local_tracks=True)
    obs, _ = env.reset()
    assert all(isinstance(v, np.ndarray) for v in obs.values())
    obs, *_ = env.step({a: 1 for a in env.agents})
    rows = env.episode_records
    assert rows["reports"] and env._last_reports == []
    assert any(calls) and env.tracker.tracks == []
    assert all(set(r).isdisjoint({"truth_id", "is_false_alarm"}) for r in rows["reports"])
    assert {r["report_id"] for r in rows["reports"]} == {r["report_id"] for r in rows["annotations"]}
    assert all(env.observation_space(a).contains(obs[a]) for a in env.agents)
    # Mutating record-only truth labels must not affect policy observations.
    before = {a: v.copy() for a, v in obs.items()}
    rows["truth"][-1]["p"][:] = 1e10
    rows["annotations"][0]["truth_id"] = 9999
    assert_equal(before, {a: env._observe(i) for i, a in enumerate(env.possible_agents)})


def test_outage_reports_are_copied_at_generation():
    sc = scenario()
    sc["stress"] = {"outages": [["all", 0, 10]]}
    env = RadarNetEnv(sc, record_episode=True, local_tracks=True)
    env.reset()
    run(env)
    assert env.episode_records["reports"]
    assert env._last_reports == []
    assert env.tracker.tracks == [] and all(lt.tracks == [] for lt in env.local_trackers)


def test_service_opt_in_and_disabled_export(tmp_path):
    with pytest.raises(ValueError, match="disabled"):
        export_episode_records(None, tmp_path)
    env, _ = run_episode(scenario(), classifier=None, record=False, record_episode=True)
    assert env.frames == [] and env.episode_records["metadata"]["completed"]


def test_false_alarm_annotations_are_separate():
    sc = scenario()
    sc["targets"] = []
    sc["radar_defaults"]["false_alarms_per_beam"] = 0.5
    env = RadarNetEnv(sc, record_episode=True)
    env.reset()
    run(env)
    records = env.episode_records
    assert records["reports"] and records["truth"] == []
    assert all(a["is_false_alarm"] and a["truth_id"] == -1 for a in records["annotations"])
    ids = [r["report_id"] for r in records["reports"]]
    assert len(ids) == len(set(ids))


def test_export_rejects_nonstandard_json_before_writing(tmp_path):
    records = make_env(record_episode=True).episode_records
    records["metadata"]["invalid"] = float("nan")
    with pytest.raises(ValueError):
        export_episode_records(records, tmp_path / "episode")
    assert not (tmp_path / "episode").exists()


@pytest.fixture
def three_reports(monkeypatch):
    """Three deterministic reports per active radar, with a stable quality tie."""
    original = env_module.radar_step

    def produce(targets, action, state, p, dt, rng, radar_index=0, terrain=None):
        _, state = original(targets, action, state, p, dt, rng, radar_index, terrain)
        reports = []
        if state.last_beams:
            for index, quality in enumerate((0.1, 0.9, 0.9)):
                report = _make_report(
                    np.array([4000.123456789 + index * 200, 0.01 * index, 0.05, -2.123456789]),
                    np.array([5.123456789, 0.001, 0.002, 0.2]), 20.123456789, 0.9,
                    p, radar_index, truth_id=None, false_alarm=True)
                report["signal_quality"] = quality
                reports.append(report)
        return reports, state

    monkeypatch.setattr(env_module, "radar_step", produce)


@pytest.mark.parametrize("stress,share,status,delivered_per_step", [
    ({}, True, "delivered", 6),
    ({"outages": [["all", 0, 10]]}, True, "outage_suppressed", 0),
    ({"max_reports_per_radar": 0}, True, "bandwidth_dropped", 0),
    ({"comm_drop_prob": 1.0}, True, "packet_lost", 0),
    ({}, False, "sharing_withheld", 0),
])
def test_each_report_has_one_final_status_and_actual_delivery(
        three_reports, monkeypatch, stress, share, status, delivered_per_step):
    sc = scenario()
    sc["share_actions"] = True
    sc["stress"] = stress
    env = RadarNetEnv(sc, seed=7, record_episode=True, local_tracks=True)
    env.reset()
    original = TrackManager.process
    central_calls = []

    def inspect(self, t, reports, radars):
        assert all(not {"truth_id", "is_false_alarm", "report_id", "final_status"}.intersection(r)
                   for r in reports)
        if self is env.tracker:
            central_calls.append(deepcopy(reports))
            step = env.step_count
            delivered = [r for r in env.episode_records["deliveries"] if r["step_index"] == step]
            by_id = {r["report_id"]: r for r in env.episode_records["reports"]}
            assert len(delivered) == len(reports) == delivered_per_step
            for index, (delivery, actual) in enumerate(zip(delivered, reports)):
                row = by_id[delivery["report_id"]]
                assert delivery["delivery_index"] == index
                assert row["radar_id"] == actual["radar"]
                assert_equal(row["z"], actual["z"])
                assert_equal(row["R"], actual["measurement_covariance"])
        return original(self, t, reports, radars)

    monkeypatch.setattr(TrackManager, "process", inspect)
    while env.agents:
        action = 1 if share else env.n_base_actions + 1
        env.step({a: action for a in env.agents})
    records = env.episode_records
    ids = [r["report_id"] for r in records["reports"]]
    assert len(ids) == len(set(ids)) == 6 * env.max_steps
    traces = records["communication"]
    assert [r["report_id"] for r in traces] == ids
    assert {r["final_status"] for r in traces} == {status}
    assert len(central_calls) == env.max_steps


@pytest.mark.parametrize("outage,loss,share", [
    (False, False, True), (False, True, True), (False, True, False),
    (False, False, False), (True, True, False),
])
def test_filter_precedence_and_stable_bandwidth_order(three_reports, tmp_path, outage, loss, share):
    sc = scenario()
    sc["share_actions"] = True
    sc["stress"] = {"max_reports_per_radar": 1, "comm_drop_prob": float(loss),
                    "outages": [["all", 0, 10]] if outage else []}
    env = RadarNetEnv(sc, record_episode=True)
    env.reset()
    while env.agents:
        env.step({a: 1 if share else env.n_base_actions + 1 for a in env.agents})
    for row in env.episode_records["communication"]:
        original_index = int(row["report_id"].split(":")[-1])
        expected = ("outage_suppressed" if outage else
                    "bandwidth_dropped" if original_index != 1 else
                    "packet_lost" if loss else "sharing_withheld" if not share else "delivered")
        assert row["final_status"] == expected
    if share and not loss and not outage:
        assert [r["report_id"] for r in env.episode_records["deliveries"]] == [
            f"{step}:{radar}:1" for step in range(1, env.max_steps + 1) for radar in range(2)]
    export_episode_records(env.episode_records, tmp_path)
    assert_equal(load_episode_records(tmp_path), env.episode_records)


def test_trace_does_not_change_stressed_simulation_or_precision(three_reports):
    sc = scenario()
    sc["share_actions"] = True
    sc["stress"] = {"max_reports_per_radar": 2, "comm_drop_prob": 0.5,
                    "outages": [[0, 0, 0.2]]}
    a = RadarNetEnv(sc, seed=7, record=True, local_tracks=True)
    b = RadarNetEnv(sc, seed=7, record=True, local_tracks=True, record_episode=True)
    a.reset()
    b.reset()
    while a.agents:
        actions = {a.possible_agents[0]: 1, a.possible_agents[1]: a.n_base_actions + 1}
        assert_equal(a.step(actions), b.step(actions))
        assert_equal(a._last_reports, b._last_reports)
        assert_equal(a.frames, b.frames)
        for name in ("rng_targets", "rng_radars", "rng_comm"):
            for left, right in zip(getattr(a, name), getattr(b, name)):
                assert_equal(left.bit_generator.state, right.bit_generator.state)
        assert_equal(a.rng_policy.bit_generator.state, b.rng_policy.bit_generator.state)
    for row in b.episode_records["reports"]:
        index = int(row["report_id"].split(":")[-1])
        assert row["z"][0] == 4000.123456789 + index * 200
        assert row["R"][0, 0] == np.float64(5.123456789) ** 2


def test_v1_roundtrip_has_no_inferred_communication(tmp_path):
    env = make_env(record_episode=True)
    run(env)
    legacy = deepcopy(env.episode_records)
    legacy["metadata"]["record_schema_version"] = 1
    legacy["metadata"].pop("communication_order")
    legacy.pop("communication")
    legacy.pop("deliveries")
    legacy.pop("radar_resources")
    legacy.pop("reward_terms")
    export_episode_records(legacy, tmp_path)
    restored = load_episode_records(tmp_path)
    assert_equal(restored, legacy)
    assert "deliveries" not in restored and "communication" not in restored


def test_v2_records_still_roundtrip_without_resource_tables(tmp_path):
    env = make_env(record_episode=True)
    run(env)
    v2 = deepcopy(env.episode_records)
    v2["metadata"]["record_schema_version"] = 2
    v2.pop("radar_resources")
    v2.pop("reward_terms")
    export_episode_records(v2, tmp_path)
    restored = load_episode_records(tmp_path)
    assert_equal(restored, v2)
    assert "radar_resources" not in restored


def test_v3_records_resources_and_reward_terms(tmp_path):
    from src.core.env import REWARD_TERMS
    env = make_env(record_episode=True)
    run(env)
    rec = env.episode_records
    steps = len(rec["transitions"])
    assert len(rec["radar_resources"]) == steps * env.n_radars
    assert len(rec["reward_terms"]) == steps
    w = env.sc["reward"]
    for row, tr in zip(rec["reward_terms"], rec["transitions"]):
        total = sum(sign * w.get(key, 0.0) * row[term] for term, (key, sign) in REWARD_TERMS.items())
        assert total == pytest.approx(row["reward"]) and row["reward"] == pytest.approx(tr["rewards"][0])
    export_episode_records(rec, tmp_path)
    assert_equal(load_episode_records(tmp_path), rec)


@pytest.mark.parametrize("version", [0, 4, True, "2"])
def test_unsupported_record_versions_fail_explicitly(tmp_path, version):
    records = make_env(record_episode=True).episode_records
    records["metadata"]["record_schema_version"] = version
    with pytest.raises(ValueError, match="unsupported episode record schema version"):
        export_episode_records(records, tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps(records["metadata"]), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported episode record schema version"):
        load_episode_records(tmp_path)


def test_v2_export_rejects_unfinished_or_inconsistent_trace(tmp_path):
    env = make_env(record_episode=True)
    run(env)
    records = env.episode_records
    records["communication"][0]["final_status"] = "pending"
    with pytest.raises(ValueError, match="unfinished status"):
        export_episode_records(records, tmp_path)
    records["communication"][0]["final_status"] = "delivered"
    records["deliveries"].pop(0)
    with pytest.raises(ValueError, match="invalid order|match exactly"):
        export_episode_records(records, tmp_path)


def test_schema_layout_mismatch_is_not_silently_read_as_v1(tmp_path):
    env = make_env(record_episode=True)
    export_episode_records(env.episode_records, tmp_path)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["record_schema_version"] = 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="array layout"):
        load_episode_records(tmp_path)


def test_empty_communication_tables_roundtrip(tmp_path):
    sc = scenario()
    sc["targets"] = []
    env = RadarNetEnv(sc, record_episode=True)
    env.reset()
    run(env)
    assert env.episode_records["communication"] == env.episode_records["deliveries"] == []
    export_episode_records(env.episode_records, tmp_path)
    with np.load(tmp_path / "records.npz", allow_pickle=False) as arrays:
        assert arrays["communication_final_status"].shape == (0,)
        assert arrays["deliveries_delivery_index"].shape == (0,)
    assert_equal(load_episode_records(tmp_path), env.episode_records)
