"""Small CPU-only canonical generation and verified, non-destructive resume."""
from copy import deepcopy
import json
import weakref

import numpy as np
import pytest

from src.core.env import _TERRAIN_CACHE
from src.core.scenario import ROOT
from src.services.scenario_planner import load_recipe, plan_episodes
from src.services.simulation_service import load_episode_records, run_episode
import src.services.synthetic_dataset_service as service


def plan(episodes=3, *, demo=False, zero=False):
    recipe = load_recipe(ROOT / "configs/datasets/planner_demo.yaml")
    recipe["episodes"] = episodes
    if not demo:
        recipe["overrides"]["duration_s"] = 1
    if zero:
        recipe["overrides"]["radar_defaults"] = {"r_min_m": 0, "r_max_m": 1,
                                                        "false_alarms_per_beam": 0}
    return plan_episodes(recipe)


def output(root, entry):
    return root / entry["output_paths"]["metadata.json"].rsplit("/", 1)[0]


@pytest.mark.parametrize("episodes,demo", [(3, False), (10, True)])
def test_cpu_generation_and_pickle_free_roundtrip(tmp_path, episodes, demo):
    planned = plan(episodes, demo=demo)
    before = deepcopy(planned)
    manifest = service.generate_dataset(planned, tmp_path)
    assert planned == before
    assert manifest["generation_status"] == "completed", manifest
    assert len(manifest["episodes"]) == episodes
    persisted = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert persisted == manifest
    for source, entry in zip(planned["episodes"], manifest["episodes"]):
        assert all(entry[key] == source[key] for key in service.IDENTITY)
        directory = output(tmp_path, entry)
        records = load_episode_records(directory)
        assert entry["runtime_episode_id"] == records["metadata"]["episode_id"]
        assert entry["runtime_episode_id"] != entry["dataset_episode_id"]
        assert records["metadata"]["record_schema_version"] == 3
        assert records["metadata"]["config_snapshot"]["episode_seed"] == entry["episode_seed"]
        for table, rows in records.items():
            if table != "metadata":
                assert entry["counts"][table] == len(rows)
        assert entry["counts"]["states"] == entry["counts"]["transitions"] + 1
        with np.load(directory / "records.npz", allow_pickle=False) as arrays:
            for key in arrays.files:
                assert arrays[key].dtype.kind != "O"
        assert entry["file_sha256"] == {name: service._file_hash(directory / name) for name in service.FILES}
    expected_files = {tmp_path / "manifest.json"} | {
        tmp_path / path for row in manifest["episodes"] for path in row["output_paths"].values()}
    assert {path for path in tmp_path.rglob("*") if path.is_file()} == expected_files
    assert not (tmp_path / ".generation.lock").exists()


def test_resume_verifies_and_skips_all_without_rewriting_episode_files(tmp_path, monkeypatch):
    planned = plan()
    first = service.generate_dataset(planned, tmp_path)
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.name in service.FILES}
    def forbidden(*args, **kwargs):
        pytest.fail("verified episodes must not be simulated again")
    monkeypatch.setattr(service, "run_episode", forbidden)
    resumed = service.generate_dataset(planned, tmp_path)
    assert resumed["last_run"]["skipped_verified"] == 3
    assert resumed["episodes"] == first["episodes"]
    assert all(path.read_bytes() == data for path, data in originals.items())


@pytest.mark.parametrize("kind", ["missing", "corrupt", "changed_counts"])
def test_resume_detects_bad_output_without_silent_reuse(tmp_path, monkeypatch, kind):
    planned = plan()
    first = service.generate_dataset(planned, tmp_path)
    entry = first["episodes"][0]
    path = output(tmp_path, entry) / "records.npz"
    if kind == "missing":
        path.unlink()
    elif kind == "corrupt":
        path.write_bytes(b"broken zip")
    else:
        first["episodes"][0]["counts"]["states"] += 1
        (tmp_path / "manifest.json").write_text(json.dumps(first), encoding="utf-8")
    def forbidden(*args, **kwargs):
        pytest.fail("bad completed output must be reported, not rerun/overwritten")
    monkeypatch.setattr(service, "run_episode", forbidden)
    resumed = service.generate_dataset(planned, tmp_path)
    assert resumed["generation_status"] == "failed"
    assert resumed["episodes"][0]["generation_status"] == "failed"
    assert resumed["episodes"][0]["error_reason"]
    assert resumed["last_run"]["skipped_verified"] == 2
    again = service.generate_dataset(planned, tmp_path)
    assert again["episodes"][0]["generation_status"] == "failed"


@pytest.mark.parametrize("change", ["recipe", "configuration", "policy"])
def test_mismatched_recipe_configuration_or_policy_rejected(tmp_path, change):
    planned = plan()
    service.generate_dataset(planned, tmp_path)
    if change == "recipe":
        planned["recipe"]["max_attempts"] += 1
    elif change == "configuration":
        planned["episodes"][0]["resolved_scenario"]["stress"]["noise_scale"] += 0.1
    with pytest.raises(ValueError, match="mismatch"):
        service.generate_dataset(planned, tmp_path, policy_name="random" if change == "policy" else "central_greedy")


def test_zero_detections_and_zero_confirmed_tracks_are_completed(tmp_path):
    manifest = service.generate_dataset(plan(zero=True), tmp_path)
    assert manifest["generation_status"] == "completed", manifest
    for entry in manifest["episodes"]:
        assert entry["counts"]["reports"] == entry["counts"]["confirmed_track_rows"] == 0
        assert entry["counts"]["states"] > 0


@pytest.mark.parametrize("stage", ["simulation", "export"])
def test_real_errors_are_persisted_and_no_partial_episode_is_published(tmp_path, monkeypatch, stage):
    def broken(*args, **kwargs):
        if stage == "export":
            directory = args[1]
            directory.mkdir(parents=True)
            (directory / "records.npz").write_bytes(b"partial")
        raise RuntimeError("injected real failure")
    monkeypatch.setattr(service, "run_episode" if stage == "simulation" else "export_episode_records", broken)
    manifest = service.generate_dataset(plan(), tmp_path)
    assert manifest["generation_status"] == "failed"
    for row in manifest["episodes"]:
        assert row["generation_status"] == "failed"
        assert "injected real failure" in row["error_reason"]
        assert not output(tmp_path, row).exists()


def test_crash_after_episode_publish_is_recovered_without_rerun(tmp_path, monkeypatch):
    planned = plan()
    manifest = service.generate_dataset(planned, tmp_path)
    row = manifest["episodes"][0]
    row["generation_status"] = "running"
    for key in ("file_sha256", "counts", "runtime_episode_id", "output_bytes"):
        row.pop(key)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(service, "run_episode", lambda *a, **kw: pytest.fail("must recover published files"))
    resumed = service.generate_dataset(planned, tmp_path)
    assert resumed["generation_status"] == "completed"
    assert resumed["last_run"]["skipped_verified"] == 3


def test_one_episode_in_memory_and_no_terrain_cache_growth(tmp_path, monkeypatch):
    previous = []
    original = service.run_episode
    before = set(_TERRAIN_CACHE)
    def run(*args, **kwargs):
        assert not previous or previous[-1]() is None
        env, summary = original(*args, **kwargs)
        previous.append(weakref.ref(env))
        return env, summary
    monkeypatch.setattr(service, "run_episode", run)
    assert service.generate_dataset(plan(), tmp_path)["generation_status"] == "completed"
    assert set(_TERRAIN_CACHE) == before
    assert all(ref() is None for ref in previous)


def test_bounded_pool_and_single_writer_guard(tmp_path, monkeypatch):
    planned = plan(10)
    monkeypatch.setattr(service, "MAX_TERRAINS", 2)
    with pytest.raises(ValueError, match="bounded limit"):
        service.generate_dataset(planned, tmp_path)
    monkeypatch.setattr(service, "MAX_TERRAINS", 16)
    (tmp_path / ".generation.lock").touch()
    with pytest.raises(FileExistsError):
        service.generate_dataset(planned, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_generation_records_match_recording_disabled_run(tmp_path):
    planned = plan(1)
    manifest = service.generate_dataset(planned, tmp_path)
    records = load_episode_records(output(tmp_path, manifest["episodes"][0]))
    episode = planned["episodes"][0]
    env, _ = run_episode(episode["resolved_scenario"], seed=episode["episode_seed"],
                         classifier=None, record=False, record_episode=False)
    np.testing.assert_array_equal(records["truth"][-1]["p"], env.targets[0].p)
    np.testing.assert_array_equal(records["truth"][-1]["v"], env.targets[0].v)
    np.testing.assert_array_equal([r["rewards"][0] for r in records["transitions"]], env.log["reward"])


def test_incomplete_staging_is_reported_and_resume_false_refuses_overwrite(tmp_path):
    planned = plan()
    episode = planned["episodes"][0]
    pending = tmp_path / episode["split"] / (episode["dataset_episode_id"] + ".pending")
    pending.mkdir(parents=True)
    manifest = service.generate_dataset(planned, tmp_path)
    assert "incomplete .pending" in manifest["episodes"][0]["error_reason"]
    with pytest.raises(ValueError, match="manifest exists"):
        service.generate_dataset(planned, tmp_path, resume=False)
