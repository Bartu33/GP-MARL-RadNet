"""Sequential CPU generation from a 6A plan; one canonical episode in memory.

Explicit call only. Completed outputs are immutable on resume. Invalid/partial
outputs are reported as failed and retained for inspection; remove the affected
episode directory (including .pending) before retrying. One writer per dataset.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time

import numpy as np

from src.core.env import _TERRAIN_CACHE
from src.core.scenario import scenario_snapshot, validate_scenario
from src.services.scenario_planner import _digest, terrain_group_id
from src.services.simulation_service import (
    export_episode_records, load_episode_records, run_episode,
)

MAX_TERRAINS = 16
FILES = ("metadata.json", "records.npz")
IDENTITY = ("dataset_episode_id", "episode_index", "split", "scenario_family",
            "terrain_group_id", "master_seed", "episode_seed",
            "resolved_configuration_hash", "recipe_version")


def _atomic_json(value, path):
    encoded = json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_plan(plan):
    if plan.get("plan_schema_version") != 1 or plan.get("recipe_version") != 1:
        raise ValueError("only planner schema/recipe version 1 is supported")
    if not isinstance(plan.get("episodes"), list) or not plan["episodes"]:
        raise ValueError("plan.episodes: expected a nonempty list")
    ids, terrains, owners, routes = set(), set(), {}, {}
    dimensions = set()
    for episode in plan["episodes"]:
        key = episode["dataset_episode_id"]
        if not isinstance(key, str) or not re.fullmatch(r"ep-\d{6,}-[0-9a-f]{12}", key) or key in ids:
            raise ValueError("dataset_episode_id: invalid or duplicate ID")
        ids.add(key)
        sc = validate_scenario(episode["resolved_scenario"])
        if len(sc["targets"]) != 1:
            raise ValueError(f"{key}: generation v1 requires one target")
        snapshot = scenario_snapshot(sc, episode_seed=episode["episode_seed"])
        if (_digest(snapshot) != episode["resolved_configuration_hash"]
                or snapshot != episode["config_snapshot"]):
            raise ValueError(f"{key}: resolved configuration hash/snapshot mismatch")
        if episode["recipe_version"] != 1 or episode["master_seed"] != plan["master_seed"]:
            raise ValueError(f"{key}: recipe version/master seed mismatch")
        spec = sc["terrain"]
        group = terrain_group_id(spec)
        if group != episode["terrain_group_id"]:
            raise ValueError(f"{key}: terrain group mismatch")
        split = episode["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError(f"{key}: unknown split")
        for groups, identity in ((owners, group), (routes, episode["route_group_id"])):
            if groups.setdefault(identity, split) != split:
                raise ValueError(f"{key}: group crosses splits")
        terrains.add(_digest([sc["site"], spec, sc["seed"]]))
        dimensions.add((sc["dt"], len(sc["radars"]), sc["max_track_slots"]))
    if len(terrains) > MAX_TERRAINS:
        raise ValueError(f"terrain pool exceeds bounded limit {MAX_TERRAINS}")
    if len(dimensions) != 1:
        raise ValueError("dt, radar count and observation/action dimensions must be fixed")


def _inspect(directory, episode, binding, expected=None):
    """Load with the canonical pickle-free reader, then verify provenance/counts."""
    hashes = {name: _file_hash(directory / name) for name in FILES}
    if expected and hashes != expected["file_sha256"]:
        raise ValueError("episode file checksum mismatch")
    records = load_episode_records(directory)
    metadata = records["metadata"]
    snapshot = metadata["config_snapshot"]
    if (metadata["record_schema_version"] != 2 or metadata["completed"] is not True
            or metadata.get("dataset_provenance") != binding
            or snapshot["scenario"] != episode["resolved_scenario"]
            or snapshot["episode_seed"] != episode["episode_seed"]):
        raise ValueError("episode completion/configuration/provenance mismatch")
    steps = episode["config_snapshot"]["simulation_steps"]
    if (metadata["recorded_steps"] != steps or len(records["states"]) != steps + 1
            or len(records["transitions"]) != steps or len(records["truth"]) != steps + 1):
        raise ValueError("episode state/transition/truth counts mismatch")
    if [row["step_index"] for row in records["states"]] != list(range(steps + 1)):
        raise ValueError("episode state indices mismatch")
    for row in records["states"]:
        if not np.isclose(row["t_s"], row["step_index"] * snapshot["scenario"]["dt"], rtol=0, atol=1e-9):
            raise ValueError("episode timestamps mismatch")
    with np.load(directory / "records.npz", allow_pickle=False) as arrays:
        if any(not np.isfinite(arrays[key]).all() for key in arrays.files if arrays[key].dtype.kind == "f"):
            raise ValueError("episode contains non-finite numeric records")
    counts = {table: len(rows) for table, rows in records.items() if table != "metadata"}
    counts["confirmed_track_rows"] = sum(row["confirmed"] for row in records["tracks"])
    counts["fused_track_rows"] = sum(row["source"] == "fused" for row in records["tracks"])
    result = {"runtime_episode_id": metadata["episode_id"], "counts": counts, "file_sha256": hashes,
              "output_bytes": sum((directory / name).stat().st_size for name in FILES)}
    if expected and any(result[key] != expected[key] for key in result):
        raise ValueError("manifest counts/runtime identity do not match episode files")
    return result


def generate_dataset(plan, output_directory, *, policy_name="central_greedy", resume=True):
    """Generate or verify a plan, returning a persisted manifest (including errors).

    No classifier, UI frames, training, resplitting or simulation changes. Errors
    are explicit failed entries; configuration conflicts raise before any run.
    Incomplete directories are never overwritten. A single-writer lock prevents
    overlapping exports; after a process crash remove the stale lock explicitly.
    """
    plan = deepcopy(plan)
    _validate_plan(plan)
    root = Path(output_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".generation.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    try:
        return _generate(plan, root, policy_name, resume)
    finally:
        lock.unlink()


def _generate(plan, root, policy_name, resume):
    manifest_path = root / "manifest.json"
    plan_hash, recipe_hash = _digest(plan), _digest(plan["recipe"])
    header = {"manifest_schema_version": 1, "plan_hash": plan_hash, "recipe_hash": recipe_hash,
              "policy_name": policy_name, "classifier": None, "worker_count": 1,
              "numpy_version": np.__version__}
    entries = [{**{key: ep[key] for key in IDENTITY}, "generation_status": "planned",
                "runtime_episode_id": None, "output_paths": {
                    name: f"{ep['split']}/{ep['dataset_episode_id']}/{name}" for name in FILES},
                "counts": None, "error_reason": None} for ep in plan["episodes"]]
    if manifest_path.exists():
        if not resume:
            raise ValueError("dataset manifest exists; use resume or a new output directory")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(manifest.get(key) != value for key, value in header.items()):
            raise ValueError("dataset recipe/plan/configuration/policy/runtime version mismatch")
        if len(manifest["episodes"]) != len(entries):
            raise ValueError("dataset manifest episode count mismatch")
        for old, new in zip(manifest["episodes"], entries):
            if any(old.get(key) != new[key] for key in (*IDENTITY, "output_paths")):
                raise ValueError("dataset manifest identity/output path mismatch")
    else:
        manifest = {**header, "generation_status": "running", "episodes": entries}
        _atomic_json(manifest, manifest_path)
    manifest["generation_status"] = "running"
    _atomic_json(manifest, manifest_path)
    wall, cpu = time.perf_counter(), time.process_time()
    skipped = 0
    for episode, entry in zip(plan["episodes"], manifest["episodes"]):
        directory = root / episode["split"] / episode["dataset_episode_id"]
        pending = directory.with_name(directory.name + ".pending")
        binding = {**{key: episode[key] for key in IDENTITY}, "plan_hash": plan_hash,
                   "recipe_hash": recipe_hash, "policy_name": policy_name}
        before = set(_TERRAIN_CACHE)
        episode_wall, episode_cpu = time.perf_counter(), time.process_time()
        try:
            if entry["generation_status"] == "completed":
                _inspect(directory, episode, binding, expected=entry)
                skipped += 1
                continue
            if pending.exists():
                raise ValueError("incomplete .pending output exists; inspect/remove it before retry")
            if directory.exists():
                # Recover a crash after directory publication but before manifest commit.
                result = _inspect(directory, episode, binding,
                                  expected=entry if "file_sha256" in entry else None)
                skipped += 1
            else:
                entry.update(generation_status="running", error_reason=None)
                _atomic_json(manifest, manifest_path)
                env, _ = run_episode(episode["resolved_scenario"], policy_name=policy_name,
                                     seed=episode["episode_seed"], classifier=None,
                                     record=False, record_episode=True)
                try:
                    env.episode_records["metadata"]["dataset_provenance"] = binding
                    export_episode_records(env.episode_records, pending)
                finally:
                    del env
                result = _inspect(pending, episode, binding)
                for name in FILES:
                    # Windows _commit/fsync requires a writable file handle.
                    with (pending / name).open("r+b") as stream:
                        os.fsync(stream.fileno())
                pending.rename(directory)
            entry.update(result, generation_status="completed", error_reason=None,
                         wall_time_s=time.perf_counter() - episode_wall,
                         cpu_time_s=time.process_time() - episode_cpu)
        except Exception as exc:
            entry.update(generation_status="failed", error_reason=f"{type(exc).__name__}: {exc}")
        finally:
            # Do not evict terrain entries owned by other callers; this generator
            # retains at most the current episode's newly-created cache entries.
            for key in set(_TERRAIN_CACHE) - before:
                del _TERRAIN_CACHE[key]
            _atomic_json(manifest, manifest_path)
    manifest.update(generation_status="completed" if all(
        row["generation_status"] == "completed" for row in manifest["episodes"]) else "failed",
        last_run={"skipped_verified": skipped, "wall_time_s": time.perf_counter() - wall,
                  "cpu_time_s": time.process_time() - cpu})
    _atomic_json(manifest, manifest_path)
    return manifest


def main():
    import argparse
    from src.services.scenario_planner import load_recipe, plan_episodes

    parser = argparse.ArgumentParser(description="Generate synthetic dataset from scenario planner recipe.")
    parser.add_argument("--recipe", default="configs/datasets/planner_100.yaml", help="Path to dataset recipe YAML")
    parser.add_argument("--output", default="data/processed/synthetic_100", help="Output directory")
    parser.add_argument("--policy", default="central_greedy", help="Policy name")
    parser.add_argument("--episodes", type=int, default=None, help="Optional override for episode count")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume (fail if manifest exists)")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    recipe = load_recipe(Path(args.recipe))
    if args.episodes is not None:
        recipe["episodes"] = args.episodes
    plan = plan_episodes(recipe)
    manifest = generate_dataset(plan, args.output, policy_name=args.policy, resume=args.resume)
    print(f"Dataset generation finished: status={manifest['generation_status']}, episodes={len(manifest['episodes'])}")


if __name__ == "__main__":
    main()

