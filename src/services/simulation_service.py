"""Run episodes, collect metrics and compare policies."""
from __future__ import annotations

import time
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.env import REPORT_FINAL_STATUSES, RadarNetEnv
from src.core.scenario import load_scenario
from src.models.baselines.policies import make_policy

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = ROOT / "models" / "checkpoints" / "flight_profile_gbm.joblib"

# Column dtype and trailing shape. None denotes the fixed episode agent count.
# v1 tables; v2 adds EPISODE_TRACE_FIELDS. Rows are sparse except states.
EPISODE_FIELDS = {
    "states": {"step_index": ("int64", ()), "t_s": ("float64", ())},
    "truth": {
        "step_index": ("int64", ()), "t_s": ("float64", ()),
        "target_id": ("int64", ()), "class_label": ("int64", ()),
        "p": ("float64", (3,)), "v": ("float64", (3,)), "orientation": ("float64", (3, 3)),
    },
    "reports": {
        "step_index": ("int64", ()), "t_s": ("float64", ()),
        "radar_id": ("int64", ()), "report_id": ("str", ()),
        "z": ("float64", (4,)), "R": ("float64", (4, 4)),
        "world_xyz": ("float64", (3,)), "snr_db": ("float64", ()),
        "signal_quality": ("float64", ()), "detection_probability": ("float64", ()),
        "doppler_hz": ("float64", ()), "detection_valid": ("int64", ()),
    },
    "annotations": {
        "report_id": ("str", ()), "truth_id": ("int64", ()), "is_false_alarm": ("bool", ()),
    },
    "tracks": {
        "step_index": ("int64", ()), "t_s": ("float64", ()),
        "source": ("str", ()), "radar_id": ("int64", ()), "track_id": ("int64", ()),
        "x": ("float64", (6,)), "P": ("float64", (6, 6)),
        "confirmed": ("bool", ()), "status": ("str", ()), "hits": ("int64", ()),
        "born_t_s": ("float64", ()), "last_update_t_s": ("float64", ()),
    },
    "transitions": {
        "step_index": ("int64", ()), "start_t_s": ("float64", ()), "end_t_s": ("float64", ()),
        "requested_actions": ("int64", None), "action_present": ("bool", None),
        "rewards": ("float64", None), "terminations": ("bool", None), "truncations": ("bool", None),
    },
}

EPISODE_TRACE_FIELDS = {
    "communication": {
        "step_index": ("int64", ()), "t_s": ("float64", ()),
        "radar_id": ("int64", ()), "report_id": ("str", ()), "final_status": ("str", ()),
    },
    "deliveries": {
        "step_index": ("int64", ()), "t_s": ("float64", ()),
        "radar_id": ("int64", ()), "report_id": ("str", ()), "delivery_index": ("int64", ()),
    },
}


def _episode_fields(version):
    """v1 has no communication evidence; never infer delivery from raw reports."""
    if type(version) is not int or version not in (1, 2):
        raise ValueError(f"unsupported episode record schema version: {version!r}; supported: 1, 2")
    return EPISODE_FIELDS if version == 1 else {**EPISODE_FIELDS, **EPISODE_TRACE_FIELDS}


def _validate_report_trace(records):
    reports = {row["report_id"]: row for row in records["reports"]}
    traces = {row["report_id"]: row for row in records["communication"]}
    if (len(reports) != len(records["reports"]) or len(traces) != len(records["communication"])
            or reports.keys() != traces.keys()):
        raise ValueError("communication: require exactly one trace per unique generated report")
    for report_id, trace in traces.items():
        if trace["final_status"] not in REPORT_FINAL_STATUSES:
            raise ValueError(f"communication.{report_id}.final_status: unknown or unfinished status")
        if any(trace[k] != reports[report_id][k] for k in ("step_index", "t_s", "radar_id")):
            raise ValueError(f"communication.{report_id}: report timestamp or radar mismatch")
    delivered = set()
    indices = {}
    for row in records["deliveries"]:
        report_id, step = row["report_id"], row["step_index"]
        if report_id in delivered or report_id not in reports:
            raise ValueError("deliveries: duplicate or unknown report ID")
        if (row["delivery_index"] != indices.get(step, 0)
                or any(row[k] != reports[report_id][k] for k in ("step_index", "t_s", "radar_id"))):
            raise ValueError(f"deliveries.{report_id}: invalid order, timestamp, or radar")
        delivered.add(report_id)
        indices[step] = indices.get(step, 0) + 1
    if delivered != {key for key, row in traces.items() if row["final_status"] == "delivered"}:
        raise ValueError("deliveries: IDs must match exactly the delivered final statuses")


def export_episode_records(records: dict, directory: str | Path) -> tuple[Path, Path]:
    """Explicit export of env.episode_records; never called by reset/step.

    NPZ keys are '<table>_<field>'. Numeric arrays retain float64 precision;
    variable-length tables use rows, not object arrays. Existing files at the
    destination are replaced, so use a separate directory per episode.
    """
    if records is None:
        raise ValueError("episode recording is disabled or reset has not been called")
    metadata = deepcopy(records["metadata"])
    fields_by_table = _episode_fields(metadata["record_schema_version"])
    if metadata["record_schema_version"] == 2:
        _validate_report_trace(records)
    elif any(table in records for table in EPISODE_TRACE_FIELDS):
        raise ValueError("schema v1 cannot contain communication/delivery tables; use v2")
    # Serialize before creating output files; reject non-standard JSON values.
    encoded = json.dumps(metadata, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    arrays = {}
    for table, fields in fields_by_table.items():
        rows = records[table]
        for field, (dtype, tail) in fields.items():
            tail = (len(metadata["agent_ids"]),) if tail is None else tail
            arrays[f"{table}_{field}"] = np.asarray([row[field] for row in rows], dtype=dtype).reshape(
                (len(rows), *tail))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata_path, records_path = directory / "metadata.json", directory / "records.npz"
    np.savez_compressed(records_path, **arrays)
    metadata_path.write_text(encoded, encoding="utf-8")
    return metadata_path, records_path


def load_episode_records(directory: str | Path) -> dict:
    """Read v1 unchanged (trace unavailable), or v2 with communication evidence.

    Unsupported versions and metadata/NPZ layout mismatches are explicit errors.
    """
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    fields_by_table = _episode_fields(metadata["record_schema_version"])
    records = {"metadata": metadata}
    with np.load(directory / "records.npz", allow_pickle=False) as arrays:
        expected = {f"{table}_{field}" for table, fields in fields_by_table.items() for field in fields}
        if set(arrays.files) != expected:
            raise ValueError("records.npz: array layout does not match declared schema version")
        for table, fields in fields_by_table.items():
            columns = {field: arrays[f"{table}_{field}"] for field in fields}
            count = len(columns["step_index"] if "step_index" in columns else columns["report_id"])
            for field, (dtype, tail) in fields.items():
                tail = (len(metadata["agent_ids"]),) if tail is None else tail
                column = columns[field]
                valid_dtype = column.dtype.kind == "U" if dtype == "str" else column.dtype == np.dtype(dtype)
                if column.shape != (count, *tail) or not valid_dtype:
                    raise ValueError(f"{table}.{field}: invalid shape or dtype")
            records[table] = [
                {field: column[i].copy() if column.ndim > 1 else column[i].item()
                 for field, column in columns.items()}
                for i in range(count)
            ]
    if metadata["record_schema_version"] == 2:
        _validate_report_trace(records)
    return records


def load_classifier(path: Path = CLASSIFIER_PATH):
    if not Path(path).exists():
        return None
    from src.models.classifier.classifier import FlightProfileClassifier
    return FlightProfileClassifier.load(path)


def single_radar_summary(env: RadarNetEnv) -> dict:
    """Structure A: each radar alone with its own local track, on the same
    detections as the fused track.  Coverage uses the network's coverage mask
    so that it is directly comparable with ``coverage_in_range``."""
    if not env.local_eval:
        return {}
    net_cov = np.concatenate([np.asarray(c, bool) for c in env.evaluator.in_cov])
    cov, longest, rmse = [], [], []
    for ev in env.local_eval:
        valid = np.concatenate([np.asarray(v, bool) for v in ev.valid])
        cov.append(float(valid[net_cov].mean()) if net_cov.any() else np.nan)
        s = ev.summary()
        longest.append(s["longest_track_s"])
        rmse.append(s["pos_rmse_m"])
    return {
        "single_best_coverage": float(np.nanmax(cov)),
        "single_best_longest_s": float(np.nanmax(longest)),
        "single_mean_rmse_m": float(np.nanmean(rmse)) if np.any(np.isfinite(rmse)) else np.nan,
    }


def per_radar_table(env: RadarNetEnv) -> pd.DataFrame:
    """Local single-radar tracks next to the fused track (one row each)."""
    rows = []
    for p, ev in zip(env.radar_params, env.local_eval):
        rows.append({"track": f"{p.name} (local)", **ev.summary()})
    rows.append({"track": "Fused (network)", **env.evaluator.summary()})
    cols = ["track", "coverage_in_range", "pos_rmse_m", "longest_track_s", "track_losses", "interruptions",
            "reacquire_s", "correct_class_continuity"]
    return pd.DataFrame(rows)[cols]


def summarize(env: RadarNetEnv) -> dict:
    lg = env.log
    return {
        # coverage, continuity, detection / track loss: see src/core/evaluation.py (METRIC_INFO)
        **env.evaluator.summary(),
        **single_radar_summary(env),
        "false_tracks_mean": float(np.mean(lg["false_tracks"])) if lg["false_tracks"] else 0.0,
        "class_accuracy": float(np.nanmean(lg["class_acc"])) if np.any(np.isfinite(lg["class_acc"])) else np.nan,
        "sensing_cost": float(np.mean(lg["energy"])) if lg["energy"] else 0.0,
        "reports_per_step": float(np.mean(lg["reports"])) if lg["reports"] else 0.0,
        "return": float(np.sum(lg["reward"])),
        # Jain fairness of the radars' cumulative tracking time (sensor-usage balance)
        "usage_fairness": float(lg["fairness"][-1]) if lg["fairness"] else 1.0,
    }


def run_episode(scenario, policy_name: str = "central_greedy", seed: int | None = None,
                classifier="auto", record: bool = True, record_episode: bool = False):
    """Run one episode.  Returns (env, summary).  ``env.frames`` holds the
    recorded frames for the 3-D viewer when ``record`` is True."""
    sc = load_scenario(scenario) if isinstance(scenario, str) else scenario
    clf = load_classifier() if classifier == "auto" else classifier
    env = RadarNetEnv(sc, classifier=clf, record=record, seed=seed, local_tracks=True,
                      record_episode=record_episode)
    obs, _ = env.reset(seed=seed)
    policy = make_policy(policy_name)
    policy.reset(env)
    decide = []
    while env.agents:
        t0 = time.perf_counter()
        actions = policy.act(env, obs)
        decide.append(time.perf_counter() - t0)
        obs, _, _, _, _ = env.step(actions)
    return env, {**summarize(env), "decision_ms": float(np.mean(decide) * 1000) if decide else np.nan}


def compare_policies(scenario, policies, seeds=(0, 1, 2), classifier="auto", progress=None) -> pd.DataFrame:
    """Run every policy on every seed.  ``progress(done, total, label)`` is
    called after each episode (used by the dashboard's progress bar)."""
    rows = []
    clf = load_classifier() if classifier == "auto" else classifier
    seeds = list(seeds)
    total, done = len(policies) * len(seeds), 0
    for pol in policies:
        for s in seeds:
            _, summ = run_episode(scenario, pol, seed=s, classifier=clf, record=False)
            rows.append({"policy": pol, "seed": s, **summ})
            done += 1
            if progress is not None:
                progress(done, total, f"{pol} · seed {s}")
    return pd.DataFrame(rows)


# Locations offered by the map picker of the 3-D view (display only: the
# simulation itself is flat-earth and does not depend on the site).
MAP_SITES = [
    {"name": "Utah Valley · US", "lat": 40.30, "lon": -111.70},
    {"name": "Bursa · Uludağ · TR", "lat": 40.14, "lon": 29.10},
    {"name": "Kayseri · Erciyes · TR", "lat": 38.58, "lon": 35.44},
    {"name": "Ankara · TR", "lat": 39.93, "lon": 32.85},
    {"name": "İzmir · TR", "lat": 38.42, "lon": 27.14},
    {"name": "Kapadokya · Göreme · TR", "lat": 38.64, "lon": 34.83},
    {"name": "Van · TR", "lat": 38.50, "lon": 43.38},
    {"name": "Zermatt · Alps · CH", "lat": 46.02, "lon": 7.75},
    {"name": "Grand Canyon · US", "lat": 36.10, "lon": -112.11},
]


def viewer_payload(env: RadarNetEnv) -> dict:
    """Static scene description + frames, consumed by the 3-D viewer."""
    sc = env.sc
    return {
        "site": sc["site"],
        "sites": [sc["site"]] + [s for s in MAP_SITES if s["name"] != sc["site"]["name"]],
        "terrain": env.terrain.to_payload(),
        "dt": env.dt,
        "area_radius": sc["area_radius_m"],
        "radars": [{
            "name": p.name, "pos": [float(x) for x in p.position], "heading": float(p.heading),
            "fov": float(np.deg2rad(p.fov_az_deg)), "rmax": float(p.r_max_m),
            "bw_az": float(np.deg2rad(p.beamwidth_az_deg)), "bw_el": float(np.deg2rad(p.beamwidth_el_deg)),
        } for p in env.radar_params],
        "classes": ["A · fast", "B · agile", "C · erratic", "D · drift"],
        "frames": env.frames,
    }


if __name__ == "__main__":
    env, summ = run_episode("nominal", "central_greedy", seed=0)
    print(summ)
