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
from src.core.targets import CLASS_SHORT
from src.core.terrain import build_terrain
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
    {"name": "Bursa · Uludag · TR", "lat": 40.14, "lon": 29.10},
    {"name": "Kayseri · Erciyes · TR", "lat": 38.58, "lon": 35.44},
    {"name": "Ankara · TR", "lat": 39.93, "lon": 32.85},
    {"name": "Izmir · TR", "lat": 38.42, "lon": 27.14},
    {"name": "Cappadocia · Goreme · TR", "lat": 38.64, "lon": 34.83},
    {"name": "Van · TR", "lat": 38.50, "lon": 43.38},
    {"name": "Zermatt · Alps · CH", "lat": 46.02, "lon": 7.75},
    {"name": "Grand Canyon · US", "lat": 36.10, "lon": -112.11},
]


def format_target_label(target_id: int, label: int, target_type: str | None = None, ground_truth_tag: bool = True) -> str:
    """Format human-readable target label using explicit target_type with legacy fallback.

    If target_type is provided:
      - 'aircraft_like': T{id} — Aircraft-like
      - 'uav_like': T{id} — UAV-like (Agile) if label==1 else T{id} — UAV-like (Drift) if label==3 else T{id} — UAV-like
      - 'missile_like': T{id} — Missile-like
      - other: T{id} — {target_type}
    If target_type is absent (legacy dataset):
      - Fallback to CLASS_SHORT[label]: truth {CLASS_SHORT[label]}
    """
    tag = " [Ground Truth]" if ground_truth_tag else ""
    if target_type and "/" in str(target_type):
        # UAV / aircraft / missile classes: "class/variant"
        from src.core.target_classes import TARGET_CLASSES
        cls, variant = str(target_type).split("/", 1)
        name = TARGET_CLASSES[cls]["name"] if cls in TARGET_CLASSES else cls
        return f"T{target_id} — {name} ({variant.replace('_', ' ')}){tag}"
    if target_type:
        tt = str(target_type).lower().strip()
        if "aircraft" in tt:
            type_str = "Aircraft-like"
        elif "missile" in tt:
            type_str = "Missile-like"
        elif "uav" in tt or "drone" in tt:
            if label == 1:
                type_str = "UAV-like (Agile)"
            elif label == 3:
                type_str = "UAV-like (Drift)"
            else:
                type_str = "UAV-like"
        else:
            type_str = target_type.replace("_", "-").title()
        return f"T{target_id} — {type_str}{tag}"
    legacy_name = CLASS_SHORT[label] if 0 <= label < len(CLASS_SHORT) else f"Class {label}"
    return f"truth {legacy_name}"


def viewer_payload(env: RadarNetEnv) -> dict:
    """Static scene description + frames, consumed by the 3-D viewer."""
    sc = env.sc
    target_labels = {}
    target_type_map = {}
    for s in env.targets:
        tt = getattr(s, "target_type", None)
        if not tt and s.id < len(sc.get("targets", [])):
            tt = sc["targets"][s.id].get("target_type")
        target_type_map[s.id] = (s.label, tt)
        target_labels[s.id] = format_target_label(s.id, s.label, tt)

    # Attach display_name to targets in env.frames
    frames = []
    for fr in env.frames:
        fr_copy = dict(fr)
        fr_copy["targets"] = [
            {
                **t,
                "target_type": target_type_map.get(t["id"], (t.get("label", 0), None))[1],
                "display_name": target_labels.get(t["id"], f"truth {CLASS_SHORT[t.get('label', 0)]}"),
            }
            for t in fr.get("targets", [])
        ]
        frames.append(fr_copy)

    target_legend = [target_labels[tid] for tid in sorted(target_labels.keys())] if target_labels else ["A · fast", "B · agile", "C · erratic", "D · drift"]

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
        "target_labels": target_labels,
        "target_legend": target_legend,
        "frames": frames,
    }


def records_to_viewer_payload(records: dict) -> dict:
    """Builds a viewer payload from Canonical Episode Records v2 for replay."""
    metadata = records.get("metadata", {})
    cfg = metadata.get("config_snapshot", {})
    sc = cfg.get("scenario", {})
    eff_radars = cfg.get("effective_radars", [])

    site = sc.get("site", {"name": "Utah Valley · US", "lat": 40.30, "lon": -111.70})
    seed = sc.get("seed", 0)
    terrain_spec = sc.get("terrain", {"kind": "procedural", "roughness": 0.5})
    terrain = build_terrain(site, terrain_spec, seed=seed)

    dt = float(sc.get("dt", 0.5))
    area_radius = float(sc.get("area_radius_m", 10000.0))

    radars_payload = []
    radars_spec = sc.get("radars", [])
    for i, r in enumerate(radars_spec):
        eff = eff_radars[i] if i < len(eff_radars) else {}
        hd_deg = float(r.get("heading_deg", 0.0))
        fov_deg = float(eff.get("fov_az_deg", r.get("fov_az_deg", 60.0)))
        rmax = float(eff.get("r_max_m", r.get("r_max_m", 14000.0)))
        bw_az_deg = float(eff.get("beamwidth_az_deg", 3.0))
        bw_el_deg = float(eff.get("beamwidth_el_deg", 6.0))
        radars_payload.append({
            "name": r.get("name", f"R{i+1}"),
            "pos": [float(x) for x in r["position"]],
            "heading": float(np.deg2rad(hd_deg)),
            "fov": float(np.deg2rad(fov_deg)),
            "rmax": rmax,
            "bw_az": float(np.deg2rad(bw_az_deg)),
            "bw_el": float(np.deg2rad(bw_el_deg)),
        })

    truth_by_step = {}
    for row in records.get("truth", []):
        truth_by_step.setdefault(row["step_index"], []).append(row)

    reports_by_step = {}
    ann_map = {a["report_id"]: a for a in records.get("annotations", [])}
    for rep in records.get("reports", []):
        reports_by_step.setdefault(rep["step_index"], []).append(rep)

    fused_tracks_by_step = {}
    for tr in records.get("tracks", []):
        if tr["source"] == "fused":
            fused_tracks_by_step.setdefault(tr["step_index"], []).append(tr)

    trans_by_step = {}
    for tr in records.get("transitions", []):
        trans_by_step[tr["step_index"]] = tr

    # Extract target types from metadata / config_snapshot
    target_types_meta = metadata.get("target_types", [])
    cfg_targets = sc.get("targets", [])
    target_type_map = {}
    for row in records.get("truth", []):
        tid = row["target_id"]
        if tid not in target_type_map:
            tt = None
            if tid < len(target_types_meta) and target_types_meta[tid]:
                tt = target_types_meta[tid]
            elif tid < len(cfg_targets) and cfg_targets[tid].get("target_type"):
                tt = cfg_targets[tid]["target_type"]
            target_type_map[tid] = (row["class_label"], tt)

    target_labels = {}
    for tid, (lbl, tt) in target_type_map.items():
        target_labels[tid] = format_target_label(tid, lbl, tt)

    max_steps = max(truth_by_step.keys()) if truth_by_step else 60
    frames = []

    for s_idx in range(max_steps + 1):
        step_truth = truth_by_step.get(s_idx, [])
        step_reports = reports_by_step.get(s_idx, [])
        step_tracks = fused_tracks_by_step.get(s_idx, [])
        step_trans = trans_by_step.get(s_idx, {})
        actions = step_trans.get("requested_actions", [1] * len(radars_spec))

        radars_frame = []
        for i, r_spec in enumerate(radars_spec):
            act = int(actions[i]) if i < len(actions) else 1
            mode = 0 if act == 0 else (1 if act == 1 else 2)
            r_pos = np.array(r_spec["position"], float)
            los_list = []
            for tgt_row in step_truth:
                los_ok = terrain.line_of_sight(r_pos, tgt_row["p"])
                los_list.append(1 if los_ok else 2)

            radars_frame.append({
                "len": None,
                "mode": mode,
                "online": True,
                "beams": [],
                "energy": 1.0,
                "heat": 1.0,
                "los": los_list,
            })

        tracks_frame = []
        for tr in step_tracks:
            pos_std = float(np.sqrt(max(0.0, np.trace(tr["P"][:3, :3]) / 3.0)))
            tracks_frame.append({
                "id": tr["track_id"],
                "p": [round(float(x), 1) for x in tr["x"][:3]],
                "v": [round(float(x), 1) for x in tr["x"][3:]],
                "sig": round(pos_std, 1),
                "conf": bool(tr["confirmed"]),
                "cls": -1,
                "pc": 0.0,
            })

        dets_frame = []
        for rep in step_reports:
            ann = ann_map.get(rep["report_id"], {})
            fa = 1 if ann.get("is_false_alarm", False) else 0
            dets_frame.append([round(float(x), 1) for x in rep["world_xyz"]] + [fa])

        targets_frame = [
            {
                "id": tgt_row["target_id"],
                "label": tgt_row["class_label"],
                "target_type": target_type_map.get(tgt_row["target_id"], (tgt_row["class_label"], None))[1],
                "display_name": target_labels.get(tgt_row["target_id"], f"truth {CLASS_SHORT[tgt_row['class_label']]}"),
                "p": [round(float(x), 1) for x in tgt_row["p"]],
                "v": [round(float(x), 1) for x in tgt_row["v"]],
            }
            for tgt_row in step_truth
        ]

        frames.append({
            "t": round(s_idx * dt, 2),
            "radars": radars_frame,
            "targets": targets_frame,
            "dets": dets_frame,
            "tracks": tracks_frame,
        })

    target_legend = [target_labels[tid] for tid in sorted(target_labels.keys())] if target_labels else ["A · fast", "B · agile", "C · erratic", "D · drift"]

    return {
        "site": site,
        "sites": [site] + [s for s in MAP_SITES if s["name"] != site["name"]],
        "terrain": terrain.to_payload(),
        "dt": dt,
        "area_radius": area_radius,
        "radars": radars_payload,
        "classes": ["A · fast", "B · agile", "C · erratic", "D · drift"],
        "target_labels": target_labels,
        "target_legend": target_legend,
        "frames": frames,
        "replay_metadata": {
            "source": "canonical_records_v2",
            "unrecorded_notice": "Beam ray traces are omitted during replay because exact micro-dwell angles are not stored in records.",
        }
    }


if __name__ == "__main__":
    env, summ = run_episode("nominal", "central_greedy", seed=0)
    print(summ)
