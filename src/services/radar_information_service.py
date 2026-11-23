"""Radar Information Analysis Service.

Provides post-hoc operational and scientific analysis of Canonical Episode
Records (v1 and v2) to answer:
"What information can be extracted from radar incoming data, and what can be
predicted/estimated about the target from this information?"

Categorizes information into three strict layers:
  Category A: Direct radar measurements (Operational sensor reports: range,
              azimuth, elevation, Doppler, SNR, measurement covariance R,
              communication transit status).
  Category B: Tracker and fusion estimates (Operational estimates: state x,
              covariance P, confirmation status, coasting duration, class probs).
  Category C: Ground truth (Evaluation only: true target ID, true type, true
              kinematics, true aspect angle, true terrain LOS occlusion).

Strict Rule: Never leak Category C ground truth into operational radar reports.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.core import geometry as geo
from src.core.terrain import build_terrain
from src.services.simulation_service import load_episode_records


def load_records_or_dir(target: Union[str, Path, dict]) -> dict:
    """Safely loads episode records from a directory path or returns dict as-is."""
    if isinstance(target, dict):
        return target
    p = Path(target)
    if p.is_dir():
        return load_episode_records(p)
    raise ValueError(f"Target must be a records dict or directory path, got: {target!r}")


# ---------------------------------------------------------------------------
# Category A: Direct Radar Measurements Analysis
# ---------------------------------------------------------------------------

def summarize_radar_measurements(
    records: dict,
    radar_id: Optional[int] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Summarizes operational radar measurements from the reports table.
    
    Contains ONLY Category A data (what the radar hardware/processor outputs).
    Never inspects ground-truth target_id or annotations.
    """
    reports = records.get("reports", [])
    t_min = t_range[0] if t_range else -np.inf
    t_max = t_range[1] if t_range else np.inf

    filtered_reports = [
        r for r in reports
        if (radar_id is None or r["radar_id"] == radar_id)
        and (t_min <= r["t_s"] <= t_max)
    ]

    total_count = len(filtered_reports)
    if total_count == 0:
        return {
            "radar_id": radar_id,
            "t_range": t_range,
            "total_reports": 0,
            "snr_db": None,
            "range_m": None,
            "azimuth_deg": None,
            "elevation_deg": None,
            "radial_velocity_mps": None,
            "doppler_hz": None,
            "measurement_uncertainty": None,
            "signal_quality": None,
            "detection_probability": None,
        }

    snrs = [r["snr_db"] for r in filtered_reports]
    z_arr = np.array([r["z"] for r in filtered_reports])  # [r, az, el, r_dot]
    qualities = [r["signal_quality"] for r in filtered_reports]
    pds = [r["detection_probability"] for r in filtered_reports]
    dopplers = [r["doppler_hz"] for r in filtered_reports]

    # Measurement sigmas from diagonal of R
    r_covs = [r["R"] for r in filtered_reports]
    sigmas_r = [np.sqrt(max(0.0, float(rc[0, 0]))) for rc in r_covs]
    sigmas_az_deg = [np.rad2deg(np.sqrt(max(0.0, float(rc[1, 1])))) for rc in r_covs]
    sigmas_el_deg = [np.rad2deg(np.sqrt(max(0.0, float(rc[2, 2])))) for rc in r_covs]
    sigmas_rdot = [np.sqrt(max(0.0, float(rc[3, 3]))) for rc in r_covs]

    def _stats(arr: Union[list, np.ndarray]) -> dict:
        a = np.asarray(arr, float)
        return {
            "mean": round(float(np.mean(a)), 3),
            "std": round(float(np.std(a)), 3),
            "median": round(float(np.median(a)), 3),
            "min": round(float(np.min(a)), 3),
            "max": round(float(np.max(a)), 3),
            "p25": round(float(np.percentile(a, 25)), 3),
            "p75": round(float(np.percentile(a, 75)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
        }

    return {
        "radar_id": radar_id,
        "t_range": t_range,
        "total_reports": total_count,
        "snr_db": _stats(snrs),
        "range_m": _stats(z_arr[:, 0]),
        "azimuth_deg": _stats(np.rad2deg(z_arr[:, 1])),
        "elevation_deg": _stats(np.rad2deg(z_arr[:, 2])),
        "radial_velocity_mps": _stats(z_arr[:, 3]),
        "doppler_hz": _stats(dopplers),
        "measurement_uncertainty": {
            "sigma_range_m": _stats(sigmas_r),
            "sigma_azimuth_deg": _stats(sigmas_az_deg),
            "sigma_elevation_deg": _stats(sigmas_el_deg),
            "sigma_radial_velocity_mps": _stats(sigmas_rdot),
        },
        "signal_quality": _stats(qualities),
        "detection_probability": _stats(pds),
    }


def summarize_communication(
    records: dict,
    radar_id: Optional[int] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Summarizes communication transit evidence for radar reports (Schema v2)."""
    comm = records.get("communication")
    if comm is None:
        return {
            "status": "unavailable_in_schema_v1",
            "total_generated": len(records.get("reports", [])),
            "delivered": None,
            "packet_lost": None,
            "delivery_rate_pct": None,
        }

    t_min = t_range[0] if t_range else -np.inf
    t_max = t_range[1] if t_range else np.inf

    filtered_comm = [
        c for c in comm
        if (radar_id is None or c["radar_id"] == radar_id)
        and (t_min <= c["t_s"] <= t_max)
    ]

    total = len(filtered_comm)
    if total == 0:
        return {
            "radar_id": radar_id,
            "total_generated": 0,
            "delivered": 0,
            "packet_lost": 0,
            "delivery_rate_pct": 0.0,
            "status_counts": {},
        }

    counts: Dict[str, int] = {}
    for c in filtered_comm:
        st = c.get("final_status", "unknown")
        counts[st] = counts.get(st, 0) + 1

    delivered = counts.get("delivered", 0)
    lost = counts.get("packet_lost", 0)
    rate = round(delivered / total * 100.0, 2) if total > 0 else 0.0

    return {
        "radar_id": radar_id,
        "total_generated": total,
        "delivered": delivered,
        "packet_lost": lost,
        "delivery_rate_pct": rate,
        "status_counts": counts,
    }


# ---------------------------------------------------------------------------
# Category B: Tracker and Fusion Estimates Analysis
# ---------------------------------------------------------------------------

def summarize_tracker_estimates(
    records: dict,
    source: str = "fused",
    radar_id: Optional[int] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Summarizes operational tracker/fusion state estimates and uncertainties.
    
    Contains Category B data (Kalman filter state x, covariance P, track status).
    Does NOT depend on simulation truth.
    """
    tracks = records.get("tracks", [])
    t_min = t_range[0] if t_range else -np.inf
    t_max = t_range[1] if t_range else np.inf

    filtered_tracks = [
        tr for tr in tracks
        if (tr["source"] == source)
        and (radar_id is None or tr["radar_id"] == radar_id)
        and (t_min <= tr["t_s"] <= t_max)
    ]

    total_track_steps = len(filtered_tracks)
    if total_track_steps == 0:
        return {
            "source": source,
            "radar_id": radar_id,
            "total_track_steps": 0,
            "unique_track_ids": 0,
            "status_breakdown": {},
            "position_uncertainty_m": None,
            "velocity_uncertainty_mps": None,
            "coasting_duration_s": None,
        }

    unique_ids = set(tr["track_id"] for tr in filtered_tracks)
    status_counts: Dict[str, int] = {}
    for tr in filtered_tracks:
        st = tr["status"]
        status_counts[st] = status_counts.get(st, 0) + 1

    # Uncertainties and coasting
    pos_stds = []
    vel_stds = []
    coasting_durations = []

    for tr in filtered_tracks:
        P = tr["P"]
        pos_var = (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0
        vel_var = (P[3, 3] + P[4, 4] + P[5, 5]) / 3.0
        pos_stds.append(float(np.sqrt(max(0.0, pos_var))))
        vel_stds.append(float(np.sqrt(max(0.0, vel_var))))
        coast_s = max(0.0, tr["t_s"] - tr["last_update_t_s"])
        coasting_durations.append(float(coast_s))

    def _stats(arr: list) -> dict:
        a = np.asarray(arr, float)
        return {
            "mean": round(float(np.mean(a)), 3),
            "std": round(float(np.std(a)), 3),
            "median": round(float(np.median(a)), 3),
            "min": round(float(np.min(a)), 3),
            "max": round(float(np.max(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
        }

    # One denominator for every share: all track steps in the window.  Status values are
    # confirmed / confirmed_coasting / tentative / tentative_coasting; "coasting" = no update
    # at that step (the track only predicted), whatever its confirmation state.
    from src.services.episode_analysis_service import track_state_breakdown
    breakdown = track_state_breakdown({"tracks": filtered_tracks}, source=source)
    confirmed_pct = breakdown["confirmed_pct"] or 0.0
    coasting_only = [d for d in coasting_durations if d > 1e-9]

    return {
        "source": source,
        "radar_id": radar_id,
        "total_track_steps": total_track_steps,
        "unique_track_ids": len(unique_ids),
        "confirmed_steps_pct": confirmed_pct,
        "state_breakdown": breakdown,
        "coasting_steps_duration_s": _stats(coasting_only) if coasting_only else None,
        "status_breakdown": status_counts,
        "position_uncertainty_m": _stats(pos_stds),
        "velocity_uncertainty_mps": _stats(vel_stds),
        "coasting_duration_s": _stats(coasting_durations),
    }


# ---------------------------------------------------------------------------
# Category C vs Operational: Visibility and Sensing Cross-Analysis
# ---------------------------------------------------------------------------

def analyze_radar_target_visibility(
    records: dict,
    radar_id: Optional[int] = None,
    target_id: Optional[int] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Evaluates radar-target visibility, line of sight, and sensing pipeline.
    
    Cross-references Category C ground truth with Category A reports and communication
    to determine exactly WHY a detection did or did not materialize.
    
    Clarifies:
      1. Report never formed (Outside range / Outside FOV / Terrain blocked / Unilluminated / Missed Pd)
      2. Report formed but lost in transit (Packet loss)
      3. Report delivered and fused
    """
    metadata = records.get("metadata", {})
    cfg = metadata.get("config_snapshot", {})
    scenario = cfg.get("scenario", {})
    effective_radars = cfg.get("effective_radars", [])
    radars_spec = scenario.get("radars", [])
    
    # Reconstruct terrain if terrain spec is present
    terrain = None
    if "terrain" in scenario and "site" in scenario:
        seed = scenario.get("seed", 0)
        terrain = build_terrain(scenario["site"], scenario["terrain"], seed=seed)

    truth_rows = records.get("truth", [])
    reports = records.get("reports", [])
    annotations = {a["report_id"]: a for a in records.get("annotations", [])}
    comm_map = {c["report_id"]: c for c in records.get("communication", [])}

    t_min = t_range[0] if t_range else -np.inf
    t_max = t_range[1] if t_range else np.inf

    # Index detections by (step_index, radar_id, truth_id)
    det_delivered: Dict[Tuple[int, int, int], bool] = {}
    det_formed: Dict[Tuple[int, int, int], dict] = {}
    for r in reports:
        rid = r["radar_id"]
        step = r["step_index"]
        ann = annotations.get(r["report_id"], {})
        tid = ann.get("truth_id", -1)
        if tid >= 0:
            key = (step, rid, tid)
            det_formed[key] = r
            c = comm_map.get(r["report_id"])
            det_delivered[key] = (c.get("final_status") == "delivered") if c else True

    # Group truth by step
    truth_by_step: Dict[int, Dict[int, dict]] = {}
    for row in truth_rows:
        if t_min <= row["t_s"] <= t_max:
            truth_by_step.setdefault(row["step_index"], {})[row["target_id"]] = row

    radar_indices = [radar_id] if radar_id is not None else list(range(len(radars_spec)))
    
    breakdown = {
        "total_looks_evaluated": 0,
        "out_of_range": 0,
        "out_of_fov": 0,
        "terrain_blocked": 0,
        "los_clear_unilluminated_or_missed": 0,
        "measurement_formed_but_lost": 0,
        "measurement_delivered": 0,
    }

    per_target_stats: Dict[int, dict] = {}

    for step_idx, step_targets in truth_by_step.items():
        for rid in radar_indices:
            r_spec = radars_spec[rid]
            eff = effective_radars[rid] if rid < len(effective_radars) else {}
            r_pos = np.array(r_spec["position"], float)
            r_hd = float(r_spec["heading_deg"])
            r_min_m = float(eff.get("r_min_m", 300.0))
            r_max_m = float(eff.get("r_max_m", 14000.0))
            fov_az = np.deg2rad(float(eff.get("fov_az_deg", 60.0)))
            el_min = np.deg2rad(float(eff.get("el_min_deg", -2.0)))
            el_max = np.deg2rad(float(eff.get("el_max_deg", 60.0)))

            target_ids = [target_id] if target_id is not None else list(step_targets.keys())
            for tid in target_ids:
                if tid not in step_targets:
                    continue
                tgt = step_targets[tid]
                tgt_p = tgt["p"]
                tgt_v = tgt["v"]

                breakdown["total_looks_evaluated"] += 1
                t_stat = per_target_stats.setdefault(tid, {
                    "looks": 0, "out_of_range": 0, "out_of_fov": 0,
                    "terrain_blocked": 0, "los_clear_no_det": 0,
                    "formed_lost": 0, "delivered": 0,
                })
                t_stat["looks"] += 1

                # Check radar-target geometry
                meas = geo.measure(tgt_p, tgt_v, r_pos, r_hd)
                r_dist, az, el, _ = meas

                in_rng = (r_min_m <= r_dist <= r_max_m)
                in_fov = (abs(az) <= fov_az and el_min <= el <= el_max)

                key = (step_idx, rid, tid)
                has_det = (key in det_formed)
                is_deliv = det_delivered.get(key, False)

                if not in_rng:
                    breakdown["out_of_range"] += 1
                    t_stat["out_of_range"] += 1
                elif not in_fov:
                    breakdown["out_of_fov"] += 1
                    t_stat["out_of_fov"] += 1
                else:
                    # In Range and In FOV
                    los_ok = True
                    if terrain is not None:
                        los_ok = bool(terrain.line_of_sight(r_pos, tgt_p))
                    
                    if not los_ok:
                        breakdown["terrain_blocked"] += 1
                        t_stat["terrain_blocked"] += 1
                    else:
                        # LOS clear
                        if has_det:
                            if is_deliv:
                                breakdown["measurement_delivered"] += 1
                                t_stat["delivered"] += 1
                            else:
                                breakdown["measurement_formed_but_lost"] += 1
                                t_stat["formed_lost"] += 1
                        else:
                            breakdown["los_clear_unilluminated_or_missed"] += 1
                            t_stat["los_clear_no_det"] += 1

    return {
        "radar_id": radar_id,
        "target_id": target_id,
        "t_range": t_range,
        "overall_sensing_breakdown": breakdown,
        "per_target_breakdown": per_target_stats,
        "notes": {
            "resimulation_required": False,
            "explanation": (
                "Visibility, FOV, and terrain LOS clearance are reconstructed deterministically "
                "from records and scenario snapshot without resimulation. Exact unilluminated beam "
                "angles and counterfactual Swerling draws require resimulation if sub-dwell detail is needed."
            )
        }
    }


# ---------------------------------------------------------------------------
# Comprehensive Episode and Dataset Analysis
# ---------------------------------------------------------------------------

def analyze_episode(
    target: Union[str, Path, dict],
    radar_id: Optional[int] = None,
    t_range: Optional[Tuple[float, float]] = None,
) -> dict:
    """Runs a complete end-to-end information analysis on a single episode."""
    records = load_records_or_dir(target)
    metadata = records.get("metadata", {})

    meas_summary = summarize_radar_measurements(records, radar_id=radar_id, t_range=t_range)
    comm_summary = summarize_communication(records, radar_id=radar_id, t_range=t_range)
    fused_tracker = summarize_tracker_estimates(records, source="fused", t_range=t_range)
    local_tracker = summarize_tracker_estimates(records, source="local", radar_id=radar_id, t_range=t_range)
    visibility = analyze_radar_target_visibility(records, radar_id=radar_id, t_range=t_range)

    return {
        "episode_id": metadata.get("episode_id", "unknown"),
        "schema_version": metadata.get("record_schema_version", 1),
        "filter": {"radar_id": radar_id, "t_range": t_range},
        "category_a_radar_measurements": meas_summary,
        "category_a_communication": comm_summary,
        "category_b_tracker_fused": fused_tracker,
        "category_b_tracker_local": local_tracker,
        "category_c_visibility_cross_analysis": visibility,
    }


def analyze_dataset(
    dataset_dir: Union[str, Path],
    split: Optional[str] = None,
    max_episodes: Optional[int] = None,
) -> dict:
    """Aggregates radar information analysis over all or selected episodes of a dataset."""
    root = Path(dataset_dir)
    manifest_path = root / "manifest.json"
    splits = [split] if split else ["train", "validation", "test"]

    episodes_found = []
    for sp in splits:
        sp_dir = root / sp
        if sp_dir.is_dir():
            for ep_dir in sorted(sp_dir.glob("ep-*")):
                episodes_found.append((sp, ep_dir))

    if max_episodes:
        episodes_found = episodes_found[:max_episodes]

    total_episodes = len(episodes_found)
    total_reports = 0
    total_delivered = 0
    total_lost = 0
    snrs_all = []
    range_all = []
    rdot_all = []

    fused_confirmed_steps = 0
    fused_total_steps = 0
    pos_uncertainties = []
    coasting_times = []

    for sp, ep_dir in episodes_found:
        records = load_episode_records(ep_dir)
        m = summarize_radar_measurements(records)
        c = summarize_communication(records)
        t = summarize_tracker_estimates(records, source="fused")

        total_reports += m["total_reports"]
        if c.get("delivered") is not None:
            total_delivered += c["delivered"]
            total_lost += c["packet_lost"]

        for r in records.get("reports", []):
            snrs_all.append(r["snr_db"])
            range_all.append(r["z"][0])
            rdot_all.append(r["z"][3])

        fused_total_steps += t["total_track_steps"]
        for tr in records.get("tracks", []):
            if tr["source"] == "fused":
                if tr["confirmed"]:
                    fused_confirmed_steps += 1
                P = tr["P"]
                pos_uncertainties.append(float(np.sqrt(max(0.0, (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0))))
                coast = max(0.0, tr["t_s"] - tr["last_update_t_s"])
                coasting_times.append(float(coast))

    def _stats(arr: list) -> dict:
        if not arr:
            return {}
        a = np.asarray(arr, float)
        return {
            "mean": round(float(np.mean(a)), 3),
            "std": round(float(np.std(a)), 3),
            "median": round(float(np.median(a)), 3),
            "min": round(float(np.min(a)), 3),
            "max": round(float(np.max(a)), 3),
            "p90": round(float(np.percentile(a, 90)), 3),
        }

    return {
        "dataset_path": str(root),
        "episodes_analyzed": total_episodes,
        "splits_included": splits,
        "radar_measurements": {
            "total_reports": total_reports,
            "mean_reports_per_episode": round(total_reports / max(1, total_episodes), 2),
            "snr_db": _stats(snrs_all),
            "range_m": _stats(range_all),
            "radial_velocity_mps": _stats(rdot_all),
        },
        "communication": {
            "total_generated": total_reports,
            "total_delivered": total_delivered,
            "total_packet_lost": total_lost,
            "overall_delivery_rate_pct": round(total_delivered / max(1, total_reports) * 100.0, 2),
        },
        "fused_tracker": {
            "total_fused_steps": fused_total_steps,
            "confirmed_step_ratio_pct": round(fused_confirmed_steps / max(1, fused_total_steps) * 100.0, 2),
            "position_uncertainty_m": _stats(pos_uncertainties),
            "coasting_duration_s": _stats(coasting_times),
        }
    }


# ---------------------------------------------------------------------------
# Time-Series Extraction for Plotting and Streamlit
# ---------------------------------------------------------------------------

def get_detection_timeline(
    records: dict,
    radar_id: Optional[int] = None,
    bin_width_s: float = 1.0,
) -> dict:
    """Returns detection count and delivered count binned over time."""
    reports = records.get("reports", [])
    comm_map = {c["report_id"]: c for c in records.get("communication", [])}
    
    t_max = max([r["t_s"] for r in reports], default=30.0)
    bins = np.arange(0.0, t_max + bin_width_s * 1.5, bin_width_s)
    n_bins = len(bins) - 1

    counts_total = np.zeros(n_bins, dtype=int)
    counts_delivered = np.zeros(n_bins, dtype=int)

    for r in reports:
        if radar_id is not None and r["radar_id"] != radar_id:
            continue
        idx = int(np.clip(r["t_s"] // bin_width_s, 0, n_bins - 1))
        counts_total[idx] += 1
        c = comm_map.get(r["report_id"])
        if c and c.get("final_status") == "delivered":
            counts_delivered[idx] += 1

    return {
        "bin_edges_s": [round(float(b), 2) for b in bins],
        "bin_centers_s": [round(float(b + bin_width_s / 2.0), 2) for b in bins[:-1]],
        "generated_counts": counts_total.tolist(),
        "delivered_counts": counts_delivered.tolist(),
    }


def get_track_uncertainty_timeline(
    records: dict,
    source: str = "fused",
    track_id: Optional[int] = None,
) -> dict:
    """Returns position uncertainty and coasting history over time for tracks."""
    tracks = records.get("tracks", [])
    timeline: Dict[float, list] = {}

    for tr in tracks:
        if tr["source"] != source:
            continue
        if track_id is not None and tr["track_id"] != track_id:
            continue
        P = tr["P"]
        pos_std = float(np.sqrt(max(0.0, (P[0, 0] + P[1, 1] + P[2, 2]) / 3.0)))
        coast_s = max(0.0, tr["t_s"] - tr["last_update_t_s"])
        timeline.setdefault(tr["t_s"], []).append({
            "track_id": tr["track_id"],
            "confirmed": tr["confirmed"],
            "pos_std_m": pos_std,
            "coasting_s": coast_s,
        })

    times = sorted(timeline.keys())
    mean_pos_std = [float(np.mean([item["pos_std_m"] for item in timeline[t]])) for t in times]
    mean_coasting = [float(np.mean([item["coasting_s"] for item in timeline[t]])) for t in times]

    return {
        "time_s": times,
        "mean_position_uncertainty_m": mean_pos_std,
        "mean_coasting_s": mean_coasting,
        "detailed": timeline,
    }
