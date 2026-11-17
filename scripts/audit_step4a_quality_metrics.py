"""Comprehensive Quality & Radar Tracking Metrics Audit across all 1,000 Episodes for Step 4A."""
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.simulation_service import load_episode_records
from src.services.diverse_dataset_service import DATASET_GROUPS

DATASET_ROOT = ROOT / "data" / "processed" / "diverse_1000_v1"

def main():
    print("=" * 80)
    print("STEP 4A AUDIT 2: FULL 1000-EPISODE RADAR & TRACKING QUALITY AUDIT")
    print("=" * 80)

    index_path = DATASET_ROOT / "diverse_1000_metadata_index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    episodes = index_data["episodes"]

    # Containers for aggregations
    group_stats = {g: defaultdict(list) for g in DATASET_GROUPS}
    diff_stats = {"easy": defaultdict(list), "medium": defaultdict(list), "hard": defaultdict(list)}
    
    # We sample episodes for deep EKF RMSE calculation to keep runtime fast and accurate
    # 1 out of every 5 episodes (200 episodes sampled across all 7 groups and 3 difficulties)
    rmse_per_group = {g: [] for g in DATASET_GROUPS}
    rmse_per_diff = {"easy": [], "medium": [], "hard": []}

    print("Auditing 1,000 episodes across 7 groups...")
    
    for idx, ep in enumerate(episodes):
        grp = ep["dataset_name"]
        diff = ep["difficulty_level"]
        q = ep["quality_summary"]
        
        tc = ep["target_count"]
        cov = q["track_coverage_pct"]
        los = q["los_clear_pct"]
        zero_det = q["zero_detection_targets"]
        truth_rows = q["truth_rows"]
        rep_rows = q["report_rows"]
        deliv = q["delivered_reports"]
        lost = q["lost_reports"]
        fa = q["false_alarms"]
        conf_tr = q["confirmed_tracks"]
        tent_tr = q["tentative_tracks"]
        
        # Accumulate in group_stats
        group_stats[grp]["target_count"].append(tc)
        group_stats[grp]["track_coverage"].append(cov)
        group_stats[grp]["los_clear"].append(los)
        group_stats[grp]["zero_det"].append(zero_det)
        group_stats[grp]["truth_rows"].append(truth_rows)
        group_stats[grp]["reports"].append(rep_rows)
        group_stats[grp]["delivered"].append(deliv)
        group_stats[grp]["lost"].append(lost)
        group_stats[grp]["false_alarms"].append(fa)
        group_stats[grp]["confirmed_tracks"].append(conf_tr)
        group_stats[grp]["tentative_tracks"].append(tent_tr)

        # Accumulate in diff_stats
        diff_stats[diff]["target_count"].append(tc)
        diff_stats[diff]["track_coverage"].append(cov)
        diff_stats[diff]["los_clear"].append(los)
        diff_stats[diff]["zero_det"].append(zero_det)
        diff_stats[diff]["false_alarms"].append(fa)

        # Compute truth-associated RMSE on sample (every 5th episode)
        if idx % 5 == 0:
            ep_dir = DATASET_ROOT / grp / ep["split"] / ep["episode_id"]
            if ep_dir.exists():
                recs = load_episode_records(ep_dir)
                truth = recs["truth"]
                tracks = recs["tracks"]
                conf_tracks = [t for t in tracks if t.get("confirmed")]
                truth_by_step = {}
                for row in truth:
                    truth_by_step.setdefault(row["step_index"], []).append(row["p"])
                
                pos_errs = []
                for tr in conf_tracks:
                    step = tr["step_index"]
                    p_est = tr["x"][:3]
                    if step in truth_by_step:
                        best_d = min(math.dist(p_est, p_t) for p_t in truth_by_step[step])
                        if best_d < 2500.0:
                            pos_errs.append(best_d)
                if pos_errs:
                    ep_rmse = math.sqrt(sum(e**2 for e in pos_errs) / len(pos_errs))
                    rmse_per_group[grp].append(ep_rmse)
                    rmse_per_diff[diff].append(ep_rmse)

    # Compile group summary
    group_summary = {}
    print("\n" + "=" * 80)
    print("GROUP-BY-GROUP METRICS SUMMARY (1,000 EPISODES)")
    print("=" * 80)
    print(f"{'Group Name':32s} | {'Eps':4s} | {'Tgts':4s} | {'Zero-Det %':10s} | {'Mean Cov %':10s} | {'Mean LOS %':10s} | {'RMSE (m)':8s} | {'Deliv %':8s}")
    print("-" * 105)

    for grp in DATASET_GROUPS:
        st = group_stats[grp]
        n_eps = len(st["track_coverage"])
        tot_tgts = sum(st["target_count"])
        tot_zero = sum(st["zero_det"])
        zero_pct = (tot_zero / tot_tgts * 100.0) if tot_tgts > 0 else 0.0
        mean_cov = float(np.mean(st["track_coverage"]))
        mean_los = float(np.mean(st["los_clear"]))
        tot_deliv = sum(st["delivered"])
        tot_reps = sum(st["reports"])
        deliv_pct = (tot_deliv / max(1, tot_reps) * 100.0)
        mean_fa = float(np.mean(st["false_alarms"]))
        mean_conf = float(np.mean(st["confirmed_tracks"]))
        mean_tent = float(np.mean(st["tentative_tracks"]))
        rmse_val = round(float(np.mean(rmse_per_group[grp])), 2) if rmse_per_group[grp] else None
        
        group_summary[grp] = {
            "episode_count": n_eps,
            "total_targets": tot_tgts,
            "zero_detection_targets": tot_zero,
            "zero_detection_pct": round(zero_pct, 1),
            "mean_track_coverage_pct": round(mean_cov, 1),
            "min_track_coverage_pct": round(float(np.min(st["track_coverage"])), 1),
            "max_track_coverage_pct": round(float(np.max(st["track_coverage"])), 1),
            "mean_los_pct": round(mean_los, 1),
            "min_los_pct": round(float(np.min(st["los_clear"])), 1),
            "max_los_pct": round(float(np.max(st["los_clear"])), 1),
            "total_reports": tot_reps,
            "delivered_reports": tot_deliv,
            "delivered_pct": round(deliv_pct, 1),
            "total_false_alarms": sum(st["false_alarms"]),
            "mean_false_alarms_per_ep": round(mean_fa, 1),
            "mean_confirmed_tracks_per_ep": round(mean_conf, 1),
            "mean_tentative_tracks_per_ep": round(mean_tent, 1),
            "mean_position_rmse_m": rmse_val,
        }
        rmse_str = f"{rmse_val:.2f} m" if rmse_val is not None else "N/A"
        print(f"{grp:32s} | {n_eps:4d} | {tot_tgts:4d} | {tot_zero:3d}/{tot_tgts:3d} ({zero_pct:4.1f}%) | {mean_cov:9.1f}% | {mean_los:9.1f}% | {rmse_str:8s} | {deliv_pct:7.1f}%")

    # Compile difficulty summary
    diff_summary = {}
    print("\n" + "=" * 80)
    print("DIFFICULTY TIER SUMMARY (300 Easy, 450 Medium, 250 Hard)")
    print("=" * 80)
    print(f"{'Difficulty Tier':16s} | {'Eps':4s} | {'Tgts':4s} | {'Zero-Det %':10s} | {'Mean Cov %':10s} | {'Mean LOS %':10s} | {'RMSE (m)':8s} | {'Mean FA/ep':10s}")
    print("-" * 90)

    for diff in ("easy", "medium", "hard"):
        st = diff_stats[diff]
        n_eps = len(st["track_coverage"])
        tot_tgts = sum(st["target_count"])
        tot_zero = sum(st["zero_det"])
        zero_pct = (tot_zero / tot_tgts * 100.0) if tot_tgts > 0 else 0.0
        mean_cov = float(np.mean(st["track_coverage"]))
        mean_los = float(np.mean(st["los_clear"]))
        mean_fa = float(np.mean(st["false_alarms"]))
        rmse_val = round(float(np.mean(rmse_per_diff[diff])), 2) if rmse_per_diff[diff] else None
        
        diff_summary[diff] = {
            "episode_count": n_eps,
            "total_targets": tot_tgts,
            "zero_detection_targets": tot_zero,
            "zero_detection_pct": round(zero_pct, 1),
            "mean_track_coverage_pct": round(mean_cov, 1),
            "mean_los_pct": round(mean_los, 1),
            "mean_false_alarms_per_ep": round(mean_fa, 1),
            "mean_position_rmse_m": rmse_val,
        }
        rmse_str = f"{rmse_val:.2f} m" if rmse_val is not None else "N/A"
        print(f"{diff.capitalize():16s} | {n_eps:4d} | {tot_tgts:4d} | {tot_zero:3d}/{tot_tgts:3d} ({zero_pct:4.1f}%) | {mean_cov:9.1f}% | {mean_los:9.1f}% | {rmse_str:8s} | {mean_fa:9.1f}")

    # Output detailed JSON report
    report_data = {
        "report_name": "step4a_final_quality_audit",
        "dataset_name": "diverse_1000_v1",
        "total_episodes": len(episodes),
        "group_summary": group_summary,
        "difficulty_summary": diff_summary,
    }
    
    out_file = ROOT / "reports" / "step4a_final_quality_audit.json"
    out_file.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print(f"\nSaved detailed audit report to {out_file}")

if __name__ == "__main__":
    main()
