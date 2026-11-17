"""Controlled experiments comparing Baseline (15s), Extended (30s, 60s),

Radar Heading/Beam Aligned (15s), and Combined Aligned+Extended (30s) across 7 representative pilot episodes.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import copy
import numpy as np

from src.interfaces.dataset_browser import load_dataset_episodes
from src.services.simulation_service import run_episode

index_data = json.loads((ROOT / "data" / "processed" / "pilot_diverse_v1" / "pilot_metadata_index.json").read_text(encoding="utf-8"))
episodes_by_group = {}
for ep in index_data["episodes"]:
    episodes_by_group.setdefault(ep["dataset_name"], []).append(ep)

REPRESENTATIVE_EPISODES = [
    (grp, eps[0]["episode_id"]) for grp, eps in episodes_by_group.items()
]

output_dir = ROOT / "reports" / "step2e_investigation"
output_dir.mkdir(parents=True, exist_ok=True)

all_results = {}

for grp, ep_id in REPRESENTATIVE_EPISODES:
    # Find metadata file
    ep_dir = ROOT / "data" / "processed" / "pilot_diverse_v1" / grp
    meta_files = list(ep_dir.glob(f"**/{ep_id}/metadata.json"))
    if not meta_files:
        print(f"Warning: metadata not found for {ep_id}")
        continue
    
    meta_path = meta_files[0]
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sc_base = meta["config_snapshot"]["scenario"]
    seed = meta.get("episode_seed") or meta.get("dataset_provenance", {}).get("episode_seed", 20262116)
    
    print(f"\n=======================================================")
    print(f"Testing Group: {grp} (Episode: {ep_id})")
    print(f"=======================================================")
    
    grp_results = {}
    
    # ----------------------------------------------------
    # Variation 1: Baseline 15s
    # ----------------------------------------------------
    env1, s1 = run_episode(sc_base, policy_name="central_greedy", seed=seed, classifier=None, record=False, record_episode=True)
    recs1 = env1.episode_records
    ann1 = {a["report_id"]: a for a in recs1.get("annotations", [])}
    truth_dets1 = sum(1 for r in recs1["reports"] if ann1.get(r["report_id"], {}).get("truth_id", -1) >= 0)
    conf_tracks1 = len(set(tr["track_id"] for tr in recs1["tracks"] if tr.get("source") == "fused" and tr.get("confirmed")))
    grp_results["baseline_15s"] = {
        "duration_s": 15.0,
        "steps": len(recs1["transitions"]),
        "truth_detections": truth_dets1,
        "false_alarms": len(recs1["reports"]) - truth_dets1,
        "confirmed_tracks": conf_tracks1,
        "track_coverage_pct": round(float(s1.get("track_coverage", 0.0)) * 100.0, 1),
        "pos_rmse_m": round(float(s1["pos_rmse_m"]), 2) if s1.get("pos_rmse_m") is not None and not np.isnan(s1["pos_rmse_m"]) else None,
    }
    
    # ----------------------------------------------------
    # Variation 2: Extended 30s
    # ----------------------------------------------------
    sc2 = copy.deepcopy(sc_base)
    sc2["duration_s"] = 30.0
    env2, s2 = run_episode(sc2, policy_name="central_greedy", seed=seed, classifier=None, record=False, record_episode=True)
    recs2 = env2.episode_records
    ann2 = {a["report_id"]: a for a in recs2.get("annotations", [])}
    truth_dets2 = sum(1 for r in recs2["reports"] if ann2.get(r["report_id"], {}).get("truth_id", -1) >= 0)
    conf_tracks2 = len(set(tr["track_id"] for tr in recs2["tracks"] if tr.get("source") == "fused" and tr.get("confirmed")))
    grp_results["extended_30s"] = {
        "duration_s": 30.0,
        "steps": len(recs2["transitions"]),
        "truth_detections": truth_dets2,
        "false_alarms": len(recs2["reports"]) - truth_dets2,
        "confirmed_tracks": conf_tracks2,
        "track_coverage_pct": round(float(s2.get("track_coverage", 0.0)) * 100.0, 1),
        "pos_rmse_m": round(float(s2["pos_rmse_m"]), 2) if s2.get("pos_rmse_m") is not None and not np.isnan(s2["pos_rmse_m"]) else None,
    }

    # ----------------------------------------------------
    # Variation 3: Extended 60s
    # ----------------------------------------------------
    sc3 = copy.deepcopy(sc_base)
    sc3["duration_s"] = 60.0
    env3, s3 = run_episode(sc3, policy_name="central_greedy", seed=seed, classifier=None, record=False, record_episode=True)
    recs3 = env3.episode_records
    ann3 = {a["report_id"]: a for a in recs3.get("annotations", [])}
    truth_dets3 = sum(1 for r in recs3["reports"] if ann3.get(r["report_id"], {}).get("truth_id", -1) >= 0)
    conf_tracks3 = len(set(tr["track_id"] for tr in recs3["tracks"] if tr.get("source") == "fused" and tr.get("confirmed")))
    grp_results["extended_60s"] = {
        "duration_s": 60.0,
        "steps": len(recs3["transitions"]),
        "truth_detections": truth_dets3,
        "false_alarms": len(recs3["reports"]) - truth_dets3,
        "confirmed_tracks": conf_tracks3,
        "track_coverage_pct": round(float(s3.get("track_coverage", 0.0)) * 100.0, 1),
        "pos_rmse_m": round(float(s3["pos_rmse_m"]), 2) if s3.get("pos_rmse_m") is not None and not np.isnan(s3["pos_rmse_m"]) else None,
    }

    # ----------------------------------------------------
    # Variation 4: Aligned Beams & Sector 15s (Realistic sector 120 deg + elevation bars covering altitude)
    # ----------------------------------------------------
    sc4 = copy.deepcopy(sc_base)
    # 120 sector focuses 100% looks into the forward valley corridor instead of scanning back into mountain
    sc4.setdefault("radar_defaults", {})
    sc4["radar_defaults"]["search_sector_deg"] = 120.0
    # Include bars at (3.0, 10.0, 20.0) so high aircraft and steep approaches are not missed
    sc4["radar_defaults"]["search_bars_deg"] = [3.0, 10.0, 20.0]
    for r in sc4["radars"]:
        r["search_sector_deg"] = 120.0
        r["search_bars_deg"] = [3.0, 10.0, 20.0]
    env4, s4 = run_episode(sc4, policy_name="central_greedy", seed=seed, classifier=None, record=False, record_episode=True)
    recs4 = env4.episode_records
    ann4 = {a["report_id"]: a for a in recs4.get("annotations", [])}
    truth_dets4 = sum(1 for r in recs4["reports"] if ann4.get(r["report_id"], {}).get("truth_id", -1) >= 0)
    conf_tracks4 = len(set(tr["track_id"] for tr in recs4["tracks"] if tr.get("source") == "fused" and tr.get("confirmed")))
    grp_results["aligned_15s"] = {
        "duration_s": 15.0,
        "steps": len(recs4["transitions"]),
        "truth_detections": truth_dets4,
        "false_alarms": len(recs4["reports"]) - truth_dets4,
        "confirmed_tracks": conf_tracks4,
        "track_coverage_pct": round(float(s4.get("track_coverage", 0.0)) * 100.0, 1),
        "pos_rmse_m": round(float(s4["pos_rmse_m"]), 2) if s4.get("pos_rmse_m") is not None and not np.isnan(s4["pos_rmse_m"]) else None,
    }

    # ----------------------------------------------------
    # Variation 5: Combined Aligned + Extended 30s
    # ----------------------------------------------------
    sc5 = copy.deepcopy(sc4)
    sc5["duration_s"] = 30.0
    env5, s5 = run_episode(sc5, policy_name="central_greedy", seed=seed, classifier=None, record=False, record_episode=True)
    recs5 = env5.episode_records
    ann5 = {a["report_id"]: a for a in recs5.get("annotations", [])}
    truth_dets5 = sum(1 for r in recs5["reports"] if ann5.get(r["report_id"], {}).get("truth_id", -1) >= 0)
    conf_tracks5 = len(set(tr["track_id"] for tr in recs5["tracks"] if tr.get("source") == "fused" and tr.get("confirmed")))
    grp_results["aligned_extended_30s"] = {
        "duration_s": 30.0,
        "steps": len(recs5["transitions"]),
        "truth_detections": truth_dets5,
        "false_alarms": len(recs5["reports"]) - truth_dets5,
        "confirmed_tracks": conf_tracks5,
        "track_coverage_pct": round(float(s5.get("track_coverage", 0.0)) * 100.0, 1),
        "pos_rmse_m": round(float(s5["pos_rmse_m"]), 2) if s5.get("pos_rmse_m") is not None and not np.isnan(s5["pos_rmse_m"]) else None,
    }
    
    all_results[grp] = grp_results
    
    print(f"  1. Baseline 15s:            Dets={truth_dets1:2d}, Conf={conf_tracks1}, Cov={grp_results['baseline_15s']['track_coverage_pct']:5.1f}%, RMSE={grp_results['baseline_15s']['pos_rmse_m']}")
    print(f"  2. Extended 30s:            Dets={truth_dets2:2d}, Conf={conf_tracks2}, Cov={grp_results['extended_30s']['track_coverage_pct']:5.1f}%, RMSE={grp_results['extended_30s']['pos_rmse_m']}")
    print(f"  3. Extended 60s:            Dets={truth_dets3:2d}, Conf={conf_tracks3}, Cov={grp_results['extended_60s']['track_coverage_pct']:5.1f}%, RMSE={grp_results['extended_60s']['pos_rmse_m']}")
    print(f"  4. Aligned 15s (120deg/el): Dets={truth_dets4:2d}, Conf={conf_tracks4}, Cov={grp_results['aligned_15s']['track_coverage_pct']:5.1f}%, RMSE={grp_results['aligned_15s']['pos_rmse_m']}")
    print(f"  5. Aligned+Ext 30s:         Dets={truth_dets5:2d}, Conf={conf_tracks5}, Cov={grp_results['aligned_extended_30s']['track_coverage_pct']:5.1f}%, RMSE={grp_results['aligned_extended_30s']['pos_rmse_m']}")

# Save experiment results to JSON
res_path = output_dir / "controlled_experiments_results.json"
res_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
print(f"\nAll experiment results saved to: {res_path}")
