"""Controlled experiments for Step 2G: Aircraft and Missile geometry and radar configuration optimization."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import copy
import math
import numpy as np

from src.core.env import _cached_terrain
from src.core.scenario import validate_scenario
from src.services.simulation_service import run_episode
from src.services.diverse_dataset_service import compute_physical_quality_metrics

# Reusable placement definitions
PLACEMENTS = {
    "valley_entrance_exit_overlook": [
        {"name": "R1", "position": [0, -4200, 15], "heading_deg": 90},
        {"name": "R2", "position": [0, 4200, 15], "heading_deg": -90},
        {"name": "R3", "position": [2200, 0, 22], "heading_deg": 180},
    ],
    "cross_valley_triangulation": [
        {"name": "R1", "position": [-2000, -2500, 20], "heading_deg": 45},
        {"name": "R2", "position": [2000, -2500, 20], "heading_deg": 135},
        {"name": "R3", "position": [0, 3800, 15], "heading_deg": -90},
    ],
    "lateral_flank_staggered": [
        {"name": "R1", "position": [-1800, -2800, 18], "heading_deg": 45},
        {"name": "R2", "position": [2000, 200, 22], "heading_deg": 160},
        {"name": "R3", "position": [-1600, 3200, 18], "heading_deg": -45},
    ],
}

def run_test_sim(name, terrain_dict, radars_spec, target_spec, duration_s=30.0, seed=20261116):
    radars = copy.deepcopy(radars_spec)
    targets = [copy.deepcopy(target_spec)]
    
    sc = {
        "name": name,
        "site": {"lat": 40.3, "lon": -111.7},
        "duration_s": duration_s,
        "dt": 0.5,
        "terrain": terrain_dict,
        "radar_defaults": {
            "search_sector_deg": radars[0].get("search_sector_deg", 120.0),
            "search_bars_deg": radars[0].get("search_bars_deg", [3.0, 10.0]),
        },
        "radars": radars,
        "targets": targets,
    }
    vsc = validate_scenario(sc)
    env, summ = run_episode(vsc, policy_name="central_greedy", seed=seed, record=False, record_episode=True)
    recs = env.episode_records
    
    annotations = recs.get("annotations", [])
    ann_by_rep = {a["report_id"]: a for a in annotations}
    truth_dets = [r for r in recs["reports"] if ann_by_rep.get(r["report_id"], {}).get("truth_id", -1) >= 0]
    false_alarms = [r for r in recs["reports"] if ann_by_rep.get(r["report_id"], {}).get("truth_id", -1) < 0]
    
    conf_tracks = [t for t in recs["tracks"] if t.get("confirmed", False)]
    conf_ids = set(t["track_id"] for t in conf_tracks)
    conf_steps = set(t["step_index"] for t in conf_tracks)
    
    coverage = summ.get("track_coverage", 0.0) * 100.0
    rmse = summ.get("pos_rmse_m")
    
    first_det_step = truth_dets[0]["step_index"] if truth_dets else None
    first_det_time = (first_det_step * 0.5) if first_det_step is not None else None
    
    return {
        "name": name,
        "truth_dets": len(truth_dets),
        "false_alarms": len(false_alarms),
        "confirmed_tracks": len(conf_ids),
        "confirmed_steps": len(conf_steps),
        "coverage_pct": coverage,
        "rmse_m": rmse,
        "first_det_s": first_det_time,
    }

terrain_canyon = {
    "source": "procedural",
    "resolution": 256,
    "family": "valley_linear_canyon",
    "texture_family": "rocky_mountain",
    "obstacle_profile": "clear",
    "valley_width_m": 3400.0,
    "relief_height_m": 800.0,
    "seed": 20261116 + 101,
}

terrain_pass = {
    "source": "procedural",
    "resolution": 256,
    "family": "valley_sinuous_pass",
    "texture_family": "snowy_highlands",
    "obstacle_profile": "sparse",
    "valley_width_m": 2800.0,
    "relief_height_m": 1100.0,
    "seed": 20261116 + 102,
}

# -------------------------------------------------------------
# EXPERIMENT 1: SINGLE AIRCRAFT
# -------------------------------------------------------------
print("==================================================================")
print("EXPERIMENT 1: SINGLE AIRCRAFT CONTROLLED CONFIGURATION TESTING")
print("==================================================================")

# Target Aircraft specs (150 m/s, high altitude)
target_aircraft_canyon = {
    "label": 0,
    "target_type": "aircraft_like",
    "position": [-50.0, -3500.0, 1850.0],
    "velocity": [0.0, 150.0, 0.0],
    "motion": {"model": "CV"},
}

aircraft_experiments = [
    {
        "name": "AC-1: Baseline v2 (Entrance/Exit, 3 bars [5, 16, 28], 120°)",
        "terrain": terrain_canyon,
        "placement": "valley_entrance_exit_overlook",
        "bars": [5.0, 16.0, 28.0],
        "sector": 120.0,
        "target": target_aircraft_canyon,
        "duration": 30.0,
    },
    {
        "name": "AC-2: Higher Elevation Bars [8, 22, 38], 120° sector",
        "terrain": terrain_canyon,
        "placement": "valley_entrance_exit_overlook",
        "bars": [8.0, 22.0, 38.0],
        "sector": 120.0,
        "target": target_aircraft_canyon,
        "duration": 30.0,
    },
    {
        "name": "AC-3: Cross-Valley Triangulation (Eliminates Zenith Blind Zone)",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [5.0, 16.0, 28.0],
        "sector": 120.0,
        "target": target_aircraft_canyon,
        "duration": 30.0,
    },
    {
        "name": "AC-4: Cross-Valley + Faster Sweep (90° sector, bars [8, 24])",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [8.0, 24.0],
        "sector": 90.0,
        "target": target_aircraft_canyon,
        "duration": 30.0,
    },
    {
        "name": "AC-5: Combined Optimal (Cross-Valley, 90° sector, bars [6, 18, 32])",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [6.0, 18.0, 32.0],
        "sector": 90.0,
        "target": target_aircraft_canyon,
        "duration": 30.0,
    },
]

ac_results = []
for exp in aircraft_experiments:
    radars = copy.deepcopy(PLACEMENTS[exp["placement"]])
    for r in radars:
        r["search_bars_deg"] = exp["bars"]
        r["search_sector_deg"] = exp["sector"]
    res = run_test_sim(exp["name"], exp["terrain"], radars, exp["target"], duration_s=exp["duration"])
    ac_results.append(res)
    rmse_str = f"{res['rmse_m']:.2f}m" if res['rmse_m'] is not None else "N/A"
    print(f"[{res['name']}] -> Truth Dets: {res['truth_dets']}, Confirmed: {res['confirmed_tracks']}, Coverage: {res['coverage_pct']:.1f}%, RMSE: {rmse_str}, First Det: {res['first_det_s']}s")

# -------------------------------------------------------------
# EXPERIMENT 2: SINGLE MISSILE
# -------------------------------------------------------------
print("\n==================================================================")
print("EXPERIMENT 2: SINGLE MISSILE CONTROLLED CONFIGURATION TESTING")
print("==================================================================")

# Target Missile specs (300 m/s, low altitude)
target_missile_canyon = {
    "label": 2,
    "target_type": "missile_like",
    "position": [0.0, -2600.0, 450.0],
    "velocity": [0.0, 300.0, 0.0],
    "motion": {"model": "CV"},
}

missile_experiments = [
    {
        "name": "MS-1: Baseline v2 (Entrance/Exit, 2 bars [3, 8], 120°)",
        "terrain": terrain_canyon,
        "placement": "valley_entrance_exit_overlook",
        "bars": [3.0, 8.0],
        "sector": 120.0,
        "target": target_missile_canyon,
        "duration": 30.0,
    },
    {
        "name": "MS-2: Cross-Valley Triangulation (Both radars view corridor)",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [3.0, 8.0],
        "sector": 120.0,
        "target": target_missile_canyon,
        "duration": 30.0,
    },
    {
        "name": "MS-3: Cross-Valley + Faster Sweep (90° sector, bars [3, 8])",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [3.0, 8.0],
        "sector": 90.0,
        "target": target_missile_canyon,
        "duration": 30.0,
    },
    {
        "name": "MS-4: Cross-Valley + Low-Elevation Fast Raster (90° sector, 1 bar [4.5])",
        "terrain": terrain_canyon,
        "placement": "cross_valley_triangulation",
        "bars": [4.5],
        "sector": 90.0,
        "target": target_missile_canyon,
        "duration": 30.0,
    },
    {
        "name": "MS-5: Staggered Flank (Inward cross-look, 90° sector, 2 bars [3, 8])",
        "terrain": terrain_canyon,
        "placement": "lateral_flank_staggered",
        "bars": [3.0, 8.0],
        "sector": 90.0,
        "target": target_missile_canyon,
        "duration": 30.0,
    },
]

ms_results = []
for exp in missile_experiments:
    radars = copy.deepcopy(PLACEMENTS[exp["placement"]])
    for r in radars:
        r["search_bars_deg"] = exp["bars"]
        r["search_sector_deg"] = exp["sector"]
    res = run_test_sim(exp["name"], exp["terrain"], radars, exp["target"], duration_s=exp["duration"])
    ms_results.append(res)
    rmse_str = f"{res['rmse_m']:.2f}m" if res['rmse_m'] is not None else "N/A"
    print(f"[{res['name']}] -> Truth Dets: {res['truth_dets']}, Confirmed: {res['confirmed_tracks']}, Coverage: {res['coverage_pct']:.1f}%, RMSE: {rmse_str}, First Det: {res['first_det_s']}s")

# Save results to a json file
res_summary = {
    "aircraft_experiments": ac_results,
    "missile_experiments": ms_results,
}
Path("reports/experiment_aircraft_missile_results.json").write_text(json.dumps(res_summary, indent=2), encoding="utf-8")
print("\nResults successfully saved to reports/experiment_aircraft_missile_results.json")
