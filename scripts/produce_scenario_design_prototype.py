"""Production script for Step 4D: Scenario Design Prototype (12 Representative Episodes).

Generates 12 purposeful, operationally coherent episodes covering all 6 templates:
1. valley_infiltration (EP-0001, EP-0002)
2. high_altitude_crossing (EP-0003, EP-0004)
3. low_altitude_missile_run (EP-0005, EP-0006)
4. coordinated_uav_approach (EP-0007, EP-0008)
5. mixed_target_incursion (EP-0009, EP-0010)
6. terrain_masking_reacquisition (EP-0011, EP-0012)

Output: data/processed/scenario_design_prototype_v1/
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.env import _cached_terrain
from src.core.scenario import scenario_snapshot, validate_scenario
from src.services.diverse_dataset_service import (
    _digest,
    compute_physical_quality_metrics,
    format_display_name,
    terrain_group_id,
    SCENARIO_RADAR_CONFIGS,
    RADAR_PLACEMENT_PROFILES_V2,
)
from src.services.synthetic_dataset_service import generate_dataset

PROTOTYPE_DIR = ROOT / "data" / "processed" / "scenario_design_prototype_v1"
PROTOTYPE_DIR.mkdir(parents=True, exist_ok=True)

PROTOTYPE_EPISODES_SPEC = [
    # -------------------------------------------------------------------------
    # 1. valley_infiltration
    # -------------------------------------------------------------------------
    {
        "template": "valley_infiltration",
        "dataset_name": "dataset_single_uav_agile_v1",
        "display_name": "EP-0001 · Forest Winding Pass · UAV Agile · 3 Radars (Valley Infiltration)",
        "terrain_family": "valley_sinuous_pass",
        "texture_family": "green_forest_valley",
        "obstacle_profile": "sparse",
        "difficulty_level": "easy",
        "radar_placement_profile": "valley_entrance_exit_overlook",
        "radar_category": "low_altitude_uav",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["uav_agile"],
        "target_types": ["uav_like"],
        "route_family": "sinuous_pass_transit",
        "operational_objective": "Stealth low-altitude UAV penetration along winding valley corridor with entrance, exit, and shoulder overlook surveillance.",
        "terrain_params": {
            "valley_width_m": 3200.0,
            "relief_height_m": 880.0,
            "sinuosity_amplitude_m": 1500.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (-400.0, -4600.0, 160.0),  # (x, y, agl)
                "velocity": (12.0, 64.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.04},
            }
        ],
    },
    {
        "template": "valley_infiltration",
        "dataset_name": "dataset_single_uav_drift_v1",
        "display_name": "EP-0002 · Mixed Canyon · UAV Drift · 3 Radars (Valley Infiltration)",
        "terrain_family": "valley_linear_canyon",
        "texture_family": "mixed_rock_vegetation",
        "obstacle_profile": "sparse",
        "difficulty_level": "easy",
        "radar_placement_profile": "lateral_flank_staggered",
        "radar_category": "slow_drift_uav",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["uav_drift"],
        "target_types": ["uav_like"],
        "route_family": "corridor_south_to_north",
        "operational_objective": "Persistent low-speed drift sensor platform slipping through staggered lateral radar flanks along canyon floor.",
        "terrain_params": {
            "valley_width_m": 3400.0,
            "relief_height_m": 880.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 3,
                "target_type": "uav_like",
                "rel_pos": (0.0, -4400.0, 180.0),
                "velocity": (1.0, 40.0, 0.0),
                "motion": {"model": "CV"},
            }
        ],
    },

    # -------------------------------------------------------------------------
    # 2. high_altitude_crossing
    # -------------------------------------------------------------------------
    {
        "template": "high_altitude_crossing",
        "dataset_name": "dataset_single_aircraft_v1",
        "display_name": "EP-0003 · Rocky Highlands · Aircraft · 3 Radars (High-Altitude Crossing)",
        "terrain_family": "basin_highlands",
        "texture_family": "rocky_mountain",
        "obstacle_profile": "clear",
        "difficulty_level": "easy",
        "radar_placement_profile": "cross_valley_triangulation",
        "radar_category": "high_altitude_aircraft",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["aircraft_cruise"],
        "target_types": ["aircraft_like"],
        "route_family": "corridor_south_to_north",
        "operational_objective": "High-altitude aircraft transit (2400m MSL) across alpine plateau; continuous multi-radar triangulation using high-elevation bars.",
        "terrain_params": {
            "basin_radius_m": 4500.0,
            "relief_height_m": 920.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (150.0, -5200.0, 2400.0),  # high altitude MSL
                "velocity": (4.0, 180.0, 0.0),
                "motion": {"model": "CV"},
            }
        ],
    },
    {
        "template": "high_altitude_crossing",
        "dataset_name": "dataset_triple_all_same_v1",
        "display_name": "EP-0004 · Snowy Highlands · 3 Aircraft · 3 Radars (High-Altitude Crossing)",
        "terrain_family": "basin_highlands",
        "texture_family": "snowy_highlands",
        "obstacle_profile": "clear",
        "difficulty_level": "medium",
        "radar_placement_profile": "cross_valley_triangulation",
        "radar_category": "high_altitude_aircraft",
        "target_profile_composition": "all_same",
        "physical_target_type_composition": "all_same",
        "target_profiles": ["aircraft_cruise", "aircraft_cruise", "aircraft_cruise"],
        "target_types": ["aircraft_like", "aircraft_like", "aircraft_like"],
        "route_family": "corridor_south_to_north",
        "operational_objective": "Three-ship high-altitude formation crossing snowy alpine basin with staggered corridors and shared tracker slot management.",
        "terrain_params": {
            "basin_radius_m": 4600.0,
            "relief_height_m": 950.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (-800.0, -5000.0, 2200.0),
                "velocity": (2.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (0.0, -5400.0, 2500.0),
                "velocity": (0.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (800.0, -5100.0, 2800.0),
                "velocity": (-2.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
        ],
    },

    # -------------------------------------------------------------------------
    # 3. low_altitude_missile_run
    # -------------------------------------------------------------------------
    {
        "template": "low_altitude_missile_run",
        "dataset_name": "dataset_single_missile_v1",
        "display_name": "EP-0005 · Dry Pass · Missile · 3 Radars (Low-Altitude Missile Run)",
        "terrain_family": "mountain_chokepoint",
        "texture_family": "dry_canyon",
        "obstacle_profile": "sparse",
        "difficulty_level": "medium",
        "radar_placement_profile": "cross_valley_triangulation",
        "radar_category": "fast_missile",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["missile_dash"],
        "target_types": ["missile_like"],
        "route_family": "chokepoint_penetration",
        "operational_objective": "Supersonic terrain-following missile dash through constricted pass; early entry detection, deep gorge shadowing, exit intercept.",
        "terrain_params": {
            "chokepoint_width_m": 1700.0,
            "open_width_m": 5500.0,
            "relief_height_m": 920.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 2,
                "target_type": "missile_like",
                "rel_pos": (0.0, -4200.0, 140.0),
                "velocity": (0.0, 300.0, 0.0),
                "motion": {"model": "CV"},
            }
        ],
    },
    {
        "template": "low_altitude_missile_run",
        "dataset_name": "dataset_single_missile_v1",
        "display_name": "EP-0006 · Rocky Pass · Missile · 3 Radars (Low-Altitude Missile Run)",
        "terrain_family": "mountain_chokepoint",
        "texture_family": "rocky_mountain",
        "obstacle_profile": "asymmetric",
        "difficulty_level": "hard",
        "radar_placement_profile": "asymmetric_ridge_chokepoint",
        "radar_category": "fast_missile",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["missile_dash"],
        "target_types": ["missile_like"],
        "route_family": "chokepoint_penetration",
        "operational_objective": "High-speed dash along steep asymmetric mountain wall testing radar elevation gating under tight corridor constraints.",
        "terrain_params": {
            "chokepoint_width_m": 1600.0,
            "open_width_m": 5200.0,
            "relief_height_m": 950.0,
            "asymmetry_ratio": 1.5,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 2,
                "target_type": "missile_like",
                "rel_pos": (-200.0, -4000.0, 150.0),
                "velocity": (4.0, 310.0, 0.0),
                "motion": {"model": "CV"},
            }
        ],
    },

    # -------------------------------------------------------------------------
    # 4. coordinated_uav_approach
    # -------------------------------------------------------------------------
    {
        "template": "coordinated_uav_approach",
        "dataset_name": "dataset_triple_all_same_v1",
        "display_name": "EP-0007 · Forest Winding Pass · 3 UAV Agile · 3 Radars (Coordinated UAV Approach)",
        "terrain_family": "valley_sinuous_pass",
        "texture_family": "green_forest_valley",
        "obstacle_profile": "sparse",
        "difficulty_level": "medium",
        "radar_placement_profile": "lateral_flank_staggered",
        "radar_category": "low_altitude_uav",
        "target_profile_composition": "all_same",
        "physical_target_type_composition": "all_same",
        "target_profiles": ["uav_agile", "uav_agile", "uav_agile"],
        "target_types": ["uav_like", "uav_like", "uav_like"],
        "route_family": "sinuous_pass_transit",
        "operational_objective": "Multi-axis coordinated UAV surveillance approach across staggered radar coverage sectors along winding valley.",
        "terrain_params": {
            "valley_width_m": 3200.0,
            "relief_height_m": 880.0,
            "sinuosity_amplitude_m": 1500.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (-600.0, -2800.0, 170.0),
                "velocity": (10.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.035},
            },
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (0.0, -2300.0, 190.0),
                "velocity": (0.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": -0.035},
            },
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (600.0, -1800.0, 210.0),
                "velocity": (-10.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.035},
            },
        ],
    },
    {
        "template": "coordinated_uav_approach",
        "dataset_name": "dataset_triple_two_plus_one_v1",
        "display_name": "EP-0008 · Grassland Ridge · 3 UAV (2 Agile + 1 Drift) · 3 Radars (Coordinated UAV Approach)",
        "terrain_family": "ridge_multi_obstacle",
        "texture_family": "grassland_valley",
        "obstacle_profile": "sparse",
        "difficulty_level": "easy",
        "radar_placement_profile": "valley_entrance_exit_overlook",
        "radar_category": "low_altitude_uav",
        "target_profile_composition": "two_plus_one",
        "physical_target_type_composition": "all_same",
        "target_profiles": ["uav_agile", "uav_agile", "uav_drift"],
        "target_types": ["uav_like", "uav_like", "uav_like"],
        "route_family": "transverse_spur_weave",
        "operational_objective": "Heterogeneous UAV scouting team: 2 agile forward scouts flanked by 1 persistent drift sensor platform navigating spur ridges.",
        "terrain_params": {
            "valley_width_m": 4200.0,
            "relief_height_m": 880.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (-500.0, -2600.0, 170.0),
                "velocity": (12.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.04},
            },
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (500.0, -2300.0, 180.0),
                "velocity": (-12.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": -0.04},
            },
            {
                "label": 3,
                "target_type": "uav_like",
                "rel_pos": (0.0, -1800.0, 220.0),
                "velocity": (2.0, 40.0, 0.0),
                "motion": {"model": "CV"},
            },
        ],
    },

    # -------------------------------------------------------------------------
    # 5. mixed_target_incursion
    # -------------------------------------------------------------------------
    {
        "template": "mixed_target_incursion",
        "dataset_name": "dataset_triple_all_different_v1",
        "display_name": "EP-0009 · Mixed Canyon · 3 Targets (All Different) · 3 Radars (Mixed-Target Incursion)",
        "terrain_family": "valley_linear_canyon",
        "texture_family": "mixed_rock_vegetation",
        "obstacle_profile": "sparse",
        "difficulty_level": "medium",
        "radar_placement_profile": "cross_valley_triangulation",
        "radar_category": "mixed_three_targets",
        "target_profile_composition": "all_different",
        "physical_target_type_composition": "all_different",
        "target_profiles": ["aircraft_cruise", "uav_agile", "missile_dash"],
        "target_types": ["aircraft_like", "uav_like", "missile_like"],
        "route_family": "corridor_south_to_north",
        "operational_objective": "Multi-tier strike package: high-altitude aircraft escort, medium-altitude agile UAV scout, and low-altitude dash missile.",
        "terrain_params": {
            "valley_width_m": 3400.0,
            "relief_height_m": 880.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (-500.0, -5000.0, 2300.0),
                "velocity": (2.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (400.0, -2200.0, 180.0),
                "velocity": (-8.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.04},
            },
            {
                "label": 2,
                "target_type": "missile_like",
                "rel_pos": (0.0, -3800.0, 140.0),
                "velocity": (0.0, 300.0, 0.0),
                "motion": {"model": "CV"},
            },
        ],
    },
    {
        "template": "mixed_target_incursion",
        "dataset_name": "dataset_triple_all_different_v1",
        "display_name": "EP-0010 · Dry Pass · 3 Targets (All Different) · 3 Radars (Mixed-Target Incursion)",
        "terrain_family": "mountain_chokepoint",
        "texture_family": "dry_canyon",
        "obstacle_profile": "sparse",
        "difficulty_level": "hard",
        "radar_placement_profile": "asymmetric_ridge_chokepoint",
        "radar_category": "mixed_three_targets",
        "target_profile_composition": "all_different",
        "physical_target_type_composition": "all_different",
        "target_profiles": ["aircraft_cruise", "uav_agile", "missile_dash"],
        "target_types": ["aircraft_like", "uav_like", "missile_like"],
        "route_family": "chokepoint_penetration",
        "operational_objective": "Joint multi-regime transit through constricted mountain pass testing resource allocation between high and low targets.",
        "terrain_params": {
            "chokepoint_width_m": 1800.0,
            "open_width_m": 5500.0,
            "relief_height_m": 900.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (400.0, -5200.0, 2400.0),
                "velocity": (-3.0, 180.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (-300.0, -2000.0, 170.0),
                "velocity": (6.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.04},
            },
            {
                "label": 2,
                "target_type": "missile_like",
                "rel_pos": (100.0, -3600.0, 130.0),
                "velocity": (2.0, 310.0, 0.0),
                "motion": {"model": "CV"},
            },
        ],
    },

    # -------------------------------------------------------------------------
    # 6. terrain_masking_reacquisition
    # -------------------------------------------------------------------------
    {
        "template": "terrain_masking_reacquisition",
        "dataset_name": "dataset_single_uav_agile_v1",
        "display_name": "EP-0011 · Rocky Ridge · UAV Agile · 3 Radars (Terrain Masking Reacquisition)",
        "terrain_family": "ridge_multi_obstacle",
        "texture_family": "rocky_mountain",
        "obstacle_profile": "sparse",
        "difficulty_level": "medium",
        "radar_placement_profile": "lateral_flank_staggered",
        "radar_category": "low_altitude_uav",
        "target_profile_composition": "single",
        "physical_target_type_composition": "single",
        "target_profiles": ["uav_agile"],
        "target_types": ["uav_like"],
        "route_family": "transverse_spur_weave",
        "operational_objective": "Intentional terrain masking behind a mountain spur followed by opportunistic sensor handover and reacquisition.",
        "terrain_params": {
            "valley_width_m": 4000.0,
            "relief_height_m": 920.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 1,
                "target_type": "uav_like",
                "rel_pos": (-400.0, -4200.0, 160.0),
                "velocity": (14.0, 65.0, 0.0),
                "motion": {"model": "CT", "turn_rate_rad_s": 0.038},
            }
        ],
    },
    {
        "template": "terrain_masking_reacquisition",
        "dataset_name": "dataset_triple_two_plus_one_v1",
        "display_name": "EP-0012 · Dry Ridge · 3 Targets (2 Aircraft + 1 Missile) · 3 Radars (Terrain Masking Reacquisition)",
        "terrain_family": "ridge_multi_obstacle",
        "texture_family": "dry_canyon",
        "obstacle_profile": "sparse",
        "difficulty_level": "hard",
        "radar_placement_profile": "valley_entrance_exit_overlook",
        "radar_category": "mixed_three_targets",
        "target_profile_composition": "two_plus_one",
        "physical_target_type_composition": "two_plus_one",
        "target_profiles": ["aircraft_cruise", "aircraft_cruise", "missile_dash"],
        "target_types": ["aircraft_like", "aircraft_like", "missile_like"],
        "route_family": "transverse_spur_weave",
        "operational_objective": "High-altitude aircraft diversion while low-flying missile ducks behind transverse ridge before re-emerging into exit radar field.",
        "terrain_params": {
            "valley_width_m": 4200.0,
            "relief_height_m": 920.0,
            "generator_version": 2,
        },
        "targets_spec": [
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (-600.0, -5000.0, 2200.0),
                "velocity": (2.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 0,
                "target_type": "aircraft_like",
                "abs_pos": (600.0, -5200.0, 2500.0),
                "velocity": (-2.0, 175.0, 0.0),
                "motion": {"model": "CV"},
            },
            {
                "label": 2,
                "target_type": "missile_like",
                "rel_pos": (0.0, -3600.0, 130.0),
                "velocity": (4.0, 300.0, 0.0),
                "motion": {"model": "CV"},
            },
        ],
    },
]


def build_scenario_design_prototype_dataset(output_dir: str | Path | None = None):
    print("=================================================================")
    print("STEP 4D: GENERATING SCENARIO DESIGN PROTOTYPE (12 EPISODES)")
    print("=================================================================")

    target_dir = Path(output_dir) if output_dir else PROTOTYPE_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    master_seed = 20261117
    plan_episodes = []

    for ep_idx, ep_spec in enumerate(PROTOTYPE_EPISODES_SPEC):
        global_ep_index = ep_idx + 1
        ep_seed = master_seed + 1000 + global_ep_index
        rng = np.random.default_rng(ep_seed)
        ter_seed = ep_seed + 777

        ter_spec = {
            "source": "procedural",
            "resolution": 256,
            "family": ep_spec["terrain_family"],
            "texture_family": ep_spec["texture_family"],
            "obstacle_profile": ep_spec["obstacle_profile"],
            "generator_version": 2,
            "seed": ter_seed,
            "parameters": ep_spec["terrain_params"],
        }
        for k in ("valley_width_m", "relief_height_m"):
            if k in ep_spec["terrain_params"]:
                ter_spec[k] = ep_spec["terrain_params"][k]

        dummy_sc = {"terrain": ter_spec, "site": {"lat": 40.30, "lon": -111.70}}
        ter_obj = _cached_terrain(dummy_sc)

        # Radars
        cat_key = ep_spec["radar_category"]
        cat_cfg = SCENARIO_RADAR_CONFIGS[cat_key]
        rad_prof_key = ep_spec["radar_placement_profile"]
        radars = deepcopy(RADAR_PLACEMENT_PROFILES_V2[rad_prof_key]["radars"])
        for r in radars:
            r["search_sector_deg"] = cat_cfg["search_sector_deg"]
            r["search_bars_deg"] = cat_cfg["search_bars_deg"]

        # Targets with terrain clearance
        targets = []
        for t_cfg in ep_spec["targets_spec"]:
            if "abs_pos" in t_cfg:
                x, y, z = t_cfg["abs_pos"]
            else:
                rx, ry, agl = t_cfg["rel_pos"]
                x, y = rx, ry
                gh = float(ter_obj.height(x, y))
                z = round(gh + agl, 1)

            t_resolved = {
                "label": t_cfg["label"],
                "target_type": t_cfg["target_type"],
                "position": [round(float(x), 1), round(float(y), 1), round(float(z), 1)],
                "velocity": [round(float(v), 1) for v in t_cfg["velocity"]],
                "motion": deepcopy(t_cfg["motion"]),
            }
            targets.append(t_resolved)

        sc = {
            "name": f"scenario_prototype_ep{global_ep_index:04d}",
            "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
            "area_radius_m": 7500,
            "duration_s": 30.0,
            "dt": 0.5,
            "max_track_slots": 3,
            "difficulty_level": ep_spec["difficulty_level"],
            "terrain": ter_spec,
            "radar_defaults": {
                "search_sector_deg": cat_cfg["search_sector_deg"],
                "search_bars_deg": cat_cfg["search_bars_deg"],
            },
            "radars": radars,
            "targets": targets,
        }

        validated_sc = validate_scenario(sc)
        ter_group_id = terrain_group_id(validated_sc["terrain"])
        fingerprint = [{k: t[k] for k in ("position", "velocity")} for t in validated_sc["targets"]]
        route_id = "route-" + _digest(fingerprint)[:20]
        snapshot = scenario_snapshot(validated_sc, episode_seed=ep_seed)
        config_hash = _digest(snapshot)
        dataset_ep_id = f"ep-{global_ep_index:06d}-" + _digest([master_seed, global_ep_index, "train", config_hash])[:12]

        episode_entry = {
            "dataset_name": "dataset_prototype_episodes",
            "dataset_episode_id": dataset_ep_id,
            "episode_index": global_ep_index,
            "display_name": ep_spec["display_name"],
            "template": ep_spec["template"],
            "difficulty_level": ep_spec["difficulty_level"],
            "split": "train",
            "scenario_family": ep_spec["template"],
            "terrain_family": ep_spec["terrain_family"],
            "terrain_seed": ter_seed,
            "terrain_group_id": ter_group_id,
            "terrain_parameters": ep_spec["terrain_params"],
            "texture_family": ep_spec["texture_family"],
            "obstacle_profile": ep_spec["obstacle_profile"],
            "generator_version": 2,
            "radar_count": 3,
            "radar_placement_profile": rad_prof_key,
            "radar_elevation_bars": cat_cfg["search_bars_deg"],
            "radar_search_sector_deg": cat_cfg["search_sector_deg"],
            "radar_category": cat_key,
            "target_count": len(targets),
            "target_types": ep_spec["target_types"],
            "motion_profiles": [t["motion"]["model"] for t in targets],
            "class_labels": [t["label"] for t in targets],
            "target_profiles": ep_spec["target_profiles"],
            "physical_target_type_composition": ep_spec["physical_target_type_composition"],
            "target_profile_composition": ep_spec["target_profile_composition"],
            "route_family": ep_spec["route_family"],
            "route_group_id": route_id,
            "operational_objective": ep_spec["operational_objective"],
            "geometry_group_id": "geom-" + _digest([ter_group_id, route_id, validated_sc["radars"]])[:20],
            "master_seed": master_seed,
            "episode_seed": ep_seed,
            "resolved_scenario": validated_sc,
            "config_snapshot": snapshot,
            "resolved_configuration_hash": config_hash,
            "recipe_version": 1,
        }
        plan_episodes.append(episode_entry)

    recipe = {
        "name": "scenario_design_prototype_v1",
        "recipe_version": 1,
        "master_seed": master_seed,
    }
    plan = {
        "plan_schema_version": 1,
        "recipe_version": 1,
        "recipe": recipe,
        "plan_version": 1,
        "recipe_hash": _digest(recipe),
        "master_seed": master_seed,
        "splits": {"train": 12, "validation": 0, "test": 0},
        "episodes": plan_episodes,
    }

    # Execute simulation generation
    output_group_dir = target_dir / "dataset_prototype_episodes"
    output_group_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating 12 canonical prototype episodes into {output_group_dir}...")
    manifest = generate_dataset(plan, output_group_dir, policy_name="central_greedy", resume=False)

    master_previews = []
    for ep_idx, (episode, entry) in enumerate(zip(plan["episodes"], manifest["episodes"])):
        ep_dir = output_group_dir / episode["split"] / episode["dataset_episode_id"]
        quality = compute_physical_quality_metrics(ep_dir)
        preview = {
            "dataset_name": "dataset_prototype_episodes",
            "episode_id": episode["dataset_episode_id"],
            "display_name": episode["display_name"],
            "template": episode["template"],
            "split": episode["split"],
            "scenario_family": episode["scenario_family"],
            "terrain_family": episode["terrain_family"],
            "terrain_seed": episode["terrain_seed"],
            "terrain_group_id": episode["terrain_group_id"],
            "terrain_parameters": episode["terrain_parameters"],
            "texture_family": episode["texture_family"],
            "obstacle_profile": episode["obstacle_profile"],
            "generator_version": 2,
            "radar_count": episode["radar_count"],
            "radar_placement_profile": episode["radar_placement_profile"],
            "target_count": episode["target_count"],
            "target_types": episode["target_types"],
            "target_profiles": episode["target_profiles"],
            "motion_profiles": episode["motion_profiles"],
            "class_labels": episode["class_labels"],
            "physical_target_type_composition": episode["physical_target_type_composition"],
            "target_profile_composition": episode["target_profile_composition"],
            "route_family": episode["route_family"],
            "episode_seed": episode["episode_seed"],
            "difficulty_level": episode["difficulty_level"],
            "operational_objective": episode["operational_objective"],
            "quality_summary": quality,
        }
        entry["episode_preview"] = preview
        entry["quality_metrics"] = quality
        master_previews.append(preview)
        print(f"  [{ep_idx+1}/12] {preview['display_name']} -> LOS: {quality['los_clear_pct']:.1f}%, Track Cov: {quality['track_coverage_pct']:.1f}%, Zero Tgts: {quality['zero_detection_targets']}")

    # Save manifest.json in group dir
    manifest_path = output_group_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Save top-level metadata index and manifest
    summary = {
        "prototype_schema_version": 1,
        "prototype_name": "scenario_design_prototype_v1",
        "total_episodes": len(master_previews),
        "split_counts": {"train": 12, "validation": 0, "test": 0},
        "template_counts": {
            "valley_infiltration": 2,
            "high_altitude_crossing": 2,
            "low_altitude_missile_run": 2,
            "coordinated_uav_approach": 2,
            "mixed_target_incursion": 2,
            "terrain_masking_reacquisition": 2,
        },
        "episodes": master_previews,
    }
    index_path = target_dir / "scenario_design_metadata_index.json"
    index_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    top_manifest = target_dir / "manifest.json"
    top_manifest.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=================================================================")
    print(">>> 12 PROTOTYPE EPISODES GENERATED & VALIDATED WITH 100% SUCCESS! <<<")
    print(f"Location: {target_dir}")
    print("=================================================================")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Produce Scenario Design Prototype Episodes")
    parser.add_argument("--output-dir", default=str(PROTOTYPE_DIR), help="Output directory path")
    args = parser.parse_args()
    build_scenario_design_prototype_dataset(output_dir=args.output_dir)
