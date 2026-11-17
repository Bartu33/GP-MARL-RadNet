"""Diverse Valley Dataset Service for GP-MARL-RadNet (Step 2C).

Orchestrates the generation, metadata indexation, and physical quality validation
of the 7 target-diverse dataset groups across procedural valley terrain families,
texture styles, obstacle profiles, and radar placement strategies:

Seven dataset groups:
1. dataset_single_aircraft_v1   (Single aircraft-like cruise)
2. dataset_single_uav_agile_v1   (Single UAV-like agile maneuver)
3. dataset_single_uav_drift_v1   (Single UAV-like drift)
4. dataset_single_missile_v1     (Single missile-like high-speed dash)
5. dataset_triple_all_same_v1    (Triple targets, all same profile)
6. dataset_triple_all_different_v1 (Triple targets, 3 distinct profiles/types)
7. dataset_triple_two_plus_one_v1  (Triple targets, 2 same profile + 1 distinct)

Guarantees:
- Strict split isolation (disjoint terrain seeds and route hashes across train/val/test).
- Exact physical vs visual boundary: Terrain.H governs radar LOS; visual 3D objects are decorative.
- Ground clearance: targets AGL >= 30m, radar antennas mast-mounted above local ground.
- Deterministic descriptive episode names (display_name).
- Canonical Episode Records v2 compliance (metadata.json + records.npz).
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from src.core.env import _cached_terrain, RadarNetEnv
from src.core.procedural_terrain import (
    SUPPORTED_TERRAIN_FAMILIES,
    SUPPORTED_TEXTURE_FAMILIES,
    SUPPORTED_OBSTACLE_PROFILES,
)
from src.core.scenario import (
    DEFAULTS,
    scenario_snapshot,
    validate_scenario,
    _merge,
)
from src.core.targets import TARGET_TYPE_PROFILES
from src.services.scenario_planner import _digest, terrain_group_id
from src.services.simulation_service import (
    export_episode_records,
    load_episode_records,
    run_episode,
)
from src.services.synthetic_dataset_service import generate_dataset

DATASET_GROUPS = (
    "dataset_single_aircraft_v1",
    "dataset_single_uav_agile_v1",
    "dataset_single_uav_drift_v1",
    "dataset_single_missile_v1",
    "dataset_triple_all_same_v1",
    "dataset_triple_all_different_v1",
    "dataset_triple_two_plus_one_v1",
)

TARGET_PROFILES = {
    "aircraft_cruise": {
        "physical_type": "aircraft_like",
        "legacy_label": 0,
        "name": "Aircraft Cruise",
        "nominal_speed": 150.0,
        "speed_range": (130.0, 180.0),
        "altitude_range": (1400.0, 2600.0),
        "motion_model": "CV",
    },
    "uav_agile": {
        "physical_type": "uav_like",
        "legacy_label": 1,
        "name": "UAV Agile",
        "nominal_speed": 15.0,
        "speed_range": (10.0, 22.0),
        "altitude_range": (120.0, 260.0),
        "motion_model": "CT",
        "turn_rate_rad_s": 0.08,
    },
    "uav_drift": {
        "physical_type": "uav_like",
        "legacy_label": 3,
        "name": "UAV Drift",
        "nominal_speed": 8.0,
        "speed_range": (5.0, 12.0),
        "altitude_range": (150.0, 320.0),
        "motion_model": "CV",
    },
    "missile_dash": {
        "physical_type": "missile_like",
        "legacy_label": 2,
        "name": "Missile Dash",
        "nominal_speed": 300.0,
        "speed_range": (260.0, 360.0),
        "altitude_range": (300.0, 550.0),
        "motion_model": "CV",
    },
}

RADAR_PLACEMENT_PROFILES = {
    "valley_entrance_exit_overlook": {
        "radars": [
            {"name": "R1", "position": [0, -4500, 15], "heading_deg": 90},
            {"name": "R2", "position": [0, 4500, 15], "heading_deg": -90},
            {"name": "R3", "position": [3200, 0, 22], "heading_deg": 180},
        ],
    },
    "cross_valley_triangulation": {
        "radars": [
            {"name": "R1", "position": [-3000, -2500, 18], "heading_deg": 45},
            {"name": "R2", "position": [3000, -2500, 18], "heading_deg": 135},
            {"name": "R3", "position": [0, 4200, 15], "heading_deg": -90},
        ],
    },
    "asymmetric_ridge_chokepoint": {
        "radars": [
            {"name": "R1", "position": [-3400, 800, 25], "heading_deg": 0},
            {"name": "R2", "position": [2000, -3800, 15], "heading_deg": 105},
            {"name": "R3", "position": [-1200, 3200, 18], "heading_deg": -60},
        ],
    },
    "lateral_flank_staggered": {
        "radars": [
            {"name": "R1", "position": [-2500, -3800, 15], "heading_deg": 30},
            {"name": "R2", "position": [2800, 500, 20], "heading_deg": 160},
            {"name": "R3", "position": [-2200, 4200, 15], "heading_deg": -45},
        ],
    },
}

# Refined Military Crest Siting & Orientation (Pilot v2)
RADAR_PLACEMENT_PROFILES_V2 = {
    "valley_entrance_exit_overlook": {
        "radars": [
            {"name": "R1", "position": [0, -4200, 15], "heading_deg": 90},
            {"name": "R2", "position": [0, 4200, 15], "heading_deg": -90},
            {"name": "R3", "position": [2200, 0, 22], "heading_deg": 180},
        ],
    },
    "cross_valley_triangulation": {
        "radars": [
            {"name": "R1", "position": [-2000, -2500, 20], "heading_deg": 45},
            {"name": "R2", "position": [2000, -2500, 20], "heading_deg": 135},
            {"name": "R3", "position": [0, 3800, 15], "heading_deg": -90},
        ],
    },
    "asymmetric_ridge_chokepoint": {
        "radars": [
            {"name": "R1", "position": [-2200, 800, 25], "heading_deg": 0},
            {"name": "R2", "position": [1800, -3000, 20], "heading_deg": 140},
            {"name": "R3", "position": [-1000, 3200, 18], "heading_deg": -60},
        ],
    },
    "lateral_flank_staggered": {
        "radars": [
            {"name": "R1", "position": [-1800, -2800, 18], "heading_deg": 45},
            {"name": "R2", "position": [2000, 200, 22], "heading_deg": 160},
            {"name": "R3", "position": [-1600, 3200, 18], "heading_deg": -45},
        ],
    },
}

SCENARIO_RADAR_CONFIGS = {
    "high_altitude_aircraft": {
        "search_sector_deg": 90.0,
        "search_bars_deg": [6.0, 18.0, 32.0],
    },
    "low_altitude_uav": {
        "search_sector_deg": 120.0,
        "search_bars_deg": [1.5, 5.0, 9.5],
    },
    "slow_drift_uav": {
        "search_sector_deg": 120.0,
        "search_bars_deg": [1.5, 5.0, 9.5],
    },
    "fast_missile": {
        "search_sector_deg": 120.0,
        "search_bars_deg": [3.0, 8.0],
    },
    "mixed_three_targets": {
        "search_sector_deg": 120.0,
        "search_bars_deg": [2.0, 7.0, 15.0, 25.0],
    },
}

GROUP_DIFFICULTY_SCHEDULE = [
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single Aircraft
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single UAV Agile
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single UAV Drift
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single Missile
    ["easy", "medium", "medium", "medium", "hard", "hard"],      # Triple All Same
    ["easy", "medium", "medium", "medium", "hard", "hard"],      # Triple All Different
    ["easy", "easy", "easy", "medium", "hard", "hard"],          # Triple Two Plus One
]

ROUTE_FAMILIES = (
    "corridor_south_to_north",
    "sinuous_pass_transit",
    "mountain_ridge_flank",
    "chokepoint_penetration",
    "transverse_spur_weave",
)


def format_display_name(
    episode_index: int,
    terrain_family: str,
    texture_family: str,
    target_profile_composition: str,
    target_profiles: List[str],
    radar_count: int,
) -> str:
    """Produces a deterministic, descriptive title for an episode.

    Examples:
    - EP-0001 · Rocky Canyon · Aircraft · 3 Radars
    - EP-0012 · Forest Valley · UAV Agile · 3 Radars
    - EP-0025 · Snowy Pass · 3 Targets (All Same) · 3 Radars
    - EP-0031 · Dry Canyon · 3 Targets (All Different) · 3 Radars
    - EP-0037 · Forest Valley · 3 UAV (2 Agile + 1 Drift) · 3 Radars
    """
    ep_prefix = f"EP-{episode_index:04d}"

    visual_prefixes = {
        "rocky_mountain": "Rocky",
        "green_forest_valley": "Forest",
        "dry_canyon": "Dry",
        "snowy_highlands": "Snowy",
        "grassland_valley": "Grassland",
        "mixed_rock_vegetation": "Mixed",
    }
    terrain_names = {
        "valley_linear_canyon": "Canyon",
        "valley_sinuous_pass": "Winding Pass",
        "mountain_chokepoint": "Pass",
        "basin_highlands": "Highlands",
        "ridge_multi_obstacle": "Ridge Corridor",
    }
    vis = visual_prefixes.get(texture_family, "Valley")
    geo = terrain_names.get(terrain_family, "Valley")
    terrain_desc = f"{vis} {geo}"

    profile_names = {
        "aircraft_cruise": "Aircraft",
        "uav_agile": "UAV Agile",
        "uav_drift": "UAV Drift",
        "missile_dash": "Missile",
    }

    if len(target_profiles) == 1:
        target_desc = profile_names.get(target_profiles[0], target_profiles[0])
    elif target_profile_composition == "all_same":
        target_desc = "3 Targets (All Same)"
    elif target_profile_composition == "all_different":
        target_desc = "3 Targets (All Different)"
    elif target_profile_composition == "two_plus_one":
        counts = {}
        for p in target_profiles:
            counts[p] = counts.get(p, 0) + 1
        if all("uav" in p for p in target_profiles):
            agile_c = counts.get("uav_agile", 0)
            drift_c = counts.get("uav_drift", 0)
            target_desc = f"3 UAV ({agile_c} Agile + {drift_c} Drift)"
        else:
            target_desc = "3 Targets (2+1)"
    else:
        target_desc = f"{len(target_profiles)} Targets"

    radar_desc = f"{radar_count} Radars"
    return f"{ep_prefix} · {terrain_desc} · {target_desc} · {radar_desc}"


def _make_target_spec(profile_key: str, route_family: str, target_idx: int, rng: np.random.Generator) -> Dict[str, Any]:
    prof = TARGET_PROFILES[profile_key]
    phys_type = prof["physical_type"]
    label = prof["legacy_label"]

    lateral_offsets = [0.0, -900.0, 900.0, -1600.0, 1600.0]
    lat_off = lateral_offsets[target_idx % len(lateral_offsets)]

    v_nom = prof["nominal_speed"]
    alt_low, alt_high = prof["altitude_range"]
    z = float(rng.uniform(alt_low, alt_high))

    if route_family == "corridor_south_to_north":
        x = lat_off + float(rng.uniform(-200.0, 200.0))
        y = -5200.0 - (target_idx * 1400.0)
        vx = float(rng.uniform(-5.0, 5.0))
        vy = v_nom * float(rng.uniform(0.92, 1.08))
    elif route_family == "sinuous_pass_transit":
        x = -1500.0 + lat_off * 0.5
        y = -5000.0 - (target_idx * 1200.0)
        vx = v_nom * 0.3
        vy = v_nom * 0.95
    elif route_family == "chokepoint_penetration":
        x = lat_off * 0.7
        y = -5500.0 - (target_idx * 1600.0)
        vx = float(rng.uniform(-8.0, 8.0))
        vy = v_nom
    elif route_family == "mountain_ridge_flank":
        x = 2400.0 + (target_idx * 400.0)
        y = -4800.0 - (target_idx * 1000.0)
        vx = -v_nom * 0.15
        vy = v_nom * 0.98
    else:  # transverse_spur_weave
        x = -800.0 + lat_off * 0.6
        y = -5000.0 - (target_idx * 1300.0)
        vx = v_nom * 0.2
        vy = v_nom * 0.95

    motion_spec: Dict[str, Any] = {"model": prof["motion_model"]}
    if prof["motion_model"] == "CT":
        turn_rate = prof.get("turn_rate_rad_s", 0.05) * (1.0 if (target_idx % 2 == 0) else -1.0)
        motion_spec["turn_rate_rad_s"] = round(turn_rate, 4)

    return {
        "label": label,
        "target_type": phys_type,
        "position": [round(x, 1), round(y, 1), round(z, 1)],
        "velocity": [round(vx, 1), round(vy, 1), 0.0],
        "motion": motion_spec,
    }


def build_diverse_pilot_episodes() -> Dict[str, List[Dict[str, Any]]]:
    """Builds explicit specifications for the 42 pilot episodes (6 per dataset group)."""
    # 42 distinct configuration templates covering all geomorphic, texture, obstacle, and target combinations.
    # Group split policy: index 0..3 -> train (4), index 4 -> validation (1), index 5 -> test (1).
    splits = ["train", "train", "train", "train", "validation", "test"]

    # Systematic cycling parameters ensuring complete coverage of 5 terrains, 6 textures, 4 obstacles
    terrain_fams = list(SUPPORTED_TERRAIN_FAMILIES)  # 5
    texture_fams = list(SUPPORTED_TEXTURE_FAMILIES)  # 6
    obstacle_profs = list(SUPPORTED_OBSTACLE_PROFILES)  # 4
    radar_profs = list(RADAR_PLACEMENT_PROFILES.keys())  # 4
    route_fams = list(ROUTE_FAMILIES)  # 5

    group_definitions = {
        "dataset_single_aircraft_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise"],
        },
        "dataset_single_uav_agile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_agile"],
        },
        "dataset_single_uav_drift_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_drift"],
        },
        "dataset_single_missile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["missile_dash"],
        },
        "dataset_triple_all_same_v1": {
            "comp_profile": "all_same",
            "comp_physical": "all_same",
            "targets_builder": lambda i, r, rng: (
                ["aircraft_cruise", "aircraft_cruise", "aircraft_cruise"] if (i % 3 == 0) else
                (["uav_agile", "uav_agile", "uav_agile"] if (i % 3 == 1) else
                 ["missile_dash", "missile_dash", "missile_dash"])
            ),
        },
        "dataset_triple_all_different_v1": {
            "comp_profile": "all_different",
            "comp_physical": "all_different",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise", "uav_agile", "missile_dash"],
        },
        "dataset_triple_two_plus_one_v1": {
            "comp_profile": "two_plus_one",
            "comp_physical": "evaluated_per_combo",
            "targets_builder": lambda i, r, rng: (
                # 2 UAV Agile + 1 UAV Drift (profile: 2+1, physical: all_same [all uav_like])
                ["uav_agile", "uav_agile", "uav_drift"] if (i % 2 == 0) else
                # 2 Aircraft + 1 Missile (profile: 2+1, physical: 2+1)
                ["aircraft_cruise", "aircraft_cruise", "missile_dash"]
            ),
        },
    }

    all_groups_episodes: Dict[str, List[Dict[str, Any]]] = {}
    global_ep_index = 1
    master_seed = 20261116

    for g_idx, (group_name, group_cfg) in enumerate(group_definitions.items()):
        group_episodes = []
        for ep_in_group in range(6):
            split = splits[ep_in_group]
            # Ensure disjoint seeds across splits to guarantee zero leakage
            split_salt = {"train": 1000, "validation": 2000, "test": 3000}[split]
            ep_seed = master_seed + split_salt + (g_idx * 50) + ep_in_group
            rng = np.random.default_rng(ep_seed)

            # Cycle geomorphic, visual and placement attributes
            cfg_idx = g_idx * 6 + ep_in_group
            ter_fam = terrain_fams[cfg_idx % len(terrain_fams)]
            tex_fam = texture_fams[cfg_idx % len(texture_fams)]
            obs_prof = obstacle_profs[cfg_idx % len(obstacle_profs)]
            rad_prof_key = radar_profs[cfg_idx % len(radar_profs)]
            route_fam = route_fams[cfg_idx % len(route_fams)]

            # Terrain parameters
            w_valley = round(float(rng.uniform(2400.0, 3600.0)), 1)
            h_relief = round(float(rng.uniform(750.0, 950.0)), 1)
            ter_seed = ep_seed + 777

            # Build targets
            target_profile_keys = group_cfg["targets_builder"](ep_in_group, route_fam, rng)
            targets = []
            for t_idx, p_key in enumerate(target_profile_keys):
                t_spec = _make_target_spec(p_key, route_fam, t_idx, rng)
                targets.append(t_spec)

            # Physical composition determination
            phys_types = [TARGET_PROFILES[p]["physical_type"] for p in target_profile_keys]
            if len(phys_types) == 1:
                phys_comp = "single"
            elif len(set(phys_types)) == 1:
                phys_comp = "all_same"
            elif len(set(phys_types)) == len(phys_types):
                phys_comp = "all_different"
            else:
                phys_comp = "two_plus_one"

            profile_comp = group_cfg["comp_profile"]

            # Display name
            disp_name = format_display_name(
                episode_index=global_ep_index,
                terrain_family=ter_fam,
                texture_family=tex_fam,
                target_profile_composition=profile_comp,
                target_profiles=target_profile_keys,
                radar_count=3,
            )

            # Construct full resolved scenario dictionary
            sc = {
                "name": f"scenario_{group_name}_ep{global_ep_index:04d}",
                "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
                "area_radius_m": 7500,
                "duration_s": 15.0,
                "dt": 0.5,
                "max_track_slots": 3,
                "terrain": {
                    "source": "procedural",
                    "resolution": 256,
                    "family": ter_fam,
                    "texture_family": tex_fam,
                    "obstacle_profile": obs_prof,
                    "valley_width_m": w_valley,
                    "relief_height_m": h_relief,
                    "seed": ter_seed,
                },
                "radar_defaults": deepcopy(DEFAULTS["radar_defaults"]),
                "radars": deepcopy(RADAR_PLACEMENT_PROFILES[rad_prof_key]["radars"]),
                "targets": targets,
            }

            validated_sc = validate_scenario(sc)
            ter_group_id = terrain_group_id(validated_sc["terrain"])

            # Compute route group id
            fingerprint = [{k: t[k] for k in ("position", "velocity")} for t in validated_sc["targets"]]
            route_id = "route-" + _digest(fingerprint)[:20]

            snapshot = scenario_snapshot(validated_sc, episode_seed=ep_seed)
            config_hash = _digest(snapshot)
            dataset_ep_id = f"ep-{global_ep_index:06d}-" + _digest([master_seed, global_ep_index, split, group_name, config_hash])[:12]

            episode_entry = {
                "dataset_name": group_name,
                "dataset_episode_id": dataset_ep_id,
                "episode_index": global_ep_index,
                "display_name": disp_name,
                "split": split,
                "scenario_family": group_name,
                "terrain_family": ter_fam,
                "terrain_seed": ter_seed,
                "terrain_group_id": ter_group_id,
                "terrain_parameters": {
                    "valley_width_m": w_valley,
                    "relief_height_m": h_relief,
                },
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "radar_count": 3,
                "radar_placement_profile": rad_prof_key,
                "target_count": len(targets),
                "target_types": phys_types,
                "motion_profiles": [t["motion"]["model"] for t in targets],
                "class_labels": [t["label"] for t in targets],
                "target_profiles": target_profile_keys,
                "physical_target_type_composition": phys_comp,
                "target_profile_composition": profile_comp,
                "route_family": route_fam,
                "route_group_id": route_id,
                "geometry_group_id": "geom-" + _digest([ter_group_id, route_id, validated_sc["radars"]])[:20],
                "master_seed": master_seed,
                "episode_seed": ep_seed,
                "resolved_scenario": validated_sc,
                "config_snapshot": snapshot,
                "resolved_configuration_hash": config_hash,
                "recipe_version": 1,
            }

            group_episodes.append(episode_entry)
            global_ep_index += 1

        all_groups_episodes[group_name] = group_episodes

    return all_groups_episodes


def make_target_spec_v2(terrain, profile_key: str, route_family: str, target_idx: int, rng: np.random.Generator) -> Dict[str, Any]:
    """Generates target kinematics with ground-relative altitude and speed-scaled ingress range."""
    prof = TARGET_PROFILES[profile_key]
    phys_type = prof["physical_type"]
    label = prof["legacy_label"]
    v_nom = prof["nominal_speed"]

    lat_offsets = [0.0, -600.0, 600.0, -1200.0, 1200.0]
    lat_off = lat_offsets[target_idx % len(lat_offsets)]

    if profile_key == "aircraft_cruise":
        base_y = -3500.0 - (target_idx * 500.0)
    elif profile_key == "missile_dash":
        base_y = -2600.0 - (target_idx * 500.0)
    elif profile_key == "uav_agile":
        base_y = -1800.0 - (target_idx * 400.0)
    elif profile_key == "uav_drift":
        base_y = -1200.0 - (target_idx * 300.0)
    else:
        base_y = -2200.0 - (target_idx * 400.0)

    if route_family == "corridor_south_to_north":
        x = lat_off + float(rng.uniform(-80.0, 80.0))
        y = base_y + float(rng.uniform(-20.0, 20.0))
        vx = float(rng.uniform(-2.0, 2.0))
        vy = v_nom * float(rng.uniform(0.97, 1.03))
    elif route_family == "sinuous_pass_transit":
        x = -500.0 + lat_off * 0.4 + float(rng.uniform(-40.0, 40.0))
        y = base_y + float(rng.uniform(-20.0, 20.0))
        vx = v_nom * 0.20
        vy = v_nom * 0.97
    elif route_family == "chokepoint_penetration":
        x = lat_off * 0.35 + float(rng.uniform(-40.0, 40.0))
        y = base_y + float(rng.uniform(-20.0, 20.0))
        vx = float(rng.uniform(-3.0, 3.0))
        vy = v_nom
    elif route_family == "mountain_ridge_flank":
        x = 1000.0 + (target_idx * 200.0) + float(rng.uniform(-40.0, 40.0))
        y = base_y + float(rng.uniform(-20.0, 20.0))
        vx = -v_nom * 0.08
        vy = v_nom * 0.98
    else:  # transverse_spur_weave
        x = -350.0 + lat_off * 0.4 + float(rng.uniform(-40.0, 40.0))
        y = base_y + float(rng.uniform(-20.0, 20.0))
        vx = v_nom * 0.15
        vy = v_nom * 0.97

    gh = float(terrain.height(x, y))
    if profile_key == "aircraft_cruise":
        z = float(rng.uniform(1800.0, 2500.0))
    elif profile_key == "missile_dash":
        z = gh + float(rng.uniform(320.0, 480.0))
    elif profile_key in ("uav_agile", "uav_drift"):
        z = gh + float(rng.uniform(150.0, 250.0))
    else:
        z = gh + float(rng.uniform(300.0, 600.0))

    motion_spec: Dict[str, Any] = {"model": prof["motion_model"]}
    if prof["motion_model"] == "CT":
        turn_rate = prof.get("turn_rate_rad_s", 0.05) * (1.0 if (target_idx % 2 == 0) else -1.0)
        motion_spec["turn_rate_rad_s"] = round(turn_rate, 4)

    return {
        "label": label,
        "target_type": phys_type,
        "position": [round(x, 1), round(y, 1), round(z, 1)],
        "velocity": [round(vx, 1), round(vy, 1), 0.0],
        "motion": motion_spec,
    }


def build_diverse_pilot_episodes_v2() -> Dict[str, List[Dict[str, Any]]]:
    """Builds explicit specifications for the 42 Pilot v2 episodes (Step 2F).

    Enhancements over v1:
    - 30-second episode duration (60 steps).
    - Scenario-aware radar elevation bars & search sector.
    - Sited inward-oriented radar placement profiles.
    - Speed-scaled target ingress positions.
    - Ground-relative altitudes (zero clearance violations).
    - Pre-defined difficulty levels (easy 31.0%, medium 45.2%, hard 23.8%).
    """
    splits = ["train", "train", "train", "train", "validation", "test"]

    terrain_fams = list(SUPPORTED_TERRAIN_FAMILIES)
    texture_fams = list(SUPPORTED_TEXTURE_FAMILIES)
    radar_profs = list(RADAR_PLACEMENT_PROFILES_V2.keys())
    route_fams = list(ROUTE_FAMILIES)

    group_definitions = {
        "dataset_single_aircraft_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise"],
            "radar_cat": lambda i, pk: "high_altitude_aircraft",
        },
        "dataset_single_uav_agile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_agile"],
            "radar_cat": lambda i, pk: "low_altitude_uav",
        },
        "dataset_single_uav_drift_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_drift"],
            "radar_cat": lambda i, pk: "slow_drift_uav",
        },
        "dataset_single_missile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["missile_dash"],
            "radar_cat": lambda i, pk: "fast_missile",
        },
        "dataset_triple_all_same_v1": {
            "comp_profile": "all_same",
            "comp_physical": "all_same",
            "targets_builder": lambda i, r, rng: (
                ["aircraft_cruise", "aircraft_cruise", "aircraft_cruise"] if (i % 3 == 0) else
                (["uav_agile", "uav_agile", "uav_agile"] if (i % 3 == 1) else
                 ["missile_dash", "missile_dash", "missile_dash"])
            ),
            "radar_cat": lambda i, pk: (
                "high_altitude_aircraft" if pk[0] == "aircraft_cruise" else
                ("low_altitude_uav" if pk[0] == "uav_agile" else "fast_missile")
            ),
        },
        "dataset_triple_all_different_v1": {
            "comp_profile": "all_different",
            "comp_physical": "all_different",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise", "uav_agile", "missile_dash"],
            "radar_cat": lambda i, pk: "mixed_three_targets",
        },
        "dataset_triple_two_plus_one_v1": {
            "comp_profile": "two_plus_one",
            "comp_physical": "evaluated_per_combo",
            "targets_builder": lambda i, r, rng: (
                ["uav_agile", "uav_agile", "uav_drift"] if (i % 2 == 0) else
                ["aircraft_cruise", "aircraft_cruise", "missile_dash"]
            ),
            "radar_cat": lambda i, pk: (
                "low_altitude_uav" if all("uav" in p for p in pk) else "mixed_three_targets"
            ),
        },
    }

    all_groups_episodes: Dict[str, List[Dict[str, Any]]] = {}
    global_ep_index = 1
    master_seed = 20261116

    for g_idx, (group_name, group_cfg) in enumerate(group_definitions.items()):
        group_episodes = []
        sched = GROUP_DIFFICULTY_SCHEDULE[g_idx]

        for ep_in_group in range(6):
            split = splits[ep_in_group]
            diff = sched[ep_in_group]
            split_salt = {"train": 1000, "validation": 2000, "test": 3000}[split]
            ep_seed = master_seed + split_salt + (g_idx * 50) + ep_in_group
            rng = np.random.default_rng(ep_seed)

            cfg_idx = g_idx * 6 + ep_in_group
            tex_fam = texture_fams[cfg_idx % len(texture_fams)]
            rad_prof_key = radar_profs[cfg_idx % len(radar_profs)]
            route_fam = route_fams[cfg_idx % len(route_fams)]

            # Difficulty-aware terrain and obstacle generation
            if diff == "easy":
                ter_candidates = ["valley_linear_canyon", "basin_highlands"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "clear"
                w_valley = round(float(rng.uniform(3200.0, 3800.0)), 1)
                h_relief = round(float(rng.uniform(650.0, 780.0)), 1)
            elif diff == "medium":
                ter_candidates = ["valley_sinuous_pass", "mountain_chokepoint", "valley_linear_canyon"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "sparse"
                w_valley = round(float(rng.uniform(2600.0, 3100.0)), 1)
                h_relief = round(float(rng.uniform(780.0, 880.0)), 1)
            else:  # hard
                ter_candidates = ["ridge_multi_obstacle", "mountain_chokepoint"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "dense" if (ep_in_group % 2 == 0) else "asymmetric"
                w_valley = round(float(rng.uniform(2100.0, 2500.0)), 1)
                h_relief = round(float(rng.uniform(900.0, 1050.0)), 1)

            ter_seed = ep_seed + 777

            # Determine radar config for this category
            target_profile_keys = group_cfg["targets_builder"](ep_in_group, route_fam, rng)
            cat_key = group_cfg["radar_cat"](ep_in_group, target_profile_keys)
            cat_cfg = SCENARIO_RADAR_CONFIGS[cat_key]

            # Terrain object for ground clearance check
            ter_spec = {
                "source": "procedural",
                "resolution": 256,
                "family": ter_fam,
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "valley_width_m": w_valley,
                "relief_height_m": h_relief,
                "seed": ter_seed,
            }
            dummy_sc = {"terrain": ter_spec, "site": {"lat": 40.30, "lon": -111.70}}
            ter_obj = _cached_terrain(dummy_sc)

            # Build targets with ground-relative altitude
            targets = []
            for t_idx, p_key in enumerate(target_profile_keys):
                t_spec = make_target_spec_v2(ter_obj, p_key, route_fam, t_idx, rng)
                targets.append(t_spec)

            # Physical composition
            phys_types = [TARGET_PROFILES[p]["physical_type"] for p in target_profile_keys]
            if len(phys_types) == 1:
                phys_comp = "single"
            elif len(set(phys_types)) == 1:
                phys_comp = "all_same"
            elif len(set(phys_types)) == len(phys_types):
                phys_comp = "all_different"
            else:
                phys_comp = "two_plus_one"

            profile_comp = group_cfg["comp_profile"]

            disp_name = format_display_name(
                episode_index=global_ep_index,
                terrain_family=ter_fam,
                texture_family=tex_fam,
                target_profile_composition=profile_comp,
                target_profiles=target_profile_keys,
                radar_count=3,
            )

            # Radars with military crest positions and scenario-aware elevation bars
            radars = deepcopy(RADAR_PLACEMENT_PROFILES_V2[rad_prof_key]["radars"])
            for r in radars:
                r["search_sector_deg"] = cat_cfg["search_sector_deg"]
                r["search_bars_deg"] = cat_cfg["search_bars_deg"]

            sc = {
                "name": f"scenario_{group_name}_ep{global_ep_index:04d}",
                "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
                "area_radius_m": 7500,
                "duration_s": 30.0,
                "dt": 0.5,
                "max_track_slots": 3,
                "difficulty_level": diff,
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
            dataset_ep_id = f"ep-{global_ep_index:06d}-" + _digest([master_seed, global_ep_index, split, group_name, config_hash])[:12]

            episode_entry = {
                "dataset_name": group_name,
                "dataset_episode_id": dataset_ep_id,
                "episode_index": global_ep_index,
                "display_name": disp_name,
                "difficulty_level": diff,
                "split": split,
                "scenario_family": group_name,
                "terrain_family": ter_fam,
                "terrain_seed": ter_seed,
                "terrain_group_id": ter_group_id,
                "terrain_parameters": {
                    "valley_width_m": w_valley,
                    "relief_height_m": h_relief,
                },
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "radar_count": 3,
                "radar_placement_profile": rad_prof_key,
                "radar_elevation_bars": cat_cfg["search_bars_deg"],
                "radar_search_sector_deg": cat_cfg["search_sector_deg"],
                "radar_category": cat_key,
                "target_count": len(targets),
                "target_types": phys_types,
                "motion_profiles": [t["motion"]["model"] for t in targets],
                "class_labels": [t["label"] for t in targets],
                "target_profiles": target_profile_keys,
                "physical_target_type_composition": phys_comp,
                "target_profile_composition": profile_comp,
                "route_family": route_fam,
                "route_group_id": route_id,
                "geometry_group_id": "geom-" + _digest([ter_group_id, route_id, validated_sc["radars"]])[:20],
                "master_seed": master_seed,
                "episode_seed": ep_seed,
                "resolved_scenario": validated_sc,
                "config_snapshot": snapshot,
                "resolved_configuration_hash": config_hash,
                "recipe_version": 1,
            }

            group_episodes.append(episode_entry)
            global_ep_index += 1

        all_groups_episodes[group_name] = group_episodes

    return all_groups_episodes


def build_canonical_plan_for_group(group_name: str, episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wraps a list of resolved episodes into a valid synthetic_dataset_service Plan v1 dict."""
    master = episodes[0]["master_seed"]
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for ep in episodes:
        split_counts[ep["split"]] += 1

    dur = episodes[0]["resolved_scenario"].get("duration_s", 15.0)

    return {
        "plan_schema_version": 1,
        "recipe_version": 1,
        "master_seed": master,
        "recipe": {
            "recipe_version": 1,
            "master_seed": master,
            "episodes": len(episodes),
            "base_scenario": "nominal",
            "scenario_family": group_name,
            "split_ratios": {s: round(count / len(episodes), 4) for s, count in split_counts.items()},
            "overrides": {"duration_s": dur},
            "terrain_pools": {
                "train": [{"seed": ep["terrain_seed"], "terrain": ep["resolved_scenario"]["terrain"]}
                          for ep in episodes if ep["split"] == "train"],
                "validation": [{"seed": ep["terrain_seed"], "terrain": ep["resolved_scenario"]["terrain"]}
                               for ep in episodes if ep["split"] == "validation"],
                "test": [{"seed": ep["terrain_seed"], "terrain": ep["resolved_scenario"]["terrain"]}
                         for ep in episodes if ep["split"] == "test"],
            },
        },
        "split_ratios": {s: round(count / len(episodes), 4) for s, count in split_counts.items()},
        "split_counts": split_counts,
        "split_allocation": "prefix-stable largest-deficit allocation",
        "geometry_validation": "initial target clearance and radar/target grid bounds",
        "grouping_policy": "terrain and initial-position/velocity groups disjoint",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    }


def compute_physical_quality_metrics(episode_dir: Path) -> Dict[str, Any]:
    """Loads Canonical Episode Records v2 and computes comprehensive physical and radar metrics."""
    records = load_episode_records(episode_dir)
    meta = records["metadata"]
    states = records["states"]
    truth = records["truth"]
    reports = records["reports"]
    tracks = records["tracks"]
    trans = records["transitions"]
    deliveries = records.get("deliveries", [])

    # Basic simulation completion
    completed = meta.get("completed", False)
    recorded_steps = meta.get("recorded_steps", len(states) - 1)
    duration_s = states[-1]["t_s"] if states else 0.0

    # Counts
    n_states = len(states)
    n_truth = len(truth)
    n_reports = len(reports)
    n_tracks = len(tracks)
    n_deliveries = len(deliveries)

    # Delivered vs lost communication packets
    communication = records.get("communication", [])
    if communication:
        delivered_cnt = sum(1 for c in communication if c.get("final_status") == "delivered")
        lost_cnt = sum(1 for c in communication if c.get("final_status") == "packet_lost")
    elif deliveries:
        delivered_cnt = len(deliveries)
        lost_cnt = max(0, n_reports - delivered_cnt)
    else:
        delivered_cnt = n_reports
        lost_cnt = 0

    # Track classification
    confirmed_cnt = sum(1 for t in tracks if t.get("confirmed", False))
    tentative_cnt = n_tracks - confirmed_cnt
    fused_cnt = sum(1 for t in tracks if t.get("source") == "fused")

    # Target & radar clearance and LOS
    scenario = meta["config_snapshot"]["scenario"]
    terrain = _cached_terrain(scenario)

    # Annotations and target-level detections
    annotations = records.get("annotations", [])
    n_targets = len(scenario["targets"])
    detections_per_target = {t_id: sum(1 for a in annotations if a.get("truth_id") == t_id) for t_id in range(n_targets)}
    zero_detection_targets = sum(1 for t_id in range(n_targets) if detections_per_target[t_id] == 0)
    false_alarms = sum(1 for a in annotations if a.get("is_false_alarm", False))

    min_agl = float("inf")
    mean_agl_acc = 0.0
    for row in truth:
        px, py, pz = row["p"]
        gh = float(terrain.height(px, py))
        agl = pz - gh
        if agl < min_agl:
            min_agl = agl
        mean_agl_acc += agl
    mean_agl = (mean_agl_acc / max(1, n_truth)) if n_truth > 0 else 0.0

    # Radar mast clearance
    rad_clearances = []
    for r in scenario["radars"]:
        rx, ry, rz = r["position"]
        gh = float(terrain.height(rx, ry))
        rad_clearances.append(rz)  # mast height in scenario spec

    min_radar_mast = min(rad_clearances) if rad_clearances else 0.0

    # LOS clear vs blocked calculation across all truth points and radars
    total_los_checks = 0
    clear_los_checks = 0
    rad_positions = [
        (r["position"][0], r["position"][1], float(terrain.height(r["position"][0], r["position"][1])) + r["position"][2])
        for r in scenario["radars"]
    ]

    for row in truth:
        t_pos = row["p"]
        for r_pos in rad_positions:
            total_los_checks += 1
            if terrain.line_of_sight(r_pos, t_pos):
                clear_los_checks += 1

    blocked_los_checks = total_los_checks - clear_los_checks
    los_clear_ratio = (clear_los_checks / max(1, total_los_checks)) if total_los_checks > 0 else 0.0

    # Tracking coverage: fraction of simulation steps with at least 1 confirmed track
    confirmed_steps = set()
    for tr in tracks:
        if tr.get("confirmed", False):
            confirmed_steps.add(tr["step_index"])
    track_coverage = len(confirmed_steps) / max(1, recorded_steps)

    # Basic RMSE between matched truth and tracks
    # We compute position error for matched target IDs
    pos_errors = []
    truth_by_step_id = {(row["step_index"], row["target_id"]): row["p"] for row in truth}
    for tr in tracks:
        step = tr["step_index"]
        tid = tr.get("matched_target_id", -1)
        if (step, tid) in truth_by_step_id:
            truth_p = truth_by_step_id[(step, tid)]
            trk_p = tr["x"][:3]
            err = math.hypot(truth_p[0] - trk_p[0], truth_p[1] - trk_p[1], truth_p[2] - trk_p[2])
            pos_errors.append(err)

    rmse_m = math.sqrt(sum(e * e for e in pos_errors) / len(pos_errors)) if pos_errors else None

    # Assessment: Challenging vs Flawed
    # An episode is physically flawed only if target crashed (min_agl < 25m) or radar underground.
    # High occlusion (low LOS ratio) is physical terrain shielding (challenging).
    is_technically_flawed = (min_agl < 25.0) or (min_radar_mast < 10.0) or (not completed)
    is_challenging = (los_clear_ratio < 0.25) or (track_coverage < 0.40)

    quality_tag = "flawed" if is_technically_flawed else ("challenging" if is_challenging else "nominal")

    reasons = []
    if is_technically_flawed:
        if min_agl < 25.0: reasons.append("Target clearance violation (<25m AGL)")
        if min_radar_mast < 10.0: reasons.append("Radar mast clearance violation (<10m)")
        if not completed: reasons.append("Simulation incomplete")
    elif is_challenging:
        if los_clear_ratio < 0.25: reasons.append(f"High physical terrain masking (LOS clear {los_clear_ratio*100:.1f}% < 25%)")
        if track_coverage < 0.40: reasons.append(f"Sparse track coverage ({track_coverage*100:.1f}% < 40%) due to simulation window and narrow 3° pencil beam raster scan")
    else:
        reasons.append("High continuous LOS visibility and track coverage")

    diff_level = scenario.get("difficulty_level") or meta.get("difficulty_level") or meta.get("config_snapshot", {}).get("scenario", {}).get("difficulty_level", "medium")

    return {
        "completed": completed,
        "difficulty_level": diff_level,
        "recorded_steps": recorded_steps,
        "duration_s": duration_s,
        "truth_rows": n_truth,
        "report_rows": n_reports,
        "track_rows": n_tracks,
        "confirmed_tracks": confirmed_cnt,
        "tentative_tracks": tentative_cnt,
        "fused_tracks": fused_cnt,
        "delivered_reports": delivered_cnt,
        "lost_reports": lost_cnt,
        "false_alarms": false_alarms,
        "detections_per_target": detections_per_target,
        "zero_detection_targets": zero_detection_targets,
        "min_target_agl_m": round(min_agl, 1),
        "mean_target_agl_m": round(mean_agl, 1),
        "min_radar_mast_m": round(min_radar_mast, 1),
        "total_los_checks": total_los_checks,
        "clear_los_checks": clear_los_checks,
        "blocked_los_checks": blocked_los_checks,
        "los_clear_pct": round(los_clear_ratio * 100.0, 1),
        "track_coverage_pct": round(track_coverage * 100.0, 1),
        "position_rmse_m": round(rmse_m, 2) if rmse_m is not None else None,
        "quality_tag": quality_tag,
        "quality_tag_reasons": reasons,
    }


def generate_pilot_dataset(output_root: str | Path = "data/processed/pilot_diverse_v1",
                           policy_name: str = "central_greedy",
                           resume: bool = True) -> Dict[str, Any]:
    """Generates all 7 diverse pilot dataset groups (42 episodes total)."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_groups_episodes = build_diverse_pilot_episodes()
    master_index_entries = []
    groups_results = {}

    for group_name in DATASET_GROUPS:
        episodes = all_groups_episodes[group_name]
        group_dir = output_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        plan = build_canonical_plan_for_group(group_name, episodes)

        manifest = generate_dataset(plan, group_dir, policy_name=policy_name, resume=resume)

        for ep_idx, (episode, entry) in enumerate(zip(plan["episodes"], manifest["episodes"])):
            ep_dir = group_dir / episode["split"] / episode["dataset_episode_id"]
            if entry.get("generation_status") == "completed" and ep_dir.exists():
                quality = compute_physical_quality_metrics(ep_dir)
                preview = {
                    "dataset_name": episode["dataset_name"],
                    "episode_id": episode["dataset_episode_id"],
                    "display_name": episode["display_name"],
                    "split": episode["split"],
                    "scenario_family": episode["scenario_family"],
                    "terrain_family": episode["terrain_family"],
                    "terrain_seed": episode["terrain_seed"],
                    "terrain_group_id": episode["terrain_group_id"],
                    "terrain_parameters": episode["terrain_parameters"],
                    "texture_family": episode["texture_family"],
                    "obstacle_profile": episode["obstacle_profile"],
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
                    "quality_summary": quality,
                }
                entry["episode_preview"] = preview
                entry["quality_metrics"] = quality
                master_index_entries.append(preview)

        # Update manifest.json with the preview & quality additions
        manifest_path = group_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        groups_results[group_name] = manifest

    pilot_summary = {
        "pilot_schema_version": 1,
        "pilot_name": "pilot_diverse_v1",
        "total_episodes": len(master_index_entries),
        "dataset_groups": list(DATASET_GROUPS),
        "split_counts": {
            "train": sum(1 for ep in master_index_entries if ep["split"] == "train"),
            "validation": sum(1 for ep in master_index_entries if ep["split"] == "validation"),
            "test": sum(1 for ep in master_index_entries if ep["split"] == "test"),
        },
        "terrain_family_counts": {
            tf: sum(1 for ep in master_index_entries if ep["terrain_family"] == tf)
            for tf in SUPPORTED_TERRAIN_FAMILIES
        },
        "texture_family_counts": {
            tx: sum(1 for ep in master_index_entries if ep["texture_family"] == tx)
            for tx in SUPPORTED_TEXTURE_FAMILIES
        },
        "obstacle_profile_counts": {
            op: sum(1 for ep in master_index_entries if ep["obstacle_profile"] == op)
            for op in SUPPORTED_OBSTACLE_PROFILES
        },
        "quality_overview": {
            "completed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["completed"]),
            "nominal_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "nominal"),
            "challenging_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "challenging"),
            "flawed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "flawed"),
            "mean_los_clear_pct": round(sum(ep["quality_summary"]["los_clear_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
            "mean_track_coverage_pct": round(sum(ep["quality_summary"]["track_coverage_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
        },
        "episodes": master_index_entries,
    }

    index_path = output_root / "pilot_metadata_index.json"
    index_path.write_text(json.dumps(pilot_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return pilot_summary


def run_pilot_quality_audit(pilot_dir: str | Path = "data/processed/pilot_diverse_v1") -> Dict[str, Any]:
    """Generates a rigorous radar and tracking quality audit report across the 42 pilot episodes.

    Computes:
    - Global overview across all 42 episodes
    - Disaggregated statistics by split (train, validation, test)
    - Disaggregated statistics by dataset group
    - Target-level detection counts & zero-detection targets
    - Detailed explanation of challenging vs nominal classifications
    """
    pilot_dir = Path(pilot_dir).resolve()
    index_file = pilot_dir / "pilot_metadata_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Pilot metadata index not found at {index_file}")

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    episodes_meta = index_data["episodes"]

    detailed_episodes = []
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    by_group: Dict[str, List[Dict[str, Any]]] = {g: [] for g in DATASET_GROUPS}

    for ep in episodes_meta:
        group_name = ep["dataset_name"]
        split = ep["split"]
        ep_id = ep["episode_id"]
        ep_path = pilot_dir / group_name / split / ep_id
        if not ep_path.exists():
            continue

        q = compute_physical_quality_metrics(ep_path)
        ep_record = {
            "episode_id": ep_id,
            "display_name": ep["display_name"],
            "dataset_group": group_name,
            "split": split,
            "terrain_family": ep["terrain_family"],
            "texture_family": ep["texture_family"],
            "obstacle_profile": ep["obstacle_profile"],
            "target_count": ep["target_count"],
            "target_types": ep["target_types"],
            "radar_count": ep["radar_count"],
            "radar_placement": ep["radar_placement_profile"],
            "metrics": q,
        }
        detailed_episodes.append(ep_record)
        by_split[split].append(ep_record)
        by_group[group_name].append(ep_record)

    def _aggregate(records_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records_list:
            return {}
        n = len(records_list)
        return {
            "episode_count": n,
            "completed_count": sum(1 for r in records_list if r["metrics"]["completed"]),
            "nominal_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "nominal"),
            "challenging_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "challenging"),
            "flawed_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "flawed"),
            "total_truth_rows": sum(r["metrics"]["truth_rows"] for r in records_list),
            "total_reports": sum(r["metrics"]["report_rows"] for r in records_list),
            "total_delivered_reports": sum(r["metrics"]["delivered_reports"] for r in records_list),
            "total_lost_reports": sum(r["metrics"]["lost_reports"] for r in records_list),
            "total_false_alarms": sum(r["metrics"]["false_alarms"] for r in records_list),
            "zero_detection_targets_total": sum(r["metrics"]["zero_detection_targets"] for r in records_list),
            "mean_los_clear_pct": round(sum(r["metrics"]["los_clear_pct"] for r in records_list) / n, 1),
            "mean_track_coverage_pct": round(sum(r["metrics"]["track_coverage_pct"] for r in records_list) / n, 1),
            "mean_target_agl_m": round(sum(r["metrics"]["mean_target_agl_m"] for r in records_list) / n, 1),
            "min_target_agl_m": round(min(r["metrics"]["min_target_agl_m"] for r in records_list), 1),
            "min_radar_mast_m": round(min(r["metrics"]["min_radar_mast_m"] for r in records_list), 1),
        }

    audit_report = {
        "audit_schema_version": 1,
        "pilot_name": "pilot_diverse_v1",
        "total_episodes_audited": len(detailed_episodes),
        "overall_summary": _aggregate(detailed_episodes),
        "split_breakdown": {
            split: _aggregate(recs) for split, recs in by_split.items()
        },
        "group_breakdown": {
            group: _aggregate(recs) for group, recs in by_group.items()
        },
        "challenging_classification_rationale": {
            "definition": (
                "An episode is classified as 'challenging' if its physical LOS clearance is below 25% "
                "or tracking coverage is below 40% over the short 15-second simulation window."
            ),
            "physical_causes": [
                "Realistic terrain shadowing: Deep linear canyons and narrow sinuous mountain passes "
                "naturally obstruct radar line-of-sight rays at low and moderate elevation angles.",
                "Pencil-beam raster scan geometry: Radars use narrow 3.0° azimuth × 6.0° elevation beams. "
                "A 120° search sector with 2 elevation bars requires 80 looks to sweep once, but a 15-second episode "
                "only executes 30 looks (37.5% sector coverage). Targets outside the initial 30 looks are not illuminated.",
                "Strict operational truth isolation: Operational radar control does not leak truth positions. "
                "Radars search realistically without cheat coordinates.",
            ],
            "conclusion": (
                "Episodes marked 'challenging' are physically valid, collision-free, and mathematically rigorous. "
                "They represent genuine, non-trivial multi-agent radar resource management challenges."
            ),
        },
        "episodes": detailed_episodes,
    }

    # Save to pilot folder and reports
    audit_path = pilot_dir / "pilot_quality_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = reports_dir / "pilot_quality_audit_report.json"
    report_copy.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return audit_report


def generate_pilot_v2_dataset(output_root: str | Path = "data/processed/pilot_diverse_v2",
                              policy_name: str = "central_greedy",
                              resume: bool = True) -> Dict[str, Any]:
    """Generates all 7 diverse pilot v2 dataset groups (42 episodes total) with Step 2F enhancements."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_groups_episodes = build_diverse_pilot_episodes_v2()
    master_index_entries = []
    groups_results = {}

    for group_name in DATASET_GROUPS:
        episodes = all_groups_episodes[group_name]
        group_dir = output_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        plan = build_canonical_plan_for_group(group_name, episodes)

        manifest = generate_dataset(plan, group_dir, policy_name=policy_name, resume=resume)

        for ep_idx, (episode, entry) in enumerate(zip(plan["episodes"], manifest["episodes"])):
            ep_dir = group_dir / episode["split"] / episode["dataset_episode_id"]
            if entry.get("generation_status") == "completed" and ep_dir.exists():
                quality = compute_physical_quality_metrics(ep_dir)
                preview = {
                    "dataset_name": episode["dataset_name"],
                    "episode_id": episode["dataset_episode_id"],
                    "display_name": episode["display_name"],
                    "difficulty_level": episode["difficulty_level"],
                    "split": episode["split"],
                    "scenario_family": episode["scenario_family"],
                    "terrain_family": episode["terrain_family"],
                    "terrain_seed": episode["terrain_seed"],
                    "terrain_group_id": episode["terrain_group_id"],
                    "terrain_parameters": episode["terrain_parameters"],
                    "texture_family": episode["texture_family"],
                    "obstacle_profile": episode["obstacle_profile"],
                    "radar_count": episode["radar_count"],
                    "radar_placement_profile": episode["radar_placement_profile"],
                    "radar_elevation_bars": episode["radar_elevation_bars"],
                    "radar_search_sector_deg": episode["radar_search_sector_deg"],
                    "radar_category": episode["radar_category"],
                    "target_count": episode["target_count"],
                    "target_types": episode["target_types"],
                    "target_profiles": episode["target_profiles"],
                    "motion_profiles": episode["motion_profiles"],
                    "class_labels": episode["class_labels"],
                    "physical_target_type_composition": episode["physical_target_type_composition"],
                    "target_profile_composition": episode["target_profile_composition"],
                    "route_family": episode["route_family"],
                    "episode_seed": episode["episode_seed"],
                    "quality_summary": quality,
                }
                entry["episode_preview"] = preview
                entry["quality_metrics"] = quality
                master_index_entries.append(preview)

        manifest_path = group_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        groups_results[group_name] = manifest

    pilot_summary = {
        "pilot_schema_version": 2,
        "pilot_name": "pilot_diverse_v2",
        "total_episodes": len(master_index_entries),
        "dataset_groups": list(DATASET_GROUPS),
        "split_counts": {
            "train": sum(1 for ep in master_index_entries if ep["split"] == "train"),
            "validation": sum(1 for ep in master_index_entries if ep["split"] == "validation"),
            "test": sum(1 for ep in master_index_entries if ep["split"] == "test"),
        },
        "difficulty_counts": {
            "easy": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "easy"),
            "medium": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "medium"),
            "hard": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "hard"),
        },
        "terrain_family_counts": {
            tf: sum(1 for ep in master_index_entries if ep["terrain_family"] == tf)
            for tf in SUPPORTED_TERRAIN_FAMILIES
        },
        "texture_family_counts": {
            tx: sum(1 for ep in master_index_entries if ep["texture_family"] == tx)
            for tx in SUPPORTED_TEXTURE_FAMILIES
        },
        "obstacle_profile_counts": {
            op: sum(1 for ep in master_index_entries if ep["obstacle_profile"] == op)
            for op in SUPPORTED_OBSTACLE_PROFILES
        },
        "quality_overview": {
            "completed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["completed"]),
            "nominal_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "nominal"),
            "challenging_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "challenging"),
            "flawed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "flawed"),
            "mean_los_clear_pct": round(sum(ep["quality_summary"]["los_clear_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
            "mean_track_coverage_pct": round(sum(ep["quality_summary"]["track_coverage_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
            "zero_detection_targets_total": sum(ep["quality_summary"]["zero_detection_targets"] for ep in master_index_entries),
        },
        "episodes": master_index_entries,
    }

    index_path = output_root / "pilot_metadata_index.json"
    index_path.write_text(json.dumps(pilot_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return pilot_summary


def run_pilot_v2_quality_audit(pilot_dir: str | Path = "data/processed/pilot_diverse_v2") -> Dict[str, Any]:
    """Generates an audit report for Pilot v2 dataset including difficulty breakdown."""
    pilot_dir = Path(pilot_dir).resolve()
    index_file = pilot_dir / "pilot_metadata_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Pilot metadata index not found at {index_file}")

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    episodes_meta = index_data["episodes"]

    detailed_episodes = []
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    by_diff: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    by_group: Dict[str, List[Dict[str, Any]]] = {g: [] for g in DATASET_GROUPS}

    for ep in episodes_meta:
        group_name = ep["dataset_name"]
        split = ep["split"]
        diff = ep.get("difficulty_level", "medium")
        ep_id = ep["episode_id"]
        ep_path = pilot_dir / group_name / split / ep_id
        if not ep_path.exists():
            continue

        q = compute_physical_quality_metrics(ep_path)
        ep_record = {
            "episode_id": ep_id,
            "display_name": ep["display_name"],
            "difficulty_level": diff,
            "dataset_group": group_name,
            "split": split,
            "terrain_family": ep["terrain_family"],
            "texture_family": ep["texture_family"],
            "obstacle_profile": ep["obstacle_profile"],
            "target_count": ep["target_count"],
            "target_types": ep["target_types"],
            "radar_count": ep["radar_count"],
            "radar_placement": ep["radar_placement_profile"],
            "radar_elevation_bars": ep.get("radar_elevation_bars"),
            "metrics": q,
        }
        detailed_episodes.append(ep_record)
        by_split[split].append(ep_record)
        by_diff.setdefault(diff, []).append(ep_record)
        by_group[group_name].append(ep_record)

    def _aggregate(records_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records_list:
            return {}
        n = len(records_list)
        return {
            "episode_count": n,
            "completed_count": sum(1 for r in records_list if r["metrics"]["completed"]),
            "nominal_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "nominal"),
            "challenging_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "challenging"),
            "flawed_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "flawed"),
            "total_truth_rows": sum(r["metrics"]["truth_rows"] for r in records_list),
            "total_reports": sum(r["metrics"]["report_rows"] for r in records_list),
            "total_delivered_reports": sum(r["metrics"]["delivered_reports"] for r in records_list),
            "total_lost_reports": sum(r["metrics"]["lost_reports"] for r in records_list),
            "total_false_alarms": sum(r["metrics"]["false_alarms"] for r in records_list),
            "zero_detection_targets_total": sum(r["metrics"]["zero_detection_targets"] for r in records_list),
            "mean_los_clear_pct": round(sum(r["metrics"]["los_clear_pct"] for r in records_list) / n, 1),
            "mean_track_coverage_pct": round(sum(r["metrics"]["track_coverage_pct"] for r in records_list) / n, 1),
            "mean_target_agl_m": round(sum(r["metrics"]["mean_target_agl_m"] for r in records_list) / n, 1),
            "min_target_agl_m": round(min(r["metrics"]["min_target_agl_m"] for r in records_list), 1),
            "min_radar_mast_m": round(min(r["metrics"]["min_radar_mast_m"] for r in records_list), 1),
        }

    audit_report = {
        "audit_schema_version": 2,
        "pilot_name": "pilot_diverse_v2",
        "total_episodes_audited": len(detailed_episodes),
        "overall_summary": _aggregate(detailed_episodes),
        "difficulty_breakdown": {
            d: _aggregate(recs) for d, recs in by_diff.items()
        },
        "split_breakdown": {
            split: _aggregate(recs) for split, recs in by_split.items()
        },
        "group_breakdown": {
            group: _aggregate(recs) for group, recs in by_group.items()
        },
        "episodes": detailed_episodes,
    }

    audit_path = pilot_dir / "pilot_v2_quality_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = reports_dir / "pilot_v2_quality_audit_report.json"
    report_copy.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return audit_report


def build_diverse_canary_episodes() -> Dict[str, List[Dict[str, Any]]]:
    """Builds explicit specifications for the 70 Canary episodes (10 per dataset group, Step 3A).

    Guarantees:
    - 70 episodes total (10 per dataset group).
    - 49 train (70.0%), 11 validation (15.7%), 10 test (14.3%).
    - Difficulty balance: 21 easy (30.0%), 28 medium (40.0%), 21 hard (30.0%).
    - Full diversity across all 5 terrain families, 6 texture families, 4 obstacle profiles.
    - Flank/cross-valley radar placements for Aircraft and Missile (eliminating zenith blind cones).
    - Disjoint train/validation/test terrain seeds and route hashes (zero leakage).
    - Ground-relative target altitudes (zero ground collisions) and speed-scaled ingress.
    - 30.0-second episode duration (60 steps).
    """
    terrain_fams = list(SUPPORTED_TERRAIN_FAMILIES)
    texture_fams = list(SUPPORTED_TEXTURE_FAMILIES)
    all_radar_profs = list(RADAR_PLACEMENT_PROFILES_V2.keys())
    flank_radar_profs = ["cross_valley_triangulation", "lateral_flank_staggered", "asymmetric_ridge_chokepoint"]
    route_fams = list(ROUTE_FAMILIES)

    group_definitions = {
        "dataset_single_aircraft_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise"],
            "radar_cat": lambda i, pk: "high_altitude_aircraft",
            "radar_profs": flank_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "validation", "test"],
        },
        "dataset_single_uav_agile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_agile"],
            "radar_cat": lambda i, pk: "low_altitude_uav",
            "radar_profs": all_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "validation", "test"],
        },
        "dataset_single_uav_drift_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_drift"],
            "radar_cat": lambda i, pk: "slow_drift_uav",
            "radar_profs": all_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "validation", "test"],
        },
        "dataset_single_missile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["missile_dash"],
            "radar_cat": lambda i, pk: "fast_missile",
            "radar_profs": flank_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "validation", "test"],
        },
        "dataset_triple_all_same_v1": {
            "comp_profile": "all_same",
            "comp_physical": "all_same",
            "targets_builder": lambda i, r, rng: (
                ["aircraft_cruise", "aircraft_cruise", "aircraft_cruise"] if (i % 3 == 0) else
                (["uav_agile", "uav_agile", "uav_agile"] if (i % 3 == 1) else
                 ["missile_dash", "missile_dash", "missile_dash"])
            ),
            "radar_cat": lambda i, pk: (
                "high_altitude_aircraft" if pk[0] == "aircraft_cruise" else
                ("low_altitude_uav" if pk[0] == "uav_agile" else "fast_missile")
            ),
            "radar_profs": all_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "test", "test"],
        },
        "dataset_triple_all_different_v1": {
            "comp_profile": "all_different",
            "comp_physical": "all_different",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise", "uav_agile", "missile_dash"],
            "radar_cat": lambda i, pk: "mixed_three_targets",
            "radar_profs": all_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "test", "test"],
        },
        "dataset_triple_two_plus_one_v1": {
            "comp_profile": "two_plus_one",
            "comp_physical": "evaluated_per_combo",
            "targets_builder": lambda i, r, rng: (
                ["uav_agile", "uav_agile", "uav_drift"] if (i % 2 == 0) else
                ["aircraft_cruise", "aircraft_cruise", "missile_dash"]
            ),
            "radar_cat": lambda i, pk: (
                "low_altitude_uav" if all("uav" in p for p in pk) else "mixed_three_targets"
            ),
            "radar_profs": all_radar_profs,
            "splits": ["train", "train", "train", "train", "train", "train", "train", "validation", "test", "test"],
        },
    }

    canary_difficulty_schedule = [
        "easy", "easy", "medium", "medium", "medium", "hard", "hard", "easy", "medium", "hard"
    ]

    all_groups_episodes: Dict[str, List[Dict[str, Any]]] = {}
    global_ep_index = 1
    master_seed = 20261201

    for g_idx, (group_name, group_cfg) in enumerate(group_definitions.items()):
        group_episodes = []
        splits = group_cfg["splits"]

        for ep_in_group in range(10):
            split = splits[ep_in_group]
            diff = canary_difficulty_schedule[ep_in_group]
            split_salt = {"train": 1000, "validation": 2000, "test": 3000}[split]
            ep_seed = master_seed + split_salt + (g_idx * 100) + ep_in_group
            rng = np.random.default_rng(ep_seed)

            # Difficulty-aware terrain and obstacle generation
            if diff == "easy":
                ter_candidates = ["valley_linear_canyon", "basin_highlands"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "clear"
                w_valley = round(float(rng.uniform(3200.0, 3800.0)), 1)
                h_relief = round(float(rng.uniform(650.0, 780.0)), 1)
            elif diff == "medium":
                ter_candidates = ["valley_sinuous_pass", "mountain_chokepoint", "valley_linear_canyon"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "sparse"
                w_valley = round(float(rng.uniform(2600.0, 3100.0)), 1)
                h_relief = round(float(rng.uniform(780.0, 880.0)), 1)
            else:  # hard
                ter_candidates = ["ridge_multi_obstacle", "mountain_chokepoint"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "dense" if (ep_in_group % 2 == 0) else "asymmetric"
                w_valley = round(float(rng.uniform(2100.0, 2500.0)), 1)
                h_relief = round(float(rng.uniform(900.0, 1050.0)), 1)

            ter_seed = ep_seed + 777
            tex_fam = texture_fams[(g_idx * 10 + ep_in_group) % len(texture_fams)]
            allowed_radars = group_cfg["radar_profs"]
            rad_prof_key = allowed_radars[ep_in_group % len(allowed_radars)]
            route_fam = route_fams[(g_idx * 10 + ep_in_group) % len(route_fams)]

            # Target profiles & kinematics
            target_profile_keys = group_cfg["targets_builder"](ep_in_group, route_fam, rng)
            cat_key = group_cfg["radar_cat"](ep_in_group, target_profile_keys)
            cat_cfg = SCENARIO_RADAR_CONFIGS[cat_key]

            # Cache terrain for height query
            ter_spec = {
                "source": "procedural",
                "resolution": 256,
                "family": ter_fam,
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "valley_width_m": w_valley,
                "relief_height_m": h_relief,
                "seed": ter_seed,
            }
            dummy_sc = {"terrain": ter_spec, "site": {"lat": 40.30, "lon": -111.70}}
            ter_obj = _cached_terrain(dummy_sc)

            targets = []
            for t_idx, p_key in enumerate(target_profile_keys):
                t_spec = make_target_spec_v2(ter_obj, p_key, route_fam, t_idx, rng)
                targets.append(t_spec)

            phys_types = [TARGET_PROFILES[p]["physical_type"] for p in target_profile_keys]
            if len(phys_types) == 1:
                phys_comp = "single"
            elif len(set(phys_types)) == 1:
                phys_comp = "all_same"
            elif len(set(phys_types)) == len(phys_types):
                phys_comp = "all_different"
            else:
                phys_comp = "two_plus_one"

            profile_comp = group_cfg["comp_profile"]

            disp_name = format_display_name(
                episode_index=global_ep_index,
                terrain_family=ter_fam,
                texture_family=tex_fam,
                target_profile_composition=profile_comp,
                target_profiles=target_profile_keys,
                radar_count=3,
            )

            radars = deepcopy(RADAR_PLACEMENT_PROFILES_V2[rad_prof_key]["radars"])
            for r in radars:
                r["search_sector_deg"] = cat_cfg["search_sector_deg"]
                r["search_bars_deg"] = cat_cfg["search_bars_deg"]

            sc = {
                "name": f"scenario_{group_name}_ep{global_ep_index:04d}",
                "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
                "area_radius_m": 7500,
                "duration_s": 30.0,
                "dt": 0.5,
                "max_track_slots": 3,
                "difficulty_level": diff,
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
            dataset_ep_id = f"ep-{global_ep_index:06d}-" + _digest([master_seed, global_ep_index, split, group_name, config_hash])[:12]

            episode_entry = {
                "dataset_name": group_name,
                "dataset_episode_id": dataset_ep_id,
                "episode_index": global_ep_index,
                "display_name": disp_name,
                "difficulty_level": diff,
                "split": split,
                "scenario_family": group_name,
                "terrain_family": ter_fam,
                "terrain_seed": ter_seed,
                "terrain_group_id": ter_group_id,
                "terrain_parameters": {
                    "valley_width_m": w_valley,
                    "relief_height_m": h_relief,
                },
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "radar_count": 3,
                "radar_placement_profile": rad_prof_key,
                "radar_elevation_bars": cat_cfg["search_bars_deg"],
                "radar_search_sector_deg": cat_cfg["search_sector_deg"],
                "radar_category": cat_key,
                "target_count": len(targets),
                "target_types": phys_types,
                "motion_profiles": [t["motion"]["model"] for t in targets],
                "class_labels": [t["label"] for t in targets],
                "target_profiles": target_profile_keys,
                "physical_target_type_composition": phys_comp,
                "target_profile_composition": profile_comp,
                "route_family": route_fam,
                "route_group_id": route_id,
                "geometry_group_id": "geom-" + _digest([ter_group_id, route_id, validated_sc["radars"]])[:20],
                "master_seed": master_seed,
                "episode_seed": ep_seed,
                "resolved_scenario": validated_sc,
                "config_snapshot": snapshot,
                "resolved_configuration_hash": config_hash,
                "recipe_version": 1,
            }

            group_episodes.append(episode_entry)
            global_ep_index += 1

        all_groups_episodes[group_name] = group_episodes

    return all_groups_episodes


def generate_canary_dataset(output_root: str | Path = "data/processed/canary_diverse_v1",
                            policy_name: str = "central_greedy",
                            resume: bool = True) -> Dict[str, Any]:
    """Generates all 7 Canary dataset groups (70 episodes total, Step 3A)."""
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_groups_episodes = build_diverse_canary_episodes()
    master_index_entries = []
    groups_results = {}

    for group_name in DATASET_GROUPS:
        episodes = all_groups_episodes[group_name]
        group_dir = output_root / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        plan = build_canonical_plan_for_group(group_name, episodes)

        manifest = generate_dataset(plan, group_dir, policy_name=policy_name, resume=resume)

        for ep_idx, (episode, entry) in enumerate(zip(plan["episodes"], manifest["episodes"])):
            ep_dir = group_dir / episode["split"] / episode["dataset_episode_id"]
            if entry.get("generation_status") == "completed" and ep_dir.exists():
                quality = compute_physical_quality_metrics(ep_dir)
                preview = {
                    "dataset_name": episode["dataset_name"],
                    "episode_id": episode["dataset_episode_id"],
                    "display_name": episode["display_name"],
                    "difficulty_level": episode["difficulty_level"],
                    "split": episode["split"],
                    "scenario_family": episode["scenario_family"],
                    "terrain_family": episode["terrain_family"],
                    "terrain_seed": episode["terrain_seed"],
                    "terrain_group_id": episode["terrain_group_id"],
                    "terrain_parameters": episode["terrain_parameters"],
                    "texture_family": episode["texture_family"],
                    "obstacle_profile": episode["obstacle_profile"],
                    "radar_count": episode["radar_count"],
                    "radar_placement_profile": episode["radar_placement_profile"],
                    "radar_elevation_bars": episode["radar_elevation_bars"],
                    "radar_search_sector_deg": episode["radar_search_sector_deg"],
                    "radar_category": episode["radar_category"],
                    "target_count": episode["target_count"],
                    "target_types": episode["target_types"],
                    "target_profiles": episode["target_profiles"],
                    "motion_profiles": episode["motion_profiles"],
                    "class_labels": episode["class_labels"],
                    "physical_target_type_composition": episode["physical_target_type_composition"],
                    "target_profile_composition": episode["target_profile_composition"],
                    "route_family": episode["route_family"],
                    "episode_seed": episode["episode_seed"],
                    "quality_summary": quality,
                }
                entry["episode_preview"] = preview
                entry["quality_metrics"] = quality
                master_index_entries.append(preview)

        manifest_path = group_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        groups_results[group_name] = manifest

    canary_summary = {
        "canary_schema_version": 1,
        "dataset_name": "canary_diverse_v1",
        "total_episodes": len(master_index_entries),
        "dataset_groups": list(DATASET_GROUPS),
        "split_counts": {
            "train": sum(1 for ep in master_index_entries if ep["split"] == "train"),
            "validation": sum(1 for ep in master_index_entries if ep["split"] == "validation"),
            "test": sum(1 for ep in master_index_entries if ep["split"] == "test"),
        },
        "difficulty_counts": {
            "easy": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "easy"),
            "medium": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "medium"),
            "hard": sum(1 for ep in master_index_entries if ep["difficulty_level"] == "hard"),
        },
        "terrain_family_counts": {
            tf: sum(1 for ep in master_index_entries if ep["terrain_family"] == tf)
            for tf in SUPPORTED_TERRAIN_FAMILIES
        },
        "texture_family_counts": {
            tx: sum(1 for ep in master_index_entries if ep["texture_family"] == tx)
            for tx in SUPPORTED_TEXTURE_FAMILIES
        },
        "obstacle_profile_counts": {
            op: sum(1 for ep in master_index_entries if ep["obstacle_profile"] == op)
            for op in SUPPORTED_OBSTACLE_PROFILES
        },
        "quality_overview": {
            "completed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["completed"]),
            "nominal_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "nominal"),
            "challenging_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "challenging"),
            "flawed_episodes": sum(1 for ep in master_index_entries if ep["quality_summary"]["quality_tag"] == "flawed"),
            "mean_los_clear_pct": round(sum(ep["quality_summary"]["los_clear_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
            "mean_track_coverage_pct": round(sum(ep["quality_summary"]["track_coverage_pct"] for ep in master_index_entries) / max(1, len(master_index_entries)), 1),
            "zero_detection_targets_total": sum(ep["quality_summary"]["zero_detection_targets"] for ep in master_index_entries),
        },
        "episodes": master_index_entries,
    }

    index_path = output_root / "canary_metadata_index.json"
    index_path.write_text(json.dumps(canary_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return canary_summary


def run_canary_quality_audit(canary_dir: str | Path = "data/processed/canary_diverse_v1") -> Dict[str, Any]:
    """Generates an audit report for Canary dataset including group, difficulty, and split breakdowns."""
    canary_dir = Path(canary_dir).resolve()
    index_file = canary_dir / "canary_metadata_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"Canary metadata index not found at {index_file}")

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    episodes_meta = index_data["episodes"]

    detailed_episodes = []
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    by_diff: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    by_group: Dict[str, List[Dict[str, Any]]] = {g: [] for g in DATASET_GROUPS}

    for ep in episodes_meta:
        group_name = ep["dataset_name"]
        split = ep["split"]
        diff = ep.get("difficulty_level", "medium")
        ep_id = ep["episode_id"]
        ep_path = canary_dir / group_name / split / ep_id
        if not ep_path.exists():
            continue

        q = compute_physical_quality_metrics(ep_path)
        ep_record = {
            "episode_id": ep_id,
            "display_name": ep["display_name"],
            "difficulty_level": diff,
            "dataset_group": group_name,
            "split": split,
            "terrain_family": ep["terrain_family"],
            "texture_family": ep["texture_family"],
            "obstacle_profile": ep["obstacle_profile"],
            "target_count": ep["target_count"],
            "target_types": ep["target_types"],
            "radar_count": ep["radar_count"],
            "radar_placement": ep["radar_placement_profile"],
            "radar_elevation_bars": ep.get("radar_elevation_bars"),
            "radar_search_sector_deg": ep.get("radar_search_sector_deg"),
            "metrics": q,
        }
        detailed_episodes.append(ep_record)
        by_split[split].append(ep_record)
        by_diff.setdefault(diff, []).append(ep_record)
        by_group[group_name].append(ep_record)

    def _aggregate(records_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records_list:
            return {}
        n = len(records_list)
        pos_rmses = [r["metrics"]["position_rmse_m"] for r in records_list if r["metrics"].get("position_rmse_m") is not None]
        mean_rmse = round(float(np.mean(pos_rmses)), 2) if pos_rmses else None

        return {
            "episode_count": n,
            "completed_count": sum(1 for r in records_list if r["metrics"]["completed"]),
            "nominal_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "nominal"),
            "challenging_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "challenging"),
            "flawed_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "flawed"),
            "total_truth_rows": sum(r["metrics"]["truth_rows"] for r in records_list),
            "total_reports": sum(r["metrics"]["report_rows"] for r in records_list),
            "total_delivered_reports": sum(r["metrics"]["delivered_reports"] for r in records_list),
            "total_lost_reports": sum(r["metrics"]["lost_reports"] for r in records_list),
            "total_false_alarms": sum(r["metrics"]["false_alarms"] for r in records_list),
            "zero_detection_targets_total": sum(r["metrics"]["zero_detection_targets"] for r in records_list),
            "mean_los_clear_pct": round(sum(r["metrics"]["los_clear_pct"] for r in records_list) / n, 1),
            "mean_track_coverage_pct": round(sum(r["metrics"]["track_coverage_pct"] for r in records_list) / n, 1),
            "mean_target_agl_m": round(sum(r["metrics"]["mean_target_agl_m"] for r in records_list) / n, 1),
            "min_target_agl_m": round(min(r["metrics"]["min_target_agl_m"] for r in records_list), 1),
            "min_radar_mast_m": round(min(r["metrics"]["min_radar_mast_m"] for r in records_list), 1),
            "mean_position_rmse_m": mean_rmse,
        }

    audit_report = {
        "audit_schema_version": 2,
        "dataset_name": "canary_diverse_v1",
        "total_episodes_audited": len(detailed_episodes),
        "overall_summary": _aggregate(detailed_episodes),
        "difficulty_breakdown": {
            d: _aggregate(recs) for d, recs in by_diff.items()
        },
        "split_breakdown": {
            split: _aggregate(recs) for split, recs in by_split.items()
        },
        "group_breakdown": {
            group: _aggregate(recs) for group, recs in by_group.items()
        },
        "episodes": detailed_episodes,
    }

    audit_path = canary_dir / "canary_quality_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = reports_dir / "canary_quality_audit_report.json"
    report_copy.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return audit_report


def build_diverse_1000_episodes() -> Dict[str, List[Dict[str, Any]]]:
    """Builds explicit specifications for the full 1000 production episodes (Step 3B).

    Exact Distribution (1000 episodes total):
    - Single Aircraft: 105 Train, 22 Val, 23 Test (150 Total)
    - Single UAV Agile: 105 Train, 22 Val, 23 Test (150 Total)
    - Single UAV Drift: 105 Train, 22 Val, 23 Test (150 Total)
    - Single Missile: 105 Train, 22 Val, 23 Test (150 Total)
    - Triple All Same: 92 Train, 20 Val, 20 Test (132 Total)
    - Triple All Different: 92 Train, 20 Val, 20 Test (132 Total)
    - Triple Two Plus One: 96 Train, 22 Val, 18 Test (136 Total)
    Total: 700 Train (70.0%), 150 Validation (15.0%), 150 Test (15.0%).

    Difficulty Target:
    - Easy: 300 (30.0%)
    - Medium: 450 (45.0%)
    - Hard: 250 (25.0%)
    """
    terrain_fams = list(SUPPORTED_TERRAIN_FAMILIES)
    texture_fams = list(SUPPORTED_TEXTURE_FAMILIES)
    all_radar_profs = list(RADAR_PLACEMENT_PROFILES_V2.keys())
    flank_radar_profs = ["cross_valley_triangulation", "lateral_flank_staggered", "asymmetric_ridge_chokepoint"]
    route_fams = list(ROUTE_FAMILIES)

    group_specs = {
        "dataset_single_aircraft_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise"],
            "radar_cat": lambda i, pk: "high_altitude_aircraft",
            "radar_profs": flank_radar_profs,
            "counts": {"train": (31, 48, 26), "validation": (7, 10, 5), "test": (7, 10, 6)},
        },
        "dataset_single_uav_agile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_agile"],
            "radar_cat": lambda i, pk: "low_altitude_uav",
            "radar_profs": all_radar_profs,
            "counts": {"train": (32, 47, 26), "validation": (7, 10, 5), "test": (6, 11, 6)},
        },
        "dataset_single_uav_drift_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["uav_drift"],
            "radar_cat": lambda i, pk: "slow_drift_uav",
            "radar_profs": all_radar_profs,
            "counts": {"train": (31, 48, 26), "validation": (7, 10, 5), "test": (7, 10, 6)},
        },
        "dataset_single_missile_v1": {
            "comp_profile": "single",
            "comp_physical": "single",
            "targets_builder": lambda i, r, rng: ["missile_dash"],
            "radar_cat": lambda i, pk: "fast_missile",
            "radar_profs": flank_radar_profs,
            "counts": {"train": (31, 47, 27), "validation": (7, 10, 5), "test": (7, 10, 6)},
        },
        "dataset_triple_all_same_v1": {
            "comp_profile": "all_same",
            "comp_physical": "all_same",
            "targets_builder": lambda i, r, rng: (
                ["aircraft_cruise", "aircraft_cruise", "aircraft_cruise"] if (i % 3 == 0) else
                (["uav_agile", "uav_agile", "uav_agile"] if (i % 3 == 1) else
                 ["missile_dash", "missile_dash", "missile_dash"])
            ),
            "radar_cat": lambda i, pk: (
                "high_altitude_aircraft" if pk[0] == "aircraft_cruise" else
                ("low_altitude_uav" if pk[0] == "uav_agile" else "fast_missile")
            ),
            "radar_profs": all_radar_profs,
            "counts": {"train": (28, 41, 23), "validation": (6, 9, 5), "test": (6, 9, 5)},
        },
        "dataset_triple_all_different_v1": {
            "comp_profile": "all_different",
            "comp_physical": "all_different",
            "targets_builder": lambda i, r, rng: ["aircraft_cruise", "uav_agile", "missile_dash"],
            "radar_cat": lambda i, pk: "mixed_three_targets",
            "radar_profs": all_radar_profs,
            "counts": {"train": (28, 41, 23), "validation": (6, 9, 5), "test": (6, 9, 5)},
        },
        "dataset_triple_two_plus_one_v1": {
            "comp_profile": "two_plus_one",
            "comp_physical": "evaluated_per_combo",
            "targets_builder": lambda i, r, rng: (
                ["uav_agile", "uav_agile", "uav_drift"] if (i % 2 == 0) else
                ["aircraft_cruise", "aircraft_cruise", "missile_dash"]
            ),
            "radar_cat": lambda i, pk: (
                "low_altitude_uav" if all("uav" in p for p in pk) else "mixed_three_targets"
            ),
            "radar_profs": all_radar_profs,
            "counts": {"train": (29, 43, 24), "validation": (6, 10, 6), "test": (5, 8, 5)},
        },
    }

    all_groups_episodes: Dict[str, List[Dict[str, Any]]] = {}
    global_ep_index = 1
    master_seed = 20261300

    for g_idx, (group_name, group_cfg) in enumerate(group_specs.items()):
        group_episodes = []
        counts = group_cfg["counts"]

        # Build list of (split, difficulty) pairs in structured interleaved order
        split_diff_pairs = []
        for split in ("train", "validation", "test"):
            e_cnt, m_cnt, h_cnt = counts[split]
            sub = ["easy"] * e_cnt + ["medium"] * m_cnt + ["hard"] * h_cnt
            perm_rng = np.random.default_rng(master_seed + g_idx * 17 + len(split))
            perm_indices = perm_rng.permutation(len(sub))
            shuffled_sub = [sub[idx] for idx in perm_indices]
            for diff in shuffled_sub:
                split_diff_pairs.append((split, diff))

        for ep_in_group, (split, diff) in enumerate(split_diff_pairs):
            split_salt = {"train": 100000, "validation": 200000, "test": 300000}[split]
            ep_seed = master_seed + split_salt + (g_idx * 1000) + ep_in_group
            rng = np.random.default_rng(ep_seed)

            # Difficulty-aware terrain and obstacle parameters
            if diff == "easy":
                ter_candidates = ["valley_linear_canyon", "basin_highlands"]
                ter_fam = ter_candidates[(g_idx * 11 + ep_in_group) % len(ter_candidates)]
                obs_prof = "clear"
                w_valley = round(float(rng.uniform(3200.0, 3800.0)), 1)
                h_relief = round(float(rng.uniform(650.0, 780.0)), 1)
            elif diff == "medium":
                ter_candidates = ["valley_sinuous_pass", "mountain_chokepoint", "valley_linear_canyon"]
                ter_fam = ter_candidates[(g_idx * 11 + ep_in_group) % len(ter_candidates)]
                obs_prof = "sparse"
                w_valley = round(float(rng.uniform(2600.0, 3100.0)), 1)
                h_relief = round(float(rng.uniform(780.0, 880.0)), 1)
            else:  # hard
                ter_candidates = ["ridge_multi_obstacle", "mountain_chokepoint"]
                ter_fam = ter_candidates[(g_idx * 11 + ep_in_group) % len(ter_candidates)]
                obs_prof = "dense" if (ep_in_group % 2 == 0) else "asymmetric"
                w_valley = round(float(rng.uniform(2100.0, 2500.0)), 1)
                h_relief = round(float(rng.uniform(900.0, 1050.0)), 1)

            ter_seed = ep_seed + 777
            tex_fam = texture_fams[(g_idx * 13 + ep_in_group) % len(texture_fams)]
            allowed_radars = group_cfg["radar_profs"]
            rad_prof_key = allowed_radars[(ep_in_group) % len(allowed_radars)]
            route_fam = route_fams[(g_idx * 7 + ep_in_group) % len(route_fams)]

            # Target profiles & kinematics
            target_profile_keys = group_cfg["targets_builder"](ep_in_group, route_fam, rng)
            cat_key = group_cfg["radar_cat"](ep_in_group, target_profile_keys)
            cat_cfg = SCENARIO_RADAR_CONFIGS[cat_key]

            # Cache terrain for height query
            ter_spec = {
                "source": "procedural",
                "resolution": 256,
                "family": ter_fam,
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "valley_width_m": w_valley,
                "relief_height_m": h_relief,
                "seed": ter_seed,
            }
            dummy_sc = {"terrain": ter_spec, "site": {"lat": 40.30, "lon": -111.70}}
            ter_obj = _cached_terrain(dummy_sc)

            targets = []
            for t_idx, p_key in enumerate(target_profile_keys):
                t_spec = make_target_spec_v2(ter_obj, p_key, route_fam, t_idx, rng)
                targets.append(t_spec)

            phys_types = [TARGET_PROFILES[p]["physical_type"] for p in target_profile_keys]
            if len(phys_types) == 1:
                phys_comp = "single"
            elif len(set(phys_types)) == 1:
                phys_comp = "all_same"
            elif len(set(phys_types)) == len(phys_types):
                phys_comp = "all_different"
            else:
                phys_comp = "two_plus_one"

            profile_comp = group_cfg["comp_profile"]

            disp_name = format_display_name(
                episode_index=global_ep_index,
                terrain_family=ter_fam,
                texture_family=tex_fam,
                target_profile_composition=profile_comp,
                target_profiles=target_profile_keys,
                radar_count=3,
            )

            radars = deepcopy(RADAR_PLACEMENT_PROFILES_V2[rad_prof_key]["radars"])
            for r in radars:
                r["search_sector_deg"] = cat_cfg["search_sector_deg"]
                r["search_bars_deg"] = cat_cfg["search_bars_deg"]

            sc = {
                "name": f"scenario_{group_name}_ep{global_ep_index:04d}",
                "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
                "area_radius_m": 7500,
                "duration_s": 30.0,
                "dt": 0.5,
                "max_track_slots": 3,
                "difficulty_level": diff,
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
            dataset_ep_id = f"ep-{global_ep_index:06d}-" + _digest([master_seed, global_ep_index, split, group_name, config_hash])[:12]

            episode_entry = {
                "dataset_name": group_name,
                "dataset_episode_id": dataset_ep_id,
                "episode_index": global_ep_index,
                "display_name": disp_name,
                "difficulty_level": diff,
                "split": split,
                "scenario_family": group_name,
                "terrain_family": ter_fam,
                "terrain_seed": ter_seed,
                "terrain_group_id": ter_group_id,
                "terrain_parameters": {
                    "valley_width_m": w_valley,
                    "relief_height_m": h_relief,
                },
                "texture_family": tex_fam,
                "obstacle_profile": obs_prof,
                "radar_count": 3,
                "radar_placement_profile": rad_prof_key,
                "radar_elevation_bars": cat_cfg["search_bars_deg"],
                "radar_search_sector_deg": cat_cfg["search_sector_deg"],
                "radar_category": cat_key,
                "target_count": len(targets),
                "target_types": phys_types,
                "motion_profiles": [t["motion"]["model"] for t in targets],
                "class_labels": [t["label"] for t in targets],
                "target_profiles": target_profile_keys,
                "physical_target_type_composition": phys_comp,
                "target_profile_composition": profile_comp,
                "route_family": route_fam,
                "route_group_id": route_id,
                "geometry_group_id": "geom-" + _digest([ter_group_id, route_id, validated_sc["radars"]])[:20],
                "master_seed": master_seed,
                "episode_seed": ep_seed,
                "resolved_scenario": validated_sc,
                "config_snapshot": snapshot,
                "resolved_configuration_hash": config_hash,
                "recipe_version": 1,
            }

            group_episodes.append(episode_entry)
            global_ep_index += 1

        all_groups_episodes[group_name] = group_episodes

    return all_groups_episodes


def run_diverse_1000_quality_audit(dataset_dir: str | Path = "data/processed/diverse_1000_v1") -> Dict[str, Any]:
    """Generates an audit report for diverse_1000_v1 dataset including group, difficulty, and split breakdowns."""
    dataset_dir = Path(dataset_dir).resolve()
    index_file = dataset_dir / "diverse_1000_metadata_index.json"
    if not index_file.exists():
        index_file = dataset_dir / "metadata_index.json"
    if not index_file.exists():
        raise FileNotFoundError(f"1000-episode metadata index not found at {dataset_dir}")

    index_data = json.loads(index_file.read_text(encoding="utf-8"))
    episodes_meta = index_data["episodes"]

    detailed_episodes = []
    by_split: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    by_diff: Dict[str, List[Dict[str, Any]]] = {"easy": [], "medium": [], "hard": []}
    by_group: Dict[str, List[Dict[str, Any]]] = {g: [] for g in DATASET_GROUPS}

    for ep in episodes_meta:
        group_name = ep["dataset_name"]
        split = ep["split"]
        diff = ep.get("difficulty_level", "medium")
        ep_id = ep["episode_id"]
        ep_path = dataset_dir / group_name / split / ep_id
        if not ep_path.exists():
            continue

        q = compute_physical_quality_metrics(ep_path)
        ep_record = {
            "episode_id": ep_id,
            "display_name": ep["display_name"],
            "difficulty_level": diff,
            "dataset_group": group_name,
            "split": split,
            "terrain_family": ep["terrain_family"],
            "texture_family": ep["texture_family"],
            "obstacle_profile": ep["obstacle_profile"],
            "target_count": ep["target_count"],
            "target_types": ep["target_types"],
            "radar_count": ep["radar_count"],
            "radar_placement": ep["radar_placement_profile"],
            "radar_elevation_bars": ep.get("radar_elevation_bars"),
            "radar_search_sector_deg": ep.get("radar_search_sector_deg"),
            "metrics": q,
        }
        detailed_episodes.append(ep_record)
        by_split[split].append(ep_record)
        by_diff.setdefault(diff, []).append(ep_record)
        by_group[group_name].append(ep_record)

    def _aggregate(records_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records_list:
            return {}
        n = len(records_list)
        pos_rmses = [r["metrics"]["position_rmse_m"] for r in records_list if r["metrics"].get("position_rmse_m") is not None]
        mean_rmse = round(float(np.mean(pos_rmses)), 2) if pos_rmses else None

        return {
            "episode_count": n,
            "completed_count": sum(1 for r in records_list if r["metrics"]["completed"]),
            "nominal_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "nominal"),
            "challenging_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "challenging"),
            "flawed_count": sum(1 for r in records_list if r["metrics"]["quality_tag"] == "flawed"),
            "total_truth_rows": sum(r["metrics"]["truth_rows"] for r in records_list),
            "total_reports": sum(r["metrics"]["report_rows"] for r in records_list),
            "total_delivered_reports": sum(r["metrics"]["delivered_reports"] for r in records_list),
            "total_lost_reports": sum(r["metrics"]["lost_reports"] for r in records_list),
            "total_false_alarms": sum(r["metrics"]["false_alarms"] for r in records_list),
            "zero_detection_targets_total": sum(r["metrics"]["zero_detection_targets"] for r in records_list),
            "mean_los_clear_pct": round(sum(r["metrics"]["los_clear_pct"] for r in records_list) / n, 1),
            "mean_track_coverage_pct": round(sum(r["metrics"]["track_coverage_pct"] for r in records_list) / n, 1),
            "mean_target_agl_m": round(sum(r["metrics"]["mean_target_agl_m"] for r in records_list) / n, 1),
            "min_target_agl_m": round(min(r["metrics"]["min_target_agl_m"] for r in records_list), 1),
            "min_radar_mast_m": round(min(r["metrics"]["min_radar_mast_m"] for r in records_list), 1),
            "mean_position_rmse_m": mean_rmse,
        }

    audit_report = {
        "audit_schema_version": 2,
        "dataset_name": "diverse_1000_v1",
        "total_episodes_audited": len(detailed_episodes),
        "overall_summary": _aggregate(detailed_episodes),
        "difficulty_breakdown": {
            d: _aggregate(recs) for d, recs in by_diff.items()
        },
        "split_breakdown": {
            split: _aggregate(recs) for split, recs in by_split.items()
        },
        "group_breakdown": {
            group: _aggregate(recs) for group, recs in by_group.items()
        },
        "episodes": detailed_episodes,
    }

    audit_path = dataset_dir / "diverse_1000_quality_audit.json"
    audit_path.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_copy = reports_dir / "diverse_1000_quality_audit_report.json"
    report_copy.write_text(json.dumps(audit_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return audit_report
