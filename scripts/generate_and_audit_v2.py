"""Generate and audit Pilot v2 dataset."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import time
from copy import deepcopy
import numpy as np

from src.core.env import _cached_terrain
from src.core.procedural_terrain import (
    SUPPORTED_TERRAIN_FAMILIES,
    SUPPORTED_TEXTURE_FAMILIES,
    SUPPORTED_OBSTACLE_PROFILES,
)
from src.core.scenario import (
    DEFAULTS,
    scenario_snapshot,
    validate_scenario,
)
from src.services.scenario_planner import _digest, terrain_group_id
from src.services.synthetic_dataset_service import generate_dataset
from src.services.diverse_dataset_service import (
    DATASET_GROUPS,
    TARGET_PROFILES,
    ROUTE_FAMILIES,
    format_display_name,
    compute_physical_quality_metrics,
)

SCENARIO_RADAR_CONFIGS = {
    "high_altitude_aircraft": {
        "search_sector_deg": 120.0,
        "search_bars_deg": [5.0, 16.0, 28.0],
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

GROUP_DIFFICULTY_SCHEDULE = [
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single Aircraft
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single UAV Agile
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single UAV Drift
    ["easy", "easy", "medium", "medium", "medium", "hard"],      # Single Missile
    ["easy", "medium", "medium", "medium", "hard", "hard"],      # Triple All Same
    ["easy", "medium", "medium", "medium", "hard", "hard"],      # Triple All Different
    ["easy", "easy", "easy", "medium", "hard", "hard"],          # Triple Two Plus One
]

def make_target_spec_v2(terrain, profile_key: str, route_family: str, target_idx: int, rng: np.random.Generator):
    prof = TARGET_PROFILES[profile_key]
    phys_type = prof["physical_type"]
    label = prof["legacy_label"]
    v_nom = prof["nominal_speed"]

    lat_offsets = [0.0, -600.0, 600.0, -1200.0, 1200.0]
    lat_off = lat_offsets[target_idx % len(lat_offsets)]

    if profile_key == "aircraft_cruise":
        base_y = -3500.0 - (target_idx * 500.0)
    elif profile_key == "missile_dash":
        base_y = -2400.0 - (target_idx * 500.0)
    elif profile_key == "uav_agile":
        base_y = -1800.0 - (target_idx * 400.0)
    elif profile_key == "uav_drift":
        base_y = -1200.0 - (target_idx * 300.0)
    else:
        base_y = -2200.0 - (target_idx * 400.0)

    if route_family == "corridor_south_to_north":
        x = lat_off + float(rng.uniform(-80.0, 80.0))
        y = base_y
        vx = float(rng.uniform(-2.0, 2.0))
        vy = v_nom * float(rng.uniform(0.97, 1.03))
    elif route_family == "sinuous_pass_transit":
        x = -500.0 + lat_off * 0.4
        y = base_y
        vx = v_nom * 0.20
        vy = v_nom * 0.97
    elif route_family == "chokepoint_penetration":
        x = lat_off * 0.35
        y = base_y
        vx = float(rng.uniform(-3.0, 3.0))
        vy = v_nom
    elif route_family == "mountain_ridge_flank":
        x = 1000.0 + (target_idx * 200.0)
        y = base_y
        vx = -v_nom * 0.08
        vy = v_nom * 0.98
    else:  # transverse_spur_weave
        x = -350.0 + lat_off * 0.4
        y = base_y
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

    motion_spec = {"model": prof["motion_model"]}
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

def build_canonical_plan_for_group_v2(group_name: str, episodes: list):
    master = episodes[0]["master_seed"]
    split_counts = {"train": 0, "validation": 0, "test": 0}
    for ep in episodes:
        split_counts[ep["split"]] += 1

    dur = episodes[0]["resolved_scenario"].get("duration_s", 30.0)

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
            "split_ratios": {"train": 4/6, "validation": 1/6, "test": 1/6},
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
        "split_ratios": {"train": 0.6667, "validation": 0.1667, "test": 0.1667},
        "split_counts": split_counts,
        "split_allocation": "prefix-stable largest-deficit allocation",
        "geometry_validation": "initial target clearance and radar/target grid bounds",
        "grouping_policy": "terrain and initial-position/velocity groups disjoint",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    }

print("Loaded generate_and_audit_v2 harness.")

def build_diverse_pilot_episodes_v2():
    splits = ["train", "train", "train", "train", "validation", "test"]

    terrain_fams = list(SUPPORTED_TERRAIN_FAMILIES)  # 5
    texture_fams = list(SUPPORTED_TEXTURE_FAMILIES)  # 6
    radar_profs = list(RADAR_PLACEMENT_PROFILES_V2.keys())  # 4
    route_fams = list(ROUTE_FAMILIES)  # 5

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

    all_groups_episodes = {}
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
                # Open valley or wide basin, clear obstacle profile
                ter_candidates = ["valley_linear_canyon", "basin_highlands"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "clear"
                w_valley = round(float(rng.uniform(3200.0, 3800.0)), 1)
                h_relief = round(float(rng.uniform(650.0, 780.0)), 1)
            elif diff == "medium":
                # Sinuous pass or chokepoint or linear canyon, sparse obstacles
                ter_candidates = ["valley_sinuous_pass", "mountain_chokepoint", "valley_linear_canyon"]
                ter_fam = ter_candidates[(g_idx + ep_in_group) % len(ter_candidates)]
                obs_prof = "sparse"
                w_valley = round(float(rng.uniform(2600.0, 3100.0)), 1)
                h_relief = round(float(rng.uniform(780.0, 880.0)), 1)
            else:  # hard
                # Multi-obstacle ridge or chokepoint, dense or asymmetric obstacles
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

def run_pilot_v2_generation(output_root="data/processed/pilot_diverse_v2"):
    out_path = Path(output_root).resolve()
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_groups_episodes = build_diverse_pilot_episodes_v2()
    master_index_entries = []
    groups_results = {}

    print(f"\nStarting Pilot v2 generation to {out_path}...")
    for group_name in DATASET_GROUPS:
        episodes = all_groups_episodes[group_name]
        group_dir = out_path / group_name
        group_dir.mkdir(parents=True, exist_ok=True)
        plan = build_canonical_plan_for_group_v2(group_name, episodes)

        print(f"Generating group: {group_name} (6 episodes)...")
        manifest = generate_dataset(plan, group_dir, policy_name="central_greedy", resume=False)

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

    t_elapsed = time.time() - t0
    print(f"\nAll 42 episodes generated in {t_elapsed:.1f}s!")

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

    index_path = out_path / "pilot_metadata_index.json"
    index_path.write_text(json.dumps(pilot_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved master index to {index_path}")
    return pilot_summary

if __name__ == "__main__":
    summary = run_pilot_v2_generation()
    print("Quality overview:")
    print(json.dumps(summary["quality_overview"], indent=2))
    print("Difficulty counts:")
    print(json.dumps(summary["difficulty_counts"], indent=2))
