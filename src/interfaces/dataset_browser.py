"""Dataset Replay Episode Browser, Filtering & Preview Service for GP-MARL-RadNet.

Provides lightweight metadata indexing, multi-criteria filtering, instant episode preview,
and Canonical Episode Records v2 loading for the target-class datasets and legacy planner datasets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

DATASET_OPTIONS = {
    "missile_salvo_50 (Missile salvos: dives, jinks, pop-up, weave; time-on-target - 50 episodes)": {
        "kind": "pilot_group",
        "group_name": "missile_salvo_50",
        "relative_dir": "data/processed/missile_salvo_50",
    },
    "missile_salvo_pilot5 (Missile salvo pilot - 5 episodes)": {
        "kind": "pilot_group",
        "group_name": "missile_salvo_pilot5",
        "relative_dir": "data/processed/missile_salvo_pilot5",
    },
    "missile_routes_pilot5 (Multi-missile routes pilot - 5 episodes)": {
        "kind": "pilot_group",
        "group_name": "missile_routes_pilot5",
        "relative_dir": "data/processed/missile_routes_pilot5",
    },
    "missile_dive_pilot10 (Missile dives and evasive manoeuvres - 10 episodes)": {
        "kind": "pilot_group",
        "group_name": "missile_dive_pilot10",
        "relative_dir": "data/processed/missile_dive_pilot10",
    },
    "class_targets_pilot5 (UAV / Aircraft / Missile - 5 episodes)": {
        "kind": "pilot_group",
        "group_name": "class_targets_pilot5",
        "relative_dir": "data/processed/class_targets_pilot5",
    },
    "multi_target_v1 (legacy)": {
        "kind": "legacy",
        "group_name": "multi_target_v1",
        "relative_dir": "data/processed/multi_target_v1",
    },
    "synthetic_100 (legacy)": {
        "kind": "legacy",
        "group_name": "synthetic_100",
        "relative_dir": "data/processed/synthetic_100",
    },
}

_CATALOG_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def load_dataset_catalog(root_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Returns available datasets present on disk."""
    available = {}
    for name, spec in DATASET_OPTIONS.items():
        p = root_dir / spec["relative_dir"]
        if p.exists() and (p.is_dir() or p.with_suffix(".json").exists()):
            available[name] = spec
    return available


def load_dataset_episodes(dataset_key: str, root_dir: Path) -> List[Dict[str, Any]]:
    """Loads lightweight metadata records for a dataset without reading NPZ files."""
    if dataset_key in _CATALOG_CACHE:
        return _CATALOG_CACHE[dataset_key]

    spec = DATASET_OPTIONS.get(dataset_key)
    if not spec:
        return []

    target_dir = root_dir / spec["relative_dir"]
    episodes: List[Dict[str, Any]] = []

    if spec["kind"] == "pilot_all":
        idx_name = spec.get("index_file", "pilot_metadata_index.json")
        index_file = target_dir / idx_name
        if not index_file.exists():
            index_file = target_dir / "pilot_metadata_index.json"
        if not index_file.exists():
            index_file = target_dir / "canary_metadata_index.json"
        if index_file.exists():
            data = json.loads(index_file.read_text(encoding="utf-8"))
            for ep in data.get("episodes", []):
                rec = dict(ep)
                rec["full_path"] = target_dir / rec["dataset_name"] / rec["split"] / rec["episode_id"]
                episodes.append(rec)

    elif spec["kind"] == "pilot_group":
        manifest_file = target_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            for entry in manifest.get("episodes", []):
                preview = entry.get("episode_preview")
                if preview:
                    rec = dict(preview)
                else:
                    rec = {
                        "dataset_name": spec["group_name"],
                        "episode_id": entry["dataset_episode_id"],
                        "display_name": f"{entry['dataset_episode_id']} · {spec['group_name']}",
                        "split": entry["split"],
                        "scenario_family": entry.get("scenario_family", spec["group_name"]),
                        "terrain_family": entry.get("terrain_family", "Unknown"),
                        "texture_family": entry.get("texture_family", "Unknown"),
                        "obstacle_profile": entry.get("obstacle_profile", "Unknown"),
                        "radar_count": 3,
                        "radar_placement_profile": "Unknown",
                        "target_count": "Unknown",
                        "target_types": ["Unknown"],
                        "target_profiles": ["Unknown"],
                        "motion_profiles": ["Unknown"],
                        "class_labels": ["Unknown"],
                        "physical_target_type_composition": "Unknown",
                        "target_profile_composition": "Unknown",
                        "route_family": "Unknown",
                        "episode_seed": entry.get("episode_seed", "Unknown"),
                        "terrain_seed": entry.get("terrain_seed", "Unknown"),
                        "quality_summary": entry.get("quality_metrics", {}),
                    }
                rec["full_path"] = target_dir / rec["split"] / rec["episode_id"]
                episodes.append(rec)

    elif spec["kind"] == "legacy":
        manifest_file = target_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            for entry in manifest.get("episodes", []):
                ep_id = entry.get("dataset_episode_id", f"ep-{entry.get('episode_index', 0):06d}")
                split = entry.get("split", "train")
                scen_fam = entry.get("scenario_family", spec["group_name"])
                idx = entry.get("episode_index", 0)

                rec = {
                    "dataset_name": spec["group_name"],
                    "episode_id": ep_id,
                    "display_name": f"EP-{idx:04d} · {scen_fam} (Legacy)",
                    "split": split,
                    "scenario_family": scen_fam,
                    "terrain_family": "Unknown",
                    "texture_family": "Unknown",
                    "obstacle_profile": "Unknown",
                    "radar_count": "Unknown",
                    "radar_placement_profile": "Unknown",
                    "target_count": "Unknown",
                    "target_types": ["Unknown"],
                    "target_profiles": ["Unknown"],
                    "motion_profiles": ["Unknown"],
                    "class_labels": ["Unknown"],
                    "physical_target_type_composition": "Unknown",
                    "target_profile_composition": "Unknown",
                    "route_family": "Unknown",
                    "episode_seed": entry.get("episode_seed", "Unknown"),
                    "terrain_seed": "Unknown",
                    "quality_summary": {},
                    "full_path": target_dir / split / ep_id,
                }
                episodes.append(rec)

    _CATALOG_CACHE[dataset_key] = episodes
    return episodes


def filter_episodes(
    episodes: List[Dict[str, Any]],
    search_text: str = "",
    split: str = "All",
    difficulty: str = "All",
    terrain_family: str = "All",
    texture_family: str = "All",
    obstacle_profile: str = "All",
    target_count: str = "All",
    target_profile_composition: str = "All",
    physical_composition: str = "All",
    target_type: str = "All",
    radar_count: str = "All",
) -> List[Dict[str, Any]]:
    """Applies multi-criteria reactive filtering over episode metadata dictionaries."""
    filtered = []
    search_lower = search_text.strip().lower()

    for ep in episodes:
        # Search text on display name or episode ID
        if search_lower:
            name_match = search_lower in ep.get("display_name", "").lower()
            id_match = search_lower in ep.get("episode_id", "").lower()
            if not (name_match or id_match):
                continue

        # Split filter
        if split != "All" and ep.get("split") != split:
            continue

        # Difficulty filter
        if difficulty != "All" and ep.get("difficulty_level") != difficulty:
            continue

        # Terrain family filter
        if terrain_family != "All" and ep.get("terrain_family") != terrain_family:
            continue

        # Texture family filter
        if texture_family != "All" and ep.get("texture_family") != texture_family:
            continue

        # Obstacle profile filter
        if obstacle_profile != "All" and ep.get("obstacle_profile") != obstacle_profile:
            continue

        # Target count filter
        if target_count != "All" and str(ep.get("target_count")) != str(target_count):
            continue

        # Target profile composition filter
        if target_profile_composition != "All" and ep.get("target_profile_composition") != target_profile_composition:
            continue

        # Physical composition filter
        if physical_composition != "All" and ep.get("physical_target_type_composition") != physical_composition:
            continue

        # Target type filter
        if target_type != "All":
            types = ep.get("target_types", [])
            if target_type not in types:
                continue

        # Radar count filter
        if radar_count != "All" and str(ep.get("radar_count")) != str(radar_count):
            continue

        filtered.append(ep)

    return filtered


def render_episode_preview_card(ep_meta: Dict[str, Any]):
    """Renders the instant metadata preview panel without reading NPZ files."""
    st.markdown(f"##### 📋 Episode Preview: `{ep_meta.get('display_name', 'Episode')}`")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"• **Episode ID:** `{ep_meta.get('episode_id', 'Unknown')}`")
        st.markdown(f"• **Dataset Group:** `{ep_meta.get('dataset_name', 'Unknown')}`")
        diff_val = ep_meta.get('difficulty_level') or 'Unknown'
        diff_icon = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(str(diff_val).lower(), "⚪")
        st.markdown(f"• **Difficulty Level:** {diff_icon} `{diff_val}`")
        st.markdown(f"• **Split:** `{ep_meta.get('split', 'Unknown')}`")
        st.markdown(f"• **Terrain Family:** `{ep_meta.get('terrain_family', 'Unknown')}`")
        st.markdown(f"• **Texture Family:** `{ep_meta.get('texture_family', 'Unknown')}`")
        st.markdown(f"• **Obstacle Profile:** `{ep_meta.get('obstacle_profile', 'Unknown')}`")
        st.markdown(f"• **Radar Count:** `{ep_meta.get('radar_count', 'Unknown')}`")
        st.markdown(f"• **Radar Placement:** `{ep_meta.get('radar_placement_profile', 'Unknown')}`")
        st.markdown(f"• **Terrain Seed:** `{ep_meta.get('terrain_seed', 'Unknown')}`")
    with c2:
        st.markdown(f"• **Target Count:** `{ep_meta.get('target_count', 'Unknown')}`")
        types_str = ", ".join(str(t) for t in ep_meta.get("target_types", ["Unknown"]))
        st.markdown(f"• **Target Types:** `{types_str}`")
        profs_str = ", ".join(str(p) for p in ep_meta.get("target_profiles", ["Unknown"]))
        st.markdown(f"• **Target Profiles:** `{profs_str}`")
        motions_str = ", ".join(str(m) for m in ep_meta.get("motion_profiles", ["Unknown"]))
        st.markdown(f"• **Motion Models:** `{motions_str}`")
        st.markdown(f"• **Profile Composition:** `{ep_meta.get('target_profile_composition', 'Unknown')}`")
        st.markdown(f"• **Physical Composition:** `{ep_meta.get('physical_target_type_composition', 'Unknown')}`")
        st.markdown(f"• **Route Family:** `{ep_meta.get('route_family', 'Unknown')}`")
        st.markdown(f"• **Episode Seed:** `{ep_meta.get('episode_seed', 'Unknown')}`")

    # Physical vs Visual & Quality Breakdown
    q = ep_meta.get("quality_summary") or {}
    los_val = f"{q['los_clear_pct']:.1f}%" if "los_clear_pct" in q else "Unknown"
    tag_val = q.get("quality_tag", "Unknown")
    cov_val = f"{q['track_coverage_pct']:.1f}%" if "track_coverage_pct" in q else "Unknown"
    rmse_val = f"{q['position_rmse_m']:.2f} m" if q.get("position_rmse_m") is not None else "Unknown"
    zero_tgts = q.get("zero_detection_targets", "Unknown")

    st.info(
        f"🏔️ **Physical vs Visual Separation:**  \n"
        f"• **Physical Obstacle:** Radar line-of-sight (LOS) occlusion is computed strictly from the `Terrain.H` elevation grid.  \n"
        f"• **Visual Decoration:** 3D trees, rocks, and shrubs are purely decorative Three.js meshes; they introduce no artificial RF attenuation.  \n"
        f"• **Difficulty & Quality:** Difficulty=`{diff_val}` · Quality=`{tag_val}` · LOS=`{los_val}` · Track Coverage=`{cov_val}` · Zero-Detection Targets=`{zero_tgts}`"
    )


def render_dataset_replay_ui(root_dir: Path, sim_service):
    """Renders the comprehensive Dataset Replay browser, filter panel, instant preview card,

    and handles loading Canonical Episode Records v2 into session state for full visualization.
    """
    st.markdown("##### A. Episode Replay Selection & Exploration")

    catalog = load_dataset_catalog(root_dir)
    if not catalog:
        st.warning(f"No valid dataset found in data/processed/ ({root_dir / 'data' / 'processed'}).")
        return

    cat_names = list(catalog.keys())
    sel_ds_name = st.selectbox(
        "Dataset",
        cat_names,
        index=0,
        key="rep_dataset_sel",
        help="Datasets present under data/processed/."
    )

    FILTER_KEYS = [
        "rep_f_search",
        "rep_f_split",
        "rep_f_diff",
        "rep_f_rcount",
        "rep_f_tcount",
        "rep_f_terrain",
        "rep_f_texture",
        "rep_f_obstacle",
        "rep_f_ttype",
        "rep_f_pcomp",
        "rep_f_phycomp",
    ]

    # Auto-reset filters when switching dataset to avoid silent filter trapping
    prev_ds = st.session_state.get("_last_rep_dataset_sel")
    if prev_ds is not None and prev_ds != sel_ds_name:
        for k in FILTER_KEYS:
            if k in st.session_state:
                del st.session_state[k]
    st.session_state["_last_rep_dataset_sel"] = sel_ds_name

    all_eps = load_dataset_episodes(sel_ds_name, root_dir)
    if not all_eps:
        st.warning(f"No episode records found in selected dataset: {sel_ds_name}")
        return

    with st.expander("🔍 Multi-Criteria Episode Filtering & Search", expanded=True):
        f_top_c1, f_top_c2 = st.columns([4, 1])
        f_top_c1.caption("Narrow down criteria to filter episodes. Use the Reset Filters button to clear all selections.")
        if f_top_c2.button("🔄 Reset Filters", key="rep_reset_filters_btn", help="Clear all filters and display all episodes"):
            for k in FILTER_KEYS:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()

        f_c1, f_c2, f_c3, f_c4, f_c5 = st.columns([3, 2, 2, 2, 2])
        search_txt = f_c1.text_input(
            "Text Search (ID / Name)",
            value="",
            key="rep_f_search",
            placeholder="e.g. Canyon, Agile, 0005"
        )

        splits = ["All"] + sorted(list({str(e.get("split")) for e in all_eps if e.get("split")}))
        if st.session_state.get("rep_f_split") not in splits:
            st.session_state["rep_f_split"] = "All"
        sel_split = f_c2.selectbox("Split", splits, key="rep_f_split")

        diffs = ["All"] + sorted(list({str(e.get("difficulty_level")) for e in all_eps if e.get("difficulty_level")}))
        if st.session_state.get("rep_f_diff") not in diffs:
            st.session_state["rep_f_diff"] = "All"
        sel_diff = f_c3.selectbox("Difficulty Level", diffs, key="rep_f_diff")

        radar_counts = ["All"] + sorted(list({str(e.get("radar_count")) for e in all_eps if e.get("radar_count") is not None}))
        if st.session_state.get("rep_f_rcount") not in radar_counts:
            st.session_state["rep_f_rcount"] = "All"
        sel_rcount = f_c4.selectbox("Radar Count", radar_counts, key="rep_f_rcount")

        tgt_counts = ["All"] + sorted(list({str(e.get("target_count")) for e in all_eps if e.get("target_count") is not None}))
        if st.session_state.get("rep_f_tcount") not in tgt_counts:
            st.session_state["rep_f_tcount"] = "All"
        sel_tcount = f_c5.selectbox("Target Count", tgt_counts, key="rep_f_tcount")

        f_r2_c1, f_r2_c2, f_r2_c3 = st.columns(3)
        terrains = ["All"] + sorted(list({str(e.get("terrain_family")) for e in all_eps if e.get("terrain_family")}))
        if st.session_state.get("rep_f_terrain") not in terrains:
            st.session_state["rep_f_terrain"] = "All"
        sel_terrain = f_r2_c1.selectbox("Terrain Family", terrains, key="rep_f_terrain")

        textures = ["All"] + sorted(list({str(e.get("texture_family")) for e in all_eps if e.get("texture_family")}))
        if st.session_state.get("rep_f_texture") not in textures:
            st.session_state["rep_f_texture"] = "All"
        sel_texture = f_r2_c2.selectbox("Texture Family", textures, key="rep_f_texture")

        obstacles = ["All"] + sorted(list({str(e.get("obstacle_profile")) for e in all_eps if e.get("obstacle_profile")}))
        if st.session_state.get("rep_f_obstacle") not in obstacles:
            st.session_state["rep_f_obstacle"] = "All"
        sel_obstacle = f_r2_c3.selectbox("Obstacle Profile", obstacles, key="rep_f_obstacle")

        f_r3_c1, f_r3_c2, f_r3_c3 = st.columns(3)
        all_types = set()
        for e in all_eps:
            for t in e.get("target_types", []):
                if t:
                    all_types.add(str(t))
        tgt_types = ["All"] + sorted(list(all_types))
        if st.session_state.get("rep_f_ttype") not in tgt_types:
            st.session_state["rep_f_ttype"] = "All"
        sel_ttype = f_r3_c1.selectbox("Target Type", tgt_types, key="rep_f_ttype")

        prof_comps = ["All"] + sorted(list({str(e.get("target_profile_composition")) for e in all_eps if e.get("target_profile_composition")}))
        if st.session_state.get("rep_f_pcomp") not in prof_comps:
            st.session_state["rep_f_pcomp"] = "All"
        sel_pcomp = f_r3_c2.selectbox("Profile Composition", prof_comps, key="rep_f_pcomp")

        phys_comps = ["All"] + sorted(list({str(e.get("physical_target_type_composition")) for e in all_eps if e.get("physical_target_type_composition")}))
        if st.session_state.get("rep_f_phycomp") not in phys_comps:
            st.session_state["rep_f_phycomp"] = "All"
        sel_phycomp = f_r3_c3.selectbox("Physical Composition", phys_comps, key="rep_f_phycomp")

    filtered_eps = filter_episodes(
        all_eps,
        search_text=search_txt,
        split=sel_split,
        difficulty=sel_diff,
        terrain_family=sel_terrain,
        texture_family=sel_texture,
        obstacle_profile=sel_obstacle,
        target_count=sel_tcount,
        target_profile_composition=sel_pcomp,
        physical_composition=sel_phycomp,
        target_type=sel_ttype,
        radar_count=sel_rcount,
    )

    st.markdown(f"**Displayed Episodes:** `{len(filtered_eps)} / {len(all_eps)}` (Total {len(all_eps)} Episodes)")

    if not filtered_eps:
        st.warning("⚠️ No episodes match the selected filter criteria. Please broaden your filters or click Reset Filters.")
        return

    # Episode Selection
    ep_sel_key = f"rep_ep_sel_{sel_ds_name}"
    curr_ep_idx = st.session_state.get(ep_sel_key)
    if curr_ep_idx is not None and (not isinstance(curr_ep_idx, int) or curr_ep_idx >= len(filtered_eps) or curr_ep_idx < 0):
        st.session_state[ep_sel_key] = 0

    selected_idx = st.selectbox(
        "Episode Selection",
        range(len(filtered_eps)),
        format_func=lambda i: filtered_eps[i]["display_name"],
        key=ep_sel_key,
        help="Select an episode to replay by descriptive name or ID."
    )
    selected_ep = filtered_eps[selected_idx]

    # Instant Preview Card
    render_episode_preview_card(selected_ep)

    # Load Button
    b_load, _ = st.columns([1, 4])
    if b_load.button("Load Replay Episode", type="primary", key="rep_load_btn"):
        with st.spinner(f"Loading Canonical Records v2: {selected_ep['display_name']}..."):
            try:
                recs = sim_service.load_episode_records(selected_ep["full_path"])
                payload_conv = sim_service.records_to_viewer_payload(recs)
                st.session_state["scen_records"] = recs
                st.session_state["scen_payload"] = payload_conv
                st.session_state["scen_mode_active"] = "replay"
                st.session_state["scen_ep_name"] = selected_ep["display_name"]
                st.session_state["scen_ep_id"] = selected_ep["episode_id"]
                st.success(
                    f"Episode loaded: **{selected_ep['display_name']}** "
                    f"({len(recs.get('reports', []))} reports, {len(recs.get('tracks', []))} tracks)"
                )
            except Exception as e:
                st.error(f"Replay loading error: {e}")
