"""GP-MARL-RadNet dashboard.

    streamlit run src/interfaces/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.core import detection as det
from src.core.evaluation import METRIC_INFO
from src.core.modes import Mode
from src.core.radar import RadarParams, antenna_gain, measurement_sigmas, snr_linear
from src.core.rcs import EllipsoidRCS, LookupTableRCS, m2_to_dbsm
from src.core.scenario import build_radars, list_scenarios, load_scenario, validate_scenario
from src.core.targets import CLASS_NAMES, CLASS_SHORT
from src.interfaces import analysis_views, dataset_browser, plots, viewer3d
from src.models.baselines.policies import LEARNED, POLICIES
from src.services import radar_info
from src.services import simulation_service as sim
from src.services import radar_information_service as r_info_srv
from src.services import radar_selection_service as r_sel_srv
from src.services import episode_analysis_service as ep_an

st.set_page_config(page_title="GP-MARL-RadNet", layout="wide", page_icon="📡")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
[data-testid="stMetricValue"]{font-family:ui-monospace,monospace}
</style>""", unsafe_allow_html=True)
analysis_views.inject_metric_css()

st.title("GP-MARL-RadNet")
st.caption("Cooperative multi-agent radar network · research and educational simulation "
           "(abstract models, synthetic targets, non-operational parameters)")

tab_scen, tab_sim, tab_lab, tab_cls, tab_exp, tab_about = st.tabs(
    ["Scenario Management", "3-D Simulation", "Radar Model Lab", "Classifier", "Experiments", "About"])

POLICY_NAMES = [p for p in POLICIES if p != "random"] + ["random"]
CKPT_DIR = ROOT / "models" / "checkpoints"
TRAINED = [n for n in LEARNED if (CKPT_DIR / f"{n}.pt").exists()]
POLICY_NAMES = TRAINED + POLICY_NAMES
POLICY_HELP = {
    "fixed_scan": "Baseline (sets radar modes): always search.",
    "round_robin": "Baseline (sets radar modes): alternate search / track, cycling through tracks.",
    "nearest": "Baseline (sets radar modes): each radar refreshes its closest track that is due, else searches.",
    "central_greedy": "Baseline (the centre sets radar modes): most uncertain due tracks first, no duplicates, "
                      "spare radars search.",
    "central_greedy_confirm": "Baseline (the centre sets radar modes): central_greedy plus one confirmation dwell "
                              "for every new tentative track.",
    "interleaved_greedy": "Like central_greedy but tracking radars keep searching (INTERLEAVED mode; "
                          "needs a scenario with interleaved_actions: true, e.g. 'interleaved').",
    "task_greedy": "Fusion-centre task manager (rule-based), the method of this project: sends track-update "
                   "requests -- first one confirmation dwell for each new tentative track, then the confirmed "
                   "tracks due for a revisit -- and each radar's local scheduler accepts, delays, partially "
                   "serves or rejects them (src/core/scheduler.py). The centre never sets radar modes.",
    "random": "Random actions (sanity check).",
    "ppo_task": "Learned fusion-centre task manager: one PPO agent sends a track-update request (or none) to "
                "each radar every second; the radars' local schedulers decide (models/checkpoints/ppo_task.pt).",
    "central_ppo": "Trained PPO policy at the fusion centre: sees the fused track table and all radars, "
                   "sends one action to each radar (models/checkpoints/central_ppo.pt).",
    "mappo": "Trained MAPPO policy (models/checkpoints/mappo.pt).",
    "ippo": "Trained independent-PPO policy (models/checkpoints/ippo.pt).",
}


@st.cache_data(show_spinner=False)
def cached_episode(scenario: str, policy: str, seed: int, use_clf: bool, site_name: str | None = None):
    sc = load_scenario(scenario)
    site = next((s for s in sim.MAP_SITES if s["name"] == site_name), None)
    if site is not None:
        sc["site"] = dict(site)
    env, summary = sim.run_episode(sc, policy, seed=seed, classifier="auto" if use_clf else None)
    return (sim.viewer_payload(env), summary, {k: list(map(float, v)) for k, v in env.log.items()},
            sim.per_radar_table(env))


# ============================================================== Scenario Management
with tab_scen:
    st.subheader("Scenario Management & Operational Analysis")
    st.caption("Interactive scenario configuration, live simulation / dataset replay, "
               "physics-informed radar observations, and explainable radar coordination.")

    scen_col1, scen_col2 = st.columns([3, 4])
    op_mode = scen_col1.radio("Operation Mode", ["Live Simulation (CPU)", "Dataset Replay"], horizontal=True, key="scen_op_mode")

    if op_mode == "Live Simulation (CPU)":
        st.markdown("##### A. Scenario Configuration")
        c_base, c_site, c_pol = st.columns([2, 2, 2])
        scen_list = list_scenarios()
        default_idx = scen_list.index("nominal") if "nominal" in scen_list else 0
        base_scen_name = c_base.selectbox("Base Scenario Template", scen_list, index=default_idx, key="scen_base_name")

        base_scen = load_scenario(base_scen_name)
        site_names = [base_scen["site"]["name"]] + [s["name"] for s in sim.MAP_SITES if s["name"] != base_scen["site"]["name"]]
        selected_site = c_site.selectbox("Map / Terrain Location", site_names, key="scen_site_sel")
        selected_policy = c_pol.selectbox("Coordination Policy", POLICY_NAMES,
                                          index=POLICY_NAMES.index("task_greedy") if "task_greedy" in POLICY_NAMES else 0,
                                          help="\n".join(f"**{k}** – {v}" for k, v in POLICY_HELP.items()
                                                         if k in POLICY_NAMES),
                                          key="scen_pol_sel")

        c_dur, c_dt, c_seed = st.columns(3)
        duration_s = c_dur.slider("Duration (s)", 10.0, 60.0, float(base_scen.get("duration_s", 30.0)), 5.0, key="scen_dur_slider")
        dt_s = c_dt.selectbox("Time Step dt (s)", [0.5, 0.25, 1.0], index=0, key="scen_dt_sel")
        seed_val = c_seed.number_input("Episode Seed", 0, 99999999, int(base_scen.get("seed", 42)), key="scen_seed_inp")

        with st.expander("Detailed Radar Configuration (R1, R2, R3)", expanded=False):
            st.caption("Radars in this scenario. Positions are in ENU coordinates (meters relative to site origin).")
            radars_conf = [dict(r) for r in base_scen.get("radars", [])]
            for ri, r in enumerate(radars_conf):
                r_col1, r_col2, r_col3, r_col4 = st.columns([1, 2, 1, 2])
                r_name = r_col1.text_input(f"Radar {ri+1} Name", value=r.get("name", f"R{ri+1}"), key=f"r_name_{ri}")
                pos_x = r_col2.number_input(f"Pos X (m) #{ri+1}", value=float(r["position"][0]), step=500.0, key=f"r_x_{ri}")
                pos_y = r_col2.number_input(f"Pos Y (m) #{ri+1}", value=float(r["position"][1]), step=500.0, key=f"r_y_{ri}")
                pos_z = r_col3.number_input(f"Mast Z (m) #{ri+1}", value=float(r["position"][2]), step=5.0, key=f"r_z_{ri}")
                hd = r_col4.slider(f"Heading Azimuth (deg) #{ri+1}", 0.0, 360.0, float(r.get("heading_deg", 90.0)), step=5.0, key=f"r_hd_{ri}")
                r["name"] = r_name
                r["position"] = [pos_x, pos_y, pos_z]
                r["heading_deg"] = hd

        with st.expander("Detailed Target Configuration (Class, Variant & Motion)", expanded=False):
            st.caption("Each target is a UAV, an aircraft or a missile. With 'planned' motion the simulator draws "
                       "the variant (unless fixed), the RCS and a valid trajectory through the area from the class "
                       "envelope at every seed; with an explicit start the target flies the chosen motion model.")
            from src.core.target_classes import TARGET_CLASSES
            targets_conf = [dict(t) for t in base_scen.get("targets", [])]
            for ti, tgt in enumerate(targets_conf):
                st.markdown(f"**Target #{ti}**")
                t_col1, t_col2, t_col3 = st.columns(3)
                curr_cls = tgt.get("target_class") or CLASS_NAMES[min(int(tgt.get("label", 0)), len(CLASS_NAMES) - 1)]
                cls_sel = t_col1.selectbox(f"Class #{ti}", CLASS_NAMES, index=CLASS_NAMES.index(curr_cls),
                                           format_func=lambda c: TARGET_CLASSES[c]["name"], key=f"tgt_cls_{ti}")
                variants = ["auto"] + list(TARGET_CLASSES[cls_sel]["variants"])
                curr_var = tgt.get("variant") if tgt.get("variant") in variants else "auto"
                var_sel = t_col2.selectbox(f"Variant #{ti}", variants, index=variants.index(curr_var),
                                           format_func=lambda v: v.replace("_", " "), key=f"tgt_var_{ti}")
                motions = ["planned (class envelope)", "CV", "CA", "CT"]
                curr_mot = (tgt["motion"].get("model", "CV") if isinstance(tgt.get("motion"), dict)
                            and "segments" not in tgt["motion"] else motions[0])
                mot_selected = t_col3.selectbox(f"Motion #{ti}", motions,
                                                index=motions.index(curr_mot) if curr_mot in motions else 0,
                                                key=f"tgt_mot_{ti}")
                new_tgt = {"target_class": cls_sel}
                if mot_selected == motions[0]:
                    if var_sel != "auto":
                        new_tgt["variant"] = var_sel
                else:
                    # explicit start: position, speed and heading
                    k_col1, k_col2, k_col3 = st.columns(3)
                    def_pos = tgt.get("position", [0.0 + ti * 1000.0, -4000.0, 500.0])
                    px = k_col1.number_input(f"Pos X (m) #{ti}", value=float(def_pos[0]), step=250.0, key=f"tgt_px_{ti}")
                    py = k_col2.number_input(f"Pos Y (m) #{ti}", value=float(def_pos[1]), step=250.0, key=f"tgt_py_{ti}")
                    pz = k_col3.number_input(f"Alt Z (m) #{ti}", value=float(def_pos[2]), step=50.0, key=f"tgt_pz_{ti}")
                    variant = var_sel if var_sel != "auto" else list(TARGET_CLASSES[cls_sel]["variants"])[0]
                    lo, hi = TARGET_CLASSES[cls_sel]["variants"][variant].speed_typical_mps
                    s_col1, s_col2 = st.columns(2)
                    spd = s_col1.number_input(f"Speed (m/s) #{ti}", value=float((lo + hi) / 2), step=10.0,
                                              key=f"tgt_spd_{ti}")
                    hdg = s_col2.slider(f"Heading Azimuth (deg) #{ti}", 0.0, 360.0, 90.0, step=5.0, key=f"tgt_hdg_{ti}")
                    motion = {"model": mot_selected}
                    if mot_selected == "CA":
                        acc = st.number_input(f"Acceleration along the heading (m/s²) #{ti}", value=2.0, step=0.5,
                                              key=f"tgt_acc_{ti}")
                        motion["acceleration_mps2"] = [float(acc * np.cos(np.radians(hdg))),
                                                       float(acc * np.sin(np.radians(hdg))), 0.0]
                    elif mot_selected == "CT":
                        motion["turn_rate_rad_s"] = float(np.radians(st.number_input(
                            f"Turn rate (deg/s, + = left) #{ti}", value=3.0, step=0.5, key=f"tgt_turn_{ti}")))
                    new_tgt.update({"variant": variant, "position": [float(px), float(py), float(pz)],
                                    "velocity": [float(spd * np.cos(np.radians(hdg))),
                                                 float(spd * np.sin(np.radians(hdg))), 0.0],
                                    "motion": motion})
                targets_conf[ti] = new_tgt

        with st.expander("Channel & Environment Parameters", expanded=False):
            ch_col1, ch_col2 = st.columns(2)
            pkt_loss = ch_col1.slider("Packet Loss Probability (stress.comm_drop_prob)", 0.0, 0.5,
                                      float(base_scen.get("stress", {}).get("comm_drop_prob", 0.0)), 0.01, key="scen_pkt_loss")
            noise_scale = ch_col2.slider("Measurement Noise Scale (stress.noise_scale)", 0.5, 3.0,
                                         float(base_scen.get("stress", {}).get("noise_scale", 1.0)), 0.1, key="scen_ns")

        # Assemble live scenario dict
        live_sc = dict(base_scen)
        live_sc["duration_s"] = duration_s
        live_sc["dt"] = dt_s
        live_sc["seed"] = int(seed_val)
        site_obj = next((s for s in sim.MAP_SITES if s["name"] == selected_site), None)
        if site_obj is not None:
            live_sc["site"] = dict(site_obj)
        live_sc["radars"] = radars_conf
        live_sc["targets"] = targets_conf
        if "stress" not in live_sc:
            live_sc["stress"] = dict(base_scen.get("stress", {}))
        else:
            live_sc["stress"] = dict(live_sc["stress"])
        live_sc["stress"]["comm_drop_prob"] = float(pkt_loss)
        live_sc["stress"]["noise_scale"] = float(noise_scale)

        b_val, b_run, _ = st.columns([1, 1, 3])
        if b_val.button("Validate Scenario", key="scen_btn_val"):
            try:
                validate_scenario(live_sc)
                st.success("Scenario validated successfully! All parameters, limits, and terrain bounds are consistent.")
            except Exception as e:
                st.error(f"Scenario Validation Error: {e}")

        if b_run.button("Run Simulation", type="primary", key="scen_btn_run"):
            try:
                validate_scenario(live_sc)
                with st.spinner("Running CPU simulation episode with Canonical Records export..."):
                    env_res, summ_res = sim.run_episode(live_sc, policy_name=selected_policy, seed=int(seed_val),
                                                         record_episode=True)
                    st.session_state["scen_records"] = env_res.episode_records
                    st.session_state["scen_payload"] = sim.viewer_payload(env_res)
                    st.session_state["scen_summary"] = summ_res
                    st.session_state["scen_mode_active"] = "live"
                st.success("Simulation finished! See panels below for 3D visualization, Radar Information, and Radar Selection.")
            except Exception as e:
                st.error(f"Simulation Execution Failed: {e}")

    else:
        # Replay Mode with Dynamic Episode Browser, Filters & Instant Preview
        dataset_browser.render_dataset_replay_ui(ROOT, sim)

    records_data = st.session_state.get("scen_records")
    viewer_payload_data = st.session_state.get("scen_payload")

    st.markdown("---")
    st.markdown("##### B. Simulation & Visualization")
    if viewer_payload_data is None or records_data is None:
        st.info("Run a Live Simulation or load an Episode Replay above to view 3-D visualization and operational radar analytics.")
    else:
        # Metrics summary row
        sm_c = st.columns(6)
        n_truth = len(records_data.get("truth", []))
        n_reps = len(records_data.get("reports", []))
        n_comm = records_data.get("communication", [])
        n_deliv = sum(1 for c in n_comm if c.get("final_status") == "delivered")
        n_lost = sum(1 for c in n_comm if c.get("final_status") == "packet_lost")
        deliv_pct = (n_deliv / n_reps * 100.0) if n_reps > 0 else 0.0
        fused_tracks_all = [tr for tr in records_data.get("tracks", []) if tr.get("source") == "fused"]
        unique_fused_ids = set(tr["track_id"] for tr in fused_tracks_all)
        confirmed_fused_ids = set(tr["track_id"] for tr in fused_tracks_all if tr.get("confirmed") or tr.get("status") == "confirmed")
        n_conf_tracks = len(confirmed_fused_ids)
        n_cumul_tracks = len(unique_fused_ids)

        n_states = len(records_data.get("states", []))
        n_steps = len(records_data.get("transitions", []))
        sm_c[0].metric("Simulation Steps", f"{n_steps} stp / {n_states} stt",
                       help=f"{n_steps} action & simulation transitions, {n_states} discrete-time states (t=0..T)")
        sm_c[1].metric("Ground Truth Rows", n_truth)
        sm_c[2].metric("Reports Generated", n_reps)
        sm_c[3].metric("Reports Delivered", f"{n_deliv} ({deliv_pct:.1f}%)")
        sm_c[4].metric("Packet Loss", f"{n_lost}")
        sm_c[5].metric("Fusion Tracks (Confirmed / Cumulative)", f"{n_conf_tracks} / {n_cumul_tracks}",
                       help=f"Confirmed track count: {n_conf_tracks}. Total unique track IDs initiated across the episode: {n_cumul_tracks}. "
                            f"Tentative tracks may be initiated due to clutter, false alarms, and track fragmentation in addition to physical targets.")

        v_opt = st.radio("Visualization Mode", ["3-D WebGL View", "2-D Top-Down View (Plotly)"], horizontal=True, key="scen_v_opt")
        if v_opt == "3-D WebGL View":
            viewer3d.render(viewer_payload_data, height=620)
            st.caption("3-D Interactive View: Left click drag = orbit · Right click drag = pan · Wheel = zoom. "
                       "In Replay mode, detections and tracks are played back directly from records.")
        else:
            fr_slider = st.slider("Simulation Step (Frame)", 0, len(viewer_payload_data["frames"]) - 1, 0, key="scen_fr_slider")
            st.plotly_chart(plots.map_2d(viewer_payload_data, fr_slider), width="stretch")

        st.markdown("---")
        st.markdown("##### C. Radar Information (Category A & B Operational Analysis)")

        info_r_col, info_t_col = st.columns([2, 4])
        sel_radar_name = info_r_col.selectbox("Radar Selection", ["All Radars (R1+R2+R3)", "Radar 1 (R1)", "Radar 2 (R2)", "Radar 3 (R3)"], key="scen_info_r")
        rid_filter = None if "All Radars" in sel_radar_name else (0 if "R1" in sel_radar_name else (1 if "R2" in sel_radar_name else 2))

        t_max_scen = float(records_data["states"][-1]["t_s"]) if records_data.get("states") else 30.0
        sel_t_range = info_t_col.slider("Time Window (s)", 0.0, t_max_scen, (0.0, t_max_scen), 0.5, key="scen_t_slider")

        meas_summary = r_info_srv.summarize_radar_measurements(records_data, radar_id=rid_filter, t_range=sel_t_range)
        comm_summary = r_info_srv.summarize_communication(records_data, radar_id=rid_filter, t_range=sel_t_range)
        track_summary = r_info_srv.summarize_tracker_estimates(records_data, source="fused", t_range=sel_t_range)

        ic = analysis_views.card_row(6)
        ic[0].metric("Selected Reports", meas_summary["total_reports"],
                     help="Radar reports (detections, including false alarms) in the selected radar and time window.")
        deliv_stat = f"{comm_summary.get('delivered', 0)} ({comm_summary.get('delivery_rate_pct', 0.0):.1f}%)" if comm_summary.get("delivered") is not None else "N/A"
        ic[1].metric("Delivery Ratio", deliv_stat,
                     help="Reports that reached the fusion centre after outages, bandwidth caps and packet loss.")
        ic[2].metric("Avg SNR", f"{meas_summary['snr_db']['mean']} dB" if meas_summary.get("snr_db") else "N/A",
                     help="Mean signal-to-noise ratio of the reports.")
        ic[3].metric("Avg Range", f"{meas_summary['range_m']['mean']:,.0f} m" if meas_summary.get("range_m") else "N/A",
                     help="Mean measured distance of the reports.")
        ic[4].metric("Range Noise (σr)", f"{meas_summary['measurement_uncertainty']['sigma_range_m']['mean']:.1f} m" if meas_summary.get("measurement_uncertainty") else "N/A",
                     help="Mean standard deviation of the range measurement (diagonal of the measurement covariance R).")
        ic[5].metric("Filter Uncertainty (σpos)", f"{track_summary['position_uncertainty_m']['mean']:.1f} m" if track_summary.get("position_uncertainty_m") else "N/A",
                     help="Mean RMS position standard deviation of the fused EKF tracks (state covariance P).")

        analysis_views.render_track_states(ep_an.track_state_breakdown(records_data, t_range=sel_t_range))
        st.markdown("###### Signal Physics & EKF Information")
        analysis_views.render_signal_physics(
            ep_an.measurement_physics(records_data, radar_id=rid_filter, t_range=sel_t_range),
            ep_an.ekf_information_gain(records_data, t_range=sel_t_range))

        # Detection Timeline Plotly Chart
        det_timeline_data = r_info_srv.get_detection_timeline(records_data, radar_id=rid_filter, bin_width_s=1.0)
        st.plotly_chart(plots.plot_detection_timeline(det_timeline_data), width="stretch")

        st.info("**Category C Ground Truth Isolation:** Operational radar observations do not include target true identity (`target_id`) or type (`target_type`). Metrics above reflect exclusively physical RF signals and EKF covariance estimates.")

        st.markdown("---")
        st.markdown("##### D. Radar Selection & Decision Analysis")

        act_summary = r_sel_srv.analyze_radar_actions(records_data)
        st.plotly_chart(plots.plot_action_distribution(act_summary["radars"]), width="stretch")

        st.markdown("###### Radar Resource Management")
        analysis_views.render_resources(ep_an.resource_metrics(records_data))

        st.markdown("###### Decision Inspector")
        st.caption("Why the selected radar took its action at a step: the shared track table it saw (uncertainty, age, "
                   "coverage), its energy reserve and the policy rule.  Reproduced by re-simulating the episode.")
        n_trans = len(records_data.get("transitions", []))
        n_rad = len(records_data.get("metadata", {}).get("agent_ids", [])) or 1
        di_c1, di_c2 = st.columns([3, 1])
        insp_step = di_c1.slider("Inspection Time Step (Step)", 1, max(1, n_trans), min(15, max(1, n_trans)), key="scen_insp_step")
        insp_rid = di_c2.selectbox("Inspection Radar", list(range(n_rad)), format_func=lambda r: f"Radar {r+1} (R{r+1})",
                                   key="scen_insp_rid")
        analysis_views.render_decision(ep_an.decision_context(records_data, insp_step, insp_rid))

        st.markdown("###### Multi-Sensor Track Transitions")
        trans_res = r_sel_srv.analyze_track_radar_transitions(records_data)
        st.caption(f"Total of **{trans_res['total_opportunistic_transitions']}** inter-radar track update transitions occurred in this episode.")
        if trans_res["transitions"]:
            df_trans = pd.DataFrame(trans_res["transitions"])[["time_s", "track_id", "from_radar", "to_radar", "time_since_last_update_s", "sigma_pos_before_m", "sigma_pos_after_m"]]
            st.dataframe(df_trans.head(15), width="stretch", hide_index=True)
        st.info("**Note:** A transition means that the next update of a fused track came from another radar. "
                "With the task-manager policies (task_greedy, ppo_task) the fusion centre sends track-update "
                "requests and each radar's own scheduler decides whether and when to serve them; with the "
                "mode-setting baselines the centre assigns the track slot directly. There is no radar-to-radar "
                "handover protocol.")

        st.markdown("---")
        st.markdown("##### E. Ground-Truth Evaluation Metrics (Category C — evaluation only)")
        st.caption("This panel uses simulation ground truth (true kinematics, report annotations, terrain LOS). None of "
                   "it is available to the radars, the tracker or the policy; Sections C and D stay ground-truth free.")
        with st.expander("Show ground-truth evaluation", expanded=False):
            analysis_views.render_ground_truth(ep_an.ground_truth_metrics(records_data))


@st.cache_data(show_spinner=False)
def sample_class_trajectories(duration_s: float = 30.0):
    """One planned trajectory per class (UAV, aircraft, missile) over procedural terrain."""
    from src.core.terrain import build_terrain
    from src.services.class_dataset_service import plan_target, resolve_recipe
    ter = build_terrain(load_scenario("nominal")["site"], {"source": "procedural"}, 0)
    cfg = resolve_recipe({})["trajectory"]
    out = []
    for lab, cls in enumerate(CLASS_NAMES):
        _, info = plan_target(cls, np.random.default_rng(lab + 1), ter, duration_s, 0.5, (0.0, 0.0), cfg)
        out.append((lab, np.asarray(info["trajectory"])))
    return out


# ============================================================== 3-D simulation
with tab_sim:
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
    scen = c1.selectbox("Scenario", list_scenarios(), index=list_scenarios().index("nominal"))
    pol = c2.selectbox("Coordination policy", POLICY_NAMES, index=POLICY_NAMES.index("task_greedy"),
                       help="\n".join(f"**{k}** – {v}" for k, v in POLICY_HELP.items()))
    seed = c3.number_input("Seed", 0, 9999, 0)
    has_clf = sim.CLASSIFIER_PATH.exists()
    use_clf = c4.toggle("Classifier", value=has_clf, disabled=not has_clf,
                        help="Train it on the Classifier tab first." if not has_clf else None)
    view = c5.radio("View", ["3-D", "2-D"], horizontal=True)
    if not TRAINED:
        st.info("**The learned task manager (ppo_task) is not in the list yet because it has not been trained.** "
                "Train once (needs PyTorch, ~30 min on a CPU):  \n"
                "`python -m src.services.training_service --task --decisions 30000`  \n"
                "The checkpoint is written to `models/checkpoints/ppo_task.pt` and then appears here and on the "
                "Experiments tab.")
    sc_site = load_scenario(scen)["site"]["name"]
    site_names = [sc_site] + [s["name"] for s in sim.MAP_SITES if s["name"] != sc_site]
    s1, s2 = st.columns([2, 5])
    site_name = s1.selectbox("Map location (terrain)", site_names,
                             help="Sets the terrain the simulation uses for line-of-sight. "
                                  "Elevation tiles are downloaded once and cached; offline a procedural terrain is used.")
    s2.caption(load_scenario(scen)["description"])

    with st.spinner("Simulating episode (first run on a new site downloads terrain)…"):
        try:
            payload, summary, log, radar_table = cached_episode(scen, pol, int(seed), bool(use_clf), site_name)
        except ValueError as e:
            if type(e).__name__ != "IncompatiblePolicy":
                raise
            st.warning(str(e))
            payload = None
    if payload is not None:
        t_src = payload.get("terrain", {}).get("source", "flat")
        st.caption({"tiles": "Terrain: real elevation (AWS Terrain Tiles) · line-of-sight active",
                    "procedural": "Terrain: procedural hills (elevation tiles unavailable) · line-of-sight active",
                    "flat": "Terrain: flat"}.get(t_src, ""))

        fmt = lambda v, f: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f.format(v)
        m = st.columns(8)
        m[0].metric("Position RMSE", fmt(summary["pos_rmse_m"], "{:.0f} m"), help=METRIC_INFO["pos_rmse_m"][0])
        m[1].metric("Coverage (in range)", fmt(summary["coverage_in_range"], "{:.0%}"), help=METRIC_INFO["coverage_in_range"][0])
        m[2].metric("Track losses", fmt(summary["track_losses"], "{}"), help=METRIC_INFO["track_losses"][0])
        m[3].metric("Longest track", fmt(summary["longest_track_s"], "{:.0f} s"), help=METRIC_INFO["longest_track_s"][0])
        m[4].metric("Detection loss", fmt(summary["detection_loss"], "{:.0%}"), help=METRIC_INFO["detection_loss"][0])
        m[5].metric("Class accuracy", fmt(summary["class_accuracy"], "{:.0%}"), help=METRIC_INFO["class_accuracy"][0])
        m[6].metric("False tracks (avg)", fmt(summary["false_tracks_mean"], "{:.2f}"), help=METRIC_INFO["false_tracks_mean"][0])
        m[7].metric("Team return", fmt(summary["return"], "{:.0f}"), help=METRIC_INFO["return"][0])
        # ------------------------------------------------ parameters (popovers) and the radar equation
        radars_p = build_radars(load_scenario(scen))
        b1, b2, b3 = st.columns([1, 1, 4])
        with b1.popover("📡 Radar parameters"):
            ptab = pd.DataFrame({p.name: [v for _, _, v in radar_info.radar_parameters(p)] for p in radars_p})
            ptab.insert(0, "parameter", [n for _, n, _ in radar_info.radar_parameters(radars_p[0])])
            ptab.insert(0, "group", [g for g, _, _ in radar_info.radar_parameters(radars_p[0])])
            st.dataframe(ptab, hide_index=True, width="stretch", height=560)
            vmax = max(np.linalg.norm(t["v"]) for fr in payload["frames"] for t in fr["targets"])
            for note in sorted(set(n.split(": ", 1)[1] for p in radars_p for n in radar_info.ambiguity_notes(p, vmax))):
                st.warning(note)
        with b2.popover("🎯 Targets"):
            st.dataframe(pd.DataFrame(payload["targets"]), hide_index=True, width="stretch")
            st.caption("RCS depends on the aspect (incidence angle in the target's body frame); the range shows "
                       "the mean RCS over random aspects. Swerling 1: the RCS also fluctuates from look to look.")
        b3.caption("Right-click a radar, target or track in the 3-D view for its parameters.")

        with st.expander("Radar equation: how range, RCS and SNR give the detection probability"):
            e1, e2, e3 = st.columns(3)
            ri_ = e1.selectbox("Radar", range(len(radars_p)), format_func=lambda i: radars_p[i].name, key="lb_r")
            ti_ = e2.selectbox("Target", range(len(payload["targets"])),
                               format_func=lambda i: f"{payload['targets'][i]['target']} · {payload['targets'][i]['class']}",
                               key="lb_t")
            fi_ = e3.slider("Time (frame)", 0, len(payload["frames"]) - 1, len(payload["frames"]) // 2, key="lb_f")
            fr = payload["frames"][fi_]
            rng_m, sig_m2, _ = fr["radars"][ri_]["link"][ti_]
            los = fr["radars"][ri_]["los"][ti_]
            sw = 1 if "1" in payload["targets"][ti_]["fluctuation"] else 0
            tg = fr["targets"][ti_]
            los_vec = np.asarray(tg["p"], float) - radars_p[ri_].pos
            r_dot = float(np.dot(tg["v"], los_vec) / max(np.linalg.norm(los_vec), 1e-9))
            lb = radar_info.link_budget(radars_p[ri_], max(rng_m, 1.0), sig_m2, swerling=sw, r_dot_mps=r_dot)
            st.markdown(f"**t = {fr['t']:.1f} s** · range **{rng_m / 1000:.2f} km** · RCS at this aspect "
                        f"**{sig_m2:.3g} m²** ({10 * np.log10(max(sig_m2, 1e-12)):.1f} dBsm) · line of sight: "
                        + {0: "outside coverage", 1: "visible", 2: "hidden by terrain"}[los])
            c1, c2 = st.columns([3, 2])
            terms = pd.DataFrame(lb["terms"], columns=["term", "dB"])
            terms.loc[len(terms)] = ["= SNR (one track dwell, beam on the target)", lb["snr_db"]]
            c1.dataframe(terms.round(2), hide_index=True, width="stretch", height=500)
            c2.metric("SNR", f"{lb['snr_db']:.1f} dB",
                      help="Signal-to-noise ratio of one track dwell with the beam on the target: the sum of the "
                           "radar-equation terms in the table.")
            rule = f"{radars_p[ri_].prf_m_of_n} of {len(radars_p[ri_].prf_stagger_hz)} PRF sub-dwells"
            c2.metric("P_D, constant RCS (Marcum)", f"{lb['pd_constant_rcs']:.3f}",
                      help=f"Detection probability of the look for a non-fluctuating target: each PRF sub-dwell "
                           f"has {lb['sub_dwell_snr_db']:.1f} dB (Marcum Q-function), detection needs {rule}.")
            c2.metric("P_D, Swerling I", f"{lb['pd_swerling1']:.3f}",
                      help=f"Same for a slowly fluctuating target (Swerling I: one RCS per look), {rule}.")
            n_prf, m_prf = len(radars_p[ri_].prf_stagger_hz), radars_p[ri_].prf_m_of_n
            if lb["n_blind_prfs"] > n_prf - m_prf:
                c2.warning(f"Radial velocity {r_dot:.1f} m/s puts this target in the clutter notch of "
                           f"{lb['n_blind_prfs']} of {n_prf} PRFs: fewer than {m_prf} can see it, so this radar "
                           "cannot detect it in this look (blind speed; zero Doppler = flying across the beam).")
            elif lb["n_blind_prfs"]:
                c2.warning(f"Radial velocity {r_dot:.1f} m/s puts this target in the clutter notch of "
                           f"{lb['n_blind_prfs']} of {n_prf} PRFs; at least {m_prf} of the other "
                           f"{n_prf - lb['n_blind_prfs']} must detect it.")
            c2.caption(f"This target uses the {lb['pd_model']} model. It would be detected with P_D = 0.5 out to "
                       f"{lb['max_range_pd50_km']:.1f} km at this RCS.")
            c2.markdown("**Measurement accuracy at this SNR**  \n" +
                        "  \n".join(f"σ {k}: {v}" for k, v in lb["sigma"].items()))
            st.caption("SNR = P_t G² λ² σ N / ((4π)³ R⁴ L k T_0 B F)  (Eq. 27): the dB terms add up to the SNR. "
                       "Search looks integrate fewer pulses and a target off the beam centre loses gain, so their "
                       "SNR is lower. Simulation truth (range, aspect) is used here for explanation only.")

        with st.expander("Single radar vs network: each radar's local track and the fused track"):
            st.dataframe(radar_table.round(3), width="stretch", hide_index=True)
            st.caption("Local tracks use only that radar's own detections (structure A); the fused track combines "
                       "all radars at the fusion centre. Coverage is measured on the network coverage mask.")

        if view == "3-D":
            viewer3d.render(payload, height=1080)
            st.caption("Drag to rotate · right-drag to pan · scroll to zoom. Terrain and imagery are streamed "
                       "from public tile services; without internet a procedural ground is shown. "
                       "Below the map: one PPI scope per radar with its own local track (L#) and the fusion centre's "
                       "fused track (T#); click a scope to enlarge it, the PPI button hides them.")
        else:
            k = st.slider("Frame", 0, len(payload["frames"]) - 1, len(payload["frames"]) // 2)
            st.plotly_chart(plots.map_2d(payload, k), width="stretch")
            st.markdown("**PPI scopes** · own detections (4 s afterglow), own local track L# (amber), "
                        "fused track T# from the fusion centre (white; open square = this radar does not feed it)")
            n_r = len(payload["radars"])
            for row in range(0, n_r, 4):
                cols = st.columns(min(4, n_r - row))
                for j, col in enumerate(cols):
                    col.plotly_chart(plots.ppi(payload, k, row + j), width="stretch", key=f"ppi_{row + j}")

        g1, g2 = st.columns(2)
        g1.plotly_chart(plots.time_series(log, ["rmse"], "Fused position error of tracked targets", "m"),
                        width="stretch")
        g2.plotly_chart(plots.time_series(log, ["tracked", "class_acc"], "Coverage and class accuracy", "fraction"),
                        width="stretch")
        st.download_button("Download this replay as a stand-alone HTML file",
                           viewer3d.build_html(payload), file_name=f"radnet_{scen}_{pol}_{seed}.html", mime="text/html")


# ============================================================== Radar model lab
with tab_lab:
    st.subheader("Physics-informed radar model")
    st.caption("Interactive views of the equations implemented in src/core (equation numbers refer to the "
               "physics-informed detection note and Alpdemir 2022).")
    p = RadarParams()
    a, b = st.columns(2)
    with a:
        pt = st.slider("Peak power Pt (W)", 100, 5000, int(p.pt_w), 100)
        gain = st.slider("Antenna gain G0 (dBi)", 20, 45, int(p.gain_dbi))
        pfa = 10 ** st.slider("log10 P_FA", -10, -3, -6)
        mode = st.radio("Mode (pulses integrated)", ["SEARCH", "TRACK"], horizontal=True)
    p.pt_w, p.gain_dbi, p.pfa = float(pt), float(gain), pfa
    n_p = p.profiles[Mode[mode]].n_pulses
    rng_km = np.linspace(0.5, 20, 300)
    g0 = antenna_gain(0.0, 0.0, p)
    snr = {f"σ = {s} m²": 10 * np.log10(snr_linear(rng_km * 1e3, s, g0, p, n_p)) for s in (0.01, 0.1, 1.0, 10.0)}
    with b:
        st.plotly_chart(plots.line(rng_km, snr, f"SNR vs range (Eq. 27, {n_p} pulses)", "range (km)", "SNR (dB)"),
                        width="stretch")
    c, d = st.columns(2)
    snr_db = np.linspace(-5, 25, 300)
    chi = 10 ** (snr_db / 10)
    from src.core.radar import look_detection_probability
    pd_curves = {"Marcum Q (Eq. 31)": det.pd_nonfluctuating(chi, pfa), "Swerling I": det.pd_swerling1(chi, pfa),
                 f"MPRF look, {p.prf_m_of_n} of {len(p.prf_stagger_hz)} PRFs (simulator)": look_detection_probability(chi, p),
                 "erfc approx (Alpdemir Eq. 1)": det.pd_erfc_approx(chi, pfa), "logistic (Eq. 34)": det.pd_logistic(snr_db)}
    c.plotly_chart(plots.line(snr_db, pd_curves, "Probability of detection", "SNR (dB)", "P_D"), width="stretch")
    sig = np.array([measurement_sigmas(x, p, p.profiles[Mode[mode]]) for x in chi])
    d.plotly_chart(plots.line(snr_db, {"σ_r (m)": sig[:, 0], "σ_α (mrad)": sig[:, 1] * 1e3,
                                       "σ_ṙ (m/s)": sig[:, 3]},
                              "SNR-dependent accuracy (Eqs. 39-46)", "SNR (dB)", "std"), width="stretch")

    e, f = st.columns(2)
    with e:
        st.markdown("**Three-radius ellipsoid RCS** (Alpdemir Eq. 14)")
        ea = st.slider("a (m)", 0.1, 6.0, 4.0)
        eb = st.slider("b (m)", 0.1, 3.0, 1.0)
        ec = st.slider("c (m)", 0.1, 3.0, 0.6)
        ell = EllipsoidRCS(ea, eb, ec)
        lut = LookupTableRCS.synthetic()
        ang = np.linspace(-180, 180, 361)
        u = [np.array([np.cos(np.deg2rad(x)), np.sin(np.deg2rad(x)), 0.0]) for x in ang]
        st.plotly_chart(plots.polar_pattern(ang, {
            "ellipsoid (dBsm+40)": m2_to_dbsm([ell.mean_rcs(v) for v in u]) + 40,
            "lookup table (dBsm+40)": m2_to_dbsm([lut.mean_rcs(v) for v in u]) + 40},
            "Horizontal-plane RCS vs aspect"), width="stretch")
    with f:
        st.markdown("**Track probability** (Alpdemir Eqs. 2-4)")
        k_looks = st.slider("k looks", 5, 60, 30)
        l_miss = st.slider("l consecutive misses = track loss", 1, 10, 4)
        pdv = np.linspace(0.05, 0.99, 120)
        st.plotly_chart(plots.line(pdv, {"P_track": [det.probability_of_track(x, k_looks, l_miss) for x in pdv]},
                                   "P_track vs single-look P_D", "P_D", "P_track"), width="stretch")
        bw = np.linspace(-6, 6, 200)
        st.plotly_chart(plots.line(bw, {"az cut": 10 * np.log10(antenna_gain(np.deg2rad(bw), 0, p) / g0)},
                                   "Gaussian beam pattern (Eq. 24)", "off-boresight (deg)", "gain (dB)", height=260),
                        width="stretch")


# ============================================================== Classifier
with tab_cls:
    st.subheader("Flight-profile classifier")
    st.caption("Kinematic features of fused tracks → gradient-boosted trees, used by the simulator at the "
               "fusion centre. Trained on fused tracks of multi-radar episodes, evaluated with an episode-wise split.")
    left, right = st.columns([1, 2])
    with left:
        n_eps = st.slider("Episodes per class", 10, 150, 40, 10,
                          help="40 episodes per class take about 10 minutes on a CPU.")
        if st.button("Generate dataset and train", type="primary"):
            import json
            from src.services import classifier_study as cs
            with st.spinner("Simulating multi-radar tracks, training GBM / MLP / GRU…"):
                d = cs.generate(n_eps, seed=int(np.random.default_rng().integers(1 << 30)))
                cs.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(cs.DATA_PATH, **d)
                cs.train_simulation_model(d)
                rep_s = cs.run_study(d)
                cs.REPORT_PATH.write_text(json.dumps(rep_s, indent=2))
            st.cache_data.clear()
            best = max((r for r in rep_s["results"] if r["train_on"] == r["test_on"] == "fused"),
                       key=lambda r: r["macro_f1"])
            st.success(f"Simulator classifier updated · best on fused tracks: {best['model']} "
                       f"macro-F1 {best['macro_f1']:.2f}")
        st.write("Model file:", "✅ present" if sim.CLASSIFIER_PATH.exists() else "— not trained yet")
        st.markdown("\n".join(f"- **{CLASS_SHORT[i]}** – `{n}`" for i, n in enumerate(CLASS_NAMES)))
    with right:
        import json
        study_path = ROOT / "models" / "metadata" / "classifier_study.json"
        rows = json.loads(study_path.read_text())["results"] if study_path.exists() else []
        gbm = next((r for r in rows if r["model"] == "gbm" and r["train_on"] == r["test_on"] == "fused"
                    and "confusion" in r and len(r["confusion"]) == len(CLASS_NAMES)), None)
        if gbm:
            st.caption(f"Simulator classifier (GBM, fused tracks) on held-out episodes: {gbm['n']} windows · "
                       f"macro-F1 {gbm['macro_f1']:.2f} · balanced accuracy {gbm['balanced_accuracy']:.2f}")
            st.plotly_chart(plots.confusion(gbm["confusion"], CLASS_SHORT), width="stretch")
        st.plotly_chart(plots.trajectories_3d(sample_class_trajectories(), "One sample trajectory per class (30 s)"),
                        width="stretch")

    st.subheader("Classifier study: MLP vs GRU, fused vs single-radar tracks")
    study = ROOT / "models" / "metadata" / "classifier_study.json"
    import json
    rep_s = json.loads(study.read_text()) if study.exists() else None
    old_study = rep_s is not None and set(rep_s.get("dataset", {}).get("windows_per_class", {}) or CLASS_NAMES) != set(CLASS_NAMES)
    if rep_s is None:
        st.info("Run once: `python -m src.services.classifier_study --episodes 40` "
                "(multi-radar episodes, ~10 min on a CPU).")
    elif old_study:
        st.warning("This study file was made with the earlier A–D behaviour classes. Re-run it for "
                   "UAV / aircraft / missile: `python -m src.services.classifier_study --episodes 40`.")
    else:
        res = pd.DataFrame(rep_s["results"]).drop(columns="confusion", errors="ignore")
        piv = res.pivot_table(index=["train_on", "test_on"], columns="model", values="macro_f1").round(3)
        st.dataframe(piv, width="stretch")
        st.caption("Macro-F1 on held-out episodes. train_on / test_on: fused = network track at the fusion "
                   "centre, local = one radar's own track, truth = true states (optimistic upper bound only).")
        with st.expander("ECE and balanced accuracy"):
            st.dataframe(res.drop(columns=[c for c in ("curves", "roc_auc_per_class", "pr_auc_per_class",
                                                       "prevalence") if c in res.columns]).round(3),
                         width="stretch", hide_index=True)
        st.markdown("**ROC-AUC and PR-AUC** (one class against the rest, macro average over classes)")
        if "roc_auc_macro" not in res.columns:
            st.info("This study file predates ROC-AUC / PR-AUC: re-run "
                    "`python -m src.services.classifier_study --episodes 40`.")
        else:
            a1, a2 = st.columns(2)
            a1.dataframe(res.pivot_table(index=["train_on", "test_on"], columns="model",
                                         values="roc_auc_macro").round(3), width="stretch")
            a1.caption("ROC-AUC: 0.5 = chance, 1 = perfect ranking of the windows of each class.")
            a2.dataframe(res.pivot_table(index=["train_on", "test_on"], columns="model",
                                         values="pr_auc_macro").round(3), width="stretch")
            a2.caption("PR-AUC (average precision): chance level = the class's share of the test windows. "
                       "It falls when a class is found only together with many false alarms, which ROC-AUC hides "
                       "when classes are unbalanced.")
            pick = [r for r in rep_s["results"] if r.get("curves")]
            if pick:
                labels = [f"{r['model']} · train {r['train_on']} → test {r['test_on']}" for r in pick]
                default = next((i for i, r in enumerate(pick) if r["model"] == "gbm"
                                and r["train_on"] == r["test_on"] == "fused"), 0)
                sel = pick[st.selectbox("Curves of", range(len(pick)), index=default,
                                        format_func=lambda i: labels[i], key="auc_sel")]
                import plotly.graph_objects as go
                g1, g2 = st.columns(2)
                for col, kind, xl, yl, title in ((g1, "roc", "false positive rate", "true positive rate", "ROC"),
                                                 (g2, "pr", "recall", "precision", "Precision-recall")):
                    fig = go.Figure()
                    for k, (name, (x, y)) in enumerate(sel["curves"][kind].items()):
                        auc = sel[f"{kind}_auc_per_class"][CLASS_NAMES.index(name)]
                        fig.add_scatter(x=x, y=y, mode="lines", name=f"{CLASS_SHORT[CLASS_NAMES.index(name)]} "
                                                                     f"({auc:.2f})")
                    if kind == "roc":
                        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                        line={"dash": "dash", "color": "#8DA2AC"})
                    fig.update_layout(template="plotly_dark", title=title, xaxis_title=xl, yaxis_title=yl,
                                      height=360, paper_bgcolor="#0E1417", plot_bgcolor="#151D21",
                                      xaxis_range=[0, 1], yaxis_range=[0, 1.02], margin={"t": 40})
                    col.plotly_chart(fig, width="stretch")
        ds_chk = rep_s["dataset"]
        st.markdown(
            f"**Dataset checks** – windows per class: {ds_chk['windows_per_class']} "
            f"(training uses a class-balanced subset) · episodes in both train and test: "
            f"{ds_chk['episodes_in_train_and_test']} · best single feature: `{ds_chk['best_single_feature']}` "
            f"({ds_chk['best_single_feature_accuracy']:.0%} accuracy alone).")
        if rep_s.get("hard_pairs") or rep_s.get("early") or rep_s.get("by_variant"):
            st.markdown("**Hard cases** – the overall scores are high because UAV, aircraft and missile differ "
                        "strongly in speed and altitude. These tables show where it is not easy (GBM = the "
                        "simulator's model, GRU = best sequence model; fused tracks of held-out episodes).")
            h1, h2 = st.columns(2)
            if rep_s.get("hard_pairs"):
                hp = pd.DataFrame(rep_s["hard_pairs"]).round(3)
                h1.dataframe(hp, hide_index=True, width="stretch")
                h1.caption("Physically overlapping variant pairs: accuracy on their windows, how often one is "
                           "taken for the other, and the best single feature on this pair alone.")
            if rep_s.get("early"):
                h2.dataframe(pd.DataFrame(rep_s["early"]).round(3), hide_index=True, width="stretch")
                h2.caption("Young tracks: only their first 3 s or 6 s (the simulator first classifies a track "
                           "after 6 s). The models were trained on 10 s windows.")
            if rep_s.get("by_variant"):
                bv = pd.DataFrame(rep_s["by_variant"]).T.reset_index().rename(columns={"index": "variant"})
                st.dataframe(bv.round(3), hide_index=True, width="stretch")
                st.caption("Accuracy per target variant; n = test windows (fast missiles give few windows).")


# ============================================================== Experiments
with tab_exp:
    st.subheader("Policy comparison")
    a, b, c = st.columns([2, 3, 1])
    scen_e = a.selectbox("Scenario ", list_scenarios(), key="exp_scen")
    pols = b.multiselect("Policies", POLICY_NAMES, default=TRAINED + ["task_greedy", "central_greedy_confirm",
                                                                     "central_greedy", "round_robin", "fixed_scan"])
    n_seeds = c.number_input("Seeds", 1, 10, 3)
    if st.button("Run comparison", type="primary"):
        bar = st.progress(0.0, text="Running episodes…")
        df = sim.compare_policies(scen_e, pols, seeds=range(int(n_seeds)),
                                  progress=lambda d, n, lab: bar.progress(d / n, text=f"Episode {d}/{n} · {lab}"))
        bar.empty()
        for pol_name, why in df.attrs.get("skipped", {}).items():
            st.warning(f"Skipped **{pol_name}**: {why}")
        st.session_state["exp_df"] = df
    df = st.session_state.get("exp_df")
    if df is not None and not df.empty:
        metrics = [c for c in df.columns if c not in ("policy", "seed")]
        default = [m for m in ("coverage_in_range", "single_best_coverage", "track_losses", "longest_track_s", "reacquire_s",
                               "correct_class_continuity", "pos_rmse_m", "return") if m in metrics]
        shown = st.multiselect("Columns", metrics, default=default, key="exp_cols",
                               help="Each cell is mean ± standard deviation over the seeds.")
        g = df.groupby("policy")[metrics]
        mean, std = g.mean(), g.std().fillna(0.0)
        table = pd.DataFrame({m: [f"{a:.3g} ± {b:.2g}" if np.isfinite(a) else "—"
                                  for a, b in zip(mean[m], std[m])] for m in shown}, index=mean.index)
        st.dataframe(table, width="stretch")
        metric = st.selectbox("Metric", [c for c in df.columns if c not in ("policy", "seed")])
        import plotly.express as px
        fig = px.box(df, x="policy", y=metric, points="all", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0E1417", plot_bgcolor="#151D21", height=380)
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False), "experiment.csv", "text/csv")
    with st.expander("What do the metrics mean?"):
        st.dataframe(pd.DataFrame([{"metric": k, "meaning": v[0], "unit / denominator": v[1]}
                                   for k, v in METRIC_INFO.items()]), width="stretch", hide_index=True)
        st.caption("Track loss counts tracks deleted by the tracker; a single missed detection is not a track loss. "
                   "Definitions: src/core/evaluation.py.")
    missing = [n for n in LEARNED if n not in TRAINED]
    if missing:
        st.info(f"Not trained yet: {', '.join(missing)}. Train with "
                + ", ".join(f"`python -m src.services.training_service {LEARNED[n]}`".replace(" `", "`") for n in missing)
                + " ; checkpoints in models/checkpoints/ appear here automatically.")


# ============================================================== About
with tab_about:
    st.markdown((ROOT / "doc" / "architecture" / "overview.md").read_text(encoding="utf-8")
                if (ROOT / "doc" / "architecture" / "overview.md").exists() else "See README.md")
