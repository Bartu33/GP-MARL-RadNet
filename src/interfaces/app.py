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
from src.core.scenario import list_scenarios, load_scenario, validate_scenario
from src.core.targets import CLASS_NAMES, CLASS_SHORT, advance, spawn_target
from src.interfaces import plots, viewer3d
from src.models.baselines.policies import LEARNED, POLICIES
from src.services import simulation_service as sim
from src.services import radar_information_service as r_info_srv
from src.services import radar_selection_service as r_sel_srv

st.set_page_config(page_title="GP-MARL-RadNet", layout="wide", page_icon="📡")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
[data-testid="stMetricValue"]{font-family:ui-monospace,monospace}
</style>""", unsafe_allow_html=True)

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
    "fixed_scan": "Always search.",
    "round_robin": "Alternate search / track, cycling through tracks.",
    "nearest": "Each radar refreshes its closest track that is due for a revisit, else searches.",
    "central_greedy": "Central heuristic: most uncertain due tracks first, no duplicates, spare radars search.",
    "interleaved_greedy": "Like central_greedy but tracking radars keep searching (INTERLEAVED mode; "
                          "needs a scenario with interleaved_actions: true, e.g. 'interleaved').",
    "random": "Random actions (sanity check).",
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
    op_mode = scen_col1.radio("Operation Mode", ["Live Simulation (CPU)", "Dataset Replay (multi_target_v1)"], horizontal=True, key="scen_op_mode")

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
                                          index=POLICY_NAMES.index("central_greedy") if "central_greedy" in POLICY_NAMES else 0,
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

        with st.expander("Detailed Target Configuration (Archetypes & Motion)", expanded=False):
            st.caption("Target archetypes, legacy labels, and kinematic motion models. Optionally specify explicit initial positions and velocities.")
            targets_conf = [dict(t) for t in base_scen.get("targets", [])]
            for ti, tgt in enumerate(targets_conf):
                st.markdown(f"**Target #{ti}**")
                t_col1, t_col2, t_col3 = st.columns(3)
                tt_options = ["aircraft_like", "uav_like", "missile_like"]
                curr_tt = tgt.get("target_type", "aircraft_like" if tgt.get("label", 0) == 0 else "uav_like")
                if curr_tt not in tt_options:
                    tt_options.append(curr_tt)
                tt_selected = t_col1.selectbox(f"Archetype #{ti}", tt_options, index=tt_options.index(curr_tt), key=f"tgt_tt_{ti}")
                lbl_val = t_col2.selectbox(f"Legacy Class #{ti} [Ground Truth]", [0, 1, 2, 3], index=int(tgt.get("label", 0)),
                                           format_func=lambda l: f"Class {l} ({CLASS_NAMES[l]})", key=f"tgt_lbl_{ti}")
                mot_dict = tgt.get("motion", {})
                mot_model = mot_dict.get("model", "CV") if isinstance(mot_dict, dict) else "CV"
                mot_selected = t_col3.selectbox(f"Motion Model #{ti}", ["CV", "CA", "CT", "SegmentedMotion"],
                                                index=["CV", "CA", "CT", "SegmentedMotion"].index(mot_model) if mot_model in ["CV", "CA", "CT", "SegmentedMotion"] else 0,
                                                key=f"tgt_mot_{ti}")
                tgt["target_type"] = tt_selected
                tgt["label"] = lbl_val

                # Explicit initial position & velocity kinematics
                has_explicit_init = "position" in tgt and "velocity" in tgt
                k_col1, k_col2, k_col3, k_col4 = st.columns(4)
                use_custom = k_col1.checkbox(f"Explicit Initial Pos/Vel #{ti}", value=has_explicit_init, key=f"tgt_custom_{ti}")
                if use_custom:
                    def_pos = tgt.get("position", [0.0 + ti * 1000.0, 0.0 + ti * 1000.0, 500.0 + ti * 200.0])
                    px = k_col2.number_input(f"Pos X (m) #{ti}", value=float(def_pos[0]), step=250.0, key=f"tgt_px_{ti}")
                    py = k_col3.number_input(f"Pos Y (m) #{ti}", value=float(def_pos[1]), step=250.0, key=f"tgt_py_{ti}")
                    pz = k_col4.number_input(f"Alt Z (m) #{ti}", value=float(def_pos[2]), step=50.0, key=f"tgt_pz_{ti}")
                    s_col1, s_col2 = st.columns(2)
                    def_vel = tgt.get("velocity", [150.0 if ti == 0 else (20.0 if ti in (1, 3) else 280.0), 0.0, 0.0])
                    def_speed = float(np.hypot(def_vel[0], def_vel[1]))
                    spd = s_col1.number_input(f"Speed (m/s) #{ti}", value=def_speed, step=10.0, key=f"tgt_spd_{ti}")
                    hdg = s_col2.slider(f"Heading Azimuth (deg) #{ti}", 0.0, 360.0, 0.0, step=5.0, key=f"tgt_hdg_{ti}")
                    vx = spd * np.cos(np.radians(hdg))
                    vy = spd * np.sin(np.radians(hdg))
                    tgt["position"] = [float(px), float(py), float(pz)]
                    tgt["velocity"] = [float(vx), float(vy), 0.0]
                    tgt["motion"] = {"model": mot_selected}
                else:
                    tgt.pop("position", None)
                    tgt.pop("velocity", None)
                    if isinstance(tgt.get("motion"), dict) and mot_selected != "SegmentedMotion":
                        tgt["motion"]["model"] = mot_selected

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
        # Replay Mode
        st.markdown("##### A. Episode Replay Selection")
        ds_root = ROOT / "data" / "processed" / "multi_target_v1"
        if not ds_root.is_dir():
            st.warning(f"multi_target_v1 dataset directory not found at {ds_root}.")
        else:
            rep_c1, rep_c2 = st.columns([1, 2])
            split_choice = rep_c1.selectbox("Dataset Split", ["train", "validation", "test"], index=0, key="rep_split_sel")
            split_dir = ds_root / split_choice
            ep_dirs = sorted([d.name for d in split_dir.glob("ep-*")]) if split_dir.is_dir() else []
            selected_ep = rep_c2.selectbox("Episode Directory", ep_dirs, index=0 if ep_dirs else None, key="rep_ep_sel")
            st.caption(f"Aktif split: **{split_choice}** ({len(ep_dirs)} episode). "
                       f"Bölünme: Prefix-stable largest-deficit tahsisi (Train: 70, Validation: 15, Test: 15).")

            if selected_ep:
                ep_full_path = split_dir / selected_ep
                if st.button("Load Replay Episode", type="primary", key="rep_load_btn"):
                    with st.spinner("Loading Canonical Episode Records v2 from disk..."):
                        recs = sim.load_episode_records(ep_full_path)
                        payload_conv = sim.records_to_viewer_payload(recs)
                        st.session_state["scen_records"] = recs
                        st.session_state["scen_payload"] = payload_conv
                        st.session_state["scen_mode_active"] = "replay"
                        st.session_state["scen_ep_name"] = selected_ep
                    st.success(f"Episode {selected_ep} loaded successfully ({len(recs['reports'])} reports, {len(recs['tracks'])} tracks).")

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
        sm_c[0].metric("Simülasyon Adımları", f"{n_steps} ad / {n_states} drm",
                       help=f"{n_steps} eylem ve simülasyon geçiş adımı (transitions), {n_states} ayrık zaman durumu (states; t=0..T)")
        sm_c[1].metric("Ground Truth Rows", n_truth)
        sm_c[2].metric("Reports Generated", n_reps)
        sm_c[3].metric("Reports Delivered", f"{n_deliv} ({deliv_pct:.1f}%)")
        sm_c[4].metric("Packet Loss", f"{n_lost}")
        sm_c[5].metric("Füzyon İzleri (Onaylı/Kümülatif)", f"{n_conf_tracks} / {n_cumul_tracks}",
                       help=f"Doğrulanan (confirmed) iz sayısı: {n_conf_tracks}. Episode boyunca başlatılan toplam tekil iz (track ID) sayısı: {n_cumul_tracks}. "
                            f"Fiziksel hedeflere ek olarak clutter, yanlış alarm ve track fragmentation sonucu geçici (tentative) izler başlatılabilir.")

        v_opt = st.radio("Visualization Mode", ["3-D WebGL View", "2-D Kuşbakışı (Plotly)"], horizontal=True, key="scen_v_opt")
        if v_opt == "3-D WebGL View":
            viewer3d.render(viewer_payload_data, height=620)
            st.caption("3-D Etkileşimli Görünüm: Sol tık sürükle = döndür · Sağ tık = kaydır · Tekerlek = yakınlaştır. "
                       "Replay modunda tespitler ve izler doğrudan kayıtlardan oynatılır.")
        else:
            fr_slider = st.slider("Zaman Adımı (Frame)", 0, len(viewer_payload_data["frames"]) - 1, 0, key="scen_fr_slider")
            st.plotly_chart(plots.map_2d(viewer_payload_data, fr_slider), width="stretch")

        st.markdown("---")
        st.markdown("##### C. Radar Information (Kategori A & B Operasyonel Analiz)")

        info_r_col, info_t_col = st.columns([2, 4])
        sel_radar_name = info_r_col.selectbox("Radar Seçimi", ["Tüm Radarlar (R1+R2+R3)", "Radar 1 (R1)", "Radar 2 (R2)", "Radar 3 (R3)"], key="scen_info_r")
        rid_filter = None if "Tüm Radarlar" in sel_radar_name else (0 if "R1" in sel_radar_name else (1 if "R2" in sel_radar_name else 2))

        t_max_scen = float(records_data["states"][-1]["t_s"]) if records_data.get("states") else 30.0
        sel_t_range = info_t_col.slider("Zaman Penceresi (s)", 0.0, t_max_scen, (0.0, t_max_scen), 0.5, key="scen_t_slider")

        meas_summary = r_info_srv.summarize_radar_measurements(records_data, radar_id=rid_filter, t_range=sel_t_range)
        comm_summary = r_info_srv.summarize_communication(records_data, radar_id=rid_filter, t_range=sel_t_range)
        track_summary = r_info_srv.summarize_tracker_estimates(records_data, source="fused", t_range=sel_t_range)

        ic = st.columns(6)
        ic[0].metric("Seçili Rapor", meas_summary["total_reports"])
        deliv_stat = f"{comm_summary.get('delivered', 0)} ({comm_summary.get('delivery_rate_pct', 0.0):.1f}%)" if comm_summary.get("delivered") is not None else "—"
        ic[1].metric("Teslim Oranı", deliv_stat)
        ic[2].metric("Ort. SNR", f"{meas_summary['snr_db']['mean']} dB" if meas_summary.get("snr_db") else "—")
        ic[3].metric("Ort. Menzil", f"{meas_summary['range_m']['mean']:.0f} m" if meas_summary.get("range_m") else "—")
        ic[4].metric("Ölçüm Gürültüsü σ_r", f"{meas_summary['measurement_uncertainty']['sigma_range_m']['mean']:.1f} m" if meas_summary.get("measurement_uncertainty") else "—",
                     help="Sensör seviyesi menzil ölçüm gürültüsü standart sapması (Measurement Covariance R)")
        ic[5].metric("Filtre Belirsizliği σ_pos", f"{track_summary['position_uncertainty_m']['mean']:.1f} m" if track_summary.get("position_uncertainty_m") else "—",
                     help="Merkezi EKF durum kestirim kovaryansı (State Estimation Covariance P)")

        st_bk = track_summary.get("status_breakdown", {})
        n_conf = st_bk.get("confirmed", 0)
        n_tent = st_bk.get("tentative", 0)
        n_coast = st_bk.get("coasted", 0)
        st.caption(f"Füzyon İz Durum Dağılımı: **{n_conf} doğrulanmış (confirmed)** iz adımı (%{track_summary.get('confirmed_steps_pct', 0.0):.1f}), "
                   f"**{n_tent} geçici (tentative)** iz adımı, **{n_coast} serbest seyir (coasted)** iz adımı.")

        # Detection Timeline Plotly Chart
        det_timeline_data = r_info_srv.get_detection_timeline(records_data, radar_id=rid_filter, bin_width_s=1.0)
        st.plotly_chart(plots.plot_detection_timeline(det_timeline_data), width="stretch")

        st.info("**Kategori C Ground Truth İzolasyonu:** Operasyonel radar ölçümlerinde hedefin gerçek kimliği (`target_id`) veya türü (`target_type`) yer almaz. Yukarıdaki metrikler yalnızca radarların fiziksel RF sinyal ve EKF kovaryans çıktısıdır.")

        st.markdown("---")
        st.markdown("##### D. Radar Selection & Decision Analysis")

        act_summary = r_sel_srv.analyze_radar_actions(records_data)
        st.plotly_chart(plots.plot_action_distribution(act_summary["radars"]), width="stretch")

        st.markdown("###### Karar Açıklayıcı (Decision Inspector)")
        st.caption("Herhangi bir zaman adımında seçilen radarın neden arama yaptığı veya neden belirli bir hedef slotunu takip ettiğini inceleyin.")

        di_c1, di_c2 = st.columns([3, 1])
        insp_step = di_c1.slider("İncelenecek Zaman Adımı (Step)", 0, len(records_data.get("transitions", [])) - 1, min(15, len(records_data.get("transitions", [])) - 1), key="scen_insp_step")
        insp_rid = di_c2.selectbox("İncelenecek Radar", [0, 1, 2], format_func=lambda r: f"Radar {r+1} (R{r+1})", key="scen_insp_rid")

        decision_data = r_sel_srv.explain_radar_selection_decision(records_data, insp_step, insp_rid)

        d_box1, d_box2 = st.columns(2)
        with d_box1:
            st.markdown(f"**Alınan Aksiyon:** `{decision_data.get('mode')} (Aksiyon {decision_data.get('action')})`")
            st.markdown(f"🟢 **Politika Kuralı (Rule-Based Fact):**  \n{decision_data.get('justification')}")
        with d_box2:
            st.markdown(f"**Adım Bilgisi:** Step {decision_data.get('step_index')} (t = {insp_step * 0.5:.1f} s)")
            st.markdown(f"🟡 **Analitik Çıkarsama (Post-Hoc Context):**  \n{decision_data.get('explanation')}")

        st.markdown("###### Çoklu Sensör İz Geçişleri (Sensor Transitions)")
        trans_res = r_sel_srv.analyze_track_radar_transitions(records_data)
        st.caption(f"Bu episode'da toplam **{trans_res['total_opportunistic_transitions']} adet** radarlar arası iz güncelleme geçişi gerçekleşmiştir.")
        if trans_res["transitions"]:
            df_trans = pd.DataFrame(trans_res["transitions"])[["time_s", "track_id", "from_radar", "to_radar", "time_since_last_update_s", "sigma_pos_before_m", "sigma_pos_after_m"]]
            st.dataframe(df_trans.head(15), width="stretch", hide_index=True)
        st.warning("**Not:** Bu geçişler merkezi kara tahta (slots) üzerinden fırsatçı sensör atamasıdır (Centralized Opportunistic Sensor Transition). Kod tabanında dağıtık otonom el sıkışma protokolü (Handover Protocol) bulunmamaktadır.")


# ============================================================== 3-D simulation
with tab_sim:
    c1, c2, c3, c4, c5 = st.columns([2, 2, 1, 1, 1])
    scen = c1.selectbox("Scenario", list_scenarios(), index=list_scenarios().index("nominal"))
    pol = c2.selectbox("Coordination policy", POLICY_NAMES, index=POLICY_NAMES.index("central_greedy"),
                       help="\n".join(f"**{k}** – {v}" for k, v in POLICY_HELP.items()))
    seed = c3.number_input("Seed", 0, 9999, 0)
    has_clf = sim.CLASSIFIER_PATH.exists()
    use_clf = c4.toggle("Classifier", value=has_clf, disabled=not has_clf,
                        help="Train it on the Classifier tab first." if not has_clf else None)
    view = c5.radio("View", ["3-D", "2-D"], horizontal=True)
    if not TRAINED:
        st.info("**No learned policy in the list yet because none has been trained.** "
                "Train once (needs PyTorch, ~1-2 h on a CPU):  \n"
                "`python -m src.services.training_service --central --steps 150000` (policy at the fusion centre), "
                "`--steps 150000` (MAPPO) or `--ippo --steps 150000` (IPPO).  \n"
                "The checkpoints are written to `models/checkpoints/` and then appear here and on the Experiments tab.")
    sc_site = load_scenario(scen)["site"]["name"]
    site_names = [sc_site] + [s["name"] for s in sim.MAP_SITES if s["name"] != sc_site]
    s1, s2 = st.columns([2, 5])
    site_name = s1.selectbox("Map location (terrain)", site_names,
                             help="Sets the terrain the simulation uses for line-of-sight. "
                                  "Elevation tiles are downloaded once and cached; offline a procedural terrain is used.")
    s2.caption(load_scenario(scen)["description"])

    with st.spinner("Simulating episode (first run on a new site downloads terrain)…"):
        payload, summary, log, radar_table = cached_episode(scen, pol, int(seed), bool(use_clf), site_name)
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
    with st.expander("Single radar vs network: each radar's local track and the fused track"):
        st.dataframe(radar_table.round(3), width="stretch", hide_index=True)
        st.caption("Local tracks use only that radar's own detections (structure A); the fused track combines "
                   "all radars at the fusion centre. Coverage is measured on the network coverage mask.")

    if view == "3-D":
        viewer3d.render(payload, height=740)
        st.caption("Drag to rotate · right-drag to pan · scroll to zoom. Terrain and imagery are streamed "
                   "from public tile services; without internet a procedural ground is shown.")
    else:
        k = st.slider("Frame", 0, len(payload["frames"]) - 1, len(payload["frames"]) // 2)
        st.plotly_chart(plots.map_2d(payload, k), width="stretch")

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
    pd_curves = {"Marcum Q (Eq. 31)": det.pd_nonfluctuating(chi, pfa), "Swerling I": det.pd_swerling1(chi, pfa),
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
                    and "confusion" in r), None)
        if gbm:
            st.caption(f"Simulator classifier (GBM, fused tracks) on held-out episodes: {gbm['n']} windows · "
                       f"macro-F1 {gbm['macro_f1']:.2f} · balanced accuracy {gbm['balanced_accuracy']:.2f}")
            st.plotly_chart(plots.confusion(gbm["confusion"], CLASS_SHORT), width="stretch")
        rng = np.random.default_rng(1)
        trajs = []
        for lab in range(4):
            s, mot = spawn_target(lab, lab, rng, 6000)
            P = [s.p.copy()]
            for _ in range(160):
                advance(s, mot, 0.5)
                P.append(s.p.copy())
            trajs.append((lab, np.array(P)))
        st.plotly_chart(plots.trajectories_3d(trajs, "One sample trajectory per class (80 s)"), width="stretch")

    st.subheader("Classifier study: MLP vs GRU, fused vs single-radar tracks")
    study = ROOT / "models" / "metadata" / "classifier_study.json"
    if not study.exists():
        st.info("Run once: `python -m src.services.classifier_study --episodes 40` "
                "(multi-radar episodes, ~10 min on a CPU).")
    else:
        import json
        rep_s = json.loads(study.read_text())
        res = pd.DataFrame(rep_s["results"]).drop(columns="confusion", errors="ignore")
        piv = res.pivot_table(index=["train_on", "test_on"], columns="model", values="macro_f1").round(3)
        st.dataframe(piv, width="stretch")
        st.caption("Macro-F1 on held-out episodes. train_on / test_on: fused = network track at the fusion "
                   "centre, local = one radar's own track, truth = true states (optimistic upper bound only).")
        with st.expander("ECE and balanced accuracy"):
            st.dataframe(res.round(3), width="stretch", hide_index=True)
        ds_chk = rep_s["dataset"]
        st.markdown(
            f"**Dataset checks** – windows per class: {ds_chk['windows_per_class']} "
            f"(training uses a class-balanced subset) · episodes in both train and test: "
            f"{ds_chk['episodes_in_train_and_test']} · best single feature: `{ds_chk['best_single_feature']}` "
            f"({ds_chk['best_single_feature_accuracy']:.0%} accuracy alone).")


# ============================================================== Experiments
with tab_exp:
    st.subheader("Policy comparison")
    a, b, c = st.columns([2, 3, 1])
    scen_e = a.selectbox("Scenario ", list_scenarios(), key="exp_scen")
    pols = b.multiselect("Policies", POLICY_NAMES, default=TRAINED + ["fixed_scan", "round_robin", "nearest", "central_greedy"])
    n_seeds = c.number_input("Seeds", 1, 10, 3)
    if st.button("Run comparison", type="primary"):
        bar = st.progress(0.0, text="Running episodes…")
        df = sim.compare_policies(scen_e, pols, seeds=range(int(n_seeds)),
                                  progress=lambda d, n, lab: bar.progress(d / n, text=f"Episode {d}/{n} · {lab}"))
        bar.empty()
        st.session_state["exp_df"] = df
    df = st.session_state.get("exp_df")
    if df is not None:
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
