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
from src.core.scenario import build_radars, list_scenarios, load_scenario
from src.core.targets import CLASS_NAMES, CLASS_SHORT, advance, spawn_target
from src.interfaces import plots, viewer3d
from src.models.baselines.policies import LEARNED, POLICIES
from src.services import radar_info
from src.services import simulation_service as sim

st.set_page_config(page_title="GP-MARL-RadNet", layout="wide", page_icon="📡")
st.markdown("""<style>
.block-container{padding-top:1.2rem}
[data-testid="stMetricValue"]{font-family:ui-monospace,monospace}
</style>""", unsafe_allow_html=True)

st.title("GP-MARL-RadNet")
st.caption("Cooperative multi-agent radar network · research and educational simulation "
           "(abstract models, synthetic targets, non-operational parameters)")

tab_sim, tab_lab, tab_cls, tab_exp, tab_about = st.tabs(
    ["3-D Simulation", "Radar Model Lab", "Classifier", "Experiments", "About"])

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
    "task_greedy": "Fusion-centre task manager (rule-based): sends track-update requests for due tracks; "
                   "each radar's local scheduler accepts, delays, partially serves or rejects them "
                   "(src/core/scheduler.py). The centre never sets radar modes.",
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
            lb = radar_info.link_budget(radars_p[ri_], max(rng_m, 1.0), sig_m2, swerling=sw)
            st.markdown(f"**t = {fr['t']:.1f} s** · range **{rng_m / 1000:.2f} km** · RCS at this aspect "
                        f"**{sig_m2:.3g} m²** ({10 * np.log10(max(sig_m2, 1e-12)):.1f} dBsm) · line of sight: "
                        + {0: "outside coverage", 1: "visible", 2: "hidden by terrain"}[los])
            c1, c2 = st.columns([3, 2])
            terms = pd.DataFrame(lb["terms"], columns=["term", "dB"])
            terms.loc[len(terms)] = ["= SNR (one track dwell, beam on the target)", lb["snr_db"]]
            c1.dataframe(terms.round(2), hide_index=True, width="stretch", height=500)
            c2.metric("SNR", f"{lb['snr_db']:.1f} dB")
            c2.metric("P_D, constant RCS (Marcum)", f"{lb['pd_constant_rcs']:.3f}")
            c2.metric("P_D, Swerling I", f"{lb['pd_swerling1']:.3f}")
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
