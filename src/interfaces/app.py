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
from src.core.modes import Mode
from src.core.radar import RadarParams, antenna_gain, measurement_sigmas, snr_linear
from src.core.rcs import EllipsoidRCS, LookupTableRCS, m2_to_dbsm
from src.core.scenario import list_scenarios, load_scenario
from src.core.targets import CLASS_NAMES, CLASS_SHORT, advance, spawn_target
from src.interfaces import plots, viewer3d
from src.models.baselines.policies import POLICIES
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
TRAINED = [n for n in ("mappo", "ippo") if (CKPT_DIR / f"{n}.pt").exists()]
POLICY_NAMES = TRAINED + POLICY_NAMES
POLICY_HELP = {
    "fixed_scan": "Always search.",
    "round_robin": "Alternate search / track, cycling through tracks.",
    "nearest": "Each radar tracks its closest track.",
    "central_greedy": "Central heuristic: most uncertain tracks first, no duplicates.",
    "interleaved_greedy": "Like central_greedy but tracking radars keep searching (INTERLEAVED mode; "
                          "needs a scenario with interleaved_actions: true, e.g. 'interleaved').",
    "random": "Random actions (sanity check).",
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
    return sim.viewer_payload(env), summary, {k: list(map(float, v)) for k, v in env.log.items()}


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
    if "mappo" not in TRAINED:
        st.info("**MAPPO is not in the list yet because it has not been trained.** "
                "Train it once (needs PyTorch, ~1-2 h on a CPU):  \n"
                "`python -m src.services.training_service --steps 150000` (MAPPO) and "
                "`python -m src.services.training_service --ippo --steps 150000` (IPPO baseline).  \n"
                "The checkpoints are written to `models/checkpoints/` and then appear here and on the Experiments tab.")
    sc_site = load_scenario(scen)["site"]["name"]
    site_names = [sc_site] + [s["name"] for s in sim.MAP_SITES if s["name"] != sc_site]
    s1, s2 = st.columns([2, 5])
    site_name = s1.selectbox("Map location (terrain)", site_names,
                             help="Sets the terrain the simulation uses for line-of-sight. "
                                  "Elevation tiles are downloaded once and cached; offline a procedural terrain is used.")
    s2.caption(load_scenario(scen)["description"])

    with st.spinner("Simulating episode (first run on a new site downloads terrain)…"):
        payload, summary, log = cached_episode(scen, pol, int(seed), bool(use_clf), site_name)
    t_src = payload.get("terrain", {}).get("source", "flat")
    st.caption({"tiles": "Terrain: real elevation (AWS Terrain Tiles) · line-of-sight active",
                "procedural": "Terrain: procedural hills (elevation tiles unavailable) · line-of-sight active",
                "flat": "Terrain: flat"}.get(t_src, ""))

    m = st.columns(6)
    m[0].metric("Position RMSE", f"{summary['pos_rmse_m']:.0f} m")
    m[1].metric("Track coverage", f"{summary['track_coverage']:.0%}")
    m[2].metric("False tracks (avg)", f"{summary['false_tracks_mean']:.2f}")
    m[3].metric("Class accuracy", "—" if np.isnan(summary["class_accuracy"]) else f"{summary['class_accuracy']:.0%}")
    m[4].metric("Sensing cost", f"{summary['sensing_cost']:.3f}")
    m[5].metric("Team return", f"{summary['return']:.0f}")

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
    st.caption("Kinematic features of fused tracks → gradient-boosted trees. Trained on synthetic "
               "tracker output, evaluated with an episode-wise split.")
    left, right = st.columns([1, 2])
    with left:
        n_eps = st.slider("Episodes per class", 10, 150, 40, 10)
        if st.button("Generate dataset and train", type="primary"):
            from src.services import dataset_service as ds
            with st.spinner("Simulating radar tracks and training…"):
                df = ds.generate(n_eps, seed=int(np.random.default_rng().integers(1 << 30)))
                ds.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(ds.DATA_PATH, index=False)
                _, rep = ds.train(df)
            st.session_state["cls_report"] = rep
            st.cache_data.clear()
            st.success(f"Macro-F1 {rep['macro_f1']:.2f} · balanced acc. {rep['balanced_accuracy']:.2f}")
        st.write("Model file:", "✅ present" if sim.CLASSIFIER_PATH.exists() else "— not trained yet")
        st.markdown("\n".join(f"- **{CLASS_SHORT[i]}** – `{n}`" for i, n in enumerate(CLASS_NAMES)))
    with right:
        rep = st.session_state.get("cls_report")
        meta = ROOT / "models" / "metadata" / "flight_profile_gbm.json"
        if rep is None and meta.exists():
            import json
            rep = json.loads(meta.read_text())
        if rep:
            st.caption(f"Held-out windows: {rep['n_test']} · macro-F1 {rep['macro_f1']:.2f} · "
                       f"balanced accuracy {rep['balanced_accuracy']:.2f}")
            st.plotly_chart(plots.confusion(rep["confusion"], CLASS_SHORT), width="stretch")
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
        agg = df.drop(columns="seed").groupby("policy").agg(["mean", "std"]).round(3)
        st.dataframe(agg, width="stretch")
        metric = st.selectbox("Metric", [c for c in df.columns if c not in ("policy", "seed")])
        import plotly.express as px
        fig = px.box(df, x="policy", y=metric, points="all", template="plotly_dark")
        fig.update_layout(paper_bgcolor="#0E1417", plot_bgcolor="#151D21", height=380)
        st.plotly_chart(fig, width="stretch")
        st.download_button("Download CSV", df.to_csv(index=False), "experiment.csv", "text/csv")
    if len(TRAINED) < 2:
        missing = [n for n in ("mappo", "ippo") if n not in TRAINED]
        st.info(f"Not trained yet: {', '.join(missing)}. Train with "
                "`python -m src.services.training_service` (add `--ippo` for IPPO); "
                "checkpoints in models/checkpoints/ appear here automatically.")


# ============================================================== About
with tab_about:
    st.markdown((ROOT / "doc" / "architecture" / "overview.md").read_text(encoding="utf-8")
                if (ROOT / "doc" / "architecture" / "overview.md").exists() else "See README.md")
