"""Streamlit views of the episode analysis (Dataset Replay, Sections C-E).

UI layer only: every number comes from ``src/services/episode_analysis_service.py``.
Layout rule for every panel: a centred row of bordered, equal-height summary cards (``st.metric`` with a
``help`` tooltip and the min–max range as a muted sub-line), then grouped detail
tables with one column per statistic and a one-sentence description per row.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Metric cards: one bordered tile per st.metric, all the same minimum height, with label (+ help icon),
# value and sub-line stacked on one vertical centre line and the stack centred vertically, so a card
# without a sub-line lines up with its neighbours. Each part is shrunk to its content and centred with
# auto margins, which works whatever display mode Streamlit gives the wrappers; !important beats the
# component's own emotion styles.
METRIC_CARD_MIN_HEIGHT = "7.25rem"
METRIC_CSS = """<style>
[data-testid="stMetric"]{box-sizing:border-box;width:100%;height:100%;min-height:%MINH%;
  padding:.7rem .6rem;text-align:center;display:flex !important;flex-direction:column !important;
  justify-content:center !important;align-items:center !important;
  background:rgba(255,255,255,.035);border:1px solid rgba(250,250,250,.14);border-radius:.6rem}
[data-testid="stMetric"] > div{display:flex !important;flex-direction:column !important;align-items:center !important;
  justify-content:center !important;width:100%;gap:.15rem}
[data-testid="stMetricLabel"]{display:flex !important;justify-content:center !important;align-items:center;
  width:fit-content !important;max-width:100%;margin-left:auto !important;margin-right:auto !important}
[data-testid="stMetricLabel"] > div{justify-content:center !important;width:auto !important;flex:0 1 auto !important}
/* the (?) help icon hangs just right of the label, so the label text itself sits on the centre line */
[data-testid="stMetricLabel"]{position:relative}
[data-testid="stMetricLabel"] > :last-child:not(:first-child){position:absolute !important;left:100%;top:50%;
  transform:translateY(-50%);margin-left:.15rem}
[data-testid="stMetricValue"]{display:flex !important;justify-content:center !important;text-align:center;width:100%}
[data-testid="stMetric"] div:has(> [data-testid="stMetricDelta"]){display:flex !important;flex-direction:column;
  align-items:center !important;width:100%}
[data-testid="stMetricDelta"]{margin-left:auto !important;margin-right:auto !important;justify-content:center !important}
[data-testid="stMetricDeltaDescription"]{text-align:center}
[data-testid="stMetric"] [data-testid="stMarkdownContainer"] p{text-align:center}
/* card rows: equal-height columns and every wrapper between column and card stretched, so cards whose
   label wraps to two lines still match their neighbours' height */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]){align-items:stretch}
[data-testid="stColumn"]:has([data-testid="stMetric"]){display:flex;flex-direction:column}
[data-testid="stColumn"]:has([data-testid="stMetric"]) > [data-testid="stVerticalBlock"],
[data-testid="stColumn"] [data-testid="stVerticalBlock"]:has(> [data-testid="stElementContainer"] [data-testid="stMetric"]){flex:1 1 auto}
[data-testid="stElementContainer"]:has([data-testid="stMetric"]){flex:1 1 auto;display:flex}
[data-testid="stElementContainer"]:has([data-testid="stMetric"]) > *{flex:1 1 auto}
</style>""".replace("%MINH%", METRIC_CARD_MIN_HEIGHT)


def inject_metric_css():
    st.markdown(METRIC_CSS, unsafe_allow_html=True)


# Card grid: every card row is laid out on the same CARD_GRID_COLS-wide grid, so a card has the same width
# in every row. A row with fewer cards is centred: with an even shortfall the cards take the middle grid
# slots (exactly under the cards of a full row); with an odd one, equal half-slot spacers sit on both sides.
CARD_GRID_COLS = 6


def card_row(n: int, grid: int = CARD_GRID_COLS) -> list:
    """Return ``n`` centred Streamlit columns for metric cards on a ``grid``-column layout."""
    if n >= grid:
        return list(st.columns(n))
    pad = grid - n
    if pad % 2 == 0:
        cols = st.columns(grid)
        return list(cols[pad // 2: pad // 2 + n])
    side = pad / 2
    cols = st.columns([side] + [1] * n + [side])
    return list(cols[1:-1])


DIST_COLUMNS =["Metric", "Unit", "Min", "Median (p50)", "p90", "Max", "Description"]
_DIST_CONFIG = {
    "Metric": st.column_config.TextColumn("Metric", width="medium"),
    "Unit": st.column_config.TextColumn("Unit", width="small", alignment="center"),
    # the four statistics share one width and right alignment, so their digits line up under the headers
    **{c: st.column_config.NumberColumn(c, width="small", alignment="right")
       for c in ("Min", "Median (p50)", "p90", "Max")},
    "Description": st.column_config.TextColumn("Description", width="large"),
}
_PCT = lambda label, help_=None: st.column_config.ProgressColumn(  # noqa: E731
    label, min_value=0, max_value=100, format="%.0f%%", help=help_)


def _fmt(v, digits: int = 1) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float) and abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}f}"


def _card(col, label: str, stat: Optional[dict], unit: str, help_: str, digits: int = 1, key: str = "p50"):
    """Median (or another key) as the value, min–max as a muted sub-line."""
    if not stat:
        col.metric(label, "N/A", help=help_)
        return
    col.metric(label, f"{_fmt(stat[key], digits)} {unit}".strip(),
               delta=f"range {_fmt(stat['min'], digits)} – {_fmt(stat['max'], digits)} {unit}".strip(),
               delta_color="off", delta_arrow="off", help=help_)


def _value_card(col, label: str, value: str, help_: str, sub: Optional[str] = None):
    if sub:
        col.metric(label, value, delta=sub, delta_color="off", delta_arrow="off", help=help_)
    else:
        col.metric(label, value, help=help_)


def _dist_row(metric: str, unit: str, stat: Optional[dict], digits: int, description: str) -> dict:
    r = (lambda k: round(float(stat[k]), digits)) if stat else (lambda k: None)
    return {"Metric": metric, "Unit": unit, "Min": r("min"), "Median (p50)": r("p50"), "p90": r("p90"),
            "Max": r("max"), "Description": description}


def _dist_table(rows: list):
    st.dataframe(pd.DataFrame(rows, columns=DIST_COLUMNS), column_config=_DIST_CONFIG,
                 width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Section C: track states, signal physics, EKF information
# ---------------------------------------------------------------------------

def render_track_states(stb: dict):
    c = stb["counts"]
    age = stb["coasting_age_s"]
    cols = card_row(4)
    _value_card(cols[0], "Fused Track Steps", f"{stb['total_track_steps']}",
                "Track-steps in the time window: one row per fused track per simulation step. "
                "Every share below uses this number as its denominator.")
    _value_card(cols[1], "Confirmed", f"{stb['confirmed_steps']} ({_fmt(stb['confirmed_pct'])}%)",
                "Steps of tracks that passed the confirmation rule (enough hits).",
                f"{c['confirmed_updated']} updated · {c['confirmed_coasting']} coasting")
    _value_card(cols[2], "Tentative", f"{stb['tentative_steps']} ({_fmt(stb['tentative_pct'])}%)",
                "Steps of candidate tracks not yet confirmed; many come from false alarms.",
                f"{c['tentative_updated']} updated · {c['tentative_coasting']} coasting")
    _value_card(cols[3], "Coasting", f"{stb['coasting_steps']} ({_fmt(stb['coasting_pct'])}%)",
                "Steps in which a track received no measurement and was only predicted. "
                "The sub-line gives the time since its last update.",
                (f"{_fmt(age['min'])} – {_fmt(age['max'])} s since update · {stb['coasting_over_5s']} > 5 s"
                 if age else None))


def render_signal_physics(phys: dict, ekf: dict):
    if not phys.get("total_reports"):
        st.info("No radar reports in this time window.")
        return
    nz = phys["noise"]
    cols = card_row(5)
    _card(cols[0], "Azimuth Cross-Range", nz["cross_range_az_m"], "m",
          "Position error across the line of sight caused by azimuth noise: range × σaz. "
          "It grows with distance even when the angle noise is small.", 0)
    _card(cols[1], "Radial Velocity |ṙ|", phys["abs_radial_velocity_mps"], "m/s",
          "Measured closing / opening speed along the line of sight.", 0)
    _card(cols[2], "Doppler Shift |fD|", phys["abs_doppler_hz"], "Hz",
          "Doppler frequency of the echo, fD = −2ṙ/λ.", 0)
    _card(cols[3], "Estimated RCS", phys["rcs_hat_dbsm"], "dBsm",
          "Radar cross-section recovered from the radar equation with the measured SNR and range, "
          "assuming beam-centre gain: a lower bound. False alarms are included.", 1)
    _card(cols[4], "Info Gain / Update", ekf["information_gain_bits"], "bits",
          "Uncertainty removed by one EKF measurement update: ½·log2(det Pprior / det Ppost).", 1)

    tabs = st.tabs(["Measurement Noise", "Signal & Target Physics", "EKF Covariance & Information Gain"])
    with tabs[0]:
        _dist_table([
            _dist_row("Range Noise (σr)", "m", nz["sigma_range_m"], 2,
                      "Standard deviation of the range measurement; shrinks with SNR."),
            _dist_row("Azimuth Noise (σaz)", "mrad", nz["sigma_azimuth_mrad"], 3,
                      "Angular noise in the horizontal plane, from beamwidth and SNR."),
            _dist_row("Elevation Noise (σel)", "mrad", nz["sigma_elevation_mrad"], 3,
                      "Angular noise in the vertical plane."),
            _dist_row("Azimuth Cross-Range", "m", nz["cross_range_az_m"], 0,
                      "Horizontal position error across the beam: range × σaz."),
            _dist_row("Elevation Cross-Range", "m", nz["cross_range_el_m"], 0,
                      "Vertical position error across the beam: range × σel."),
            _dist_row("Radial Velocity Noise (σṙ)", "m/s", nz["sigma_radial_velocity_mps"], 3,
                      "Noise of the measured line-of-sight speed; set by dwell time and SNR."),
            _dist_row("Doppler Noise (σfD)", "Hz", nz["sigma_doppler_hz"], 1,
                      "The same noise expressed in frequency: 2σṙ/λ."),
        ])
    with tabs[1]:
        _dist_table([
            _dist_row("Signal-to-Noise Ratio", "dB", phys["snr_db"], 1,
                      "Echo strength above receiver noise; drives detection and measurement accuracy."),
            _dist_row("Range", "m", phys["range_m"], 0, "Measured distance from the radar to the report."),
            _dist_row("Radial Velocity |ṙ|", "m/s", phys["abs_radial_velocity_mps"], 1,
                      "Speed along the line of sight, without sign."),
            _dist_row("Doppler Shift |fD|", "Hz", phys["abs_doppler_hz"], 0, "Echo frequency shift, without sign."),
            _dist_row("Estimated RCS", "dBsm", phys["rcs_hat_dbsm"], 1,
                      "Radar-equation estimate of target size (beam-centre gain: a lower bound)."),
        ])
        st.caption(f"{phys['total_reports']} reports in the window. {phys['rcs_note'].capitalize()}.")
    with tabs[2]:
        _dist_table([
            _dist_row("Position Uncertainty Drop (Δσpos)", "m", ekf["delta_sigma_pos_m"], 0,
                      "How much one update shrinks the RMS position standard deviation."),
            _dist_row("Position Covariance Drop (Δtr P)", "m²", ekf["delta_trace_pos_m2"], 0,
                      "Reduction of the trace of the position covariance per update."),
            _dist_row("Information Gain", "bits", ekf["information_gain_bits"], 2,
                      "Entropy removed from the 6-D state per update."),
            _dist_row("Coasting Growth (Δσpos)", "m / step", ekf["coasting_sigma_growth_m_per_step"], 1,
                      "How fast position uncertainty grows while a track gets no measurement."),
        ])
        st.caption(f"{ekf['updates']} update steps. The prior is rebuilt from the previous posterior with the "
                   f"tracker's constant-velocity model (process noise q = {ekf['process_noise_q']:g}).")


# ---------------------------------------------------------------------------
# Section D: radar resources and decision inspector
# ---------------------------------------------------------------------------

def render_resources(res: dict):
    if res["radars"] is None:
        st.warning(f"Radar energy / thermal / channel state: {res['source']}")
        return
    rs = res["radars"]
    cols = card_row(6)
    _value_card(cols[0], "Total Energy", f"{sum(r['energy_total_j'] for r in rs):,.0f} J",
                "Energy used by all radars over the episode (transmit + processing).",
                f"{min(r['energy_total_j'] for r in rs):,.0f} – {max(r['energy_total_j'] for r in rs):,.0f} J per radar")
    duty = [r["duty_cycle_pct"] for r in rs]
    _value_card(cols[1], "Duty Cycle", f"{sum(duty) / len(duty):.1f}%",
                "Share of time the radars were radiating (dwell time / episode time), averaged over radars.",
                f"{min(duty):.1f} – {max(duty):.1f}% per radar")
    _value_card(cols[2], "Min Energy Reserve", f"{min(r['energy_reserve_min_pct'] for r in rs):.0f}%",
                "Lowest battery / capacitor reserve reached by any radar; below 15% the policy forces rest.")
    _value_card(cols[3], "Min Thermal Margin", f"{min(r['thermal_margin_min_pct'] for r in rs):.0f}%",
                "Smallest remaining headroom to the maximum transmitter temperature (100% = ambient).",
                f"peak {max(r['heat_peak_k'] for r in rs):.1f} K")
    chan = [r["channel_utilisation_mean_pct"] for r in rs]
    _value_card(cols[4], "Channel Use", f"{sum(chan) / len(chan):.0f}%",
                "Average share of each radar's tracking channels busy with recently updated tracks.",
                f"{min(chan):.0f} – {max(chan):.0f}% per radar")
    _value_card(cols[5], "Load Fairness (Jain)", f"{res['jain_fairness_track_load']}",
                "Jain index of the tracking load across radars: 1 = perfectly balanced, 1/N = one radar does all.",
                f"energy fairness {res['jain_fairness_energy']}")
    df = pd.DataFrame([{
        "Radar": r["radar"], "Energy (J)": r["energy_total_j"], "Energy / Step (J)": r["energy_mean_j_per_step"],
        "Duty Cycle": r["duty_cycle_pct"], "Min Reserve": r["energy_reserve_min_pct"],
        "Min Thermal Margin": r["thermal_margin_min_pct"], "Channel Use": r["channel_utilisation_mean_pct"],
        "Peak Heat (K)": r["heat_peak_k"], "Track Load (s)": r["track_load_s"],
        "Modes Flown": " · ".join(f"{k} {v}" for k, v in r["mode_steps"].items()),
    } for r in rs])
    with st.expander("Per-Radar Resource Breakdown"):
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "Duty Cycle": _PCT("Duty Cycle", "Radiating time / episode time"),
            "Min Reserve": _PCT("Min Reserve", "Lowest energy reserve reached"),
            "Min Thermal Margin": _PCT("Min Thermal Margin", "Smallest headroom to the temperature limit"),
            "Channel Use": _PCT("Channel Use", "Mean share of busy tracking channels"),
            "Modes Flown": st.column_config.TextColumn(
                "Modes Flown", help="Modes actually flown, including the simulator's fall-backs "
                                    "(empty slot / out of coverage → SEARCH, energy or heat limit → IDLE)"),
        })
        st.caption(f"Source: {res['source']}.")


def render_decision(dec: dict):
    if "error" in dec:
        st.warning(dec["error"])
        return
    cands = dec["candidates"]
    me = dec["radar"]
    cover_due = [c for c in cands if c["coverable_by_this_radar"] and (c["due"] or c["needs_confirmation_dwell"])]
    cols = card_row(5)
    _value_card(cols[0], "Requested Action", dec["requested_action"],
                "Action the policy chose for this radar, decided on the state before this step.",
                f"t = {dec['decision_time_s']:.1f} s · {dec['policy']}")
    _value_card(cols[1], "Mode Flown", dec["flown_mode"] or "N/A",
                "Mode the simulator actually executed; differs from the request when the slot was empty, "
                "the track was out of coverage, or energy / heat forced a rest.")
    _value_card(cols[2], "Tracks in Table", f"{len(cands)}",
                "Tracks in the shared track table the policy saw.",
                f"{sum(c['coverable_by_this_radar'] for c in cands)} inside {me}'s FOV and range gate")
    _value_card(cols[3], "Due & Coverable", f"{len(cover_due)}",
                f"Tracks that needed a dwell (unconfirmed, older than the revisit time or too uncertain) "
                f"and that {me} could reach.")
    _value_card(cols[4], "Energy Reserve", f"{dec['energy_reserve_pct']:.0f}%",
                "Radar energy reserve at decision time; below 15% the policy rests the radar.",
                f"thermal margin {dec['thermal_margin_pct']:.0f}%")
    st.markdown("**Numeric reason**  \n" + "  \n".join(f"• {r}" for r in dec["reasons"]))
    reproduced = "—" if dec["policy_reproduced"] is None else f"{dec['policy_reproduced']:.0%}"
    st.caption(f"Reproduced by re-simulation (verified: {dec['replay_verified']}; policy actions reproduced: {reproduced}).")
    if not cands:
        st.caption("The shared track table was empty at this decision time.")
        return
    df = pd.DataFrame([{
        "Slot": c["slot"], "Track": f"T{c['track_id']}", "Confirmed": c["confirmed"],
        "σpos (m)": c["sigma_pos_m"], "Age (s)": c["age_s"], "Hits": c["hits"], "Due": c["due"],
        "Confirm Dwell": c["needs_confirmation_dwell"], "Range (m)": c["range_m"], "Azimuth (°)": c["az_deg"],
        "In FOV": c["in_fov"], "In Range": c["in_range"], "Coverable By": ", ".join(c["coverable_by"]) or "—",
        "Assigned To": ", ".join(c["assigned_to"]) or "—",
    } for c in cands])
    with st.expander("Candidate Tracks at Decision Time", expanded=True):
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "σpos (m)": st.column_config.NumberColumn("σpos (m)", help="RMS position standard deviation of the track",
                                                      format="%.0f"),
            "Age (s)": st.column_config.NumberColumn("Age (s)", help="Time since the track's last measurement update",
                                                     format="%.1f"),
            "Due": st.column_config.CheckboxColumn("Due", help="Needs a revisit: unconfirmed, older than the revisit "
                                                               "time, or σpos above the threshold"),
            "Confirm Dwell": st.column_config.CheckboxColumn("Confirm Dwell", help="Unconfirmed track waiting for its "
                                                                                   "confirmation dwell"),
            "In FOV": st.column_config.CheckboxColumn("In FOV", help=f"Inside {me}'s azimuth field of view"),
            "In Range": st.column_config.CheckboxColumn("In Range", help=f"Inside {me}'s maximum range"),
            "Range (m)": st.column_config.NumberColumn("Range (m)", format="%.0f"),
            "Coverable By": st.column_config.TextColumn("Coverable By", help="Radars whose FOV and range gate hold it"),
            "Assigned To": st.column_config.TextColumn("Assigned To", help="Radars that tracked it in this step"),
        })


# ---------------------------------------------------------------------------
# Section E: ground-truth evaluation (Category C)
# ---------------------------------------------------------------------------

def render_ground_truth(gt: dict):
    dm, tk, rw = gt["detection"], gt["tracking"], gt["reward"]
    tabs = st.tabs(["Tracking Accuracy", "Detection & Terrain Masking", "Track Life Cycle", "Reward Breakdown"])
    with tabs[0]:
        cols = card_row(4)
        _value_card(cols[0], "Position RMSE", f"{_fmt(tk['position_rmse_m'], 0)} m",
                    "Root-mean-square distance between confirmed tracks and the true targets they follow.",
                    f"{tk['matched_target_steps']} matched target-steps")
        _value_card(cols[1], "Velocity RMSE", f"{_fmt(tk['velocity_rmse_mps'], 0)} m/s",
                    "Root-mean-square error of the track velocity against the true velocity.")
        _value_card(cols[2], "Position NEES", _fmt(tk["nees_pos_mean"], 1),
                    "Normalised estimation error squared: error weighted by the filter's own covariance. "
                    "A consistent filter averages 3 (position); larger means over-confident.",
                    f"{_fmt(tk['nees_pos_within_95_pct'], 0)}% inside the 95% bound")
        _value_card(cols[3], "State NEES", _fmt(tk["nees_state_mean"], 1),
                    "NEES of the full position + velocity state; a consistent filter averages 6.",
                    f"{_fmt(tk['nees_state_within_95_pct'], 0)}% inside the 95% bound")
        if not tk["matched_target_steps"]:
            st.caption("No confirmed track was matched to a target in this episode, so accuracy is not available.")
    with tabs[1]:
        cols = card_row(5)
        _value_card(cols[0], "Detection Probability", _fmt(dm["pd_overall"], 2),
                    "Share of all target-steps with at least one true detection.")
        _value_card(cols[1], "PD given Line of Sight", _fmt(dm["pd_given_los"], 2), dm["pd_note"] + ".",
                    f"{dm['missed_detections_in_los']} missed in line of sight")
        _value_card(cols[2], "True / False Reports", f"{dm['true_detections']} / {dm['false_alarms']}",
                    "Reports caused by a real target vs. false alarms (noise / clutter).",
                    f"{_fmt(dm['false_alarm_share_pct'], 0)}% false")
        _value_card(cols[3], "False Alarm Rate", f"{_fmt(dm['far_per_second'], 2)} /s",
                    "False-alarm reports per second of episode, all radars together.",
                    f"{_fmt(dm['far_per_radar_step'], 3)} per radar-step")
        _value_card(cols[4], "Line of Sight Clear", f"{_fmt(dm['los_clear_pct'], 0)}%",
                    "Share of target-steps in which at least one radar had range, field of view and an unobstructed "
                    "line of sight.",
                    f"terrain {_fmt(dm['terrain_blocked_pct'], 0)}% · out of coverage {_fmt(dm['out_of_coverage_pct'], 0)}%")
        df = pd.DataFrame([{"Target": k, "LOS Clear": v["clear"], "Terrain Blocked": v["terrain"],
                            "Out of Coverage": v["out_of_coverage"]} for k, v in dm["per_target_los_pct"].items()])
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "LOS Clear": _PCT("LOS Clear", "Some radar sees the target"),
            "Terrain Blocked": _PCT("Terrain Blocked", "In range and FOV of a radar, but hidden by terrain"),
            "Out of Coverage": _PCT("Out of Coverage", "Outside every radar's range / field of view"),
        })
    with tabs[2]:
        cols = card_row(3)
        _value_card(cols[0], "Track Completeness", f"{_fmt(tk['track_completeness_pct'], 0)}%",
                    "Share of target-steps in which a confirmed track followed the target.")
        _value_card(cols[1], "Track Drops", f"{tk['track_drops']}",
                    "Times the track following a target was deleted by the tracker.")
        _value_card(cols[2], "Identity Switches", f"{tk['id_switches']}",
                    "Times a target changed from one track ID to another.")
        df = pd.DataFrame([{"Target": k, "Completeness": v["completeness_pct"], "First in LOS (s)": v["first_in_los_s"],
                            "First Tracked (s)": v["first_tracked_s"], "Initiation Latency (s)": v["initiation_latency_s"]}
                           for k, v in tk["per_target"].items()])
        st.dataframe(df, width="stretch", hide_index=True, column_config={
            "Completeness": _PCT("Completeness", "Share of steps the target was tracked"),
            "Initiation Latency (s)": st.column_config.NumberColumn(
                "Initiation Latency (s)", help="First confirmed track minus first time in line of sight", format="%.1f"),
        })
    with tabs[3]:
        if rw["terms"] is None:
            st.warning(f"Reward component breakdown: {rw['source']}")
            return
        terms = sorted(rw["terms"], key=lambda t: t["contribution_sum"])
        worst = terms[0]
        cols = card_row(3)
        _value_card(cols[0], "Episode Reward", f"{rw['reward_sum']:.1f}",
                    "Sum of the team reward over the episode (env._team_reward).",
                    f"{rw['reward_mean']:.3f} per step")
        _value_card(cols[1], "Largest Penalty", worst["term"].replace("_", " "),
                    "Reward term that cost the most over the episode.", f"{worst['contribution_sum']:.1f}")
        _value_card(cols[2], "Reconstruction Error", f"{rw['max_reconstruction_error']:.1e}",
                    "Largest per-step difference between the recorded reward and the sum of its terms (0 = exact).")
        fig = go.Figure(go.Bar(x=[t["contribution_sum"] for t in terms],
                               y=[t["term"].replace("_", " ") for t in terms], orientation="h",
                               marker_color=["#ff6b6b" if t["contribution_sum"] < 0 else "#4FD1C5" for t in terms]))
        fig.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=30, b=10),
                          title="Contribution over the episode (sign · weight · term)",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        # (no expander here: this view is rendered inside the Section E expander, and Streamlit forbids nesting)
        st.markdown("**Reward Term Details**")
        st.dataframe(pd.DataFrame([{
            "Term": t["term"].replace("_", " "), "Weight": t["weight_key"], "w": t["weight"],
            "Raw Mean": t["raw_mean"], "Contribution / Step": t["contribution_mean"],
            "Contribution (Episode)": t["contribution_sum"]} for t in rw["terms"]]),
            width="stretch", hide_index=True, column_config={
                "w": st.column_config.NumberColumn("w", help="Weight from the scenario's reward block"),
                "Raw Mean": st.column_config.NumberColumn("Raw Mean", help="Mean unweighted term per step", format="%.4f"),
                "Contribution / Step": st.column_config.NumberColumn(
                    "Contribution / Step", help="sign · weight · term, averaged over steps", format="%.4f"),
                "Contribution (Episode)": st.column_config.NumberColumn(
                    "Contribution (Episode)", help="sign · weight · term, summed over the episode", format="%.2f"),
            })
        st.caption(f"Source: {rw['source']}.")
