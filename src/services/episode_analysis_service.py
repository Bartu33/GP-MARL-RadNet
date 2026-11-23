"""Episode analysis for the Dataset Replay screen: signal physics, EKF information,
radar resource management, decision context and ground-truth evaluation.

Information layers (as in ``radar_information_service``):

* Category A -- radar reports (z, R, SNR, Doppler) and radar parameters.
* Category B -- tracker output (x, P, status) and radar resource state.
* Category C -- ground truth (truth table, report annotations, terrain LOS);
  used only by :func:`ground_truth_metrics`, which the UI shows in its own,
  explicitly labelled panel.

Data sources.  Schema v3 records carry the per-step radar resources and the
unweighted team-reward terms.  v1/v2 records do not; for them the episode is
re-simulated from ``config_snapshot`` + ``episode_seed`` + the recorded actions
(the simulator is deterministic given those), and the replay is accepted only
if its rewards and report counts match the recorded ones step by step.  The
replay also yields the decision context (shared track table, coverage, due
flags, energy reserve) that no schema version stores.  Anything that cannot
be recovered is returned as ``None`` with a reason, never guessed.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core import geometry as geo
from src.core.env import REWARD_TERMS, RadarNetEnv, _cached_terrain, jain_fairness
from src.core.radar import Mode, antenna_gain, los_status, snr_linear
from src.core.scenario import build_radars
from src.models.tracking.ekf import cv_process_noise, cv_transition

CHI2_95 = {3: 7.815, 6: 12.592}          # 95 % quantiles of chi-square, 3 and 6 degrees of freedom
MODE_NAMES = {int(Mode.IDLE): "IDLE", int(Mode.SEARCH): "SEARCH", int(Mode.TRACK): "TRACK",
              int(Mode.OFFLINE): "OFFLINE", int(Mode.INTERLEAVED): "INTERLEAVED"}


def stats(values, digits: int = 3) -> Optional[dict]:
    """min / p50 / p90 / max / mean of finite values, or None when there are none."""
    a = np.asarray([v for v in np.ravel(np.asarray(values, float)) if np.isfinite(v)], float)
    if a.size == 0:
        return None
    r = lambda x: round(float(x), digits)          # noqa: E731
    return {"n": int(a.size), "min": r(a.min()), "p50": r(np.median(a)), "p90": r(np.percentile(a, 90)),
            "max": r(a.max()), "mean": r(a.mean())}


def _pct(num, den) -> Optional[float]:
    return round(100.0 * num / den, 2) if den else None


def _radar_params(records) -> list:
    """Radar parameters with the terrain-adjusted antenna height, as the simulator uses them."""
    sc = records["metadata"]["config_snapshot"]["scenario"]
    params = build_radars(sc)
    terrain = _cached_terrain(sc)
    for p in params:
        x, y, mast = p.position
        p.position = (x, y, float(terrain.height(x, y)) + mast)
    return params


def _in_window(t, t_range):
    return t_range is None or (t_range[0] <= t <= t_range[1])


# ---------------------------------------------------------------------------
# Category B: fused track states
# ---------------------------------------------------------------------------

def track_state_breakdown(records: dict, source: str = "fused", t_range=None) -> dict:
    """Track-step counts by confirmation and update state; every share has the same denominator.

    A track step is *coasting* when the track got no measurement update at that
    step (status suffix ``_coasting``, i.e. last update earlier than the step time).
    """
    rows = [tr for tr in records.get("tracks", []) if tr["source"] == source and _in_window(tr["t_s"], t_range)]
    total = len(rows)
    counts = {"confirmed_updated": 0, "confirmed_coasting": 0, "tentative_updated": 0, "tentative_coasting": 0}
    ages = []
    for tr in rows:
        age = max(0.0, float(tr["t_s"] - tr["last_update_t_s"]))
        coasting = str(tr["status"]).endswith("_coasting") or age > 1e-9
        key = ("confirmed" if tr["confirmed"] else "tentative") + ("_coasting" if coasting else "_updated")
        counts[key] += 1
        if coasting:
            ages.append(age)
    confirmed = counts["confirmed_updated"] + counts["confirmed_coasting"]
    coasting = counts["confirmed_coasting"] + counts["tentative_coasting"]
    return {
        "source": source, "total_track_steps": total, "counts": counts,
        "confirmed_steps": confirmed, "tentative_steps": total - confirmed, "coasting_steps": coasting,
        "confirmed_pct": _pct(confirmed, total), "tentative_pct": _pct(total - confirmed, total),
        "coasting_pct": _pct(coasting, total),
        "shares_pct": {k: _pct(v, total) for k, v in counts.items()},
        "coasting_age_s": stats(ages, 2),
        "coasting_over_5s": int(sum(a > 5.0 for a in ages)),
    }


# ---------------------------------------------------------------------------
# Category A: measurement and signal physics (no ground truth)
# ---------------------------------------------------------------------------

def _report_modes(records) -> Dict[Tuple[int, int], int]:
    """(step, radar) -> mode actually flown (recorded v3 resources, else the requested action)."""
    res = records.get("radar_resources")
    if res:
        return {(r["step_index"], r["radar_id"]): int(r["mode"]) for r in res}
    out = {}
    for tr in records.get("transitions", []):
        for i, a in enumerate(np.asarray(tr["requested_actions"]).tolist()):
            out[(tr["step_index"], i)] = int(Mode.IDLE) if a == 0 else int(Mode.SEARCH) if a == 1 else int(Mode.TRACK)
    return out


def measurement_physics(records: dict, radar_id: Optional[int] = None, t_range=None) -> dict:
    """Noise breakdown, Doppler / radial velocity, SNR, range and recovered RCS of radar reports.

    RCS is recovered from the radar equation, sigma_hat = SNR / SNR(r, 1 m^2, G0, n_pulses),
    assuming the target sat at the beam centre (gain G0).  A target off the beam axis
    had a lower gain, so sigma_hat is a lower bound of its mean RCS.  The pulse count
    is that of the mode the radar flew (search or track profile).  False alarms are
    included: they cannot be told apart without ground truth.
    """
    reps = [r for r in records.get("reports", [])
            if (radar_id is None or r["radar_id"] == radar_id) and _in_window(r["t_s"], t_range)]
    if not reps:
        return {"total_reports": 0}
    params = _radar_params(records)
    modes = _report_modes(records)
    z = np.array([r["z"] for r in reps])
    R = np.array([r["R"] for r in reps])
    sig = np.sqrt(np.clip(np.diagonal(R, axis1=1, axis2=2), 0.0, None))
    lam = np.array([params[r["radar_id"]].wavelength for r in reps])
    rng_m = z[:, 0]
    rcs = []
    for r in reps:
        p = params[r["radar_id"]]
        mode = modes.get((r["step_index"], r["radar_id"]), int(Mode.SEARCH))
        profile = p.profiles[Mode.TRACK if mode in (int(Mode.TRACK), int(Mode.INTERLEAVED)) else Mode.SEARCH]
        unit = float(snr_linear(r["z"][0], 1.0, antenna_gain(0.0, 0.0, p), p, profile.n_pulses))
        rcs.append(10 * math.log10(max(10 ** (r["snr_db"] / 10) / unit, 1e-12)))
    return {
        "total_reports": len(reps),
        "noise": {
            "sigma_range_m": stats(sig[:, 0], 2),
            "sigma_azimuth_mrad": stats(sig[:, 1] * 1e3, 3),
            "sigma_elevation_mrad": stats(sig[:, 2] * 1e3, 3),
            "cross_range_az_m": stats(rng_m * sig[:, 1], 1),        # r * sigma_az
            "cross_range_el_m": stats(rng_m * sig[:, 2], 1),        # r * sigma_el
            "sigma_radial_velocity_mps": stats(sig[:, 3], 3),
            "sigma_doppler_hz": stats(2 * sig[:, 3] / lam, 2),      # f_D = -2 r_dot / lambda
        },
        "abs_radial_velocity_mps": stats(np.abs(z[:, 3]), 1),
        "abs_doppler_hz": stats(np.abs([r["doppler_hz"] for r in reps]), 1),
        "snr_db": stats([r["snr_db"] for r in reps], 2),
        "range_m": stats(rng_m, 0),
        "rcs_hat_dbsm": stats(rcs, 2),
        "rcs_note": "beam-centre gain assumed: lower bound of the mean RCS; includes false alarms",
    }


# ---------------------------------------------------------------------------
# Category B: EKF information gain
# ---------------------------------------------------------------------------

def ekf_information_gain(records: dict, source: str = "fused", t_range=None) -> dict:
    """Uncertainty reduction per measurement update of the tracker.

    For each update step the prior is rebuilt from the previous recorded posterior
    with the tracker's own CV prediction (F P F' + Q(dt, q)); then
    delta sigma_pos = sigma_pos(prior) - sigma_pos(post), delta tr(P_pos), and the
    information gain 0.5 log2(det P_prior / det P_post) in bits (6-D state).
    """
    q = float(records["metadata"]["config_snapshot"]["scenario"].get("tracker", {}).get("q", 40.0))
    by_track: Dict[int, list] = {}
    for tr in records.get("tracks", []):
        if tr["source"] == source:
            by_track.setdefault(tr["track_id"], []).append(tr)
    d_sigma, d_trace, bits, coast_growth = [], [], [], []
    for rows in by_track.values():
        rows.sort(key=lambda r: r["step_index"])
        for prev, cur in zip(rows, rows[1:]):
            if not _in_window(cur["t_s"], t_range):
                continue
            dt = float(cur["t_s"] - prev["t_s"])
            if dt <= 0:
                continue
            F = cv_transition(dt)
            prior = F @ prev["P"] @ F.T + cv_process_noise(dt, q)
            post = cur["P"]
            s_prior = math.sqrt(max(np.trace(prior[:3, :3]) / 3, 0.0))
            s_post = math.sqrt(max(np.trace(post[:3, :3]) / 3, 0.0))
            if cur["last_update_t_s"] > prev["last_update_t_s"] + 1e-9:          # updated in this step
                d_sigma.append(s_prior - s_post)
                d_trace.append(float(np.trace(prior[:3, :3]) - np.trace(post[:3, :3])))
                sp, lp = np.linalg.slogdet(prior)
                sq, lq = np.linalg.slogdet(post)
                if sp > 0 and sq > 0:
                    bits.append(0.5 * (lp - lq) / math.log(2))
            else:
                coast_growth.append(s_post - math.sqrt(max(np.trace(prev["P"][:3, :3]) / 3, 0.0)))
    return {
        "updates": len(d_sigma),
        "delta_sigma_pos_m": stats(d_sigma, 1),
        "delta_trace_pos_m2": stats(d_trace, 0),
        "information_gain_bits": stats(bits, 2),
        "coasting_sigma_growth_m_per_step": stats(coast_growth, 1),
        "process_noise_q": q,
    }


# ---------------------------------------------------------------------------
# replay (resources, reward terms and decision context for any schema version)
# ---------------------------------------------------------------------------

_REPLAY_CACHE: Dict[str, dict] = {}


def _slot_context(env, policy) -> dict:
    """What the policy saw before deciding: the shared track table and the radars' state."""
    from src.models.baselines.policies import REVISIT_S, STD_DUE_M, _coverable, _due, _low_energy
    confirm_sent = getattr(policy, "confirm_sent", None)
    slots = []
    for s, tr in enumerate(env.slots):
        if tr is None:
            continue
        per_radar = []
        for i, p in enumerate(env.radar_params):
            r, a, e, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
            per_radar.append({"range_m": round(float(r), 0), "az_deg": round(math.degrees(a), 1),
                              "el_deg": round(math.degrees(e), 1), "in_fov": bool(abs(a) <= math.radians(p.fov_az_deg)),
                              "in_range": bool(r <= p.r_max_m), "coverable": bool(_coverable(env, i, tr))})
        fresh = (not tr.confirmed and confirm_sent is not None and confirm_sent.get(tr.id) != tr.hits)
        slots.append({
            "slot": s, "track_id": int(tr.id), "confirmed": bool(tr.confirmed), "hits": int(tr.hits),
            "sigma_pos_m": round(float(tr.ekf.pos_std), 1), "age_s": round(float(env.t - tr.last_update_t), 2),
            "born_t_s": round(float(tr.born_t), 2), "due": bool(_due(env, tr)),
            "needs_confirmation_dwell": bool(fresh), "radars": per_radar,
        })
    radars = [{"energy_frac": round(float(st.energy_j / p.e_max_j), 3),
               "heat_margin": round(float(1 - np.clip((st.heat_k - p.h_ambient_k) / (p.h_max_k - p.h_ambient_k), 0, 1)), 3),
               "low_energy": bool(_low_energy(env, i)), "online": bool(st.online)}
              for i, (p, st) in enumerate(zip(env.radar_params, env.radar_states))]
    return {"t_s": round(float(env.t), 2), "slots": slots, "radars": radars,
            "rules": {"revisit_s": REVISIT_S, "sigma_due_m": STD_DUE_M, "low_energy_frac": 0.15}}


def replay_episode(records: dict) -> dict:
    """Re-simulate the episode with its recorded actions; verified against the records.

    Returns {"verified", "mismatch", "radar_resources", "reward_terms", "contexts",
    "evaluator", "policy_actions_match"}.  Cached per episode id.
    """
    meta = records["metadata"]
    key = str(meta.get("episode_id"))
    if key in _REPLAY_CACHE:
        return _REPLAY_CACHE[key]
    from src.models.baselines.policies import make_policy
    snap = meta["config_snapshot"]
    seed = snap["episode_seed"]
    policy_name = (meta.get("dataset_provenance") or {}).get("policy_name")
    env = RadarNetEnv(snap["scenario"], classifier=None, seed=seed, local_tracks=bool(meta.get("local_tracks_enabled", True)),
                      record_episode=True)
    obs, _ = env.reset(seed=seed)
    policy = None
    if policy_name:
        try:
            policy = make_policy(policy_name)
            policy.reset(env)
        except Exception:                                  # noqa: BLE001 - unknown / learned policy
            policy = None
    contexts, mismatch, policy_match = [], None, []
    reports_by_step: Dict[int, int] = {}
    for r in records.get("reports", []):
        reports_by_step[r["step_index"]] = reports_by_step.get(r["step_index"], 0) + 1
    for tr in records.get("transitions", []):
        contexts.append(_slot_context(env, policy))
        acts = {a: int(v) for a, v, present in zip(env.possible_agents, np.asarray(tr["requested_actions"]).tolist(),
                                                     np.asarray(tr["action_present"]).tolist()) if present}
        if policy is not None:
            proposed = policy.act(env, obs)
            policy_match.append(all(int(proposed.get(a, 1)) == v for a, v in acts.items()))
        n_before = len(env.episode_records["reports"])
        obs, rewards, _, _, _ = env.step(acts)
        n_reports = len(env.episode_records["reports"]) - n_before
        rec_reward = float(np.asarray(tr["rewards"])[0]) if len(tr["rewards"]) else 0.0
        got = float(next(iter(rewards.values()))) if rewards else 0.0
        if mismatch is None and (not math.isclose(got, rec_reward, rel_tol=1e-9, abs_tol=1e-9)
                                 or n_reports != reports_by_step.get(tr["step_index"], 0)):
            mismatch = {"step_index": tr["step_index"], "recorded_reward": rec_reward, "replayed_reward": got,
                        "recorded_reports": reports_by_step.get(tr["step_index"], 0), "replayed_reports": n_reports}
    out = {
        "verified": mismatch is None, "mismatch": mismatch, "policy_name": policy_name,
        "policy_actions_match": (float(np.mean(policy_match)) if policy_match else None),
        "radar_resources": env.episode_records["radar_resources"],
        "reward_terms": env.episode_records["reward_terms"],
        "contexts": contexts, "evaluator": env.evaluator.summary(),
    }
    _REPLAY_CACHE[key] = out
    return out


def _resources(records) -> Tuple[Optional[list], Optional[list], str]:
    """(radar_resources, reward_terms, source) -- recorded (v3), verified replay, or unavailable."""
    if records.get("radar_resources") and records.get("reward_terms"):
        return records["radar_resources"], records["reward_terms"], "recorded (schema v3)"
    try:
        rep = replay_episode(records)
    except Exception as exc:                                # noqa: BLE001
        return None, None, f"N/A (not recorded; replay failed: {type(exc).__name__})"
    if not rep["verified"]:
        return None, None, f"N/A (not recorded; replay diverged at step {rep['mismatch']['step_index']})"
    return rep["radar_resources"], rep["reward_terms"], "replayed from config + seed + actions (verified)"


# ---------------------------------------------------------------------------
# Section D: radar resource management
# ---------------------------------------------------------------------------

def resource_metrics(records: dict) -> dict:
    """Energy, duty cycle, thermal margin, channel use and load fairness per radar."""
    res, _, source = _resources(records)
    params = _radar_params(records)
    if res is None:
        return {"source": source, "radars": None}
    dt = float(records["metadata"]["config_snapshot"]["scenario"]["dt"])
    out, energy_tot, load = [], [], []
    for i, p in enumerate(params):
        rows = sorted([r for r in res if r["radar_id"] == i], key=lambda r: r["step_index"])
        if not rows:
            continue
        used = np.array([r["energy_used_j"] for r in rows])
        dwell = np.array([r["dwell_s"] for r in rows])
        heat = np.array([r["heat_k"] for r in rows])
        margin = 1 - np.clip((heat - p.h_ambient_k) / (p.h_max_k - p.h_ambient_k), 0, 1)
        reserve = np.array([r["energy_j"] for r in rows]) / p.e_max_j
        chan = 1 - np.array([r["channels_free"] for r in rows]) / max(p.c_max, 1)
        modes = [MODE_NAMES.get(int(r["mode"]), str(r["mode"])) for r in rows]
        out.append({
            "radar": p.name, "steps": len(rows),
            "energy_total_j": round(float(used.sum()), 1), "energy_mean_j_per_step": round(float(used.mean()), 2),
            "duty_cycle_pct": round(100 * float(dwell.sum()) / (len(rows) * dt), 2),
            "energy_reserve_min_pct": round(100 * float(reserve.min()), 1),
            "heat_peak_k": round(float(heat.max()), 2), "thermal_margin_min_pct": round(100 * float(margin.min()), 1),
            "channel_utilisation_mean_pct": round(100 * float(chan.mean()), 1),
            "mode_steps": {m: modes.count(m) for m in sorted(set(modes))},
            "track_load_s": round(float(rows[-1]["track_load_s"]), 2),
        })
        energy_tot.append(float(used.sum()))
        load.append(float(rows[-1]["track_load_s"]))
    return {"source": source, "radars": out,
            "jain_fairness_energy": round(jain_fairness(energy_tot), 3) if energy_tot else None,
            "jain_fairness_track_load": round(jain_fairness(load), 3) if load else None}


def reward_breakdown(records: dict) -> dict:
    """Mean weighted contribution of every team-reward term (env._team_reward), checked against the reward."""
    _, terms, source = _resources(records)
    if terms is None:
        return {"source": source, "terms": None}
    w = records["metadata"]["config_snapshot"]["scenario"]["reward"]
    rows = []
    for term, (key, sign) in REWARD_TERMS.items():
        vals = np.array([r[term] for r in terms])
        contrib = sign * float(w.get(key, 0.0)) * vals
        rows.append({"term": term, "weight_key": key, "weight": float(w.get(key, 0.0)),
                     "raw_mean": round(float(vals.mean()), 4), "contribution_mean": round(float(contrib.mean()), 4),
                     "contribution_sum": round(float(contrib.sum()), 3)})
    total = np.array([r["reward"] for r in terms])
    rebuilt = sum(np.array([r[t] for r in terms]) * s * float(w.get(k, 0.0)) for t, (k, s) in REWARD_TERMS.items())
    return {"source": source, "terms": rows, "reward_sum": round(float(total.sum()), 3),
            "reward_mean": round(float(total.mean()), 4),
            "max_reconstruction_error": float(np.max(np.abs(rebuilt - total))) if len(total) else 0.0}


# ---------------------------------------------------------------------------
# Section D: decision inspector
# ---------------------------------------------------------------------------

def decision_context(records: dict, step_index: int, radar_id: int) -> dict:
    """Numeric reason for radar ``radar_id``'s action in transition ``step_index`` (1-based).

    The action of transition k was chosen on the state after step k-1; the
    replay reproduces that state (track table, uncertainty, age, coverage,
    energy) exactly.  The rule text follows the recorded policy.
    """
    trans = {tr["step_index"]: tr for tr in records.get("transitions", [])}
    if step_index not in trans:
        return {"error": f"step {step_index} not in 1..{len(trans)}"}
    tr = trans[step_index]
    acts = np.asarray(tr["requested_actions"]).tolist()
    if radar_id >= len(acts):
        return {"error": f"radar {radar_id + 1} does not exist in this episode ({len(acts)} radars)"}
    action = int(acts[radar_id])
    try:
        rep = replay_episode(records)
    except Exception as exc:                                # noqa: BLE001
        return {"error": f"replay failed: {type(exc).__name__}: {exc}"}
    ctx = rep["contexts"][step_index - 1]
    res = {(r["step_index"], r["radar_id"]): r for r in rep["radar_resources"]}.get((step_index, radar_id))
    flown = MODE_NAMES.get(int(res["mode"])) if res else None
    me = ctx["radars"][radar_id]
    candidates = []
    for s in ctx["slots"]:
        rr = s["radars"][radar_id]
        candidates.append({"slot": s["slot"], "track_id": s["track_id"], "confirmed": s["confirmed"],
                           "sigma_pos_m": s["sigma_pos_m"], "age_s": s["age_s"], "hits": s["hits"],
                           "due": s["due"], "needs_confirmation_dwell": s["needs_confirmation_dwell"],
                           "range_m": rr["range_m"], "az_deg": rr["az_deg"], "in_fov": rr["in_fov"],
                           "in_range": rr["in_range"], "coverable_by_this_radar": rr["coverable"],
                           "coverable_by": [f"R{i + 1}" for i, x in enumerate(s["radars"]) if x["coverable"]],
                           "assigned_to": [f"R{i + 1}" for i, a in enumerate(acts) if a == 2 + s["slot"]]})
    requested = "IDLE" if action == 0 else "SEARCH" if action == 1 else f"TRACK slot {action - 2}"
    reasons = []
    pol = rep.get("policy_name") or "unknown"
    if action == 0:
        reasons.append(f"energy reserve {me['energy_frac']:.0%} < {ctx['rules']['low_energy_frac']:.0%}: forced rest"
                       if me["low_energy"] else "IDLE requested although the energy reserve was sufficient "
                                                f"({me['energy_frac']:.0%})")
    elif action == 1:
        cover_due = [c for c in candidates if c["coverable_by_this_radar"] and (c["due"] or c["needs_confirmation_dwell"])]
        if not candidates:
            reasons.append("the track table was empty: nothing to track, so search")
        elif not cover_due:
            reasons.append("no due / unconfirmed track lay inside this radar's FOV and range gate")
        else:
            taken = [c for c in cover_due if c["assigned_to"]]
            if len(taken) == len(cover_due):
                reasons.append("every coverable due track was already served by another (closer) radar: "
                               + ", ".join(f"T{c['track_id']}→{'/'.join(c['assigned_to'])}" for c in taken))
            if step_index % 3 == 0 and pol.startswith("central_greedy"):
                reasons.append(f"step {step_index} is a search-reservation step of {pol} (every 3rd step one radar keeps searching)")
            if not reasons:
                reasons.append("coverable due tracks existed but the policy kept this radar searching")
    else:
        slot = action - 2
        c = next((c for c in candidates if c["slot"] == slot), None)
        if c is None:
            reasons.append(f"slot {slot} was empty: the simulator fell back to SEARCH")
        else:
            why = ("unconfirmed track: confirmation dwell" if c["needs_confirmation_dwell"] or not c["confirmed"]
                   else f"due for revisit: age {c['age_s']:.1f} s ≥ {ctx['rules']['revisit_s']} s"
                   if c["age_s"] >= ctx["rules"]["revisit_s"]
                   else f"due for revisit: σ_pos {c['sigma_pos_m']:.0f} m > {ctx['rules']['sigma_due_m']:.0f} m")
            reasons.append(f"T{c['track_id']} ({why}); σ_pos {c['sigma_pos_m']:.0f} m, age {c['age_s']:.1f} s")
            if not c["coverable_by_this_radar"]:
                reasons.append("track outside this radar's FOV / range gate: the simulator fell back to SEARCH")
            elif len(c["coverable_by"]) > 1:
                reasons.append(f"closest of the radars that could cover it ({', '.join(c['coverable_by'])}); "
                               f"range {c['range_m']:.0f} m")
    return {
        "step_index": step_index, "decision_time_s": ctx["t_s"], "radar": f"R{radar_id + 1}",
        "policy": pol, "policy_reproduced": rep.get("policy_actions_match"),
        "requested_action": requested, "flown_mode": flown,
        "energy_reserve_pct": round(100 * me["energy_frac"], 1), "thermal_margin_pct": round(100 * me["heat_margin"], 1),
        "candidates": candidates, "reasons": reasons, "replay_verified": rep["verified"],
    }


# ---------------------------------------------------------------------------
# Category C: ground-truth evaluation (evaluation only)
# ---------------------------------------------------------------------------

def _match(targets_p, confirmed):
    """env.match_targets: greedy, gate 3 sigma_pos + 500 m."""
    pairs = []
    for ti, p in targets_p.items():
        for k, tr in enumerate(confirmed):
            d = float(np.linalg.norm(tr["x"][:3] - p))
            sp = math.sqrt(max(np.trace(tr["P"][:3, :3]) / 3, 0.0))
            if d < 3 * sp + 500.0:
                pairs.append((d, ti, k))
    pairs.sort()
    used_t, used_k, out = set(), set(), {}
    for d, ti, k in pairs:
        if ti in used_t or k in used_k:
            continue
        used_t.add(ti)
        used_k.add(k)
        out[ti] = confirmed[k]
    return out


def ground_truth_metrics(records: dict) -> dict:
    """Tracking accuracy, detection, terrain masking, track life cycle and reward terms vs. ground truth."""
    meta = records["metadata"]
    sc = meta["config_snapshot"]["scenario"]
    dt = float(sc["dt"])
    params = _radar_params(records)
    terrain = _cached_terrain(sc)
    truth: Dict[int, Dict[int, dict]] = {}
    for row in records.get("truth", []):
        truth.setdefault(row["step_index"], {})[row["target_id"]] = row
    tracks_by_step: Dict[int, list] = {}
    for tr in records.get("tracks", []):
        if tr["source"] == "fused":
            tracks_by_step.setdefault(tr["step_index"], []).append(tr)
    ann = {a["report_id"]: a for a in records.get("annotations", [])}
    reports = records.get("reports", [])
    steps = sorted(s for s in truth if s >= 1)
    target_ids = sorted({row["target_id"] for row in records.get("truth", [])})

    class _T:                                   # minimal target for los_status
        def __init__(self, p, v):
            self.p, self.v = np.asarray(p, float), np.asarray(v, float)

    # --- detection and terrain masking --------------------------------------
    detected = {(r["step_index"], ann[r["report_id"]]["truth_id"]) for r in reports
                if not ann[r["report_id"]]["is_false_alarm"] and ann[r["report_id"]]["truth_id"] >= 0}
    los = {}                                   # (step, tid) -> 'clear' | 'terrain' | 'out_of_coverage'
    for s in steps:
        for tid, row in truth[s].items():
            st = [los_status(_T(row["p"], row["v"]), p, terrain) for p in params]
            los[(s, tid)] = "clear" if 1 in st else "terrain" if 2 in st else "out_of_coverage"
    n_ts = len(los)
    clear = [k for k, v in los.items() if v == "clear"]
    n_tp = sum(1 for r in reports if not ann[r["report_id"]]["is_false_alarm"])
    n_fa = len(reports) - n_tp
    radar_steps = len(steps) * len(params)
    per_target_los = {}
    for tid in target_ids:
        vals = [los[(s, tid)] for s in steps if (s, tid) in los]
        per_target_los[f"T{tid}"] = {k: _pct(vals.count(k), len(vals)) for k in ("clear", "terrain", "out_of_coverage")}
    detection = {
        "reports_total": len(reports), "true_detections": n_tp, "false_alarms": n_fa,
        "false_alarm_share_pct": _pct(n_fa, len(reports)),
        "pd_overall": round(len(detected) / n_ts, 4) if n_ts else None,
        "pd_given_los": round(sum(k in detected for k in clear) / len(clear), 4) if clear else None,
        "missed_detections_in_los": int(sum(k not in detected for k in clear)),
        "far_per_radar_step": round(n_fa / radar_steps, 4) if radar_steps else None,
        "far_per_second": round(n_fa / (len(steps) * dt), 4) if steps else None,
        "los_clear_pct": _pct(len(clear), n_ts),
        "terrain_blocked_pct": _pct(sum(v == "terrain" for v in los.values()), n_ts),
        "out_of_coverage_pct": _pct(sum(v == "out_of_coverage" for v in los.values()), n_ts),
        "per_target_los_pct": per_target_los,
        "pd_note": "P_D | LOS counts a target-step as an opportunity when some radar has range, FOV and terrain "
                   "line of sight, whether or not a beam was on it",
    }

    # --- tracking accuracy and life cycle -----------------------------------
    pos_sq, vel_sq, nees3, nees6 = [], [], [], []
    assigned = {tid: None for tid in target_ids}
    drops = switches = 0
    first_cov, first_trk, tracked_steps = {}, {}, {tid: 0 for tid in target_ids}
    for s in steps:
        trs = tracks_by_step.get(s, [])
        ids = {tr["track_id"] for tr in trs}
        confirmed = [tr for tr in trs if tr["confirmed"]]
        match = _match({tid: row["p"] for tid, row in truth[s].items()}, confirmed)
        for tid in target_ids:
            prev = assigned[tid]
            if prev is not None and prev not in ids:
                drops += 1
                assigned[tid] = prev = None
            if los.get((s, tid)) == "clear":
                first_cov.setdefault(tid, s)
            tr = match.get(tid)
            if tr is None:
                continue
            if prev is not None and tr["track_id"] != prev:
                switches += 1
            assigned[tid] = tr["track_id"]
            tracked_steps[tid] += 1
            first_trk.setdefault(tid, s)
            row = truth[s][tid]
            e = np.concatenate([tr["x"][:3] - row["p"], tr["x"][3:] - row["v"]])
            pos_sq.append(float(e[:3] @ e[:3]))
            vel_sq.append(float(e[3:] @ e[3:]))
            try:
                nees3.append(float(e[:3] @ np.linalg.solve(tr["P"][:3, :3], e[:3])))
                nees6.append(float(e @ np.linalg.solve(tr["P"], e)))
            except np.linalg.LinAlgError:
                pass
    per_target = {}
    for tid in target_ids:
        lat = ((first_trk[tid] - first_cov[tid]) * dt if tid in first_trk and tid in first_cov
               and first_trk[tid] >= first_cov[tid] else None)
        per_target[f"T{tid}"] = {
            "completeness_pct": _pct(tracked_steps[tid], len(steps)),
            "first_in_los_s": first_cov[tid] * dt if tid in first_cov else None,
            "first_tracked_s": first_trk[tid] * dt if tid in first_trk else None,
            "initiation_latency_s": lat,
        }
    tracking = {
        "matched_target_steps": len(pos_sq),
        "position_rmse_m": round(math.sqrt(np.mean(pos_sq)), 1) if pos_sq else None,
        "velocity_rmse_mps": round(math.sqrt(np.mean(vel_sq)), 1) if vel_sq else None,
        "nees_pos_mean": round(float(np.mean(nees3)), 2) if nees3 else None,
        "nees_pos_within_95_pct": _pct(sum(x <= CHI2_95[3] for x in nees3), len(nees3)),
        "nees_state_mean": round(float(np.mean(nees6)), 2) if nees6 else None,
        "nees_state_within_95_pct": _pct(sum(x <= CHI2_95[6] for x in nees6), len(nees6)),
        "nees_note": "a consistent filter has mean NEES ≈ 3 (position) / 6 (state) and ~95 % of samples "
                     "inside the 95 % chi-square bound",
        "track_drops": drops, "id_switches": switches,
        "track_completeness_pct": _pct(sum(tracked_steps.values()), len(steps) * len(target_ids)),
        "per_target": per_target,
    }
    return {"category": "C (ground truth, evaluation only)", "detection": detection, "tracking": tracking,
            "reward": reward_breakdown(records)}
