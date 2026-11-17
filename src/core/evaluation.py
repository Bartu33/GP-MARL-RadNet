"""Evaluation-only track metrics (uses simulation truth; never fed to agents).

Definitions follow Table 2 of the project proposal and the advisor meeting of
29 Oct 2026.  They are computed per target and per step, then summarised per
episode by :meth:`TrackEvaluator.summary`.

* detection opportunity: a radar is online and transmitting, the target is
  inside its range / field of view with terrain line of sight, and the target
  lies inside one of the beams of that step (half-power width).
* detection loss: an opportunity without a target-originated detection from
  that radar (counted at the radar, before any communication loss).
* valid track: the target is matched to a confirmed fused track (see
  ``RadarNetEnv.match_targets``).
* in coverage: at least one online radar has the target in range / field of
  view with line of sight.
* track loss: the fused track that was following a target is deleted by the
  tracker's confirmation/deletion rules.  A single missed detection is not a
  track loss -- the tracker coasts through missing measurements.
* interruption: a period without a valid track after the target was first
  tracked; reacquisition time is the length of an interruption that ends with
  the target tracked again.
* identity switch: the target becomes matched to a different track id while
  the previous one still exists.
* NEES (normalised estimation error squared): for a matched track with state
  estimate x and covariance P, e = x - x_true (position and velocity) and
  NEES = e^T P^-1 e.  For a consistent filter it is chi-square distributed
  with 6 degrees of freedom, so its mean is about 6 and about 95 % of the
  values lie inside [1.24, 14.45] (Bar-Shalom, Li & Kirubarajan 2001,
  Sec. 5.4).  A mean far above 6 means the tracker is over-confident
  (P too small), far below 6 that it is too cautious.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo

NEES_DOF = 6
NEES_95 = (1.2373, 14.4494)       # two-sided 95 % interval of chi-square(6)
from .radar import los_status

# Shown in the dashboard next to the experiment table: name -> (meaning, unit / denominator).
METRIC_INFO = {
    "pos_rmse_m": ("RMS position error of the fused tracks matched to targets", "m, over matched target-steps"),
    "pos_err_median_m": ("Median position error of matched fused tracks (robust to the few large errors "
                         "of new or coasting tracks that dominate the RMSE)", "m, over matched target-steps"),
    "track_coverage": ("Target-steps with a valid confirmed track, all targets (also out of range)",
                       "fraction of all target-steps"),
    "coverage_in_range": ("Target-steps with a valid track while at least one radar can see the target "
                          "(range, field of view, line of sight)", "fraction of in-coverage target-steps"),
    "detection_loss": ("Detection opportunities (target inside a beam of a transmitting radar, "
                       "with line of sight) that produced no detection", "fraction of opportunities"),
    "track_losses": ("Fused tracks following a target that the tracker deleted", "count per episode"),
    "interruptions": ("Times a target lost its valid track after it had been tracked", "count per episode"),
    "longest_track_s": ("Longest uninterrupted valid-track interval, mean over targets", "s"),
    "interrupted_s": ("Time without a valid track after first acquisition, mean over targets", "s"),
    "reacquire_s": ("Mean length of interruptions that ended with the target tracked again", "s"),
    "correct_class_continuity": ("Target-steps with a valid track whose predicted class is correct",
                                 "fraction of all target-steps"),
    "id_switches": ("A target became matched to a different track while the old one still existed",
                    "count per episode"),
    "nees_mean": ("Mean NEES of matched fused tracks (position + velocity); about 6 for a consistent "
                  "filter, larger = over-confident covariance", "chi-square(6) units, over matched target-steps"),
    "nees_in_95": ("Matched target-steps whose NEES lies inside the 95 % chi-square(6) interval "
                   "[1.24, 14.45]; about 0.95 for a consistent filter", "fraction of matched target-steps"),
    "false_tracks_mean": ("Confirmed tracks not matched to any target", "mean per step"),
    "class_accuracy": ("Predicted class correct, over matched tracks with a class estimate",
                       "fraction of matched target-steps"),
    "sensing_cost": ("Transmit + processing energy per step, normalised by each radar's budget",
                     "mean per step"),
    "reports_per_step": ("Radar reports delivered to the fusion centre (after communication loss)",
                         "mean per step"),
    "return": ("Sum of the team reward over the episode", "reward units"),
    "usage_fairness": ("Jain index of cumulative tracking time per radar (1 = balanced)", "0-1"),
    # task-manager interface (track-update requests answered by the radars' local schedulers)
    "requests_per_step": ("Track-update requests sent by the fusion centre to all radars", "mean per step"),
    "request_accept_rate": ("Requests served in the next step with a full track dwell", "fraction of requests"),
    "request_delay_rate": ("Requests queued and served later, before their deadline", "fraction of requests"),
    "request_partial_rate": ("Requests served with the short interleaved dwell (radar low on energy / heat margin)",
                             "fraction of requests"),
    "request_reject_rate": ("Requests refused by the local scheduler (coverage, queue full, deadline, resources)",
                            "fraction of requests"),
    "request_expired_rate": ("Accepted requests dropped before service (deadline passed or track deleted)",
                             "fraction of requests"),
    "decision_ms": ("Time the coordination policy needs to choose the actions of all radars", "ms per step (CPU)"),
    # structure A: every radar on its own (local single-radar tracks, same detections)
    "single_best_coverage": ("Coverage in range reached by the best single radar using only its own "
                             "local track (structure A); compare with coverage_in_range of the fused track",
                             "fraction of in-coverage target-steps of the network"),
    "single_best_longest_s": ("Longest uninterrupted local track of the best single radar, mean over targets",
                              "s"),
    "single_mean_rmse_m": ("RMS position error of local single-radar tracks, mean over radars",
                           "m, over matched target-steps"),
}


def nees(x, P, p_true, v_true) -> float:
    """Normalised estimation error squared of a 6-D [position, velocity] estimate."""
    e = np.asarray(x, float)[:6] - np.concatenate([p_true, v_true])
    return float(e @ np.linalg.solve(np.asarray(P, float)[:6, :6], e))


def in_beam(meas, beams, p) -> bool:
    """True when the measured direction lies inside one of the beams (half-power width)."""
    _, a, e, _ = meas
    ha, he = np.deg2rad(p.beamwidth_az_deg) / 2, np.deg2rad(p.beamwidth_el_deg) / 2
    return any(abs(geo.wrap_angle(a - b[0])) <= ha and abs(e - b[1]) <= he for b in beams)


def runs(valid) -> dict:
    """Continuity statistics of one target's boolean valid-track series."""
    v = np.asarray(valid, bool)
    longest = cur = 0
    for x in v:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    first = int(np.argmax(v)) if v.any() else len(v)
    gaps, g = [], 0
    for x in v[first:]:
        if x:
            if g:
                gaps.append((g, True))       # interruption that ended with reacquisition
            g = 0
        else:
            g += 1
    if g:
        gaps.append((g, False))              # still interrupted at the end of the episode
    return {
        "longest": longest,
        "interruptions": len(gaps),
        "interrupted": sum(n for n, _ in gaps),
        "reacquired": [n for n, ok in gaps if ok],
    }


class TrackEvaluator:
    def __init__(self, n_targets: int, dt: float):
        self.dt = dt
        self.valid = [[] for _ in range(n_targets)]
        self.in_cov = [[] for _ in range(n_targets)]
        self.correct = [[] for _ in range(n_targets)]
        self.assigned = [None] * n_targets          # track id currently following each target
        self.track_losses = 0
        self.id_switches = 0
        self.det_opp = 0
        self.det_hit = 0
        self.sq_err = []                            # squared position errors of matched target-steps
        self.nees = []                              # NEES of matched target-steps (tracks that expose P)

    def detections(self, targets, radar_params, radar_states, reports_by_radar, terrain):
        """Count detection opportunities and losses of one step (call before comm loss)."""
        for i, (p, st) in enumerate(zip(radar_params, radar_states)):
            if not st.online or not st.last_beams:
                continue
            hits = {d["truth_id"] for d in reports_by_radar.get(i, []) if not d["is_false_alarm"]}
            for tgt in targets:
                if los_status(tgt, p, terrain) != 1:
                    continue
                if not in_beam(geo.measure(tgt.p, tgt.v, p.pos, p.heading), st.last_beams, p):
                    continue
                self.det_opp += 1
                self.det_hit += tgt.id in hits

    def tracks(self, targets, radar_params, radar_states, match, track_ids, terrain) -> dict:
        """Per-target track state after fusion; ``match`` maps target index -> Track.
        ``radar_params`` / ``radar_states`` define "in coverage" (all radars for
        the fused track, one radar for its local track).  Returns the events of
        this step, used by the continuity terms of the team reward."""
        losses = interrupted = 0
        for ti, tgt in enumerate(targets):
            tr = match.get(ti)
            prev = self.assigned[ti]
            was_tracked = any(self.valid[ti])
            if prev is not None and prev not in track_ids:
                self.track_losses += 1               # the track that followed this target was deleted
                losses += 1
                self.assigned[ti] = prev = None
            if tr is not None:
                if prev is not None and tr.id != prev:
                    self.id_switches += 1
                self.assigned[ti] = tr.id
            cov = any(st.online and los_status(tgt, p, terrain) == 1
                      for p, st in zip(radar_params, radar_states))
            ok = tr is not None
            if ok:
                self.sq_err.append(float(np.sum((tr.x[:3] - tgt.p) ** 2)))
                P = getattr(tr, "P", None)
                if P is not None:
                    self.nees.append(nees(tr.x, P, tgt.p, tgt.v))
            elif was_tracked:
                interrupted += 1
            self.valid[ti].append(ok)
            self.in_cov[ti].append(cov)
            self.correct[ti].append(ok and tr.class_probs is not None
                                    and int(np.argmax(tr.class_probs)) == tgt.label)
        return {"track_losses": losses, "interrupted": interrupted}

    def summary(self) -> dict:
        n_steps = sum(len(v) for v in self.valid)
        if n_steps == 0:
            return {}
        valid = np.concatenate([np.asarray(v, bool) for v in self.valid])
        cov = np.concatenate([np.asarray(c, bool) for c in self.in_cov])
        st = [runs(v) for v in self.valid]
        reacq = [n for s in st for n in s["reacquired"]]
        return {
            "pos_rmse_m": float(np.sqrt(np.mean(self.sq_err))) if self.sq_err else np.nan,
            "pos_err_median_m": float(np.sqrt(np.median(self.sq_err))) if self.sq_err else np.nan,
            "track_coverage": float(valid.mean()),
            "coverage_in_range": float(valid[cov].mean()) if cov.any() else np.nan,
            "detection_loss": float(1 - self.det_hit / self.det_opp) if self.det_opp else np.nan,
            "track_losses": int(self.track_losses),
            "interruptions": int(sum(s["interruptions"] for s in st)),
            "longest_track_s": float(np.mean([s["longest"] for s in st]) * self.dt),
            "interrupted_s": float(np.mean([s["interrupted"] for s in st]) * self.dt),
            "reacquire_s": float(np.mean(reacq) * self.dt) if reacq else np.nan,
            "correct_class_continuity": float(np.mean(np.concatenate(
                [np.asarray(c, bool) for c in self.correct]))),
            "id_switches": int(self.id_switches),
            "nees_mean": float(np.mean(self.nees)) if self.nees else np.nan,
            "nees_in_95": float(np.mean([NEES_95[0] <= v <= NEES_95[1] for v in self.nees]))
            if self.nees else np.nan,
        }
