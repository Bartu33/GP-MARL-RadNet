"""RadarNetEnv -- cooperative multi-radar environment (PettingZoo ParallelEnv).

One agent per radar.  At every step each radar chooses

    0         IDLE   (rest: recover energy, cool down)
    1         SEARCH (next positions of its search raster)
    2 .. K+1  TRACK slot k (one long dwell on the predicted position of the
              k-th entry of the shared track table)

Observations are fixed-size vectors built as in [PN] Section 12 (Eq. (59)):
a block per track slot with a validity mask, the *predicted* measurement of
that track relative to this radar and its uncertainty, followed by the
radar's own normalised resource vector (Eq. (57)) and mode.  Agents never
see simulation truth; the team reward is computed from truth by the
environment (allowed during centralised training).

Team reward (same for every agent), following the project brief:
    r = - w_track * normalised track error of matched targets
        - w_miss  * fraction of targets without a confirmed track
        + w_class * (probability assigned to the true class - chance)
        - w_cost  * energy spent / max energy per step
        - w_false * number of confirmed tracks not matching any target
"""
from __future__ import annotations

import functools

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.core import geometry as geo
from src.core.evaluation import TrackEvaluator
from src.core.modes import Mode
from src.core.radar import RadarAction, RadarState, radar_step, resource_vector, snr_linear, antenna_gain
from src.core.radar import los_status
from src.core.rcs import incidence_in_body
from src.core.scenario import build_radars
from src.core.scheduler import LocalScheduler
from src.core.targets import CLASS_NAMES, advance, keep_above_ground, spawn_target
from src.core.terrain import build_terrain
from src.models.classifier.classifier import accumulate
from src.models.tracking.tracker import TrackManager

SLOT_FEATURES = 12
OWN_FEATURES = 6 + 4 + 2           # resources + mode one-hot + (online, time fraction)


class RadarNetEnv(ParallelEnv):
    metadata = {"name": "radar_net_v0", "render_modes": []}

    def __init__(self, scenario: dict, classifier=None, record: bool = False, seed: int | None = None,
                 local_tracks: bool = False):
        self.sc = scenario
        self.classifier = classifier
        self.record = record
        # local_tracks: every radar also runs its own single-radar tracker on its own
        # detections (structure A of the evaluation); off during training for speed
        self.local_tracks = local_tracks
        self.dt = float(scenario["dt"])
        self.max_steps = int(round(scenario["duration_s"] / self.dt))
        self.K = int(scenario["max_track_slots"])
        self.radar_params = build_radars(scenario)
        self.terrain = _cached_terrain(scenario)
        for p in self.radar_params:                 # position z = mast height above local ground
            x, y, mast = p.position
            p.position = (x, y, float(self.terrain.height(x, y)) + mast)
        self.n_radars = len(self.radar_params)
        self.possible_agents = [f"radar_{i}" for i in range(self.n_radars)]
        self.agents = list(self.possible_agents)
        self._seed = scenario.get("seed", 0) if seed is None else seed
        self.obs_dim = self.K * SLOT_FEATURES + OWN_FEATURES
        # optional extra actions: INTERLEAVED (search + track) on slot k
        self.interleaved = bool(scenario.get("interleaved_actions", False))
        # optional measurement-sharing decision: every radar action also says whether this
        # radar's reports are sent to the fusion centre this step (fusion participation)
        self.share = bool(scenario.get("share_actions", False))
        self.n_base_actions = 2 + self.K * (2 if self.interleaved else 1)

    # ------------------------------------------------------------------ spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        # a < n_base: sensing action a, reports shared; a >= n_base: action a - n_base, reports withheld
        return spaces.Discrete(self.n_base_actions * (2 if self.share else 1))

    def state(self) -> np.ndarray:
        """Global state for a centralised critic: all agents' observations."""
        return np.concatenate([self._obs[a] for a in self.possible_agents])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        # Independent random streams (all derived from one seed).  Target motion
        # does not depend on what the radars do, so every policy run with the
        # same seed sees exactly the same target trajectories (paired seeds).
        #   stream 0: targets (one child per target: spawn, RCS, manoeuvres)
        #   stream 1: communication losses
        #   stream 2: everything else (e.g. the random baseline policy)
        #   stream 3 + i: detections / noise / false alarms of radar i
        root = np.random.SeedSequence(self._seed)
        streams = root.spawn(3 + self.n_radars)
        self.rng_targets = [np.random.default_rng(s) for s in streams[0].spawn(len(self.sc["targets"]))]
        self.rng_comm = np.random.default_rng(streams[1])
        self.rng = np.random.default_rng(streams[2])
        self.rng_radar = [np.random.default_rng(s) for s in streams[3:]]
        self.agents = list(self.possible_agents)
        self.t = 0.0
        self.step_count = 0
        self.radar_states = [RadarState.initial(p) for p in self.radar_params]
        self.targets, self.motions = [], []
        for i, spec in enumerate(self.sc["targets"]):
            s, m = spawn_target(i, int(spec["label"]), self.rng_targets[i], self.sc["area_radius_m"],
                                position=spec.get("position"))
            keep_above_ground(s, self.terrain)
            self.targets.append(s)
            self.motions.append(m)
        tcfg = self.sc.get("tracker", {})
        self.tracker = TrackManager(**tcfg)
        self.local_trackers = [TrackManager(**tcfg) for _ in range(self.n_radars)] if self.local_tracks else []
        self.local_eval = [TrackEvaluator(len(self.targets), self.dt) for _ in self.local_trackers]
        self.slots: list = [None] * self.K
        self.frames: list[dict] = []
        self.log = {k: [] for k in ("t", "rmse", "tracked", "false_tracks", "energy", "reports",
                                    "class_acc", "reward", "fairness")}
        self.track_load = np.zeros(self.n_radars)         # cumulative seconds spent tracking per radar
        self.evaluator = TrackEvaluator(len(self.targets), self.dt)   # truth-based metrics, evaluation only
        self._last_reports = []
        # PPI scopes (display only, no random numbers): every radar's own reports before
        # communication, report counts [produced at the radar, received at the centre],
        # and which fused track each local track feeds: (radar, local id) -> (fused id, t)
        self._own_reports = [[] for _ in range(self.n_radars)]
        self._comm = [[0, 0] for _ in range(self.n_radars)]
        self._links: dict = {}
        self._beam_len = [None] * self.n_radars
        # local schedulers: used when the centre sends track-update requests (step_requests)
        self.schedulers = [LocalScheduler(p, self.dt, **self.sc.get("scheduler", {})) for p in self.radar_params]
        self.last_answers: dict = {}
        self.last_report_counts: dict = {}
        self.track_events = {"info_gain": 0.0, "confirmed_deleted": 0, "newly_confirmed": 0}
        self._update_slots()
        self._obs = {a: self._observe(i) for i, a in enumerate(self.possible_agents)}
        if self.record:
            self._record_frame([])
        return dict(self._obs), {a: {} for a in self.agents}

    # ------------------------------------------------------------------ step
    def step_requests(self, requests: dict):
        """Task-manager interface: ``requests`` maps radar index -> list of
        ``scheduler.Request``.  Each radar's local scheduler answers the requests
        (``self.last_answers``) and chooses its own look for this step; the centre
        never sets a radar mode.  Returns the same tuple as :meth:`step`."""
        t_next = self.t + self.dt
        tracks = {tr.id: tr for tr in self.tracker.tracks}
        self.last_answers, looks = {}, {}
        for i, sch in enumerate(self.schedulers):
            st = self.radar_states[i]
            self.last_answers[i] = sch.submit(self.t, requests.get(i, []), tracks, st)
            looks[i] = sch.next_action(t_next, tracks, st)
        served = {i: sch.last_served for i, sch in enumerate(self.schedulers) if sch.last_served is not None}
        out = self.step({a: 1 for a in self.agents}, radar_actions=looks)
        # requested dwells: did the radar detect the track it was asked to look at?
        alive = {tr.id: tr for tr in self.tracker.tracks}
        for i, tid in served.items():
            if tid in alive:
                tr = alive[tid]
                tr.record_dwell(i, self.t, tr.updated_by.get(i) == self.t)
        return out

    def scheduler_summary(self) -> dict:
        """Request statistics of all local schedulers (empty if no request was sent)."""
        tot = {}
        for sch in self.schedulers:
            for k, v in sch.summary().items():
                if k != "reasons":
                    tot[k] = tot.get(k, 0) + v
        n = tot.get("requests", 0)
        if n == 0:
            return {}
        return {
            "requests_per_step": n / max(self.step_count, 1),
            "request_accept_rate": tot["accepted"] / n,
            "request_delay_rate": tot["delayed"] / n,
            "request_partial_rate": tot["partial"] / n,
            "request_reject_rate": tot["rejected"] / n,
            "request_expired_rate": tot["expired"] / n,
        }

    def step(self, actions: dict, radar_actions: dict | None = None):
        """``radar_actions`` (radar index -> (RadarAction, beam length)) overrides the
        decoded agent actions; it is used by :meth:`step_requests`."""
        self.t += self.dt
        self.step_count += 1
        stress = self.sc["stress"]

        for s, m in zip(self.targets, self.motions):
            advance(s, m, self.dt, terrain=self.terrain)
        self.tracker.predict(self.dt)
        for lt in self.local_trackers:
            lt.predict(self.dt)

        fail = stress.get("fail_radar")
        if fail is not None and self.t >= stress.get("fail_time_s", 0.0):
            self.radar_states[int(fail)].online = False

        all_reports, energy, at_radar = [], 0.0, {}
        self.withheld = 0
        for i, (p, st) in enumerate(zip(self.radar_params, self.radar_states)):
            a = int(actions.get(self.possible_agents[i], 1))
            shared = a < self.n_base_actions
            if radar_actions is not None and i in radar_actions:
                act, self._beam_len[i] = radar_actions[i]
            else:
                act = self._decode_action(i, a % self.n_base_actions)
            reports, _ = radar_step(self.targets, act, st, p, self.dt, self.rng_radar[i], radar_index=i,
                                    terrain=self.terrain)
            if self._in_outage(i):
                reports = []                     # radar transmits but detects nothing (e.g. jamming)
            at_radar[i] = reports
            energy += st.last_energy_used_j / p.e_max_j
            if st.mode in (Mode.TRACK, Mode.INTERLEAVED):
                self.track_load[i] += p.profiles[st.mode].dwell_s
            # communication model: bandwidth cap then random packet loss
            cap = stress.get("max_reports_per_radar")
            if cap is not None and len(reports) > cap:
                reports = sorted(reports, key=lambda d: -d["signal_quality"])[:cap]
            if self.rng_comm.random() < stress.get("comm_drop_prob", 0.0):
                reports = []
            if not shared:                       # radar keeps its reports (local track only)
                self.withheld += len(reports)
                reports = []
            all_reports.extend(reports)
        self._last_reports = all_reports
        self.last_report_counts = {}                       # reports per radar that reached the centre
        for d in all_reports:
            self.last_report_counts[d["radar"]] = self.last_report_counts.get(d["radar"], 0) + 1
        self.evaluator.detections(self.targets, self.radar_params, self.radar_states, at_radar, self.terrain)

        # truth-free track events of this step (used by the task-manager reward)
        logdet = lambda tr: float(np.linalg.slogdet(tr.ekf.P[:3, :3])[1])
        pre = {tr.id: (logdet(tr), tr.confirmed) for tr in self.tracker.tracks}
        self.tracker.process(self.t, all_reports, self.radar_params)
        post = {tr.id: tr for tr in self.tracker.tracks}
        self.track_events = {
            # reduction of log det of the position covariance by this step's updates (nats)
            "info_gain": sum(max(0.0, pre[k][0] - logdet(tr)) for k, tr in post.items() if k in pre),
            "confirmed_deleted": sum(1 for k, (_, c) in pre.items() if c and k not in post),
            "newly_confirmed": sum(1 for k, tr in post.items() if tr.confirmed and not pre.get(k, (0, False))[1]),
        }
        for i, lt in enumerate(self.local_trackers):        # at the radar: before communication loss
            lt.process(self.t, at_radar[i], self.radar_params)
        self._update_ppi(at_radar, all_reports)
        self._update_channels()
        if self.classifier is not None and self.step_count % 4 == 0:      # every 2 s
            probs = self.classifier.classify_tracks(self.tracker.tracks)
            for tr in self.tracker.tracks:
                if tr.id in probs:
                    tr.class_probs = accumulate(tr.class_probs, probs[tr.id])
        self._update_slots()

        match = self.match_targets()
        events = self.evaluator.tracks(self.targets, self.radar_params, self.radar_states, match,
                                       {tr.id for tr in self.tracker.tracks}, self.terrain)
        for i, (lt, ev) in enumerate(zip(self.local_trackers, self.local_eval)):
            ev.tracks(self.targets, [self.radar_params[i]], [self.radar_states[i]], self.match_targets(lt),
                      {tr.id for tr in lt.tracks}, self.terrain)
        reward, metrics = self._team_reward(energy, match, events)
        for k, v in metrics.items():
            self.log[k].append(v)
        self.log["t"].append(self.t)
        self.log["reward"].append(reward)

        self._obs = {a: self._observe(i) for i, a in enumerate(self.possible_agents)}
        done = self.step_count >= self.max_steps
        rewards = {a: reward for a in self.agents}
        terms = {a: False for a in self.agents}
        truncs = {a: done for a in self.agents}
        infos = {a: {"metrics": metrics} for a in self.agents}
        if self.record:
            self._record_frame(all_reports)
        if done:
            self.agents = []
        return dict(self._obs), rewards, terms, truncs, infos

    # ------------------------------------------------------------------ helpers
    def _update_ppi(self, at_radar: dict, delivered: list):
        """Bookkeeping for the PPI scopes: each radar's own reports, how many
        reached the fusion centre, and which fused track every local track
        feeds (the fused track updated by the same report)."""
        self._own_reports = [list(at_radar.get(i, [])) for i in range(self.n_radars)]
        recv = [0] * self.n_radars
        for d in delivered:
            recv[d["radar"]] += 1
        self._comm = [[len(self._own_reports[i]), recv[i]] for i in range(self.n_radars)]
        central = self.tracker.last_assoc
        for i, lt in enumerate(self.local_trackers):
            for d in self._own_reports[i]:
                k = id(d["z"])
                if k in lt.last_assoc and k in central:
                    self._links[(i, lt.last_assoc[k])] = (central[k], self.t)
        fused = {tr.id for tr in self.tracker.tracks}
        local = {(i, tr.id) for i, lt in enumerate(self.local_trackers) for tr in lt.tracks}
        self._links = {k: v for k, v in self._links.items()
                       if k in local and v[0] in fused and self.t - v[1] <= 15.0}

    def link_of(self, radar: int, local_id: int):
        """Fused track id fed by the same reports as this radar's local track (or None)."""
        v = self._links.get((radar, local_id))
        return v[0] if v else None

    def _in_outage(self, i: int) -> bool:
        """Scenario-defined detection outages: ``stress.outages`` = [[radar | "all", t0, t1], ...]."""
        for r, t0, t1 in self.sc["stress"].get("outages") or []:
            if (r == "all" or int(r) == i) and t0 <= self.t < t1:
                return True
        return False

    def _decode_action(self, i: int, a: int) -> RadarAction:
        self._beam_len[i] = None
        if a == 0:
            return RadarAction(Mode.IDLE)
        if a == 1:
            return RadarAction(Mode.SEARCH)
        mode = Mode.TRACK
        k = a - 2
        if self.interleaved and k >= self.K:
            mode, k = Mode.INTERLEAVED, k - self.K
        tr = self.slots[k] if 0 <= k < self.K else None
        if tr is None:
            return RadarAction(Mode.SEARCH)          # empty slot -> fall back to search
        p = self.radar_params[i]
        meas = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
        if abs(meas[1]) > np.deg2rad(p.fov_az_deg) or meas[0] > p.r_max_m:
            return RadarAction(Mode.SEARCH)          # outside this radar's coverage
        self._beam_len[i] = float(meas[0]) * 1.15
        return RadarAction(mode, beam=(meas[1], meas[2]))

    def _update_slots(self):
        """Shared, ordered track table: confirmed tracks first, then the most
        uncertain.  The same ordering is shown to every agent so that
        "slot k" means the same track for everyone."""
        trs = sorted(self.tracker.tracks, key=lambda t: (not t.confirmed, -t.ekf.pos_std))
        self.slots = (trs[: self.K] + [None] * self.K)[: self.K]

    def _update_channels(self):
        for i, (p, st) in enumerate(zip(self.radar_params, self.radar_states)):
            busy = sum(1 for tr in self.tracker.tracks if self.t - tr.updated_by.get(i, -1e9) < 3.0)
            st.channels_free = max(0, p.c_max - busy)

    def _observe(self, i: int) -> np.ndarray:
        p, st = self.radar_params[i], self.radar_states[i]
        blocks = []
        for tr in self.slots:
            if tr is None:
                blocks.append(np.zeros(SLOT_FEATURES))
                continue
            r, a, e, rdot = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
            in_fov = float(abs(a) <= np.deg2rad(p.fov_az_deg) and r <= p.r_max_m)
            snr_pred = 10 * np.log10(max(1e-12, float(snr_linear(r, 0.1, antenna_gain(0, 0, p), p,
                                                                 p.profiles[Mode.TRACK].n_pulses))))
            cp = tr.class_probs.max() if tr.class_probs is not None else 1.0 / len(CLASS_NAMES)
            blocks.append(np.array([
                1.0, r / p.r_max_m, np.sin(a), np.cos(a), e, rdot / 300.0,
                np.log10(1 + tr.ekf.pos_std) / 3.0, float(tr.confirmed),
                min(1.0, (self.t - tr.last_update_t) / 10.0), in_fov, snr_pred / 30.0, cp,
            ]))
        # [idle, search, track, offline]; an interleaved look sets search and track
        mode_1h = np.array([0.0, 1.0, 1.0, 0.0]) if st.mode == Mode.INTERLEAVED else np.eye(4)[int(st.mode)]
        own = np.concatenate([resource_vector(st, p, p.profiles[Mode.TRACK].total_dwell_s),
                              mode_1h, [float(st.online), self.step_count / self.max_steps]])
        return np.concatenate(blocks + [own]).astype(np.float32)

    def match_targets(self, tracker=None):
        """Greedy truth-to-track assignment (evaluation only); fused tracker by default."""
        confirmed = (tracker or self.tracker).confirmed
        pairs = []
        for ti, tgt in enumerate(self.targets):
            for k, tr in enumerate(confirmed):
                d = np.linalg.norm(tr.x[:3] - tgt.p)
                if d < 3 * tr.ekf.pos_std + 500.0:
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

    def _team_reward(self, energy: float, match: dict, events: dict):
        w = self.sc["reward"]
        n = len(self.targets)
        errs, cls_gain, cls_hits = [], [], []
        for ti, tr in match.items():
            tgt = self.targets[ti]
            errs.append(np.linalg.norm(tr.x[:3] - tgt.p))
            if tr.class_probs is not None:
                cls_gain.append(tr.class_probs[tgt.label] - 1.0 / len(CLASS_NAMES))
                cls_hits.append(float(np.argmax(tr.class_probs) == tgt.label))
        err_term = np.mean([min(3.0, e / w["err_scale_m"]) for e in errs]) if errs else 3.0
        miss = (n - len(match)) / max(n, 1)
        false_tracks = len(self.tracker.confirmed) - len(match)
        fair = jain_fairness(self.track_load)
        r = (-w["w_track"] * err_term - w["w_miss"] * miss
             - w.get("w_fair", 0.0) * (1.0 - fair)
             + w["w_class"] * (np.mean(cls_gain) if cls_gain else 0.0)
             - w["w_cost"] * energy / max(self.n_radars, 1) * 10
             - w["w_false"] * false_tracks
             # continuity (main objective): targets that lost their track after being
             # tracked, and fused tracks deleted by the tracker this step
             - w.get("w_interrupt", 0.0) * events["interrupted"] / max(n, 1)
             - w.get("w_loss", 0.0) * events["track_losses"]
             # communication cost: reports sent to the fusion centre per radar
             - w.get("w_comm", 0.0) * len(self._last_reports) / max(self.n_radars, 1))
        metrics = {
            "rmse": float(np.sqrt(np.mean(np.square(errs)))) if errs else np.nan,
            "tracked": len(match) / max(n, 1),
            "false_tracks": false_tracks,
            "energy": energy,
            "reports": len(self._last_reports),
            "class_acc": float(np.mean(cls_hits)) if cls_hits else np.nan,
            "fairness": fair,
        }
        return float(r), metrics

    # ------------------------------------------------------------------ recording for the UI
    @staticmethod
    def _link(p, tgt):
        r = float(np.linalg.norm(tgt.p - p.pos))
        sig = float(tgt.rcs.mean_rcs(incidence_in_body(p.pos, tgt.p, tgt.R_WT)))
        chi = float(snr_linear(r, sig, antenna_gain(0, 0, p), p, p.profiles[Mode.TRACK].n_pulses))
        return [round(r), round(sig, 4), round(10 * np.log10(max(chi, 1e-30)), 1)]

    def _record_frame(self, reports):
        r1 = lambda a: [round(float(x), 1) for x in a]
        radars = []
        for i, (p, st) in enumerate(zip(self.radar_params, self.radar_states)):
            blen = self._beam_len[i] if st.mode in (Mode.TRACK, Mode.INTERLEAVED) else None
            radars.append({
                "len": round(blen) if blen else None,
                "mode": int(st.mode), "online": bool(st.online),
                "beams": [[round(float(a), 4), round(float(e), 4)] for a, e in st.last_beams],
                "energy": round(st.energy_j / p.e_max_j, 3),
                "heat": round(1 - np.clip((st.heat_k - p.h_ambient_k) / (p.h_max_k - p.h_ambient_k), 0, 1), 3),
                # per target: 0 out of coverage, 1 line of sight, 2 hidden by terrain
                "los": [los_status(s, p, self.terrain) if st.online else 0 for s in self.targets],
                # per target: [range m, mean RCS at this aspect m², SNR dB of a boresight track dwell]
                "link": [self._link(p, s) for s in self.targets],
            })
        tracks = []
        for tr in self.tracker.tracks:
            cp = tr.class_probs
            tracks.append({
                "id": tr.id, "p": r1(tr.x[:3]), "v": r1(tr.x[3:]),
                "sig": round(tr.ekf.pos_std, 1), "conf": bool(tr.confirmed),
                "cls": int(np.argmax(cp)) if cp is not None else -1,
                "pc": round(float(np.max(cp)), 3) if cp is not None else 0.0,
                # radars whose reports updated this fused track in the last 3 s
                "by": sorted(r for r, tt in tr.updated_by.items() if self.t - tt <= 3.0),
            })
        self.frames.append({
            "t": round(self.t, 2),
            "radars": radars,
            "targets": [{"id": s.id, "label": s.label, "p": r1(s.p), "v": r1(s.v)} for s in self.targets],
            "tracks": tracks,
            "dets": [r1(d["world_xyz"]) + [int(d["is_false_alarm"]), d["radar"]] for d in reports],
            # confirmed local (single-radar) tracks per radar, for per-radar displays
            "ltracks": [[{"id": tr.id, "p": r1(tr.x[:3]), "v": r1(tr.x[3:]), "sig": round(tr.ekf.pos_std, 1),
                          "fid": self.link_of(i, tr.id)} for tr in lt.confirmed]
                        for i, lt in enumerate(self.local_trackers)],
            # PPI scopes: every radar's own reports (also those not delivered) and the
            # report count per radar [produced at the radar, received by the fusion centre]
            "own": [[r1(d["world_xyz"]) + [int(d["is_false_alarm"])] for d in reps] for reps in self._own_reports],
            "comm": [list(c) for c in self._comm],
        })


_TERRAIN_CACHE: dict = {}


def _cached_terrain(sc: dict):
    """Build (or reuse) the terrain of a scenario; training creates many envs."""
    spec = sc.get("terrain", {"source": "flat"})
    key = (sc["site"]["lat"], sc["site"]["lon"], str(spec), sc.get("seed", 0))
    if key not in _TERRAIN_CACHE:
        _TERRAIN_CACHE[key] = build_terrain(sc["site"], spec, seed=sc.get("seed", 0))
    return _TERRAIN_CACHE[key]


def jain_fairness(x) -> float:
    """Jain's fairness index (sum x)^2 / (n sum x^2) of the per-radar tracking
    load: 1 = perfectly balanced, 1/n = one radar does all the tracking."""
    x = np.asarray(x, float)
    if x.sum() <= 0:
        return 1.0
    return float(x.sum() ** 2 / (len(x) * np.sum(x ** 2)))
