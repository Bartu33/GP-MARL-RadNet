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
from src.core.modes import Mode
from src.core.radar import RadarAction, RadarState, radar_step, resource_vector, snr_linear, antenna_gain
from src.core.radar import los_status
from src.core.scenario import build_radars
from src.core.targets import CLASS_NAMES, advance, keep_above_ground, spawn_target
from src.core.terrain import build_terrain
from src.models.classifier.classifier import accumulate
from src.models.tracking.tracker import TrackManager

SLOT_FEATURES = 12
OWN_FEATURES = 6 + 4 + 2           # resources + mode one-hot + (online, time fraction)


class RadarNetEnv(ParallelEnv):
    metadata = {"name": "radar_net_v0", "render_modes": []}

    def __init__(self, scenario: dict, classifier=None, record: bool = False, seed: int | None = None):
        self.sc = scenario
        self.classifier = classifier
        self.record = record
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

    # ------------------------------------------------------------------ spaces
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return spaces.Discrete(2 + self.K * (2 if self.interleaved else 1))

    def state(self) -> np.ndarray:
        """Global state for a centralised critic: all agents' observations."""
        return np.concatenate([self._obs[a] for a in self.possible_agents])

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed = seed
        self.rng = np.random.default_rng(self._seed)
        self.agents = list(self.possible_agents)
        self.t = 0.0
        self.step_count = 0
        self.radar_states = [RadarState.initial(p) for p in self.radar_params]
        self.targets, self.motions = [], []
        for i, spec in enumerate(self.sc["targets"]):
            s, m = spawn_target(i, int(spec["label"]), self.rng, self.sc["area_radius_m"],
                                position=spec.get("position"))
            keep_above_ground(s, self.terrain)
            self.targets.append(s)
            self.motions.append(m)
        self.tracker = TrackManager()
        self.slots: list = [None] * self.K
        self.frames: list[dict] = []
        self.log = {k: [] for k in ("t", "rmse", "tracked", "false_tracks", "energy", "reports",
                                    "class_acc", "reward", "fairness")}
        self.track_load = np.zeros(self.n_radars)         # cumulative seconds spent tracking per radar
        self._last_reports = []
        self._beam_len = [None] * self.n_radars
        self._update_slots()
        self._obs = {a: self._observe(i) for i, a in enumerate(self.possible_agents)}
        if self.record:
            self._record_frame([])
        return dict(self._obs), {a: {} for a in self.agents}

    # ------------------------------------------------------------------ step
    def step(self, actions: dict):
        self.t += self.dt
        self.step_count += 1
        stress = self.sc["stress"]

        for s, m in zip(self.targets, self.motions):
            advance(s, m, self.dt, terrain=self.terrain)
        self.tracker.predict(self.dt)

        fail = stress.get("fail_radar")
        if fail is not None and self.t >= stress.get("fail_time_s", 0.0):
            self.radar_states[int(fail)].online = False

        all_reports, energy = [], 0.0
        for i, (p, st) in enumerate(zip(self.radar_params, self.radar_states)):
            act = self._decode_action(i, int(actions.get(self.possible_agents[i], 1)))
            reports, _ = radar_step(self.targets, act, st, p, self.dt, self.rng, radar_index=i,
                                    terrain=self.terrain)
            energy += st.last_energy_used_j / p.e_max_j
            if st.mode in (Mode.TRACK, Mode.INTERLEAVED):
                self.track_load[i] += p.profiles[st.mode].dwell_s
            # communication model: bandwidth cap then random packet loss
            cap = stress.get("max_reports_per_radar")
            if cap is not None and len(reports) > cap:
                reports = sorted(reports, key=lambda d: -d["signal_quality"])[:cap]
            if self.rng.random() < stress.get("comm_drop_prob", 0.0):
                reports = []
            all_reports.extend(reports)
        self._last_reports = all_reports

        self.tracker.process(self.t, all_reports, self.radar_params)
        self._update_channels()
        if self.classifier is not None and self.step_count % 4 == 0:      # every 2 s
            probs = self.classifier.classify_tracks(self.tracker.tracks)
            for tr in self.tracker.tracks:
                if tr.id in probs:
                    tr.class_probs = accumulate(tr.class_probs, probs[tr.id])
        self._update_slots()

        reward, metrics = self._team_reward(energy)
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

    def match_targets(self):
        """Greedy truth-to-track assignment (evaluation only)."""
        pairs = []
        for ti, tgt in enumerate(self.targets):
            for k, tr in enumerate(self.tracker.confirmed):
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
            out[ti] = self.tracker.confirmed[k]
        return out

    def _team_reward(self, energy: float):
        w = self.sc["reward"]
        match = self.match_targets()
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
             - w["w_false"] * false_tracks)
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
            })
        tracks = []
        for tr in self.tracker.tracks:
            cp = tr.class_probs
            tracks.append({
                "id": tr.id, "p": r1(tr.x[:3]), "v": r1(tr.x[3:]),
                "sig": round(tr.ekf.pos_std, 1), "conf": bool(tr.confirmed),
                "cls": int(np.argmax(cp)) if cp is not None else -1,
                "pc": round(float(np.max(cp)), 3) if cp is not None else 0.0,
            })
        self.frames.append({
            "t": round(self.t, 2),
            "radars": radars,
            "targets": [{"id": s.id, "label": s.label, "p": r1(s.p), "v": r1(s.v)} for s in self.targets],
            "tracks": tracks,
            "dets": [r1(d["world_xyz"]) + [int(d["is_false_alarm"]), d["radar"]] for d in reports],
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
