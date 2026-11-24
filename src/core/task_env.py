"""Single-agent task-manager environment (Gymnasium) for the fusion-centre RL.

One agent at the fusion centre.  Every decision period (``decision_s``,
default 1 s = 2 simulator steps) it sees the task-manager observation of
``src/core/observation.py`` and chooses, for every radar, either one track
slot to request an update for or "no request" (the radar then follows its own
scheduler: queued requests, otherwise search).  The requests go to the radars'
local schedulers, which accept, delay, partially serve or reject them; the
agent never sets a radar mode.

Action:  MultiDiscrete([K + 1] * N): radar i -> slot k (0..K-1) or K = no request.
         ``action_mask()`` marks the slots radar i cannot cover.
Reward (truth-free, per decision period; weights in ``DEFAULT_REWARD``):
    + w_gain   * information gain of the fused tracks (reduction of log det P_pos, nats)
    - w_aoi    * mean age of information of the confirmed tracks (/ 10 s)
    - w_del    * confirmed tracks deleted by the tracker
    + w_new    * tracks newly confirmed (rewards search that finds targets)
    - w_rej    * requests rejected by the radars
    - w_dup    * the same track requested from more than one radar
    - w_redundant * requests for tracks that are not due (confirmed, updated less
               than REVISIT_S ago and still accurate): the radar time is better
               spent searching
Simulation truth is used only by the evaluation metrics, never by the agent.

The first reward (information gain weighted 0.2, no redundancy term) taught the
agent to keep refreshing tracks it already had instead of searching; the gain
weight is therefore small and the age-of-information, deletion, new-track and
redundancy terms carry the decision (see doc/research/gozlem_gereksinimleri.md).

``greedy_action`` is the rule-based task manager (``task_greedy``) expressed in
this action space; it is the teacher for behaviour cloning before PPO.
"""
from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces

from src.core.env import RadarNetEnv
from src.core.observation import _coverable, central_observation, observation_size
from src.core.scheduler import Request, Response
from src.models.baselines.policies import _due

DEFAULT_REWARD = {"w_gain": 0.05, "w_aoi": 2.0, "w_del": 2.0, "w_new": 1.0, "w_rej": 0.05, "w_dup": 0.2,
                  "w_redundant": 0.2}


class TaskManagerEnv(Env):
    metadata = {"render_modes": []}

    def __init__(self, scenario: dict, decision_s: float = 1.0, deadline_s: float = 2.0,
                 reward: dict | None = None, local_tracks: bool = False, record: bool = False,
                 classifier=None):
        self.sc = scenario
        self.sim = RadarNetEnv(scenario, classifier=classifier, record=record, local_tracks=local_tracks)
        self.substeps = max(1, int(round(decision_s / self.sim.dt)))
        self.deadline_s = deadline_s
        self.w = {**DEFAULT_REWARD, **(reward or {})}
        K, N = self.sim.K, self.sim.n_radars
        self.K, self.N = K, N
        self.action_space = spaces.MultiDiscrete([K + 1] * N)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(observation_size(K, N),), dtype=np.float32)
        self._obs = None

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        self._confirm_sent = {}            # teacher: track id -> hits when its confirmation dwell was asked
        self._obs = central_observation(self.sim)
        return self._obs["vector"], {"action_mask": self.action_mask()}

    def action_mask(self) -> np.ndarray:
        """(N, K+1) booleans: slot k is selectable for radar i if the radar can
        cover that track; 'no request' (last column) is always allowed."""
        m = np.zeros((self.N, self.K + 1), bool)
        m[:, :self.K] = self._obs["quality_mask"]
        m[:, self.K] = True
        return m

    def requests(self, action) -> tuple[dict, int]:
        """Turn an action into scheduler requests; also returns the number of
        duplicate requests (same track asked from several radars)."""
        reqs, dup, _ = self._requests(action)
        return reqs, dup

    def _requests(self, action):
        reqs, seen, dup, redundant = {}, set(), 0, 0
        mask = self.action_mask()
        for i, k in enumerate(np.asarray(action, int)):
            if k >= self.K or not mask[i, k]:
                continue
            tr = self.sim.slots[k]
            if tr is None:
                continue
            dup += tr.id in seen
            redundant += not _due(self.sim, tr)
            seen.add(tr.id)
            reqs[i] = [Request(tr.id, deadline_s=self.sim.t + self.deadline_s, issued_t=self.sim.t)]
        return reqs, dup, redundant

    def greedy_action(self) -> np.ndarray:
        """Teacher: the rule-based task manager (``task_greedy``) in this action
        space.  Requests in the order of ``policies.task_order`` -- a confirmation
        dwell for each new tentative track (newest first), then the confirmed
        tracks due for a revisit (most uncertain first) -- each to the nearest
        radar that can cover it and has not been given a request in this
        decision; radars without a request search."""
        from src.models.baselines.policies import task_order
        sim, mask = self.sim, self.action_mask()
        slot = {tr.id: k for k, tr in enumerate(sim.slots) if tr is not None}
        if not hasattr(self, "_confirm_sent"):
            self._confirm_sent = {}
        action = np.full(self.N, self.K)
        for tr, confirm in task_order(sim, self._confirm_sent):
            k = slot.get(tr.id)
            if k is None:
                continue
            free = [i for i in range(self.N) if action[i] == self.K and mask[i, k]]
            if free:
                i = min(free, key=lambda j: np.linalg.norm(tr.x[:3] - sim.radar_params[j].pos))
                action[i] = k
                if confirm:
                    self._confirm_sent[tr.id] = tr.hits
        return action

    def step(self, action):
        reqs, dup, redundant = self._requests(action)
        gain = deleted = new = 0.0
        rejected = 0
        done = False
        for s in range(self.substeps):
            _, _, _, truncs, _ = self.sim.step_requests(reqs if s == 0 else {})
            if s == 0:
                rejected = sum(a.response == Response.REJECTED for ans in self.sim.step_answers.values()
                               for a in ans)
            ev = self.sim.track_events
            gain += ev["info_gain"]
            deleted += ev["confirmed_deleted"]
            new += ev["newly_confirmed"]
            done = not self.sim.agents
            if done:
                break
        conf = [tr for tr in self.sim.tracker.tracks if tr.confirmed]
        aoi = np.mean([min(1.0, (self.sim.t - tr.last_update_t) / 10.0) for tr in conf]) if conf else 0.0
        w = self.w
        reward = (w["w_gain"] * gain - w["w_aoi"] * aoi - w["w_del"] * deleted + w["w_new"] * new
                  - w["w_rej"] * rejected - w["w_dup"] * dup - w["w_redundant"] * redundant)
        self._obs = central_observation(self.sim)
        info = {"action_mask": self.action_mask(), "info_gain": gain, "aoi": aoi, "deleted": deleted,
                "new_confirmed": new, "rejected": rejected, "duplicates": dup, "redundant": redundant}
        return self._obs["vector"], float(reward), False, done, info


class TaskPolicy:
    """Wraps a trained task-manager network so that ``run_episode`` can use it
    like the rule-based ``task_greedy``: one decision every ``substeps``
    simulator steps, requests go through ``env.step_requests``."""
    name = "ppo_task"

    def __init__(self, net, greedy: bool = True, decision_s: float = 1.0, deadline_s: float = 2.0):
        self.net, self.greedy = net, greedy
        self.decision_s, self.deadline_s = decision_s, deadline_s

    def reset(self, env):
        self.wrap = TaskManagerEnv.__new__(TaskManagerEnv)
        self.wrap.sim, self.wrap.K, self.wrap.N = env, env.K, env.n_radars
        self.wrap.deadline_s = self.deadline_s
        self.substeps = max(1, int(round(self.decision_s / env.dt)))
        need = self.net.obs_dim, self.net.n_radars, self.net.n_choices
        have = observation_size(env.K, env.n_radars), env.n_radars, env.K + 1
        if need != have:
            from src.models.marl.mappo import IncompatiblePolicy
            raise IncompatiblePolicy(
                f"{self.name} was trained for {need[1]} radars and {need[2] - 1} track slots; scenario "
                f"'{env.sc.get('name')}' has {have[1]} radars and {have[2] - 1} track slots. Train it on this "
                f"scenario: python -m src.services.training_service --task --scenario {env.sc.get('name')}")

    def act(self, env, obs):                  # not used: run_episode calls requests()
        return {a: 1 for a in env.agents}

    def requests(self, env) -> dict:
        if env.step_count % self.substeps:
            return {}
        self.wrap._obs = central_observation(env)
        action = self.net.act(self.wrap._obs["vector"], self.wrap.action_mask(), greedy=self.greedy)
        return self.wrap.requests(action)[0]
