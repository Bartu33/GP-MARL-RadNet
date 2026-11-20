"""Transparent, non-learning coordination baselines.

Every policy maps the environment to a dict ``{agent_name: action}`` using
the same action encoding as RadarNetEnv (0 idle, 1 search, 2+k track slot k).
They read only information an operator would have: the shared track table
and each radar's own geometry and resource state -- never truth.
"""
from __future__ import annotations

import numpy as np

from src.core import geometry as geo


class Policy:
    name = "base"

    def reset(self, env):
        pass

    def act(self, env, obs) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


def _coverable(env, i, tr) -> bool:
    p = env.radar_params[i]
    r, a, _, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
    return abs(a) <= np.deg2rad(p.fov_az_deg) and r <= p.r_max_m


def _low_energy(env, i, frac=0.15) -> bool:
    st, p = env.radar_states[i], env.radar_params[i]
    return st.energy_j / p.e_max_j < frac


REVISIT_S = 2.0          # a confirmed track is refreshed at most this often
STD_DUE_M = 150.0        # ... or earlier when its position uncertainty grows beyond this


def _due(env, tr) -> bool:
    """Adaptive revisit: a track needs a dedicated dwell only when it is still
    tentative, has not been updated for ``REVISIT_S`` or has become uncertain.
    Otherwise the radar's time is better spent searching for new targets."""
    return (not tr.confirmed or env.t - tr.last_update_t >= REVISIT_S
            or tr.ekf.pos_std > STD_DUE_M)


class FixedScan(Policy):
    """Always search; tracks are maintained only by search detections."""
    name = "fixed_scan"

    def act(self, env, obs):
        return {a: 1 for a in env.agents}


class RoundRobin(Policy):
    """Alternate search and tracking, cycling through the track slots."""
    name = "round_robin"

    def reset(self, env):
        self.k = 0

    def act(self, env, obs):
        self.k += 1
        out = {}
        for i, a in enumerate(env.agents):
            if (self.k + i) % 2 == 0:
                out[a] = 1
                continue
            filled = [s for s, tr in enumerate(env.slots) if tr is not None and _coverable(env, i, tr)]
            out[a] = 2 + filled[(self.k // 2 + i) % len(filled)] if filled else 1
        return out


class NearestSensor(Policy):
    """Each radar refreshes the closest coverable track that is due for a
    revisit and searches otherwise."""
    name = "nearest"

    def act(self, env, obs):
        out = {}
        for i, a in enumerate(env.agents):
            if _low_energy(env, i):
                out[a] = 0
                continue
            best, bd = None, np.inf
            p = env.radar_params[i]
            for s, tr in enumerate(env.slots):
                if tr is None or not _due(env, tr) or not _coverable(env, i, tr):
                    continue
                d = np.linalg.norm(tr.x[:3] - p.pos)
                if d < bd:
                    best, bd = s, d
            out[a] = 2 + best if best is not None else 1
        return out


class GreedyCentral(Policy):
    """Centralised heuristic: assign radars to the most uncertain tracks that
    are due for a revisit, without duplicates; spare radars search.  Every
    third step one radar is also reserved for search."""
    name = "central_greedy"

    def reset(self, env):
        self.k = 0

    def act(self, env, obs):
        self.k += 1
        out = {a: 1 for a in env.agents}
        free = [i for i in range(len(env.agents)) if not _low_energy(env, i)]
        for i in range(len(env.agents)):
            if i not in free:
                out[env.agents[i]] = 0
        if free and self.k % 3 == 0:
            free.pop(self.k % len(free))
        order = sorted([(s, tr) for s, tr in enumerate(env.slots) if tr is not None and _due(env, tr)],
                       key=lambda st: (not st[1].confirmed, -st[1].ekf.pos_std))
        for s, tr in order:
            cands = [i for i in free if _coverable(env, i, tr)]
            if not cands:
                continue
            i = min(cands, key=lambda j: np.linalg.norm(tr.x[:3] - env.radar_params[j].pos))
            out[env.agents[i]] = 2 + s
            free.remove(i)
        return out


class InterleavedGreedy(GreedyCentral):
    """Like ``central_greedy``, but assigned radars use the INTERLEAVED mode
    (search sweep + track dwell in the same step) instead of a pure track
    dwell, so no radar ever stops surveying.  Needs a scenario with
    ``interleaved_actions: true``; otherwise it behaves like central_greedy."""
    name = "interleaved_greedy"

    def act(self, env, obs):
        out = super().act(env, obs)
        if getattr(env, "interleaved", False):
            out = {a: (v + env.K if v >= 2 else v) for a, v in out.items()}
        return out


class TaskManagerGreedy(Policy):
    """Rule-based fusion-centre task manager (structure B of the proposal).

    It does not set radar modes.  Every step it sends track-update requests
    (``scheduler.Request``) for the tracks that are due for a revisit, most
    uncertain first, each to the coverable radar with the fewest requests in
    this step (ties: nearest).  A track that already has a pending request at
    some radar is not requested again (no redundant illumination).  The
    radars' local schedulers accept, delay, partially serve or reject them.
    Deadline = now + ``REVISIT_S``; priority = position uncertainty / 100 m.
    """
    name = "task_greedy"

    def act(self, env, obs):              # not used: run_episode calls requests()
        return {a: 1 for a in env.agents}

    def requests(self, env) -> dict:
        from src.core.scheduler import Request
        pending = {q.req.track_id for sch in env.schedulers for q in sch.queue}
        load = {i: 0 for i in range(env.n_radars)}
        out: dict = {}
        due = sorted([tr for tr in env.tracker.tracks if _due(env, tr) and tr.id not in pending],
                     key=lambda tr: (not tr.confirmed, -tr.ekf.pos_std))
        for tr in due:
            cands = [i for i in range(env.n_radars)
                     if env.radar_states[i].online and _coverable(env, i, tr)]
            if not cands:
                continue
            i = min(cands, key=lambda j: (load[j], np.linalg.norm(tr.x[:3] - env.radar_params[j].pos)))
            load[i] += 1
            out.setdefault(i, []).append(Request(tr.id, deadline_s=env.t + REVISIT_S,
                                                 priority=tr.ekf.pos_std / 100.0, issued_t=env.t))
        return out


class RandomPolicy(Policy):
    name = "random"

    def act(self, env, obs):
        return {a: int(env.rng.integers(env.action_space(a).n)) for a in env.agents}


POLICIES = {cls.name: cls for cls in (FixedScan, RoundRobin, NearestSensor, GreedyCentral, InterleavedGreedy,
                                       TaskManagerGreedy, RandomPolicy)}


# learned policies -> training_service flags.  The fusion-centre task manager
# (single PPO agent sending requests) is the learned policy of the project; the
# earlier policies that set radar modes directly (central_ppo, MAPPO, IPPO) are
# kept in the code for reference but no longer offered in the dashboard.
LEARNED = {"ppo_task": "--task"}
LEGACY = {"central_ppo": "--central", "mappo": "", "ippo": "--ippo"}


def make_policy(name: str) -> Policy:
    if name in POLICIES:
        return POLICIES[name]()
    if name == "ppo_task":
        from src.core.task_env import TaskPolicy
        from src.models.rl.ppo import DEFAULT_CKPT, load_net  # requires torch
        if not DEFAULT_CKPT.exists():
            raise FileNotFoundError(f"no trained ppo_task checkpoint at {DEFAULT_CKPT}; "
                                    "run: python -m src.services.training_service --task")
        return TaskPolicy(load_net(DEFAULT_CKPT))
    if name in LEGACY:
        from src.models.marl.mappo import DEFAULT_CKPT, MAPPOPolicy  # requires torch
        path = DEFAULT_CKPT.with_name(f"{name}.pt")
        if not path.exists():
            raise FileNotFoundError(f"no trained {name} checkpoint at {path}; "
                                    f"run: python -m src.services.training_service {LEGACY[name]}")
        pol = MAPPOPolicy.load(path)
        pol.name = name
        return pol
    raise KeyError(f"unknown policy '{name}'")
