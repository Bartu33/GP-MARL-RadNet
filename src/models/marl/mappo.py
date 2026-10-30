"""Minimal MAPPO (multi-agent PPO with a centralised critic) for RadarNetEnv.

Design (CTDE -- centralised training, decentralised execution):
* one actor network shared by all radars (parameter sharing); its input is
  the radar's own observation plus a one-hot agent id,
* one critic network that sees the global state ``env.state()`` and
  predicts the value of the shared team reward,
* PPO clipped surrogate objective with generalised advantage estimation.

Written from the published algorithm descriptions (Schulman et al. 2017,
"Proximal Policy Optimization Algorithms"; Yu et al. 2022, "The Surprising
Effectiveness of PPO in Cooperative Multi-Agent Games").  Setting
``centralised_critic=False`` turns it into independent PPO (IPPO) with
shared parameters, which is the learning baseline of the project.

Requires PyTorch (see requirements-ml.txt).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = ROOT / "models" / "checkpoints" / "mappo.pt"


def mlp(n_in, n_out, hidden=128):
    return nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                         nn.Linear(hidden, n_out))


@dataclass
class MAPPOConfig:
    lr: float = 3e-4
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 512
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden: int = 128
    centralised_critic: bool = True


class MAPPOAgent(nn.Module):
    def __init__(self, obs_dim: int, state_dim: int, n_actions: int, n_agents: int, cfg: MAPPOConfig):
        super().__init__()
        self.cfg, self.n_agents, self.n_actions = cfg, n_agents, n_actions
        self.obs_dim, self.state_dim = obs_dim, state_dim
        self.actor = mlp(obs_dim + n_agents, n_actions, cfg.hidden)
        critic_in = state_dim if cfg.centralised_critic else obs_dim + n_agents
        self.critic = mlp(critic_in, 1, cfg.hidden)

    def actor_input(self, obs: torch.Tensor, agent_idx: torch.Tensor) -> torch.Tensor:
        return torch.cat([obs, nn.functional.one_hot(agent_idx, self.n_agents).float()], -1)

    def dist(self, obs, agent_idx):
        return torch.distributions.Categorical(logits=self.actor(self.actor_input(obs, agent_idx)))

    def value(self, state, obs, agent_idx):
        x = state if self.cfg.centralised_critic else self.actor_input(obs, agent_idx)
        return self.critic(x).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.obs, self.state, self.agent, self.act, self.logp, self.rew, self.val, self.done = ([] for _ in range(8))

    def add(self, **kw):
        for k, v in kw.items():
            getattr(self, k).append(v)


def compute_gae(rew, val, done, last_val, gamma, lam):
    """rew/val/done: (T, N) arrays; returns advantages and returns."""
    T = len(rew)
    adv = np.zeros_like(rew)
    gae = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + val


class MAPPOTrainer:
    def __init__(self, env_fn, cfg: MAPPOConfig = MAPPOConfig(), seed: int = 0):
        self.env_fn, self.cfg = env_fn, cfg
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        env = env_fn()
        env.reset(seed=seed)
        self.agents = list(env.possible_agents)
        self.N = len(self.agents)
        self.net = MAPPOAgent(env.obs_dim, env.obs_dim * self.N, env.action_space(self.agents[0]).n, self.N, cfg)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        self.history = []

    def collect(self, n_steps: int):
        buf = RolloutBuffer()
        env = self.env_fn()
        obs, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        idx = torch.arange(self.N)
        ep_ret, ep_rets = 0.0, []
        for _ in range(n_steps):
            o = torch.as_tensor(np.stack([obs[a] for a in self.agents]))
            s = torch.as_tensor(env.state()).float().expand(self.N, -1)
            with torch.no_grad():
                d = self.net.dist(o, idx)
                a = d.sample()
                v = self.net.value(s, o, idx)
            obs, rew, term, trunc, _ = env.step({ag: int(a[i]) for i, ag in enumerate(self.agents)})
            done = float(any(trunc.values()) or any(term.values()))
            r = float(next(iter(rew.values()))) if rew else 0.0
            ep_ret += r
            buf.add(obs=o.numpy(), state=s.numpy(), agent=idx.numpy(), act=a.numpy(),
                    logp=d.log_prob(a).numpy(), rew=np.full(self.N, r), val=v.numpy(), done=np.full(self.N, done))
            if done:
                ep_rets.append(ep_ret)
                ep_ret = 0.0
                obs, _ = env.reset(seed=int(self.rng.integers(1 << 30)))
        with torch.no_grad():
            o = torch.as_tensor(np.stack([obs[a] for a in self.agents]))
            s = torch.as_tensor(env.state()).float().expand(self.N, -1)
            last_val = self.net.value(s, o, idx).numpy()
        return buf, last_val, ep_rets

    def update(self, buf: RolloutBuffer, last_val):
        c = self.cfg
        rew, val, done = (np.array(x, dtype=np.float32) for x in (buf.rew, buf.val, buf.done))
        adv, ret = compute_gae(rew, val, done, last_val, c.gamma, c.gae_lambda)
        flat = lambda x: torch.as_tensor(np.concatenate(x)).float() if np.ndim(x[0]) > 1 else torch.as_tensor(np.array(x).reshape(-1))
        obs, state = flat(buf.obs), flat(buf.state)
        agent = torch.as_tensor(np.array(buf.agent).reshape(-1)).long()
        act = torch.as_tensor(np.array(buf.act).reshape(-1)).long()
        logp_old = torch.as_tensor(np.array(buf.logp).reshape(-1)).float()
        adv_t = torch.as_tensor(adv.reshape(-1)).float()
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret.reshape(-1)).float()
        n = len(act)
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, c.minibatch):
                mb = perm[start:start + c.minibatch]
                d = self.net.dist(obs[mb], agent[mb])
                ratio = torch.exp(d.log_prob(act[mb]) - logp_old[mb])
                pg = -torch.min(ratio * adv_t[mb], torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * adv_t[mb]).mean()
                v_loss = (self.net.value(state[mb], obs[mb], agent[mb]) - ret_t[mb]).pow(2).mean()
                ent = d.entropy().mean()
                loss = pg + c.value_coef * v_loss - c.entropy_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                stats.append((pg.item(), v_loss.item(), ent.item()))
        return np.mean(stats, axis=0)

    def train(self, total_steps: int, rollout: int = 960, log=print):
        done_steps = 0
        while done_steps < total_steps:
            buf, last_val, ep_rets = self.collect(rollout)
            pg, vl, ent = self.update(buf, last_val)
            done_steps += rollout
            mean_ret = float(np.mean(ep_rets)) if ep_rets else float("nan")
            self.history.append({"steps": done_steps, "episode_return": mean_ret, "pg_loss": pg,
                                 "value_loss": vl, "entropy": ent})
            log(f"steps {done_steps:>8d}  return {mean_ret:8.1f}  entropy {ent:5.2f}  v_loss {vl:8.2f}")
        return self.history

    def save(self, path: Path = DEFAULT_CKPT):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "cfg": self.cfg.__dict__,
                    "dims": (self.net.obs_dim, self.net.state_dim, self.net.n_actions, self.N),
                    "history": self.history}, path)


class MAPPOPolicy:
    """Decentralised execution: each radar samples from the shared actor
    using only its own observation."""
    name = "mappo"

    def __init__(self, net: MAPPOAgent, greedy: bool = True):
        self.net, self.greedy = net.eval(), greedy

    @classmethod
    def load(cls, path: Path = DEFAULT_CKPT, greedy: bool = True):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        obs_dim, state_dim, n_act, n_ag = ck["dims"]
        net = MAPPOAgent(obs_dim, state_dim, n_act, n_ag, MAPPOConfig(**ck["cfg"]))
        net.load_state_dict(ck["state_dict"])
        return cls(net, greedy)

    @classmethod
    def load_default(cls):
        if not DEFAULT_CKPT.exists():
            raise FileNotFoundError(f"no trained MAPPO checkpoint at {DEFAULT_CKPT}; "
                                    "run: python -m src.services.training_service")
        return cls.load(DEFAULT_CKPT)

    def reset(self, env):
        pass

    def act(self, env, obs) -> dict:
        names = env.possible_agents
        o = torch.as_tensor(np.stack([obs[a] for a in names]))
        with torch.no_grad():
            d = self.net.dist(o, torch.arange(len(names)))
            a = d.probs.argmax(-1) if self.greedy else d.sample()
        return {n: int(a[i]) for i, n in enumerate(names) if n in env.agents}
