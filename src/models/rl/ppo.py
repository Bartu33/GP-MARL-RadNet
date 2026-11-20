"""Proximal Policy Optimisation for the single fusion-centre agent.

Plain PPO (Schulman et al. 2017) with generalised advantage estimation
(Schulman et al. 2016): one actor-critic network, no hierarchy, no
multi-agent structure.  The action is MultiDiscrete (one choice per radar);
the policy factorises over radars and invalid choices are removed with the
action mask before sampling.

Optional warm start (behaviour cloning): the network first imitates the
rule-based task manager (``TaskManagerEnv.greedy_action``) on recorded
decisions -- actor by cross-entropy, critic by the discounted returns of those
episodes -- and PPO then fine-tunes it.  The learning algorithm is still plain
PPO; only its starting point changes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CKPT = ROOT / "models" / "checkpoints" / "ppo_task.pt"


@dataclass
class PPOConfig:
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.95            # 1 s decisions: an effective horizon of ~20 s
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 256
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rollout: int = 1024            # decisions per update


class TaskActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_radars: int, n_choices: int, hidden: int = 256):
        super().__init__()
        self.obs_dim, self.n_radars, self.n_choices = obs_dim, n_radars, n_choices
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_radars * n_choices)
        self.v = nn.Linear(hidden, 1)

    def forward(self, obs, mask):
        h = self.body(obs)
        logits = self.pi(h).view(-1, self.n_radars, self.n_choices)
        logits = logits.masked_fill(~mask, -1e9)
        return torch.distributions.Categorical(logits=logits), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray, greedy: bool = True) -> np.ndarray:
        d, _ = self(torch.as_tensor(obs, dtype=torch.float32)[None], torch.as_tensor(mask)[None])
        a = d.probs.argmax(-1) if greedy else d.sample()
        return a[0].numpy()


@dataclass
class PPOTrainer:
    env_fn: callable
    cfg: PPOConfig = field(default_factory=PPOConfig)
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        self.env = self.env_fn()
        self.net = TaskActorCritic(self.env.observation_space.shape[0], self.env.N, self.env.K + 1, self.cfg.hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.cfg.lr)
        self.history = []
        self._obs, info = self.env.reset(seed=int(self.rng.integers(1 << 30)))
        self._mask = info["action_mask"]
        self._ep_ret, self._ep_rets = 0.0, []

    def collect(self):
        c, env = self.cfg, self.env
        O, M, A, LP, R, D, V = [], [], [], [], [], [], []
        for _ in range(c.rollout):
            o = torch.as_tensor(self._obs, dtype=torch.float32)[None]
            m = torch.as_tensor(self._mask)[None]
            with torch.no_grad():
                d, v = self.net(o, m)
                a = d.sample()
                lp = d.log_prob(a).sum(-1)
            obs2, r, term, trunc, info = env.step(a[0].numpy())
            O.append(self._obs); M.append(self._mask); A.append(a[0].numpy())
            LP.append(float(lp)); R.append(r); D.append(term or trunc); V.append(float(v))
            self._ep_ret += r
            if term or trunc:
                self._ep_rets.append(self._ep_ret)
                self._ep_ret = 0.0
                obs2, info = env.reset(seed=int(self.rng.integers(1 << 30)))
            self._obs, self._mask = obs2, info["action_mask"]
        with torch.no_grad():
            _, last_v = self.net(torch.as_tensor(self._obs, dtype=torch.float32)[None],
                                 torch.as_tensor(self._mask)[None])
        R, D, V = np.array(R), np.array(D, float), np.array(V)
        adv = np.zeros_like(R)
        gae, nxt = 0.0, float(last_v)
        for t in reversed(range(len(R))):
            delta = R[t] + c.gamma * nxt * (1 - D[t]) - V[t]
            gae = delta + c.gamma * c.gae_lambda * (1 - D[t]) * gae
            adv[t], nxt = gae, V[t]
        return (np.array(O, np.float32), np.array(M), np.array(A), np.array(LP, np.float32),
                adv.astype(np.float32), (adv + V).astype(np.float32))

    def update(self, batch):
        c = self.cfg
        O, M, A, LP, ADV, RET = (torch.as_tensor(x) for x in batch)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = len(O)
        stats = []
        for _ in range(c.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, c.minibatch):
                i = perm[s:s + c.minibatch]
                d, v = self.net(O[i], M[i])
                lp = d.log_prob(A[i]).sum(-1)
                ratio = torch.exp(lp - LP[i])
                pg = -torch.min(ratio * ADV[i], torch.clamp(ratio, 1 - c.clip, 1 + c.clip) * ADV[i]).mean()
                vl = ((v - RET[i]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + c.vf_coef * vl - c.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), c.max_grad_norm)
                self.opt.step()
                stats.append((pg.item(), vl.item(), ent.item()))
        return np.mean(stats, axis=0)

    # ------------------------------------------------------------ warm start
    def collect_expert(self, episodes: int, log=print) -> dict:
        """Run the rule-based teacher and record (obs, mask, action, return-to-go)."""
        c, env = self.cfg, self.env_fn()
        O, M, A, G = [], [], [], []
        for ep in range(episodes):
            obs, info = env.reset(seed=int(self.rng.integers(1 << 30)))
            rew, done = [], False
            while not done:
                a = env.greedy_action()
                O.append(obs); M.append(info["action_mask"]); A.append(a)
                obs, r, term, trunc, info = env.step(a)
                rew.append(r)
                done = term or trunc
            g, ret = 0.0, []
            for r in reversed(rew):
                g = r + c.gamma * g
                ret.append(g)
            G.extend(reversed(ret))
            if (ep + 1) % 20 == 0:
                log(f"teacher episodes {ep + 1}/{episodes}")
        return {"obs": np.array(O, np.float32), "mask": np.array(M), "action": np.array(A),
                "ret": np.array(G, np.float32)}

    def behaviour_clone(self, data: dict, epochs: int = 30, lr: float = 1e-3, log=print) -> float:
        """Fit the actor to the teacher's choices and the critic to its returns.
        Returns the share of radar choices that match the teacher."""
        O, M, A, G = (torch.as_tensor(data[k]) for k in ("obs", "mask", "action", "ret"))
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        n = len(O)
        for ep in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.cfg.minibatch):
                i = perm[s:s + self.cfg.minibatch]
                d, v = self.net(O[i], M[i])
                loss = -d.log_prob(A[i]).sum(-1).mean() + self.cfg.vf_coef * ((v - G[i]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.max_grad_norm)
                opt.step()
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                with torch.no_grad():
                    d, _ = self.net(O, M)
                    acc = float((d.probs.argmax(-1) == A).float().mean())
                log(f"behaviour cloning epoch {ep + 1}/{epochs}  loss {loss.item():.3f}  match {acc:.3f}")
        self.history.append({"decisions": 0, "behaviour_cloning_match": acc, "teacher_samples": n})
        return acc

    def train(self, total_decisions: int, log=print, save_path: Path | None = None, save_every: int = 5):
        done = 0
        while done < total_decisions:
            batch = self.collect()
            pg, vl, ent = self.update(batch)
            done += self.cfg.rollout
            ret = float(np.mean(self._ep_rets[-10:])) if self._ep_rets else float("nan")
            self.history.append({"decisions": done, "episode_return": ret, "pg_loss": float(pg),
                                 "value_loss": float(vl), "entropy": float(ent)})
            log(f"decisions {done:>7d}  return {ret:8.2f}  entropy {ent:5.2f}  v_loss {vl:7.3f}")
            if save_path is not None and len(self.history) % save_every == 0:
                self.save(save_path)
        return self.history

    def save(self, path: Path = DEFAULT_CKPT):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = self.net
        torch.save({"state_dict": n.state_dict(), "cfg": self.cfg.__dict__,
                    "dims": (n.obs_dim, n.n_radars, n.n_choices), "history": self.history,
                    "scenario": self.env.sc.get("name"), "reward": self.env.w}, path)


def load_net(path: Path = DEFAULT_CKPT) -> TaskActorCritic:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    obs_dim, n_radars, n_choices = ck["dims"]
    net = TaskActorCritic(obs_dim, n_radars, n_choices, ck["cfg"].get("hidden", 256))
    net.load_state_dict(ck["state_dict"])
    return net.eval()
