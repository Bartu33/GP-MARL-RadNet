"""Single-agent fusion-centre task manager: environment, SNR features, PPO."""
import numpy as np
import pytest

from src.core.observation import TRACK_FEATURES, QUALITY_FEATURES, central_observation, snr_statistics
from src.core.scenario import load_scenario
from src.core.task_env import TaskManagerEnv


def _sc(name="nominal", duration=20):
    sc = load_scenario(name)
    sc["terrain"] = {"source": "procedural"}
    sc["duration_s"] = duration
    return sc


def _run(env, steps, choose):
    obs, info = env.reset(seed=0)
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(choose(info["action_mask"]))
        if term or trunc:
            break
    return obs, info


def test_spaces_and_one_decision_per_second():
    env = TaskManagerEnv(_sc())
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs.astype(np.float32))
    assert env.action_space.nvec.tolist() == [env.K + 1] * env.N
    assert info["action_mask"].shape == (env.N, env.K + 1) and info["action_mask"][:, -1].all()
    t0 = env.sim.t
    env.step([env.K] * env.N)
    assert env.sim.t == pytest.approx(t0 + 1.0)            # 1-s decision = 2 simulator steps


def test_episode_ends_and_reward_is_finite():
    env = TaskManagerEnv(_sc(duration=10))
    obs, info = env.reset(seed=1)
    rets, done, n = [], False, 0
    while not done:
        obs, r, term, trunc, info = env.step([env.K] * env.N)
        rets.append(r)
        done, n = term or trunc, n + 1
    assert n == 10 and np.all(np.isfinite(rets))


def test_actions_become_requests_to_the_chosen_radar_only():
    env = TaskManagerEnv(_sc())
    _run(env, 6, lambda m: [m.shape[1] - 1] * m.shape[0])           # let tracks form
    m = env.action_mask()
    i, k = np.argwhere(m[:, :-1])[0]
    action = [env.K] * env.N
    action[i] = k
    reqs, dup = env.requests(action)
    assert list(reqs) == [i] and reqs[i][0].track_id == env.sim.slots[k].id and dup == 0
    assert reqs[i][0].deadline_s == pytest.approx(env.sim.t + 2.0)


def test_masked_choice_is_ignored_and_duplicates_are_counted():
    env = TaskManagerEnv(_sc())
    _run(env, 6, lambda m: [m.shape[1] - 1] * m.shape[0])
    m = env.action_mask()
    blocked = np.argwhere(~m[:, :-1])
    if len(blocked):
        i, k = blocked[0]
        action = [env.K] * env.N
        action[i] = k
        assert env.requests(action)[0] == {}                       # radar cannot cover it: no request
    k = int(np.flatnonzero(m[:, :-1].sum(0) >= 2)[0])               # a track two radars can cover
    radars = np.flatnonzero(m[:, k])[:2]
    action = [env.K] * env.N
    for i in radars:
        action[i] = k
    _, dup = env.requests(action)
    assert dup == 1


def test_snr_features_follow_the_reports():
    env = TaskManagerEnv(_sc())
    _run(env, 10, lambda m: [int(np.flatnonzero(r)[0]) for r in m])
    o = central_observation(env.sim)
    base = TRACK_FEATURES.index("instantaneous SNR: last report / 30 dB")
    for k, tr in enumerate(env.sim.slots):
        if tr is None or not tr.snr:
            continue
        last, mean, std = snr_statistics(tr)
        f = o["tracks"][k]
        assert f[base] == pytest.approx(last / 30)
        assert f[base + 1] == pytest.approx(mean / 30)
        allv = sorted((x for h in tr.snr.values() for x in h), key=lambda x: x[0])[-10:]
        db = np.array([v for _, v in allv])
        assert mean == pytest.approx(10 * np.log10(np.mean(10 ** (db / 10))))   # linear mean, not dB mean
        assert mean >= db.mean() - 1e-9                                       # Jensen: linear mean >= dB mean


def test_requested_dwells_record_detections():
    env = TaskManagerEnv(_sc(duration=30))
    _run(env, 20, lambda m: [int(np.flatnonzero(r)[0]) for r in m])        # always request the first slot
    rec = [h for tr in env.sim.tracker.tracks for h in tr.dwells.values()]
    assert rec, "no requested dwell was recorded"
    hits = [d for h in rec for _, d in h]
    assert 0.0 <= np.mean(hits) <= 1.0
    idx = QUALITY_FEATURES.index("detection rate of this radar's requested dwells on the track (0.5 = no history)")
    q = central_observation(env.sim)["quality"][..., idx]
    assert np.all((q >= 0) & (q <= 1))


def test_ppo_learns_shapes_and_policy_runs_in_the_simulator(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from src.models.baselines.policies import make_policy
    from src.models.rl import ppo
    from src.services.simulation_service import run_episode
    sc = _sc(duration=8)
    trainer = ppo.PPOTrainer(lambda: TaskManagerEnv(sc), cfg=ppo.PPOConfig(rollout=32, minibatch=16, epochs=1))
    hist = trainer.train(64, log=lambda *a: None)
    assert len(hist) == 2 and np.isfinite(hist[-1]["value_loss"])
    path = tmp_path / "ppo_task.pt"
    trainer.save(path)
    net = ppo.load_net(path)
    mask = np.ones((net.n_radars, net.n_choices), bool)
    mask[:, :-1] = False                                           # only "no request" allowed
    a = net.act(np.zeros(net.obs_dim, np.float32), mask)
    assert a.tolist() == [net.n_choices - 1] * net.n_radars        # masked choices are never taken
    monkeypatch.setattr(ppo, "DEFAULT_CKPT", path)
    _, s = run_episode(sc, "ppo_task", seed=0, classifier=None, record=False)
    assert 0 <= s["coverage_in_range"] <= 1 and s["requests_per_step"] > 0


def test_trained_task_policy_refuses_another_scenario_size(tmp_path):
    pytest.importorskip("torch")
    from src.core.env import RadarNetEnv
    from src.core.task_env import TaskPolicy
    from src.models.marl.mappo import IncompatiblePolicy
    from src.models.rl.ppo import TaskActorCritic
    env = TaskManagerEnv(_sc())
    net = TaskActorCritic(env.observation_space.shape[0], env.N, env.K + 1, 32)
    TaskPolicy(net).reset(env.sim)
    multi = RadarNetEnv(_sc("multi_target"))
    multi.reset(seed=0)
    with pytest.raises(IncompatiblePolicy):
        TaskPolicy(net).reset(multi)


def test_teacher_action_is_valid_and_gives_each_radar_at_most_one_track():
    env = TaskManagerEnv(_sc())
    obs, info = env.reset(seed=0)
    for _ in range(8):
        a = env.greedy_action()
        m = env.action_mask()
        assert all(m[i, k] for i, k in enumerate(a))                       # never a masked choice
        chosen = [k for k in a if k < env.K]
        assert len(chosen) == len(set(chosen))                              # no track to two radars
        obs, r, term, trunc, info = env.step(a)


def test_new_reward_prefers_the_teacher_over_doing_nothing():
    env = TaskManagerEnv(_sc(duration=30))
    rets = {}
    for name in ("teacher", "none"):
        obs, info = env.reset(seed=3)
        total, done = 0.0, False
        while not done:
            a = env.greedy_action() if name == "teacher" else [env.K] * env.N
            obs, r, term, trunc, info = env.step(a)
            total, done = total + r, term or trunc
        rets[name] = total
    assert rets["teacher"] > rets["none"]


def test_redundant_requests_are_counted():
    env = TaskManagerEnv(_sc())
    obs, info = env.reset(seed=0)
    for _ in range(6):
        obs, r, term, trunc, info = env.step(env.greedy_action())
    from src.models.baselines.policies import _due
    fresh = [k for k, tr in enumerate(env.sim.slots) if tr is not None and not _due(env.sim, tr)]
    m = env.action_mask()
    pairs = [(i, k) for k in fresh for i in range(env.N) if m[i, k]]
    if pairs:
        i, k = pairs[0]
        a = [env.K] * env.N
        a[i] = k
        assert env._requests(a)[2] == 1


def test_behaviour_cloning_imitates_the_teacher():
    pytest.importorskip("torch")
    from src.models.rl import ppo
    sc = _sc(duration=10)
    tr = ppo.PPOTrainer(lambda: TaskManagerEnv(sc), cfg=ppo.PPOConfig(minibatch=32))
    data = tr.collect_expert(6, log=lambda *a: None)
    assert len(data["obs"]) == 60 and data["action"].shape == (60, tr.env.N)
    match = tr.behaviour_clone(data, epochs=60, lr=3e-3, log=lambda *a: None)
    assert match > 0.9                                                     # fits the teacher on its own data
