"""Observation used by the learning side: the agents' current observation
(RadarNetEnv._observe) and the prototype task-manager observation
(src/core/observation.py) must be correct, consistent and free of truth."""
import numpy as np
import pytest

from src.core import geometry as geo
from src.core import observation as ob
from src.core.env import RadarNetEnv, SLOT_FEATURES
from src.core.modes import Mode
from src.core.scenario import load_scenario
from src.models.baselines.policies import make_policy


def _env(name="nominal", steps=40, policy="task_greedy", seed=0):
    sc = load_scenario(name)
    sc["terrain"] = {"source": "procedural"}
    env = RadarNetEnv(sc)
    obs, _ = env.reset(seed=seed)
    pol = make_policy(policy)
    pol.reset(env)
    for _ in range(steps):
        if hasattr(pol, "requests"):
            obs, *_ = env.step_requests(pol.requests(env))
        else:
            obs, *_ = env.step(pol.act(env, obs))
    return env, obs


# ---------------------------------------------------------------- current agent observation
def test_agent_observation_slots_match_the_track_table():
    env, obs = _env(policy="central_greedy")
    for i, a in enumerate(env.agents):
        o = obs[a]
        assert o.shape == (env.obs_dim,) and np.all(np.isfinite(o))
        p = env.radar_params[i]
        for k, tr in enumerate(env.slots):
            blk = o[k * SLOT_FEATURES:(k + 1) * SLOT_FEATURES]
            if tr is None:
                assert np.all(blk == 0)
                continue
            r, az, _, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
            assert blk[1] == pytest.approx(r / p.r_max_m, rel=1e-5)
            assert blk[2] ** 2 + blk[3] ** 2 == pytest.approx(1.0, abs=1e-5)
            assert blk[6] == pytest.approx(np.log10(1 + tr.ekf.pos_std) / 3, rel=1e-5)     # uncertainty
            assert blk[7] == float(tr.confirmed)
            assert blk[8] == pytest.approx(min(1, (env.t - tr.last_update_t) / 10), abs=1e-6)  # age of info


def test_agent_resource_features_follow_the_radar_state():
    env, obs = _env(policy="central_greedy")
    for i, a in enumerate(env.agents):
        own = obs[a][env.K * SLOT_FEATURES:]
        p, st = env.radar_params[i], env.radar_states[i]
        assert own[0] == pytest.approx(st.energy_j / p.e_max_j, rel=1e-5)
        assert own[3] == pytest.approx(st.channels_free / p.c_max, rel=1e-5)
        assert np.all((own[:6] >= 0) & (own[:6] <= 1))


def test_every_agent_sees_the_same_slot_order():
    env, obs = _env(policy="central_greedy")
    conf = [o[:env.K * SLOT_FEATURES].reshape(env.K, SLOT_FEATURES)[:, [0, 6, 7]] for o in obs.values()]
    for c in conf[1:]:
        assert np.allclose(c, conf[0])           # validity, uncertainty, confirmation: radar-independent


def test_observations_never_use_truth():
    env, obs = _env(policy="central_greedy")
    before = {a: env._observe(i).copy() for i, a in enumerate(env.agents)}
    cen = ob.central_observation(env)["vector"].copy()
    for t in env.targets:                        # move every true target far away
        t.p = t.p + 5000.0
        t.v = -t.v
    for i, a in enumerate(env.agents):
        assert np.array_equal(env._observe(i), before[a])
    assert np.array_equal(ob.central_observation(env)["vector"], cen)


# ---------------------------------------------------------------- task-manager observation
def test_central_observation_shapes_masks_and_ranges():
    env, _ = _env()
    o = ob.central_observation(env)
    K, N = env.K, env.n_radars
    assert o["tracks"].shape == (K, len(ob.TRACK_FEATURES))
    assert o["radars"].shape == (N, len(ob.RADAR_FEATURES))
    assert o["quality"].shape == (N, K, len(ob.QUALITY_FEATURES))
    assert o["vector"].shape == (ob.observation_size(K, N),) and np.all(np.isfinite(o["vector"]))
    assert np.array_equal(o["track_mask"], [tr is not None for tr in env.slots])
    assert np.all(o["tracks"][~o["track_mask"]] == 0)
    assert np.all(o["quality"][~o["quality_mask"]] == 0)
    assert not np.any(o["quality_mask"][:, ~o["track_mask"]])
    r = o["radars"]
    assert np.all((r[:, 5:9] >= 0) & (r[:, 5:9] <= 1))
    assert np.all((o["quality"][..., 3] >= 0))               # an update never loses information


def test_quality_mask_matches_coverage():
    env, _ = _env()
    o = ob.central_observation(env)
    for i, p in enumerate(env.radar_params):
        for k, tr in enumerate(env.slots):
            if tr is not None:
                assert o["quality_mask"][i, k] == (env.radar_states[i].online and ob._coverable(p, tr))


def test_information_gain_larger_for_uncertain_tracks():
    env, _ = _env()
    i, tr = next((i, tr) for tr in env.slots if tr is not None
                 for i in range(env.n_radars) if ob._coverable(env.radar_params[i], tr))
    q0, _ = ob.service_quality(env, i, tr)
    P0 = tr.ekf.P.copy()
    tr.ekf.P = P0 * 25.0                         # 5x larger position / velocity sigma
    q1, _ = ob.service_quality(env, i, tr)
    tr.ekf.P = P0
    assert q1[3] > q0[3] > 0


def test_last_answers_and_nis_are_reported():
    env, _ = _env()
    o = ob.central_observation(env)
    for i, answers in env.last_answers.items():
        frac = o["radars"][i, 9:13]
        if answers:
            assert frac.sum() == pytest.approx(1.0)
        else:
            assert np.all(frac == 0)
    k = np.flatnonzero(o["nis_valid"])
    assert len(k) > 0 and np.all(o["tracks"][k, 12] >= 0)


def test_track_features_follow_the_tracker():
    env, _ = _env()
    o = ob.central_observation(env)
    for k, tr in enumerate(env.slots):
        if tr is None:
            continue
        f = o["tracks"][k]
        assert f[7] == pytest.approx(np.log10(1 + tr.ekf.pos_std) / 3)
        assert f[11] == pytest.approx(min(1, (env.t - tr.last_update_t) / 10))
        assert f[9] == float(tr.confirmed)
        assert 0 <= f[15] <= 1


def test_blind_share_is_small_when_the_radial_velocity_is_unknown():
    """A fresh track (unknown velocity) must not look blind; a hovering target
    with a well-known zero radial velocity is blind at every PRF."""
    from types import SimpleNamespace
    from src.core.observation import blind_share
    from src.core.radar import RadarParams
    p = RadarParams(position=(0, 0, 0))
    x = np.array([5000.0, 0.0, 300.0, 0.0, 0.0, 0.0])
    fresh = SimpleNamespace(x=x, ekf=SimpleNamespace(P=np.diag([100.0] * 3 + [120.0 ** 2] * 3)))
    hover = SimpleNamespace(x=x, ekf=SimpleNamespace(P=np.diag([100.0] * 3 + [0.2 ** 2] * 3)))
    assert blind_share(fresh, p) < 0.05
    assert blind_share(hover, p) > 0.95
