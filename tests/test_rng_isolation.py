"""Episode randomness is reproducible and isolated from sensing decisions."""
from copy import deepcopy

import numpy as np
import pytest

import src.core.env as env_module
from src.core.env import RadarNetEnv
from src.core.scenario import load_scenario
from src.models.baselines.policies import RandomPolicy


def scenario():
    sc = load_scenario("nominal")
    sc["terrain"] = {"source": "flat"}
    sc["duration_s"] = 12
    # Produce plentiful reports, including false alarms, without relying on
    # a particular random target happening to intersect a narrow search beam.
    sc["radar_defaults"]["false_alarms_per_beam"] = 0.5
    return sc


def make_env(sc=None, seed=7):
    env = RadarNetEnv(scenario() if sc is None else sc, seed=seed)
    env.reset()
    return env


def truth(env):
    return np.array([np.concatenate((s.p, s.v, s.R_WT.ravel())) for s in env.targets])


def assert_reports_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for key in a:
            np.testing.assert_equal(a[key], b[key], err_msg=key)


@pytest.fixture
def raw_reports(monkeypatch):
    """Capture radar output before bandwidth, loss, or sharing decisions."""
    captured = {}
    original = env_module.radar_step

    def capture(*args, **kwargs):
        reports, state = original(*args, **kwargs)
        captured[kwargs["radar_index"]] = deepcopy(reports)
        return reports, state

    monkeypatch.setattr(env_module, "radar_step", capture)
    return captured


def step_search(env, captured):
    env.step({a: 1 for a in env.agents})
    return deepcopy(captured)


def test_same_seed_actions_reproduce_truth_and_reports(raw_reports):
    a, b = make_env(), make_env()
    count = 0
    while a.agents:
        ra = step_search(a, raw_reports)
        rb = step_search(b, raw_reports)
        np.testing.assert_array_equal(truth(a), truth(b))
        for i in ra:
            assert_reports_equal(ra[i], rb[i])
            count += len(ra[i])
        assert_reports_equal(a._last_reports, b._last_reports)
    assert count > 0


@pytest.mark.parametrize("other_action", [0, "random"])
def test_different_actions_preserve_truth(other_action):
    a, b = make_env(), make_env()
    policy = RandomPolicy()
    different = False
    while a.agents:
        actions = (policy.act(b, b._obs) if other_action == "random"
                   else {ag: other_action for ag in b.agents})
        different |= any(v != 1 for v in actions.values())
        a.step({ag: 1 for ag in a.agents})
        b.step(actions)
        np.testing.assert_array_equal(truth(a), truth(b))
    assert different


@pytest.mark.parametrize("component", ["policy", "comm"])
def test_extra_rng_draws_preserve_truth_and_raw_reports(component, raw_reports):
    a, b = make_env(), make_env()
    count = 0
    while a.agents:
        if component == "policy":
            b.rng_policy.random(37)
        else:
            for rng in b.rng_comm:
                rng.random(37)
        ra = step_search(a, raw_reports)
        rb = step_search(b, raw_reports)
        np.testing.assert_array_equal(truth(a), truth(b))
        for i in ra:
            assert_reports_equal(ra[i], rb[i])
            count += len(ra[i])
    assert count > 0


def test_packet_loss_preserves_truth_and_raw_reports(raw_reports):
    sc = scenario()
    a = make_env(deepcopy(sc))
    sc["stress"]["comm_drop_prob"] = 1.0
    b = make_env(sc)
    count = 0
    while a.agents:
        ra = step_search(a, raw_reports)
        rb = step_search(b, raw_reports)
        np.testing.assert_array_equal(truth(a), truth(b))
        for i in ra:
            assert_reports_equal(ra[i], rb[i])
        count += len(a._last_reports)
        assert b._last_reports == []
    assert count > 0


def test_extra_radar_draws_do_not_change_other_radars(raw_reports):
    a, b = make_env(), make_env()
    changed = False
    count = 0
    while a.agents:
        b.rng_radars[0].random(37)
        ra = step_search(a, raw_reports)
        rb = step_search(b, raw_reports)
        np.testing.assert_array_equal(truth(a), truth(b))
        changed |= len(ra[0]) != len(rb[0])
        for i in range(1, a.n_radars):
            assert_reports_equal(ra[i], rb[i])
            count += len(ra[i])
    assert changed and count > 0


def test_extra_target_draws_do_not_change_other_targets():
    a, b = make_env(), make_env()
    changed = False
    while a.agents:
        # ErraticSlow consumes randomness at every integration substep.
        b.rng_targets[2].random(37)
        a.step({ag: 1 for ag in a.agents})
        b.step({ag: 1 for ag in b.agents})
        ta, tb = truth(a), truth(b)
        np.testing.assert_array_equal(ta[[0, 1, 3]], tb[[0, 1, 3]])
        changed |= not np.array_equal(ta[2], tb[2])
    assert changed


@pytest.mark.parametrize("population", ["targets", "radars"])
def test_population_sizes_do_not_shift_other_streams(population):
    sc = scenario()
    a = make_env(deepcopy(sc))
    sc[population].append(deepcopy(sc[population][-1]))
    b = make_env(sc)
    np.testing.assert_array_equal(truth(a), truth(b)[:len(a.targets)])
    for name in ("rng_targets", "rng_radars", "rng_comm"):
        for left, right in zip(getattr(a, name), getattr(b, name)):
            np.testing.assert_array_equal(left.random(20), right.random(20))
    np.testing.assert_array_equal(a.rng_policy.random(20), b.rng_policy.random(20))


@pytest.mark.parametrize("explicit_seed", [True, False])
def test_reset_replays_all_streams(explicit_seed, raw_reports):
    env = make_env(seed=1)
    env.reset(seed=19)

    def episode():
        result = []
        while env.agents:
            actions = RandomPolicy().act(env, env._obs)
            env.step(actions)
            result.append((truth(env), deepcopy(raw_reports), deepcopy(env._last_reports), actions))
        return result

    first = episode()
    env.reset(seed=19) if explicit_seed else env.reset()
    second = episode()
    for (ta, ra, da, aa), (tb, rb, db, ab) in zip(first, second):
        np.testing.assert_array_equal(ta, tb)
        assert aa == ab
        for i in ra:
            assert_reports_equal(ra[i], rb[i])
        assert_reports_equal(da, db)


def test_different_seeds_change_stochastic_states():
    assert not np.array_equal(truth(make_env(seed=7)), truth(make_env(seed=8)))


def test_legacy_rng_and_environment_contracts():
    env = make_env()
    assert env.rng is env.rng_policy
    assert isinstance(env.rng, np.random.Generator)
    assert env.obs_dim == env.K * 12 + 12
    assert env.action_space("radar_0").n == 2 + env.K
    obs, info = env.reset(options={})
    assert set(obs) == set(info) == set(env.agents)
    assert all(env.observation_space(a).contains(obs[a]) for a in env.agents)
    assert len(env.step({a: 1 for a in env.agents})) == 5


def test_random_policy_uses_policy_rng():
    env = make_env()
    # Keep a legacy RNG available, but ensure RandomPolicy does not use it.
    env.rng = object()
    acts = RandomPolicy().act(env, env._obs)
    assert all(env.action_space(a).contains(v) for a, v in acts.items())
