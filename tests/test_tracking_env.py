import numpy as np
import pytest

from src.core.env import RadarNetEnv
from src.core.scenario import list_scenarios, load_scenario
from src.models.classifier.features import FEATURE_NAMES, window_features
from src.models.tracking.ekf import EKF
from src.core.radar import RadarParams
from src.core import geometry as geo


def test_ekf_converges_on_straight_line():
    rng = np.random.default_rng(0)
    radar, hd = np.zeros(3), 0.0
    R = np.diag([20.0 ** 2, 0.003 ** 2, 0.003 ** 2, 0.5 ** 2])
    x_true = np.array([6000.0, 1000.0, 800.0, -80.0, 20.0, 0.0])
    z0 = geo.measure(x_true[:3], x_true[3:], radar, hd) + rng.normal(0, np.sqrt(np.diag(R)))
    ekf = EKF.from_detection({"z": z0, "measurement_covariance": R}, radar, hd, q=1.0)
    errs = []
    for _ in range(80):
        x_true[:3] += x_true[3:] * 0.5
        ekf.predict(0.5)
        z = geo.measure(x_true[:3], x_true[3:], radar, hd) + rng.normal(0, np.sqrt(np.diag(R)))
        ekf.update(z, R, radar, hd)
        errs.append(np.linalg.norm(ekf.x[:3] - x_true[:3]))
    assert np.mean(errs[-20:]) < 25.0
    assert np.linalg.norm(ekf.x[3:] - x_true[3:]) < 5.0


@pytest.mark.parametrize("name", list_scenarios())
def test_env_runs_and_spaces_hold(name):
    sc = load_scenario(name)
    sc["duration_s"] = 15
    sc["terrain"] = {"source": "procedural"}      # keep tests offline
    env = RadarNetEnv(sc, record=True)
    obs, _ = env.reset(seed=0)
    for a in env.agents:
        assert env.observation_space(a).contains(obs[a])
    while env.agents:
        acts = {a: env.action_space(a).sample() for a in env.agents}
        obs, rew, term, trunc, info = env.step(acts)
        assert len(set(rew.values())) <= 1          # shared team reward
    assert len(env.frames) == env.max_steps + 1
    assert env.state().shape == (env.n_radars * env.obs_dim,)


def test_pettingzoo_parallel_api():
    from pettingzoo.test import parallel_api_test
    sc = load_scenario("nominal")
    sc["duration_s"] = 10
    parallel_api_test(RadarNetEnv(sc), num_cycles=20)


def test_window_features_shape_and_hover():
    t = np.arange(0, 10, 0.5)
    hover = np.zeros((len(t), 6))
    hover[:, 2] = 100.0
    f = window_features(t, hover)
    assert f.shape == (len(FEATURE_NAMES),)
    assert f[FEATURE_NAMES.index("hover_frac")] == 1.0


def test_jain_fairness():
    from src.core.env import jain_fairness
    assert jain_fairness([1, 1, 1, 1]) == pytest.approx(1.0)
    assert jain_fairness([1, 0, 0, 0]) == pytest.approx(0.25)


def test_interleaved_actions_extend_action_space():
    sc = load_scenario("interleaved")
    sc["duration_s"] = 10
    sc["terrain"] = {"source": "procedural"}
    env = RadarNetEnv(sc)
    env.reset(seed=0)
    assert env.action_space(env.agents[0]).n == 2 + 2 * env.K
    base = load_scenario("nominal")
    assert RadarNetEnv(base).action_space("radar_0").n == 2 + base["max_track_slots"]   # unchanged default
