"""Baselines with adaptive revisit, single-radar local tracks, continuity reward,
fusion-centre policy and the classifier study."""
from types import SimpleNamespace

import numpy as np
import pytest

from src.core.env import RadarNetEnv
from src.core.scenario import load_scenario
from src.models.baselines import policies as pol
from src.services.simulation_service import per_radar_table, run_episode


def _sc(name, **over):
    sc = load_scenario(name)
    sc["terrain"] = {"source": "procedural"}
    sc.update(over)
    return sc


def test_revisit_rule():
    env = SimpleNamespace(t=10.0)
    ekf = SimpleNamespace(pos_std=40.0)
    fresh = SimpleNamespace(confirmed=True, last_update_t=9.5, ekf=ekf)
    stale = SimpleNamespace(confirmed=True, last_update_t=10.0 - pol.REVISIT_S, ekf=ekf)
    tentative = SimpleNamespace(confirmed=False, last_update_t=9.9, ekf=ekf)
    vague = SimpleNamespace(confirmed=True, last_update_t=9.9, ekf=SimpleNamespace(pos_std=pol.STD_DUE_M + 1))
    assert not pol._due(env, fresh)
    assert pol._due(env, stale) and pol._due(env, tentative) and pol._due(env, vague)


def test_greedy_searches_when_no_track_is_due():
    env = RadarNetEnv(_sc("nominal", duration_s=5))
    obs, _ = env.reset(seed=0)
    for tr in env.tracker.tracks:
        tr.confirmed, tr.last_update_t = True, env.t
    p = pol.make_policy("central_greedy")
    p.reset(env)
    p.k = 1                                            # not a reserved-search step
    acts = p.act(env, obs)
    assert all(a in (0, 1) for a in acts.values())     # nobody wastes a dwell on fresh tracks


def test_local_tracks_give_structure_a_metrics():
    env, s = run_episode(_sc("nominal", duration_s=40), "central_greedy", seed=0, classifier=None, record=True)
    for k in ("single_best_coverage", "single_best_longest_s", "single_mean_rmse_m", "pos_err_median_m"):
        assert k in s
    assert 0.0 <= s["single_best_coverage"] <= 1.0
    tab = per_radar_table(env)
    assert len(tab) == env.n_radars + 1 and tab["track"].iloc[-1].startswith("Fused")
    assert len(env.frames[-1]["ltracks"]) == env.n_radars


def test_training_env_has_no_local_trackers():
    env = RadarNetEnv(_sc("nominal", duration_s=5))
    env.reset(seed=0)
    assert env.local_trackers == [] and env.local_eval == []


def _return(w_interrupt, w_loss):
    sc = _sc("track_loss")
    sc["reward"] = dict(sc["reward"], w_interrupt=w_interrupt, w_loss=w_loss)
    _, s = run_episode(sc, "fixed_scan", seed=0, classifier=None, record=False)
    return s["return"], s["track_losses"]


def test_continuity_terms_penalise_track_loss():
    r_off, losses = _return(0.0, 0.0)
    r_on, _ = _return(0.5, 1.0)
    assert losses >= 1
    assert r_on < r_off - 0.99 * losses                # at least w_loss per deleted track


def test_central_actor_sees_global_state():
    torch = pytest.importorskip("torch")
    from src.models.marl.mappo import MAPPOAgent, MAPPOConfig
    net = MAPPOAgent(obs_dim=5, state_dim=20, n_actions=3, n_agents=4, cfg=MAPPOConfig(central_actor=True))
    assert net.actor[0].in_features == 20 + 4
    obs, state, idx = torch.zeros(4, 5), torch.randn(4, 20), torch.arange(4)
    d = net.dist(obs, idx, state)
    assert d.probs.shape == (4, 3)


def test_classifier_study_runs_and_splits_by_episode():
    pytest.importorskip("torch")
    from src.services import classifier_study as cs
    d = cs.generate(3, duration_s=50.0, seed=3)
    assert d["seq"].shape[1:] == (cs.SEQ_LEN, len(cs.SEQ_NAMES))
    assert set(np.unique(d["source"])) <= {"fused", "local", "truth", "fused_early3", "fused_early6"}
    rep = cs.run_study(d)
    assert rep["dataset"]["episodes_in_train_and_test"] == 0
    for r in rep["results"]:
        assert 0.0 <= r["macro_f1"] <= 1.0 and 0.0 <= r["ece"] <= 1.0
    assert {r["model"] for r in rep["results"]} >= {"mlp", "gru"}


def test_ece_of_perfectly_calibrated_predictions_is_zero():
    from src.services.classifier_study import ece
    proba = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert ece(proba, np.array([0, 1])) == pytest.approx(0.0)
    assert ece(np.array([[0.9, 0.1]]), np.array([1])) == pytest.approx(0.9)


def test_share_action_withholds_reports_from_the_centre():
    sc = _sc("share_control", duration_s=10)
    env = RadarNetEnv(sc, local_tracks=True)
    env.reset(seed=0)
    base = env.n_base_actions
    assert env.action_space("radar_0").n == 2 * base
    withheld = 0
    while env.agents:
        env.step({a: base + 1 for a in env.agents})       # search, keep reports local
        assert env._last_reports == []
        withheld += env.withheld
    assert withheld > 0                                   # radars did detect, the centre got nothing
    assert any(lt.tracks for lt in env.local_trackers)    # local tracks still formed
    assert env.tracker.tracks == []


def test_comm_cost_lowers_reward_when_sharing():
    sc = _sc("share_control", duration_s=20)
    _, on = run_episode(sc, "fixed_scan", seed=0, classifier=None, record=False)
    sc["reward"] = dict(sc["reward"], w_comm=0.0)
    _, off = run_episode(sc, "fixed_scan", seed=0, classifier=None, record=False)
    assert on["reports_per_step"] > 0 and on["return"] < off["return"]


def test_decision_time_is_reported():
    _, s = run_episode(_sc("nominal", duration_s=5), "central_greedy", seed=0, classifier=None, record=False)
    assert s["decision_ms"] > 0


def test_simulation_classifier_trained_on_fused_tracks(tmp_path):
    from src.models.classifier.classifier import FlightProfileClassifier
    from src.services import classifier_study as cs
    d = cs.generate(3, duration_s=50.0, seed=3)
    meta = cs.train_simulation_model(d, path=tmp_path / "checkpoints" / "sim.joblib")
    clf = FlightProfileClassifier.load(tmp_path / "checkpoints" / "sim.joblib")
    assert clf.fitted and meta["trained_on"].startswith("fused")
    assert clf.predict_proba(d["feat"][:3]).shape == (3, 3)                    # UAV / aircraft / missile


def test_trained_policy_refuses_a_scenario_of_another_size(monkeypatch):
    """A network trained on nominal (6 track slots) cannot run multi_target (12 slots):
    it must say so clearly, and the policy comparison skips it instead of crashing."""
    pytest.importorskip("torch")
    from src.models.marl.mappo import IncompatiblePolicy, MAPPOAgent, MAPPOConfig, MAPPOPolicy
    from src.services import simulation_service as sim
    nominal = RadarNetEnv(_sc("nominal"))
    nominal.reset(seed=0)
    n_act = nominal.action_space("radar_0").n
    net = MAPPOAgent(nominal.obs_dim, nominal.obs_dim * nominal.n_radars, n_act, nominal.n_radars,
                     MAPPOConfig(central_actor=True))
    pol = MAPPOPolicy(net)
    pol.name = "central_ppo"
    pol.reset(nominal)                                    # same size: fine
    multi = RadarNetEnv(_sc("multi_target"))
    multi.reset(seed=0)
    with pytest.raises(IncompatiblePolicy, match="trained for"):
        pol.reset(multi)

    real = sim.make_policy
    monkeypatch.setattr(sim, "make_policy", lambda name: pol if name == "central_ppo" else real(name))
    sc = _sc("multi_target", duration_s=5)
    df = sim.compare_policies(sc, ["central_ppo", "fixed_scan"], seeds=[0, 1], classifier=None)
    assert set(df.policy) == {"fixed_scan"} and "central_ppo" in df.attrs["skipped"]
