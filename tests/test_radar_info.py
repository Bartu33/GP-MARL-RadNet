"""Parameters shown in the dashboard / viewer, the radar-equation budget and ROC/PR AUC."""
import numpy as np
import pytest

from src.core.radar import RadarParams, antenna_gain, snr_linear
from src.core.scenario import load_scenario
from src.services import radar_info as ri


def test_budget_terms_add_up_to_the_simulator_snr():
    p = RadarParams()
    for r, s in [(5000.0, 1.0), (12000.0, 0.05), (900.0, 30.0)]:
        b = ri.link_budget(p, r, s)
        snr = 10 * np.log10(snr_linear(r, s, antenna_gain(0, 0, p), p, p.profiles[ri.Mode.TRACK].n_pulses))
        assert sum(v for _, v in b["terms"]) == pytest.approx(snr, abs=1e-6)
        assert b["snr_db"] == pytest.approx(snr, abs=1e-9)


def test_budget_detection_and_range_for_pd_half():
    p = RadarParams()
    b = ri.link_budget(p, 5000.0, 1.0)
    assert 0 <= b["pd_swerling1"] <= 1 and 0 <= b["pd_constant_rcs"] <= 1
    r50 = b["max_range_pd50_km"] * 1000
    near = ri.link_budget(p, r50 * 0.99, 1.0)["pd_constant_rcs"]
    far = ri.link_budget(p, r50 * 1.01, 1.0)["pd_constant_rcs"]
    assert near > 0.5 > far                                            # bisection found the P_D = 0.5 range


def test_parameter_table_and_derived_values():
    p = RadarParams(fc_hz=10e9, bandwidth_hz=1e6)
    rows = {k: v for _, k, v in ri.radar_parameters(p)}
    assert rows["wavelength λ = c / f_c"] == "3.00 cm"
    assert rows["range resolution c / 2B"] == "150 m"
    assert rows["unambiguous range c / 2·PRF"] == "9.4 km"             # nominal MPRF 16 kHz
    assert rows["unambiguous radial velocity ±λ·PRF / 4"] == "±120 m/s"
    assert rows["unambiguous range over the stagger c / 2·gcd(PRF)"] == "74.9 km"   # 14/16/18 kHz -> gcd 2 kHz


def test_ambiguity_warning_only_when_targets_are_too_fast():
    p = RadarParams()                                                # PRF stagger resolves both
    assert ri.ambiguity_notes(p, 20.0) == []
    assert ri.ambiguity_notes(p, 900.0) == []
    single = RadarParams(prf_stagger_hz=(16e3,))                      # one PRF: both limits bite
    assert any("radial velocity" in n for n in ri.ambiguity_notes(single, 230.0))
    assert any("unambiguous range" in n for n in ri.ambiguity_notes(single, 20.0))


def test_viewer_payload_carries_parameters_and_links():
    from src.services import simulation_service as sim
    sc = load_scenario("nominal")
    sc["terrain"] = {"source": "procedural"}
    sc["duration_s"] = 5
    env, _ = sim.run_episode(sc, "central_greedy", seed=0, classifier=None)
    pl = sim.viewer_payload(env)
    assert len(pl["targets"]) == len(env.targets)
    assert all(len(r["params"]) > 20 for r in pl["radars"])
    link = pl["frames"][-1]["radars"][0]["link"]
    assert len(link) == len(env.targets)
    p, t = env.radar_params[0], env.targets[0]
    assert link[0][0] == pytest.approx(np.linalg.norm(t.p - p.pos), abs=1.0)


def test_auc_scores():
    from src.services.classifier_study import auc_scores
    y = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    perfect = np.eye(4)[y] * 0.9 + 0.025
    s = auc_scores(perfect, y)
    assert s["roc_auc_macro"] == pytest.approx(1.0) and s["pr_auc_macro"] == pytest.approx(1.0)
    flat = np.full((8, 4), 0.25)
    s = auc_scores(flat, y)
    assert s["roc_auc_macro"] == pytest.approx(0.5)
    assert s["pr_auc_macro"] == pytest.approx(0.25)                    # chance = class share
    s = auc_scores(perfect[:4], y[:4])                                   # classes 2, 3 absent
    assert s["roc_auc_per_class"][2] is None and s["roc_auc_macro"] == pytest.approx(1.0)
    assert set(s["curves"]["roc"]) == {"A_fast_cruise", "B_agile_hover"}
