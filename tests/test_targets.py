"""Analytic motion, finite segment schedules, and legacy compatibility."""
from copy import deepcopy
import math
import re

import numpy as np
import pytest

from src.core.env import RadarNetEnv
from src.core.geometry import body_rotation
from src.core.rcs import ConstantRCS
from src.core.scenario import validate_scenario
from src.core.targets import (
    CA, CT, CV, SegmentedMotion, TargetState, advance, build_motion, spawn_target,
)
from src.core.terrain import Terrain
from src.services.simulation_service import export_episode_records, load_episode_records


def state(p=(0, 0, 1000), v=(10, 0, 0)):
    return TargetState(0, 0, np.array(p, dtype=float), np.array(v, dtype=float), ConstantRCS(1))


def model(spec):
    return build_motion(spec, np.random.default_rng(7), 1)


SEGMENTS = [
    {"model": "CV", "duration_s": 0.2},
    {"model": "CT", "duration_s": 0.3, "turn_rate_rad_s": math.pi / 0.6},
    {"model": "CA", "duration_s": 0.5, "acceleration_mps2": [2, 0, 0]},
]


def scenario(motion=None):
    return {"dt": 0.25, "duration_s": 1, "terrain": {"source": "flat"},
            "radars": [{}], "targets": [{"label": 0, "position": [4000, 0, 1000],
                "velocity": [10, 0, 0], "motion": motion or {"segments": deepcopy(SEGMENTS)}}]}


def test_cv_exact_no_homing_and_zero_bank():
    s = state(p=(1, 2, 1000), v=(3, -4, 2))
    s.a_lat = 20
    model({"model": "CV"}).step(s, 2)
    np.testing.assert_array_equal(s.p, [7, -6, 1004])
    np.testing.assert_array_equal(s.v, [3, -4, 2])
    assert s.a_lat == 0


def test_ca_exact_and_lateral_acceleration():
    s = state(p=(1, 2, 1000), v=(3, -4, 2))
    m = model({"model": "CA", "acceleration_mps2": [2, 3, -1]})
    m.step(s, 2)
    np.testing.assert_array_equal(s.p, [11, 0, 1002])
    np.testing.assert_array_equal(s.v, [7, 2, 0])
    assert s.a_lat == pytest.approx(17 / np.sqrt(53), rel=1e-14)


@pytest.mark.parametrize("omega,time,position,velocity", [
    (1, math.pi / 2, [10, 10], [0, 10]),
    (1, 2 * math.pi, [0, 0], [10, 0]),
    (-1, math.pi / 2, [10, -10], [0, -10]),
])
def test_ct_known_turns(omega, time, position, velocity):
    s = state(v=(10, 0, 3))
    model({"model": "CT", "turn_rate_rad_s": omega}).step(s, time)
    np.testing.assert_allclose(s.p[:2], position, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(s.v[:2], velocity, rtol=1e-14, atol=1e-12)
    assert s.p[2] == pytest.approx(1000 + 3 * time, rel=1e-14)
    assert s.v[2] == 3
    assert s.a_lat == omega * 10


@pytest.mark.parametrize("omega", [0, 1e-15, -1e-15])
def test_ct_zero_and_tiny_turn_rate_are_stable(omega):
    s = state(v=(100, 0, 2))
    model({"model": "CT", "turn_rate_rad_s": omega}).step(s, 1)
    assert s.p[0] == pytest.approx(100, rel=1e-14)
    assert s.p[1] == pytest.approx(50 * omega, rel=1e-14, abs=1e-28)
    assert s.v[1] == pytest.approx(100 * omega, rel=1e-14, abs=1e-28)
    assert s.p[2] == 1002 and s.v[2] == 2


def test_ct_preserves_horizontal_speed_and_vertical_motion():
    s = state(v=(3, 4, -2))
    m = model({"model": "CT", "turn_rate_rad_s": 0.2})
    for _ in range(100):
        m.step(s, 0.1)
    assert np.hypot(*s.v[:2]) == pytest.approx(5, rel=1e-13)
    assert s.v[2] == -2
    assert s.p[2] == pytest.approx(980, rel=1e-13)


@pytest.mark.parametrize("spec", [
    {"model": "CV"}, {"model": "CA", "acceleration_mps2": [1, 2, -0.5]},
    {"model": "CT", "turn_rate_rad_s": -0.3}, {"segments": SEGMENTS},
])
def test_single_step_and_partitioned_steps_agree(spec):
    a, b = state(v=(3, 4, 2)), state(v=(3, 4, 2))
    ma, mb = model(spec), model(spec)
    ma.step(a, 1)
    for h in (0.1, 0.15, 0.35, 0.4):
        mb.step(b, h)
    np.testing.assert_allclose(a.p, b.p, rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(a.v, b.v, rtol=1e-14, atol=1e-12)
    assert a.a_lat == pytest.approx(b.a_lat, rel=1e-14, abs=1e-12)


def test_one_step_crosses_multiple_segments_with_known_result():
    s = state()
    m = model({"segments": SEGMENTS})
    assert isinstance(m, SegmentedMotion)
    m.step(s, 1)
    radius = 6 / math.pi
    np.testing.assert_allclose(s.p, [2 + radius + 0.25, radius + 5, 1000], rtol=1e-14, atol=1e-12)
    np.testing.assert_allclose(s.v, [1, 10, 0], rtol=1e-14, atol=1e-12)
    assert m.segment_index == 2 and m.elapsed == 1


def test_segment_boundaries_preserve_position_velocity_and_update_bank():
    s = state()
    m = model({"segments": SEGMENTS})
    m.step(s, 0.2)
    np.testing.assert_array_equal(s.p, [2, 0, 1000])
    np.testing.assert_array_equal(s.v, [10, 0, 0])
    assert m.segment_index == 1
    assert s.a_lat == pytest.approx(10 * math.pi / 0.6)
    p, v = s.p.copy(), s.v.copy()
    m.step(s, 0)
    np.testing.assert_array_equal(s.p, p)
    np.testing.assert_array_equal(s.v, v)
    m.step(s, 0.3)
    assert m.segment_index == 2
    np.testing.assert_allclose(s.v, [0, 10, 0], atol=1e-12)
    assert s.a_lat == pytest.approx(-2, rel=1e-14)


def test_schedule_exhaustion_is_explicit_and_does_not_partly_advance():
    s = state()
    m = model({"segments": [{"model": "CV", "duration_s": 0.1}]})
    before = s.p.copy()
    with pytest.raises(ValueError, match="exceeds total"):
        m.step(s, 0.2)
    np.testing.assert_array_equal(s.p, before)
    assert m.elapsed == 0


@pytest.mark.parametrize("dt", [-1, True, float("nan"), float("inf")])
def test_direct_motion_calls_reject_invalid_timesteps(dt):
    for spec in ({"model": "CV"}, {"model": "CA", "acceleration_mps2": [0, 0, 0]},
                 {"model": "CT", "turn_rate_rad_s": 0.1}, {"segments": SEGMENTS}):
        with pytest.raises(ValueError, match="dt"):
            model(spec).step(state(), dt)


def test_direct_constructors_validate_parameters_and_copy_acceleration():
    rng = np.random.default_rng(1)
    assert isinstance(CV(rng, 1), CV)
    with pytest.raises(ValueError, match="acceleration_mps2"):
        CA(rng, 1, acceleration_mps2=[1, False, 0])
    with pytest.raises(ValueError, match="turn_rate_rad_s"):
        CT(rng, 1, turn_rate_rad_s=float("nan"))
    acceleration = np.array([1.0, 0, 0])
    m = CA(rng, 1, acceleration_mps2=acceleration)
    acceleration[:] = 100
    s = state()
    m.step(s, 1)
    np.testing.assert_array_equal(s.v, [11, 0, 0])


def test_segment_times_reject_overflow_and_unrepresentable_boundaries():
    for durations in ([1e308, 1e308], [1e20, 1e-20]):
        with pytest.raises(ValueError, match="duration_s"):
            model({"segments": [{"model": "CV", "duration_s": duration} for duration in durations]})


@pytest.mark.parametrize("motion,path", [
    (None, "motion"), ("CV", "motion"), ({}, "motion.model"),
    ({"model": "unknown"}, "motion.model"), ({"model": True}, "motion.model"),
    ({"model": "CV", "acceleration_mps2": [0, 0, 0]}, "acceleration_mps2"),
    ({"model": "CA"}, "acceleration_mps2"),
    ({"model": "CA", "acceleration_mps2": [0, 1]}, "acceleration_mps2"),
    ({"model": "CA", "acceleration_mps2": [0, True, 0]}, "acceleration_mps2[1]"),
    ({"model": "CA", "acceleration_mps2": [0, float("nan"), 0]}, "acceleration_mps2[1]"),
    ({"model": "CA", "acceleration_mps2": [[0], [1], [2]]}, "acceleration_mps2[0]"),
    ({"model": "CT"}, "turn_rate_rad_s"),
    ({"model": "CT", "turn_rate_rad_s": float("inf")}, "turn_rate_rad_s"),
    ({"model": "CT", "turn_rate_rad_s": False}, "turn_rate_rad_s"),
    ({"model": "CT", "turn_rate_rad_s": "0.1"}, "turn_rate_rad_s"),
    ({"segments": []}, "segments"), ({"segments": None}, "segments"),
    ({"segments": [None]}, "segments[0]"),
    ({"segments": [{"model": "CV"}]}, "duration_s"),
    ({"segments": [{"model": "CV", "duration_s": 0}]}, "duration_s"),
    ({"segments": [{"model": "CV", "duration_s": -1}]}, "duration_s"),
    ({"segments": [{"model": "CV", "duration_s": True}]}, "duration_s"),
    ({"segments": [{"model": "CV", "duration_s": float("nan")}]}, "duration_s"),
    ({"segments": [], "model": "CV"}, "motion"),
])
def test_invalid_motion_parameters_have_paths(motion, path):
    sc = scenario()
    sc["targets"][0]["motion"] = motion
    with pytest.raises(ValueError, match=re.escape(path)):
        validate_scenario(sc)


@pytest.mark.parametrize("missing", ["position", "velocity"])
def test_explicit_motion_requires_initial_state(missing):
    sc = scenario()
    del sc["targets"][0][missing]
    with pytest.raises(ValueError, match=f"targets\\[0\\].{missing}"):
        validate_scenario(sc)


def test_velocity_without_motion_is_not_silently_ignored():
    sc = scenario()
    del sc["targets"][0]["motion"]
    with pytest.raises(ValueError, match="velocity"):
        validate_scenario(sc)


def test_schedule_must_cover_rounded_effective_episode_duration():
    sc = scenario({"segments": [{"model": "CV", "duration_s": 0.5}]})
    sc.update(dt=0.5, duration_s=0.6)
    validate_scenario(sc)             # effective duration is 0.5, not 0.6
    sc["duration_s"] = 0.76          # effective duration is now 1.0
    with pytest.raises(ValueError, match="effective episode duration"):
        validate_scenario(sc)


def test_roundoff_at_schedule_end_and_reset():
    sc = scenario({"segments": [{"model": "CV", "duration_s": 0.1},
                                {"model": "CT", "duration_s": 0.8, "turn_rate_rad_s": 0.2}]})
    sc.update(dt=0.3, duration_s=0.9)
    env = RadarNetEnv(sc)
    env.reset(seed=1)
    while env.agents:
        env.step({a: 0 for a in env.agents})
    final = env.targets[0].p.copy()
    env.reset(seed=1)
    assert env.motions[0].elapsed == 0
    while env.agents:
        env.step({a: 0 for a in env.agents})
    np.testing.assert_array_equal(env.targets[0].p, final)


def test_many_small_steps_do_not_accumulate_false_schedule_overrun():
    s = state(v=(1, 0, 0))
    m = model({"segments": [{"model": "CV", "duration_s": 100}]})
    for _ in range(3333):
        m.step(s, 0.03)
    m.step(s, 0.01)
    assert m.elapsed == 100
    assert s.p[0] == pytest.approx(100, rel=1e-13)


def test_class_labels_do_not_choose_explicit_motion_and_no_rng_draws_in_motion():
    states, motions = [], []
    for label in range(4):
        s, m = spawn_target(label, label, np.random.default_rng(label), 1,
                            position=[4000, 0, 1000], velocity=[10, 0, 0],
                            motion={"segments": SEGMENTS})
        states.append(s)
        motions.append(m)
    rng_states = [deepcopy(m.rng.bit_generator.state) for m in motions]
    for _ in range(4):
        for s, m in zip(states, motions):
            advance(s, m, 0.25)
        for s in states[1:]:
            np.testing.assert_array_equal(s.p, states[0].p)
            np.testing.assert_array_equal(s.v, states[0].v)
            np.testing.assert_array_equal(s.R_WT, states[0].R_WT)
    assert [m.rng.bit_generator.state for m in motions] == rng_states


# Captured from the unchanged legacy implementation before adding analytic models.
@pytest.mark.parametrize("label,expected,rng_state", [
    (0, [-24.43301744909958, 1468.368524053259, 1007.6508530611347,
         -131.92688079317574, -7.2258298197846145, -2.288895609778427, -15.259409104590638],
     140513848826398230406913279291277526521),
    (1, [1058.5219988868462, 2250.4172833235475, 102.6293765101066,
         14.618466364733212, 15.555369825267421, 3.830082699071991, 0],
     7319938573329922607306772083434381644),
    (2, [963.9264110707777, 2114.5774488493603, 186.7162480236644,
         5.785113960428896, -8.87310928894478, -0.14461776590391884, 3.5518326103655395],
     281955145873677190922765594099328208576),
    (3, [979.3221029762027, 2168.9760546840575, 604.6597242794118,
         -1.634434876352521, -2.500863800392157, 1.471188385547803, 0],
     122178296102132076795768006699553331752),
])
def test_legacy_seeded_trajectories_and_rng_are_unchanged(label, expected, rng_state):
    rng = np.random.default_rng(123)
    s, m = spawn_target(0, label, rng, 6500)
    for _ in range(20):
        advance(s, m, 0.5)
    np.testing.assert_allclose(np.concatenate([s.p, s.v, [s.a_lat]]), expected, rtol=1e-14, atol=1e-12)
    assert rng.bit_generator.state["state"]["state"] == rng_state


def test_terrain_clearance_is_an_explicit_departure_from_analytic_motion():
    s = state(p=(0, 0, 35), v=(1, 0, -10))
    advance(s, model({"model": "CV"}), 1, terrain=Terrain.flat())
    assert s.p[2] == 30 and s.v[2] == 0
    assert s.p[2] != 35 - 10


def test_initial_clearance_updates_orientation_and_copies_inputs():
    sc = scenario({"model": "CV"})
    sc["targets"][0].update(position=[4000, 0, 10], velocity=[10, 0, -2])
    before = deepcopy(sc)
    env = RadarNetEnv(sc)
    env.reset()
    target = env.targets[0]
    assert target.p[2] == 30 and target.v[2] == 0
    np.testing.assert_array_equal(target.R_WT, body_rotation(target.v, 0))
    assert sc == before


def test_explicit_motion_recording_and_policy_independence(tmp_path):
    a = RadarNetEnv(scenario(), seed=7)
    b = RadarNetEnv(scenario(), seed=7, record_episode=True)
    a.reset()
    b.reset()
    while a.agents:
        ra = a.step({ag: 1 for ag in a.agents})
        rb = b.step({ag: 1 for ag in b.agents})
        for key in ra[0]:
            np.testing.assert_array_equal(ra[0][key], rb[0][key])
        assert ra[1:4] == rb[1:4]
        np.testing.assert_array_equal(a.targets[0].p, b.targets[0].p)
        assert a.rng_radars[0].bit_generator.state == b.rng_radars[0].bit_generator.state
    export_episode_records(b.episode_records, tmp_path)
    loaded = load_episode_records(tmp_path)
    assert loaded["metadata"]["record_schema_version"] == 2
    assert loaded["metadata"]["config_snapshot"]["scenario"]["targets"][0]["motion"] == {"segments": SEGMENTS}
    c = RadarNetEnv(loaded["metadata"]["config_snapshot"]["scenario"], seed=7)
    c.reset()
    while c.agents:
        c.rng_policy.random(37)
        c.rng_comm[0].random(37)
        c.step({ag: 0 for ag in c.agents})
        expected = loaded["truth"][c.step_count]
        np.testing.assert_array_equal(c.targets[0].p, expected["p"])
