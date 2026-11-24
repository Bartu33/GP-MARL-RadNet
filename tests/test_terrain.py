"""Terrain height model and line-of-sight occlusion."""
import numpy as np
import pytest

from src.core import geometry as geo
from src.core.env import RadarNetEnv
from src.core.modes import Mode
from src.core.radar import RadarAction, RadarParams, RadarState, los_status, radar_step
from src.core.rcs import ConstantRCS
from src.core.scenario import build_radars, load_scenario
from src.core.targets import TargetState, advance, spawn_class_target, spawn_target
from src.core.terrain import Terrain, build_terrain
from src.services.simulation_service import export_episode_records, load_episode_records


def ridge(height=500.0):
    """A north-south ridge at x = 0 on a 20 km grid."""
    n, cell = 200, 100.0
    x = (np.arange(n) + 0.5) * cell - 10_000
    H = np.tile(height * np.exp(-(x / 400.0) ** 2), (n, 1))
    return Terrain(H, -10_000, 10_000, cell, "procedural")


def test_flat_terrain_has_line_of_sight():
    t = Terrain.flat()
    assert t.line_of_sight([0, 0, 15], [8000, 0, 200])


def test_ridge_blocks_low_target_but_not_high_one():
    t = ridge()
    assert t.height(0, 0) > 490
    assert not t.line_of_sight([-5000, 0, 15], [5000, 0, 200])
    assert t.line_of_sight([-5000, 0, 15], [5000, 0, 2000])


def test_earth_curvature_hides_very_low_far_target():
    t = Terrain.flat()
    # 4/3-Earth bulge at mid-path of 60 km is ~53 m
    assert not t.line_of_sight([0, 0, 20], [60_000, 0, 20])
    assert t.line_of_sight([0, 0, 20], [60_000, 0, 400])


def test_radar_step_respects_occlusion():
    t = ridge()
    p = RadarParams(position=(-5000, 0, 15), heading_deg=0.0, false_alarms_per_beam=0.0)
    # moving target: a stationary one would sit in the clutter notch (blind speed)
    tgt = TargetState(id=0, label=0, p=np.array([4000.0, 0, 300.0]), v=np.array([-50.0, 0, 0]),
                      rcs=ConstantRCS(10.0), swerling=0)
    tgt.update_attitude()
    assert los_status(tgt, p, t) == 2 and los_status(tgt, p, None) == 1
    rng = np.random.default_rng(0)
    beam = (0.0, np.arctan2(285.0, 9000.0))
    hits_open = sum(len(radar_step([tgt], RadarAction(Mode.TRACK, beam), RadarState.initial(p), p, 0.5, rng)[0])
                    for _ in range(50))
    hits_ridge = sum(len(radar_step([tgt], RadarAction(Mode.TRACK, beam), RadarState.initial(p), p, 0.5, rng,
                                    terrain=t)[0]) for _ in range(50))
    assert hits_open > 40 and hits_ridge == 0


def test_procedural_terrain_datum_is_central_median():
    t = Terrain.procedural(40.3, -111.7, seed=0)
    g = np.linspace(-7000, 7000, 15)
    X, Y = np.meshgrid(g, g)
    assert abs(np.median(t.height(X, Y))) < 1.0


@pytest.fixture(scope="module", params=[512, 1024])
def valley(request):
    sc = load_scenario("valley_pass")
    sc["terrain"]["resolution"] = request.param
    terrain = build_terrain(sc["site"], sc["terrain"], seed=sc["seed"])
    radars = build_radars(sc)
    for radar in radars:
        x, y, mast = radar.position
        radar.position = (x, y, float(terrain.height(x, y)) + mast)
    return sc, terrain, radars


def valley_target(time):
    target = TargetState(0, 0, np.array([0., -5000 + 50 * time, 200.]),
                         np.array([0., 50., 0.]), ConstantRCS(10), swerling=0)
    target.update_attitude()
    return target


# Geometry layer: no sensors, scheduling, detection draws, or tracking involved.
@pytest.mark.parametrize("time,expected", [
    (0, (True, True)), (55, (False, True)), (100, (True, True)),
    (145, (True, False)), (200, (True, True)),
])
def test_valley_los_geometry_at_both_resolutions(valley, time, expected):
    _, terrain, radars = valley
    target = valley_target(time)
    for radar, clear in zip(radars, expected):
        assert geo.measure(target.p, target.v, radar.pos, radar.heading)[0] < radar.r_max_m
        assert terrain.line_of_sight(radar.pos, target.p) == clear
        assert terrain.line_of_sight(radar.pos, target.p, step_m=5) == clear
        assert los_status(target, radar, terrain) == (1 if clear else 2)
        assert los_status(target, radar, Terrain.flat()) == 1


def test_explicit_hills_are_only_the_gaussian_sum(valley):
    sc, terrain, _ = valley
    rows, cols = terrain.H.shape
    x = terrain.x_west + (np.arange(cols) + 0.5) * terrain.cell
    y = terrain.y_north - (np.arange(rows) + 0.5) * terrain.cell
    X, Y = np.meshgrid(x, y)
    expected = sum(h["height_m"] * np.exp(-((X - h["center_m"][0]) ** 2
                       + (Y - h["center_m"][1]) ** 2) / (2 * h["sigma_m"] ** 2))
                   for h in sc["terrain"]["hills"])
    np.testing.assert_allclose(terrain.H, expected, rtol=1e-14, atol=1e-12)
    assert terrain.base_msl == 0 and terrain.H.min() >= 0
    assert terrain.height(-2000, -1500) > 790
    assert terrain.height(2000, 1500) > 790


def test_minimum_agl_over_entire_interpolated_valley_route(valley):
    _, terrain, radars = valley
    # At fixed x=0, bilinear terrain is piecewise linear in y. Its maximum
    # on this route occurs at a row-centre knot or a route endpoint.
    knots = terrain.y_north - (np.arange(terrain.H.shape[0]) + 0.5) * terrain.cell
    y = np.r_[-5000., knots[(knots >= -5000) & (knots <= 5000)], 5000.]
    assert 200 - np.max(terrain.height(0, y)) >= 30
    # Avoid the out-of-grid edge clamp masking an invalid scenario extent.
    for x, north in [(0, -5000), (0, 5000), *(r.position[:2] for r in radars)]:
        assert terrain.x_west < x < terrain.x_west + terrain.cell * terrain.H.shape[1]
        assert terrain.y_north - terrain.cell * terrain.H.shape[0] < north < terrain.y_north


def test_clearance_does_not_modify_analytic_cv_route(valley):
    sc, terrain, _ = valley
    spec = sc["targets"][0]
    target, motion = spawn_class_target(0, spec["target_class"], spec["variant"], np.random.default_rng(0),
                                        sc["area_radius_m"], position=spec["position"], velocity=spec["velocity"],
                                        motion=spec["motion"])
    for step in range(1, 401):
        advance(target, motion, sc["dt"], terrain=terrain)
        np.testing.assert_array_equal(target.p, [0, -5000 + 25 * step, 200])
        np.testing.assert_array_equal(target.v, [0, 50, 0])


def test_explicit_hills_ignore_seed_and_do_not_create_rng(monkeypatch):
    sc = load_scenario("valley_pass")

    def forbidden(*args, **kwargs):
        pytest.fail("explicit hill construction must not create or consume an RNG")

    monkeypatch.setattr(np.random, "default_rng", forbidden)
    a = build_terrain(sc["site"], sc["terrain"], seed=1)
    b = build_terrain(sc["site"], sc["terrain"], seed=999)
    np.testing.assert_array_equal(a.H, b.H)
    empty = Terrain.procedural(40.3, -111.7, resolution=32, hills=[])
    assert not empty.H.any() and empty.base_msl == 0


def test_legacy_procedural_values_are_unchanged():
    terrain = Terrain.procedural(40.30, -111.70, seed=23, resolution=64)
    np.testing.assert_allclose(
        [terrain.base_msl, terrain.height(-2000, -1500), terrain.height(0, 0), terrain.height(2000, 1500)],
        [1270.7346138288258, 128.3286927760213, 164.71170264956027, 82.12415584435713],
        rtol=1e-14, atol=1e-12)


# Sensor layer: truth-pointed beams are a unit-test fixture, not a runtime policy.
# Fresh resource state per look isolates LOS from thermal/energy scheduling.
@pytest.mark.parametrize("time,expected", [(55, (False, True)), (145, (True, False))])
def test_valley_radar_detections_follow_los_with_no_false_alarms(valley, time, expected):
    _, terrain, radars = valley
    target = valley_target(time)
    for radar, clear in zip(radars, expected):
        assert radar.false_alarms_per_beam == 0
        beam = tuple(geo.measure(target.p, target.v, radar.pos, radar.heading)[1:3])
        hits, flat_hits = 0, 0
        rng = np.random.default_rng(3)
        for _ in range(50):
            for ground in (terrain, Terrain.flat()):
                reports, _ = radar_step([target], RadarAction(Mode.TRACK, beam),
                                        RadarState.initial(radar), radar, 0.5, rng, terrain=ground)
                assert all(not r["is_false_alarm"] and r["truth_id"] == target.id for r in reports)
                if ground is terrain:
                    hits += len(reports)
                else:
                    flat_hits += len(reports)
        assert flat_hits > 40
        assert hits > 40 if clear else hits == 0


def test_valley_episode_recording_is_passive_and_v2_roundtrip(tmp_path):
    # Integration layer: use the actual scenario and existing trackers, but
    # do not make tracking success a condition for accepting LOS geometry.
    sc = load_scenario("valley_pass")
    a = RadarNetEnv(sc, seed=7, local_tracks=True)
    b = RadarNetEnv(sc, seed=7, local_tracks=True, record_episode=True)
    np.testing.assert_equal(a.reset(), b.reset())
    while a.agents:
        actions = {agent: 1 for agent in a.agents}
        np.testing.assert_equal(a.step(actions), b.step(actions))
        np.testing.assert_equal(a._last_reports, b._last_reports)
        for left, right in zip([a.tracker, *a.local_trackers], [b.tracker, *b.local_trackers]):
            assert len(left.tracks) == len(right.tracks)
            for lt, rt in zip(left.tracks, right.tracks):
                np.testing.assert_array_equal(lt.x, rt.x)
                np.testing.assert_array_equal(lt.P, rt.P)
        np.testing.assert_array_equal(b.targets[0].p, [0, -5000 + 25 * b.step_count, 200])
    for name in ("rng_targets", "rng_radars", "rng_comm"):
        for left, right in zip(getattr(a, name), getattr(b, name)):
            assert left.bit_generator.state == right.bit_generator.state
    assert a.rng_policy.bit_generator.state == b.rng_policy.bit_generator.state
    records = b.episode_records
    assert records["metadata"]["completed"]
    assert records["metadata"]["record_schema_version"] == 3
    assert records["metadata"]["config_snapshot"]["scenario"]["terrain"]["hills"] == sc["terrain"]["hills"]
    assert len(records["states"]) == 401
    export_episode_records(records, tmp_path)
    np.testing.assert_equal(load_episode_records(tmp_path), records)
