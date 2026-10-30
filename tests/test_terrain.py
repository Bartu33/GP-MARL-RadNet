"""Terrain height model and line-of-sight occlusion."""
import numpy as np

from src.core.modes import Mode
from src.core.radar import RadarAction, RadarParams, RadarState, los_status, radar_step
from src.core.rcs import ConstantRCS
from src.core.targets import TargetState
from src.core.terrain import Terrain


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
    tgt = TargetState(id=0, label=0, p=np.array([4000.0, 0, 300.0]), v=np.zeros(3),
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
