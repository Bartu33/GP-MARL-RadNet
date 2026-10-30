"""Synthetic aerial targets: motion models and labelled behaviour classes.

The project brief asks for "labelled synthetic target types through
differences in motion and measurement signatures" with non-operational
parameters.  Four abstract behaviour archetypes are defined; none of them
models a specific real platform.

=====  ================  ==============================================
label  name              signature (all parameters drawn from ranges)
=====  ================  ==============================================
0      A_fast_cruise     high speed, wide coordinated turns (bank-to-turn,
                         Alpdemir 2022 Eqs. 6-8), nearly constant altitude,
                         large anisotropic ellipsoid RCS
1      B_agile_hover     stop-and-go: hover segments and short dashes,
                         abrupt heading changes, climbs/descents, small RCS
2      C_erratic_slow    slow, random-walk heading, short-period vertical
                         oscillation, very small fluctuating RCS
3      D_drift           almost passive drift with a wind vector, slow
                         climb, small turbulence, medium RCS
=====  ================  ==============================================

Ranges overlap on purpose so the classification problem is not trivial.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import body_rotation
from .rcs import ConstantRCS, EllipsoidRCS, LookupTableRCS, RCSModel

CLASS_NAMES = ["A_fast_cruise", "B_agile_hover", "C_erratic_slow", "D_drift"]
CLASS_SHORT = ["A · fast", "B · agile", "C · erratic", "D · drift"]
G = 9.81


@dataclass
class TargetState:
    id: int
    label: int
    p: np.ndarray
    v: np.ndarray
    rcs: RCSModel
    swerling: int = 1
    a_lat: float = 0.0
    R_WT: np.ndarray = field(default_factory=lambda: np.eye(3))

    def update_attitude(self):
        self.R_WT = body_rotation(self.v, self.a_lat)


class MotionModel:
    """Base class.  ``step`` advances ``state`` in place by ``dt`` seconds."""

    def __init__(self, rng: np.random.Generator, area_radius: float, center=(0.0, 0.0)):
        self.rng = rng
        self.area_radius = area_radius
        self.center = np.asarray(center, float)

    def _homing_turn(self, p, heading):
        """Signed heading error towards the area centre when outside the area."""
        d = self.center - p[:2]
        if np.linalg.norm(d) < self.area_radius:
            return None
        err = np.arctan2(d[1], d[0]) - heading
        return (err + np.pi) % (2 * np.pi) - np.pi

    def step(self, s: TargetState, dt: float):  # pragma: no cover - interface
        raise NotImplementedError


class FastCruise(MotionModel):
    """Bank-to-turn kinematics: x' = v cos psi, y' = v sin psi, psi' = u / v."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0)):
        super().__init__(rng, area_radius, center)
        self.speed = rng.uniform(110, 230)
        self.u = 0.0
        self.u_cmd = 0.0
        self.t_next = 0.0
        self.alt_rate = 0.0

    def step(self, s, dt):
        psi = np.arctan2(s.v[1], s.v[0])
        self.t_next -= dt
        if self.t_next <= 0:
            self.t_next = self.rng.uniform(8, 25)
            self.u_cmd = self.rng.choice([0.0, 0.0, 1.0, -1.0]) * self.rng.uniform(5, 25)
            self.alt_rate = self.rng.normal(0, 1.5)
        home = self._homing_turn(s.p, psi)
        if home is not None:
            self.u_cmd = np.sign(home) * 25.0
        self.u += np.clip(self.u_cmd - self.u, -8 * dt, 8 * dt)      # roll-rate limit
        psi += self.u / self.speed * dt
        vz = self.alt_rate
        s.v = np.array([self.speed * np.cos(psi), self.speed * np.sin(psi), vz])
        s.p = s.p + s.v * dt
        s.a_lat = self.u


class AgileHover(MotionModel):
    """Alternating hover / dash segments with abrupt heading changes."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0)):
        super().__init__(rng, area_radius, center)
        self.t_next = 0.0
        self.v_cmd = np.zeros(3)

    def step(self, s, dt):
        self.t_next -= dt
        if self.t_next <= 0:
            hover = self.rng.random() < 0.4
            self.t_next = self.rng.uniform(2, 6) if hover else self.rng.uniform(3, 12)
            speed = self.rng.uniform(0, 2) if hover else self.rng.uniform(8, 26)
            psi = self.rng.uniform(-np.pi, np.pi)
            home = self._homing_turn(s.p, psi)
            if home is not None:
                psi += home
            vz = self.rng.normal(0, 2.5)
            if s.p[2] < 40:
                vz = abs(vz)
            self.v_cmd = np.array([speed * np.cos(psi), speed * np.sin(psi), vz])
        dv = np.clip(self.v_cmd - s.v, -6 * dt, 6 * dt)               # agile, but finite accel
        a = dv / dt
        s.v = s.v + dv
        s.p = s.p + s.v * dt
        horiz = np.array([-s.v[1], s.v[0]]) / max(np.hypot(*s.v[:2]), 1e-6)
        s.a_lat = float(np.dot(a[:2], horiz))


class ErraticSlow(MotionModel):
    """Slow flight with a random-walk turn rate and a short-period vertical
    oscillation (flapping-like)."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0)):
        super().__init__(rng, area_radius, center)
        self.speed = rng.uniform(7, 19)
        self.turn_rate = 0.0
        self.osc_amp = rng.uniform(0.4, 2.5)
        self.osc_freq = rng.uniform(0.3, 1.2)
        self.phase = rng.uniform(0, 2 * np.pi)
        self.t = 0.0

    def step(self, s, dt):
        self.t += dt
        psi = np.arctan2(s.v[1], s.v[0])
        # Ornstein-Uhlenbeck turn rate
        self.turn_rate += -0.5 * self.turn_rate * dt + 0.35 * np.sqrt(dt) * self.rng.normal()
        home = self._homing_turn(s.p, psi)
        if home is not None:
            self.turn_rate = 0.3 * np.sign(home)
        psi += self.turn_rate * dt
        speed = self.speed * (1 + 0.15 * np.sin(0.2 * self.t + self.phase))
        vz = (self.osc_amp * 2 * np.pi * self.osc_freq
              * np.cos(2 * np.pi * self.osc_freq * self.t + self.phase)) + self.rng.normal(0, 0.3)
        s.v = np.array([speed * np.cos(psi), speed * np.sin(psi), vz])
        s.p = s.p + s.v * dt
        s.a_lat = speed * self.turn_rate


class Drift(MotionModel):
    """Passive drift with a slowly rotating wind vector and turbulence."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0)):
        super().__init__(rng, area_radius, center)
        self.wind_speed = rng.uniform(2, 9)
        self.wind_dir = rng.uniform(-np.pi, np.pi)
        self.climb = rng.uniform(-0.3, 1.8)
        self.gust = np.zeros(3)

    def step(self, s, dt):
        self.wind_dir += 0.01 * self.rng.normal() * np.sqrt(dt)
        home = self._homing_turn(s.p, self.wind_dir)
        if home is not None:
            self.wind_dir += np.sign(home) * 0.02 * dt
        self.gust += -0.3 * self.gust * dt + 0.25 * np.sqrt(dt) * self.rng.normal(size=3)
        s.v = np.array([self.wind_speed * np.cos(self.wind_dir),
                        self.wind_speed * np.sin(self.wind_dir), self.climb]) + self.gust
        s.p = s.p + s.v * dt
        s.a_lat = 0.0


MOTION_BY_LABEL = {0: FastCruise, 1: AgileHover, 2: ErraticSlow, 3: Drift}
ALT_RANGE = {0: (700, 2200), 1: (40, 350), 2: (40, 700), 3: (300, 1600)}


def default_rcs(label: int, rng: np.random.Generator) -> tuple[RCSModel, int]:
    """Synthetic RCS model and Swerling case for each archetype."""
    if label == 0:
        return EllipsoidRCS(rng.uniform(3, 5), rng.uniform(0.8, 1.3), rng.uniform(0.5, 0.8)), 1
    if label == 1:
        return EllipsoidRCS(rng.uniform(0.25, 0.4), rng.uniform(0.25, 0.4), rng.uniform(0.08, 0.15)), 1
    if label == 2:
        return ConstantRCS(rng.uniform(0.008, 0.05)), 1
    return LookupTableRCS.synthetic(nose_dbsm=-8, side_dbsm=-2, tail_dbsm=-6, top_dbsm=0), 0


def spawn_target(tid: int, label: int, rng: np.random.Generator, area_radius: float,
                 center=(0.0, 0.0), position=None) -> tuple[TargetState, MotionModel]:
    """Create a target of a given class at a random (or given) position."""
    center = np.asarray(center, float)
    if position is None:
        ang = rng.uniform(-np.pi, np.pi)
        rad = area_radius * np.sqrt(rng.uniform(0.1, 0.8))
        xy = center + rad * np.array([np.cos(ang), np.sin(ang)])
        position = [xy[0], xy[1], rng.uniform(*ALT_RANGE[label])]
    motion = MOTION_BY_LABEL[label](rng, area_radius, center)
    psi = rng.uniform(-np.pi, np.pi)
    speed0 = getattr(motion, "speed", 5.0)
    rcs, sw = default_rcs(label, rng)
    s = TargetState(id=tid, label=label, p=np.asarray(position, float),
                    v=np.array([speed0 * np.cos(psi), speed0 * np.sin(psi), 0.0]),
                    rcs=rcs, swerling=sw)
    s.update_attitude()
    return s, motion


MIN_AGL_M = 30.0


def keep_above_ground(s: TargetState, terrain=None, min_agl: float = MIN_AGL_M):
    """Hold the target at least ``min_agl`` above the local ground."""
    ground = float(terrain.height(s.p[0], s.p[1])) if terrain is not None else 0.0
    if s.p[2] < ground + min_agl:
        s.p[2] = ground + min_agl
        s.v[2] = max(s.v[2], 0.0)


def advance(s: TargetState, motion: MotionModel, dt: float, substeps: int = 5, terrain=None):
    """Integrate a target over dt with a few Euler sub-steps."""
    h = dt / substeps
    for _ in range(substeps):
        motion.step(s, h)
        keep_above_ground(s, terrain)
    s.update_attitude()
