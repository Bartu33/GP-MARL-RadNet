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
import math
from numbers import Real

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


def _motion_number(value, path, positive=False):
    try:
        valid = (not isinstance(value, (bool, np.bool_)) and isinstance(value, Real)
                 and math.isfinite(value))
    except OverflowError:
        valid = False
    if not valid or (positive and value <= 0):
        raise ValueError(f"{path}: expected a finite {'positive ' if positive else ''}number")
    return float(value)


def _motion_vector(value, path):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1) or len(value) != 3):
        raise ValueError(f"{path}: expected a vector of length 3")
    return np.array([_motion_number(v, f"{path}[{i}]") for i, v in enumerate(value)])


def validate_motion(spec, path="motion", *, segment=False):
    """Validate deterministic motion input; return total duration for a schedule."""
    if not isinstance(spec, dict):
        raise ValueError(f"{path}: expected a mapping")
    if "segments" in spec and not segment:
        if set(spec) != {"segments"}:
            raise ValueError(f"{path}: segments cannot be combined with model or other fields")
        segments = spec["segments"]
        if not isinstance(segments, list) or not segments:
            raise ValueError(f"{path}.segments: expected a nonempty list")
        durations = []
        end = 0.0
        for i, entry in enumerate(segments):
            child = f"{path}.segments[{i}]"
            validate_motion(entry, child, segment=True)
            durations.append(float(entry["duration_s"]))
            try:
                next_end = math.fsum(durations)
            except OverflowError:
                next_end = math.inf
            if not math.isfinite(next_end) or next_end <= end:
                raise ValueError(f"{child}.duration_s: cumulative time must be finite and increasing")
            end = next_end
        return end
    kind = spec.get("model")
    if not isinstance(kind, str) or kind not in ("CV", "CA", "CT"):
        raise ValueError(f"{path}.model: expected CV, CA, or CT")
    parameter = {"CA": "acceleration_mps2", "CT": "turn_rate_rad_s"}.get(kind)
    allowed = {"model"} | ({parameter} if parameter else set()) | ({"duration_s"} if segment else set())
    for key in spec:
        if key not in allowed:
            raise ValueError(f"{path}.{key}: unsupported motion field")
    if segment:
        _motion_number(spec.get("duration_s"), f"{path}.duration_s", positive=True)
    if kind == "CA":
        _motion_vector(spec.get(parameter), f"{path}.{parameter}")
    elif kind == "CT":
        _motion_number(spec.get(parameter), f"{path}.{parameter}")


def _motion_dt(dt):
    dt = _motion_number(dt, "dt")
    if dt < 0:
        raise ValueError("dt: must be >= 0")
    return dt


class CV(MotionModel):
    """Exact constant velocity in world ENU coordinates; no homing or RNG draws."""

    def step(self, s, dt):
        dt = _motion_dt(dt)
        s.p = s.p + s.v * dt
        s.a_lat = 0.0


class CA(MotionModel):
    """Exact constant world-frame acceleration (m/s²)."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0), *, acceleration_mps2):
        super().__init__(rng, area_radius, center)
        self.acceleration = _motion_vector(acceleration_mps2, "acceleration_mps2")

    def step(self, s, dt):
        dt = _motion_dt(dt)
        s.p = s.p + s.v * dt + 0.5 * self.acceleration * dt ** 2
        s.v = s.v + self.acceleration * dt
        speed = np.hypot(*s.v[:2])
        s.a_lat = float(self.acceleration[:2] @ np.array([-s.v[1], s.v[0]]) / speed) if speed > 0 else 0.0


class CT(MotionModel):
    """Exact horizontal turn, positive CCW about +z, with constant vertical speed."""

    def __init__(self, rng, area_radius, center=(0.0, 0.0), *, turn_rate_rad_s):
        super().__init__(rng, area_radius, center)
        self.turn_rate = _motion_number(turn_rate_rad_s, "turn_rate_rad_s")

    def step(self, s, dt):
        dt = _motion_dt(dt)
        theta = _motion_number(self.turn_rate * dt, "turn_rate_rad_s * dt")
        # np.sinc(x) = sin(pi*x)/(pi*x). Half-angle form avoids cancellation
        # in (1-cos(theta))/omega and gives the CV limit even at omega == 0.
        A = dt * np.sinc(theta / np.pi)
        B = dt * (theta / 2) * np.sinc(theta / (2 * np.pi)) ** 2
        vx, vy, vz = s.v
        s.p = s.p + np.array([A * vx - B * vy, B * vx + A * vy, dt * vz])
        c, sn = np.cos(theta), np.sin(theta)
        s.v = np.array([c * vx - sn * vy, sn * vx + c * vy, vz])
        s.a_lat = float(self.turn_rate * np.hypot(vx, vy))


class SegmentedMotion(MotionModel):
    """Finite deterministic schedule; split steps at every exact segment boundary.

    Position and velocity are inherited. At a boundary, lateral acceleration
    uses the next segment (right-continuous). No implicit repeat or tail mode.
    """

    def __init__(self, rng, area_radius, segments, center=(0.0, 0.0)):
        super().__init__(rng, area_radius, center)
        validate_motion({"segments": segments})
        durations = [float(entry["duration_s"]) for entry in segments]
        self.ends = [math.fsum(durations[:i + 1]) for i in range(len(durations))]
        self.models = [build_motion({k: v for k, v in entry.items() if k != "duration_s"},
                                    rng, area_radius, center) for entry in segments]
        self.elapsed = 0.0
        self._time_error = 0.0
        self.segment_index = 0

    def step(self, s, dt):
        dt = _motion_dt(dt)
        if dt == 0:
            self.models[self.segment_index].step(s, 0.0)
            return
        # Compensated clock accumulation prevents thousands of small steps
        # from falsely exhausting an otherwise valid finite schedule.
        increment = dt - self._time_error
        end = self.elapsed + increment
        time_error = (end - self.elapsed) - increment
        if not math.isfinite(end):
            raise ValueError("motion.segments: elapsed time must remain finite")
        tolerance = 32 * np.finfo(float).eps * max(1.0, self.ends[-1], end)
        if end > self.ends[-1] + tolerance:
            raise ValueError("motion.segments: step exceeds total schedule duration")
        end = min(end, self.ends[-1])
        while self.elapsed < end:
            boundary = self.ends[self.segment_index]
            next_time = min(end, boundary)
            self.models[self.segment_index].step(s, next_time - self.elapsed)
            self.elapsed = next_time
            if self.elapsed >= boundary and self.segment_index + 1 < len(self.models):
                self.segment_index += 1
        self._time_error = time_error
        self.models[self.segment_index].step(s, 0.0)  # update bank state at exact boundaries


def build_motion(spec, rng, area_radius, center=(0.0, 0.0)):
    validate_motion(spec)
    if "segments" in spec:
        return SegmentedMotion(rng, area_radius, spec["segments"], center)
    cls = {"CV": CV, "CA": CA, "CT": CT}[spec["model"]]
    return cls(rng, area_radius, center, **{k: v for k, v in spec.items() if k != "model"})


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
                 center=(0.0, 0.0), position=None, *, velocity=None, motion=None) -> tuple[TargetState, MotionModel]:
    """Spawn a legacy behavior, or explicit motion with required position/velocity.

    Class still chooses the default RCS. Explicit motion is deterministic and
    independent of class; the legacy branch retains its original RNG order.
    """
    if motion is not None:
        p = _motion_vector(position, "position")
        v = _motion_vector(velocity, "velocity")
        model = build_motion(motion, rng, area_radius, center)
        rcs, sw = default_rcs(label, rng)
        s = TargetState(id=tid, label=label, p=p.copy(), v=v.copy(), rcs=rcs, swerling=sw)
        model.step(s, 0.0)
        s.update_attitude()
        return s, model
    if velocity is not None:
        raise ValueError("velocity: requires an explicit motion definition")
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
    """Advance with the existing substeps and sampled ground-clearance clamp.

    Legacy models retain their integration. CV/CA/CT propagate analytically
    within each substep; terrain clearance may alter that analytic trajectory.
    """
    h = dt / substeps
    for _ in range(substeps):
        motion.step(s, h)
        keep_above_ground(s, terrain)
    s.update_attitude()
