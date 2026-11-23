"""Missile terminal-dive episodes over open rolling terrain, defended by 1-3 radars.

Scenario: a supersonic missile (``missile/terminal_dive``) cruises at high
altitude towards a defended point, pushes over into a steep dive and is
followed until it is a few hundred metres above the ground near that point
(impact is not simulated).  The radars stand on high ground around the
defended point, face the threat axis and search the elevation band the
threat crosses, as a point-defence deployment would.

Profiles (one per episode, cycled):

* ``steep_dive``     high cruise, push-over to 55-70 deg, straight dive
* ``shallow_dive``   medium-high cruise, 25-35 deg glide-dive
* ``jink_dive``      push-over to 40-55 deg, evasive S-jinks (8-15 g, 1-2 s) all the way down
* ``dive_pullout``   steep dive, pull-out at low altitude, short low-level dash
* ``weave_dive``     S-weave during cruise, push-over, dive with a final jink

Every manoeuvre is a CV / CT / CA segment inside the variant envelope of
``src/core/target_classes.py``; the push-over and pull-out arcs are split into
CA pieces of <= 12 deg so speed stays nearly constant along the arc.  The
whole trajectory is propagated against the terrain at planning time.
Plans use the planner schema with recipe version 2 and are written by
``synthetic_dataset_service.generate_dataset``.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math

import numpy as np

from src.core.env import _cached_terrain
from src.core.scenario import scenario_snapshot, validate_scenario
from src.core.target_classes import TARGET_CLASSES, RCS_ASPECTS, variant_spec
from src.services.class_dataset_service import (
    CHECK_STEP_S, RECIPE_VERSION, _effective_radar, check_trajectory, propagate,
)
from src.services.scenario_planner import SPLITS, _digest, _json_value, _physical_numbers, terrain_group_id

G = 9.81
CLASS, VARIANT = "missile", "terminal_dive"
DIVE_PROFILES = ("steep_dive", "shallow_dive", "jink_dive", "dive_pullout", "weave_dive")
EVASIVE_PROFILES = ("popup_dive", "barrel_roll_dive", "dogleg_serpentine", "break_turn_dive", "random_jink_dive")
PROFILES = DIVE_PROFILES + EVASIVE_PROFILES
ARC_PIECE_RAD = math.radians(12.0)

DEFAULT_RECIPE = {
    "recipe_version": RECIPE_VERSION,
    "dataset_name": "missile_dive_pilot10",
    "master_seed": 20261124,
    "episodes": 5,
    "profiles": list(PROFILES),
    "dt": 0.5,
    "max_track_slots": 4,
    "area_radius_m": 12000.0,
    "site": {"name": "Rolling Hills Test Range", "lat": 40.30, "lon": -111.70},
    "radar_count": [1, 3],
    "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "terrain": {"resolution": 256, "hills": [12, 20], "height_m": [60.0, 450.0],
                "sigma_m": [700.0, 2500.0], "extent_m": 13000.0},
    "defended_point_radius_m": 1500.0,
    "radar": {
        "distance_from_point_m": [1500.0, 7000.0],
        "min_separation_m": 2000.0,
        "high_ground_quantile": 0.80,     # sites drawn from the highest 20 % of candidate cells
        "local_high_quantile": 0.80,      # ... that are above 80 % of the ground within 1.5 km
        "mast_m": [8.0, 20.0],
        "heading_jitter_deg": 10.0,
        "max_bars": 6,
        "defaults": {"fov_az_deg": 45.0, "search_sector_deg": 70.0, "r_max_m": 20000.0,
                     "el_max_deg": 70.0},
    },
    # Tracking a 15-20 g diving missile: wide prior on the unobserved tangential velocity,
    # white-acceleration PSD for ~14 g manoeuvres, and short-lived candidates so false
    # alarms leave the track table quickly (used with the central_greedy_confirm policy).
    "tracker": {"tentative_timeout_s": 1.0, "init_v_std_mps": 400.0, "q": 10000.0},
    "policy": "central_greedy_confirm",
    "coverage": {"min_visible_fraction": 0.5, "min_visible_fraction_low_level": 0.3, "sample_step_s": 0.5, "max_radar_attempts": 200,
                 "max_episode_attempts": 60},
}


def _merge(base, over):
    out = deepcopy(base)
    for key, value in (over or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) \
            else deepcopy(value)
    return out


def resolve_recipe(recipe=None):
    r = _merge(DEFAULT_RECIPE, recipe or {})
    if r["recipe_version"] != RECIPE_VERSION:
        raise ValueError(f"recipe_version: expected {RECIPE_VERSION}")
    if set(r) - set(DEFAULT_RECIPE):
        raise ValueError(f"recipe: unsupported fields {sorted(set(r) - set(DEFAULT_RECIPE))}")
    if not r["profiles"] or set(r["profiles"]) - set(PROFILES):
        raise ValueError(f"profiles: subset of {list(PROFILES)}")
    lo, hi = r["radar_count"]
    if not 1 <= lo <= hi <= 3:
        raise ValueError("radar_count: 1 <= lo <= hi <= 3")
    ratios = [Fraction(str(r["split_ratios"][s])) for s in SPLITS]
    if sum(ratios) != 1:
        raise ValueError("split_ratios: must sum to 1")
    return r


# ---------------------------------------------------------------------------
# terrain
# ---------------------------------------------------------------------------

def rolling_hills(rng, cfg):
    """Open rolling countryside: a sum of isolated Gaussian hills, no valley walls."""
    n = int(rng.integers(cfg["hills"][0], cfg["hills"][1] + 1))
    ext = cfg["extent_m"]
    return [{"center_m": [round(float(x), 1) for x in rng.uniform(-ext, ext, 2)],
             "height_m": round(float(rng.uniform(*cfg["height_m"])), 1),
             "sigma_m": round(float(rng.uniform(*cfg["sigma_m"])), 1)} for _ in range(n)]


# ---------------------------------------------------------------------------
# trajectory
# ---------------------------------------------------------------------------

def _dir(psi, gamma):
    """Unit velocity for heading psi and flight-path angle gamma (negative = descending)."""
    return np.array([math.cos(gamma) * math.cos(psi), math.cos(gamma) * math.sin(psi), math.sin(gamma)])


def _ceil_dt(t, dt):
    return max(dt, math.ceil(t / dt - 1e-9) * dt)


def _arc(v_from, v_to, a_n, dt):
    """CA pieces taking velocity v_from to v_to along a great-circle arc at about ``a_n``."""
    s0, s1 = np.linalg.norm(v_from), np.linalg.norm(v_to)
    u0, u1 = v_from / s0, v_to / s1
    angle = math.acos(float(np.clip(u0 @ u1, -1.0, 1.0)))
    n = max(1, math.ceil(angle / ARC_PIECE_RAD))
    pieces, v = [], v_from.copy()
    for k in range(1, n + 1):
        f = k / n
        # slerp of the direction, linear speed change
        u = (math.sin((1 - f) * angle) * u0 + math.sin(f * angle) * u1) / math.sin(angle) if angle > 1e-9 else u1
        v_next = (s0 + f * (s1 - s0)) * u
        dv = v_next - v
        t = _ceil_dt(np.linalg.norm(dv) / a_n, dt)
        pieces.append({"model": "CA", "acceleration_mps2": [round(float(c), 5) for c in dv / t], "duration_s": t})
        v = v_next
    return pieces


def _run(v0, segments):
    """Velocity after the segments and the vertical displacement they produce."""
    P, V = propagate([0.0, 0.0, 0.0], v0, {"segments": segments}, sum(s["duration_s"] for s in segments))
    return V[-1], P[-1]


def dive_profile(profile, rng, dt, spec):
    """Segments of one dive profile starting level at the origin, heading +x.

    Returns (v0, segments, kinds, terminal_agl, start_agl_nominal).  The caller
    rotates the approach azimuth by rotating v0 and every acceleration.
    """
    cap_n = 0.8 * spec.max_lateral_accel_mps2
    speed = float(rng.uniform(*{"shallow_dive": (600.0, 750.0)}.get(profile, (750.0, 950.0))))
    h0 = float(rng.uniform(*{"steep_dive": (9000.0, 12000.0), "shallow_dive": (5000.0, 7000.0),
                             "jink_dive": (8000.0, 11000.0), "dive_pullout": (7000.0, 10000.0),
                             "weave_dive": (7000.0, 10000.0)}[profile]))
    gamma = -math.radians(float(rng.uniform(*{"steep_dive": (55.0, 70.0), "shallow_dive": (25.0, 35.0),
                                              "jink_dive": (40.0, 55.0), "dive_pullout": (45.0, 60.0),
                                              "weave_dive": (35.0, 50.0)}[profile])))
    terminal_agl = float(rng.uniform(150.0, 400.0))
    v0 = speed * _dir(0.0, 0.0)
    segs, kinds = [], []

    cruise = _ceil_dt(float(rng.uniform(3.0, 8.0)), dt)
    if profile == "weave_dive":
        omega = float(rng.uniform(0.4, 0.8)) * min(spec.max_turn_rate_rad_s, cap_n / speed)
        half = _ceil_dt(cruise / 2 + 2.0, dt)
        segs += [{"model": "CT", "turn_rate_rad_s": round(omega, 5), "duration_s": half},
                 {"model": "CT", "turn_rate_rad_s": round(-omega, 5), "duration_s": 2 * half},
                 {"model": "CT", "turn_rate_rad_s": round(omega, 5), "duration_s": half}]
        kinds.append("weave")
    else:
        segs.append({"model": "CV", "duration_s": cruise})
        kinds.append("cruise")

    v, _ = _run(v0, segs)
    psi = math.atan2(v[1], v[0])
    s_dive = min(spec.speed_range_mps[1] - 20.0, speed + float(rng.uniform(0.0, 120.0)))  # gravity gain
    a_n = float(rng.uniform(0.5, 1.0)) * cap_n
    segs += _arc(v, s_dive * _dir(psi, gamma), a_n, dt)
    kinds.append("push_over")
    v, p = _run(v0, segs)

    if profile == "dive_pullout":
        # pull out so that the level dash flies at dash_agl; arc altitude loss ~ R (1 - cos gamma), R = v^2 / a_n
        dash_agl = float(rng.uniform(150.0, 300.0))
        a_po = float(rng.uniform(0.6, 1.0)) * cap_n
        loss = s_dive ** 2 / a_po * (1 - math.cos(gamma))
        t_dive = _ceil_dt(max(dt, (h0 + p[2] - (dash_agl + loss)) / (-v[2])), dt)
        segs.append({"model": "CV", "duration_s": t_dive})
        kinds.append("dive")
        v, p = _run(v0, segs)
        segs += _arc(v, np.linalg.norm(v) * _dir(math.atan2(v[1], v[0]), 0.0), a_po, dt)
        kinds.append("pull_out")
        segs.append({"model": "CV", "duration_s": _ceil_dt(float(rng.uniform(3.0, 6.0)), dt)})
        kinds.append("low_dash")
        return v0, segs, kinds, dash_agl, h0

    remaining = h0 + p[2] - terminal_agl          # p[2] < 0 after the push-over
    t_dive = _ceil_dt(max(dt, remaining / (-v[2])), dt)
    if profile == "jink_dive":
        # alternating horizontal turns of 8-15 g; equal and opposite halves keep the mean heading
        vh = math.hypot(v[0], v[1])
        left, sign = t_dive, float(rng.choice([-1.0, 1.0]))
        first = True
        while left > 1e-9:
            g_load = float(rng.uniform(8.0, 15.0)) * G
            omega = sign * min(spec.max_turn_rate_rad_s, g_load / vh)
            span = _ceil_dt(float(rng.uniform(1.0, 2.0)), dt)
            piece = min(left, span / 2 if first else span)   # half-length first jink centres the weave
            segs.append({"model": "CT", "turn_rate_rad_s": round(omega, 5), "duration_s": piece})
            left -= piece
            sign, first = -sign, False
        kinds.append("jinks")
    elif profile == "weave_dive":
        jink = _ceil_dt(min(3.0, t_dive / 3), dt)
        vh = math.hypot(v[0], v[1])
        omega = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.0)) * min(spec.max_turn_rate_rad_s, cap_n / vh)
        segs += [{"model": "CV", "duration_s": max(dt, t_dive - jink)},
                 {"model": "CT", "turn_rate_rad_s": round(omega, 5), "duration_s": jink}]
        kinds += ["dive", "jink"]
    else:
        segs.append({"model": "CV", "duration_s": t_dive})
        kinds.append("dive")
    return v0, segs, kinds, terminal_agl, h0


def _rotate(vec, psi):
    c, s = math.cos(psi), math.sin(psi)
    return [c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]]


def in_sector(angle, sector):
    """Map an angle drawn on [-pi, pi) into ``sector`` = (centre, half width) without another draw."""
    if sector is None:
        return angle
    centre, half = sector
    return float(centre + angle / math.pi * half)


def plan_dive_target(profile, rng, terrain, point, dt, sector=None):
    """Missile on ``profile`` ending near the defended ``point``; returns (target, info) or raises.

    ``sector`` (centre, half width in rad) limits the threat axis: the direction from
    the defended point towards the side the missile comes from.
    """
    spec = variant_spec(CLASS, VARIANT)
    rejected = {}
    for _ in range(200):
        v0, segs, kinds, terminal_agl, h0 = dive_profile(profile, rng, dt, spec)
        duration = sum(s["duration_s"] for s in segs)
        approach = float(rng.uniform(-math.pi, math.pi))      # direction the missile flies towards
        if sector is not None:
            approach = in_sector(approach, sector) + math.pi
        v0w = _rotate(v0, approach)
        segw = []
        for s in segs:
            s = dict(s)
            if s["model"] == "CA":
                s["acceleration_mps2"] = [round(float(c), 5) for c in _rotate(s["acceleration_mps2"], approach)]
            segw.append(s)
        P, _ = propagate([0.0, 0.0, 0.0], v0w, {"segments": segw}, duration)
        end = P[-1]
        # aim: the dive line (or the low dash) ends just short of the defended point
        aim = np.asarray(point, float) + rng.uniform(-300.0, 300.0, 2)
        start_xy = aim - end[:2]
        ground_end = float(terrain.height(*aim))
        z0 = ground_end + terminal_agl - end[2]
        position = [round(float(start_xy[0]), 2), round(float(start_xy[1]), 2), round(float(z0), 2)]
        velocity = [round(float(c), 4) for c in v0w]
        P, V = propagate(position, velocity, {"segments": segw}, duration)
        reason = check_trajectory(spec, P, V, terrain)
        start_agl = z0 - float(terrain.height(*start_xy))
        if reason is None and not 2000.0 <= start_agl <= spec.altitude_agl_range_m[1]:
            reason = "start_altitude"
        if reason is None:
            target = {"target_class": CLASS, "variant": VARIANT, "label": TARGET_CLASSES[CLASS]["label"],
                      "position": position, "velocity": velocity, "motion": {"segments": segw},
                      "rcs_dbsm": {a: round(float(rng.uniform(*spec.rcs_dbsm[a])), 3) for a in RCS_ASPECTS}}
            return target, {"profile": profile, "maneuvers": kinds, "trajectory": P, "duration_s": duration,
                            "start_agl_m": round(start_agl, 1), "approach_deg": round(math.degrees(approach), 1),
                            "aim": aim}
        rejected[reason] = rejected.get(reason, 0) + 1
    raise ValueError(f"{profile}: no valid trajectory ({rejected})")


# ---------------------------------------------------------------------------
# evasive profiles: a small autopilot flies the missile in 0.25 s CA pieces
# ---------------------------------------------------------------------------

PIECE_S = 0.25
LOW_LEVEL_PROFILES = ("popup_dive", "dogleg_serpentine")


class _Autopilot:
    """Guidance + manoeuvre overlay -> piecewise-constant acceleration within the envelope.

    Each 0.25 s piece is one CA segment; the commanded acceleration is split
    into along-track and normal parts and clipped to the variant limits
    (a_long, 0.85 a_n,max, heading rate <= 0.95 omega_max, |v_z|, speed band),
    then the state is advanced exactly as the simulator CA model does.
    """

    def __init__(self, spec, terrain, p, v):
        self.spec, self.terrain = spec, terrain
        self.p, self.v = np.asarray(p, float), np.asarray(v, float)
        self.t = 0.0
        self.segs = []
        self.a_n_max = 0.85 * spec.max_lateral_accel_mps2

    def agl(self):
        return float(self.p[2] - self.terrain.height(self.p[0], self.p[1]))

    def ground_ahead(self, seconds=(0.0, 0.5, 1.0, 2.0, 3.0, 4.0)):
        pts = self.p[:2][None, :] + np.array(seconds)[:, None] * self.v[:2][None, :]
        return float(np.max(self.terrain.height(pts[:, 0], pts[:, 1])))

    def track(self, heading, gamma, speed, tau=0.8):
        """Acceleration steering the velocity towards (heading, flight-path angle, speed)."""
        return (speed * _dir(heading, gamma) - self.v) / tau

    def pursue(self, aim3, speed, tau=0.8):
        d = np.asarray(aim3, float) - self.p
        return (speed * d / np.linalg.norm(d) - self.v) / tau

    def normal_basis(self):
        u = self.v / np.linalg.norm(self.v)
        ref = np.array([0.0, 0.0, 1.0]) if abs(u[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n1 = np.cross(u, ref)
        n1 /= np.linalg.norm(n1)
        return n1, np.cross(u, n1)          # n1 lateral (horizontal), n2 completes the normal plane

    def step(self, a_cmd, h=PIECE_S):
        spec, v = self.spec, self.v
        s = float(np.linalg.norm(v))
        u = v / s
        a_l = float(a_cmd @ u)
        a_n = a_cmd - a_l * u
        a_l = float(np.clip(a_l, -spec.max_long_accel_mps2, spec.max_long_accel_mps2))
        lo, hi = spec.speed_range_mps
        if s + a_l * h > hi - 10.0:
            a_l = (hi - 10.0 - s) / h
        if s + a_l * h < lo + 10.0:
            a_l = (lo + 10.0 - s) / h
        n = float(np.linalg.norm(a_n))
        if n > self.a_n_max:
            a_n *= self.a_n_max / n
        vh = v[:2]
        nh = float(np.linalg.norm(vh))
        if nh > 1.0:                         # heading-rate cap on the horizontal lateral part
            e = np.array([-vh[1], vh[0]]) / nh
            lat = float(a_n[:2] @ e)
            cap = 0.95 * spec.max_turn_rate_rad_s * nh
            if abs(lat) > cap:
                a_n[:2] -= (lat - math.copysign(cap, lat)) * e
        a = a_l * u + a_n
        vz_cap = 0.98 * spec.max_climb_rate_mps
        if abs(v[2] + a[2] * h) > vz_cap:
            a[2] = (math.copysign(vz_cap, v[2] + a[2] * h) - v[2]) / h
        piece = {"model": "CA", "acceleration_mps2": [round(float(c), 4) for c in a], "duration_s": h}
        a = np.array(piece["acceleration_mps2"])
        self.segs.append(piece)
        self.p = self.p + v * h + 0.5 * a * h * h
        self.v = v + a * h
        self.t += h


def _heading_to(p, xy):
    return math.atan2(xy[1] - p[1], xy[0] - p[0])


def _tf_gamma(ap, agl_cmd, lim_deg=25.0):
    """Terrain following: climb or descend towards ground-ahead + agl_cmd."""
    z_cmd = ap.ground_ahead() + agl_cmd
    lead = max(400.0, 2.0 * float(np.linalg.norm(ap.v[:2])))
    return float(np.clip(math.atan2(z_cmd - ap.p[2], lead), -math.radians(lim_deg), math.radians(lim_deg)))


def fly_evasive(profile, rng, terrain, aim, dt, spec, max_t=60.0, sector=None):
    """Fly one evasive profile; returns (position, velocity, segments, phases, start_agl, approach_deg)."""
    g = G
    aim = np.asarray(aim, float)
    aim3 = np.array([aim[0], aim[1], float(terrain.height(*aim))])
    terminal_agl = float(rng.uniform(150.0, 350.0))
    low = profile in LOW_LEVEL_PROFILES
    speed = float(rng.uniform(500.0, 650.0) if low else rng.uniform(700.0, 900.0))
    bearing = in_sector(float(rng.uniform(-math.pi, math.pi)), sector)   # from the aim point towards the launch side

    if profile == "dogleg_serpentine":
        r_w = float(rng.uniform(5000.0, 7000.0))
        wp = aim + r_w * np.array([math.cos(bearing), math.sin(bearing)])
        delta = float(rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(40.0, 70.0)))
        start_xy = wp + float(rng.uniform(5000.0, 7000.0)) * np.array([math.cos(bearing + delta),
                                                                        math.sin(bearing + delta)])
    else:
        wp = None
        start_xy = aim + float(rng.uniform(11000.0, 13500.0)) * np.array([math.cos(bearing), math.sin(bearing)])
    psi0 = _heading_to(start_xy, wp if wp is not None else aim)

    if profile == "popup_dive":
        r_pop = float(rng.uniform(5000.0, 7000.0))
        gamma_pop = math.radians(rng.uniform(25.0, 35.0))
        h_pop = float(rng.uniform(1200.0, 2200.0))
    elif profile == "barrel_roll_dive":
        depress = math.radians(rng.uniform(35.0, 50.0))
        roll_a = float(rng.uniform(12.0, 22.0)) * g
        roll_w = 2 * math.pi / float(rng.uniform(1.2, 2.0))
        roll_sign = float(rng.choice([-1.0, 1.0]))
    elif profile == "dogleg_serpentine":
        serp_a = float(rng.uniform(8.0, 14.0)) * g
        serp_T = float(rng.uniform(2.0, 3.5))
        r_pop = float(rng.uniform(2500.0, 3500.0))
        h_pop = float(rng.uniform(400.0, 700.0))
    elif profile == "break_turn_dive":
        t_break = float(rng.uniform(4.0, 8.0))
        break_len = float(rng.uniform(2.0, 3.5))
        break_sign = float(rng.choice([-1.0, 1.0]))
        boost = float(rng.uniform(80.0, 150.0))
        depress = math.radians(rng.uniform(40.0, 55.0))
        dive_break_agl = float(rng.uniform(2500.0, 3500.0))
        dive_break_len = 2.0
    else:  # random_jink_dive
        depress = math.radians(rng.uniform(30.0, 45.0))
        jink_until, jink = 0.0, np.zeros(3)

    if low:
        agl_cmd = float(rng.uniform(40.0, 80.0))
        z0 = float(terrain.height(*start_xy)) + agl_cmd
    elif profile == "barrel_roll_dive" or profile == "break_turn_dive" or profile == "random_jink_dive":
        # start well below the dive threshold so a level cruise precedes the push-over
        dist0 = float(np.hypot(*(aim - start_xy)))
        look = depress - math.radians(rng.uniform(12.0, 20.0))
        z0 = aim3[2] + float(np.clip(dist0 * math.tan(look), 4000.0, 10000.0))
    else:
        z0 = float(terrain.height(*aim)) + float(rng.uniform(6000.0, 10000.0))
    ap = _Autopilot(spec, terrain, [start_xy[0], start_xy[1], z0], speed * _dir(psi0, 0.0))
    start_agl = ap.agl()

    phases, mode, t_mode = [], "cruise", 0.0
    while ap.t < max_t:
        dist = float(np.hypot(*(aim - ap.p[:2])))
        on_grid = abs(ap.t / dt - round(ap.t / dt)) < 1e-9
        # stop on the dt grid before the next step would take the missile below the terminal altitude
        if mode in ("dive", "terminal") and on_grid and ap.agl() + min(0.0, ap.v[2]) * dt < terminal_agl:
            break
        if profile == "popup_dive":
            if mode == "cruise" and dist <= r_pop:
                mode, t_mode = "pop_up", ap.t
            elif mode == "pop_up" and ap.agl() >= h_pop:
                mode, t_mode = "dive", ap.t
            if mode == "cruise":
                a = ap.track(_heading_to(ap.p, aim), _tf_gamma(ap, agl_cmd), speed)
            elif mode == "pop_up":
                a = ap.track(_heading_to(ap.p, aim), gamma_pop, speed)
            else:
                a = ap.pursue(aim3, speed + 100.0)
        elif profile == "barrel_roll_dive":
            if mode == "cruise" and math.atan2(ap.p[2] - aim3[2], dist) >= depress:
                mode, t_mode = "dive", ap.t
            if mode == "cruise":
                a = ap.track(_heading_to(ap.p, aim), 0.0, speed)
            else:
                n1, n2 = ap.normal_basis()
                ph = roll_sign * roll_w * (ap.t - t_mode)
                a = ap.pursue(aim3, speed + 120.0, tau=1.5) + roll_a * (math.cos(ph) * n1 + math.sin(ph) * n2)
        elif profile == "dogleg_serpentine":
            if mode == "cruise" and float(np.hypot(*(wp - ap.p[:2]))) <= 800.0:
                mode, t_mode = "turn_in", ap.t
            elif mode == "turn_in" and abs(math.remainder(
                    _heading_to(ap.p, aim) - math.atan2(ap.v[1], ap.v[0]), 2 * math.pi)) < 0.1:
                mode, t_mode = "serpentine", ap.t
            elif mode == "serpentine" and dist <= r_pop:
                mode, t_mode = "pop_up", ap.t
            elif mode == "pop_up" and ap.agl() >= h_pop:
                mode, t_mode = "dive", ap.t
            if mode == "cruise":
                a = ap.track(_heading_to(ap.p, wp), _tf_gamma(ap, agl_cmd), speed)
            elif mode in ("turn_in", "serpentine"):
                a = ap.track(_heading_to(ap.p, aim), _tf_gamma(ap, agl_cmd), speed, tau=0.5)
                if mode == "serpentine":
                    n1, _ = ap.normal_basis()
                    a = a + serp_a * math.sin(2 * math.pi * (ap.t - t_mode) / serp_T) * n1
            elif mode == "pop_up":
                a = ap.track(_heading_to(ap.p, aim), math.radians(30.0), speed)
            else:
                a = ap.pursue(aim3, speed + 80.0)
        elif profile == "break_turn_dive":
            if mode == "cruise" and ap.t >= t_break:
                mode, t_mode = "break", ap.t
            elif mode == "break" and ap.t - t_mode >= break_len:
                mode, t_mode = "reacquire", ap.t
            elif mode == "reacquire" and math.atan2(ap.p[2] - aim3[2], dist) >= depress:
                mode, t_mode = "dive", ap.t
            elif mode == "dive" and ap.agl() <= dive_break_agl:
                mode, t_mode = "dive_break", ap.t
            elif mode == "dive_break" and ap.t - t_mode >= dive_break_len:
                mode, t_mode = "terminal", ap.t
            if mode == "cruise":
                a = ap.track(_heading_to(ap.p, aim), 0.0, speed)
            elif mode == "break":
                n1, _ = ap.normal_basis()
                a = break_sign * 40.0 * g * n1 - (ap.v[2] / 0.8) * np.array([0.0, 0.0, 1.0])
            elif mode == "reacquire":
                a = ap.track(_heading_to(ap.p, aim), 0.0, speed + boost, tau=0.6)
            elif mode == "dive_break":
                n1, _ = ap.normal_basis()
                a = ap.pursue(aim3, speed + boost) - break_sign * 25.0 * g * n1
            else:
                a = ap.pursue(aim3, speed + boost)
        else:  # random_jink_dive
            if mode == "cruise" and math.atan2(ap.p[2] - aim3[2], dist) >= depress:
                mode, t_mode = "dive", ap.t
            if mode == "cruise":
                a = ap.track(_heading_to(ap.p, aim), 0.0, speed)
            else:
                if ap.t >= jink_until:
                    n1, n2 = ap.normal_basis()
                    phi = float(rng.uniform(-math.pi, math.pi))
                    jink = float(rng.uniform(8.0, 20.0)) * g * (math.cos(phi) * n1 + math.sin(phi) * n2)
                    jink_until = ap.t + float(rng.uniform(0.75, 1.5))
                a = ap.pursue(aim3, speed + 100.0, tau=1.2) + jink
        if not phases or phases[-1] != mode:
            phases.append(mode)
        ap.step(np.asarray(a, float))
    else:
        raise ValueError(f"{profile}: terminal altitude not reached in {max_t:.0f} s")
    if ap.t < 8.0:
        raise ValueError(f"{profile}: episode too short")
    names = {"cruise": "terrain_follow" if low else "cruise", "pop_up": "pop_up", "dive": "dive",
             "terminal": "terminal_dive", "turn_in": "dogleg_turn", "serpentine": "serpentine",
             "break": "break_turn", "reacquire": "boost_reacquire", "dive_break": "dive_break"}
    if profile == "barrel_roll_dive":
        names["dive"] = "barrel_roll_dive"
    if profile == "random_jink_dive":
        names["dive"] = "random_jinks"
    start_v = speed * _dir(psi0, 0.0)
    return ([round(float(start_xy[0]), 2), round(float(start_xy[1]), 2), round(z0, 2)],
            [round(float(c), 4) for c in start_v], ap.segs, [names[m] for m in phases], start_agl,
            math.degrees(psi0))


def plan_evasive_target(profile, rng, terrain, point, dt, sector=None):
    """Missile flying an evasive profile onto the defended ``point``; returns (target, info) or raises."""
    spec = variant_spec(CLASS, VARIANT)
    rejected = {}
    for _ in range(200):
        aim = np.asarray(point, float) + rng.uniform(-300.0, 300.0, 2)
        try:
            position, velocity, segs, phases, start_agl, approach = fly_evasive(profile, rng, terrain, aim, dt, spec, sector=sector)
        except ValueError as exc:
            reason = str(exc).split(":")[-1].strip()
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        duration = round(sum(s["duration_s"] for s in segs), 6)
        P, V = propagate(position, velocity, {"segments": segs}, duration)
        reason = check_trajectory(spec, P, V, terrain)
        if reason is None and profile not in LOW_LEVEL_PROFILES and start_agl < 2000.0:
            reason = "start_altitude"
        if reason is None:
            target = {"target_class": CLASS, "variant": VARIANT, "label": TARGET_CLASSES[CLASS]["label"],
                      "position": position, "velocity": velocity, "motion": {"segments": segs},
                      "rcs_dbsm": {a: round(float(rng.uniform(*spec.rcs_dbsm[a])), 3) for a in RCS_ASPECTS}}
            return target, {"profile": profile, "maneuvers": phases, "trajectory": P, "duration_s": duration,
                            "start_agl_m": round(start_agl, 1), "approach_deg": round(approach, 1), "aim": aim}
        rejected[reason] = rejected.get(reason, 0) + 1
    raise ValueError(f"{profile}: no valid trajectory ({rejected})")


# ---------------------------------------------------------------------------
# radars: high ground around the defended point, facing the threat axis
# ---------------------------------------------------------------------------

def _candidate_sites(terrain, point, cfg):
    lo, hi = cfg["distance_from_point_m"]
    g = np.arange(-hi, hi + 1, 250.0)
    X, Y = np.meshgrid(point[0] + g, point[1] + g)
    d = np.hypot(X - point[0], Y - point[1])
    keep = (d >= lo) & (d <= hi)
    xy = np.column_stack([X[keep], Y[keep]])
    h = terrain.height(xy[:, 0], xy[:, 1])
    high = h >= np.quantile(h, cfg["high_ground_quantile"])
    xy, h = xy[high], h[high]
    # local high point: above most of the ground within 1.5 km (hill tops and crests, not slopes)
    off = np.linspace(-1500.0, 1500.0, 9)
    OX, OY = np.meshgrid(off, off)
    local = np.array([np.mean(terrain.height(x + OX, y + OY) <= hh) for (x, y), hh in zip(xy, h)])
    return xy[local >= cfg["local_high_quantile"]]


def _bars(params, trajectory, beam_el_deg, max_bars, floor_deg=1.5):
    """Contiguous elevation bars (one beamwidth apart) over the threat's elevation band.

    When the band is wider than ``max_bars`` beams, the window holding most of
    the in-range trajectory samples is searched.
    """
    rel = trajectory - params.pos
    rng_ = np.linalg.norm(rel, axis=1)
    inside = (rng_ >= params.r_min_m) & (rng_ <= params.r_max_m)
    if not inside.any():
        return [3.0]
    el = np.degrees(np.arcsin(rel[inside, 2] / rng_[inside]))
    first = float(np.clip(el.min() + beam_el_deg / 2, floor_deg, params.el_max_deg - beam_el_deg / 2))
    last = float(np.clip(el.max() - beam_el_deg / 2, first, params.el_max_deg - beam_el_deg / 2))
    n = int(min(max_bars, math.ceil((last - first) / beam_el_deg - 1e-9) + 1))
    best, bars = -1, None
    for lo in np.arange(first, max(first, last - (n - 1) * beam_el_deg) + 0.5, 0.5):
        cand = lo + beam_el_deg * np.arange(n)
        hits = int(np.sum(np.min(np.abs(el[:, None] - cand[None, :]), axis=1) <= beam_el_deg / 2))
        if hits > best:
            best, bars = hits, cand
    return [round(float(x), 1) for x in bars]


def _visible_mask(trajectory, params, terrain, every):
    """Range / FOV / inside a search bar / terrain line of sight, per sample."""
    from src.core import geometry as geo
    from src.core.radar import visible
    samples = trajectory[::every]
    bars = np.deg2rad(np.asarray(params.search_bars_deg))
    half = np.deg2rad(0.5 * params.beamwidth_el_deg)
    out = np.zeros(len(samples), bool)
    for k, pt in enumerate(samples):
        meas = geo.measure(pt, np.zeros(3), params.pos, params.heading)
        out[k] = bool(visible(meas, params) and np.min(np.abs(bars - meas[2])) <= half
                      and terrain.line_of_sight(params.pos, pt))
    return out


def plan_dive_radars(n, rng, terrain, point, trajectory, cfg, coverage, dt, low_level=False):
    """Radars on local high ground facing the threat.

    ``low_level``: the threat hugs the terrain, so hill-top radars may look down
    (el_min -5 deg, bars from -2 deg) and a lower visible share is accepted;
    terrain masking is the point of those profiles.
    """
    sites = _candidate_sites(terrain, point, cfg)
    every = max(1, int(round(coverage["sample_step_s"] / CHECK_STEP_S)))
    best = -1.0
    for _ in range(coverage["max_radar_attempts"]):
        order = rng.permutation(len(sites))
        chosen = []
        for i in order:
            if all(np.hypot(*(sites[i] - c)) >= cfg["min_separation_m"] for c in chosen):
                chosen.append(sites[i])
            if len(chosen) == n:
                break
        if len(chosen) < n:
            continue
        specs, masks = [], []
        for k, xy in enumerate(chosen):
            # face the part of the threat path that is inside the radar's range
            d = np.linalg.norm(trajectory[:, :2] - xy, axis=1)
            near = trajectory[d <= cfg["defaults"]["r_max_m"] * 0.8]
            look = (near if len(near) else trajectory)[:, :2].mean(axis=0)
            heading = math.degrees(math.atan2(look[1] - xy[1], look[0] - xy[0]))
            heading += float(rng.uniform(-cfg["heading_jitter_deg"], cfg["heading_jitter_deg"]))
            spec = {"name": f"R{k + 1}",
                    "position": [round(float(xy[0]), 1), round(float(xy[1]), 1),
                                 round(float(rng.uniform(*cfg["mast_m"])), 1)],
                    "heading_deg": round((heading + 180) % 360 - 180, 2)}
            if low_level:
                spec["el_min_deg"] = -5.0
            params = _effective_radar(spec, cfg["defaults"], terrain)
            spec["search_bars_deg"] = _bars(params, trajectory, params.beamwidth_el_deg, cfg["max_bars"],
                                            floor_deg=-2.0 if low_level else 1.5)
            params = _effective_radar(spec, cfg["defaults"], terrain)
            specs.append(spec)
            masks.append(_visible_mask(trajectory, params, terrain, every))
        seen = np.any(masks, axis=0)
        fraction = float(seen.mean())
        # every radar must contribute, and the network must hold the threat long enough
        need, each = ((coverage["min_visible_fraction_low_level"], 0.1) if low_level
                      else (coverage["min_visible_fraction"], 0.2))
        if fraction >= need and all(m.mean() >= each for m in masks):
            return specs, fraction, [round(float(m.mean()), 3) for m in masks]
        best = max(best, fraction)
    raise ValueError(f"no {n}-radar layout covers the dive (best {best:.0%})")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def plan_missile_dive_episodes(recipe=None):
    r = resolve_recipe(recipe)
    master, dt = int(r["master_seed"]), float(r["dt"])
    family = r["dataset_name"]
    ratios = [Fraction(str(r["split_ratios"][s])) for s in SPLITS]
    split_counts = [0, 0, 0]
    episodes, owners = [], {}
    span = list(range(r["radar_count"][0], r["radar_count"][1] + 1))
    for index in range(r["episodes"]):
        which = max(range(3), key=lambda j: (index + 1) * ratios[j] - split_counts[j])
        split = SPLITS[which]
        split_counts[which] += 1
        rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(6, index)))
        episode_seed = int(rng.integers(0, 2 ** 63, dtype=np.int64))
        profile = r["profiles"][index % len(r["profiles"])]
        cycle = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(7, index // len(span))))
        n_radars = int(cycle.permutation(span)[index % len(span)])
        errors = []
        for _ in range(r["coverage"]["max_episode_attempts"]):
            seed = int(rng.integers(0, 2 ** 31))
            terrain_spec = {"source": "procedural", "resolution": r["terrain"]["resolution"],
                            "hills": rolling_hills(rng, r["terrain"])}
            terrain = _cached_terrain({"site": r["site"], "terrain": terrain_spec, "seed": seed})
            ang, rad = rng.uniform(-math.pi, math.pi), r["defended_point_radius_m"] * math.sqrt(rng.uniform())
            point = np.array([rad * math.cos(ang), rad * math.sin(ang)])
            try:
                if profile in EVASIVE_PROFILES:
                    target, info = plan_evasive_target(profile, rng, terrain, point, dt)
                else:
                    target, info = plan_dive_target(profile, rng, terrain, point, dt)
                radars, fraction, per_radar = plan_dive_radars(n_radars, rng, terrain, point, info["trajectory"],
                                                               r["radar"], r["coverage"], dt,
                                                               low_level=profile in LOW_LEVEL_PROFILES)
                break
            except ValueError as exc:
                errors.append(str(exc))
        else:
            raise ValueError(f"episode_index={index} ({profile}): planning failed: {errors[-3:]}")

        sc = validate_scenario({
            "name": f"{family}_ep{index:04d}_{profile}",
            "description": f"missile/terminal_dive {profile}, {n_radars} radar(s) on high ground "
                           f"around the defended point",
            "site": r["site"], "dt": dt, "duration_s": info["duration_s"], "area_radius_m": r["area_radius_m"],
            "seed": seed, "max_track_slots": r["max_track_slots"], "terrain": terrain_spec,
            "radar_defaults": r["radar"]["defaults"], "radars": radars, "targets": [target],
            "tracker": r["tracker"],
        })
        group = terrain_group_id(sc["terrain"])
        route = "route-" + _digest(_physical_numbers([{k: target[k] for k in ("position", "velocity")}]))[:20]
        for key in (group, route):
            if owners.setdefault(key, split) != split:
                raise ValueError(f"episode_index={index}: {key} crosses splits")
        snapshot = scenario_snapshot(sc, episode_seed=episode_seed)
        config_hash = _digest(snapshot)
        episodes.append({
            "dataset_episode_id": f"ep-{index:06d}-" + _digest([master, index, split, family, config_hash])[:12],
            "episode_index": index, "split": split, "scenario_family": family,
            "terrain_group_id": group, "route_group_id": route,
            "geometry_group_id": "geometry-" + _digest([group, route, sc["radars"]])[:20],
            "master_seed": master, "episode_seed": episode_seed, "terrain_seed": seed,
            "terrain_family": "rolling_hills", "texture_family": None, "obstacle_profile": "open",
            "dive_profile": profile, "approach_deg": info["approach_deg"], "start_agl_m": info["start_agl_m"],
            "defended_point": [round(float(x), 1) for x in point],
            "radar_count": n_radars, "target_count": 1,
            "target_classes": [CLASS], "target_variants": [VARIANT], "class_labels": [target["label"]],
            "maneuvers": [info["maneuvers"]],
            "planned_visible_fraction": [round(fraction, 3)], "planned_visible_fraction_per_radar": per_radar,
            "resolved_scenario": sc, "config_snapshot": snapshot,
            "resolved_configuration_hash": config_hash, "recipe_version": RECIPE_VERSION,
            "generation_status": "planned",
        })
    return _json_value({
        "plan_schema_version": 1, "recipe_version": RECIPE_VERSION, "master_seed": master, "recipe": r,
        "split_ratios": dict(zip(SPLITS, map(float, ratios))), "split_counts": dict(zip(SPLITS, split_counts)),
        "split_allocation": "prefix-stable largest-deficit allocation",
        "label_space": {c: TARGET_CLASSES[c]["label"] for c in TARGET_CLASSES},
        "geometry_validation": "dive propagated against terrain and the terminal_dive envelope; radar "
                               "layout accepted only above the minimum visible fraction",
        "grouping_policy": "one terrain and route per episode; groups disjoint across splits",
        "fixed_dimensions": "dt and max_track_slots fixed; radar count and duration vary per episode",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    })
