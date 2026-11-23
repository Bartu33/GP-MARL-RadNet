"""UAV / aircraft / missile dataset: 1-3 radars, 1-3 mixed-class targets, valley terrain.

Planning (:func:`plan_class_episodes`) only builds configurations; it never
steps the environment.  The plan uses the planner schema (plan_schema_version 1)
with ``recipe_version`` 2, so :func:`generate_dataset` writes, verifies and
resumes it exactly like the other datasets (manifest, ``.pending`` directories,
split-disjoint terrain and route groups).  Recipe version 2 differs from 1 in
one respect: the radar count may vary per episode (1-3 agents).

Target envelopes come from ``src/core/target_classes.py``.  Every trajectory
is a deterministic CV / CT / CA segment schedule built inside its variant's
limits (turn rate <= min(omega_max, a_lat_max / v), |a_long| <= a_long_max,
|v_z| <= v_z_max, speed inside the hard band) and then propagated against the
terrain at planning time; a candidate is rejected if it would leave its AGL
band, so the simulator's 30 m ground clamp never has to act.  Radar sites are
accepted only if every target is inside some radar's range / field of view /
search elevation with terrain line of sight for a minimum share of the episode.

:func:`audit_class_dataset` reads the written episodes back and compares the
realised kinematics, RCS, SNR and Doppler with the variant bands.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
import math
from pathlib import Path

import numpy as np

from src.core import geometry as geo
from src.core.env import _cached_terrain
from src.core.radar import visible
from src.core.rcs import incidence_in_body, m2_to_dbsm
from src.core.scenario import build_radars, scenario_snapshot, validate_scenario
from src.core.target_classes import CLASS_ORDER, TARGET_CLASSES, rcs_model, sample_instance, variant_spec
from src.core.targets import CA, CT, CV, MIN_AGL_M, SegmentedMotion, TargetState
from src.services.scenario_planner import SPLITS, _digest, _json_value, _physical_numbers, terrain_group_id

RECIPE_VERSION = 2
CHECK_STEP_S = 0.1                 # propagation step of the planning-time trajectory check
AGL_MARGIN_M = 2.0                 # keep clear of the simulator's 30 m clamp
GRID_MARGIN_M = 500.0

# Share of segments per manoeuvre kind and variant.  "hover" (multirotor only)
# brakes to a standstill at a_long_max and holds.
MANEUVER_WEIGHTS = {
    "multirotor": {"CV": 0.30, "CT": 0.20, "CA": 0.30, "hover": 0.20},
    "fixed_wing": {"CV": 0.40, "CT": 0.45, "CA": 0.15},
    "transport": {"CV": 0.60, "CT": 0.30, "CA": 0.10},
    "fighter": {"CV": 0.30, "CT": 0.50, "CA": 0.20},
    "subsonic_cruise": {"CV": 0.50, "CT": 0.40, "CA": 0.10},
    "supersonic": {"CV": 0.40, "CT": 0.40, "CA": 0.20},
}
# Variants below this initial AGL fly terrain-relative: their constant climb
# rate follows the ground trend along the planned leg (within the climb limit).
TERRAIN_RELATIVE_BELOW_AGL_M = 500.0

# Rugged terrain pool: high relief, narrow valleys and steep flanks, so that
# ridges block line of sight and low-level routes have to follow the valleys.
# Ranges are sampled per episode; "parameters" are the generator knobs of
# src/core/procedural_terrain.py (orientation, ridge width, asymmetry, ...).
RUGGED_PROFILES = {
    "deep_canyon": {"family": "valley_linear_canyon", "valley_width_m": (2600.0, 4000.0),
                    "relief_height_m": (950.0, 1450.0), "ridge_width_m": (900.0, 1500.0),
                    "obstacle_profiles": ["sparse", "dense", "asymmetric"]},
    "twisting_gorge": {"family": "valley_sinuous_pass", "valley_width_m": (3000.0, 4400.0),
                       "relief_height_m": (900.0, 1400.0), "ridge_width_m": (900.0, 1500.0),
                       "obstacle_profiles": ["dense", "asymmetric"]},
    "alpine_chokepoint": {"family": "mountain_chokepoint", "valley_width_m": (2800.0, 4200.0),
                          "relief_height_m": (900.0, 1400.0), "ridge_width_m": (900.0, 1600.0),
                          "obstacle_profiles": ["sparse", "dense", "asymmetric"]},
    "knife_ridges": {"family": "ridge_multi_obstacle", "valley_width_m": (2800.0, 4200.0),
                     "relief_height_m": (950.0, 1450.0), "ridge_width_m": (900.0, 1500.0),
                     "obstacle_profiles": ["dense", "asymmetric"]},
    "highland_bowl": {"family": "basin_highlands", "valley_width_m": (3000.0, 4400.0),
                      "relief_height_m": (850.0, 1300.0), "ridge_width_m": (1000.0, 1700.0),
                      "obstacle_profiles": ["sparse", "dense"]},
}
ROUTE_PATTERNS = ("converging", "diverging", "hub")
ZONE_NAMES = ("Zone A", "Zone B", "Zone C", "Zone D")

DEFAULT_MISSILE_FOCUS = {
    "missile_count": [2, 3],
    "route_patterns": list(ROUTE_PATTERNS),
    "variant_weights": {"subsonic_cruise": 0.5, "supersonic": 0.5},
    "anchor_radius_m": 2000.0,        # reference point drawn within this distance of the centre
    "anchor_tolerance_m": 700.0,      # a route must pass this close to the reference point
    "min_heading_separation_deg": 40.0,
    "relaxed_heading_separation_deg": 8.0,   # after a third of the attempts: in-trail salvos in narrow valleys
    "inbound_phase": [0.50, 0.85],    # share of the episode flown before reaching the reference point
    "outbound_phase": [0.05, 0.20],
    "downrange_zone_m": 5500.0,       # outbound routes get a zone where they leave this circle
    "facility_count": [0, 2],
    "zone_radius_m": [350.0, 600.0],
    "zone_height_m": [60.0, 220.0],
    "episode_attempts": 10,           # anchor / route redraws per terrain
    "max_route_attempts": 150,        # trajectory draws per missile and variant before the next variant
    "facility_radius_m": [150.0, 300.0],
    "facility_height_m": [40.0, 120.0],
}

DEFAULT_RECIPE = {
    "recipe_version": RECIPE_VERSION,
    "dataset_name": "class_targets_v1",
    "master_seed": 20261123,
    "episodes": 5,
    "duration_s": 30.0,
    "dt": 0.5,
    "max_track_slots": 4,
    "area_radius_m": 7500.0,
    "site": {"name": "Valley Test Range", "lat": 40.30, "lon": -111.70},
    "radar_count": [1, 3],
    "target_count": [1, 3],
    "class_weights": {"uav": 1.0, "aircraft": 1.0, "missile": 1.0},
    "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "terrain": {
        "resolution": 256,
        "families": ["valley_linear_canyon", "valley_sinuous_pass", "mountain_chokepoint",
                     "basin_highlands", "ridge_multi_obstacle"],
        "textures": ["rocky_mountain", "green_forest_valley", "dry_canyon", "snowy_highlands",
                     "grassland_valley", "mixed_rock_vegetation"],
        "obstacle_profiles": ["clear", "sparse", "dense", "asymmetric"],
        "valley_width_m": [2200.0, 3800.0],
        "relief_height_m": [650.0, 1050.0],
        "terrain_rerolls": 4,       # missile_focus only: next profile's terrain if no route fits
        # names from RUGGED_PROFILES: when set, each episode takes the next profile of a
        # shuffled cycle (family, width, relief, ridge shape, orientation) instead of the
        # families / valley_width_m / relief_height_m above.  None = legacy sampling.
        "rugged_profiles": None,
    },
    "radar": {
        "distance_from_centre_m": [1500.0, 6000.0],
        "min_separation_m": 2500.0,
        "mast_m": [10.0, 25.0],
        "heading_jitter_deg": 20.0,
        "defaults": {"fov_az_deg": 60.0, "search_sector_deg": 120.0, "r_max_m": 14000.0,
                     "search_bars_deg": [2.0, 7.0, 15.0, 25.0]},
    },
    "trajectory": {"cpa_radius_m": 3500.0, "segments": [1, 3], "max_attempts": 600},
    # Missile-focused episodes (None = the mixed-class behaviour above, unchanged).
    # missile_count missiles per episode share a reference point ("Zone A"): the
    # routes converge on it, diverge from it, or both ("hub").  Remaining targets
    # (target_count - missiles) are UAV / aircraft.  Zones are visual markers only.
    "missile_focus": None,
    "coverage": {"min_visible_fraction": 0.3, "sample_step_s": 1.0, "max_radar_attempts": 300,
                 "max_episode_attempts": 30},
}


# ---------------------------------------------------------------------------
# recipe
# ---------------------------------------------------------------------------

def _merge(base, over):
    out = deepcopy(base)
    for key, value in (over or {}).items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) \
            else deepcopy(value)
    return out


def resolve_recipe(recipe: dict | None = None) -> dict:
    """Defaults + overrides, with the checks the planner relies on."""
    r = _merge(DEFAULT_RECIPE, recipe or {})
    if r["recipe_version"] != RECIPE_VERSION:
        raise ValueError(f"recipe_version: expected {RECIPE_VERSION}")
    unknown = set(r) - set(DEFAULT_RECIPE)
    if unknown:
        raise ValueError(f"recipe: unsupported fields {sorted(unknown)}")
    for key in ("radar_count", "target_count"):
        lo, hi = r[key]
        if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= 3):
            raise ValueError(f"{key}: expected [lo, hi] with 1 <= lo <= hi <= 3")
    if set(r["class_weights"]) - set(TARGET_CLASSES) or not any(w > 0 for w in r["class_weights"].values()):
        raise ValueError(f"class_weights: positive weights over {list(TARGET_CLASSES)}")
    if not (isinstance(r["episodes"], int) and r["episodes"] >= 1):
        raise ValueError("episodes: expected a positive integer")
    ratios = [Fraction(str(r["split_ratios"][s])) for s in SPLITS]
    if sum(ratios) != 1 or any(x <= 0 for x in ratios):
        raise ValueError("split_ratios: positive and summing to 1")
    steps = r["duration_s"] / r["dt"]
    if abs(steps - round(steps)) > 1e-9:
        raise ValueError("duration_s must be a multiple of dt")
    rugged = r["terrain"].get("rugged_profiles")
    if rugged is not None and (not rugged or set(rugged) - set(RUGGED_PROFILES)):
        raise ValueError(f"terrain.rugged_profiles: non-empty subset of {list(RUGGED_PROFILES)}")
    if r["missile_focus"] is not None:
        r["missile_focus"] = _resolve_focus(r["missile_focus"], r)
    return r


def _resolve_focus(over, r):
    focus = _merge(DEFAULT_MISSILE_FOCUS, over)
    unknown = set(focus) - set(DEFAULT_MISSILE_FOCUS)
    if unknown:
        raise ValueError(f"missile_focus: unsupported fields {sorted(unknown)}")
    lo, hi = focus["missile_count"]
    if not (isinstance(lo, int) and isinstance(hi, int) and 1 <= lo <= hi <= len(ZONE_NAMES) - 1):
        raise ValueError(f"missile_focus.missile_count: [lo, hi] with 1 <= lo <= hi <= {len(ZONE_NAMES) - 1}")
    if lo > r["target_count"][1] or hi > 3:
        raise ValueError("missile_focus.missile_count: cannot exceed target_count (at most 3 targets)")
    if not focus["route_patterns"] or set(focus["route_patterns"]) - set(ROUTE_PATTERNS):
        raise ValueError(f"missile_focus.route_patterns: non-empty subset of {list(ROUTE_PATTERNS)}")
    allowed = {"subsonic_cruise", "supersonic"}
    weights = focus["variant_weights"]
    if set(weights) - allowed or not any(w > 0 for w in weights.values()):
        raise ValueError(f"missile_focus.variant_weights: positive weights over {sorted(allowed)}")
    for key in ("inbound_phase", "outbound_phase"):
        a, b = focus[key]
        if not 0.0 < a <= b < 1.0:
            raise ValueError(f"missile_focus.{key}: 0 < lo <= hi < 1")
    for key in ("anchor_radius_m", "anchor_tolerance_m", "min_heading_separation_deg", "downrange_zone_m"):
        if not focus[key] > 0:
            raise ValueError(f"missile_focus.{key}: expected a positive number")
    return focus


# ---------------------------------------------------------------------------
# trajectories
# ---------------------------------------------------------------------------

class _Point:
    """Minimal target state for propagating the motion models."""

    def __init__(self, p, v):
        self.p = np.asarray(p, float)
        self.v = np.asarray(v, float)
        self.a_lat = 0.0


def _turn_cap(spec, speed):
    cap = spec.max_turn_rate_rad_s
    if speed > 1.0:
        cap = min(cap, spec.max_lateral_accel_mps2 / speed)
    return cap


def _durations(rng, total, n, dt):
    """Split ``total`` into ``n`` segments on the dt grid, each >= 4 dt."""
    slots = int(round(total / dt))
    if n == 1 or slots < 8 * n:
        return [total]
    cuts = sorted(rng.choice(np.arange(4, slots - 3), size=n - 1, replace=False))
    edges = [0, *cuts, slots]
    return [float((b - a) * dt) for a, b in zip(edges, edges[1:])]


def _build_segments(spec, variant, speed_lo, speed_hi, v0, rng, durations):
    """Segment schedule within the variant's limits; returns (segments, kinds)."""
    weights = MANEUVER_WEIGHTS[variant]
    kinds_all, w = list(weights), np.array(list(weights.values()))
    state = _Point(np.zeros(3), v0)
    segments, kinds = [], []
    for duration in durations:
        kind = kinds_all[int(rng.choice(len(kinds_all), p=w / w.sum()))]
        vh = state.v[:2]
        speed_h = float(np.hypot(*vh))
        if kind == "CT" and speed_h < 1.0:
            kind = "CA"            # a turn needs forward speed
        if kind == "CV":
            pieces = [{"model": "CV", "duration_s": duration}]
        elif kind == "CT":
            omega = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 1.0) * _turn_cap(spec, np.linalg.norm(state.v)))
            pieces = [{"model": "CT", "turn_rate_rad_s": round(omega, 5), "duration_s": duration}]
        elif kind == "CA":
            if speed_h >= 1.0:
                direction = vh / speed_h
            else:
                psi = rng.uniform(-np.pi, np.pi)
                direction = np.array([np.cos(psi), np.sin(psi)])
            speed = float(np.linalg.norm(state.v))
            accel = float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * spec.max_long_accel_mps2)
            # keep the total speed inside the hard band at the end of the segment
            vz = state.v[2]
            target_speed = float(np.clip(speed + accel * duration, speed_lo, speed_hi))
            target_h = math.sqrt(max(target_speed ** 2 - vz ** 2, 0.0))
            if speed_h < 1.0:
                accel_h = target_h / duration
            else:
                accel_h = (target_h - speed_h) / duration
            a = [round(float(accel_h * direction[0]), 5), round(float(accel_h * direction[1]), 5), 0.0]
            pieces = [{"model": "CA", "acceleration_mps2": a, "duration_s": duration}]
        else:  # hover: brake every velocity component at a_long_max, then hold
            speed = float(np.linalg.norm(state.v))
            t_brake = max(0.5, math.ceil(speed / spec.max_long_accel_mps2 / 0.5) * 0.5)
            if t_brake >= duration or speed < 0.05:
                pieces = [{"model": "CV", "duration_s": duration}]
            else:
                a = [round(float(-c / t_brake), 6) for c in state.v]
                pieces = [{"model": "CA", "acceleration_mps2": a, "duration_s": t_brake},
                          {"model": "CV", "duration_s": duration - t_brake}]
        for piece in pieces:
            model = _model(piece)
            model.step(state, piece["duration_s"])
            if piece["model"] == "CA" and kind == "hover":
                state.v = np.where(np.abs(state.v) < 1e-6, 0.0, state.v)
        segments.extend(pieces)
        kinds.append(kind)
    return segments, kinds


def _model(piece):
    rng = np.random.default_rng(0)            # CV/CA/CT draw no random numbers
    if piece["model"] == "CV":
        return CV(rng, 1.0)
    if piece["model"] == "CT":
        return CT(rng, 1.0, turn_rate_rad_s=piece["turn_rate_rad_s"])
    return CA(rng, 1.0, acceleration_mps2=piece["acceleration_mps2"])


def propagate(position, velocity, motion, duration, step=CHECK_STEP_S):
    """Positions/velocities of an explicit motion at ``step`` spacing (no terrain clamp)."""
    rng = np.random.default_rng(0)
    model = SegmentedMotion(rng, 1.0, motion["segments"]) if "segments" in motion else _model(motion)
    s = _Point(position, velocity)
    n = int(round(duration / step))
    P, V = [s.p.copy()], [s.v.copy()]
    for _ in range(n):
        model.step(s, step)
        P.append(s.p.copy())
        V.append(s.v.copy())
    return np.array(P), np.array(V)


def _inside_grid(terrain, xy, margin):
    if terrain.source == "flat":
        return True
    size = terrain.cell * terrain.H.shape[1]
    return bool(np.all((xy[:, 0] >= terrain.x_west + margin) & (xy[:, 0] <= terrain.x_west + size - margin)
                       & (xy[:, 1] <= terrain.y_north - margin) & (xy[:, 1] >= terrain.y_north - size + margin)))


def check_trajectory(spec, P, V, terrain):
    """Return None if the propagated path respects the variant envelope, else a reason."""
    speed = np.linalg.norm(V, axis=1)
    lo, hi = spec.speed_range_mps
    if speed.min() < lo - 1e-3 or speed.max() > hi + 1e-3:
        return "speed_band"
    if np.abs(V[:, 2]).max() > spec.max_climb_rate_mps + 1e-6:
        return "climb_rate"
    if not _inside_grid(terrain, P[:, :2], GRID_MARGIN_M):
        return "outside_grid"
    agl = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
    a_lo, a_hi = spec.altitude_agl_range_m
    if agl.min() < max(a_lo, MIN_AGL_M) + AGL_MARGIN_M:
        return "agl_low"
    if agl.max() > a_hi:
        return "agl_high"
    return None


def _leg(rng, terrain, centre, cpa_radius, length, low, candidates=12):
    """Closest-approach point, heading and fraction of the leg flown before it.

    Low-level flight follows low ground, as low-level routes do: the CPA is the
    lowest of a few random points, and the heading is drawn among the
    candidates whose straight leg has the flattest ground profile.
    """
    def point():
        ang, rad = rng.uniform(-np.pi, np.pi), cpa_radius * math.sqrt(rng.uniform())
        return np.asarray(centre, float) + rad * np.array([math.cos(ang), math.sin(ang)])

    f = float(rng.uniform(0.2, 0.8))
    if not low:
        return point(), float(rng.uniform(-np.pi, np.pi)), f
    pts = [point() for _ in range(4)]
    cpa = min(pts, key=lambda q: float(terrain.height(*q)))
    psis = rng.uniform(-np.pi, np.pi, candidates)
    s = np.linspace(-f, 1.0 - f, 25) * length
    relief = []
    for psi in psis:
        xy = cpa[None, :] + s[:, None] * np.array([math.cos(psi), math.sin(psi)])[None, :]
        h = terrain.height(xy[:, 0], xy[:, 1])
        relief.append(float(h.max() - h.min()))
    best = np.argsort(relief)[:3]
    return cpa, float(psis[best[int(rng.integers(len(best)))]]), f


def _angle_gap(a, b):
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _route_leg(rng, terrain, route, length, low, relaxed=False, candidates=48):
    """Leg through the route's reference point, or None if no heading is separated enough.

    The closest point lies a fraction of ``route["tolerance_m"]`` from the reference
    point; ``f`` is drawn from the route's phase (late = inbound, early = outbound).
    Headings closer than the minimum separation to an already planned route are dropped;
    low-level flyers then take the flattest of the remaining ones.
    """
    anchor = np.asarray(route["anchor"], float)
    ang, rad = rng.uniform(-np.pi, np.pi), 0.4 * route["tolerance_m"] * math.sqrt(rng.uniform())
    cpa = anchor + rad * np.array([math.cos(ang), math.sin(ang)])
    f = float(rng.uniform(*route["phase"]))
    sep = math.radians(route["relaxed_separation_deg"] if relaxed else route["min_separation_deg"])
    psis = [p for p in rng.uniform(-np.pi, np.pi, candidates)
            if all(_angle_gap(p, q) >= sep for q in route["headings"])]
    if not psis:
        return None
    if low:
        # terrain-following flies a straight line between the end heights: rank headings by the
        # largest ground deviation from that chord and drop the hopeless ones without propagating
        s_ = np.linspace(-f, 1.0 - f, 25) * length
        dev = []
        for psi in psis:
            xy = cpa[None, :] + s_[:, None] * np.array([math.cos(psi), math.sin(psi)])[None, :]
            h = terrain.height(xy[:, 0], xy[:, 1])
            chord = h[0] + (h[-1] - h[0]) * np.linspace(0.0, 1.0, len(h))
            dev.append(float(np.abs(h - chord).max()))
        keep = [i for i in np.argsort(dev) if dev[i] <= route.get("max_chord_dev_m", 200.0)][:3]
        if not keep:
            return None
        psi = psis[int(keep[int(rng.integers(len(keep)))])]
    else:
        psi = psis[int(rng.integers(len(psis)))]
    return cpa, float(psi), f


def plan_target(class_name, rng, terrain, duration, dt, centre, cfg, variant=None, route=None):
    """Draw one target of ``class_name`` and a valid trajectory through the area.

    The leg is placed so that its closest point to the area lies within
    ``cpa_radius_m`` of ``centre``; ``f`` of the leg is flown before that point.
    With ``route`` (anchor, tolerance_m, phase, headings, min_separation_deg) the leg
    instead passes through the route's reference point on a heading separated from
    the other routes, and the realised path must stay within the tolerance of it.
    """
    if variant is None:
        # variants with a leg model here; missile/terminal_dive episodes come from missile_dive_service
        names = [v for v in TARGET_CLASSES[class_name]["variants"] if v in MANEUVER_WEIGHTS]
        w = np.array([TARGET_CLASSES[class_name]["variants"][v].weight for v in names])
        variant = names[int(rng.choice(len(names), p=w / w.sum()))]
    instance = sample_instance(class_name, rng, variant=variant)
    spec = variant_spec(class_name, instance["variant"])
    rejected = {}
    for attempt in range(cfg["max_attempts"]):
        if attempt and attempt % 50 == 0:      # a fresh instance now and then (speed / altitude)
            instance = sample_instance(class_name, rng, variant=instance["variant"])
        speed = instance["speed_mps"]
        agl0 = instance["altitude_agl_m"] if attempt == 0 else float(
            rng.uniform(*(spec.altitude_agl_typical_m if rng.random() < 0.85 else spec.altitude_agl_range_m)))
        low = agl0 < TERRAIN_RELATIVE_BELOW_AGL_M
        if route is None:
            cpa, psi, f = _leg(rng, terrain, centre, cfg["cpa_radius_m"], speed * duration, low)
        else:
            leg = _route_leg(rng, terrain, route, speed * duration, low, relaxed=attempt >= cfg["max_attempts"] // 3)
            if leg is None:
                rejected["no_feasible_heading"] = rejected.get("no_feasible_heading", 0) + 1
                continue
            cpa, psi, f = leg
        u = np.array([math.cos(psi), math.sin(psi)])
        start_xy = cpa - u * speed * duration * f
        end_xy = start_xy + u * speed * duration
        ground0 = float(terrain.height(*start_xy))
        vz_cap = spec.max_climb_rate_mps if spec.can_hover else min(spec.max_climb_rate_mps, 0.5 * speed)
        if low:
            trend = (float(terrain.height(*end_xy)) - ground0) / duration
            vz = trend + 0.3 * instance["climb_rate_mps"]
        else:
            vz = instance["climb_rate_mps"]
        vz = float(np.clip(vz, -vz_cap, vz_cap))
        vh = math.sqrt(max(speed ** 2 - vz ** 2, 0.0))
        p0 = [float(start_xy[0]), float(start_xy[1]), ground0 + agl0]
        v0 = [vh * u[0], vh * u[1], vz]
        n_seg = int(rng.integers(cfg["segments"][0], cfg["segments"][1] + 1))
        segments, kinds = _build_segments(spec, instance["variant"], *spec.speed_range_mps,
                                          np.array(v0), rng, _durations(rng, duration, n_seg, dt))
        motion = {"segments": segments}
        P, V = propagate(p0, v0, motion, duration)
        reason = check_trajectory(spec, P, V, terrain)
        if reason is None:
            target = {
                "target_class": class_name, "variant": instance["variant"], "label": instance["label"],
                "position": [round(x, 3) for x in p0], "velocity": [round(x, 4) for x in v0],
                "motion": motion, "rcs_dbsm": {k: round(v, 3) for k, v in instance["rcs_dbsm"].items()},
            }
            # rounding must not break the checked trajectory
            P, V = propagate(target["position"], target["velocity"], motion, duration)
            if check_trajectory(spec, P, V, terrain) is None:
                miss = None if route is None else float(
                    np.min(np.linalg.norm(P[:, :2] - np.asarray(route["anchor"]), axis=1)))
                if miss is not None and miss > route["tolerance_m"]:
                    reason = "anchor_miss"
                else:
                    return target, {"attempts": attempt + 1, "rejected": rejected, "maneuvers": kinds,
                                    "trajectory": P, "heading_rad": psi}
            else:
                reason = "rounding"
        rejected[reason] = rejected.get(reason, 0) + 1
    raise ValueError(f"{class_name}/{instance['variant']}: no valid trajectory in "
                     f"{cfg['max_attempts']} attempts ({rejected})")


# ---------------------------------------------------------------------------
# radars and coverage
# ---------------------------------------------------------------------------

def _effective_radar(spec, defaults, terrain):
    sc = validate_scenario({"radars": [spec], "radar_defaults": defaults})
    p = build_radars(sc)[0]
    x, y, mast = p.position
    p.position = (x, y, float(terrain.height(x, y)) + mast)
    return p


def visible_fraction(trajectory, radars, terrain, every):
    """Share of samples in which some radar has range / FOV / search elevation and LOS."""
    samples = trajectory[::every]
    seen = np.zeros(len(samples), bool)
    for p in radars:
        el_top = max(p.search_bars_deg) + 0.5 * p.beamwidth_el_deg
        for k, pt in enumerate(samples):
            if seen[k]:
                continue
            meas = geo.measure(pt, np.zeros(3), p.pos, p.heading)
            if (visible(meas, p) and meas[2] <= np.deg2rad(el_top)
                    and terrain.line_of_sight(p.pos, pt)):
                seen[k] = True
    return float(seen.mean())


def plan_radars(n, rng, terrain, centre, aim, trajectories, cfg, coverage, step_s):
    every = max(1, int(round(coverage["sample_step_s"] / step_s)))
    best = (-1.0, None)
    for _ in range(coverage["max_radar_attempts"]):
        specs = []
        for i in range(n):
            for _ in range(200):
                ang = rng.uniform(-np.pi, np.pi)
                dist = rng.uniform(*cfg["distance_from_centre_m"])
                xy = np.asarray(centre, float) + dist * np.array([math.cos(ang), math.sin(ang)])
                if all(np.hypot(*(xy - np.array(s["position"][:2]))) >= cfg["min_separation_m"] for s in specs):
                    break
            else:
                break
            heading = math.degrees(math.atan2(aim[1] - xy[1], aim[0] - xy[0]))
            heading += rng.uniform(-cfg["heading_jitter_deg"], cfg["heading_jitter_deg"])
            specs.append({"name": f"R{i + 1}",
                          "position": [round(float(xy[0]), 1), round(float(xy[1]), 1),
                                       round(float(rng.uniform(*cfg["mast_m"])), 1)],
                          "heading_deg": round(float((heading + 180) % 360 - 180), 2)})
        if len(specs) < n:
            continue
        params = [_effective_radar(s, cfg["defaults"], terrain) for s in specs]
        fractions = [visible_fraction(t, params, terrain, every) for t in trajectories]
        if min(fractions) >= coverage["min_visible_fraction"]:
            return specs, fractions
        if min(fractions) > best[0]:
            best = (min(fractions), fractions)
    raise ValueError(f"no {n}-radar layout covers every target >= {coverage['min_visible_fraction']:.0%} "
                     f"(best minimum {best[0]:.0%})")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _balanced_pick(counts, weights, rng):
    """Class whose count lags its weight share most (ties broken randomly)."""
    total = sum(counts.values()) + 1
    wsum = sum(weights.values())
    deficit = {c: weights[c] / wsum * total - counts[c] for c in weights if weights[c] > 0}
    top = max(deficit.values())
    options = sorted(c for c, d in deficit.items() if d >= top - 1e-9)
    return options[int(rng.integers(len(options)))]


def _cycle_counts(master, stream, index, lo, hi):
    """Counts cycle through lo..hi, shuffled per cycle, so small datasets cover every count."""
    span = list(range(lo, hi + 1))
    cycle_rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(stream, index // len(span))))
    return int(cycle_rng.permutation(span)[index % len(span)])


def _rugged_terrain_spec(r, rng, master, index, which, retry=0):
    """Terrain spec from the next rugged profile of a shuffled cycle (see RUGGED_PROFILES)."""
    t_cfg = r["terrain"]
    names = t_cfg["rugged_profiles"]
    prof_name = names[(_cycle_counts(master, 8, index, 0, len(names) - 1) + retry) % len(names)]   # a reroll moves on to the next profile
    prof = RUGGED_PROFILES[prof_name]
    obstacle = prof["obstacle_profiles"][int(rng.integers(len(prof["obstacle_profiles"])))]
    params = {"valley_angle_deg": round(float(rng.uniform(0.0, 180.0)), 1),
              "ridge_width_m": round(float(rng.uniform(*prof["ridge_width_m"])), 1)}
    if obstacle == "asymmetric":
        params["asymmetry_ratio"] = round(float(rng.uniform(1.3, 1.7)), 2)
    spec = {
        "source": "procedural", "resolution": t_cfg["resolution"], "family": prof["family"],
        "texture_family": t_cfg["textures"][int(rng.integers(len(t_cfg["textures"])))],
        "obstacle_profile": obstacle,
        "valley_width_m": round(float(rng.uniform(*prof["valley_width_m"])), 1),
        "relief_height_m": round(float(rng.uniform(*prof["relief_height_m"])), 1),
        "parameters": params,
        "seed": int(np.random.SeedSequence(
            master, spawn_key=(3, which, index) + ((retry,) if retry else ())).generate_state(1)[0]),
    }
    return spec, prof_name


def _focus_composition(r, rng, master, index, n_targets, class_counts):
    """Shuffled class list of a missile-focused episode (missiles drawn first, the rest UAV / aircraft)."""
    focus = r["missile_focus"]
    n_missiles = min(n_targets, _cycle_counts(master, 6, index, *focus["missile_count"]))
    others = {c: float(r["class_weights"].get(c, 0.0)) for c in CLASS_ORDER if c != "missile"}
    classes = ["missile"] * n_missiles
    class_counts["missile"] += n_missiles
    for _ in range(n_targets - n_missiles):
        if not any(w > 0 for w in others.values()):
            raise ValueError("missile_focus: target_count exceeds missile_count and no UAV / aircraft weight is positive")
        c = _balanced_pick(class_counts, others, rng)
        class_counts[c] += 1
        classes.append(c)
    patterns = focus["route_patterns"]
    cyc = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(7, index // len(patterns))))
    pattern = patterns[int(cyc.permutation(len(patterns))[index % len(patterns)])]
    return [classes[i] for i in rng.permutation(len(classes))], pattern


def _route_phases(pattern, n_missiles, rng):
    """'in' (reaches the reference point late) or 'out' (leaves it early) per missile."""
    if pattern == "converging":
        return ["in"] * n_missiles
    if pattern == "diverging":
        return ["out"] * n_missiles
    if n_missiles == 1:
        return ["in"]
    n_in = int(rng.integers(1, n_missiles))                 # hub: at least one inbound and one outbound
    return ["in"] * n_in + ["out"] * (n_missiles - n_in)


def _plan_focus_targets(r, rng, terrain, classes, pattern, duration, dt, centre):
    """Targets of a missile-focused episode with routes tied to one reference point.

    Returns (targets, info, route); route holds the anchor, per-missile phases and headings.
    Raises ValueError if a route cannot be placed.
    """
    focus = r["missile_focus"]
    cfg = {**r["trajectory"], "max_attempts": focus["max_route_attempts"]}
    # reference point: the lowest of a few draws, so that low-level routes can follow the ground
    draws = []
    for _ in range(8):
        ang, rad = rng.uniform(-np.pi, np.pi), focus["anchor_radius_m"] * math.sqrt(rng.uniform())
        draws.append(np.asarray(centre, float) + rad * np.array([math.cos(ang), math.sin(ang)]))
    anchor = min(draws, key=lambda q: float(terrain.height(*q)))
    phases = iter(_route_phases(pattern, classes.count("missile"), rng))
    names = [v for v in focus["variant_weights"] if focus["variant_weights"][v] > 0]
    w = np.array([focus["variant_weights"][v] for v in names], float)
    headings, route_phases, targets, info = [], [], [], []
    for c in classes:
        if c != "missile":
            t, meta = plan_target(c, rng, terrain, duration, dt, centre, r["trajectory"])
        else:
            phase = next(phases)
            route = {"anchor": anchor, "tolerance_m": focus["anchor_tolerance_m"],
                     "phase": focus["inbound_phase"] if phase == "in" else focus["outbound_phase"],
                     "headings": headings, "min_separation_deg": focus["min_heading_separation_deg"],
                     "relaxed_separation_deg": focus["relaxed_heading_separation_deg"]}
            first = int(rng.choice(len(names), p=w / w.sum()))
            order = [names[first]] + [v for v in names if v != names[first]]
            for k, variant in enumerate(order):      # a variant that cannot fly the route yields to another
                try:
                    t, meta = plan_target(c, rng, terrain, duration, dt, centre, cfg, variant=variant, route=route)
                    break
                except ValueError:
                    if k == len(order) - 1:
                        raise
            headings.append(meta["heading_rad"])
            route_phases.append(phase)
        targets.append(t)
        info.append(meta)
    return targets, info, {"anchor": anchor, "phases": route_phases, "headings": headings}


def _build_zones(r, rng, terrain, route, targets, info, radars):
    """Abstract reference markers: Zone A at the anchor, a zone downrange of every outbound route,
    and a few facilities clear of the anchor, the zones and the radars."""
    focus = r["missile_focus"]

    def draw(key):
        return round(float(rng.uniform(*focus[key])), 1)

    def xy_of(q):
        return [round(float(q[0]), 1), round(float(q[1]), 1)]

    anchor = np.asarray(route["anchor"], float)
    zones = [{"name": ZONE_NAMES[0], "kind": "zone", "position": xy_of(anchor),
              "radius_m": draw("zone_radius_m"), "height_m": draw("zone_height_m")}]
    taken = [anchor]
    missiles = [m for t, m in zip(targets, info) if t["target_class"] == "missile"]
    for phase, meta in zip(route["phases"], missiles):
        if phase != "out":
            continue
        P = meta["trajectory"][:, :2]
        far = np.nonzero(np.linalg.norm(P - anchor, axis=1) >= focus["downrange_zone_m"])[0]
        q = P[int(far[0])] if far.size else P[-1]
        zones.append({"name": ZONE_NAMES[len(zones)], "kind": "zone", "position": xy_of(q),
                      "radius_m": draw("zone_radius_m"), "height_m": draw("zone_height_m")})
        taken.append(q)
    sites = [np.asarray(p["position"][:2], float) for p in radars]
    n_fac = int(rng.integers(focus["facility_count"][0], focus["facility_count"][1] + 1))
    placed = 0
    for _ in range(n_fac * 40):
        if placed >= n_fac:
            break
        ang, rad = rng.uniform(-np.pi, np.pi), r["area_radius_m"] * 0.8 * math.sqrt(rng.uniform())
        q = np.array([rad * math.cos(ang), rad * math.sin(ang)])
        if (all(np.linalg.norm(q - o) >= 1500.0 for o in taken + sites)
                and _inside_grid(terrain, q[None, :], GRID_MARGIN_M)):
            placed += 1
            taken.append(q)
            zones.append({"name": f"Facility {placed}", "kind": "facility", "position": xy_of(q),
                          "radius_m": draw("facility_radius_m"), "height_m": draw("facility_height_m")})
    return zones


def plan_class_episodes(recipe: dict | None = None) -> dict:
    """Deterministic plan for the recipe; JSON-ready, consumed by generate_dataset."""
    r = resolve_recipe(recipe)
    master = int(r["master_seed"])
    ratios = [Fraction(str(r["split_ratios"][s])) for s in SPLITS]
    duration, dt = float(r["duration_s"]), float(r["dt"])
    family = r["dataset_name"]
    centre = (0.0, 0.0)
    split_counts, class_counts = [0, 0, 0], {c: 0 for c in CLASS_ORDER}
    weights = {c: float(r["class_weights"].get(c, 0.0)) for c in CLASS_ORDER}
    episodes, terrain_owner, route_owner = [], {}, {}
    for index in range(r["episodes"]):
        which = max(range(3), key=lambda j: (index + 1) * ratios[j] - split_counts[j])
        split = SPLITS[which]
        split_counts[which] += 1
        rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(2, index)))
        episode_seed = int(rng.integers(0, 2 ** 63, dtype=np.int64))
        n_radars = _cycle_counts(master, 4, index, *r["radar_count"])
        n_targets = _cycle_counts(master, 5, index, *r["target_count"])

        rerolls = r["terrain"]["terrain_rerolls"] if r["missile_focus"] else 1
        for retry in range(rerolls):
            t_cfg = r["terrain"]
            rugged_profile = None
            if t_cfg.get("rugged_profiles"):
                terrain_spec, rugged_profile = _rugged_terrain_spec(r, rng, master, index, which, retry)
            else:
                terrain_spec = {
                    "source": "procedural", "resolution": t_cfg["resolution"],
                    "family": t_cfg["families"][int(rng.integers(len(t_cfg["families"])))],
                    "texture_family": t_cfg["textures"][int(rng.integers(len(t_cfg["textures"])))],
                    "obstacle_profile": t_cfg["obstacle_profiles"][int(rng.integers(len(t_cfg["obstacle_profiles"])))],
                    "valley_width_m": round(float(rng.uniform(*t_cfg["valley_width_m"])), 1),
                    "relief_height_m": round(float(rng.uniform(*t_cfg["relief_height_m"])), 1),
                    # split in the seed keeps terrain groups split-owned by construction
                    "seed": int(np.random.SeedSequence(master, spawn_key=(3, which, index)).generate_state(1)[0]),
                }
            terrain = _cached_terrain({"site": r["site"], "terrain": terrain_spec, "seed": terrain_spec["seed"]})

            focus = r["missile_focus"]
            classes, pattern = [], None
            if focus:
                classes, pattern = _focus_composition(r, rng, master, index, n_targets, class_counts)
            else:
                for _ in range(n_targets):
                    c = _balanced_pick(class_counts, weights, rng)
                    class_counts[c] += 1
                    classes.append(c)

            errors = []
            for _ in range(r["missile_focus"]["episode_attempts"] if focus else r["coverage"]["max_episode_attempts"]):
                try:
                    targets, info, rinfo = [], [], None
                    if focus:
                        targets, info, rinfo = _plan_focus_targets(r, rng, terrain, classes, pattern,
                                                                   duration, dt, centre)
                        aim = rinfo["anchor"]
                    else:
                        for c in classes:
                            t, meta = plan_target(c, rng, terrain, duration, dt, centre, r["trajectory"])
                            targets.append(t)
                            info.append(meta)
                        aim = np.mean([m["trajectory"][len(m["trajectory"]) // 2, :2] for m in info], axis=0)
                    radars, fractions = plan_radars(n_radars, rng, terrain, centre, aim,
                                                    [m["trajectory"] for m in info], r["radar"],
                                                    r["coverage"], CHECK_STEP_S)
                    break
                except ValueError as exc:
                    errors.append(str(exc))
            else:
                if retry == rerolls - 1:
                    raise ValueError(f"episode_index={index}: planning failed: {errors[-3:]}")
                continue                    # no route fits this terrain: draw one of the next profile
            break

        zones = _build_zones(r, rng, terrain, rinfo, targets, info, radars) if focus else None
        scenario = {
            "name": f"{family}_ep{index:04d}",
            "description": f"{n_radars} radar(s), targets: "
                           + ", ".join(f"{t['target_class']}/{t['variant']}" for t in targets)
                           + (f"; {pattern} routes" if focus else ""),
            "site": r["site"], "dt": dt, "duration_s": duration, "area_radius_m": r["area_radius_m"],
            "seed": terrain_spec["seed"], "max_track_slots": r["max_track_slots"],
            "terrain": terrain_spec, "radar_defaults": r["radar"]["defaults"],
            "radars": radars, "targets": targets,
        }
        if zones:
            scenario["zones"] = zones
        sc = validate_scenario(scenario)
        group = terrain_group_id(sc["terrain"])
        if terrain_owner.setdefault(group, split) != split:
            raise ValueError(f"episode_index={index}: terrain group crosses splits")
        route = "route-" + _digest(_physical_numbers(
            [{k: t[k] for k in ("position", "velocity")} for t in sc["targets"]]))[:20]
        if route_owner.setdefault(route, split) != split:
            raise ValueError(f"episode_index={index}: route group crosses splits")
        snapshot = scenario_snapshot(sc, episode_seed=episode_seed)
        config_hash = _digest(snapshot)
        episodes.append({
            "dataset_episode_id": f"ep-{index:06d}-" + _digest([master, index, split, family, config_hash])[:12],
            "episode_index": index, "split": split, "scenario_family": family,
            "terrain_group_id": group, "route_group_id": route,
            "geometry_group_id": "geometry-" + _digest([group, route, sc["radars"]])[:20],
            "master_seed": master, "episode_seed": episode_seed, "terrain_seed": terrain_spec["seed"],
            "terrain_family": terrain_spec["family"], "texture_family": terrain_spec["texture_family"],
            "obstacle_profile": terrain_spec["obstacle_profile"],
            "radar_count": n_radars, "target_count": n_targets,
            "target_classes": [t["target_class"] for t in targets],
            "target_variants": [t["variant"] for t in targets],
            "class_labels": [t["label"] for t in targets],
            "maneuvers": [m["maneuvers"] for m in info],
            "planned_visible_fraction": [round(x, 3) for x in fractions],
            "trajectory_attempts": [m["attempts"] for m in info],
            "resolved_scenario": sc, "config_snapshot": snapshot,
            "resolved_configuration_hash": config_hash, "recipe_version": RECIPE_VERSION,
            "generation_status": "planned",
        })
        if rugged_profile:
            episodes[-1]["rugged_profile"] = rugged_profile
        if focus:
            episodes[-1].update({
                "route_pattern": pattern, "missile_count": classes.count("missile"),
                "route_phases": rinfo["phases"],
                "route_headings_deg": [round(math.degrees(h), 1) for h in rinfo["headings"]],
                "anchor": [round(float(x), 1) for x in rinfo["anchor"]], "zones": zones,
            })
    return _json_value({
        "plan_schema_version": 1, "recipe_version": RECIPE_VERSION, "master_seed": master,
        "recipe": r, "split_ratios": dict(zip(SPLITS, map(float, ratios))),
        "split_counts": dict(zip(SPLITS, split_counts)),
        "split_allocation": "prefix-stable largest-deficit allocation",
        "label_space": {c: TARGET_CLASSES[c]["label"] for c in CLASS_ORDER},
        "geometry_validation": "full trajectory propagated against terrain and variant envelope; "
                               "radar layout accepted only above the minimum visible fraction",
        "grouping_policy": "terrain seeds are split-owned; terrain and route groups disjoint across splits",
        "fixed_dimensions": "dt and max_track_slots fixed; radar count varies per episode",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    })


# ---------------------------------------------------------------------------
# audit of written episodes
# ---------------------------------------------------------------------------

def _stats(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return None
    return {"min": float(x.min()), "p50": float(np.median(x)), "max": float(x.max())}


def audit_episode(directory) -> dict:
    """Realised kinematics / RCS / SNR / Doppler per target vs. its variant bands."""
    from src.services.simulation_service import load_episode_records

    rec = load_episode_records(Path(directory))
    meta = rec["metadata"]
    sc = meta["config_snapshot"]["scenario"]
    terrain = _cached_terrain(sc)
    radars = build_radars(sc)
    for p in radars:
        x, y, mast = p.position
        p.position = (x, y, float(terrain.height(x, y)) + mast)
    dt = sc["dt"]
    truth = {}
    for row in rec["truth"]:
        truth.setdefault(row["target_id"], []).append(row)
    reports = {}
    for rep, ann in zip(rec["reports"], rec["annotations"]):
        if ann["truth_id"] >= 0:
            reports.setdefault(ann["truth_id"], []).append(rep)
    out = []
    for tid, spec_t in enumerate(sc["targets"]):
        spec = variant_spec(spec_t["target_class"], spec_t["variant"])
        rows = sorted(truth[tid], key=lambda r: r["step_index"])
        P = np.array([r["p"] for r in rows])
        V = np.array([r["v"] for r in rows])
        speed = np.linalg.norm(V, axis=1)
        agl = P[:, 2] - terrain.height(P[:, 0], P[:, 1])
        acc = np.diff(V, axis=0) / dt
        vh = V[:-1, :2]
        nh = np.linalg.norm(vh, axis=1)
        moving = nh > 1.0
        # normal (manoeuvre) acceleration in 3-D: turns, push-overs and pull-outs alike
        v_mid = 0.5 * (V[:-1] + V[1:])
        s_mid = np.linalg.norm(v_mid, axis=1)
        fast = s_mid > 1.0
        u_mid = v_mid[fast] / s_mid[fast, None]
        a_f = acc[fast]
        a_lat = np.linalg.norm(a_f - (a_f * u_mid).sum(1)[:, None] * u_mid, axis=1)
        heading = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
        # heading is undefined near standstill (multirotor leaving a hover): both ends must move
        # it is also undefined in a near-vertical dive (azimuth of a vertical velocity): the horizontal speed
        # must be a real share of the speed at both ends
        h0, h1 = nh, np.linalg.norm(V[1:, :2], axis=1)
        both_moving = (moving & (h1 > 1.0) & (h0 > 0.2 * np.linalg.norm(V[:-1], axis=1))
                       & (h1 > 0.2 * np.linalg.norm(V[1:], axis=1)))
        turn = np.abs(np.diff(heading) / dt)[both_moving]
        model = rcs_model({"rcs_dbsm": spec_t["rcs_dbsm"]})
        rcs_seen = [m2_to_dbsm(model.mean_rcs(incidence_in_body(p.pos, r["p"], r["orientation"])))
                    for r in rows for p in radars]
        reps = reports.get(tid, [])
        snr = [r["snr_db"] for r in reps]
        doppler = [abs(r["doppler_hz"]) for r in reps]
        rdot = [abs(r["z"][3]) for r in reps]
        detected_steps = len({r["step_index"] for r in reps})
        # tracking: a confirmed fused track within 500 m of the truth at that step
        fused = {}
        for tr in rec["tracks"]:
            if tr["source"] == "fused" and tr["confirmed"]:
                fused.setdefault(tr["step_index"], []).append(np.asarray(tr["x"][:3]))
        tracked = [r["step_index"] for r in rows
                   if any(np.linalg.norm(x - r["p"]) <= 500.0 for x in fused.get(r["step_index"], []))]
        first_report = min((r["step_index"] for r in reps), default=None)
        # reference markers: closest approach of the realised path to every zone
        zone_dist = {z["name"]: float(np.min(np.linalg.norm(P[:, :2] - np.asarray(z["position"]), axis=1)))
                     for z in sc.get("zones", []) if z["kind"] != "facility"} if spec_t["target_class"] == "missile" else {}
        checks = {
            "speed_in_hard_band": bool(spec.speed_range_mps[0] - 0.5 <= speed.min()
                                       and speed.max() <= spec.speed_range_mps[1] + 0.5),
            "agl_in_hard_band": bool(agl.min() >= spec.altitude_agl_range_m[0] - 1
                                     and agl.max() <= spec.altitude_agl_range_m[1] + 1),
            "climb_within_limit": bool(np.abs(V[:, 2]).max() <= spec.max_climb_rate_mps + 0.5),
            "lateral_accel_within_limit": bool(a_lat.size == 0
                                               or a_lat.max() <= spec.max_lateral_accel_mps2 * 1.05 + 0.5),
            "turn_rate_within_limit": bool(turn.size == 0 or turn.max() <= spec.max_turn_rate_rad_s * 1.05 + 1e-3),
            "never_clamped": bool(agl.min() >= MIN_AGL_M + 0.5),
        }
        out.append({
            "target_id": tid, "class": spec_t["target_class"], "variant": spec_t["variant"],
            "label": spec_t["label"], "swerling": spec.swerling,
            "speed_mps": _stats(speed), "agl_m": _stats(agl), "vz_mps": _stats(V[:, 2]),
            "lateral_accel_g": _stats(a_lat / 9.81) if a_lat.size else None,
            "turn_rate_deg_s": _stats(np.degrees(turn)) if turn.size else None,
            "rcs_seen_dbsm": _stats(rcs_seen), "rcs_table_dbsm": spec_t["rcs_dbsm"],
            "reports": len(reps), "detected_step_share": detected_steps / max(1, len(rows) - 1),
            "tracked_step_share": len(tracked) / max(1, len(rows) - 1),
            "first_report_t_s": None if first_report is None else first_report * dt,
            "first_tracked_t_s": None if not tracked else min(tracked) * dt,
            "reports_per_radar": {int(k): int(v) for k, v in zip(*np.unique(
                [r["radar_id"] for r in reps], return_counts=True))} if reps else {},
            "zone_min_dist_m": zone_dist,
            "snr_db": _stats(snr), "abs_doppler_khz": _stats(np.array(doppler) / 1e3) if doppler else None,
            "abs_rdot_mps": _stats(rdot),
            "checks": checks,
        })
    zones = sc.get("zones", [])
    grid_ok = all(_inside_grid(terrain, np.asarray([z["position"]], float), 0.0) for z in zones)
    return {"radars": len(radars), "targets": out, "zones": zones, "zones_inside_terrain": bool(grid_ok),
            "terrain": {k: sc["terrain"].get(k) for k in ("family", "relief_height_m", "valley_width_m")},
            "terrain_relief_m": float(terrain.H.max() - terrain.H.min()) if terrain.source != "flat" else 0.0,
            "false_alarm_reports": sum(a["is_false_alarm"] for a in rec["annotations"]),
            "label_space": meta.get("label_space")}


def audit_class_dataset(dataset_dir) -> dict:
    root = Path(dataset_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = []
    for entry in manifest["episodes"]:
        if entry["generation_status"] != "completed":
            episodes.append({"dataset_episode_id": entry["dataset_episode_id"],
                             "status": entry["generation_status"], "error": entry["error_reason"]})
            continue
        directory = root / entry["split"] / entry["dataset_episode_id"]
        episodes.append({"dataset_episode_id": entry["dataset_episode_id"], "split": entry["split"],
                         "status": "completed", **audit_episode(directory)})
    all_checks = [all(t["checks"].values()) for e in episodes if e["status"] == "completed" for t in e["targets"]]
    return {"dataset_dir": str(root), "episodes": episodes,
            "targets_checked": len(all_checks), "targets_passing_all_checks": int(sum(all_checks))}


def attach_previews(dataset_dir, plan: dict) -> None:
    """Add the dataset browser's ``episode_preview`` to every manifest entry.

    Only descriptive fields are added; identity fields that resume checks
    compare are untouched.
    """
    from src.services.synthetic_dataset_service import _atomic_json

    path = Path(dataset_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    by_id = {e["dataset_episode_id"]: e for e in plan["episodes"]}
    for entry in manifest["episodes"]:
        e = by_id[entry["dataset_episode_id"]]
        kinds = [f"{c}/{v}" for c, v in zip(e["target_classes"], e["target_variants"])]
        composition = ("single" if e["target_count"] == 1 else
                       "all_same" if len(set(e["target_classes"])) == 1 else
                       "all_different" if len(set(e["target_classes"])) == e["target_count"] else "two_plus_one")
        entry["episode_preview"] = {
            "dataset_name": e["scenario_family"], "episode_id": e["dataset_episode_id"],
            "display_name": f"EP-{e['episode_index']:04d} · "
                            + (f"{e['dive_profile']} · " if "dive_profile" in e else "")
                            + (f"{e['route_pattern']} · " if "route_pattern" in e else "")
                            + f"{e['radar_count']}R · " + ", ".join(kinds),
            "split": e["split"], "scenario_family": e["scenario_family"],
            "terrain_family": e["terrain_family"], "texture_family": e["texture_family"],
            "obstacle_profile": e["obstacle_profile"], "radar_count": e["radar_count"],
            "radar_placement_profile": "sampled", "target_count": e["target_count"],
            "target_types": kinds, "target_profiles": e["target_variants"],
            "motion_profiles": ["+".join(m) for m in e["maneuvers"]], "class_labels": e["class_labels"],
            "physical_target_type_composition": composition, "target_profile_composition": composition,
            "route_family": e.get("route_pattern", "sampled"), "episode_seed": e["episode_seed"], "terrain_seed": e["terrain_seed"],
            "quality_summary": {"planned_visible_fraction": e["planned_visible_fraction"]},
        }
    _atomic_json(manifest, path)


def _fmt(st, f="{:.0f}"):
    return "—" if st is None else f"{f.format(st['min'])} / {f.format(st['p50'])} / {f.format(st['max'])}"


def audit_markdown(audit: dict, plan: dict) -> str:
    by_id = {e["dataset_episode_id"]: e for e in plan["episodes"]}
    lines = ["# Hedef sınıfları pilot dataset denetimi", "",
             f"Dataset: `{audit['dataset_dir']}`  ",
             f"Tüm kontrolleri geçen hedef: **{audit['targets_passing_all_checks']} / {audit['targets_checked']}**", "",
             "Değerler `min / medyan / max`. Bantlar `docs/target_classification.md` ve "
             "`src/core/target_classes.py` ile aynı kaynaktan.", ""]
    for e in audit["episodes"]:
        p = by_id[e["dataset_episode_id"]]
        lines.append(f"## {e['dataset_episode_id']} ({p['split']})")
        if e["status"] != "completed":
            lines += [f"Üretim başarısız: {e.get('error')}", ""]
            continue
        sc = p["resolved_scenario"]
        lines += [f"- Arazi: {p['terrain_family']} / {p['obstacle_profile']} / {p['texture_family']}",
                  *([f"- Arazi profili: **{p['rugged_profile']}**, kabartı {e['terrain_relief_m']:.0f} m "
                     f"(ayarlanan relief {e['terrain']['relief_height_m']:.0f} m, vadi {e['terrain']['valley_width_m']:.0f} m)"]
                    if "rugged_profile" in p else []),
                  *([f"- Rota deseni: **{p['route_pattern']}** ({p['missile_count']} füze, fazlar {p['route_phases']}, "
                     f"yönler {p['route_headings_deg']}°)",
                     "- Bölgeler: " + "; ".join(f"{z['name']} [{z['kind']}] ({z['position'][0]:.0f}, {z['position'][1]:.0f})"
                                                for z in e['zones']) + f" — arazi içinde: {e['zones_inside_terrain']}",
                     "- Füze–bölge en yakın geçiş (m): " + "; ".join(
                         f"T{t['target_id']} " + ", ".join(f"{k} {v:.0f}" for k, v in t['zone_min_dist_m'].items())
                         for t in e['targets'] if t['zone_min_dist_m'])] if "route_phases" in p else []),
                  *([f"- Arazi: **{p['terrain_kind']}**, ölçülen kabartı {e['terrain_relief_m']:.0f} m",
                     f"- Salvo: **{p['route_pattern']}**, {p['missile_count']} füze, süre {sc['duration_s']:g} s "
                     f"(hedefe eşzamanlı varış)",
                     "- Profiller: " + "; ".join(
                         f"T{k} {prof} → {zone or '—'} (tehdit ekseni {ax if ax is not None else '—'}°, "
                         f"giriş {lead:+.1f} s)" for k, (prof, zone, ax, lead) in enumerate(
                             zip(p['salvo_profiles'], p['target_zones'], p['threat_axes_deg'], p['lead_in_s']))),
                     "- Bölgeler: " + "; ".join(f"{z['name']} [{z['kind']}] ({z['position'][0]:.0f}, {z['position'][1]:.0f})"
                                                for z in e['zones']) + f" — arazi içinde: {e['zones_inside_terrain']}",
                     "- Füze–bölge en yakın geçiş (m): " + "; ".join(
                         f"T{t['target_id']} " + ", ".join(f"{k} {v:.0f}" for k, v in t['zone_min_dist_m'].items())
                         for t in e['targets'] if t['zone_min_dist_m'])] if "salvo_profiles" in p else []),
                  f"- Radar: {p['radar_count']} — " + ", ".join(
                      f"{r['name']} ({r['position'][0]:.0f}, {r['position'][1]:.0f}) yön {r['heading_deg']:.0f}°"
                      for r in sc["radars"]),
                  f"- Planlanan görünürlük oranı: {p['planned_visible_fraction']}"
                  + (f" (radar başına {p['planned_visible_fraction_per_radar']})"
                     if "planned_visible_fraction_per_radar" in p else ""),
                  *([f"- Dalış profili: **{p['dive_profile']}**, yaklaşma yönü {p['approach_deg']:.0f}°, "
                     f"başlangıç AGL {p['start_agl_m']:.0f} m, süre {sc['duration_s']:g} s, "
                     f"korunan nokta {p['defended_point']}"] if "dive_profile" in p else []),
                  "- Arama barları (°): " + "; ".join(f"{r['name']} {r.get('search_bars_deg', sc['radar_defaults'].get('search_bars_deg'))}"
                                                     for r in sc["radars"]),
                  f"- Yanlış alarm raporu: {e['false_alarm_reports']}", "",
                  "| T | Sınıf/varyant | Manevra | Hız m/s | AGL m | v_z m/s | a_n g | Dönüş °/s | "
                  "RCS (görülen) dBsm | Rapor (radar başına) | Tespit adımı | İzlenen adım | İlk rapor / iz s | "
                  "SNR dB | \\|ṙ\\| m/s | Kontroller |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for t, man in zip(e["targets"], p["maneuvers"]):
            failed = [k for k, ok in t["checks"].items() if not ok]
            lines.append(
                f"| T{t['target_id']} | {t['class']}/{t['variant']} (L{t['label']}) | {'+'.join(man)} | "
                f"{_fmt(t['speed_mps'])} | {_fmt(t['agl_m'])} | {_fmt(t['vz_mps'], '{:.1f}')} | "
                f"{_fmt(t['lateral_accel_g'], '{:.2f}')} | {_fmt(t['turn_rate_deg_s'], '{:.1f}')} | "
                f"{_fmt(t['rcs_seen_dbsm'], '{:.1f}')} | {t['reports']} {t['reports_per_radar']} | "
                f"{t['detected_step_share']:.0%} | {t['tracked_step_share']:.0%} | "
                f"{t['first_report_t_s'] if t['first_report_t_s'] is not None else '—'} / "
                f"{t['first_tracked_t_s'] if t['first_tracked_t_s'] is not None else '—'} | "
                f"{_fmt(t['snr_db'], '{:.1f}')} | {_fmt(t['abs_rdot_mps'])} | "
                f"{'tamam' if not failed else 'HATA: ' + ', '.join(failed)} |")
        lines.append("")
    lines += ["## Varyant bantları (referans)", "",
              "| Varyant | Hız m/s (sert) | AGL m (sert) | max \\|v_z\\| | max a_n g | max dönüş °/s |",
              "|---|---|---|---|---|---|"]
    for c in CLASS_ORDER:
        for v, s in TARGET_CLASSES[c]["variants"].items():
            lines.append(f"| {c}/{v} | {s.speed_range_mps[0]:g}–{s.speed_range_mps[1]:g} | "
                         f"{s.altitude_agl_range_m[0]:g}–{s.altitude_agl_range_m[1]:g} | {s.max_climb_rate_mps:g} | "
                         f"{s.max_lateral_accel_mps2 / 9.81:.1f} | {math.degrees(s.max_turn_rate_rad_s):.1f} |")
    return "\n".join(lines) + "\n"
