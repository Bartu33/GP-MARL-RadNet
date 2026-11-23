"""Missile salvo episodes: 1-3 missiles with realistic attack profiles, time-on-target.

Every missile flies one attack profile inside the envelope of its variant in
``src/core/target_classes.py`` (the class structure used by the classifier):

* ``missile/terminal_dive`` -- the ten profiles of ``missile_dive_service``:
  high-altitude dives (steep, shallow, jink, pull-out, weave) and evasive
  profiles flown by its autopilot (pop-up, barrel roll, dogleg + serpentine,
  break turn, random 3-D jinks).
* ``missile/subsonic_cruise`` -- ``cruise_weave``: terrain following at
  40-90 m AGL, optional dogleg through a turn point, S-weave in the last few
  kilometres (<= the variant's 0.15 rad/s and 4 g) and an optional terminal
  pop-up / descent inside the 20 m/s climb limit.
* ``missile/supersonic`` -- ``supersonic_skim``: low-level dash, optional
  dogleg, terminal high-g weave (<= 0.2 rad/s, 15 g).

The salvo is a time-on-target attack: all missiles reach their terminal point
at the end of the episode.  Episode length is set by the low-level missiles
(their flight cannot be extended without re-flying the terrain); a shorter
high-altitude flight gets a level lead-in, a longer one joins the episode
mid-cruise.  Route patterns:

* ``converging`` -- every missile on Zone A, threat axes at least
  ``min_axis_separation_deg`` apart;
* ``diverging``  -- one threat axis (+- ``axis_jitter_deg``), every missile on
  its own zone (A, B, C) a few kilometres apart.

Radars stand on local high ground facing the threat (``plan_dive_radars``);
the layout is kept only if each target is visible (range, FOV, search bar,
terrain line of sight) for a minimum share of the episode.  Zones and
facilities are abstract markers drawn by the 3-D viewer.  Plans use the
planner schema (recipe version 2) and are written by
``synthetic_dataset_service.generate_dataset``.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction
import math

import numpy as np

from src.core.env import _cached_terrain
from src.core.scenario import scenario_snapshot, validate_scenario
from src.core.target_classes import RCS_ASPECTS, TARGET_CLASSES, variant_spec
from src.services.class_dataset_service import (
    CHECK_STEP_S, GRID_MARGIN_M, RECIPE_VERSION, RUGGED_PROFILES, _effective_radar, _inside_grid,
    check_trajectory, plan_target, propagate,
)
from src.services.missile_dive_service import (
    EVASIVE_PROFILES, PROFILES as DIVE_SERVICE_PROFILES, _Autopilot, _bars, _candidate_sites, _dir, _heading_to,
    _tf_gamma, _visible_mask, plan_dive_target, plan_evasive_target, rolling_hills,
)
from src.services.scenario_planner import SPLITS, _digest, _json_value, _physical_numbers, terrain_group_id

CLASS = "missile"
LOW_PROFILES = ("cruise_weave", "supersonic_skim")
PROFILE_VARIANT = {**{p: "terminal_dive" for p in DIVE_SERVICE_PROFILES},
                   "cruise_weave": "subsonic_cruise", "supersonic_skim": "supersonic"}
# flights that start low: a level lead-in would cut through the terrain, so they set the episode length
LOW_LEVEL = ("popup_dive", "dogleg_serpentine") + LOW_PROFILES
PATTERNS = ("converging", "diverging")
ZONE_NAMES = ("Zone A", "Zone B", "Zone C")
TERRAIN_KINDS = ("rolling_hills",) + tuple(RUGGED_PROFILES)

DEFAULT_RECIPE = {
    "recipe_version": RECIPE_VERSION,
    "dataset_name": "missile_salvo_50",
    "master_seed": 20261127,
    "episodes": 50,
    # weave_dive needs 23-29 km of ground track: it rarely fits the 30 km terrain grid, so it is
    # left out by default (weaving is covered by jink_dive, dogleg_serpentine and the low-level profiles)
    "profiles": [p for p in PROFILE_VARIANT if p != "weave_dive"],
    "patterns": list(PATTERNS),
    "missile_count": [1, 3],
    "other_targets": [0, 1],           # UAV / aircraft traffic flying through the area
    "dt": 0.5,
    "max_track_slots": 4,
    "area_radius_m": 12000.0,
    "site": {"name": "Salvo Test Range", "lat": 40.30, "lon": -111.70},
    "radar_count": [1, 3],
    "split_ratios": {"train": 0.70, "validation": 0.15, "test": 0.15},
    "duration_s": [14.0, 34.0],        # accepted episode length (time-on-target window)
    "terrain": {"resolution": 256, "kinds": list(TERRAIN_KINDS),
                "hills": [12, 20], "height_m": [60.0, 450.0], "sigma_m": [700.0, 2500.0], "extent_m": 13000.0,
                "textures": ["rocky_mountain", "green_forest_valley", "dry_canyon", "snowy_highlands",
                             "grassland_valley", "mixed_rock_vegetation"]},
    "salvo": {"min_axis_separation_deg": 50.0, "axis_jitter_deg": 20.0, "sector_half_width_deg": 12.0,
              "zone_spacing_m": [3000.0, 5000.0]},
    "zones": {"defended_point_radius_m": 1500.0, "zone_radius_m": [350.0, 600.0],
              "zone_height_m": [60.0, 220.0], "facility_count": [0, 2],
              "facility_radius_m": [150.0, 300.0], "facility_height_m": [40.0, 120.0]},
    "radar": {
        "distance_from_point_m": [1500.0, 7000.0],
        "min_separation_m": 2000.0,
        "high_ground_quantile": 0.80,
        "local_high_quantile": 0.80,
        "mast_m": [8.0, 20.0],
        "heading_jitter_deg": 10.0,
        "max_bars": 8,
        "low_site_max_rel_m": 600.0,   # low-level episodes: extra sites at most this far above the zone
        "defaults": {"fov_az_deg": 45.0, "search_sector_deg": 70.0, "r_max_m": 20000.0, "el_max_deg": 70.0},
    },
    "tracker": {"tentative_timeout_s": 1.0, "init_v_std_mps": 400.0, "q": 10000.0},
    "policy": "central_greedy_confirm",
    "coverage": {"min_visible_fraction": 0.5, "min_visible_fraction_low_level": 0.3,
                 "min_visible_fraction_traffic": 0.25, "sample_step_s": 0.5,
                 "max_radar_attempts": 150, "max_episode_attempts": 40},
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
    if not r["profiles"] or set(r["profiles"]) - set(PROFILE_VARIANT):
        raise ValueError(f"profiles: non-empty subset of {list(PROFILE_VARIANT)}")
    if not r["patterns"] or set(r["patterns"]) - set(PATTERNS):
        raise ValueError(f"patterns: non-empty subset of {list(PATTERNS)}")
    if not r["terrain"]["kinds"] or set(r["terrain"]["kinds"]) - set(TERRAIN_KINDS):
        raise ValueError(f"terrain.kinds: non-empty subset of {list(TERRAIN_KINDS)}")
    lo, hi = r["missile_count"]
    o_lo, o_hi = r["other_targets"]
    if not (1 <= lo <= hi <= len(ZONE_NAMES) and 0 <= o_lo <= o_hi and hi + o_hi <= r["max_track_slots"]):
        raise ValueError("missile_count / other_targets: 1 <= missiles <= 3, missiles + traffic <= max_track_slots")
    lo, hi = r["radar_count"]
    if not 1 <= lo <= hi <= 3:
        raise ValueError("radar_count: 1 <= lo <= hi <= 3")
    if sum(Fraction(str(r["split_ratios"][s])) for s in SPLITS) != 1:
        raise ValueError("split_ratios: must sum to 1")
    return r


# ---------------------------------------------------------------------------
# low-level cruise / supersonic profiles (autopilot inside the variant envelope)
# ---------------------------------------------------------------------------

def fly_low_level(profile, rng, terrain, aim, dt, spec, sector, max_t=60.0):
    """Fly ``cruise_weave`` or ``supersonic_skim``; returns (position, velocity, segments, phases, axis_deg).

    The threat axis inside ``sector`` is the flattest of a few candidates: a
    terrain-following missile with a limited climb rate takes the low ground.
    """
    aim = np.asarray(aim, float)
    aim3 = np.array([aim[0], aim[1], float(terrain.height(*aim))])
    subsonic = profile == "cruise_weave"
    speed = float(rng.uniform(225.0, 265.0) if subsonic else rng.uniform(520.0, 650.0))
    agl_cmd = float(rng.uniform(40.0, 90.0) if subsonic else rng.uniform(50.0, 120.0))
    dist0 = float(rng.uniform(6500.0, 8000.0) if subsonic else rng.uniform(11000.0, 13000.0))
    centre, half = sector
    cands = centre + rng.uniform(-half, half, 24 if half > 1.0 else 8)
    s_ = np.linspace(0.0, dist0, 30)

    def rise(b):
        xy = aim[None, :] + s_[:, None] * np.array([math.cos(b), math.sin(b)])[None, :]
        h = terrain.height(xy[:, 0], xy[:, 1])
        return float(np.max(np.abs(np.diff(h))))           # steepest step between samples
    bearing = float(min(cands, key=rise))
    dogleg = rng.random() < 0.5
    if dogleg:
        r_w = dist0 * float(rng.uniform(0.45, 0.6))
        wp = aim + r_w * np.array([math.cos(bearing), math.sin(bearing)])
        delta = float(rng.choice([-1.0, 1.0]) * math.radians(rng.uniform(30.0, 50.0)))
        start_xy = wp + (dist0 - r_w) * np.array([math.cos(bearing + delta), math.sin(bearing + delta)])
    else:
        wp = None
        start_xy = aim + dist0 * np.array([math.cos(bearing), math.sin(bearing)])
    psi0 = _heading_to(start_xy, wp if wp is not None else aim)
    z0 = float(terrain.height(*start_xy)) + agl_cmd
    ap = _Autopilot(spec, terrain, [start_xy[0], start_xy[1], z0], speed * _dir(psi0, 0.0))
    # lateral manoeuvre limit: the autopilot also caps the heading rate at 0.95 omega_max
    lat_cap = min(ap.a_n_max, 0.95 * spec.max_turn_rate_rad_s * speed)
    weave_a = float(rng.uniform(0.6, 0.95)) * lat_cap
    weave_T = float(rng.uniform(4.0, 7.0) if subsonic else rng.uniform(2.0, 3.5))
    weave_r = float(rng.uniform(3000.0, 4500.0) if subsonic else rng.uniform(5000.0, 7000.0))
    end_r = float(rng.uniform(600.0, 1000.0) if subsonic else rng.uniform(800.0, 1500.0))
    popup = subsonic and rng.random() < 0.5
    pop_r = float(rng.uniform(2200.0, 2800.0))
    pop_agl = float(rng.uniform(150.0, 185.0))
    climb = math.asin(min(0.9 * spec.max_climb_rate_mps / speed, 0.5))

    mode, t_mode, phases = "cruise", 0.0, []
    while ap.t < max_t:
        dist = float(np.hypot(*(aim - ap.p[:2])))
        on_grid = abs(ap.t / dt - round(ap.t / dt)) < 1e-9
        if on_grid and dist <= end_r:
            break
        if mode == "cruise" and wp is not None and float(np.hypot(*(wp - ap.p[:2]))) <= 600.0:
            mode, t_mode = "turn_in", ap.t
        elif mode in ("cruise", "turn_in") and (wp is None or mode == "turn_in") and dist <= weave_r:
            mode, t_mode = "weave", ap.t
        elif mode == "weave" and popup and dist <= pop_r:
            mode, t_mode = "pop_up", ap.t
        elif mode == "pop_up" and ap.agl() >= pop_agl:
            mode, t_mode = "terminal_descent", ap.t
        target_xy = wp if (mode == "cruise" and wp is not None) else aim
        if mode in ("cruise", "turn_in"):
            a = ap.track(_heading_to(ap.p, target_xy), _tf_gamma(ap, agl_cmd), speed)
        elif mode == "weave":
            n1, _ = ap.normal_basis()
            a = ap.track(_heading_to(ap.p, aim), _tf_gamma(ap, agl_cmd), speed, tau=1.5) \
                + weave_a * math.sin(2 * math.pi * (ap.t - t_mode) / weave_T) * n1
        elif mode == "pop_up":
            a = ap.track(_heading_to(ap.p, aim), climb, speed)
        else:
            # descend at the climb limit, never below the terrain-following height
            gamma = -climb if ap.agl() > agl_cmd + 15.0 else _tf_gamma(ap, agl_cmd)
            a = ap.track(_heading_to(ap.p, aim3), gamma, speed)
        if not phases or phases[-1] != mode:
            phases.append(mode)
        ap.step(np.asarray(a, float))
    else:
        raise ValueError(f"{profile}: terminal point not reached in {max_t:.0f} s")
    names = {"cruise": "terrain_follow", "turn_in": "dogleg_turn", "weave": "weave" if subsonic else "terminal_weave",
             "pop_up": "pop_up", "terminal_descent": "terminal_descent"}
    return ([round(float(start_xy[0]), 2), round(float(start_xy[1]), 2), round(z0, 2)],
            [round(float(c), 4) for c in speed * _dir(psi0, 0.0)], ap.segs, [names[m] for m in phases],
            math.degrees(bearing))


def plan_low_level_target(profile, rng, terrain, point, dt, sector, attempts=120):
    spec = variant_spec(CLASS, PROFILE_VARIANT[profile])
    rejected = {}
    for _ in range(attempts):
        aim = np.asarray(point, float) + rng.uniform(-300.0, 300.0, 2)
        try:
            position, velocity, segs, phases, axis = fly_low_level(profile, rng, terrain, aim, dt, spec, sector)
        except ValueError as exc:
            reason = str(exc).split(":")[-1].strip()
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        duration = round(sum(s["duration_s"] for s in segs), 6)
        P, V = propagate(position, velocity, {"segments": segs}, duration)
        reason = check_trajectory(spec, P, V, terrain)
        if reason is None:
            target = {"target_class": CLASS, "variant": PROFILE_VARIANT[profile],
                      "label": TARGET_CLASSES[CLASS]["label"], "position": position, "velocity": velocity,
                      "motion": {"segments": segs},
                      "rcs_dbsm": {a: round(float(rng.uniform(*spec.rcs_dbsm[a])), 3) for a in RCS_ASPECTS}}
            return target, {"profile": profile, "maneuvers": phases, "trajectory": P, "duration_s": duration,
                            "approach_deg": round(axis, 1), "aim": aim}
        rejected[reason] = rejected.get(reason, 0) + 1
    raise ValueError(f"{profile}: no valid trajectory ({rejected})")


# ---------------------------------------------------------------------------
# time-on-target alignment
# ---------------------------------------------------------------------------

def turn_rate_ok(spec, V, every):
    """Heading rate on the simulator's dt grid, measured as the audit does, within the variant limit.

    The autopilot caps the heading rate with the horizontal speed at the start of each
    0.25 s piece; in a steep dive that speed falls inside the piece, so this check is
    needed on top of check_trajectory.
    """
    V = V[::every]
    dt = CHECK_STEP_S * every
    nh = np.linalg.norm(V[:, :2], axis=1)
    sp = np.linalg.norm(V, axis=1)
    ok = (nh[:-1] > 1.0) & (nh[1:] > 1.0) & (nh[:-1] > 0.2 * sp[:-1]) & (nh[1:] > 0.2 * sp[1:])
    heading = np.unwrap(np.arctan2(V[:, 1], V[:, 0]))
    turn = np.abs(np.diff(heading) / dt)[ok]
    return turn.size == 0 or float(turn.max()) <= spec.max_turn_rate_rad_s


def _state_at(target, t):
    if t <= 0:
        return np.asarray(target["position"], float), np.asarray(target["velocity"], float)
    n = max(1, int(round(t / CHECK_STEP_S)))
    P, V = propagate(target["position"], target["velocity"], target["motion"], t, step=t / n)
    return P[-1], V[-1]


def align_to_duration(target, duration_s, total, dt):
    """Return a copy of ``target`` whose motion lasts exactly ``total`` and ends where it ended.

    Longer flights join mid-way (the first part is dropped); shorter ones get a
    level CV lead-in along the initial velocity.  Returns (target, lead_in_s).
    """
    out = deepcopy(target)
    segs = target["motion"]["segments"]
    shift = round(duration_s - total, 6)
    if abs(shift) < 1e-9:
        return out, 0.0
    if shift < 0:                     # lead-in
        lead = -shift
        p = np.asarray(target["position"], float) - np.asarray(target["velocity"], float) * lead
        out["position"] = [round(float(x), 2) for x in p]
        out["motion"] = {"segments": [{"model": "CV", "duration_s": lead}] + deepcopy(segs)}
        return out, lead
    p, v = _state_at(target, shift)
    rest, elapsed = [], 0.0
    for seg in segs:
        end = elapsed + seg["duration_s"]
        if end > shift + 1e-9:
            seg = dict(seg)
            seg["duration_s"] = round(end - max(elapsed, shift), 6)
            rest.append(seg)
        elapsed = end
    out["position"] = [round(float(x), 2) for x in p]
    out["velocity"] = [round(float(x), 4) for x in v]
    out["motion"] = {"segments": rest}
    return out, -shift


# ---------------------------------------------------------------------------
# terrain, zones
# ---------------------------------------------------------------------------

def _terrain_spec(kind, rng, r, seed):
    t = r["terrain"]
    if kind == "rolling_hills":
        return {"source": "procedural", "resolution": t["resolution"], "hills": rolling_hills(rng, t)}
    prof = RUGGED_PROFILES[kind]
    obstacle = prof["obstacle_profiles"][int(rng.integers(len(prof["obstacle_profiles"])))]
    params = {"valley_angle_deg": round(float(rng.uniform(0.0, 180.0)), 1),
              "ridge_width_m": round(float(rng.uniform(*prof["ridge_width_m"])), 1)}
    if obstacle == "asymmetric":
        params["asymmetry_ratio"] = round(float(rng.uniform(1.3, 1.7)), 2)
    return {"source": "procedural", "resolution": t["resolution"], "family": prof["family"],
            "texture_family": t["textures"][int(rng.integers(len(t["textures"])))],
            "obstacle_profile": obstacle,
            "valley_width_m": round(float(rng.uniform(*prof["valley_width_m"])), 1),
            "relief_height_m": round(float(rng.uniform(*prof["relief_height_m"])), 1),
            "parameters": params, "seed": seed}


def _aim_points(pattern, n, rng, terrain, r):
    """Zone positions: one defended point (converging) or one per missile (diverging), on low ground."""
    zr = r["zones"]["defended_point_radius_m"]

    def low_point(around, radius):
        pts = [np.asarray(around, float) + radius * math.sqrt(rng.uniform())
               * np.array([math.cos(a), math.sin(a)]) for a in rng.uniform(-math.pi, math.pi, 6)]
        return min(pts, key=lambda q: float(terrain.height(*q)))

    first = low_point((0.0, 0.0), zr)
    if pattern == "converging" or n == 1:
        return [first]
    points = [first]
    for _ in range(n - 1):
        for _ in range(60):
            ang = rng.uniform(-math.pi, math.pi)
            q = first + float(rng.uniform(*r["salvo"]["zone_spacing_m"])) * np.array([math.cos(ang), math.sin(ang)])
            if all(np.linalg.norm(q - o) >= r["salvo"]["zone_spacing_m"][0] for o in points) \
                    and _inside_grid(terrain, q[None, :], 3000.0):
                points.append(q)
                break
        else:
            raise ValueError("diverging: no room for another zone")
    return points


# ground distance a profile needs between the launch side and its aim point (grid room)
# (about the 10th percentile of the profile's start-to-aim distance; the planners draw until one fits)
ROOM_M = {"steep_dive": 11700.0, "shallow_dive": 13000.0, "jink_dive": 13600.0, "dive_pullout": 18900.0,
          "weave_dive": 23300.0, "popup_dive": 13500.0, "barrel_roll_dive": 13500.0, "dogleg_serpentine": 13500.0,
          "break_turn_dive": 13500.0, "random_jink_dive": 13500.0,
          "cruise_weave": 8500.0, "supersonic_skim": 13500.0}


def _room(terrain, point, angle, margin=GRID_MARGIN_M):
    """Distance from ``point`` to the terrain grid edge along ``angle``."""
    size = terrain.cell * terrain.H.shape[1]
    lo = np.array([terrain.x_west + margin, terrain.y_north - size + margin])
    hi = np.array([terrain.x_west + size - margin, terrain.y_north - margin])
    d = np.array([math.cos(angle), math.sin(angle)])
    t = [((hi[i] if d[i] > 0 else lo[i]) - point[i]) / d[i] for i in range(2) if abs(d[i]) > 1e-9]
    return float(min(t))


def _sectors(pattern, profiles, points, rng, r, terrain, base, tries=300):
    """Threat-axis sector (centre, half width) per missile, with grid room for its profile.

    Missile 0 is already planned on threat axis ``base``; its sector is returned as (base, 0).
    """
    s = r["salvo"]
    n = len(profiles)
    half = math.radians(s["sector_half_width_deg"])
    jit = math.radians(s["axis_jitter_deg"])
    sep = math.radians(s["min_axis_separation_deg"])
    for _ in range(tries):
        if pattern == "diverging":
            centres = [base] + [base + float(rng.uniform(-jit, jit)) for _ in range(n - 1)]
        else:
            spread = float(rng.choice([-1.0, 1.0]) * rng.uniform(sep, max(sep, 2 * math.pi / n - 2 * half)))
            centres = [base + k * spread for k in range(n)]
        if all(min(_room(terrain, points[k % len(points)], c + o) for o in (-half, 0.0, half)) >= ROOM_M[p]
               for k, (p, c) in enumerate(zip(profiles, centres)) if k > 0):
            return [(base, 0.0)] + [(c, half) for c in centres[1:]]
    raise ValueError(f"{pattern}: no threat axes with grid room for {profiles}")


def _zones(points, rng, terrain, r, radars, trajectories):
    z = r["zones"]

    def draw(key):
        return round(float(rng.uniform(*z[key])), 1)

    zones = [{"name": ZONE_NAMES[i], "kind": "zone", "position": [round(float(q[0]), 1), round(float(q[1]), 1)],
              "radius_m": draw("zone_radius_m"), "height_m": draw("zone_height_m")} for i, q in enumerate(points)]
    taken = [np.asarray(q, float) for q in points] + [np.asarray(p["position"][:2], float) for p in radars]
    n_fac = int(rng.integers(z["facility_count"][0], z["facility_count"][1] + 1))
    placed = 0
    for _ in range(40 * n_fac):
        if placed >= n_fac:
            break
        ang, rad = rng.uniform(-math.pi, math.pi), 6000.0 * math.sqrt(rng.uniform())
        q = np.array([rad * math.cos(ang), rad * math.sin(ang)])
        if all(np.linalg.norm(q - o) >= 1500.0 for o in taken) and _inside_grid(terrain, q[None, :], GRID_MARGIN_M):
            placed += 1
            taken.append(q)
            zones.append({"name": f"Facility {placed}", "kind": "facility",
                          "position": [round(float(q[0]), 1), round(float(q[1]), 1)],
                          "radius_m": draw("facility_radius_m"), "height_m": draw("facility_height_m")})
    return zones


# ---------------------------------------------------------------------------
# radars
# ---------------------------------------------------------------------------

def plan_salvo_radars(n, rng, terrain, point, trajectories, needs, r):
    """Radars on local high ground around the salvo, facing the threat, bars over its elevation band.

    A layout is kept only if every target is visible (range, FOV, inside a search
    bar, terrain line of sight) for at least its share ``needs[i]`` of the episode
    and every radar sees something.  Returns (specs, shares).
    """
    cfg, cov = r["radar"], r["coverage"]
    every = max(1, int(round(cov["sample_step_s"] / CHECK_STEP_S)))
    low_level = any(need < cov["min_visible_fraction"] for need in needs)
    sites = _candidate_sites(terrain, point, cfg)
    if low_level:
        # against low-level threats radars also stand on lower shoulders and the valley floor:
        # from a crest far above the valley the missile is below the search fan or behind the edge
        lo, hi = cfg["distance_from_point_m"]
        g = np.arange(-hi, hi + 1, 250.0)
        X, Y = np.meshgrid(point[0] + g, point[1] + g)
        d = np.hypot(X - point[0], Y - point[1])
        xy = np.column_stack([X[(d >= lo) & (d <= hi)], Y[(d >= lo) & (d <= hi)]])
        rel = terrain.height(xy[:, 0], xy[:, 1]) - float(terrain.height(*point))
        low_sites = xy[rel <= cfg["low_site_max_rel_m"]]
        if len(low_sites):
            sites = np.vstack([sites, low_sites[rng.permutation(len(low_sites))[:len(sites) or 200]]])
    stacked = np.vstack(trajectories)
    best = -1.0
    for _ in range(cov["max_radar_attempts"]):
        chosen = []
        for i in rng.permutation(len(sites)):
            if all(np.hypot(*(sites[i] - c)) >= cfg["min_separation_m"] for c in chosen):
                chosen.append(sites[i])
            if len(chosen) == n:
                break
        if len(chosen) < n:
            continue
        specs, masks = [], []
        for k, xy in enumerate(chosen):
            # face a random target's in-range path: radars of one layout look at different threats
            traj = trajectories[int(rng.integers(len(trajectories)))]
            d = np.linalg.norm(traj[:, :2] - xy, axis=1)
            near = traj[d <= cfg["defaults"]["r_max_m"] * 0.8]
            look = (near if len(near) else traj)[:, :2].mean(axis=0)
            heading = math.degrees(math.atan2(look[1] - xy[1], look[0] - xy[0]))
            heading += float(rng.uniform(-cfg["heading_jitter_deg"], cfg["heading_jitter_deg"]))
            spec = {"name": f"R{k + 1}",
                    "position": [round(float(xy[0]), 1), round(float(xy[1]), 1),
                                 round(float(rng.uniform(*cfg["mast_m"])), 1)],
                    "heading_deg": round((heading + 180) % 360 - 180, 2)}
            if low_level:
                # a radar on a ridge looks down into the valley: a missile 1 km below at 5 km is at -11 deg
                spec["el_min_deg"] = -20.0
            params = _effective_radar(spec, cfg["defaults"], terrain)
            spec["search_bars_deg"] = _bars(params, stacked, params.beamwidth_el_deg, cfg["max_bars"],
                                            floor_deg=-18.0 if low_level else 1.5)
            params = _effective_radar(spec, cfg["defaults"], terrain)
            specs.append(spec)
            masks.append([_visible_mask(t, params, terrain, every) for t in trajectories])
        shares = [float(np.any([m[i] for m in masks], axis=0).mean()) for i in range(len(trajectories))]
        margin = min(s_ - need for s_, need in zip(shares, needs))
        useful = all(max(float(x.mean()) for x in m) >= 0.1 for m in masks)
        if margin >= 0 and useful:
            return specs, shares
        best = max(best, margin)
    raise ValueError(f"radar layout: a target below its visible share (best margin {best:+.2f})")


# ---------------------------------------------------------------------------
# plan
# ---------------------------------------------------------------------------

def _cycle(master, stream, index, items):
    rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(stream, index // len(items))))
    return items[int(rng.permutation(len(items))[index % len(items)])]


def plan_salvo_episodes(recipe=None):
    r = resolve_recipe(recipe)
    master, dt = int(r["master_seed"]), float(r["dt"])
    family = r["dataset_name"]
    ratios = [Fraction(str(r["split_ratios"][s])) for s in SPLITS]
    split_counts = [0, 0, 0]
    episodes, owners = [], {}
    profiles = list(r["profiles"])
    profile_slot = 0
    for index in range(r["episodes"]):
        which = max(range(3), key=lambda j: (index + 1) * ratios[j] - split_counts[j])
        split = SPLITS[which]
        split_counts[which] += 1
        rng = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(10, index)))
        episode_seed = int(rng.integers(0, 2 ** 63, dtype=np.int64))
        n_radars = _cycle(master, 11, index, list(range(r["radar_count"][0], r["radar_count"][1] + 1)))
        n_missiles = _cycle(master, 12, index, list(range(r["missile_count"][0], r["missile_count"][1] + 1)))
        n_other = _cycle(master, 13, index, list(range(r["other_targets"][0], r["other_targets"][1] + 1)))
        pattern = _cycle(master, 14, index, list(r["patterns"])) if n_missiles > 1 else "single"
        kind = _cycle(master, 15, index, list(r["terrain"]["kinds"]))
        # profiles: a global round robin over a shuffled list, so every profile appears about equally often
        chosen = []
        for _ in range(n_missiles):
            perm = np.random.default_rng(np.random.SeedSequence(master, spawn_key=(16, profile_slot // len(profiles))))
            chosen.append(profiles[int(perm.permutation(len(profiles))[profile_slot % len(profiles)])])
            profile_slot += 1

        errors = []
        substituted = []
        n_att = r["coverage"]["max_episode_attempts"]
        for attempt in range(n_att):
            if attempt == (3 * n_att) // 4 and n_missiles > 1:
                # last resort: the profile needing the most room gives way to a short steep dive
                k = max(range(n_missiles), key=lambda i: ROOM_M[chosen[i]])
                if chosen[k] != "steep_dive":
                    substituted.append({"was": chosen[k], "now": "steep_dive"})
                    chosen[k] = "steep_dive"
            seed = int(rng.integers(0, 2 ** 31))
            terrain_spec = _terrain_spec(kind, rng, r, seed)
            terrain = _cached_terrain({"site": r["site"], "terrain": terrain_spec, "seed": seed})
            try:
                points = _aim_points(pattern, n_missiles, rng, terrain, r)
                # the profile that needs the most room flies first, on any axis that fits the grid;
                # the others take their axes relative to it
                # (a subsonic cruise missile is the most terrain-bound: it picks its valley first)
                # (after half of the attempts the room-hungry profile goes first instead)
                valley_first = attempt < n_att // 2
                order = sorted(range(n_missiles),
                               key=lambda k: (valley_first and chosen[k] != "cruise_weave", -ROOM_M[chosen[k]]))
                planned = [None] * n_missiles
                sectors = None
                for j, k in enumerate(order):
                    profile, point = chosen[k], points[k % len(points)]
                    sector = None if j == 0 else sectors[j]
                    if profile in LOW_PROFILES:
                        sector = sector or (float(rng.uniform(-math.pi, math.pi)), math.pi)
                        t, info = plan_low_level_target(profile, rng, terrain, point, dt, sector)
                    elif profile in EVASIVE_PROFILES:
                        t, info = plan_evasive_target(profile, rng, terrain, point, dt, sector=sector)
                    else:
                        t, info = plan_dive_target(profile, rng, terrain, point, dt, sector=sector)
                    # threat axis: from the aim point towards the launch side
                    axis = math.atan2(t["position"][1] - info["aim"][1], t["position"][0] - info["aim"][0])
                    info["approach_deg"] = round(math.degrees(axis), 1)
                    info["zone"] = ZONE_NAMES[k % len(points)]
                    planned[k] = (t, info)
                    if j == 0:
                        sectors = _sectors(pattern, [chosen[i] for i in order], [points[i % len(points)] for i in order],
                                           rng, r, terrain, axis)
                # time on target: low-level flights set the length, the others are aligned to it
                low = [i["duration_s"] for t, i in planned if i["profile"] in LOW_LEVEL]
                total = min(low) if low else max(i["duration_s"] for _, i in planned)
                total = round(math.floor(total / dt + 1e-9) * dt, 6)
                if not r["duration_s"][0] <= total <= r["duration_s"][1]:
                    raise ValueError(f"time on target: {total:.1f} s outside {r['duration_s']}")
                targets, infos = [], []
                for t, info in planned:
                    t2, lead = align_to_duration(t, info["duration_s"], total, dt)
                    spec = variant_spec(CLASS, t2["variant"])
                    P, V = propagate(t2["position"], t2["velocity"], t2["motion"], total)
                    reason = check_trajectory(spec, P, V, terrain)
                    if reason is None and not turn_rate_ok(spec, V, int(round(dt / CHECK_STEP_S))):
                        reason = "turn_rate"
                    if reason is not None:
                        raise ValueError(f"{info['profile']}: aligned trajectory {reason}")
                    info = {**info, "trajectory": P, "lead_in_s": round(lead, 2)}
                    targets.append(t2)
                    infos.append(info)
                needs = [r["coverage"]["min_visible_fraction_low_level"] if i["profile"] in LOW_LEVEL
                         else r["coverage"]["min_visible_fraction"] for i in infos]
                for _ in range(n_other):
                    cls = ("uav", "aircraft")[int(rng.integers(2))]
                    t, meta = plan_target(cls, rng, terrain, total, dt, tuple(points[0]),
                                          {"cpa_radius_m": 5000.0, "segments": [1, 3], "max_attempts": 300})
                    targets.append(t)
                    infos.append({"profile": f"traffic/{t['variant']}", "maneuvers": meta["maneuvers"],
                                  "trajectory": meta["trajectory"], "lead_in_s": 0.0, "zone": None,
                                  "approach_deg": None})
                    needs.append(r["coverage"]["min_visible_fraction_traffic"])
                centroid = np.mean(points, axis=0)
                radars, shares = plan_salvo_radars(n_radars, rng, terrain, centroid,
                                                   [i["trajectory"] for i in infos], needs, r)
                break
            except ValueError as exc:
                errors.append(str(exc))
        else:
            summary = Counter(e.split("(")[0].strip() for e in errors).most_common(6)
            raise ValueError(f"episode_index={index} ({kind}, {pattern}, {chosen}): planning failed "
                             f"after {len(errors)} attempts: {summary}; last: {errors[-2:]}")

        zones = _zones(points, rng, terrain, r, radars, [i["trajectory"] for i in infos])
        missiles = [i for i in infos if i["zone"] is not None]
        sc = validate_scenario({
            "name": f"{family}_ep{index:04d}",
            "description": f"{len(missiles)}-missile {pattern} salvo (" + ", ".join(i["profile"] for i in missiles)
                           + f"), {n_other} traffic, {n_radars} radar(s) on high ground",
            "site": r["site"], "dt": dt, "duration_s": total, "area_radius_m": r["area_radius_m"],
            "seed": seed, "max_track_slots": r["max_track_slots"], "terrain": terrain_spec,
            "radar_defaults": r["radar"]["defaults"], "radars": radars, "targets": targets,
            "tracker": r["tracker"], "zones": zones,
        })
        group = terrain_group_id(sc["terrain"])
        route = "route-" + _digest(_physical_numbers([{k: t[k] for k in ("position", "velocity")} for t in targets]))[:20]
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
            "terrain_family": terrain_spec.get("family", "rolling_hills"), "terrain_kind": kind,
            "texture_family": terrain_spec.get("texture_family"),
            "obstacle_profile": terrain_spec.get("obstacle_profile", "open"),
            "route_pattern": pattern, "missile_count": len(missiles), "salvo_profiles": [i["profile"] for i in infos],
            "target_zones": [i["zone"] for i in infos], "threat_axes_deg": [i["approach_deg"] for i in infos],
            "lead_in_s": [i["lead_in_s"] for i in infos], "zones": zones,
            "planning_attempts": len(errors) + 1, "profile_substitutions": substituted, "planning_rejections": [e[:120] for e in errors],
            "radar_count": n_radars, "target_count": len(targets),
            "target_classes": [t["target_class"] for t in targets], "target_variants": [t["variant"] for t in targets],
            "class_labels": [t["label"] for t in targets], "maneuvers": [i["maneuvers"] for i in infos],
            "planned_visible_fraction": [round(s, 3) for s in shares],
            "resolved_scenario": sc, "config_snapshot": snapshot,
            "resolved_configuration_hash": config_hash, "recipe_version": RECIPE_VERSION,
            "generation_status": "planned",
        })
    return _json_value({
        "plan_schema_version": 1, "recipe_version": RECIPE_VERSION, "master_seed": master, "recipe": r,
        "split_ratios": dict(zip(SPLITS, map(float, ratios))), "split_counts": dict(zip(SPLITS, split_counts)),
        "split_allocation": "prefix-stable largest-deficit allocation",
        "label_space": {c: TARGET_CLASSES[c]["label"] for c in TARGET_CLASSES},
        "geometry_validation": "every missile propagated against terrain and its variant envelope after "
                               "time-on-target alignment; radar layout accepted only if every target meets "
                               "its visible share",
        "grouping_policy": "one terrain and route set per episode; groups disjoint across splits",
        "fixed_dimensions": "dt and max_track_slots fixed; radar count, target count and duration vary",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    })
