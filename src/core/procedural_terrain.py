"""Procedural valley terrain generator for GP-MARL-RadNet.

Supports diverse geomorphic families with continuous ENU height fields,
parametric ridges, corridors, bottlenecks, basins, and physical obstacle profiles:
- valley_linear_canyon: Parallel mountain ranges flanking a straight valley corridor.
- valley_sinuous_pass: Winding S-shaped valley flanked by curving mountain ranges.
- mountain_chokepoint: Pass constricted between massive towering mountain shoulders.
- basin_highlands: High mountain rim surrounding an expansive amphitheater basin with entry/exit passes.
- ridge_multi_obstacle: Valley corridor featuring transverse spur ridges and isolated knolls.

Physical obstacle profiles:
- clear: Smooth corridor without additional local knolls. Zero decorative 3D objects.
- sparse: Few (4-6) physical knolls in the corridor, plus sparse decorative 3D objects.
- dense: Many (10-16) physical knolls creating radar shadows, plus dense decorative 3D objects.
- asymmetric: Left mountain wall is markedly taller and steeper than right wall.

Texture families (6 styles):
- rocky_mountain: High rugged slate granite peaks, talus slopes, alpine gravel.
- green_forest_valley: Lush forest green valley floor, pine slopes, summit crags.
- dry_canyon: Sandstone, terracotta red-rock strata, pale arid silt valley floor.
- snowy_highlands: Pure snow/ice on high peaks, frost slopes, cold slate valley floor.
- grassland_valley: Savanna/grassland valley floor, golden prairie foothills, weathered crests.
- mixed_rock_vegetation: Stepped transitions: riparian green -> moss/rock mosaic -> granite cliffs.

Heights are strictly non-negative (H >= 0.0), with the valley floor at datum 0m.
Integrates directly with Terrain, RadarNetEnv, line_of_sight, and ScenarioPlanner.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

SUPPORTED_TERRAIN_FAMILIES = (
    "valley_linear_canyon",
    "valley_sinuous_pass",
    "mountain_chokepoint",
    "basin_highlands",
    "ridge_multi_obstacle",
)

SUPPORTED_OBSTACLE_PROFILES = ("clear", "sparse", "dense", "asymmetric")

SUPPORTED_TEXTURE_FAMILIES = (
    "rocky_mountain",
    "green_forest_valley",
    "dry_canyon",
    "snowy_highlands",
    "grassland_valley",
    "mixed_rock_vegetation",
)


def _rotate(X: np.ndarray, Y: np.ndarray, angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    rad = math.radians(angle_deg)
    u = X * math.cos(rad) + Y * math.sin(rad)
    v = -X * math.sin(rad) + Y * math.cos(rad)
    return u, v


def _apply_knolls(H: np.ndarray, u: np.ndarray, v: np.ndarray, W: float, profile: str, rng: np.random.Generator) -> np.ndarray:
    if profile in ("clear", "asymmetric"):
        return H
    n_knolls = rng.integers(4, 7) if profile == "sparse" else rng.integers(10, 16)
    for _ in range(n_knolls):
        cu = rng.uniform(-W * 0.35, W * 0.35)
        cv = rng.uniform(-7500.0, 7500.0)
        amp = rng.uniform(110.0, 220.0) if profile == "sparse" else rng.uniform(150.0, 320.0)
        sigma = rng.uniform(250.0, 450.0)
        H = H + amp * np.exp(-0.5 * (((u - cu) / sigma) ** 2 + ((v - cv) / sigma) ** 2))
    return H


def generate_valley_linear_canyon(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                  params: Dict[str, Any] | None = None,
                                  obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 3000.0))
    H_relief = float(params.get("relief_height_m", 850.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_w = float(params.get("ridge_width_m", 2500.0))

    asym = 1.0
    if obstacle_profile == "asymmetric" or params.get("asymmetry_ratio"):
        asym = float(params.get("asymmetry_ratio", 1.45))

    u, v = _rotate(X, Y, angle_deg)
    h_L = H_relief * asym
    h_R = H_relief / asym
    sigma_L = ridge_w * (0.8 if asym > 1.0 else 1.0)
    sigma_R = ridge_w * (1.2 if asym > 1.0 else 1.0)
    D_L = W * 0.5 + sigma_L * 1.1
    D_R = W * 0.5 + sigma_R * 1.1

    phi = rng.uniform(0, 2 * np.pi, 6)
    mod_L = 1.0 + 0.22 * np.sin(2 * np.pi * v / 12000.0 + phi[0]) + 0.12 * np.cos(2 * np.pi * v / 5500.0 + phi[1])
    mod_R = 1.0 + 0.22 * np.cos(2 * np.pi * v / 12000.0 + phi[2]) + 0.12 * np.sin(2 * np.pi * v / 5500.0 + phi[3])

    inner_L = np.exp(-0.5 * ((u + D_L) / sigma_L) ** 2)
    outer_L = 0.5 * np.exp(-0.5 * ((u + D_L) / (2.2 * sigma_L)) ** 2) + 0.5
    H_L = h_L * mod_L * np.where(u > -D_L, inner_L, outer_L)

    inner_R = np.exp(-0.5 * ((u - D_R) / sigma_R) ** 2)
    outer_R = 0.5 * np.exp(-0.5 * ((u - D_R) / (2.2 * sigma_R)) ** 2) + 0.5
    H_R = h_R * mod_R * np.where(u < D_R, inner_R, outer_R)

    mask = np.clip((np.abs(u) - W * 0.35) / (W * 0.45), 0.0, 1.0) ** 1.5
    H = (np.where(u <= 0, H_L, 0.0) + np.where(u > 0, H_R, 0.0)) * mask
    H = _apply_knolls(H, u, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_valley_sinuous_pass(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                 params: Dict[str, Any] | None = None,
                                 obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 2800.0))
    H_relief = float(params.get("relief_height_m", 850.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    A_sin = float(params.get("sinuosity_amplitude_m", 2400.0))
    L_sin = float(params.get("sinuosity_wavelength_m", 14000.0))
    ridge_w = float(params.get("ridge_width_m", 2500.0))

    asym = 1.0
    if obstacle_profile == "asymmetric" or params.get("asymmetry_ratio"):
        asym = float(params.get("asymmetry_ratio", 1.45))

    u, v = _rotate(X, Y, angle_deg)
    phi = rng.uniform(0, 2 * np.pi, 4)
    u_c = A_sin * np.sin(2 * np.pi * v / L_sin + phi[0]) + 0.35 * A_sin * np.cos(4 * np.pi * v / L_sin + phi[1])
    u_rel = u - u_c

    h_L = H_relief * asym
    h_R = H_relief / asym
    sigma_L = ridge_w * (0.8 if asym > 1.0 else 1.0)
    sigma_R = ridge_w * (1.2 if asym > 1.0 else 1.0)
    D_L = W * 0.5 + sigma_L * 1.1
    D_R = W * 0.5 + sigma_R * 1.1

    mod_L = 1.0 + 0.20 * np.sin(2 * np.pi * v / 11000.0 + phi[2])
    mod_R = 1.0 + 0.20 * np.cos(2 * np.pi * v / 11000.0 + phi[3])

    inner_L = np.exp(-0.5 * ((u_rel + D_L) / sigma_L) ** 2)
    outer_L = 0.5 * np.exp(-0.5 * ((u_rel + D_L) / (2.2 * sigma_L)) ** 2) + 0.5
    H_L = h_L * mod_L * np.where(u_rel > -D_L, inner_L, outer_L)

    inner_R = np.exp(-0.5 * ((u_rel - D_R) / sigma_R) ** 2)
    outer_R = 0.5 * np.exp(-0.5 * ((u_rel - D_R) / (2.2 * sigma_R)) ** 2) + 0.5
    H_R = h_R * mod_R * np.where(u_rel < D_R, inner_R, outer_R)

    mask = np.clip((np.abs(u_rel) - W * 0.35) / (W * 0.45), 0.0, 1.0) ** 1.5
    H = (np.where(u_rel <= 0, H_L, 0.0) + np.where(u_rel > 0, H_R, 0.0)) * mask
    H = _apply_knolls(H, u_rel, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_mountain_chokepoint(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                 params: Dict[str, Any] | None = None,
                                 obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W_choke = float(params.get("chokepoint_width_m", 1600.0))
    W_open = float(params.get("open_width_m", 5500.0))
    H_relief = float(params.get("relief_height_m", 850.0))
    sigma_v = float(params.get("chokepoint_length_m", 3500.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_w = float(params.get("ridge_width_m", 2500.0))

    asym = 1.0
    if obstacle_profile == "asymmetric" or params.get("asymmetry_ratio"):
        asym = float(params.get("asymmetry_ratio", 1.45))

    u, v = _rotate(X, Y, angle_deg)
    v0 = rng.uniform(-800.0, 800.0)

    W = W_choke + (W_open - W_choke) * (1.0 - np.exp(-0.5 * ((v - v0) / sigma_v) ** 2))
    H_peak_boost = 320.0 * np.exp(-0.5 * ((v - v0) / (1.2 * sigma_v)) ** 2)

    h_L = (H_relief + H_peak_boost) * asym
    h_R = (H_relief + H_peak_boost) / asym
    sigma_L = ridge_w * (0.8 if asym > 1.0 else 1.0)
    sigma_R = ridge_w * (1.2 if asym > 1.0 else 1.0)
    D_L = W * 0.5 + sigma_L * 1.1
    D_R = W * 0.5 + sigma_R * 1.1

    inner_L = np.exp(-0.5 * ((u + D_L) / sigma_L) ** 2)
    outer_L = 0.5 * np.exp(-0.5 * ((u + D_L) / (2.2 * sigma_L)) ** 2) + 0.5
    H_L = h_L * np.where(u > -D_L, inner_L, outer_L)

    inner_R = np.exp(-0.5 * ((u - D_R) / sigma_R) ** 2)
    outer_R = 0.5 * np.exp(-0.5 * ((u - D_R) / (2.2 * sigma_R)) ** 2) + 0.5
    H_R = h_R * np.where(u < D_R, inner_R, outer_R)

    mask = np.clip((np.abs(u) - W * 0.35) / (W * 0.45), 0.0, 1.0) ** 1.5
    H = (np.where(u <= 0, H_L, 0.0) + np.where(u > 0, H_R, 0.0)) * mask
    H = _apply_knolls(H, u, v, W_choke, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_basin_highlands(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                             params: Dict[str, Any] | None = None,
                             obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    R_basin = float(params.get("basin_radius_m", 4500.0))
    H_relief = float(params.get("relief_height_m", 850.0))
    aspect = float(params.get("aspect_ratio", 1.15))
    sigma_rim = float(params.get("rim_width_m", 2200.0))
    num_gaps = int(params.get("num_gaps", 2))

    asym = 1.0
    if obstacle_profile == "asymmetric" or params.get("asymmetry_ratio"):
        asym = float(params.get("asymmetry_ratio", 1.45))

    u, v = _rotate(X, Y, float(params.get("valley_angle_deg", 0.0)))
    r_ell = np.sqrt(u ** 2 + (v / aspect) ** 2)
    theta = np.arctan2(v, u)

    inner_rim = np.exp(-0.5 * ((r_ell - R_basin) / sigma_rim) ** 2)
    outer_rim = 0.5 * np.exp(-0.5 * ((r_ell - R_basin) / (2.2 * sigma_rim)) ** 2) + 0.5
    H_rim = H_relief * np.where(r_ell < R_basin, inner_rim, outer_rim)

    if asym != 1.0:
        H_rim = H_rim * np.where(u < 0, asym, 1.0 / asym)

    mask = np.clip((r_ell - (R_basin - 1.0 * sigma_rim)) / (0.7 * sigma_rim), 0.0, 1.0) ** 1.5

    gap_angles = [float(k * 2 * np.pi / num_gaps + rng.uniform(-0.3, 0.3)) for k in range(num_gaps)]
    gap_mask = np.ones_like(H_rim)
    for g_ang in gap_angles:
        d_theta = np.abs((theta - g_ang + np.pi) % (2 * np.pi) - np.pi)
        gap_mask *= (1.0 - np.exp(-0.5 * (d_theta / 0.28) ** 2))

    H = H_rim * mask * gap_mask
    H = _apply_knolls(H, u, v, R_basin * 0.8, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_ridge_multi_obstacle(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                  params: Dict[str, Any] | None = None,
                                  obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 4200.0))
    H_relief = float(params.get("relief_height_m", 900.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_w = float(params.get("ridge_width_m", 2500.0))
    num_spurs = int(params.get("num_spurs", 6))

    u, v = _rotate(X, Y, angle_deg)
    sigma_L = ridge_w
    sigma_R = ridge_w
    D_L = W * 0.5 + sigma_L * 1.1
    D_R = W * 0.5 + sigma_R * 1.1

    inner_L = np.exp(-0.5 * ((u + D_L) / sigma_L) ** 2)
    outer_L = 0.5 * np.exp(-0.5 * ((u + D_L) / (2.2 * sigma_L)) ** 2) + 0.5
    H_L = H_relief * np.where(u > -D_L, inner_L, outer_L)

    inner_R = np.exp(-0.5 * ((u - D_R) / sigma_R) ** 2)
    outer_R = 0.5 * np.exp(-0.5 * ((u - D_R) / (2.2 * sigma_R)) ** 2) + 0.5
    H_R = H_relief * np.where(u < D_R, inner_R, outer_R)

    mask = np.clip((np.abs(u) - W * 0.35) / (W * 0.45), 0.0, 1.0) ** 1.5
    H = (np.where(u <= 0, H_L, 0.0) + np.where(u > 0, H_R, 0.0)) * mask

    v_spurs = np.linspace(-6500.0, 6500.0, num_spurs)
    for i, v_sp in enumerate(v_spurs):
        side = -1.0 if (i % 2 == 0) else 1.0
        u_origin = side * W * 0.5
        u_target = side * W * 0.15
        h_spur = rng.uniform(280.0, 420.0)
        c_u = (u_origin + u_target) * 0.5
        c_v = v_sp + rng.uniform(-400.0, 400.0)
        spur_angle = math.radians(side * 25.0)
        u_prime = (u - c_u) * math.cos(spur_angle) + (v - c_v) * math.sin(spur_angle)
        v_prime = -(u - c_u) * math.sin(spur_angle) + (v - c_v) * math.cos(spur_angle)
        spur = h_spur * np.exp(-0.5 * ((u_prime / 900.0) ** 2 + (v_prime / 450.0) ** 2))
        H = np.maximum(H, spur)

    H = _apply_knolls(H, u, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_environment_objects(H: np.ndarray, x_west: float, y_north: float, cell: float,
                                 obstacle_profile: str = "clear", seed: int = 0,
                                 texture_family: str = "rocky_mountain") -> List[Dict[str, Any]]:
    """Generates deterministic 3D environment object instances placed on terrain surface.

    These are decorative objects (trees, rock boulders, shrubs) for visual scale
    and realism in the 3D viewer. They do NOT modify the continuous elevation matrix
    and do NOT introduce unmodelled electromagnetic attenuation into radar physics.
    """
    if obstacle_profile == "clear":
        return []

    rng = np.random.default_rng(seed + 777)
    rows, cols = H.shape

    if obstacle_profile == "sparse":
        n_targets = int(rng.integers(70, 120))
    elif obstacle_profile == "dense":
        n_targets = int(rng.integers(300, 500))
    elif obstacle_profile == "asymmetric":
        n_targets = int(rng.integers(180, 300))
    else:
        n_targets = 0

    if texture_family in ("green_forest_valley", "grassland_valley"):
        type_probs = [0.65, 0.15, 0.20]  # tree, shrub, rock
    elif texture_family in ("dry_canyon", "rocky_mountain"):
        type_probs = [0.10, 0.20, 0.70]
    elif texture_family == "snowy_highlands":
        type_probs = [0.20, 0.10, 0.70]
    else:  # mixed_rock_vegetation
        type_probs = [0.45, 0.25, 0.30]

    objects = []
    attempts = 0
    max_attempts = n_targets * 4

    while len(objects) < n_targets and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(-7500.0, 7500.0))
        y = float(rng.uniform(-7500.0, 7500.0))

        c = float(np.clip((x - x_west) / cell - 0.5, 0, cols - 1.001))
        r = float(np.clip((y_north - y) / cell - 0.5, 0, rows - 1.001))
        c0, r0 = int(c), int(r)
        fc, fr = c - c0, r - r0
        top = H[r0, c0] * (1.0 - fc) + H[r0, c0 + 1] * fc
        bot = H[r0 + 1, c0] * (1.0 - fc) + H[r0 + 1, c0 + 1] * fc
        h = float(top * (1.0 - fr) + bot * fr)

        if obstacle_profile == "asymmetric" and x > 1500.0 and rng.random() < 0.65:
            continue

        roll = float(rng.random())
        if roll < type_probs[0]:
            obj_type = "tree"
            if h > 650.0:
                continue
            scale = float(rng.uniform(0.7, 1.4))
        elif roll < type_probs[0] + type_probs[1]:
            obj_type = "shrub"
            if h > 750.0:
                continue
            scale = float(rng.uniform(0.6, 1.2))
        else:
            obj_type = "rock"
            scale = float(rng.uniform(0.6, 1.8))

        rot = float(rng.uniform(0.0, 2.0 * math.pi))
        objects.append({
            "type": obj_type,
            "x": round(x, 1),
            "y": round(y, 1),
            "h": round(h, 1),
            "scale": round(scale, 2),
            "rot": round(rot, 2),
        })

    return objects


# ===========================================================================
# V2 Natural Geomorphic Terrain Generators & Slope-Aware Object Placement (Step 4D)
# ===========================================================================

def _quintic_flank(u_abs: np.ndarray, W_half: float | np.ndarray, D: float,
                   H_peak: float | np.ndarray, sigma_outer: float) -> np.ndarray:
    """C1/C2 continuous mountain ridge profile with flat/gentle valley floor and outer falloff."""
    span = np.maximum(100.0, D - W_half)
    x = np.clip((u_abs - W_half) / span, 0.0, 1.0)
    # Quintic smoothstep: 10x^3 - 15x^4 + 6x^5
    s = x * x * x * (x * (x * 6.0 - 15.0) + 10.0)
    h_inner = H_peak * s

    dist_outer = np.maximum(0.0, u_abs - D)
    h_outer = H_peak * np.exp(-0.5 * (dist_outer / sigma_outer) ** 2)
    return np.where(u_abs < D, h_inner, h_outer)


def _apply_knolls_v2(H: np.ndarray, u: np.ndarray, v: np.ndarray, W: float, profile: str, rng: np.random.Generator) -> np.ndarray:
    """Natural, well-sampled rounded knolls preventing low-poly pyramid artifacts."""
    if profile in ("clear", "asymmetric"):
        return H
    n_knolls = 3 if profile == "sparse" else 6
    for _ in range(n_knolls):
        cu = rng.uniform(-W * 0.28, W * 0.28)
        cv = rng.uniform(-6000.0, 6000.0)
        amp = rng.uniform(70.0, 130.0) if profile == "sparse" else rng.uniform(100.0, 170.0)
        sigma = rng.uniform(750.0, 1150.0)
        knoll = amp * np.exp(-0.5 * (((u - cu) / sigma) ** 2 + ((v - cv) / sigma) ** 2))
        H += knoll
    return H


def generate_valley_linear_canyon_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                     params: Dict[str, Any] | None = None,
                                     obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 3200.0))
    H_relief = float(params.get("relief_height_m", 880.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_d = float(params.get("ridge_distance_m", 5600.0))
    sigma_outer = float(params.get("ridge_width_m", 3800.0))

    asym = float(params.get("asymmetry_ratio", 1.45)) if (obstacle_profile == "asymmetric" or params.get("asymmetry_ratio")) else 1.0
    u, v = _rotate(X, Y, angle_deg)

    phi = rng.uniform(0, 2 * np.pi, 6)
    mod_L = 1.0 + 0.16 * np.sin(2 * np.pi * v / 11000.0 + phi[0]) + 0.07 * np.cos(2 * np.pi * v / 4500.0 + phi[1])
    mod_R = 1.0 + 0.16 * np.cos(2 * np.pi * v / 11000.0 + phi[2]) + 0.07 * np.sin(2 * np.pi * v / 4500.0 + phi[3])

    h_L_peak = H_relief * asym * mod_L
    h_R_peak = (H_relief / asym) * mod_R
    D_L = ridge_d * (0.88 if asym > 1.0 else 1.0)
    D_R = ridge_d * (1.12 if asym > 1.0 else 1.0)

    H_L = _quintic_flank(np.maximum(0.0, -u), W * 0.45, D_L, h_L_peak, sigma_outer)
    H_R = _quintic_flank(np.maximum(0.0, u), W * 0.45, D_R, h_R_peak, sigma_outer)

    alluvial = 20.0 * np.sin(2 * np.pi * v / 4200.0 + phi[4]) * np.cos(2 * np.pi * u / 2400.0 + phi[5])
    H = np.where(u <= 0, H_L, H_R) + np.maximum(0.0, alluvial) * np.exp(-0.5 * (u / (W * 0.4)) ** 2)

    H = _apply_knolls_v2(H, u, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_valley_sinuous_pass_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                    params: Dict[str, Any] | None = None,
                                    obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 3000.0))
    H_relief = float(params.get("relief_height_m", 880.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    A_sin = float(params.get("sinuosity_amplitude_m", 1500.0))
    L_sin = float(params.get("sinuosity_wavelength_m", 15000.0))
    ridge_d = float(params.get("ridge_distance_m", 6000.0))
    sigma_outer = float(params.get("ridge_width_m", 4000.0))

    asym = float(params.get("asymmetry_ratio", 1.45)) if (obstacle_profile == "asymmetric" or params.get("asymmetry_ratio")) else 1.0
    u, v = _rotate(X, Y, angle_deg)
    phi = rng.uniform(0, 2 * np.pi, 6)

    u_c = A_sin * np.sin(2 * np.pi * v / L_sin + phi[0]) + 0.15 * A_sin * np.cos(4 * np.pi * v / L_sin + phi[1])
    u_rel = u - u_c

    mod_L = 1.0 + 0.16 * np.sin(2 * np.pi * v / 10500.0 + phi[2]) + 0.07 * np.cos(2 * np.pi * v / 4500.0 + phi[3])
    mod_R = 1.0 + 0.16 * np.cos(2 * np.pi * v / 10500.0 + phi[4]) + 0.07 * np.sin(2 * np.pi * v / 4500.0 + phi[5])

    h_L_peak = H_relief * asym * mod_L
    h_R_peak = (H_relief / asym) * mod_R
    D_L = ridge_d * (0.88 if asym > 1.0 else 1.0)
    D_R = ridge_d * (1.12 if asym > 1.0 else 1.0)

    H_L = _quintic_flank(np.maximum(0.0, -u_rel), W * 0.42, D_L, h_L_peak, sigma_outer)
    H_R = _quintic_flank(np.maximum(0.0, u_rel), W * 0.42, D_R, h_R_peak, sigma_outer)

    alluvial = 20.0 * np.sin(2 * np.pi * v / 4200.0 + phi[4])
    H = np.where(u_rel <= 0, H_L, H_R) + np.maximum(0.0, alluvial) * np.exp(-0.5 * (u_rel / (W * 0.4)) ** 2)

    H = _apply_knolls_v2(H, u_rel, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_mountain_chokepoint_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                    params: Dict[str, Any] | None = None,
                                    obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W_choke = float(params.get("chokepoint_width_m", 1800.0))
    W_open = float(params.get("open_width_m", 5500.0))
    H_relief = float(params.get("relief_height_m", 900.0))
    sigma_v = float(params.get("chokepoint_length_m", 3800.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_d = float(params.get("ridge_distance_m", 5600.0))
    sigma_outer = float(params.get("ridge_width_m", 3800.0))

    asym = float(params.get("asymmetry_ratio", 1.45)) if (obstacle_profile == "asymmetric" or params.get("asymmetry_ratio")) else 1.0
    u, v = _rotate(X, Y, angle_deg)
    v0 = rng.uniform(-500.0, 500.0)

    choke_factor = np.exp(-0.5 * ((v - v0) / sigma_v) ** 2)
    W = W_open - (W_open - W_choke) * choke_factor
    H_peak_boost = 240.0 * np.exp(-0.5 * ((v - v0) / (1.2 * sigma_v)) ** 2)

    h_L_peak = (H_relief + H_peak_boost) * asym
    h_R_peak = (H_relief + H_peak_boost) / asym
    D_L = ridge_d * (0.88 if asym > 1.0 else 1.0)
    D_R = ridge_d * (1.12 if asym > 1.0 else 1.0)

    H_L = _quintic_flank(np.maximum(0.0, -u), W * 0.42, D_L, h_L_peak, sigma_outer)
    H_R = _quintic_flank(np.maximum(0.0, u), W * 0.42, D_R, h_R_peak, sigma_outer)
    H = np.where(u <= 0, H_L, H_R)

    H = _apply_knolls_v2(H, u, v, W_choke, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_basin_highlands_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                params: Dict[str, Any] | None = None,
                                obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    R_basin = float(params.get("basin_radius_m", 4500.0))
    H_relief = float(params.get("relief_height_m", 900.0))
    aspect = float(params.get("aspect_ratio", 1.2))
    sigma_rim = float(params.get("rim_width_m", 3200.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))

    u, v = _rotate(X, Y, angle_deg)
    theta = np.arctan2(v, u)
    phi = rng.uniform(0, 2 * np.pi, 6)
    r_mod = 1.0 + 0.10 * np.cos(3 * theta + phi[0]) + 0.06 * np.sin(5 * theta + phi[1])
    r_ell = np.sqrt(u ** 2 + (v / aspect) ** 2) / r_mod

    basin_base = float(params.get("basin_base_m", 160.0))
    floor_undulation = 40.0 * np.sin(2 * np.pi * u / 3600.0 + phi[2]) * np.cos(2 * np.pi * v / 3200.0 + phi[3])
    H_floor = np.maximum(0.0, basin_base + floor_undulation)

    rim_elev_mod = 1.0 + 0.22 * np.sin(2 * theta + phi[4]) + 0.12 * np.cos(4 * theta + phi[5])
    h_rim_local = H_relief * rim_elev_mod

    dist_from_rim = r_ell - R_basin
    H_rim = h_rim_local * np.exp(-0.5 * (dist_from_rim / sigma_rim) ** 2)

    num_passes = int(params.get("num_passes", 2))
    pass_angles = [float(k * np.pi + rng.uniform(-0.25, 0.25)) for k in range(num_passes)]
    pass_mask = np.ones_like(H_rim)
    for p_ang in pass_angles:
        d_th = np.abs((theta - p_ang + np.pi) % (2 * np.pi) - np.pi)
        pass_mask *= (1.0 - 0.60 * np.exp(-0.5 * (d_th / 0.35) ** 2))

    H_rim = H_rim * pass_mask
    floor_blend = np.clip((R_basin - r_ell) / (sigma_rim * 0.75), 0.0, 1.0)
    smooth_b = floor_blend * floor_blend * (3.0 - 2.0 * floor_blend)
    H = H_floor * smooth_b + H_rim * (1.0 - smooth_b * 0.5)

    H = _apply_knolls_v2(H, u, v, R_basin * 0.8, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_ridge_multi_obstacle_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                     params: Dict[str, Any] | None = None,
                                     obstacle_profile: str = "clear") -> np.ndarray:
    params = params or {}
    rng = np.random.default_rng(seed)
    W = float(params.get("valley_width_m", 4200.0))
    H_relief = float(params.get("relief_height_m", 900.0))
    angle_deg = float(params.get("valley_angle_deg", 0.0))
    ridge_d = float(params.get("ridge_distance_m", 6000.0))
    sigma_outer = float(params.get("ridge_width_m", 3800.0))
    num_spurs = int(params.get("num_spurs", 5))

    u, v = _rotate(X, Y, angle_deg)
    phi = rng.uniform(0, 2 * np.pi, 6)

    mod_L = 1.0 + 0.15 * np.sin(2 * np.pi * v / 10000.0 + phi[0])
    mod_R = 1.0 + 0.15 * np.cos(2 * np.pi * v / 10000.0 + phi[1])
    h_L_peak = H_relief * mod_L
    h_R_peak = H_relief * mod_R

    H_L = _quintic_flank(np.maximum(0.0, -u), W * 0.40, ridge_d, h_L_peak, sigma_outer)
    H_R = _quintic_flank(np.maximum(0.0, u), W * 0.40, ridge_d, h_R_peak, sigma_outer)
    H = np.where(u <= 0, H_L, H_R)

    # Smooth buttress spur ridges extending into valley
    v_spurs = np.linspace(-5500.0, 5500.0, num_spurs)
    for i, v_sp in enumerate(v_spurs):
        side = -1.0 if (i % 2 == 0) else 1.0
        c_u = side * W * 0.42
        c_v = v_sp + rng.uniform(-250.0, 250.0)
        h_spur = rng.uniform(180.0, 280.0)
        spur_angle = math.radians(side * 20.0)
        u_p = (u - c_u) * math.cos(spur_angle) + (v - c_v) * math.sin(spur_angle)
        v_p = -(u - c_u) * math.sin(spur_angle) + (v - c_v) * math.cos(spur_angle)
        spur = h_spur * np.exp(-0.5 * ((u_p / 1400.0) ** 2 + (v_p / 800.0) ** 2))

        blend_factor = np.clip(1.0 - H / (H_relief + 1.0), 0.0, 1.0) ** 1.5
        H += spur * blend_factor

    H = _apply_knolls_v2(H, u, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)


def generate_environment_objects_v2(H: np.ndarray, x_west: float, y_north: float, cell: float,
                                    obstacle_profile: str = "clear", seed: int = 0,
                                    texture_family: str = "rocky_mountain") -> List[Dict[str, Any]]:
    """Generates slope-aware, biome-coherent decorative 3D environment objects (Step 4D V2)."""
    if obstacle_profile == "clear":
        return []

    rng = np.random.default_rng(seed + 777)
    rows, cols = H.shape

    # Precompute terrain slope matrix in degrees
    gy, gx = np.gradient(H, cell, cell)
    slopes_deg = np.rad2deg(np.arctan(np.hypot(gx, gy)))

    if obstacle_profile == "sparse":
        n_targets = int(rng.integers(60, 100))
    elif obstacle_profile == "dense":
        n_targets = int(rng.integers(240, 380))
    elif obstacle_profile == "asymmetric":
        n_targets = int(rng.integers(140, 220))
    else:
        n_targets = 0

    if texture_family in ("green_forest_valley", "grassland_valley"):
        type_probs = [0.65, 0.15, 0.20]  # tree, shrub, rock
    elif texture_family in ("dry_canyon", "rocky_mountain"):
        type_probs = [0.08, 0.18, 0.74]
    elif texture_family == "snowy_highlands":
        type_probs = [0.12, 0.10, 0.78]
    else:  # mixed_rock_vegetation
        type_probs = [0.45, 0.25, 0.30]

    objects = []
    attempts = 0
    max_attempts = n_targets * 6

    while len(objects) < n_targets and attempts < max_attempts:
        attempts += 1
        x = float(rng.uniform(-7500.0, 7500.0))
        y = float(rng.uniform(-7500.0, 7500.0))

        c = int(np.clip((x - x_west) / cell - 0.5, 0, cols - 1))
        r = int(np.clip((y_north - y) / cell - 0.5, 0, rows - 1))
        loc_h = float(H[r, c])
        loc_slope = float(slopes_deg[r, c])

        if obstacle_profile == "asymmetric" and x > 1500.0 and rng.random() < 0.65:
            continue

        roll = float(rng.random())
        if roll < type_probs[0]:
            # Tree: requires gentle slope (< 24 deg) and below tree line (< 600m)
            if loc_slope > 24.0 or loc_h > 600.0:
                continue
            if texture_family == "dry_canyon" and (loc_h > 180.0 or loc_slope > 14.0):
                continue
            if texture_family == "snowy_highlands" and (loc_h > 260.0 or loc_slope > 16.0):
                continue
            obj_type = "tree"
            scale = float(rng.uniform(0.7, 1.4))
        elif roll < type_probs[0] + type_probs[1]:
            # Shrub: gentle-to-moderate slope (< 28 deg), elevation < 720m
            if loc_slope > 28.0 or loc_h > 720.0:
                continue
            obj_type = "shrub"
            scale = float(rng.uniform(0.6, 1.2))
        else:
            # Rock: naturally on slopes (10 deg - 36 deg) or talus aprons
            if loc_slope < 8.0 and loc_h < 120.0 and rng.random() < 0.60:
                continue
            if loc_slope > 36.0:
                continue
            obj_type = "rock"
            scale = float(rng.uniform(0.6, 1.8))

        rot = float(rng.uniform(0.0, 2.0 * math.pi))
        objects.append({
            "type": obj_type,
            "x": round(x, 1),
            "y": round(y, 1),
            "h": round(loc_h, 1),
            "scale": round(scale, 2),
            "rot": round(rot, 2),
        })

    return objects


def generate_procedural_terrain_matrix(X: np.ndarray, Y: np.ndarray, family: str,
                                       seed: int = 0, params: Dict[str, Any] | None = None,
                                       obstacle_profile: str = "clear",
                                       generator_version: int = 1) -> np.ndarray:
    if family not in SUPPORTED_TERRAIN_FAMILIES:
        raise ValueError(f"unknown terrain family {family!r}; supported: {SUPPORTED_TERRAIN_FAMILIES}")
    if obstacle_profile not in SUPPORTED_OBSTACLE_PROFILES:
        raise ValueError(f"unknown obstacle profile {obstacle_profile!r}; supported: {SUPPORTED_OBSTACLE_PROFILES}")

    eff_version = generator_version
    if params and ("generator_version" in params or "version" in params):
        eff_version = int(params.get("generator_version", params.get("version", generator_version)))

    if eff_version == 2:
        generators_v2 = {
            "valley_linear_canyon": generate_valley_linear_canyon_v2,
            "valley_sinuous_pass": generate_valley_sinuous_pass_v2,
            "mountain_chokepoint": generate_mountain_chokepoint_v2,
            "basin_highlands": generate_basin_highlands_v2,
            "ridge_multi_obstacle": generate_ridge_multi_obstacle_v2,
        }
        return generators_v2[family](X, Y, seed=seed, params=params, obstacle_profile=obstacle_profile)

    generators = {
        "valley_linear_canyon": generate_valley_linear_canyon,
        "valley_sinuous_pass": generate_valley_sinuous_pass,
        "mountain_chokepoint": generate_mountain_chokepoint,
        "basin_highlands": generate_basin_highlands,
        "ridge_multi_obstacle": generate_ridge_multi_obstacle,
    }
    return generators[family](X, Y, seed=seed, params=params, obstacle_profile=obstacle_profile)


def build_procedural_valley_terrain(lat: float, lon: float, spec: dict,
                                    seed: int = 0, resolution: int = 512) -> Any:
    """Builds a continuous procedural Terrain instance for a geomorphic family."""
    from .terrain import _grid_extent, Terrain

    res = spec.get("resolution", resolution)
    family = spec["family"]
    obstacle_profile = spec.get("obstacle_profile", "clear")
    texture_family = spec.get("texture_family", "rocky_mountain")
    params = dict(spec.get("parameters", {}))
    for key in ("valley_width_m", "relief_height_m", "valley_angle_deg", "ridge_width_m", "generator_version"):
        if key in spec and key not in params:
            params[key] = spec[key]

    eff_gen_ver = int(spec.get("generator_version", spec.get("version", params.get("generator_version", 1))))
    eff_seed = int(spec.get("seed", seed))
    x_west, y_north, size = _grid_extent(lat, lon)
    cell = size / res
    axis = (np.arange(res) + 0.5) * cell
    X, Y = np.meshgrid(x_west + axis, y_north - axis)

    H = generate_procedural_terrain_matrix(X, Y, family, seed=eff_seed, params=params,
                                          obstacle_profile=obstacle_profile,
                                          generator_version=eff_gen_ver)

    if eff_gen_ver == 2:
        visual_objects = generate_environment_objects_v2(H, x_west, y_north, cell,
                                                         obstacle_profile=obstacle_profile,
                                                         seed=eff_seed,
                                                         texture_family=texture_family)
    else:
        visual_objects = generate_environment_objects(H, x_west, y_north, cell,
                                                      obstacle_profile=obstacle_profile,
                                                      seed=eff_seed,
                                                      texture_family=texture_family)

    terrain = Terrain(H, x_west, y_north, cell, source="procedural", base_msl=0.0)
    terrain.family = family
    terrain.terrain_family = family
    terrain.texture_family = texture_family
    terrain.obstacle_profile = obstacle_profile
    terrain.terrain_seed = eff_seed
    terrain.visual_objects = visual_objects
    terrain.generator_version = eff_gen_ver
    terrain.physical_vs_visual = {
        "physical_radar_occlusion": "Terrain.H",
        "visual_decorative_objects": len(visual_objects),
        "visual_rf_attenuation_modelled": False,
    }
    terrain.valley_width_m = spec.get("valley_width_m")
    terrain.relief_height_m = spec.get("relief_height_m")
    terrain.terrain_parameters = params
    return terrain
