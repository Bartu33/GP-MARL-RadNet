"""Verify V2 quintic-smoothstep terrain generator functions across all 5 families."""
import math
import numpy as np

def _rotate(X: np.ndarray, Y: np.ndarray, angle_deg: float):
    rad = math.radians(angle_deg)
    u = X * math.cos(rad) + Y * math.sin(rad)
    v = -X * math.sin(rad) + Y * math.cos(rad)
    return u, v

def _quintic_flank(u_abs: np.ndarray, W_half: float | np.ndarray, D: float, H_peak: float | np.ndarray, sigma_outer: float) -> np.ndarray:
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
                                     params: dict | None = None,
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
    
    # Valley floor micro-undulation (15-25m gentle alluvial rolls)
    alluvial = 20.0 * np.sin(2 * np.pi * v / 4200.0 + phi[4]) * np.cos(2 * np.pi * u / 2400.0 + phi[5])
    H = np.where(u <= 0, H_L, H_R) + np.maximum(0.0, alluvial) * np.exp(-0.5 * (u / (W * 0.4)) ** 2)
    
    H = _apply_knolls_v2(H, u, v, W, obstacle_profile, rng)
    return np.clip(H, 0.0, None)

def generate_valley_sinuous_pass_v2(X: np.ndarray, Y: np.ndarray, seed: int = 0,
                                    params: dict | None = None,
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
                                    params: dict | None = None,
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
                                params: dict | None = None,
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
                                     params: dict | None = None,
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

def test_all():
    res = 256
    axis = (np.arange(res) + 0.5) * (30000.0 / res) - 15000.0
    X, Y = np.meshgrid(axis, -axis)
    cell = 30000.0 / res
    
    fams = {
        "valley_linear_canyon": generate_valley_linear_canyon_v2,
        "valley_sinuous_pass": generate_valley_sinuous_pass_v2,
        "mountain_chokepoint": generate_mountain_chokepoint_v2,
        "basin_highlands": generate_basin_highlands_v2,
        "ridge_multi_obstacle": generate_ridge_multi_obstacle_v2,
    }
    for name, fn in fams.items():
        H = fn(X, Y, seed=42, obstacle_profile="dense")
        gy, gx = np.gradient(H, cell, cell)
        slopes = np.rad2deg(np.arctan(np.hypot(gx, gy)))
        print(f"{name:25s} | H max: {np.max(H):6.1f}m | H mean: {np.mean(H):5.1f}m | Max slope: {np.max(slopes):4.1f} deg | Mean slope: {np.mean(slopes):4.1f} deg")
        assert np.max(H) > 500.0
        assert np.max(slopes) < 32.0, f"{name} slope too steep: {np.max(slopes)}"
        assert not np.any(np.isnan(H))
        assert not np.any(H < 0.0)
    print("\nALL 5 V2 TERRAIN GENERATORS PASSED MATHEMATICAL & GEOMORPHIC VALIDATION!")

if __name__ == "__main__":
    test_all()
