"""Coordinate frames and the ideal radar measurement function.

References
----------
[PN]  "Physics-Informed Simulation of Three-Dimensional Radar Detections",
      capstone technical note (Oct 2026) -- Sections 2-3, Eqs. (4)-(14).

Conventions
-----------
* World frame F_W is local ENU in metres: x = east, y = north, z = up.
* A radar's heading psi is the azimuth of its boresight measured
  counter-clockwise from +x (east), in radians.
* Sensor frame F_R follows [PN] Eq. (7): x forward along boresight,
  y to the right, z up.  Azimuth alpha = atan2(y, x) is therefore positive
  to the right of boresight (clockwise seen from above), elevation theta is
  positive upwards.
"""
from __future__ import annotations

import numpy as np

C_LIGHT = 299_792_458.0  # m/s


def wrap_angle(a):
    """Wrap an angle (or array of angles) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def sensor_basis(heading: float) -> np.ndarray:
    """Matrix whose columns are the sensor axes (forward, right, up) in F_W.

    It plays the role of R_WR in [PN] Eqs. (5)-(6): p_rel = R_WR^T (p_T - p_R).
    """
    c, s = np.cos(heading), np.sin(heading)
    forward = np.array([c, s, 0.0])
    right = np.array([s, -c, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.column_stack([forward, right, up])


def world_to_sensor(p_target, v_target, p_radar, heading, v_radar=None):
    """Relative position / velocity of a target in the radar frame, Eqs. (5)-(6)."""
    R = sensor_basis(heading)
    v_radar = np.zeros(3) if v_radar is None else np.asarray(v_radar, float)
    p_rel = R.T @ (np.asarray(p_target, float) - np.asarray(p_radar, float))
    v_rel = R.T @ (np.asarray(v_target, float) - v_radar)
    return p_rel, v_rel


def spherical_from_sensor(p_rel, v_rel):
    """Ideal measurement h(x) = [r, alpha, theta, r_dot], Eqs. (8)-(14)."""
    x, y, z = p_rel
    rho = np.hypot(x, y)
    r = np.sqrt(rho * rho + z * z)
    alpha = np.arctan2(y, x)                 # Eq. (9)
    theta = np.arctan2(z, rho)               # Eq. (10)
    r_dot = float(np.dot(p_rel, v_rel) / max(r, 1e-9))  # Eq. (12)
    return np.array([r, alpha, theta, r_dot])


def measure(p_target, v_target, p_radar, heading):
    """Convenience: world-frame state -> ideal [r, alpha, theta, r_dot]."""
    p_rel, v_rel = world_to_sensor(p_target, v_target, p_radar, heading)
    return spherical_from_sensor(p_rel, v_rel)


def doppler_shift(r_dot: float, wavelength: float) -> float:
    """Monostatic Doppler frequency, Eq. (13).

    Sign convention: an approaching target (r_dot < 0) gives a positive shift.
    """
    return -2.0 * r_dot / wavelength


def direction_from_angles(alpha, theta) -> np.ndarray:
    """Unit vector in the sensor frame for azimuth alpha and elevation theta."""
    return np.array([np.cos(theta) * np.cos(alpha),
                     np.cos(theta) * np.sin(alpha),
                     np.sin(theta)])


def sensor_point_to_world(r, alpha, theta, p_radar, heading) -> np.ndarray:
    """Inverse of the position part of h(x): spherical (sensor) -> world point."""
    R = sensor_basis(heading)
    return np.asarray(p_radar, float) + R @ (r * direction_from_angles(alpha, theta))


def measurement_jacobian(p_target, v_target, p_radar, heading) -> np.ndarray:
    """Analytic 4x6 Jacobian of h(x) w.r.t. the world state [p, v].

    Used by the EKF.  Derived by differentiating Eqs. (8)-(12) in the
    sensor frame and chaining with d p_rel / d p_W = R_WR^T.
    """
    R = sensor_basis(heading)
    p_rel, v_rel = world_to_sensor(p_target, v_target, p_radar, heading)
    x, y, z = p_rel
    rho2 = max(x * x + y * y, 1e-9)
    rho = np.sqrt(rho2)
    r2 = rho2 + z * z
    r = np.sqrt(r2)
    pv = float(p_rel @ v_rel)

    H_s = np.zeros((4, 6))
    H_s[0, :3] = p_rel / r
    H_s[1, :3] = [-y / rho2, x / rho2, 0.0]
    H_s[2, :3] = [-x * z / (r2 * rho), -y * z / (r2 * rho), rho / r2]
    H_s[3, :3] = v_rel / r - pv * p_rel / r ** 3
    H_s[3, 3:] = p_rel / r

    H = np.zeros((4, 6))
    H[:, :3] = H_s[:, :3] @ R.T
    H[:, 3:] = H_s[:, 3:] @ R.T
    return H


def body_rotation(velocity, lateral_accel: float = 0.0, g: float = 9.81) -> np.ndarray:
    """Approximate target body-to-world rotation R_WT (columns = body x, y, z).

    Body x follows the velocity vector (yaw + flight-path angle); the bank
    angle mu = atan(u / g) of a coordinated turn (Alpdemir 2022, Eq. 12)
    rolls the body about its x axis.
    """
    v = np.asarray(velocity, float)
    speed = np.linalg.norm(v)
    if speed < 1e-6:
        return np.eye(3)
    bx = v / speed
    up = np.array([0.0, 0.0, 1.0])
    by = np.cross(up, bx)
    if np.linalg.norm(by) < 1e-6:          # flying straight up/down
        by = np.array([0.0, 1.0, 0.0])
    by /= np.linalg.norm(by)
    bz = np.cross(bx, by)
    mu = np.arctan2(lateral_accel, g)
    c, s = np.cos(mu), np.sin(mu)
    by_r = c * by + s * bz
    bz_r = -s * by + c * bz
    return np.column_stack([bx, by_r, bz_r])
