"""Extended Kalman filter for a 3-D constant-velocity target observed in
radar spherical coordinates [r, alpha, theta, r_dot].

Written from the standard EKF equations (e.g. Bar-Shalom, Li & Kirubarajan,
"Estimation with Applications to Tracking and Navigation", 2001).  The
measurement function and its Jacobian come from src/core/geometry.py
(physics note Eqs. (8)-(14)).
"""
from __future__ import annotations

import numpy as np

from src.core import geometry as geo


def cv_transition(dt: float) -> np.ndarray:
    F = np.eye(6)
    F[:3, 3:] = dt * np.eye(3)
    return F


def cv_process_noise(dt: float, q: float) -> np.ndarray:
    """Discretised continuous white-noise-acceleration model with PSD q."""
    I = np.eye(3)
    return q * np.block([[dt ** 3 / 3 * I, dt ** 2 / 2 * I],
                         [dt ** 2 / 2 * I, dt * I]])


class EKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray, q: float = 40.0):
        self.x = np.asarray(x0, float).copy()
        self.P = np.asarray(P0, float).copy()
        self.q = q

    # -- prediction ---------------------------------------------------------
    def predict(self, dt: float):
        F = cv_transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + cv_process_noise(dt, self.q)

    # -- update -------------------------------------------------------------
    def innovation(self, z, R, radar_pos, heading):
        h = geo.measure(self.x[:3], self.x[3:], radar_pos, heading)
        H = geo.measurement_jacobian(self.x[:3], self.x[3:], radar_pos, heading)
        nu = np.asarray(z, float) - h
        nu[1] = geo.wrap_angle(nu[1])
        S = H @ self.P @ H.T + R
        return nu, S, H

    def mahalanobis2(self, z, R, radar_pos, heading) -> float:
        nu, S, _ = self.innovation(z, R, radar_pos, heading)
        return float(nu @ np.linalg.solve(S, nu))

    def update(self, z, R, radar_pos, heading):
        nu, S, H = self.innovation(z, R, radar_pos, heading)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ nu
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)

    # -- helpers ------------------------------------------------------------
    @property
    def pos_std(self) -> float:
        """RMS position standard deviation sqrt(trace(P_pos) / 3)."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))

    @classmethod
    def from_detection(cls, det: dict, radar_pos, heading, q: float = 40.0,
                       v_std: float = 120.0) -> "EKF":
        """One-point initialisation: position from the spherical report, the
        radial velocity along the line of sight, unknown tangential velocity."""
        r, a, e, rdot = det["z"]
        p = geo.sensor_point_to_world(r, a, e, radar_pos, heading)
        u = (p - np.asarray(radar_pos, float)) / max(r, 1e-6)
        v = rdot * u
        # position covariance: map the spherical sigmas through the inverse transform
        sr, sa, se, _ = np.sqrt(np.diag(det["measurement_covariance"]))
        R = geo.sensor_basis(heading)
        d_r = geo.direction_from_angles(a, e)
        d_a = np.array([-np.cos(e) * np.sin(a), np.cos(e) * np.cos(a), 0.0])
        d_e = np.array([-np.sin(e) * np.cos(a), -np.sin(e) * np.sin(a), np.cos(e)])
        J = R @ np.column_stack([d_r, r * d_a, r * d_e])
        P_pos = J @ np.diag([sr ** 2, sa ** 2, se ** 2]) @ J.T + np.eye(3) * 1.0
        P_vel = v_std ** 2 * (np.eye(3) - np.outer(u, u)) + 25.0 * np.outer(u, u)
        P0 = np.zeros((6, 6))
        P0[:3, :3] = P_pos
        P0[3:, 3:] = P_vel + 1.0 * np.eye(3)
        return cls(np.concatenate([p, v]), P0, q)
