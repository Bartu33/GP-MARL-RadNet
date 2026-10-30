"""Radar cross section (RCS) models.

References
----------
[PN]  Physics-informed radar detection note, Section 4, Eqs. (15)-(22).
[AL]  M. N. Alpdemir, "Tactical UAV path optimization under radar threat
      using deep reinforcement learning", Neural Computing and Applications,
      2022 -- Eq. (14): three-radius (ellipsoid) RCS in the optical region.

All models return the *mean* RCS in m^2 for an incidence direction given in
the target body frame.  Fluctuations are applied separately with
:func:`sample_swerling`.  Every parameter used in the scenarios is synthetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def dbsm_to_m2(sigma_dbsm):
    """Eq. (19)."""
    return 10.0 ** (np.asarray(sigma_dbsm, float) / 10.0)


def m2_to_dbsm(sigma_m2):
    return 10.0 * np.log10(np.maximum(np.asarray(sigma_m2, float), 1e-30))


def aspect_angles(u_body) -> tuple[float, float]:
    """Aspect azimuth psi and aspect elevation theta of a body-frame unit vector."""
    ux, uy, uz = u_body
    return float(np.arctan2(uy, ux)), float(np.arctan2(uz, np.hypot(ux, uy)))


def incidence_in_body(p_radar, p_target, R_WT) -> np.ndarray:
    """Unit line of sight target -> radar expressed in the body frame, Eqs. (15)-(16)."""
    u_w = np.asarray(p_radar, float) - np.asarray(p_target, float)
    u_w /= max(np.linalg.norm(u_w), 1e-9)
    return R_WT.T @ u_w


class RCSModel:
    name = "base"

    def mean_rcs(self, u_body) -> float:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class ConstantRCS(RCSModel):
    """Isotropic target: the same RCS from every aspect."""
    sigma_m2: float = 1.0
    name: str = "constant"

    def mean_rcs(self, u_body) -> float:
        return float(self.sigma_m2)


@dataclass
class EllipsoidRCS(RCSModel):
    """Three-radius geometric RCS of an ellipsoid with semi-axes a, b, c (m).

    For a unit incidence direction u = (u_x, u_y, u_z) in the body frame the
    optical-region RCS is

        sigma = pi a^2 b^2 c^2 / (a^2 u_x^2 + b^2 u_y^2 + c^2 u_z^2)^2 .

    [AL] Eq. (14) is this same expression written with the aspect, elevation
    and bank angles of the engagement geometry; the body-frame vector form is
    used here because the simulator already has R_WT available.
    """
    a: float = 1.0
    b: float = 0.5
    c: float = 0.3
    name: str = "ellipsoid"

    def mean_rcs(self, u_body) -> float:
        ux, uy, uz = u_body
        den = (self.a ** 2 * ux ** 2 + self.b ** 2 * uy ** 2 + self.c ** 2 * uz ** 2) ** 2
        return float(np.pi * self.a ** 2 * self.b ** 2 * self.c ** 2 / max(den, 1e-12))


@dataclass
class LookupTableRCS(RCSModel):
    """2-D table sigma_dBsm = T(psi, theta) with bilinear interpolation, Eqs. (18)-(19)."""
    psi_grid: np.ndarray
    theta_grid: np.ndarray
    table_dbsm: np.ndarray            # shape (len(psi_grid), len(theta_grid))
    name: str = "lookup"
    _interp: RegularGridInterpolator = field(init=False, repr=False)

    def __post_init__(self):
        self._interp = RegularGridInterpolator(
            (self.psi_grid, self.theta_grid), self.table_dbsm,
            bounds_error=False, fill_value=None)

    def mean_rcs(self, u_body) -> float:
        psi, theta = aspect_angles(u_body)
        return float(dbsm_to_m2(self._interp([[psi, theta]])[0]))

    @classmethod
    def synthetic(cls, nose_dbsm=-10.0, side_dbsm=5.0, tail_dbsm=-3.0,
                  top_dbsm=8.0, n_psi=73, n_theta=37) -> "LookupTableRCS":
        """A deliberately synthetic pattern with distinct nose / side / tail /
        top aspects, as suggested at the end of [PN] Section 4.2."""
        psi = np.linspace(-np.pi, np.pi, n_psi)
        theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
        P, T = np.meshgrid(psi, theta, indexing="ij")
        front = 0.5 * (1 + np.cos(P))           # 1 at the nose, 0 at the tail
        side = np.abs(np.sin(P))
        horiz = (front * nose_dbsm + (1 - front) * tail_dbsm) * (1 - side) + side * side_dbsm
        lobes = 1.5 * np.cos(6 * P) * np.cos(T)  # small ripple, purely illustrative
        table = (1 - np.abs(np.sin(T))) * (horiz + lobes) + np.abs(np.sin(T)) * top_dbsm
        return cls(psi, theta, table)


@dataclass
class ScatteringCentreRCS(RCSModel):
    """Coherent sum of M point scatterers, Eqs. (20)-(21).

    positions: (M, 3) body-frame scatterer locations d_i in metres.
    strengths: (M,) nominal RCS sigma_i in m^2.
    """
    positions: np.ndarray
    strengths: np.ndarray
    wavelength: float = 0.03
    name: str = "scattering"

    def mean_rcs(self, u_body) -> float:
        phase = -1j * 4.0 * np.pi / self.wavelength * (self.positions @ np.asarray(u_body))
        amp = np.sum(np.sqrt(self.strengths) * np.exp(phase))
        return float(np.abs(amp) ** 2)


def sample_swerling(mean_sigma: float, case: int, rng: np.random.Generator) -> float:
    """Instantaneous RCS sigma_k = mean * X_k with E[X_k] = 1, Eq. (22).

    case 0      -> non-fluctuating (X = 1)
    case 1 / 2  -> X ~ Exponential(1)              (chi-square, 2 d.o.f.)
    case 3 / 4  -> X ~ Gamma(shape=2, scale=1/2)   (chi-square, 4 d.o.f.)

    At this detection-level fidelity one draw is made per look, so cases 1/2
    (and 3/4) differ only in how often the caller re-draws.
    """
    if case == 0:
        return float(mean_sigma)
    if case in (1, 2):
        return float(mean_sigma * rng.exponential(1.0))
    if case in (3, 4):
        return float(mean_sigma * rng.gamma(2.0, 0.5))
    raise ValueError(f"unknown Swerling case {case}")


def build_rcs(spec: dict) -> RCSModel:
    """Create an RCS model from a scenario dictionary."""
    kind = spec.get("model", "constant")
    if kind == "constant":
        return ConstantRCS(spec.get("sigma_m2", 1.0))
    if kind == "ellipsoid":
        return EllipsoidRCS(spec.get("a", 1.0), spec.get("b", 0.5), spec.get("c", 0.3))
    if kind == "lookup":
        return LookupTableRCS.synthetic(**spec.get("params", {}))
    if kind == "scattering":
        return ScatteringCentreRCS(np.asarray(spec["positions"], float),
                                   np.asarray(spec["strengths"], float),
                                   spec.get("wavelength", 0.03))
    raise ValueError(f"unknown RCS model '{kind}'")
