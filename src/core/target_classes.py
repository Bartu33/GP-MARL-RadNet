"""Physically grounded target classes: UAV, aircraft and missile.

Single source of truth for the three target classes used by the dataset
generator.  Every parameter is tied to a quantity the radar model in
``src/core/radar.py`` actually produces or that the tracker derives from it:

* measured per report : range r, azimuth, elevation, radial velocity r_dot
  (-> Doppler f_D = -2 r_dot / lambda), SNR (-> P_D, signal quality,
  measurement noise sigma_r / sigma_rdot)
* hidden but recoverable : RCS, through the radar equation
  sigma_hat = SNR (4 pi)^3 R^4 L k T0 B F / (P_t G^2 lambda^2 n)
* track level (EKF state) : altitude, ground speed, vertical speed,
  longitudinal / lateral acceleration, turn rate

Each class has two physically distinct variants so the classes overlap where
real platforms overlap (fighter vs. subsonic cruise missile in speed, fixed-wing
UAV vs. slow transport in speed, ...) and separate where physics separates
them (RCS, altitude, manoeuvre load).

Values are generic open-literature figures (radar textbooks and public
performance envelopes of whole platform categories), not the parameters of
any specific platform.  RCS values are X-band (fc = 10 GHz, the simulator
default) mean values per aspect in dBsm, fed to ``LookupTableRCS.synthetic``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .rcs import LookupTableRCS

G = 9.81
RCS_ASPECTS = ("nose", "side", "tail", "top")


@dataclass(frozen=True)
class VariantSpec:
    """Physical envelope of one platform variant.

    ``*_range`` tuples are hard limits; ``*_typical`` tuples are the band
    that holds most of the samples (see :func:`sample_instance`).
    """
    description: str
    weight: float                                 # sampling probability inside the class
    speed_range_mps: tuple[float, float]
    speed_typical_mps: tuple[float, float]
    altitude_agl_range_m: tuple[float, float]
    altitude_agl_typical_m: tuple[float, float]
    max_climb_rate_mps: float                     # |v_z| limit
    max_long_accel_mps2: float                    # along-track speed change
    max_lateral_accel_mps2: float                 # normal load (turns, pull-ups)
    max_turn_rate_rad_s: float                    # heading rate limit
    rcs_dbsm: dict = field(default_factory=dict)  # aspect -> (lo, hi) mean dBsm, X-band
    swerling: int = 1
    can_hover: bool = False

    def __post_init__(self):
        for name in ("speed", "altitude_agl"):
            lo, hi = getattr(self, f"{name}_range_mps" if name == "speed" else f"{name}_range_m")
            tlo, thi = getattr(self, f"{name}_typical_mps" if name == "speed" else f"{name}_typical_m")
            if not (lo <= tlo <= thi <= hi):
                raise ValueError(f"{self.description}: typical {name} band must lie inside the hard range")
        if set(self.rcs_dbsm) != set(RCS_ASPECTS):
            raise ValueError(f"{self.description}: rcs_dbsm needs exactly {RCS_ASPECTS}")
        # a coordinated turn needs a_lat = v * omega; the turn-rate cap must be reachable
        v_min = max(self.speed_range_mps[0], 1.0)
        if self.max_turn_rate_rad_s > self.max_lateral_accel_mps2 / v_min + 1e-9 and not self.can_hover:
            raise ValueError(f"{self.description}: turn-rate cap exceeds a_lat / v_min")


TARGET_CLASSES: dict[str, dict] = {
    "uav": {
        "label": 0,
        "name": "UAV",
        "variants": {
            "multirotor": VariantSpec(
                description="Small multirotor (quadcopter / hexacopter, < 25 kg)",
                weight=0.5,
                speed_range_mps=(0.0, 20.0), speed_typical_mps=(2.0, 12.0),
                altitude_agl_range_m=(30.0, 400.0), altitude_agl_typical_m=(40.0, 150.0),
                max_climb_rate_mps=8.0, max_long_accel_mps2=5.0,
                max_lateral_accel_mps2=8.0, max_turn_rate_rad_s=1.0,
                rcs_dbsm={"nose": (-22.0, -12.0), "side": (-20.0, -10.0),
                          "tail": (-22.0, -12.0), "top": (-15.0, -5.0)},
                swerling=1, can_hover=True,
            ),
            "fixed_wing": VariantSpec(
                description="Small/tactical fixed-wing UAV (2-6 m span, piston/electric)",
                weight=0.5,
                speed_range_mps=(18.0, 62.0), speed_typical_mps=(22.0, 45.0),
                altitude_agl_range_m=(100.0, 1500.0), altitude_agl_typical_m=(200.0, 800.0),
                max_climb_rate_mps=5.0, max_long_accel_mps2=3.0,
                max_lateral_accel_mps2=10.0, max_turn_rate_rad_s=0.35,
                rcs_dbsm={"nose": (-15.0, -5.0), "side": (-5.0, 3.0),
                          "tail": (-15.0, -5.0), "top": (0.0, 8.0)},
                swerling=1,
            ),
        },
    },
    "aircraft": {
        "label": 1,
        "name": "Aircraft",
        "variants": {
            "transport": VariantSpec(
                description="Transport / airliner / large turboprop",
                weight=0.5,
                speed_range_mps=(80.0, 200.0), speed_typical_mps=(100.0, 160.0),
                altitude_agl_range_m=(300.0, 4000.0), altitude_agl_typical_m=(800.0, 3000.0),
                max_climb_rate_mps=15.0, max_long_accel_mps2=2.0,
                max_lateral_accel_mps2=6.0, max_turn_rate_rad_s=0.06,
                rcs_dbsm={"nose": (10.0, 18.0), "side": (15.0, 25.0),
                          "tail": (10.0, 18.0), "top": (18.0, 28.0)},
                swerling=1,
            ),
            "fighter": VariantSpec(
                description="Fighter / jet trainer / attack aircraft",
                weight=0.5,
                speed_range_mps=(120.0, 320.0), speed_typical_mps=(180.0, 280.0),
                altitude_agl_range_m=(100.0, 4000.0), altitude_agl_typical_m=(150.0, 2000.0),
                max_climb_rate_mps=60.0, max_long_accel_mps2=10.0,
                max_lateral_accel_mps2=80.0, max_turn_rate_rad_s=0.35,
                rcs_dbsm={"nose": (0.0, 7.0), "side": (7.0, 15.0),
                          "tail": (3.0, 10.0), "top": (10.0, 18.0)},
                swerling=1,
            ),
        },
    },
    "missile": {
        "label": 2,
        "name": "Missile",
        "variants": {
            "subsonic_cruise": VariantSpec(
                description="Subsonic cruise missile, terrain following",
                weight=0.5,
                speed_range_mps=(200.0, 280.0), speed_typical_mps=(220.0, 260.0),
                altitude_agl_range_m=(30.0, 200.0), altitude_agl_typical_m=(30.0, 100.0),
                max_climb_rate_mps=20.0, max_long_accel_mps2=5.0,
                max_lateral_accel_mps2=40.0, max_turn_rate_rad_s=0.15,
                rcs_dbsm={"nose": (-20.0, -10.0), "side": (-3.0, 3.0),
                          "tail": (-12.0, -5.0), "top": (-2.0, 5.0)},
                swerling=3,
            ),
            "supersonic": VariantSpec(
                description="Supersonic missile (ramjet / rocket), low-altitude dash",
                weight=0.3,
                speed_range_mps=(450.0, 950.0), speed_typical_mps=(550.0, 800.0),
                altitude_agl_range_m=(30.0, 800.0), altitude_agl_typical_m=(30.0, 200.0),
                max_climb_rate_mps=50.0, max_long_accel_mps2=30.0,
                max_lateral_accel_mps2=150.0, max_turn_rate_rad_s=0.2,
                rcs_dbsm={"nose": (-12.0, -5.0), "side": (0.0, 6.0),
                          "tail": (-8.0, -2.0), "top": (2.0, 8.0)},
                swerling=3,
            ),
            "terminal_dive": VariantSpec(
                description="Supersonic manoeuvring missile ending in a steep terminal dive "
                            "(high-altitude dive or terrain-following pop-up attack)",
                weight=0.2,
                speed_range_mps=(450.0, 1100.0), speed_typical_mps=(600.0, 950.0),
                altitude_agl_range_m=(30.0, 15000.0), altitude_agl_typical_m=(3000.0, 12000.0),
                # |v_z| up to almost the full speed: the dive is steep, not a climb capability
                max_climb_rate_mps=1000.0, max_long_accel_mps2=60.0,
                # push-over into the dive and pull-out are ~10-30 g manoeuvres
                max_lateral_accel_mps2=300.0, max_turn_rate_rad_s=0.3,
                rcs_dbsm={"nose": (-12.0, -5.0), "side": (0.0, 6.0),
                          "tail": (-8.0, -2.0), "top": (2.0, 8.0)},
                swerling=3,
            ),
        },
    },
}

CLASS_ORDER = tuple(sorted(TARGET_CLASSES, key=lambda k: TARGET_CLASSES[k]["label"]))


def variant_spec(class_name: str, variant: str) -> VariantSpec:
    try:
        return TARGET_CLASSES[class_name]["variants"][variant]
    except KeyError:
        raise ValueError(f"unknown target class/variant {class_name!r}/{variant!r}") from None


def _banded(rng: np.random.Generator, hard, typical, p_typical: float = 0.85) -> float:
    """Draw from the typical band with probability ``p_typical``, else from the hard range."""
    lo, hi = typical if rng.random() < p_typical else hard
    return float(rng.uniform(lo, hi))


def sample_instance(class_name: str, rng: np.random.Generator, variant: str | None = None) -> dict:
    """Draw one concrete target of ``class_name`` within its physical envelope.

    Returns the variant, initial speed / AGL altitude / climb rate and the
    per-aspect mean RCS (dBsm) together with the Swerling case.
    """
    cls = TARGET_CLASSES[class_name]
    names = list(cls["variants"])
    if variant is None:
        w = np.array([cls["variants"][n].weight for n in names])
        variant = names[int(rng.choice(len(names), p=w / w.sum()))]
    spec = variant_spec(class_name, variant)
    speed = _banded(rng, spec.speed_range_mps, spec.speed_typical_mps)
    climb = float(rng.triangular(-spec.max_climb_rate_mps, 0.0, spec.max_climb_rate_mps))
    if not spec.can_hover:
        # flight-path angle of non-hovering platforms stays within +-30 deg
        climb = float(np.clip(climb, -0.5 * speed, 0.5 * speed))
    return {
        "class_name": class_name,
        "label": cls["label"],
        "variant": variant,
        "speed_mps": speed,
        "altitude_agl_m": _banded(rng, spec.altitude_agl_range_m, spec.altitude_agl_typical_m),
        "climb_rate_mps": climb,
        "rcs_dbsm": {a: float(rng.uniform(*spec.rcs_dbsm[a])) for a in RCS_ASPECTS},
        "swerling": spec.swerling,
    }


def rcs_model(instance: dict) -> LookupTableRCS:
    """Aspect-dependent RCS table of a sampled instance (X-band mean values)."""
    r = instance["rcs_dbsm"]
    return LookupTableRCS.synthetic(nose_dbsm=r["nose"], side_dbsm=r["side"],
                                    tail_dbsm=r["tail"], top_dbsm=r["top"])
