"""Scenario files (YAML) -> simulator objects."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from .radar import RadarParams

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "configs" / "scenarios"

DEFAULTS = {
    "name": "unnamed",
    "description": "",
    "site": {"name": "Utah Valley · US", "lat": 40.30, "lon": -111.70},
    "dt": 0.5,
    "duration_s": 120.0,
    "area_radius_m": 7000.0,
    "seed": 0,
    "max_track_slots": 6,
    "radar_defaults": {},
    "radars": [],
    "targets": [],
    "terrain": {"source": "flat", "resolution": 512},   # flat | procedural | tiles | auto
    "stress": {
        "fail_radar": None,        # index of a radar that goes offline
        "fail_time_s": 30.0,
        "comm_drop_prob": 0.0,     # probability that a radar's reports are lost in a step
        "max_reports_per_radar": None,  # bandwidth cap (top-k reports by quality)
        "noise_scale": 1.0,        # multiplies measurement floors and k-factors
    },
    "interleaved_actions": False,   # add INTERLEAVED (search + track) actions per track slot
    "reward": {"w_track": 1.0, "w_miss": 1.0, "w_class": 0.5, "w_cost": 0.2, "w_false": 0.2,
               "w_fair": 0.0,       # optional load-equalisation term (1 - Jain fairness of tracking load)
               "err_scale_m": 150.0},
}


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _merge(DEFAULTS, raw)


def build_radars(sc: dict) -> list[RadarParams]:
    radars = []
    scale = sc["stress"].get("noise_scale", 1.0)
    for spec in sc["radars"]:
        kw = _merge(sc.get("radar_defaults", {}), spec)
        for key in ("position", "search_bars_deg"):
            if key in kw:
                kw[key] = tuple(kw[key])
        p = RadarParams(**kw)
        if scale != 1.0:
            p.k_r *= scale
            p.k_ang *= scale
            p.k_rdot *= scale
            p.floor_r_m *= scale
            p.floor_ang_rad *= scale
            p.floor_rdot_mps *= scale
        radars.append(p)
    return radars
