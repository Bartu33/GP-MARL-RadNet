"""Scenario files (YAML) -> simulator objects."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, fields
import json
import math
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml

from .detection import PD_MODELS
from .radar import RadarParams
from .targets import MOTION_BY_LABEL, TARGET_TYPE_PROFILES, validate_motion

ROOT = Path(__file__).resolve().parents[2]
SCENARIO_DIR = ROOT / "configs" / "scenarios"
RNG_LAYOUT_VERSION = 1

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
        "outages": [],             # [[radar index | "all", t0_s, t1_s], ...]: no detections in the window
    },
    "interleaved_actions": False,   # add INTERLEAVED (search + track) actions per track slot
    "share_actions": False,         # each action also decides whether the radar sends its reports to the centre
    "reward": {"w_track": 1.0, "w_miss": 1.0, "w_class": 0.5, "w_cost": 0.2, "w_false": 0.2,
               "w_fair": 0.0,       # optional load-equalisation term (1 - Jain fairness of tracking load)
               "w_interrupt": 0.5,  # continuity: share of targets that lost their track after being tracked
               "w_loss": 1.0,       # continuity: per fused track deleted by the tracker
               "w_comm": 0.0,       # communication: reports sent to the fusion centre per radar and step
               "err_scale_m": 150.0},
}


def _merge(base: dict, over: dict) -> dict:
    out = deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def list_scenarios() -> list[str]:
    return sorted(p.stem for p in SCENARIO_DIR.glob("*.yaml"))


def load_scenario(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f"{name_or_path}.yaml"
    with open(path, encoding="utf-8") as f:
        # Keep safe construction, but reject duplicate keys before they can
        # silently overwrite configuration (including within nested mappings).
        node = yaml.compose(f, Loader=yaml.SafeLoader)
    if node is None:
        raise ValueError("scenario: expected a mapping, got empty YAML")
    _check_yaml_keys(node)
    raw = yaml.safe_load(yaml.serialize(node))
    return validate_scenario(raw)


def _check_yaml_keys(node, path="", ancestors=()):
    if id(node) in ancestors:
        raise ValueError(f"{path or 'scenario'}: recursive YAML aliases are unsupported")
    ancestors = (*ancestors, id(node))
    if isinstance(node, yaml.MappingNode):
        seen = set()
        for key, value in node.value:
            if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                raise ValueError(f"{path or 'scenario'}: keys must be strings; YAML merge keys are unsupported")
            child = f"{path}.{key.value}" if path else key.value
            if key.value in seen:
                raise ValueError(f"{child}: duplicate YAML key (line {key.start_mark.line + 1})")
            seen.add(key.value)
            _check_yaml_keys(value, child, ancestors)
    elif isinstance(node, yaml.SequenceNode):
        for i, value in enumerate(node.value):
            _check_yaml_keys(value, f"{path}[{i}]", ancestors)


def _mapping(value, path, allowed):
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a mapping")
    for key in value:
        if key not in allowed:
            raise ValueError(f"{path}.{key}: unsupported configuration field")


def _number(value, path, *, minimum=None, maximum=None, positive=False, integer=False):
    kind = Integral if integer else Real
    valid = not isinstance(value, (bool, np.bool_)) and isinstance(value, kind)
    try:
        finite = valid and (integer or math.isfinite(value))
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{path}: expected a finite {'integer' if integer else 'number'}")
    if positive and value <= 0:
        raise ValueError(f"{path}: must be > 0")
    if minimum is not None and value < minimum:
        raise ValueError(f"{path}: must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{path}: must be <= {maximum}")


def _sequence(value, path, length=None):
    if (not isinstance(value, (list, tuple, np.ndarray))
            or (isinstance(value, np.ndarray) and value.ndim != 1)
            or (length is not None and len(value) != length)):
        raise ValueError(f"{path}: expected a sequence" + (f" of length {length}" if length is not None else ""))


def _text(value, path):
    if not isinstance(value, str):
        raise ValueError(f"{path}: expected a string")


def _radar_spec(spec, path, relationships=True):
    allowed = {f.name for f in fields(RadarParams)} - {"profiles"}
    _mapping(spec, path, allowed)
    p = RadarParams(**spec)
    values = asdict(p)
    for key, value in values.items():
        field_path = f"{path}.{key}"
        if key in ("name", "pd_model"):
            _text(value, field_path)
        elif key == "quantize":
            if not isinstance(value, bool):
                raise ValueError(f"{field_path}: expected a boolean")
        elif key in ("position", "bias_std", "search_bars_deg"):
            length = {"position": 3, "bias_std": 4}.get(key)
            _sequence(value, field_path, length)
            if len(value) == 0:
                raise ValueError(f"{field_path}: must not be empty")
            for i, v in enumerate(value):
                _number(v, f"{field_path}[{i}]", minimum=0 if key == "bias_std" else None)
                if key == "search_bars_deg":
                    _number(v, f"{field_path}[{i}]", minimum=-90, maximum=90)
        elif key != "profiles":
            _number(value, field_path, integer=key == "c_max")
    for key in ("fc_hz", "bandwidth_hz", "t0_k", "r_max_m", "beamwidth_az_deg", "beamwidth_el_deg",
                "e_max_j", "c_max", "efficiency", "fov_az_deg", "search_sector_deg"):
        _number(values[key], f"{path}.{key}", positive=True)
    for key in ("pt_w", "r_min_m", "k_r", "k_ang", "k_rdot", "floor_r_m", "floor_ang_rad",
                "floor_rdot_mps", "false_alarms_per_beam", "e_recovery_w", "e_processing_j",
                "thermal_a", "thermal_b", "h_ambient_k"):
        _number(values[key], f"{path}.{key}", minimum=0)
    for key, limit in (("efficiency", 1), ("fov_az_deg", 180), ("search_sector_deg", 360),
                       ("beamwidth_az_deg", 360), ("beamwidth_el_deg", 180)):
        _number(values[key], f"{path}.{key}", maximum=limit)
    if not 0 < p.pfa < 1:
        raise ValueError(f"{path}.pfa: must be strictly between 0 and 1")
    if relationships and p.r_max_m <= p.r_min_m:
        raise ValueError(f"{path}.r_max_m: must be greater than r_min_m")
    for key in ("el_min_deg", "el_max_deg"):
        _number(values[key], f"{path}.{key}", minimum=-90, maximum=90)
    if relationships and p.el_min_deg >= p.el_max_deg:
        raise ValueError(f"{path}.el_max_deg: require -90 <= el_min_deg < el_max_deg <= 90")
    if not -1 <= p.bias_rho <= 1:
        raise ValueError(f"{path}.bias_rho: must be between -1 and 1")
    if relationships and p.h_max_k <= p.h_ambient_k:
        raise ValueError(f"{path}.h_max_k: must be greater than h_ambient_k")
    if relationships and p.snr_good_db <= p.snr_threshold_db:
        raise ValueError(f"{path}.snr_good_db: must be greater than snr_threshold_db")
    if p.pd_model not in PD_MODELS:
        raise ValueError(f"{path}.pd_model: unsupported detection model")


def validate_scenario(scenario: dict) -> dict:
    """Resolve defaults and validate without mutation, I/O, or random draws.

    Return a detached configuration. Lists still replace entire default lists.
    Unknown fields (including unsupported radar mode/profile input) are errors.
    """
    _mapping(scenario, "scenario", DEFAULTS)
    sc = _merge(DEFAULTS, scenario)
    for key in ("name", "description"):
        _text(sc[key], key)
    for key in ("dt", "duration_s", "area_radius_m"):
        _number(sc[key], key, positive=True)
    ratio = sc["duration_s"] / sc["dt"]
    if not math.isfinite(ratio) or round(ratio) < 1:
        raise ValueError("duration_s: duration_s / dt must produce at least one finite simulation step")
    _number(sc["seed"], "seed", minimum=0, integer=True)
    _number(sc["max_track_slots"], "max_track_slots", positive=True, integer=True)
    for key in ("interleaved_actions", "share_actions"):
        if not isinstance(sc[key], bool):
            raise ValueError(f"{key}: expected a boolean")
    _mapping(sc["site"], "site", DEFAULTS["site"])
    _text(sc["site"]["name"], "site.name")
    _number(sc["site"]["lat"], "site.lat", minimum=-85.05112878, maximum=85.05112878)
    _number(sc["site"]["lon"], "site.lon", minimum=-180, maximum=180)
    terrain = sc["terrain"]
    if isinstance(terrain, str):
        terrain = {"source": terrain, "resolution": 512}
    _mapping(terrain, "terrain", (*DEFAULTS["terrain"], "hills"))
    source = terrain.get("source", "flat")
    if not isinstance(source, str) or source not in ("flat", "procedural", "tiles", "auto"):
        raise ValueError("terrain.source: expected flat, procedural, tiles, or auto")
    res = terrain.get("resolution", 512)
    _number(res, "terrain.resolution", minimum=2, integer=True)
    if source in ("tiles", "auto") and (res > 1024 or 1024 % res):
        raise ValueError("terrain.resolution: tiles/auto require a divisor of 1024 between 2 and 1024")
    if "hills" in terrain:
        if source != "procedural":
            raise ValueError("terrain.hills: supported only with source procedural")
        if not isinstance(terrain["hills"], list):
            raise ValueError("terrain.hills: expected a list")
        for i, hill in enumerate(terrain["hills"]):
            path = f"terrain.hills[{i}]"
            _mapping(hill, path, ("center_m", "height_m", "sigma_m"))
            _sequence(hill.get("center_m"), f"{path}.center_m", 2)
            for j, value in enumerate(hill["center_m"]):
                _number(value, f"{path}.center_m[{j}]")
            for key in ("height_m", "sigma_m"):
                _number(hill.get(key), f"{path}.{key}", positive=True)
    for key in ("radars", "targets"):
        if not isinstance(sc[key], list):
            raise ValueError(f"{key}: expected a list")
    if not sc["radars"]:
        raise ValueError("radars: at least one radar is required")
    # Validate defaults' fields before merging, then numeric relationships on
    # each effective radar (an individual override may complete the defaults).
    allowed = {f.name for f in fields(RadarParams)} - {"profiles"}
    _mapping(sc["radar_defaults"], "radar_defaults", allowed)
    _radar_spec(sc["radar_defaults"], "radar_defaults", relationships=False)
    for i, spec in enumerate(sc["radars"]):
        path = f"radars[{i}]"
        _mapping(spec, path, allowed)
        _radar_spec(_merge(sc["radar_defaults"], spec), path)
    for i, target in enumerate(sc["targets"]):
        path = f"targets[{i}]"
        _mapping(target, path, ("label", "position", "velocity", "motion", "target_type"))
        if "label" not in target:
            raise ValueError(f"{path}.label: required field")
        _number(target["label"], f"{path}.label", integer=True)
        if target["label"] not in MOTION_BY_LABEL:
            raise ValueError(f"{path}.label: unsupported target class")
        if "target_type" in target:
            _text(target["target_type"], f"{path}.target_type")
            if target["target_type"] not in TARGET_TYPE_PROFILES:
                raise ValueError(f"{path}.target_type: unknown target type {target['target_type']!r}")
        if "position" in target:
            _sequence(target["position"], f"{path}.position", 3)
            for j, v in enumerate(target["position"]):
                _number(v, f"{path}.position[{j}]")
        if "motion" in target:
            total = validate_motion(target["motion"], f"{path}.motion")
            for field in ("position", "velocity"):
                if field not in target:
                    raise ValueError(f"{path}.{field}: required with explicit motion")
            _sequence(target["velocity"], f"{path}.velocity", 3)
            for j, value in enumerate(target["velocity"]):
                _number(value, f"{path}.velocity[{j}]")
            effective_duration = int(round(ratio)) * sc["dt"]
            if total is not None:
                tolerance = 32 * np.finfo(float).eps * max(1.0, total, effective_duration)
                if total + tolerance < effective_duration:
                    raise ValueError(f"{path}.motion.segments: total duration must cover effective episode duration")
        elif "velocity" in target:
            raise ValueError(f"{path}.velocity: requires an explicit motion definition")
    stress = sc["stress"]
    _mapping(stress, "stress", DEFAULTS["stress"])
    _number(stress["noise_scale"], "stress.noise_scale", positive=True)
    _number(stress["comm_drop_prob"], "stress.comm_drop_prob", minimum=0, maximum=1)
    _number(stress["fail_time_s"], "stress.fail_time_s", minimum=0)
    if stress["fail_radar"] is not None:
        _number(stress["fail_radar"], "stress.fail_radar", minimum=0,
                maximum=len(sc["radars"]) - 1, integer=True)
    if stress["max_reports_per_radar"] is not None:
        _number(stress["max_reports_per_radar"], "stress.max_reports_per_radar", minimum=0, integer=True)
    if not isinstance(stress["outages"], list):
        raise ValueError("stress.outages: expected a list")
    for i, outage in enumerate(stress["outages"]):
        path = f"stress.outages[{i}]"
        _sequence(outage, path, 3)
        radar, t0, t1 = outage
        if not isinstance(radar, str) or radar != "all":
            _number(radar, f"{path}[0]", minimum=0, maximum=len(sc["radars"]) - 1, integer=True)
        _number(t0, f"{path}[1]", minimum=0)
        _number(t1, f"{path}[2]", minimum=0)
        if t1 <= t0:
            raise ValueError(f"{path}[2]: must be greater than the outage start")
    _mapping(sc["reward"], "reward", DEFAULTS["reward"])
    for key, value in sc["reward"].items():
        _number(value, f"reward.{key}", positive=key == "err_scale_m")
    # Scaling can overflow even when both operands are finite. Check without
    # altering the scenario or applying the scale a second time later.
    for i, radar in enumerate(build_radars(sc)):
        for key in ("k_r", "k_ang", "k_rdot", "floor_r_m", "floor_ang_rad", "floor_rdot_mps"):
            _number(getattr(radar, key), f"radars[{i}].{key}", minimum=0)
    return sc


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


def _json_value(value):
    """Detach NumPy scalars/arrays and tuples into JSON's primitive types."""
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def scenario_snapshot(scenario: dict, *, episode_seed=None, radar_params=None, terrain=None) -> dict:
    """Create an inspectable snapshot; no disk, terrain loading, or RNG draws.

    Pass the environment's radar_params to capture actual terrain-adjusted
    positions. Effective radars are diagnostic output, never scenario input.
    Replay with RadarNetEnv(snapshot['scenario'], seed=snapshot['episode_seed']).
    """
    sc = validate_scenario(scenario)
    seed = sc["seed"] if episode_seed is None else episode_seed
    _number(seed, "episode_seed", minimum=0, integer=True)
    steps = int(round(sc["duration_s"] / sc["dt"]))
    effective = build_radars(sc) if radar_params is None else radar_params
    source = sc["terrain"] if isinstance(sc["terrain"], str) else sc["terrain"]["source"]
    snapshot = {
        "snapshot_version": 1,
        "rng_layout_version": RNG_LAYOUT_VERSION,
        "scenario": sc,
        "effective_radars": [
            {**asdict(p), "profiles": {mode.name: asdict(profile) for mode, profile in p.profiles.items()}}
            for p in effective
        ],
        "radar_position_reference": "terrain_adjusted" if radar_params is not None else "input_mast_height",
        "episode_seed": seed,
        "scenario_seed": sc["seed"],
        "terrain_seed": sc["seed"],
        "simulation_steps": steps,
        "effective_duration_s": steps * sc["dt"],
        "terrain_provenance": {
            "requested_source": source,
            "actual_source": terrain.source if terrain is not None else None,
            "data_hash": None,
            "missing": ["data_hash"] if terrain is not None else ["actual_source", "data_hash"],
        },
    }
    return _json_value(snapshot)


def save_snapshot(snapshot: dict, path: str | Path) -> Path:
    """Explicitly save a snapshot, with deterministic formatting and no NaN."""
    data = json.dumps(snapshot, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.write_text(data, encoding="utf-8")
    return path
