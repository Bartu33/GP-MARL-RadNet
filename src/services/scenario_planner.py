"""Configuration-only dataset planning. Never constructs or steps an environment.

v1 supports one explicit-motion target, fixed sensor layout/spaces and split-owned
flat/explicit-hill pools. Family metadata is descriptive, not a holdout engine.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

from src.core.scenario import (
    _check_yaml_keys, _json_value, _mapping, _merge, _number, _sequence,
    load_scenario, scenario_snapshot, validate_scenario,
)
from src.core.terrain import build_terrain

SPLITS = ("train", "validation", "test")
# Explicit IDs are part of planner v1's RNG contract; never enumerate recipe keys.
FIELDS = {
    "targets[0].label": (0, ("targets", 0, "label")),
    "targets[0].position": (1, ("targets", 0, "position")),
    "targets[0].velocity": (2, ("targets", 0, "velocity")),
    "targets[0].motion": (3, ("targets", 0, "motion")),
    "stress.noise_scale": (4, ("stress", "noise_scale")),
    "stress.comm_drop_prob": (5, ("stress", "comm_drop_prob")),
}


def _canonical(value):
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _physical_numbers(value):
    """Normalize equivalent int/float spellings for physical group identities."""
    if isinstance(value, dict):
        return {key: _physical_numbers(v) for key, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_physical_numbers(v) for v in value]
    return (float(value) or 0.0) if isinstance(value, Real) else value


def load_recipe(path):
    """Read safe YAML with the same duplicate-key protection as scenarios."""
    with open(path, encoding="utf-8") as file:
        node = yaml.compose(file, Loader=yaml.SafeLoader)
    if node is None:
        raise ValueError("recipe: expected a mapping, got empty YAML")
    _check_yaml_keys(node)
    recipe = yaml.safe_load(yaml.serialize(node))
    if not isinstance(recipe, dict):
        raise ValueError("recipe: expected a mapping")
    return recipe


def _apply(sc, field, value):
    target = sc
    *parents, key = FIELDS[field][1]
    for part in parents:
        target = target[part]
    target[key] = deepcopy(value)


def terrain_group_id(terrain):
    """Continuous ENU surface identity, not a seed or grid-resolution identity.

    Explicit hill order, numeric spelling and seed do not change the group.
    Coincident Gaussians are merged. Empty explicit hills equal a flat plane.
    Only validated recipe terrain specs should be passed here.
    """
    if terrain["source"] == "flat":
        return "terrain-" + _digest({"surface": "flat"})[:20]
    if terrain["source"] != "procedural" or "hills" not in terrain:
        raise ValueError("terrain: planner v1 supports only flat or explicit procedural hills")
    grouped = {}
    for hill in terrain["hills"]:
        key = tuple(float(v) or 0.0 for v in (*hill["center_m"], hill["sigma_m"]))
        grouped.setdefault(key, []).append(float(hill["height_m"]))
    hills = []
    for (x, y, sigma), heights in sorted(grouped.items()):
        try:
            height = math.fsum(heights)
        except OverflowError as exc:
            raise ValueError("terrain.hills: coincident heights overflow") from exc
        hills.append([x, y, sigma, height])
    surface = {"surface": "gaussian_sum", "hills": hills} if hills else {"surface": "flat"}
    return "terrain-" + _digest(surface)[:20]


def _distribution(field, spec, base):
    _mapping(spec, f"randomization.{field}", ("choice", "uniform"))
    if len(spec) != 1:
        raise ValueError(f"randomization.{field}: supply exactly one choice or uniform")
    if "choice" in spec:
        values = spec["choice"]
        if not isinstance(values, list) or not values:
            raise ValueError(f"randomization.{field}.choice: expected a nonempty list")
    else:
        if field in ("targets[0].label", "targets[0].motion"):
            raise ValueError(f"randomization.{field}: uniform is not supported for this field")
        bounds = spec["uniform"]
        _mapping(bounds, f"randomization.{field}.uniform", ("low", "high"))
        values = [bounds.get("low"), bounds.get("high")]
        for name, value in zip(("low", "high"), values):
            path = f"randomization.{field}.uniform.{name}"
            if field in ("targets[0].position", "targets[0].velocity"):
                _sequence(value, path, 3)
                for i, v in enumerate(value):
                    _number(v, f"{path}[{i}]")
            else:
                _number(value, path)
        if np.any(np.asarray(values[1]) < np.asarray(values[0])):
            raise ValueError(f"randomization.{field}.uniform: high must be >= low")
    for value in values:
        probe = deepcopy(base)
        _apply(probe, field, value)
        try:
            validate_scenario(probe)
        except ValueError as exc:
            raise ValueError(f"randomization.{field}: {exc}") from exc


def _prepare(recipe):
    _mapping(recipe, "recipe", ("recipe_version", "master_seed", "episodes", "base_scenario",
                               "scenario_family", "overrides", "randomization", "terrain_pools",
                               "split_ratios", "max_attempts"))
    for key in ("recipe_version", "master_seed", "episodes"):
        _number(recipe.get(key), key, minimum=1 if key != "master_seed" else 0, integer=True)
    if recipe["recipe_version"] != 1:
        raise ValueError("recipe_version: only version 1 is supported")
    if not isinstance(recipe.get("base_scenario"), str):
        raise ValueError("base_scenario: expected an existing scenario name or path")
    base = load_scenario(recipe["base_scenario"])
    overrides = recipe.get("overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides: expected a mapping")
    base = validate_scenario(_merge(base, overrides))
    if len(base["targets"]) != 1 or "motion" not in base["targets"][0]:
        raise ValueError("targets: planner v1 requires exactly one explicit-motion target")
    family = recipe.get("scenario_family", base["name"])
    if not isinstance(family, str) or not family:
        raise ValueError("scenario_family: expected a nonempty string")
    attempts = recipe.get("max_attempts", 10)
    _number(attempts, "max_attempts", positive=True, integer=True)
    weights = recipe.get("split_ratios", {"train": 0.7, "validation": 0.15, "test": 0.15})
    _mapping(weights, "split_ratios", SPLITS)
    ratios = []
    for split in SPLITS:
        _number(weights.get(split), f"split_ratios.{split}", positive=True, maximum=1)
        ratios.append(Fraction(str(weights[split])))
    if sum(ratios) != 1:
        raise ValueError("split_ratios: must sum to 1")
    randomization = recipe.get("randomization", {})
    _mapping(randomization, "randomization", FIELDS)
    for field, distribution in randomization.items():
        _distribution(field, distribution, base)
    pools = recipe.get("terrain_pools")
    _mapping(pools, "terrain_pools", SPLITS)
    prepared, owners = {}, {}
    for split in SPLITS:
        entries = pools.get(split)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"terrain_pools.{split}: require a nonempty split-owned pool")
        prepared[split] = []
        for i, entry in enumerate(entries):
            path = f"terrain_pools.{split}[{i}]"
            _mapping(entry, path, ("terrain", "seed"))
            _mapping(entry.get("terrain"), f"{path}.terrain", ("source", "resolution", "hills"))
            candidate = deepcopy(base)
            candidate["terrain"] = deepcopy(entry.get("terrain"))
            candidate["seed"] = entry.get("seed", 0)
            try:
                candidate = validate_scenario(candidate)
                group = terrain_group_id(candidate["terrain"])
            except ValueError as exc:
                raise ValueError(f"{path}: {exc}") from exc
            if group in owners and owners[group] != split:
                raise ValueError(f"{path}: terrain group already belongs to {owners[group]}; split leakage")
            owners[group] = split
            # Limited explicit pools only: no tile downloads or episode RNGs.
            ground = build_terrain(candidate["site"], candidate["terrain"], seed=candidate["seed"])
            prepared[split].append((candidate["terrain"], candidate["seed"], group, ground))
    return base, family, attempts, ratios, randomization, prepared


def _geometry(sc, ground):
    def inside(p, path):
        if ground.source != "flat" and not (
                ground.x_west <= p[0] <= ground.x_west + ground.cell * ground.H.shape[1]
                and ground.y_north - ground.cell * ground.H.shape[0] <= p[1] <= ground.y_north):
            raise ValueError(f"{path}: outside terrain grid (edge clamp is not accepted)")
    for i, radar in enumerate(sc["radars"]):
        p = radar.get("position", sc["radar_defaults"].get("position", (0, 0, 10)))
        inside(p, f"radars[{i}].position")
        if p[2] < 0:
            raise ValueError(f"radars[{i}].position[2]: mast is below local ground")
    p = sc["targets"][0]["position"]
    inside(p, "targets[0].position")
    if p[2] < float(ground.height(p[0], p[1])) + 30:
        raise ValueError("targets[0].position[2]: initial AGL must be >= 30 m without clearance clamping")


def plan_episodes(recipe):
    """Return a deterministic JSON-ready plan; no simulation or automatic files.

    Prefix-stable weighted split scheduling approximates quotas to integer
    episode counts. Geometry retries redraw ONLY position; other field streams
    and episode seeds never advance. Routes are conservatively grouped by
    initial position/velocity: motion variants of those starts stay together.
    """
    base, family, max_attempts, ratios, distributions, pools = _prepare(recipe)
    master = int(recipe["master_seed"])
    def rng(namespace, index, field=0, attempt=0):
        return np.random.default_rng(np.random.SeedSequence(master, spawn_key=(namespace, 1, index, field, attempt)))
    def sample(field, index, attempt=0):
        stream = rng(600, index, FIELDS[field][0], attempt)
        distribution = distributions[field]
        if "choice" in distribution:
            return deepcopy(distribution["choice"][int(stream.integers(len(distribution["choice"])))])
        bounds = distribution["uniform"]
        return _json_value(stream.uniform(bounds["low"], bounds["high"]))
    counts = [0, 0, 0]
    episodes, route_owners = [], {}
    for index in range(int(recipe["episodes"])):
        which = max(range(3), key=lambda j: (index + 1) * ratios[j] - counts[j])
        split = SPLITS[which]
        counts[which] += 1
        pool_index = int(rng(602, index).integers(len(pools[split])))
        terrain, terrain_seed, terrain_id, ground = pools[split][pool_index]
        episode_seed = int(rng(601, index).integers(0, 2 ** 63, dtype=np.int64))
        values = {field: sample(field, index) for field in distributions}
        rejected = []
        for attempt in range(max_attempts):
            if attempt and "targets[0].position" in distributions:
                values["targets[0].position"] = sample("targets[0].position", index, attempt)
            sc = deepcopy(base)
            sc["terrain"], sc["seed"] = deepcopy(terrain), terrain_seed
            for field, value in values.items():
                _apply(sc, field, value)
            sc = validate_scenario(sc)
            try:
                _geometry(sc, ground)
            except ValueError as exc:
                rejected.append({"attempt": attempt, "reason": str(exc)})
                continue
            break
        else:
            raise ValueError(f"episode_index={index}: geometry failed after {max_attempts} attempts: {_canonical(rejected)}")
        target = sc["targets"][0]
        route_id = "route-" + _digest(_physical_numbers(
            {k: target[k] for k in ("position", "velocity")}))[:20]
        if route_id in route_owners and route_owners[route_id] != split:
            raise ValueError(f"episode_index={index}: route group already belongs to {route_owners[route_id]}; "
                             "vary initial position/velocity; split leakage")
        route_owners[route_id] = split
        snapshot = scenario_snapshot(sc, episode_seed=episode_seed)
        config_hash = _digest(snapshot)
        dataset_id = f"ep-{index:06d}-" + _digest([master, index, split, family, config_hash])[:12]
        episodes.append({
            "dataset_episode_id": dataset_id, "episode_index": index, "split": split,
            "scenario_family": family, "terrain_group_id": terrain_id, "route_group_id": route_id,
            "geometry_group_id": "geometry-" + _digest([terrain_id, route_id, sc["radars"], sc["radar_defaults"]])[:20],
            "master_seed": master, "episode_seed": episode_seed, "terrain_seed": terrain_seed,
            "sampled_values": deepcopy(values), "resolved_scenario": sc, "config_snapshot": snapshot,
            "resolved_configuration_hash": config_hash, "recipe_version": 1,
            "generation_status": "planned", "geometry_attempts": attempt + 1, "rejected_attempts": rejected,
        })
    return _json_value({
        "plan_schema_version": 1, "recipe_version": 1, "master_seed": master,
        "recipe": deepcopy(recipe),
        "split_ratios": dict(zip(SPLITS, map(float, ratios))), "split_counts": dict(zip(SPLITS, counts)),
        "split_allocation": "prefix-stable largest-deficit allocation; finite counts use integer quotas",
        "geometry_validation": "initial target clearance and radar/target grid bounds; full route not simulated",
        "grouping_policy": "terrain and initial-position/velocity groups disjoint; fixed sensor layout may be shared; family is metadata only",
        "configuration_hash_basis": "canonical config_snapshot JSON, SHA-256",
        "episodes": episodes,
    })


def export_plan(plan, path):
    """Write only on explicit request; sorted standard JSON without timestamps."""
    data = json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path = Path(path)
    path.write_text(data, encoding="utf-8")
    return path
