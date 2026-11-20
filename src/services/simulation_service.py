"""Run episodes, collect metrics and compare policies."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.env import RadarNetEnv
from src.core.scenario import load_scenario
from src.models.baselines.policies import make_policy

ROOT = Path(__file__).resolve().parents[2]
CLASSIFIER_PATH = ROOT / "models" / "checkpoints" / "flight_profile_gbm.joblib"


def load_classifier(path: Path = CLASSIFIER_PATH):
    if not Path(path).exists():
        return None
    from src.models.classifier.classifier import FlightProfileClassifier
    return FlightProfileClassifier.load(path)


def single_radar_summary(env: RadarNetEnv) -> dict:
    """Structure A: each radar alone with its own local track, on the same
    detections as the fused track.  Coverage uses the network's coverage mask
    so that it is directly comparable with ``coverage_in_range``."""
    if not env.local_eval:
        return {}
    net_cov = np.concatenate([np.asarray(c, bool) for c in env.evaluator.in_cov])
    cov, longest, rmse = [], [], []
    for ev in env.local_eval:
        valid = np.concatenate([np.asarray(v, bool) for v in ev.valid])
        cov.append(float(valid[net_cov].mean()) if net_cov.any() else np.nan)
        s = ev.summary()
        longest.append(s["longest_track_s"])
        rmse.append(s["pos_rmse_m"])
    return {
        "single_best_coverage": float(np.nanmax(cov)),
        "single_best_longest_s": float(np.nanmax(longest)),
        "single_mean_rmse_m": float(np.nanmean(rmse)) if np.any(np.isfinite(rmse)) else np.nan,
    }


def per_radar_table(env: RadarNetEnv) -> pd.DataFrame:
    """Local single-radar tracks next to the fused track (one row each)."""
    rows = []
    for p, ev in zip(env.radar_params, env.local_eval):
        rows.append({"track": f"{p.name} (local)", **ev.summary()})
    rows.append({"track": "Fused (network)", **env.evaluator.summary()})
    cols = ["track", "coverage_in_range", "pos_rmse_m", "longest_track_s", "track_losses", "interruptions",
            "reacquire_s", "correct_class_continuity", "nees_mean", "nees_in_95"]
    return pd.DataFrame(rows)[cols]


def summarize(env: RadarNetEnv) -> dict:
    lg = env.log
    return {
        # coverage, continuity, detection / track loss: see src/core/evaluation.py (METRIC_INFO)
        **env.evaluator.summary(),
        **single_radar_summary(env),
        **env.scheduler_summary(),
        "false_tracks_mean": float(np.mean(lg["false_tracks"])) if lg["false_tracks"] else 0.0,
        "class_accuracy": float(np.nanmean(lg["class_acc"])) if np.any(np.isfinite(lg["class_acc"])) else np.nan,
        "sensing_cost": float(np.mean(lg["energy"])) if lg["energy"] else 0.0,
        "reports_per_step": float(np.mean(lg["reports"])) if lg["reports"] else 0.0,
        "return": float(np.sum(lg["reward"])),
        # Jain fairness of the radars' cumulative tracking time (sensor-usage balance)
        "usage_fairness": float(lg["fairness"][-1]) if lg["fairness"] else 1.0,
    }


def run_episode(scenario, policy_name: str = "central_greedy", seed: int | None = None,
                classifier="auto", record: bool = True):
    """Run one episode.  Returns (env, summary).  ``env.frames`` holds the
    recorded frames for the 3-D viewer when ``record`` is True."""
    sc = load_scenario(scenario) if isinstance(scenario, str) else scenario
    clf = load_classifier() if classifier == "auto" else classifier
    env = RadarNetEnv(sc, classifier=clf, record=record, seed=seed, local_tracks=True)
    obs, _ = env.reset(seed=seed)
    policy = make_policy(policy_name)
    policy.reset(env)
    decide = []
    while env.agents:
        t0 = time.perf_counter()
        if hasattr(policy, "requests"):                     # task manager: requests -> local schedulers
            reqs = policy.requests(env)
            decide.append(time.perf_counter() - t0)
            obs, _, _, _, _ = env.step_requests(reqs)
        else:
            actions = policy.act(env, obs)
            decide.append(time.perf_counter() - t0)
            obs, _, _, _, _ = env.step(actions)
    return env, {**summarize(env), "decision_ms": float(np.mean(decide) * 1000) if decide else np.nan}


def compare_policies(scenario, policies, seeds=(0, 1, 2), classifier="auto", progress=None) -> pd.DataFrame:
    """Run every policy on every seed.  ``progress(done, total, label)`` is
    called after each episode (used by the dashboard's progress bar)."""
    rows, skipped = [], {}
    clf = load_classifier() if classifier == "auto" else classifier
    seeds = list(seeds)
    total, done = len(policies) * len(seeds), 0
    for pol in policies:
        for s in seeds:
            try:
                _, summ = run_episode(scenario, pol, seed=s, classifier=clf, record=False)
            except ValueError as e:           # trained policy built for another scenario size
                if type(e).__name__ != "IncompatiblePolicy":
                    raise
                skipped[pol] = str(e)
                done += len(seeds) - seeds.index(s)
                if progress is not None:
                    progress(done, total, f"{pol} skipped")
                break
            rows.append({"policy": pol, "seed": s, **summ})
            done += 1
            if progress is not None:
                progress(done, total, f"{pol} · seed {s}")
    df = pd.DataFrame(rows)
    df.attrs["skipped"] = skipped             # policy -> reason (shown by the dashboard)
    return df


# Locations offered by the map picker of the 3-D view (display only: the
# simulation itself is flat-earth and does not depend on the site).
MAP_SITES = [
    {"name": "Utah Valley · US", "lat": 40.30, "lon": -111.70},
    {"name": "Bursa · Uludağ · TR", "lat": 40.14, "lon": 29.10},
    {"name": "Kayseri · Erciyes · TR", "lat": 38.58, "lon": 35.44},
    {"name": "Ankara · TR", "lat": 39.93, "lon": 32.85},
    {"name": "İzmir · TR", "lat": 38.42, "lon": 27.14},
    {"name": "Kapadokya · Göreme · TR", "lat": 38.64, "lon": 34.83},
    {"name": "Van · TR", "lat": 38.50, "lon": 43.38},
    {"name": "Zermatt · Alps · CH", "lat": 46.02, "lon": 7.75},
    {"name": "Grand Canyon · US", "lat": 36.10, "lon": -112.11},
]


def viewer_payload(env: RadarNetEnv) -> dict:
    """Static scene description + frames, consumed by the 3-D viewer."""
    sc = env.sc
    return {
        "site": sc["site"],
        "sites": [sc["site"]] + [s for s in MAP_SITES if s["name"] != sc["site"]["name"]],
        "terrain": env.terrain.to_payload(),
        "dt": env.dt,
        "area_radius": sc["area_radius_m"],
        "radars": [{
            "name": p.name, "pos": [float(x) for x in p.position], "heading": float(p.heading),
            "fov": float(np.deg2rad(p.fov_az_deg)), "rmax": float(p.r_max_m),
            "bw_az": float(np.deg2rad(p.beamwidth_az_deg)), "bw_el": float(np.deg2rad(p.beamwidth_el_deg)),
        } for p in env.radar_params],
        "classes": ["A · fast", "B · agile", "C · erratic", "D · drift"],
        "frames": env.frames,
    }


if __name__ == "__main__":
    env, summ = run_episode("nominal", "central_greedy", seed=0)
    print(summ)
