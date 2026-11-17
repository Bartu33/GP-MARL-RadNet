"""Tracking study: local single-radar tracks vs. the fused track.

For every scenario, policy and seed the same detections feed (a) one local
tracker per radar and (b) the fusion centre's tracker; target trajectories are
identical across policies for a given seed (independent random streams, see
``RadarNetEnv.reset``).  Metrics are those of ``src/core/evaluation.py``:

* coverage   -- target-steps with a valid track while at least one radar of
                the *network* can see the target, so local and fused values
                share one denominator (``coverage_net``); ``coverage_own`` uses
                each radar's own coverage instead.
* pos_rmse_m / pos_err_median_m -- position error of matched tracks.
* nees_mean / nees_in_95 -- filter consistency (about 6 and 0.95 when the
                covariance matches the real error).

Run:
    python -m src.services.tracking_report --seeds 10
Outputs go to ``data/processed/tracking_report/`` (CSV tables and PNG figures).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.scenario import load_scenario
from src.services.simulation_service import run_episode

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed" / "tracking_report"
SCENARIOS = ["nominal", "high_noise", "multi_target"]
POLICIES = ["fixed_scan", "central_greedy"]
METRICS = ["coverage_net", "coverage_own", "pos_rmse_m", "pos_err_median_m", "nees_mean", "nees_in_95",
           "longest_track_s", "track_losses", "interruptions"]


def _scenario(name: str, terrain: str) -> dict:
    sc = load_scenario(name)
    if terrain != "auto":
        sc["terrain"] = {"source": terrain}
    return sc


def track_rows(env, scenario: str, policy: str, seed: int) -> list[dict]:
    """One row per local radar track and one for the fused track."""
    net_cov = np.concatenate([np.asarray(c, bool) for c in env.evaluator.in_cov])
    rows = []
    sources = [(f"{p.name} (local)", ev) for p, ev in zip(env.radar_params, env.local_eval)]
    sources.append(("Fused", env.evaluator))
    for name, ev in sources:
        s = ev.summary()
        valid = np.concatenate([np.asarray(v, bool) for v in ev.valid])
        rows.append({
            "scenario": scenario, "policy": policy, "seed": seed, "track": name,
            "coverage_net": float(valid[net_cov].mean()) if net_cov.any() else np.nan,
            "coverage_own": s["coverage_in_range"],
            **{k: s[k] for k in METRICS if k in s and k not in ("coverage_net", "coverage_own")},
        })
    return rows


def run_study(scenarios=SCENARIOS, policies=POLICIES, seeds=range(10), terrain="procedural"):
    """Returns (per-track rows, episode summaries, the first episode's env per scenario)."""
    rows, episodes, examples = [], [], {}
    for scen in scenarios:
        for pol in policies:
            for seed in seeds:
                env, s = run_episode(_scenario(scen, terrain), pol, seed=seed, classifier=None, record=False)
                rows += track_rows(env, scen, pol, seed)
                episodes.append({"scenario": scen, "policy": pol, "seed": seed, **s})
                if pol == "central_greedy" and seed == min(seeds):
                    examples[scen] = env
    return pd.DataFrame(rows), pd.DataFrame(episodes), examples


def _closest_to_dof(x):
    """Most consistent NEES: the value nearest to its expected value (6)."""
    x = x.dropna()
    return float(x.iloc[np.argmin(np.abs(x.to_numpy() - 6.0))]) if len(x) else np.nan


# "best" local radar per metric: higher is better, except errors / losses (lower)
# and NEES (closest to 6)
BEST_AGG = {**{m: "max" for m in METRICS}, "pos_rmse_m": "min", "pos_err_median_m": "min",
            "track_losses": "min", "interruptions": "min", "nees_mean": _closest_to_dof}


def summary_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation over seeds; local radars pooled as 'Local (mean of radars)'
    and 'Local (best per metric)' next to the fused track."""
    loc = rows[rows.track != "Fused"]
    per_seed = loc.groupby(["scenario", "policy", "seed"])
    mean_local = per_seed[METRICS].mean().reset_index().assign(track="Local (mean of radars)")
    best_local = per_seed[METRICS].agg(BEST_AGG).reset_index()
    best_local = best_local.assign(track="Local (best per metric)")
    fused = rows[rows.track == "Fused"]
    allr = pd.concat([mean_local, best_local, fused[["scenario", "policy", "seed", "track"] + METRICS]])
    g = allr.groupby(["scenario", "policy", "track"])[METRICS]
    out = g.mean().round(3).astype(str) + " ± " + g.std().round(3).astype(str)
    return out.reset_index()


# ---------------------------------------------------------------- figures
LOCAL_COLORS = ["#9ecae1", "#6baed6", "#3182bd", "#08519c", "#74c476", "#31a354"]
FUSED_COLOR = "#c0392b"


def _bar(ax, rows, metric, policy, title, ref=None, robust=False):
    """Bars per scenario and track.  mean ± std, or with ``robust`` the median
    with the 25-75 % range (for skewed metrics such as NEES)."""
    sub = rows[rows.policy == policy]
    tracks = [t for t in sub.track.unique() if t != "Fused"] + ["Fused"]
    scen = [s for s in SCENARIOS if s in sub.scenario.unique()]
    w = 0.8 / len(tracks)
    for j, tr in enumerate(tracks):
        vals = [sub[(sub.scenario == s) & (sub.track == tr)][metric].dropna() for s in scen]
        if robust:
            m = np.array([v.median() for v in vals])
            e = np.array([[mi - v.quantile(0.25), v.quantile(0.75) - mi] for v, mi in zip(vals, m)]).T
        else:
            m = [v.mean() for v in vals]
            e = [v.std() for v in vals]
        color = FUSED_COLOR if tr == "Fused" else LOCAL_COLORS[j % len(LOCAL_COLORS)]
        ax.bar(np.arange(len(scen)) + (j - (len(tracks) - 1) / 2) * w, m, w, yerr=e, capsize=2,
               label=tr, color=color, error_kw={"lw": 0.8})
    if ref is not None:
        ax.axhline(ref, color="k", ls="--", lw=1)
    ax.set_xticks(range(len(scen)), scen)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def make_figures(rows: pd.DataFrame, examples: dict, out: Path, policy="central_greedy"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    _bar(axes[0], rows, "coverage_net", policy, "Coverage in range (network mask)")
    _bar(axes[1], rows, "pos_rmse_m", policy, "Position RMSE [m]")
    _bar(axes[2], rows, "nees_mean", policy, "Episode NEES, median and 25-75 % (dashed: consistent = 6)",
         ref=6.0, robust=True)
    axes[0].set_ylim(0, 1)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, frameon=False)
    fig.suptitle(f"Local single-radar tracks vs. fused track ({policy}; coverage and RMSE: mean ± std over seeds)")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "local_vs_fused.png", dpi=150)
    plt.close(fig)

    for scen, env in examples.items():
        sources = [(f"{p.name}", ev) for p, ev in zip(env.radar_params, env.local_eval)] + [("Fused", env.evaluator)]
        n_t = len(env.targets)
        fig, axes = plt.subplots(n_t, 1, figsize=(12, 0.55 * len(sources) * n_t + 1), sharex=True)
        axes = np.atleast_1d(axes)
        t = np.arange(len(env.evaluator.valid[0])) * env.dt
        for ti, ax in enumerate(axes):
            cov = np.asarray(env.evaluator.in_cov[ti], bool)
            ax.fill_between(t, -0.5, len(sources) - 0.5, where=~cov, color="0.85", step="mid", lw=0)
            for k, (name, ev) in enumerate(sources):
                v = np.asarray(ev.valid[ti], bool)
                ax.fill_between(t, k - 0.35, k + 0.35, where=v, step="mid", lw=0,
                                color=FUSED_COLOR if name == "Fused" else "#3182bd")
            ax.set_yticks(range(len(sources)), [s for s, _ in sources], fontsize=7)
            ax.set_ylim(-0.6, len(sources) - 0.4)
            ax.set_ylabel(f"T{ti} (cls {env.targets[ti].label})", fontsize=7)
        axes[-1].set_xlabel("time [s]")
        fig.suptitle(f"{scen}: when each target has a valid track (grey: no radar can see it)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"timeline_{scen}.png", dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--terrain", default="procedural", help="procedural (offline, default) | auto")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows, episodes, examples = run_study(a.scenarios, a.policies, range(a.seeds), a.terrain)
    rows.to_csv(a.out / "tracks_per_episode.csv", index=False)
    episodes.to_csv(a.out / "episodes.csv", index=False)
    table = summary_table(rows)
    table.to_csv(a.out / "summary_local_vs_fused.csv", index=False)
    make_figures(rows, examples, a.out)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.to_string(index=False))
    print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()
