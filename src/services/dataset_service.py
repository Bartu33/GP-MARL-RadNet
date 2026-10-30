"""Generate a labelled flight-profile dataset and train the classifier.

Each sample episode puts one synthetic target of a random class in front of
one radar.  The radar runs the full detection model and the EKF tracker,
either tracking the target (high-quality track) or only searching (noisier
track), so the classifier is trained on realistic *tracker output*.  The
longest-lived track that follows the target is cut into sliding windows.

    python -m src.services.dataset_service --episodes 120 --train
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.env import RadarNetEnv
from src.core.scenario import DEFAULTS, _merge
from src.core.targets import CLASS_NAMES
from src.models.classifier.features import FEATURE_NAMES, sliding_windows

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "flight_profile_windows.csv"
MODEL_PATH = ROOT / "models" / "checkpoints" / "flight_profile_gbm.joblib"


def _episode_scenario(label: int, rng: np.random.Generator, duration_s: float) -> dict:
    ang = rng.uniform(-0.6, 0.6)
    rng_m = rng.uniform(2500, 9000)
    return _merge(DEFAULTS, {
        "name": "dataset", "duration_s": duration_s, "area_radius_m": 4000,
        "radar_defaults": {"fov_az_deg": 60, "search_sector_deg": 120},
        "radars": [{"name": "R1", "position": [-rng_m * np.cos(ang), -rng_m * np.sin(ang), 15],
                    "heading_deg": float(np.rad2deg(ang))}],
        "targets": [{"label": label}],
        "max_track_slots": 3,
    })


def generate(n_per_class: int = 60, duration_s: float = 80.0, window_s: float = 10.0,
             seed: int = 0, track_fraction: float = 0.6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    ep = 0
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            sc = _episode_scenario(label, rng, duration_s)
            env = RadarNetEnv(sc, seed=int(rng.integers(1 << 31)))
            env.reset()
            use_track = rng.random() < track_fraction
            best = {}
            while env.agents:
                a = 2 if (use_track and env.slots[0] is not None) else 1
                env.step({"radar_0": a})
                for ti, tr in env.match_targets().items():
                    best[tr.id] = tr
            if not best:
                ep += 1
                continue
            tr = max(best.values(), key=lambda t: len(t.history))
            t = np.array([h[0] for h in tr.history])
            X = np.array([h[1] for h in tr.history])
            F, ends = sliding_windows(t, X, window_s=window_s, stride_s=2.5)
            for f, e in zip(F, ends):
                rows.append([ep, label, int(use_track), e, *f])
            ep += 1
    return pd.DataFrame(rows, columns=["episode", "label", "track_mode", "t_end", *FEATURE_NAMES])


def train(df: pd.DataFrame, model_path: Path = MODEL_PATH, test_frac: float = 0.25, seed: int = 0):
    """Episode-wise train/test split (windows of one episode never straddle
    the split), fit the GBM, report macro-F1 and save the model."""
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

    from src.models.classifier.classifier import FlightProfileClassifier

    rng = np.random.default_rng(seed)
    eps = df["episode"].unique()
    test_eps = set(rng.choice(eps, int(len(eps) * test_frac), replace=False))
    is_test = df["episode"].isin(test_eps)
    Xtr, ytr = df.loc[~is_test, FEATURE_NAMES].values, df.loc[~is_test, "label"].values
    Xte, yte = df.loc[is_test, FEATURE_NAMES].values, df.loc[is_test, "label"].values
    clf = FlightProfileClassifier().fit(Xtr, ytr)
    pred = clf.predict_proba(Xte).argmax(1)
    report = {
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(yte, pred)),
        "confusion": confusion_matrix(yte, pred, labels=range(len(CLASS_NAMES))).tolist(),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }
    clf.save(model_path)
    meta = model_path.parent.parent / "metadata" / (model_path.stem + ".json")
    meta.parent.mkdir(parents=True, exist_ok=True)
    meta.write_text(json.dumps(report, indent=2))
    return clf, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60, help="episodes per class")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    df = generate(args.episodes, seed=args.seed)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"saved {len(df)} windows -> {DATA_PATH}")
    if args.train:
        _, rep = train(df)
        print({k: v for k, v in rep.items() if k != "confusion"})
        print("confusion (rows=true):")
        for row in rep["confusion"]:
            print("  ", row)


if __name__ == "__main__":
    main()
