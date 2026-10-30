"""Flight-profile target classifier.

Model 1 of the project plan: engineered kinematic features + gradient
boosted trees (scikit-learn ``HistGradientBoostingClassifier``).  A temporal
1-D CNN on the raw window (Model 2) can be added later in ``tcn.py``; both
expose ``predict_proba(features)`` so the simulator does not care which one
is plugged in.

Per-window probabilities are accumulated over the life of a track with a
forgetting factor (a simple recursive Bayesian combination), which gives a
confidence that grows as evidence accumulates.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from src.core.targets import CLASS_NAMES

from .features import FEATURE_NAMES, window_features


class FlightProfileClassifier:
    def __init__(self, n_classes: int = len(CLASS_NAMES), window_s: float = 10.0, **gbm_kwargs):
        self.n_classes = n_classes
        self.window_s = window_s
        params = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                      l2_regularization=1e-3, random_state=0)
        params.update(gbm_kwargs)
        self.model = HistGradientBoostingClassifier(**params)
        self.fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        self.fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            return np.full((len(X), self.n_classes), 1.0 / self.n_classes)
        proba = np.zeros((len(X), self.n_classes))
        proba[:, self.model.classes_] = self.model.predict_proba(X)
        return proba

    def history_features(self, history, pos_std: float = 0.0) -> np.ndarray | None:
        """Feature vector of the most recent window of a track history, or
        None while the track is too short."""
        if len(history) < 6:
            return None
        t = np.array([h[0] for h in history])
        m = t >= t[-1] - self.window_s
        if m.sum() < 6 or t[-1] - t[m][0] < 0.6 * self.window_s:
            return None
        X = np.array([h[1] for h in history])[m]
        return window_features(t[m], X, pos_std)

    def classify_history(self, history, pos_std: float = 0.0) -> np.ndarray | None:
        """Class probabilities for the most recent window of a track history."""
        f = self.history_features(history, pos_std)
        return None if f is None else self.predict_proba(f[None, :])[0]

    def classify_tracks(self, tracks) -> dict:
        """One batched prediction for many tracks: {track id: probabilities}.
        Much faster than one call per track (tree ensembles have a large
        per-call overhead)."""
        ids, feats = [], []
        for tr in tracks:
            f = self.history_features(tr.history, tr.ekf.pos_std)
            if f is not None:
                ids.append(tr.id)
                feats.append(f)
        if not feats:
            return {}
        return dict(zip(ids, self.predict_proba(np.array(feats))))

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "fitted": self.fitted, "n_classes": self.n_classes,
                     "window_s": self.window_s, "features": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: str | Path) -> "FlightProfileClassifier":
        d = joblib.load(path)
        obj = cls(d["n_classes"], d["window_s"])
        obj.model, obj.fitted = d["model"], d["fitted"]
        return obj


def accumulate(prev: np.ndarray | None, new: np.ndarray, forget: float = 0.85,
               eps: float = 1e-3) -> np.ndarray:
    """Recursive log-probability accumulation with forgetting factor."""
    log_new = np.log(np.clip(new, eps, 1.0))
    if prev is None:
        log_post = log_new
    else:
        log_post = forget * np.log(np.clip(prev, eps, 1.0)) + log_new
    log_post -= log_post.max()
    p = np.exp(log_post)
    return p / p.sum()
