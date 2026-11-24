"""Classifier study: dataset checks, MLP vs GRU, fused vs single-radar tracks.

Answers three questions raised by the advisor (29 Oct 2026):

1. Do we generate a sound dataset?  Class balance, per-class feature
   statistics, a check that no single feature separates the classes on its
   own, and an episode-wise split (windows of one episode never appear in
   both training and test data).
2. Which classifier works better on track data: a feature MLP or a GRU that
   reads the window as a sequence?  (Gradient boosting is kept as reference.)
3. Does classification from the fused network track beat classification
   from a single radar's local track?  Every episode has several radars, so
   the fused track and the local tracks see the same target and detections.

A model trained on true target states is reported only as an optimistic
upper bound, never as a radar result.

    python -m src.services.classifier_study --episodes 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.core.env import RadarNetEnv
from src.core.scenario import DEFAULTS, _merge
from src.core.targets import CLASS_NAMES
from src.models.baselines.policies import make_policy
from src.models.classifier.features import FEATURE_NAMES, window_features

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "synthetic" / "classifier_study.npz"
REPORT_PATH = ROOT / "models" / "metadata" / "classifier_study.json"
SIM_MODEL_PATH = ROOT / "models" / "checkpoints" / "flight_profile_gbm.joblib"   # used by the simulator

WINDOW_S, STRIDE_S, SEQ_LEN = 10.0, 2.5, 20
SEQ_NAMES = ["vx", "vy", "vz", "speed", "alt_km", "pos_std_log"]


# ---------------------------------------------------------------- dataset
def _scenario(label: int, rng: np.random.Generator, duration_s: float, n_radars: int = 3) -> dict:
    """One target, several radars around it at random ranges and bearings."""
    radars = []
    for k in range(n_radars):
        bearing = rng.uniform(0, 2 * np.pi) if k == 0 else radars[0]["_b"] + k * 2 * np.pi / n_radars + rng.uniform(-0.4, 0.4)
        rng_m = rng.uniform(4000, 9000)
        pos = [rng_m * np.cos(bearing), rng_m * np.sin(bearing), 15]
        radars.append({"name": f"R{k + 1}", "position": pos,
                       "heading_deg": float(np.rad2deg(np.arctan2(-pos[1], -pos[0]))), "_b": bearing})
    for r in radars:
        r.pop("_b")
    return _merge(DEFAULTS, {
        "name": "classifier_study", "duration_s": duration_s, "area_radius_m": 3000,
        "radar_defaults": {"fov_az_deg": 60, "search_sector_deg": 120, "r_max_m": 14000},
        "radars": radars, "targets": [{"target_class": CLASS_NAMES[label]}], "max_track_slots": 3,
    })


def _sequence(t: np.ndarray, X: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Window resampled to SEQ_LEN steps; only quantities that do not depend on
    where the radars stand (velocities, speed, altitude, track uncertainty)."""
    tt = np.linspace(t[0], t[-1], SEQ_LEN)
    v = np.stack([np.interp(tt, t, X[:, k]) for k in (3, 4, 5)], 1)
    alt = np.interp(tt, t, X[:, 2]) / 1000.0
    ps = np.log10(1 + np.interp(tt, t, std))
    return np.column_stack([v / 100.0, np.linalg.norm(v, axis=1) / 100.0, alt, ps]).astype(np.float32)


def _windows(hist, stds):
    """Sliding windows of one track history -> (features, sequences)."""
    t = np.array([h[0] for h in hist])
    X = np.array([h[1] for h in hist])
    std = np.asarray(stds, float)
    feats, seqs = [], []
    start = t[0]
    while start + WINDOW_S <= t[-1] + 1e-9:
        m = (t >= start) & (t <= start + WINDOW_S)
        if m.sum() >= 6:
            feats.append(window_features(t[m], X[m], float(std[m].mean())))
            seqs.append(_sequence(t[m], X[m], std[m]))
        start += STRIDE_S
    return feats, seqs


EARLY_S = (3.0, 6.0)       # windows at the start of a fused track: 6 s = first classification in the simulator
# physically overlapping variant pairs (speed / altitude envelopes overlap): the hard cases
HARD_PAIRS = [("aircraft/fighter", "missile/subsonic_cruise"), ("uav/fixed_wing", "aircraft/transport")]


def _early(hist, stds, length_s):
    """The first ``length_s`` seconds of a track history as one window (None if too short)."""
    t = np.array([h[0] for h in hist])
    if t[-1] - t[0] < length_s - 1e-9:
        return None
    X = np.array([h[1] for h in hist])
    std = np.asarray(stds, float)
    m = t <= t[0] + length_s + 1e-9
    if m.sum() < 6:
        return None
    return window_features(t[m], X[m], float(std[m].mean())), _sequence(t[m], X[m], std[m])


def generate(n_per_class: int = 40, duration_s: float = 40.0, seed: int = 0, n_radars: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    out = {k: [] for k in ("feat", "seq", "y", "episode", "source", "radar", "variant")}
    ep = 0
    for label in range(len(CLASS_NAMES)):
        for _ in range(n_per_class):
            env = RadarNetEnv(_scenario(label, rng, duration_s, n_radars), seed=int(rng.integers(1 << 31)),
                              local_tracks=True)
            obs, _ = env.reset()
            pol = make_policy("central_greedy")
            pol.reset(env)
            truth, seen = [], {"fused": {}, **{f"r{i}": {} for i in range(n_radars)}}
            while env.agents:
                obs, *_ = env.step(pol.act(env, obs))
                tgt = env.targets[0]
                truth.append((env.t, np.concatenate([tgt.p, tgt.v])))
                for key, trk in [("fused", None)] + [(f"r{i}", lt) for i, lt in enumerate(env.local_trackers)]:
                    tr = env.match_targets(trk).get(0)
                    if tr is not None:
                        rec = seen[key].setdefault(tr.id, ([], []))
                        rec[0].append((env.t, tr.x.copy()))
                        rec[1].append(tr.ekf.pos_std)
            variant = f"{env.targets[0].target_class}/{env.targets[0].variant}"

            def add(f, s, src, radar):
                out["feat"].append(f); out["seq"].append(s); out["y"].append(label); out["episode"].append(ep)
                out["source"].append(src); out["radar"].append(radar); out["variant"].append(variant)

            for key, tracks in seen.items():
                for hist, stds in tracks.values():
                    if len(hist) < 2:
                        continue
                    F, S = _windows(hist, stds)
                    src = "fused" if key == "fused" else "local"
                    for f, s in zip(F, S):
                        add(f, s, src, key)
                    if key == "fused":                       # evaluation only: young tracks
                        for length in EARLY_S:
                            w = _early(hist, stds, length)
                            if w is not None:
                                add(*w, f"fused_early{length:g}", key)
            F, S = _windows(truth, np.zeros(len(truth)))         # upper bound: true states
            for f, s in zip(F, S):
                add(f, s, "truth", "truth")
            ep += 1
    return {k: np.array(v) for k, v in out.items()}


# ---------------------------------------------------------------- checks
def check_dataset(d: dict, test_eps: set) -> dict:
    """Balance, per-class feature means, single-feature separability, leakage."""
    from sklearn.tree import DecisionTreeClassifier

    fused = d["source"] == "fused"
    y, F, ep = d["y"][fused], d["feat"][fused], d["episode"][fused]
    counts = np.bincount(y, minlength=len(CLASS_NAMES))
    tr, te = ~np.isin(ep, list(test_eps)), np.isin(ep, list(test_eps))
    stumps = {}
    for j, name in enumerate(FEATURE_NAMES):                 # depth-2 tree on one feature only
        m = DecisionTreeClassifier(max_depth=2, random_state=0).fit(F[tr, j:j + 1], y[tr])
        stumps[name] = float((m.predict(F[te, j:j + 1]) == y[te]).mean())
    best = max(stumps, key=stumps.get)
    means = {CLASS_NAMES[c]: {n: float(F[y == c, j].mean()) for j, n in enumerate(FEATURE_NAMES)
                              if n in ("speed_mean", "acc_mean", "turn_rate_absmean", "alt_mean", "hover_frac")}
             for c in range(len(CLASS_NAMES))}
    return {
        "windows_per_class": dict(zip(CLASS_NAMES, counts.tolist())),
        "windows_per_variant": ({v: int(n) for v, n in zip(*np.unique(d["variant"][fused], return_counts=True))}
                                if "variant" in d else {}),
        "class_balance_min_over_max": float(counts.min() / max(counts.max(), 1)),
        "episodes_in_train_and_test": int(len(set(ep[tr]) & set(ep[te]))),
        "best_single_feature": best,
        "best_single_feature_accuracy": stumps[best],
        "class_feature_means": means,
    }


# ---------------------------------------------------------------- models
def ece(proba: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the top-class confidence."""
    conf, pred = proba.max(1), proba.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def _downsample(x, y, n: int = 60):
    idx = np.unique(np.linspace(0, len(x) - 1, min(n, len(x))).round().astype(int))
    return [np.asarray(x)[idx].round(4).tolist(), np.asarray(y)[idx].round(4).tolist()]


def auc_scores(proba, y) -> dict:
    """One-vs-rest ROC-AUC and PR-AUC (average precision) per class and their
    macro averages.  A class without positive or without negative test windows
    gets None and is left out of the average.  PR-AUC is reported next to
    ROC-AUC because with unbalanced classes ROC-AUC stays high even when the
    rare class is found with many false alarms; the chance level of PR-AUC is
    the class's share of the test windows (``prevalence``)."""
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    proba, y = np.asarray(proba, float), np.asarray(y, int)
    roc, pr, prev, curves = [], [], [], {"roc": {}, "pr": {}}
    for c in range(len(CLASS_NAMES)):
        pos = y == c
        prev.append(float(pos.mean()) if len(y) else None)
        if pos.all() or not pos.any():
            roc.append(None)
            pr.append(None)
            continue
        roc.append(float(roc_auc_score(pos, proba[:, c])))
        pr.append(float(average_precision_score(pos, proba[:, c])))
        fpr, tpr, _ = roc_curve(pos, proba[:, c])
        prec, rec, _ = precision_recall_curve(pos, proba[:, c])
        curves["roc"][CLASS_NAMES[c]] = _downsample(fpr, tpr)
        curves["pr"][CLASS_NAMES[c]] = _downsample(rec, prec)
    mean = lambda v: float(np.mean([x for x in v if x is not None])) if any(x is not None for x in v) else None
    return {"roc_auc_macro": mean(roc), "pr_auc_macro": mean(pr), "roc_auc_per_class": roc,
            "pr_auc_per_class": pr, "prevalence": prev, "curves": curves}


def _scores(proba, y) -> dict:
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
    pred = proba.argmax(1)
    return {"macro_f1": float(f1_score(y, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "ece": ece(proba, y), "n": int(len(y)),
            "confusion": confusion_matrix(y, pred, labels=range(len(CLASS_NAMES))).tolist(),
            **auc_scores(proba, y)}


def fit_gbm(X, y):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, random_state=0).fit(X, y)


def fit_mlp(X, y):
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # early stopping holds out 10 % for validation: only meaningful with enough
    # windows (tiny test datasets would leave fewer than two validation samples)
    early = len(y) >= 50
    return make_pipeline(StandardScaler(), MLPClassifier((64, 64), alpha=1e-3, max_iter=600,
                                                         early_stopping=early, random_state=0)).fit(X, y)


class GRUClassifier:
    """One-layer GRU over the window sequence, last hidden state -> classes."""

    def __init__(self, n_in: int, n_classes: int, hidden: int = 64, epochs: int = 40, seed: int = 0):
        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        self.torch = torch
        self.gru = nn.GRU(n_in, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_classes)
        self.epochs = epochs
        self.mu = self.sd = None

    def _t(self, S):
        return self.torch.as_tensor((S - self.mu) / self.sd, dtype=self.torch.float32)

    def _logits(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])

    def fit(self, S, y):
        torch = self.torch
        self.mu, self.sd = S.mean((0, 1)), S.std((0, 1)) + 1e-6
        x, yt = self._t(S), torch.as_tensor(y, dtype=torch.long)
        params = list(self.gru.parameters()) + list(self.head.parameters())
        opt = torch.optim.Adam(params, lr=3e-3)
        n = len(yt)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, 128):
                mb = perm[i:i + 128]
                loss = torch.nn.functional.cross_entropy(self._logits(x[mb]), yt[mb])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, S):
        with self.torch.no_grad():
            return self.torch.softmax(self._logits(self._t(S)), -1).numpy()


def run_study(d: dict, test_frac: float = 0.3, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    test_eps = set()                                         # episode-wise split, stratified by class
    for c in range(len(CLASS_NAMES)):
        eps = np.unique(d["episode"][d["y"] == c])
        test_eps |= set(rng.choice(eps, max(1, int(round(len(eps) * test_frac))), replace=False).tolist())
    is_te = np.isin(d["episode"], list(test_eps))
    report = {"dataset": check_dataset(d, test_eps), "results": []}

    def split(src):
        m = d["source"] == src
        tr = np.flatnonzero(m & ~is_te)
        # class-balanced training set: fast targets leave coverage sooner and give fewer windows
        n_min = min(np.sum(d["y"][tr] == c) for c in range(len(CLASS_NAMES)))
        if n_min > 0:
            tr = np.concatenate([rng.choice(tr[d["y"][tr] == c], n_min, replace=False)
                                 for c in range(len(CLASS_NAMES))])
        te = m & is_te
        return d["feat"][tr], d["seq"][tr], d["y"][tr], d["feat"][te], d["seq"][te], d["y"][te]

    data = {src: split(src) for src in ("fused", "local", "truth")}
    for train_src in ("fused", "local", "truth"):
        Ftr, Str, ytr, *_ = data[train_src]
        if len(np.unique(ytr)) < len(CLASS_NAMES):
            continue
        models = {"gbm": fit_gbm(Ftr, ytr), "mlp": fit_mlp(Ftr, ytr),
                  "gru": GRUClassifier(Str.shape[2], len(CLASS_NAMES)).fit(Str, ytr)}
        if train_src == "fused":
            fused_models = models
        for test_src in ("fused", "local") if train_src != "truth" else ("truth",):
            _, _, _, Fte, Ste, yte = data[test_src]
            if len(yte) == 0:
                continue
            for name, m in models.items():
                proba = m.predict_proba(Ste if name == "gru" else Fte)
                report["results"].append({"model": name, "train_on": train_src, "test_on": test_src,
                                          **_scores(proba, yte)})
    if "variant" in d and "fused_models" in locals():
        report.update(hard_case_analysis(d, is_te, fused_models))
    return report


def hard_case_analysis(d: dict, is_te: np.ndarray, models: dict) -> dict:
    """Where the easy overall score hides the hard cases (models trained on fused
    tracks, tested on fused tracks of held-out episodes):

    * by_variant: accuracy per target variant;
    * hard_pairs: the physically overlapping variant pairs, accuracy on their
      windows and how often each is taken for the other;
    * early: tracks seen only for their first 3 s or 6 s (6 s is when the
      simulator first classifies a track).
    Only GBM (the simulator's model) and GRU (the best sequence model)."""
    from sklearn.metrics import f1_score
    out = {"by_variant": {}, "hard_pairs": [], "early": []}
    m = (d["source"] == "fused") & is_te
    y, var = d["y"][m], d["variant"][m]
    pred = {"gbm": models["gbm"].predict_proba(d["feat"][m]).argmax(1),
            "gru": models["gru"].predict_proba(d["seq"][m]).argmax(1)}
    for v in sorted(set(var)):
        k = var == v
        out["by_variant"][v] = {"n": int(k.sum()), **{f"{n}_accuracy": float((p[k] == y[k]).mean())
                                                       for n, p in pred.items()}}
    for a, b in HARD_PAIRS:
        ka, kb = var == a, var == b
        if not (ka.any() and kb.any()):
            continue
        la, lb = int(y[ka][0]), int(y[kb][0])
        row = {"pair": f"{a} vs {b}", "n": [int(ka.sum()), int(kb.sum())]}
        for n, p in pred.items():
            row[f"{n}_accuracy"] = float(np.concatenate([p[ka] == la, p[kb] == lb]).mean())
            row[f"{n}_{a}_taken_for_{b.split('/')[0]}"] = float((p[ka] == lb).mean())
            row[f"{n}_{b}_taken_for_{a.split('/')[0]}"] = float((p[kb] == la).mean())
        # the advisor's check on the hard pair: can one feature alone separate it?
        from sklearn.tree import DecisionTreeClassifier
        tr = (d["source"] == "fused") & ~is_te & np.isin(d["variant"], [a, b])
        te = m.copy(); te[m] = ka | kb
        best = 0.0
        for j, name in enumerate(FEATURE_NAMES):
            st = DecisionTreeClassifier(max_depth=2, random_state=0).fit(d["feat"][tr, j:j + 1], d["y"][tr])
            acc = float((st.predict(d["feat"][te, j:j + 1]) == d["y"][te]).mean())
            if acc > best:
                best, row["best_single_feature"] = acc, name
        row["best_single_feature_accuracy"] = best
        out["hard_pairs"].append(row)
    for length in EARLY_S:
        k = (d["source"] == f"fused_early{length:g}") & is_te
        if not k.any():
            continue
        ye = d["y"][k]
        row = {"window_s": length, "n": int(k.sum())}
        for n, p in (("gbm", models["gbm"].predict_proba(d["feat"][k]).argmax(1)),
                     ("gru", models["gru"].predict_proba(d["seq"][k]).argmax(1))):
            row[f"{n}_macro_f1"] = float(f1_score(ye, p, average="macro", labels=range(len(CLASS_NAMES))))
            row[f"{n}_accuracy"] = float((p == ye).mean())
        out["early"].append(row)
    return out


def train_simulation_model(d: dict, path: Path = SIM_MODEL_PATH, seed: int = 0) -> dict:
    """Train the simulator's classifier on fused-track windows (the evidence the
    fusion centre actually has), class-balanced, and save it where the
    dashboard and RadarNetEnv load it from."""
    from src.models.classifier.classifier import FlightProfileClassifier

    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(d["source"] == "fused")
    if len(idx) == 0:
        raise ValueError("no fused-track windows in the dataset")
    n_min = min(np.sum(d["y"][idx] == c) for c in range(len(CLASS_NAMES)))
    if n_min > 0:                       # class-balanced; with a missing class, train on what exists
        idx = np.concatenate([rng.choice(idx[d["y"][idx] == c], n_min, replace=False)
                              for c in range(len(CLASS_NAMES))])
    clf = FlightProfileClassifier(window_s=WINDOW_S).fit(d["feat"][idx], d["y"][idx])
    clf.save(path)
    meta = {"trained_on": "fused tracks of multi-radar episodes (classifier_study)",
            "n_train": int(len(idx)), "windows_per_class": int(n_min)}
    meta_path = path.parent.parent / "metadata" / (path.stem + ".json")
    old = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**old, **meta}, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=40, help="episodes per class")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--from-data", action="store_true",
                    help="reuse data/synthetic/classifier_study.npz instead of simulating again")
    ap.add_argument("--no-sim-model", action="store_true",
                    help="do not replace the simulator's classifier with the fused-track GBM")
    args = ap.parse_args()
    if args.from_data and DATA_PATH.exists():
        d = dict(np.load(DATA_PATH))
    else:
        d = generate(args.episodes, seed=args.seed)
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(DATA_PATH, **d)
    if not args.no_sim_model:
        print("simulator classifier:", train_simulation_model(d, seed=args.seed), "->", SIM_MODEL_PATH)
    rep = run_study(d, seed=args.seed)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep["dataset"], indent=1)[:1500])
    for key in ("by_variant", "hard_pairs", "early"):
        if key in rep:
            print(key, json.dumps(rep[key], indent=1))
    for r in rep["results"]:
        print(f"{r['model']:4s} train={r['train_on']:6s} test={r['test_on']:6s} "
              f"macroF1={r['macro_f1']:.3f} balAcc={r['balanced_accuracy']:.3f} ECE={r['ece']:.3f} "
              f"ROC-AUC={r['roc_auc_macro'] or float('nan'):.3f} PR-AUC={r['pr_auc_macro'] or float('nan'):.3f} n={r['n']}")


if __name__ == "__main__":
    main()
