"""Multi-radar track manager with centralised measurement fusion.

Every radar report is processed sequentially by one shared EKF per track
(centralised measurement-level fusion, weighting each report by its own
covariance R).  Data association is greedy global-nearest-neighbour with a
chi-square gate on the Mahalanobis distance.  Simple M-hit confirmation and
time-out deletion manage the track table.  The tracker never sees simulation
truth: ``truth_id`` / ``is_false_alarm`` fields of the reports are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ekf import EKF

GATE_CHI2_4DOF_99 = 13.28


@dataclass
class Track:
    id: int
    ekf: EKF
    born_t: float
    last_update_t: float
    hits: int = 1
    confirmed: bool = False
    history: list = field(default_factory=list)       # [(t, x(6)), ...]
    updated_by: dict = field(default_factory=dict)    # radar index -> last time
    class_probs: np.ndarray | None = None

    @property
    def x(self):
        return self.ekf.x

    @property
    def P(self):
        return self.ekf.P


class TrackManager:
    def __init__(self, q: float = 40.0, confirm_hits: int = 3, tentative_timeout_s: float = 8.0,
                 confirmed_timeout_s: float = 15.0, max_pos_std_m: float = 2500.0,
                 init_min_quality: float = 0.0, history_len: int = 400):
        self.q = q
        self.confirm_hits = confirm_hits
        self.tentative_timeout_s = tentative_timeout_s
        self.confirmed_timeout_s = confirmed_timeout_s
        self.max_pos_std_m = max_pos_std_m
        self.init_min_quality = init_min_quality
        self.history_len = history_len
        self.tracks: list[Track] = []
        self._next_id = 1

    def predict(self, dt: float):
        for tr in self.tracks:
            tr.ekf.predict(dt)

    def process(self, t: float, detections: list[dict], radars):
        """Associate and fuse all reports of this step.  ``radars`` is the list
        of RadarParams (positions/headings needed by the measurement model)."""
        by_radar: dict[int, list[dict]] = {}
        for d in detections:
            by_radar.setdefault(d["radar"], []).append(d)

        for ri, dets in by_radar.items():
            rp = radars[ri]
            pos, hd = rp.pos, rp.heading
            pairs = []
            for di, d in enumerate(dets):
                for ti, tr in enumerate(self.tracks):
                    d2 = tr.ekf.mahalanobis2(d["z"], d["measurement_covariance"], pos, hd)
                    if d2 < GATE_CHI2_4DOF_99:
                        pairs.append((d2, di, ti))
            pairs.sort()
            used_d, used_t = set(), set()
            for d2, di, ti in pairs:
                if di in used_d or ti in used_t:
                    continue
                used_d.add(di)
                used_t.add(ti)
                tr = self.tracks[ti]
                d = dets[di]
                tr.ekf.update(d["z"], d["measurement_covariance"], pos, hd)
                tr.hits += 1
                tr.last_update_t = t
                tr.updated_by[ri] = t
                if tr.hits >= self.confirm_hits:
                    tr.confirmed = True
            for di, d in enumerate(dets):
                if di in used_d or d["signal_quality"] < self.init_min_quality:
                    continue
                ekf = EKF.from_detection(d, pos, hd, q=self.q)
                self.tracks.append(Track(self._next_id, ekf, t, t, updated_by={ri: t}))
                self._next_id += 1

        self._prune(t)
        for tr in self.tracks:
            tr.history.append((t, tr.ekf.x.copy()))
            if len(tr.history) > self.history_len:
                tr.history.pop(0)

    def _prune(self, t: float):
        keep = []
        for tr in self.tracks:
            silent = t - tr.last_update_t
            timeout = self.confirmed_timeout_s if tr.confirmed else self.tentative_timeout_s
            if silent > timeout or tr.ekf.pos_std > self.max_pos_std_m:
                continue
            keep.append(tr)
        self.tracks = keep

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]
