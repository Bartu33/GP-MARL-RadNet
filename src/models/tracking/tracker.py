"""Multi-radar track manager with centralised measurement fusion.

Every radar report is processed sequentially by one shared EKF per track
(centralised measurement-level fusion, weighting each report by its own
covariance R).  Data association is greedy global-nearest-neighbour with a
chi-square gate on the Mahalanobis distance; a report outside every gate is
not used to update any track (outlier rejection) and starts a new tentative
track instead.  Track management:

* confirmation -- M-of-N: a tentative track is confirmed when it received at
  least ``confirm_hits`` (M) associated reports during the last
  ``confirm_window`` (N) tracker steps.  Reports of different radars in the
  same step count separately: two radars seeing the same object at the same
  place is independent evidence, while one radar can add at most one report
  per track and step.
* coasting -- a track that gets no report keeps being predicted (its
  covariance grows); a single miss never deletes it.
* deletion -- a track is deleted when it has not been updated for
  ``tentative_timeout_s`` / ``confirmed_timeout_s`` seconds, or when its
  position uncertainty exceeds ``max_pos_std_m``.

Choosing N is a trade-off.  A radar in search mode revisits a target only
every few seconds, so a short window (e.g. 3-of-5) leaves many real tracks
unconfirmed; a long one confirms slowly-revisited targets but also keeps more
false tracks.  Measured with central_greedy, 5 seeds (dt = 0.5 s):

    N (steps)   coverage_in_range        false confirmed tracks / step
                nominal   high_noise     nominal   high_noise
       8         0.79       0.44          0.71       0.52
      16         0.79       0.50          0.61       0.84
      24         0.81       0.57          0.82       0.96
      48         0.84       0.63          0.83       0.98

The default N = 48 (24 s) reproduces the earlier "3 hits, no window" rule
(a tentative track is deleted after 8 s without reports anyway).  It can be
set per scenario under ``tracker: {confirm_hits, confirm_window}``.  The tracker never sees simulation
truth: ``truth_id`` / ``is_false_alarm`` fields of the reports are ignored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ekf import EKF

GATE_CHI2_4DOF_99 = 13.28
SNR_HISTORY = 10          # reports / requested dwells kept per radar and track


@dataclass
class Track:
    id: int
    ekf: EKF
    born_t: float
    last_update_t: float
    hits: int = 1                                     # number of updates (all radars)
    confirmed: bool = False
    hit_steps: list = field(default_factory=list)     # tracker step of every update (one entry per report)
    history: list = field(default_factory=list)       # [(t, x(6)), ...]
    updated_by: dict = field(default_factory=dict)    # radar index -> last time
    snr: dict = field(default_factory=dict)           # radar index -> [(t, SNR dB)] of associated reports
    dwells: dict = field(default_factory=dict)        # radar index -> [(t, detected)] of requested dwells
    class_probs: np.ndarray | None = None

    @property
    def x(self):
        return self.ekf.x

    @property
    def P(self):
        return self.ekf.P

    def record_snr(self, radar: int, t: float, snr_db, keep: int = SNR_HISTORY):
        if snr_db is None:
            return
        h = self.snr.setdefault(radar, [])
        h.append((t, float(snr_db)))
        del h[:-keep]

    def record_dwell(self, radar: int, t: float, detected: bool, keep: int = SNR_HISTORY):
        h = self.dwells.setdefault(radar, [])
        h.append((t, bool(detected)))
        del h[:-keep]

    def coasting(self, t: float) -> bool:
        """True when the track was not updated at time ``t`` (predicted only)."""
        return self.last_update_t < t


class TrackManager:
    def __init__(self, q: float = 40.0, confirm_hits: int = 3, confirm_window: int = 48,
                 tentative_timeout_s: float = 8.0,
                 confirmed_timeout_s: float = 15.0, max_pos_std_m: float = 2500.0,
                 init_min_quality: float = 0.0, history_len: int = 400):
        self.q = q
        if confirm_window < confirm_hits:
            raise ValueError("confirm_window (N) must be at least confirm_hits (M)")
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.tentative_timeout_s = tentative_timeout_s
        self.confirmed_timeout_s = confirmed_timeout_s
        self.max_pos_std_m = max_pos_std_m
        self.init_min_quality = init_min_quality
        self.history_len = history_len
        self.tracks: list[Track] = []
        self._next_id = 1
        # report -> track id of the last process() call, keyed by id(report["z"]);
        # read by the fusion centre to link local tracks to integrated tracks (PPI scopes)
        self.last_assoc: dict[int, int] = {}
        self.step = 0                                  # tracker steps processed so far

    def predict(self, dt: float):
        for tr in self.tracks:
            tr.ekf.predict(dt)

    def process(self, t: float, detections: list[dict], radars):
        """Associate and fuse all reports of this step.  ``radars`` is the list
        of RadarParams (positions/headings needed by the measurement model)."""
        self.last_assoc = {}
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
                self.last_assoc[id(d["z"])] = tr.id
                tr.record_snr(ri, t, d.get("snr_db"))
                tr.hit_steps.append(self.step)
                if not tr.confirmed and self._m_of_n(tr):
                    tr.confirmed = True
            for di, d in enumerate(dets):
                if di in used_d or d["signal_quality"] < self.init_min_quality:
                    continue
                ekf = EKF.from_detection(d, pos, hd, q=self.q)
                tr = Track(self._next_id, ekf, t, t, updated_by={ri: t}, hit_steps=[self.step])
                tr.record_snr(ri, t, d.get("snr_db"))
                self.tracks.append(tr)
                self.last_assoc[id(d["z"])] = self._next_id
                self._next_id += 1

        self._prune(t)
        self.step += 1
        for tr in self.tracks:
            tr.history.append((t, tr.ekf.x.copy()))
            if len(tr.history) > self.history_len:
                tr.history.pop(0)

    def _m_of_n(self, tr: Track) -> bool:
        """At least M reports during the last N tracker steps (including the current one)."""
        recent = [s for s in tr.hit_steps if s > self.step - self.confirm_window]
        tr.hit_steps = recent                          # older steps can never count again
        return len(recent) >= self.confirm_hits

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
