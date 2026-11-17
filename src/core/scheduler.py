"""Local radar scheduler: answers the fusion centre's track-update requests.

The fusion centre (later: the RL task manager) does not set radar modes.  It
sends *requests* ("update track j, with this priority, before this time") and
each radar's own scheduler decides, with deterministic rules, how to answer
and what to do in each step:

* ACCEPTED -- served in the next step with a full track dwell;
* DELAYED  -- queued and served in a later step, still before the deadline;
* PARTIAL  -- served, but with the shorter INTERLEAVED dwell (search sweep +
  short track dwell) because the radar cannot afford a full track dwell now
  (energy or heat), so the update is less accurate;
* REJECTED -- not served, with a reason: radar offline, track outside this
  radar's coverage, no free tracking channel (queue full), deadline cannot be
  met, or not enough energy / heat margin even for the short dwell.

Scheduling rule: one dwell per step, earliest deadline first (ties: higher
priority first, then older request).  Every ``search_period``-th step is
reserved for search so that a busy radar keeps surveying for new targets
(the same reservation as the ``central_greedy`` baseline).  A request whose
deadline passes in the queue, or whose track is deleted by the tracker,
expires.  The queue length is limited by the radar's tracking channels
(``RadarParams.c_max``).  This follows the priority / time-budget scheduling
of multifunction radars (MathWorks, "Adaptive tracking of maneuvering targets
with managed radar") at the abstraction level of this simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from . import geometry as geo
from .modes import INTERLEAVE_SEARCH_BEAMS, Mode
from .radar import RadarAction, RadarParams, RadarState


class Response(str, Enum):
    ACCEPTED = "accepted"
    DELAYED = "delayed"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class Request:
    """A track-update request from the fusion centre to one radar."""
    track_id: int
    deadline_s: float                 # latest simulation time at which the update is useful
    priority: float = 1.0             # larger = more important
    issued_t: float = 0.0


@dataclass
class Answer:
    request: Request
    response: Response
    reason: str = ""
    planned_t: float | None = None    # when the scheduler expects to serve it


@dataclass
class _Queued:
    req: Request
    partial: bool


def _coverable(p: RadarParams, tr) -> bool:
    r, a, e, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
    return (abs(a) <= np.deg2rad(p.fov_az_deg) and p.r_min_m <= r <= p.r_max_m
            and np.deg2rad(p.el_min_deg) <= e <= np.deg2rad(p.el_max_deg))


def dwell_affordable(p: RadarParams, s: RadarState, mode: Mode) -> bool:
    """Same feasibility test as ``radar_step`` (energy and heat after the dwell)."""
    if mode == Mode.TRACK:
        dwell = p.profiles[Mode.TRACK].dwell_s
    else:
        dwell = (INTERLEAVE_SEARCH_BEAMS * p.profiles[Mode.SEARCH].dwell_s
                 + p.profiles[Mode.INTERLEAVED].dwell_s)
    cost = p.pt_w * dwell / p.efficiency + p.e_processing_j
    return cost <= s.energy_j and s.heat_k + p.thermal_a * p.pt_w * dwell <= p.h_max_k


@dataclass
class LocalScheduler:
    params: RadarParams
    dt: float
    search_period: int = 3            # every n-th step is a search step (0 = no reservation)
    low_energy_frac: float = 0.15     # below this the radar rests when it has nothing to serve
    queue: list = field(default_factory=list)
    step_index: int = 0
    counts: dict = field(default_factory=lambda: {r.value: 0 for r in Response} | {"expired": 0, "served": 0})
    reasons: dict = field(default_factory=dict)
    last_served: int | None = None    # track id served in the last step (None: search / idle)

    # -------------------------------------------------------------- requests
    def _is_search_step(self, k: int) -> bool:
        return self.search_period > 0 and k % self.search_period == 0

    def _order(self, items):
        return sorted(items, key=lambda q: (q.req.deadline_s, -q.req.priority, q.req.issued_t))

    def _planned_times(self, t: float, items) -> dict:
        """Simulate the queue: time at which each queued request would be served."""
        out, k, now = {}, self.step_index + 1, t
        for q in self._order(items):
            now += self.dt
            while self._is_search_step(k):
                k += 1
                now += self.dt
            out[id(q)] = now
            k += 1
        return out

    def submit(self, t: float, requests, tracks: dict, state: RadarState) -> list[Answer]:
        answers = []
        for req in requests:
            ans = self._submit_one(t, req, tracks, state)
            self.counts[ans.response.value] += 1
            if ans.response == Response.REJECTED:
                self.reasons[ans.reason] = self.reasons.get(ans.reason, 0) + 1
            answers.append(ans)
        return answers

    def _reject(self, req, reason):
        return Answer(req, Response.REJECTED, reason)

    def _submit_one(self, t, req: Request, tracks: dict, state: RadarState) -> Answer:
        p = self.params
        tr = tracks.get(req.track_id)
        if not state.online:
            return self._reject(req, "offline")
        if tr is None:
            return self._reject(req, "unknown_track")
        if not _coverable(p, tr):
            return self._reject(req, "out_of_coverage")
        partial = not dwell_affordable(p, state, Mode.TRACK)
        if partial and not dwell_affordable(p, state, Mode.INTERLEAVED):
            return self._reject(req, "resources")
        # a new request for a track already queued replaces the old one
        others = [q for q in self.queue if q.req.track_id != req.track_id]
        if len(others) >= p.c_max:
            return self._reject(req, "queue_full")
        new = _Queued(req, partial)
        plan = self._planned_times(t, others + [new])
        if plan[id(new)] > req.deadline_s + 1e-9:
            return self._reject(req, "deadline")
        self.queue = others + [new]
        if partial:
            resp = Response.PARTIAL
        elif plan[id(new)] <= t + self.dt + 1e-9:
            resp = Response.ACCEPTED
        else:
            resp = Response.DELAYED
        return Answer(req, resp, planned_t=plan[id(new)])

    # -------------------------------------------------------------- one step
    def next_action(self, t: float, tracks: dict, state: RadarState) -> tuple[RadarAction, float | None]:
        """Action for the step that ends at time ``t``.  Returns (action, beam range
        limit in m or None)."""
        p = self.params
        self.step_index += 1
        self.last_served = None
        keep = []
        for q in self.queue:                                  # expire stale requests
            if q.req.deadline_s < t - 1e-9 or q.req.track_id not in tracks:
                self.counts["expired"] += 1
            else:
                keep.append(q)
        self.queue = keep
        if not state.online:
            return RadarAction(Mode.SEARCH), None
        if self.queue and not self._is_search_step(self.step_index):
            for q in self._order(self.queue):
                tr = tracks[q.req.track_id]
                if not _coverable(p, tr):                     # the target left the coverage meanwhile
                    continue
                mode = Mode.TRACK
                if q.partial or not dwell_affordable(p, state, Mode.TRACK):
                    mode = Mode.INTERLEAVED
                    if not dwell_affordable(p, state, Mode.INTERLEAVED):
                        break                                 # cannot transmit now; try next step
                self.queue.remove(q)
                self.counts["served"] += 1
                self.last_served = q.req.track_id
                r, a, e, _ = geo.measure(tr.x[:3], tr.x[3:], p.pos, p.heading)
                return RadarAction(mode, beam=(a, e)), float(r) * 1.15
        if state.energy_j / p.e_max_j < self.low_energy_frac:
            return RadarAction(Mode.IDLE), None
        return RadarAction(Mode.SEARCH), None

    def summary(self) -> dict:
        n = sum(self.counts[r.value] for r in Response)
        return {"requests": n, **self.counts, "reasons": dict(self.reasons)}
