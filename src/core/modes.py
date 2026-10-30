"""Radar operating modes and search-beam scheduling.

Three modes are modelled at the abstraction level of the detection note
(no waveforms):

* ``SEARCH`` -- the beam steps through a raster of azimuth positions and
  elevation bars covering the radar's search sector.  Each look visits
  ``beams_per_look`` neighbouring raster cells with a short dwell, so the
  radar covers a wide area with low integration gain.  Detections made in
  search also update existing tracks (track-while-scan behaviour).
* ``TRACK``  -- one long dwell centred on the predicted position of a chosen
  track.  More pulses are integrated (higher SNR, better accuracy), but the
  radar spends more energy and sees only what is inside that beam.
* ``INTERLEAVED`` -- time-shared search and track in the same step: a short
  search sweep (``INTERLEAVE_SEARCH_BEAMS`` raster cells) plus one medium
  track dwell.  It balances ("equalises") the radar's time and energy between
  surveillance and tracking.  This is our interpretation of the mode called
  *equalization* in the meeting notes; see doc/research/radar_modes.md.
* ``IDLE``   -- the radar does not transmit; it recovers energy and cools.

A radar may also be forced into IDLE by the resource model (empty energy
budget or thermal limit) or be ``OFFLINE`` after an agent failure.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class Mode(IntEnum):
    IDLE = 0
    SEARCH = 1
    TRACK = 2
    OFFLINE = 3
    INTERLEAVED = 4


@dataclass(frozen=True)
class ModeProfile:
    n_pulses: int          # pulses integrated per beam position (coherent gain)
    pri_s: float           # pulse repetition interval
    beams_per_look: int    # beam positions visited in one simulator step

    @property
    def dwell_s(self) -> float:
        """Dwell per beam position = coherent processing interval T_CPI."""
        return self.n_pulses * self.pri_s

    @property
    def total_dwell_s(self) -> float:
        return self.dwell_s * self.beams_per_look


DEFAULT_PROFILES = {
    Mode.SEARCH: ModeProfile(n_pulses=8, pri_s=250e-6, beams_per_look=8),
    Mode.TRACK: ModeProfile(n_pulses=96, pri_s=250e-6, beams_per_look=1),
    # track part of an interleaved look; the search part uses the SEARCH profile
    Mode.INTERLEAVED: ModeProfile(n_pulses=48, pri_s=250e-6, beams_per_look=1),
}
INTERLEAVE_SEARCH_BEAMS = 4


class SearchRaster:
    """Cyclic raster of (azimuth, elevation) beam centres in the sensor frame."""

    def __init__(self, sector_deg: float, beamwidth_az_deg: float,
                 elevation_bars_deg=(3.0, 9.0, 15.0)):
        half = np.deg2rad(sector_deg) / 2.0
        step = np.deg2rad(beamwidth_az_deg)
        n_az = max(1, int(np.ceil(2 * half / step)))
        az = np.linspace(-half + step / 2, half - step / 2, n_az) if n_az > 1 else np.array([0.0])
        cells = []
        for k, el in enumerate(np.deg2rad(elevation_bars_deg)):
            row = az if k % 2 == 0 else az[::-1]        # serpentine scan
            cells.extend((float(a), float(el)) for a in row)
        self.cells = cells
        self.pointer = 0

    def next_positions(self, n: int):
        out = [self.cells[(self.pointer + i) % len(self.cells)] for i in range(n)]
        self.pointer = (self.pointer + n) % len(self.cells)
        return out

    @property
    def revisit_looks(self) -> int:
        return len(self.cells)
