# Radar operating modes (Task 8)

Implemented in `src/core/modes.py`, used by `src/core/radar.py::radar_step`
and selected by the agents through the environment's action space
(`src/core/env.py`).

| Mode | What the radar does in one step (0.5 s) | Effect in the model |
|---|---|---|
| **SEARCH** | Visits 8 consecutive cells of a serpentine azimuth × elevation raster over its search sector; 32 pulses (2 ms) per cell. | Wide coverage, low integration gain → lower SNR and larger measurement covariance. Revisit time = raster cells / 8 (default 80 / 8 = 10 steps = 5 s). |
| **TRACK** | One long dwell (384 pulses, 24 ms) on the *predicted* position of a chosen track. | +10.8 dB coherent gain vs. a search cell and finer radial-velocity resolution (λ / 2T_CPI), but only targets inside that beam are seen and more energy and heat are spent. |
| **INTERLEAVED** | Time-shared: 4 search cells (32 pulses each) **and** one medium track dwell (192 pulses, 12 ms) in the same step. | Keeps surveying while tracking; +7.8 dB on the tracked target vs. a search cell; costs between a search look and a track dwell. |
| **IDLE** | No transmission. | Recovers energy and cools down. Also forced automatically when the energy budget or the thermal limit would be exceeded (the search raster does not advance then). |
| **OFFLINE** | Agent failure (stress test). | No reports. |

**Waveform.** All modes use a medium-PRF stagger of 14, 16 and 18 kHz. Every look (each search
cell and each track dwell) is split into three coherent sub-dwells, one per PRF, each with a third of
the pulses (a 384-pulse track dwell = 3 × 128). A target is reported when at least **2 of the 3**
sub-dwells detect it; its range and radial velocity are then unfolded over the PRFs that detected it
(Chinese remainder). Over all three PRFs range is unambiguous to 74.9 km and radial velocity to
±7555 m/s; over a pair only to ±839–1079 m/s, so a very fast target seen by just two PRFs is reported
with folded radial velocity. A sub-dwell whose PRF puts the target in the ±1 m/s clutter notch cannot
detect it (blind speed); with one PRF blind the other two must both detect, and a zero-Doppler
(tangential) target is never reported. Each sub-dwell has a third of the look's SNR, so P_D = 0.5 needs
about 16 dB of look SNR instead of 11.5 dB for one coherent dwell; Doppler resolution is that of a
sub-dwell (λ / 2 T_sub), and the accuracy of a report uses the SNR of the sub-dwells that detected it.
Dwell times, and so energy, heat and time budgets, are those of the earlier 4 kHz waveform.

**Track-while-scan** comes for free: every detection — from a search,
track or interleaved look — goes to the tracker, so search also refreshes
tracks.

## Actions

| Action | Default scenarios | With `interleaved_actions: true` |
|---|---|---|
| `0` | IDLE | IDLE |
| `1` | SEARCH | SEARCH |
| `2 … K+1` | TRACK slot k | TRACK slot k |
| `K+2 … 2K+1` | — | INTERLEAVED slot k |

The default action space is unchanged, so models trained on the default
scenarios (e.g. `mappo.pt`) keep working. The `interleaved` scenario turns the
extra actions on.

## Resource trade-off

Physics note Section 10: a search look costs ~33 J, a track dwell ~49 J, an
interleaved look ~41 J, and the radar recovers 30 J per step. A radar can
track continuously for only ~25 s before it must search, interleave or rest,
and it also heats up. This is the trade-off the MARL agents should learn.

## The "equalization" mode

The meeting notes mention a mode called *equalization*. We found no radar
mode with this name in the resource-management literature (searches on phased-array
search/track scheduling describe search, track, track-while-scan and
"tracking-and-searching" modes, e.g. the multifunction-radar scheduling
example in the MathWorks Radar Toolbox documentation). Two readings are
plausible, and **both are now implemented** so the team can keep whichever
the advisor confirms:

1. **Equalising time between search and track inside one radar** →
   `Mode.INTERLEAVED` above (search sweep + track dwell in every step).
2. **Equalising the load across the radars of the network** → the
   *usage fairness* metric: Jain's index
   J = (Σ xᵢ)² / (n Σ xᵢ²) of each radar's cumulative tracking time
   (1 = perfectly balanced, 1/n = one radar does all the tracking). It is
   reported for every episode (`usage_fairness` on the Experiments tab) and can
   be added to the team reward with the weight `reward.w_fair`
   (`interleaved` scenario: 0.5). The project brief also lists
   "fairness of sensor usage" among the coordination metrics.

If the advisor meant something else, rename `INTERLEAVED` or add a new
`Mode` with its own `ModeProfile` (pulses, PRI, beams per look); the rest of
the code picks it up.

## First observations (3 seeds, `interleaved` scenario, Utah terrain)

| Policy | Pos RMSE (m) | Coverage | False tracks | Usage fairness | Team return |
|---|---|---|---|---|---|
| fixed_scan | 136 | 0.76 | 0.87 | 1.00 | −183 |
| round_robin | 68 | 0.81 | 0.50 | 1.00 | −97 |
| nearest | 48 | 0.38 | 0.50 | 1.00 | −251 |
| central_greedy | 18 | 0.49 | 0.57 | 1.00 | −152 |
| interleaved_greedy | 35 | 0.81 | 1.36 | 1.00 | −105 |

Switching the same greedy assignment from TRACK to INTERLEAVED raises
coverage from 0.49 to 0.81 (radars no longer stop searching) at the cost of
a larger position error and more false tracks. In these symmetric
scenarios every radar ends up with a similar tracking load, so fairness is
≈ 1 for all rule-based policies; it becomes informative when radars fail or
when a learned policy concentrates the work on a few radars.
