# Architecture

## One environment step (`src/core/env.py`)

1. **Targets move** — `core/targets.py` integrates each synthetic target
   (four behaviour classes, see below).
2. **Agents act** — each radar chooses *idle*, *search* or *track slot k*
   (`core/modes.py`). A track action steers one long dwell at the predicted
   position of the k-th track of the shared track table.
3. **Radars observe** — `core/radar.py::radar_step` runs the physics-informed
   detection model for every beam position: visibility (range, field of view,
   terrain line-of-sight) → aspect-dependent RCS →
   Gaussian beam gain → radar-equation SNR → P_D gate → SNR-dependent noisy
   report `[r, α, θ, ṙ]` with covariance R → false alarms → energy / thermal /
   dwell / channel resources.
4. **Messages shared** — a communication model can cap the number of reports per
   radar (bandwidth) and drop packets (intermittent links).
5. **Tracks fused** — `models/tracking/tracker.py`: one EKF per track,
   centralised sequential measurement fusion, gated nearest-neighbour
   association, M-hit confirmation, time-out deletion.
6. **Targets classified** — `models/classifier/`: kinematic features of the last
   10 s of each fused track → gradient-boosted trees → class probabilities,
   accumulated over time.
7. **Team reward** — − track error − missed targets + class confidence on the
   true class − energy use − false tracks (truth is used only here, for scoring).

## Layers and folders

| Layer | Folder | Main modules |
|---|---|---|
| Interfaces | `src/interfaces/` | `app.py` (Streamlit), `viewer3d.py` + `assets/viewer3d.html` (three.js), `plots.py` |
| Services | `src/services/` | `simulation_service.py` (episodes, comparisons, viewer payload), `dataset_service.py`, `training_service.py` |
| Models | `src/models/` | `tracking/ekf.py`, `tracking/tracker.py`, `classifier/features.py`, `classifier/classifier.py`, `baselines/policies.py`, `marl/mappo.py` |
| Core simulation | `src/core/` | `geometry.py`, `rcs.py`, `detection.py`, `radar.py`, `modes.py`, `targets.py`, `scenario.py`, `env.py` |
| Configuration | `configs/scenarios/` | `nominal`, `high_noise`, `radar_failure`, `comm_limited` |

## Agent interface

* **Action** (`Discrete(2 + K)`): `0` idle, `1` search, `2 … K+1` track slot.
* **Observation** (fixed size, physics note Eq. 59): for each of the K track
  slots — validity mask, predicted range / azimuth (sin, cos) / elevation /
  radial velocity relative to this radar, log position uncertainty, confirmed
  flag, time since last update, in-coverage flag, predicted track-mode SNR,
  top class probability; then the radar's own resource vector (Eq. 57), mode
  one-hot, online flag and episode progress.
* **Global state** for a centralised critic: concatenation of all observations.

## Target classes (synthetic, `core/targets.py`)

| Label | Name | Motion signature | RCS model |
|---|---|---|---|
| 0 | A · fast cruise | 110–230 m/s, bank-to-turn coordinated turns, near-constant altitude | three-radius ellipsoid, Swerling I |
| 1 | B · agile hover | stop-and-go, hover segments, abrupt heading changes, climbs | small ellipsoid, Swerling I |
| 2 | C · erratic slow | 7–19 m/s, random-walk turn rate, short-period vertical oscillation | very small constant, Swerling I |
| 3 | D · drift | 2–9 m/s wind drift, slow climb, turbulence | synthetic lookup table |

## Equations → code

| Physics note | Code |
|---|---|
| Eqs. 5–14 frames, h(x), Doppler | `geometry.world_to_sensor`, `spherical_from_sensor`, `doppler_shift` |
| Eqs. 15–22 aspect, lookup RCS, scattering centres, Swerling | `rcs.py` |
| Eq. 24 Gaussian beam | `radar.antenna_gain` |
| Eqs. 25–28 radar equation, SNR | `radar.snr_linear`, `radar.noise_power` |
| Eqs. 29–34 P_D models | `detection.py` |
| Eqs. 35–47 noisy report, accuracy, bias, quantisation | `radar.radar_step`, `radar.measurement_sigmas` |
| Eqs. 48–50 signal quality | `radar.signal_quality` |
| Eqs. 51–57 resources | `radar.radar_step` (update block), `radar.resource_vector` |
| Eq. 58 false alarms | `radar.radar_step` |
| Section 5 terrain occlusion | `terrain.Terrain.line_of_sight`, `radar.los_status` |
| Eq. 59 MARL observation | `env.RadarNetEnv._observe` |
| Section 16 verification | `tests/test_radar_model.py` |
| Alpdemir Eq. 1 / 2–4 / 6–8 / 14 | `detection.pd_erfc_approx` / `detection.run_of_misses_probability`, `aggregate_track_probability` / `targets.FastCruise` / `rcs.EllipsoidRCS` |

## Known simplifications / next steps

* Terrain line-of-sight is checked along the straight radar-target path on a
  ~60 m height grid with a 4/3-Earth curvature correction (`core/terrain.py`);
  there is no diffraction, multipath or clutter from the terrain itself.
* No range / Doppler ambiguity, no clutter map, no micro-Doppler.
* Association is greedy GNN; JPDA or MHT would handle crossing targets better.
* Classifier Model 2 (1-D CNN on the raw window) and IMM tracking are open items.
* MAPPO is implemented but not yet tuned; training curves and comparisons
  against the baselines are the next milestone.
