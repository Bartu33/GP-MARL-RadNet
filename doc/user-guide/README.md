# User guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the classifier (once)

```bash
python -m src.services.classifier_study --episodes 40
```

Targets are UAVs, aircraft and missiles (`src/core/target_classes.py`). The
study simulates 40 three-radar episodes per class, cuts every track into
10-second windows and trains GBM, MLP and GRU classifiers on fused tracks,
single-radar (local) tracks and true states (upper bound only). The report
(`models/metadata/classifier_study.json`, also on the **Classifier** tab) gives
macro-F1, balanced accuracy, ECE, ROC-AUC and PR-AUC with an episode-wise
split, and the hard cases: accuracy per variant, the physically overlapping
variant pairs (fighter vs subsonic cruise missile, fixed-wing UAV vs transport
aircraft) and young tracks seen for only 3 s or 6 s. It also replaces the
simulator's classifier (`models/checkpoints/flight_profile_gbm.joblib`) with
the GBM trained on fused tracks (`--no-sim-model` keeps the old one,
`--from-data` reuses the saved dataset). A classifier trained on the earlier
A–D behaviour classes is not loaded; the simulator then runs without one.
Background on RL-based classification: `doc/research/rl_classification.md`.

## 3. Dashboard

```bash
streamlit run src/interfaces/app.py
```

| Tab | What you can do |
|---|---|
| **Scenario Management** | **Live simulation**: start from a scenario template, edit radars (position, heading), targets (class, variant, planned or explicit motion) and channel stress, validate and run it. **Dataset replay**: pick a dataset in `data/processed/`, filter its episodes (split, difficulty, radar and target counts, terrain), preview and load one. Both open the 3-D view and the analysis panels: radar information (signal physics, EKF information), radar selection and decision inspector (resource use, which radar served which track and why, radar-to-radar transitions) and, kept apart, ground-truth evaluation. |
| **3-D Simulation** | Pick a scenario, a coordination policy, a seed and a map location (the location sets the terrain used for line-of-sight); the episode is simulated and replayed over satellite-textured terrain. Buttons in the view: camera presets, orbit, and layer toggles (beams, coverage fans, trails, detections, truth). Bottom bar: play / pause, single step, speed, replay scrubber. The panel on the left shows each radar's mode, energy and thermal margin. Switch to **2-D** for an offline top-down view. A replay can be downloaded as a stand-alone HTML file. |
| **Radar Model Lab** | Sliders for power, gain, P_FA and mode; plots of SNR vs range, P_D models, SNR-dependent accuracy, ellipsoid / lookup-table RCS patterns, probability of track and the beam pattern. |
| **Classifier** | Generate data and train; confusion matrix; one sample trajectory per class; the classifier study with ROC / PR curves and the hard cases. |
| **Experiments** | Run several policies over several seeds on a scenario; table of mean ± std and box plots; CSV download. |

### Colours in the 3-D view

* radar / beam: teal = search, blue = track, purple = interleaved (track part; its search cells are teal), grey = idle, red = offline
* coloured spheres = true targets (orange UAV, purple aircraft, green missile)
* white box + wire sphere = confirmed fused track and its 3σ uncertainty;
  the label shows the predicted class and confidence
* small points = detections of the current step (red = false alarm)
* thin lines radar → target: green = line of sight, red = hidden by terrain

### Following a target

Click a target sphere or a confirmed track box: the camera zooms in and
follows it while the replay plays (you can still rotate and zoom around it).
A panel shows its altitude, speed, how many radars have line of sight (and how
many are blocked by terrain) and which track follows it, or for a track its
class, uncertainty and error to the nearest truth. Click empty ground, the
✕ button or a camera preset to stop following.

### PPI scopes

Below the map there is one **PPI scope** per radar, synchronised with the
replay (one update per simulation step). Each scope shows what that radar
knows on its own next to what the fusion centre knows. The **PPI** button
hides them; in the **2-D** view the same scopes are drawn with Plotly (offline).

| Symbol | Meaning |
|---|---|
| shaded sector | the radar's field of view; north up, rings every 2 km out to r_max |
| teal / blue wedge | search / track beams of this step |
| small fading dots | the radar's own detections, 4 s afterglow (red = false alarm) |
| amber diamond `L#` | the radar's own confirmed local track (structure A) |
| white square `T#` | fused track from the fusion centre; dashed / open when this radar does not feed it |
| dashed amber line `L5→T12` | the fused track updated by the same reports as the local track |
| coloured dot | truth (follows the **Truth** button) |
| header `own · net · to FC a/b` | confirmed local tracks, confirmed fused tracks, reports received by the centre / produced by the radar |

**Click a scope to enlarge it.** The enlarged scope fits its view to the
tracks and follows them; the mouse wheel zooms (down to a ~400 m wide view),
dragging pans, a double click or **Fit** fits the view again and **Full range**
shows the whole coverage. Range rings adapt to the zoom (50 m … 5 km), a scale
bar sits bottom right and, when the radar is off screen, its direction and
distance are shown top left. Uncertainty circles (3σ) are drawn for both the
local and the fused tracks. A table compares, for every confirmed local track,
the fused track it feeds: position difference, σ of the local track, σ of the
fused track and how many radars fed the fused track in the last 3 s; fused
tracks this radar has no own track for are listed below it. Close with ✕, Esc
or a click outside.

Local tracks are recorded when the episode runs with `local_tracks=True`
(the dashboard and `run_episode` do); otherwise the scopes show detections and
fused tracks only.

## 4. Scenarios

YAML files in `configs/scenarios/`. Copy one and edit:

* `radars`: name, position `[east, north, up]` in metres, `heading_deg`
  (boresight, counter-clockwise from east); any `RadarParams` field can be
  overridden here or in `radar_defaults`.
* `targets`: list of `{target_class: uav | aircraft | missile}`, optionally with a `variant`
  (e.g. `fixed_wing`, `fighter`, `supersonic`). The simulator then draws the variant, the RCS and a
  valid trajectory through the area from the class envelope at every seed; a very fast missile that
  cannot fly the whole episode inside the terrain grid flies its planned part and then leaves the area.
  An explicit target adds `variant`, `position`, `velocity` and `motion` (`CV`, `CA` with
  `acceleration_mps2`, `CT` with `turn_rate_rad_s`, or `segments`). The earlier `{label: 0..3}`
  behaviour targets still load but are not scored by the classifier.
* `share_actions: true` lets every radar action also decide whether its
  reports are sent to the fusion centre (fusion participation); with
  `reward.w_comm` each delivered report costs reward (`share_control.yaml`).
* `interleaved_actions: true` adds the INTERLEAVED (search + track) actions;
  `reward.w_fair` adds a load-equalisation term (see doc/research/radar_modes.md).
* `stress`: `fail_radar`, `fail_time_s`, `comm_drop_prob`,
  `max_reports_per_radar`, `noise_scale`, `outages` (list of
  `[radar index or all, t0, t1]`: the radar transmits but detects nothing in
  that window; `track_loss.yaml` uses a 20 s outage (25–45 s) on all radars to force
  track loss and reacquisition).
* `site`: latitude / longitude of the area; it selects the terrain.
* `terrain`: `{source: auto}` downloads real elevation tiles once (cached in
  `data/raw/terrain/`) and falls back to procedural hills offline; `flat`,
  `procedural` and `tiles` force one source. Terrain blocks radar line of
  sight, and targets keep at least 30 m above the ground.

## 5. Target-class datasets

Recipes in `configs/datasets/` describe a dataset of UAV / aircraft / missile
episodes: number of episodes, duration, radar and target counts, class
weights, train / validation / test split, terrain profiles and, for missile
datasets, the route pattern (converging, diverging, hub), salvo or terminal
dive. One script plans, simulates and audits a dataset:

```bash
python scripts/produce_class_targets_dataset.py --recipe configs/datasets/missile_routes_pilot5.yaml \
    --output data/processed/missile_routes_pilot5 --report reports/missile_routes_pilot5
```

Each episode is written as Canonical Episode Records (`metadata.json`,
`records.npz`) under its split; `manifest.json` keeps the SHA-256 of every
file and the report folder gets the plan and an audit that compares the
realised targets with their variant envelopes. Episodes can be opened from
**Scenario Management → Dataset replay**; the replay re-simulates the episode
with its recorded actions and is verified against the records, so a dataset
must be regenerated after a change to the radar model or the tracker (its
episode folder names, which carry a configuration hash, change too).

## 6. Metrics

Every number in the dashboard is defined in `src/core/evaluation.py`
(`METRIC_INFO`); the Experiments tab shows the same table under
"What do the metrics mean?". The main ones:

| Metric | Meaning |
|---|---|
| `coverage_in_range` | share of target-steps with a valid confirmed track while at least one radar can see the target |
| `track_coverage` | the same over all target-steps, also when no radar can see the target |
| `detection_loss` | share of detection opportunities (target inside a beam of a transmitting radar, line of sight) without a detection |
| `track_losses` | fused tracks following a target that the tracker deleted; one missed detection is not a track loss |
| `interruptions`, `interrupted_s`, `reacquire_s` | gaps without a valid track after first acquisition, their total length and the mean length of gaps that ended with reacquisition |
| `longest_track_s` | longest uninterrupted valid-track interval (mean over targets) |
| `correct_class_continuity` | share of target-steps with a valid track and the correct predicted class |
| `id_switches` | a target became matched to another track while the old one still existed |
| `pos_rmse_m`, `pos_err_median_m` | RMS and median position error of matched fused tracks |
| `decision_ms` | time the coordination policy needs per step for all radars (inference time) |
| `single_best_coverage`, `single_best_longest_s`, `single_mean_rmse_m` | the same, for each radar alone with its own local track (structure A); the best radar is reported |

On the 3-D Simulation tab, "Single radar vs network" lists every radar's
local track next to the fused track.

## 7. Training the task manager (PPO, optional)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m src.services.classifier_study --episodes 40       # first: the observation uses the classifier
python -m src.services.training_service --task --decisions 30000
```

One PPO agent at the fusion centre sends, every second, a track-update
request (or none) to each radar; the radars' own schedulers decide. Training
first imitates the rule-based task manager `task_greedy` (200 teacher episodes,
behaviour cloning), then fine-tunes with PPO; the checkpoint
`models/checkpoints/ppo_task.pt` and its learning history appear as `ppo_task`
in the policy lists. A checkpoint fits only scenarios with the track-slot and
radar counts it was trained on (nominal: 6 slots, 4 radars); others skip it with
a warning. The earlier mode-setting learners (`--central`, `--ippo`, MAPPO) are
kept in the code but hidden from the dashboard.

## 8. Tests

```bash
pytest
```
