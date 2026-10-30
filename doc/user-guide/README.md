# User guide

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the classifier (once)

```bash
python -m src.services.dataset_service --episodes 60 --train
```

This simulates ~240 short single-radar episodes, cuts every fused track into
10-second windows, trains the gradient-boosted classifier and prints macro-F1,
balanced accuracy and the confusion matrix. Files written:

* `data/synthetic/flight_profile_windows.csv`
* `models/checkpoints/flight_profile_gbm.joblib`
* `models/metadata/flight_profile_gbm.json`

The same can be done from the **Classifier** tab of the dashboard.

## 3. Dashboard

```bash
streamlit run src/interfaces/app.py
```

| Tab | What you can do |
|---|---|
| **3-D Simulation** | Pick a scenario, a coordination policy, a seed and a map location (the location sets the terrain used for line-of-sight); the episode is simulated and replayed over satellite-textured terrain. Buttons in the view: camera presets, orbit, and layer toggles (beams, coverage fans, trails, detections, truth). Bottom bar: play / pause, single step, speed, replay scrubber. The panel on the left shows each radar's mode, energy and thermal margin. Switch to **2-D** for an offline top-down view. A replay can be downloaded as a stand-alone HTML file. |
| **Radar Model Lab** | Sliders for power, gain, P_FA and mode; plots of SNR vs range, P_D models, SNR-dependent accuracy, ellipsoid / lookup-table RCS patterns, probability of track and the beam pattern. |
| **Classifier** | Generate data and train; confusion matrix; one sample trajectory per class. |
| **Experiments** | Run several policies over several seeds on a scenario; table of mean ± std and box plots; CSV download. |

### Colours in the 3-D view

* radar / beam: teal = search, blue = track, purple = interleaved (track part; its search cells are teal), grey = idle, red = offline
* coloured spheres = true targets (orange A, purple B, green C, yellow D)
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

## 4. Scenarios

YAML files in `configs/scenarios/`. Copy one and edit:

* `radars`: name, position `[east, north, up]` in metres, `heading_deg`
  (boresight, counter-clockwise from east); any `RadarParams` field can be
  overridden here or in `radar_defaults`.
* `targets`: list of `{label: 0..3}` (optionally `position`).
* `interleaved_actions: true` adds the INTERLEAVED (search + track) actions;
  `reward.w_fair` adds a load-equalisation term (see doc/research/radar_modes.md).
* `stress`: `fail_radar`, `fail_time_s`, `comm_drop_prob`,
  `max_reports_per_radar`, `noise_scale`.
* `site`: latitude / longitude of the area; it selects the terrain.
* `terrain`: `{source: auto}` downloads real elevation tiles once (cached in
  `data/raw/terrain/`) and falls back to procedural hills offline; `flat`,
  `procedural` and `tiles` force one source. Terrain blocks radar line of
  sight, and targets keep at least 30 m above the ground.

## 5. MARL training (optional)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m src.services.training_service --scenario nominal --steps 200000
python -m src.services.training_service --ippo      # IPPO baseline
```

After training, `mappo` appears in the policy list of the dashboard.

## 6. Tests

```bash
pytest
```
