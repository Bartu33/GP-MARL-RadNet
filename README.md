# GP-MARL-RadNet

**Cooperative Multi-Agent Reinforcement Learning for an Intelligent Radar Network** —
adaptive target tracking, identification and sensor-resource coordination in a
simulated environment.

> Research and educational simulation only. All sensor and motion models are
> abstract, targets are synthetic and every parameter is non-operational.

See [doc/architecture/overview.md](doc/architecture/overview.md) for the
architecture and [doc/research/](doc/research/) for the research notes.

## What is in the repository

| Area | Where | Status |
|---|---|---|
| Physics-informed radar detection model (geometry, RCS, radar equation, P_D gate, SNR-dependent noise, false alarms, resource state) | `src/core/radar.py`, `geometry.py`, `rcs.py`, `detection.py` | working, unit-tested |
| Radar operating modes (search raster, track dwell, idle) | `src/core/modes.py` | working |
| Synthetic targets — 4 behaviour classes, motion models | `src/core/targets.py` | working |
| Multi-agent environment (PettingZoo `ParallelEnv`) | `src/core/env.py`, scenarios in `configs/scenarios/` | working, API-tested |
| EKF tracker + centralised multi-radar fusion | `src/models/tracking/` | working |
| Flight-profile classifier (kinematic features + gradient boosting) | `src/models/classifier/` | working |
| Rule-based baselines (fixed scan, round-robin, nearest, central greedy) | `src/models/baselines/` | working |
| MAPPO / IPPO | `src/models/marl/mappo.py`, `src/services/training_service.py` | first implementation, not tuned |
| Dashboard: 3-D terrain view, radar model lab, classifier, experiments | `src/interfaces/` | working |

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/Bartu33/GP-MARL-RadNet.git
cd GP-MARL-RadNet
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
# optional, for MAPPO training:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Running

```bash
# 1) train the flight-profile classifier on synthetic tracker output (~30 s)
python -m src.services.dataset_service --episodes 60 --train

# 2) open the dashboard
streamlit run src/interfaces/app.py

# 3) run the tests
pytest

# 4) (optional) train MAPPO, then pick "mappo" in the dashboard
python -m src.services.training_service --scenario nominal --steps 200000
```

The 3-D view loads three.js from `cdn.jsdelivr.net`, satellite imagery from Esri
World Imagery and elevation from AWS Terrain Tiles, so it needs an internet
connection and a WebGL-capable browser. Without tiles it falls back to a
procedural ground; the **2-D** view works fully offline. Any replay can be
downloaded from the dashboard as a stand-alone HTML file.

## Repository layout

```
configs/scenarios/   YAML scenario files (radars, targets, stress settings)
src/core/            simulator: geometry, RCS, detection, radar, modes, targets, env
src/models/          tracking (EKF), classifier, baselines, marl (MAPPO)
src/services/        episode runner, dataset generation, training
src/interfaces/      Streamlit dashboard, 3-D viewer (three.js), plots
tests/               verification tests (physics note Section 16) and env API tests
doc/                 architecture, research notes, user guide
data/, models/       generated datasets and checkpoints (git-ignored)
ref/                 references (papers are not committed)
```

## References used in the code

* *Physics-Informed Simulation of Three-Dimensional Radar Detections* — capstone
  technical note (team document). Equation numbers in `src/core` refer to it.
* M. N. Alpdemir, "Tactical UAV path optimization under radar threat using deep
  reinforcement learning", *Neural Computing and Applications*, 2022 — only the
  radar-side relations are used: P_D approximation (Eq. 1), probability of track
  (Eqs. 2–4), three-radius ellipsoid RCS (Eq. 14) and bank-to-turn kinematics
  (Eqs. 6–8). Its threat-evasion and kill-probability parts are out of scope.
* Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to
  Tracking and Navigation*, Wiley, 2001 — EKF.
* J. Schulman et al., "Proximal Policy Optimization Algorithms", 2017;
  C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent
  Games", NeurIPS 2022 — MAPPO.

All code in this repository is original; no third-party source code was copied.
Third-party libraries are used through their public APIs under their own licences
(three.js — MIT; Streamlit, NumPy, SciPy, scikit-learn, PettingZoo, Gymnasium,
Plotly — BSD/Apache/MIT). Map tiles are displayed with attribution; check the
providers' terms before any public deployment.

to run:
"pip install -r requirements.txt
python -m src.services.dataset_service --episodes 60 --train 
streamlit run src/interfaces/app.py
pytest"