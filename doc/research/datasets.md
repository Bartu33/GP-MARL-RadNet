# Range, Doppler and micro-Doppler datasets (Task 5)

## Conclusion

The project brief requires a *fully simulated* environment, and our radar
model works at detection level (reports `[r, α, θ, ṙ, SNR]`), not with raw
signals. **Our own synthetic generator is therefore the main data source**
(`src/services/dataset_service.py`). Public datasets are useful for

1. checking that synthetic kinematics and Doppler values are plausible, and
2. a possible extension that adds a signal-level (micro-Doppler) classifier,
   which the physics note lists as an optional Stage-4 research extension.

None of the public datasets below is needed to run, train or evaluate the
system, so no licence restriction blocks the project.

## Comparison

| Dataset | Data type | Targets | Sensor | Size | Access / licence | Use for us |
|---|---|---|---|---|---|---|
| **Own synthetic data** (`data/synthetic/`) | Fused EKF tracks → 10 s windows of kinematic features | 4 behaviour classes (A–D) | Simulated radar of this repo | ~2 700 windows per 60 episodes/class (scalable) | Ours, regenerate any time | Main training / test data of the flight-profile classifier |
| [DIAT-µSAT](https://ieee-dataport.org/documents/diat-msat-micro-doppler-signature-dataset-small-unmanned-aerial-vehicle-suav) | Micro-Doppler spectrogram images | Small drones, bionic bird, mini-helicopter | CW X-band | 4 849 images | IEEE DataPort; free educational access on e-mail request from an institutional address; cite the original papers | Micro-Doppler extension (aerial targets, closest to our domain) |
| [60 GHz mmWave UAS and birds](https://ieee-dataport.org/open-access/millimeter-wave-60-ghz-radar-measurements-uas-and-birds) | Raw amplitude + phase (.pkl) | Small drones, bionic bird | 60 GHz FMCW | — | IEEE DataPort open access (account may be needed); documentation incomplete | Micro-Doppler extension; signal processing practice |
| RRN-ATR-Net (IEEE DataPort, DOI 10.21227/2q85-nz22) | Radar echoes | Quadrotor, fixed-wing, helicopter, bionic bird | TI AWR1642BOOST, anechoic chamber | 1 200 samples | IEEE DataPort | Range-Doppler classification reference |
| MAFAT Radar Challenge ([CodaLab forum](https://competitions.codalab.org/forums/22085/4642/), [paper](https://www.ee.bgu.ac.il/~haimp/2021_MAFAT_Radar_Challenge_Conference_Paper.pdf)) | Doppler-pulse I/Q segments, 128 × 32 per sample | Humans vs animals (ground targets) | Doppler-pulse radars | 6 656 labelled training segments + 49 071 auxiliary | Competition data hosted on CodaLab; terms set by the organisers (check before use) | Example of *track-segment* classification from Doppler data; method reference |
| [CARRADA](https://arxiv.org/abs/2005.01456) | Range-angle-Doppler tensors + camera | Pedestrians, cyclists, cars | Automotive FMCW | — | CC BY-NC-SA 4.0 (non-commercial) | Data-format and annotation reference |
| [RADDet](https://export.arxiv.org/pdf/2105.00363) | Range-azimuth-Doppler tensors with boxes | Dynamic road users | Automotive FMCW | — | See paper / repository | Data-format reference |
| CRUW / ROD2021 ([CVPRW 2021](https://openaccess.thecvf.com/content/CVPR2021W/WAD/html/Wang_Rethinking_of_Radars_Role_A_Camera-Radar_Dataset_and_Systematic_Annotator_CVPRW_2021_paper.html), [RODNet](https://openaccess.thecvf.com/content/WACV2021/papers/Wang_RODNet_Radar_Object_Detection_Using_Cross-Modal_Supervision_WACV_2021_paper.pdf)) | Range-azimuth maps (RAMaps) + synchronised camera | Road objects | Automotive radar | ~400 k frames | Project site cruwdataset.org; **licence not verified** — read it there before use | Data-format reference (range-azimuth) |
| [OpenSky Network](https://en.wikipedia.org/wiki/OpenSky_Network) | ADS-B aircraft trajectories | Real aircraft | ADS-B receivers | Very large | Research access on request; licence limited to non-profit research / education ([licence text](https://zenodo.org/records/6325961/files/LICENSE.txt?download=1)) | Sanity check of class-A (fast cruise) kinematics |

**Relevance ranking for this project:** own synthetic data ≫ DIAT-µSAT
(aerial, micro-Doppler) > 60 GHz UAS/birds > RRN-ATR-Net > MAFAT (method
reference) > OpenSky (kinematics check) > CARRADA / RADDet / CRUW (automotive,
format reference only).

## Usage conditions — checklist before downloading anything

- [ ] Read the licence: commercial use? redistribution? derivative works?
      (CARRADA: non-commercial; OpenSky: non-profit research / education.)
- [ ] Does access need an institutional e-mail or an account (DIAT-µSAT,
      IEEE DataPort, OpenSky)?
- [ ] Which papers must be cited?
- [ ] **Never commit the data** to this repository. Put a `README.md` and,
      if allowed, a download script under `ref/datasets/<name>/`; keep the
      files locally (`data/raw/` is git-ignored for large content).
- [ ] Record the version / download date in that README.

## Access request template (DIAT-µSAT)

Send from your university address, with the subject line the dataset page asks
for:

> Dear authors,
>
> I am a final-year student at [UNIVERSITY], working on the capstone project
> "Cooperative Multi-Agent Reinforcement Learning for an Intelligent Radar
> Network" under the supervision of [ADVISOR]. We would like to use the
> DIAT-µSAT dataset for non-commercial, educational research — to compare
> our simulated target signatures with measured micro-Doppler data. We will
> cite your papers and will not redistribute the data.
>
> Kind regards,
> [NAME], [STUDENT ID]

## Why micro-Doppler is not in the simulator yet

Micro-Doppler (rotor blades, wing flapping) needs a signal-level model:
pulse-by-pulse returns, coherent processing and time-frequency analysis.
The physics note deliberately stops at detection level (Stage 2–3) and puts
waveform-level simulation in Stage 4 as an optional extension, because it
is much more expensive and not required for the main research question.
