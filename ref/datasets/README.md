# References

Third-party papers are **not** committed to this repository (copyright).
Keep your own copies in `ref/papers/` (git-ignored) and list them here:
`physics_informed_radar_detection_model.pdf` (PN) and
`Alpdemir-2022-Neural_Computing_and_Applications.pdf` (AL).

Equation-by-equation use of PN and AL in the code (which file, and whether the
simulation uses it) is in Section 3 of
[doc/research/Gorev5-8_Arastirma_Kaynaklari.pdf](../doc/research/Gorev5-8_Arastirma_Kaynaklari.pdf).

| Key | Reference | Used for |
|---|---|---|
| PN | *Physics-Informed Simulation of Three-Dimensional Radar Detections*, capstone technical note, Oct 2026 (team document) | Radar detection model, `src/core` |
| AL | M. N. Alpdemir, "Tactical UAV path optimization under radar threat using deep reinforcement learning", *Neural Computing and Applications*, 2022, doi:10.1007/s00521-021-06702-3 | Ellipsoid RCS and bank-to-turn kinematics (simulation); P_D approximation and probability of track (Radar Model Lab, tests) |
| BS | Y. Bar-Shalom, X. R. Li, T. Kirubarajan, *Estimation with Applications to Tracking and Navigation*, Wiley, 2001 | EKF |
| PPO | J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347, 2017 | PPO |
| MAPPO | C. Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", NeurIPS 2022 | MAPPO |
| SW | P. Swerling, "Probability of detection for fluctuating targets", IRE Trans. IT, 1960 | Swerling models |
