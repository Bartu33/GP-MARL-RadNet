# GP-MARL-RadNet — Canary Dataset Production & Quality Audit Report

**Step 3A: Production Configuration Fixes & Canary Dataset (70 Episodes)**  
**Date:** 2026-11-16  
**Git Branch:** `feature/valley-diverse-datasets`  
**Dataset Directory:** `data/processed/canary_diverse_v1/`  
**Generation Policy:** `central_greedy`  
**Generation Time:** 19.76 seconds (0.282 s/ep)  
**Total Storage:** 6.09 MB (89.1 KB/ep)  

---

## 1. Executive Summary & Quality Verdict

In Step 3A, the production configurations for Single Aircraft (cross-valley triangulation, 90° sector, `[6.0, 18.0, 32.0]` elevation bars) and Single Missile (cross-valley triangulation, 120° sector, `[3.0, 8.0]` elevation bars) were integrated into the primary dataset generator.

A comprehensive **70-episode Canary dataset** (`data/processed/canary_diverse_v1/`) was successfully produced across all 7 dataset groups (10 episodes per group), spanning 5 procedural terrain families, 6 texture families, and 4 obstacle profiles.

### Overall Canary Performance Summary

| Metric | Canary v1 (70 Episodes) | Pilot v2 (42 Episodes) | Pilot v1 (Baseline) |
| :--- | :---: | :---: | :---: |
| **Total Episodes** | **70 episodes** | 42 episodes | 42 episodes |
| **Completed Episodes** | **70 / 70 (100%)** | 42 / 42 (100%) | 42 / 42 (100%) |
| **Flawed Episodes** | **0 flawed (0.0%)** | 0 flawed (0.0%) | 0 flawed (0.0%) |
| **Mean LOS Visibility** | **85.8%** | 86.5% | 62.8% |
| **Mean Track Coverage** | **35.9%** | 28.4% | 9.3% |
| **Total Delivered Reports** | **2047** | 1,044 | 153 |
| **Zero-Detection Targets** | **33** | 32 / 78 (41.0%) | 61 / 78 (78.2%) |
| **Mean Position RMSE** | **None m** | 14.8 m | 18.2 m |

---

## 2. Seven Dataset Groups Performance Breakdown

| Dataset Group | Episodes | Targets | Zero-Det Targets | Delivered Reports | Mean LOS | Mean Coverage | Mean RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dataset_single_aircraft_v1` | 10 | 10 | 0 / 10 | 199 | 100.0% | **21.0%** | N/A |
| `dataset_single_uav_agile_v1` | 10 | 10 | 3 / 10 | 218 | 76.7% | **33.5%** | N/A |
| `dataset_single_uav_drift_v1` | 10 | 10 | 2 / 10 | 211 | 73.3% | **31.0%** | N/A |
| `dataset_single_missile_v1` | 10 | 10 | 3 / 10 | 179 | 85.1% | **9.0%** | N/A |
| `dataset_triple_all_same_v1` | 10 | 30 | 6 / 30 | 551 | 88.9% | **52.7%** | N/A |
| `dataset_triple_all_different_v1` | 10 | 30 | 12 / 30 | 300 | 88.7% | **63.8%** | N/A |
| `dataset_triple_two_plus_one_v1` | 10 | 30 | 7 / 30 | 389 | 87.9% | **40.0%** | N/A |

---

## 3. Difficulty Level Breakdown (Easy, Medium, Hard)

| Difficulty Level | Episode Count | Share (%) | Delivered Reports | Mean LOS | Zero-Det Targets | Mean Track Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **EASY** | 21 | 30.0% | 803 | 99.0% | 4 | **52.5%** |
| **MEDIUM** | 28 | 40.0% | 756 | 79.8% | 15 | **34.3%** |
| **HARD** | 21 | 30.0% | 488 | 80.6% | 14 | **21.2%** |

---

## 4. Split Isolation & Leakage Verification

- **Train Episodes:** 49 (70.0%)
- **Validation Episodes:** 11 (15.7%)
- **Test Episodes:** 10 (14.3%)
- **Leakage Status:** 100% disjoint terrain seeds and route fingerprints. Zero cross-split contamination.

---

## 5. Storage & Production Projections (1,000 Episodes)

- **Total Canary Size (70 episodes):** 6.09 MB
- **Mean Episode Size:** 89.1 KB (Min: 53.7 KB, Max: 271.2 KB)
- **Canary Generation Time:** 19.76 s (0.282 s/ep)
- **1,000 Episodes Projected CPU Time:** **4.7 minutes**
- **1,000 Episodes Projected Disk Storage:** **87.0 MB**
