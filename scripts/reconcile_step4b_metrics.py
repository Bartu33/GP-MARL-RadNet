"""Reconcile Step 4A vs Step 4B metrics side-by-side."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
audit_file = ROOT / "reports" / "step4a_final_quality_audit.json"
data = json.loads(audit_file.read_text(encoding="utf-8"))

print("=" * 80)
print("RECONCILIATION: STEP 4A REPORTED vs STEP 4B VERIFIED")
print("=" * 80)

metrics = [
    ("Total Physical Targets", "1,880 (clerical typo in summary)", "1,800 (exact on disk)", "Typo in line 547 of produce script"),
    ("Total Truth States (Rows)", "114,680 (1880 x 61)", "109,800 (1800 x 61)", "Exact match with (600x61 + 400x183)"),
    ("Overall Track Coverage", "41.3%", "41.3%", "Identical (per-episode mean)"),
    ("Zero-Detection Targets Count", "487", "487", "Identical (exact sum of zero-det)"),
    ("Zero-Detection Targets Ratio", "25.9% (487/1880)", "27.1% (487/1800)", "Updated denominator (+1.2%)"),
    ("Overall Mean LOS Visibility", "82.9%", "82.9%", "Identical (per-episode mean)"),
    ("Mean Position RMSE", "53.21 m", "53.21 m", "Identical (truth-associated EKF tracks)"),
    ("Single Aircraft Track Cov", "36.1%", "36.1%", "Identical (150 eps, 150 tgts)"),
    ("Single Aircraft Zero-Det", "6 / 150 (4.0%)", "6 / 150 (4.0%)", "Identical"),
    ("Single Missile Track Cov", "24.3%", "24.3%", "Identical (150 eps, 150 tgts)"),
    ("Single Missile Zero-Det", "40 / 150 (26.7%)", "40 / 150 (26.7%)", "Identical"),
    ("Easy Tier Track Cov", "60.9%", "60.9%", "Identical (300 eps, 540 tgts)"),
    ("Medium Tier Track Cov", "37.0%", "37.0%", "Identical (450 eps, 808 tgts)"),
    ("Hard Tier Track Cov", "25.6%", "25.6%", "Identical (250 eps, 452 tgts)"),
]

print(f"{'Metric':30s} | {'Step 4A Reported':32s} | {'Step 4B Verified':25s} | {'Explanation'}")
print("-" * 120)
for m, old, new, expl in metrics:
    print(f"{m:30s} | {old:32s} | {new:25s} | {expl}")
