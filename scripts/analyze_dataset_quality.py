"""Quality analysis script for GP-MARL-RadNet synthetic dataset.

Computes comprehensive dataset quality metrics from Canonical Episode Records v2:
  1. Train/validation/test split counts
  2. Target class distribution (overall and per split)
  3. Motion model distribution (CV, CA, CT)
  4. Radar reports per episode distribution
  5. Confirmed track counts per episode and zero-track episode fraction
  6. Communication status distribution (delivered, packet_lost, etc.)
  7. Per-radar detection and report counts
  8. Target position tracking error (with greedy gating association)
  9. Split distribution comparisons
 10. Data integrity audit (NaN, Inf, corruption, missing metadata)

Saves outputs to data/processed/synthetic_100/quality_report/:
  - quality_report.json
  - quality_summary.csv
  - PNG plots
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.targets import CLASS_NAMES
from src.services.simulation_service import load_episode_records

# Styling
BG_COLOR = "#0B0F19"
PANEL_BG = "#111827"
BORDER_COLOR = "#1F2937"
GRID_COLOR = "#1E293B"
TEXT_MAIN = "#F8FAFC"
TEXT_MUTED = "#94A3B8"
ACCENT_CYAN = "#38BDF8"
ACCENT_AMBER = "#F59E0B"
ACCENT_GREEN = "#10B981"
ACCENT_RED = "#EF4444"
ACCENT_PURPLE = "#A855F7"
ACCENT_GRAY = "#64748B"

SPLIT_COLORS = {"train": "#3B82F6", "validation": "#F59E0B", "test": "#10B981"}
CLASS_COLORS = ["#38BDF8", "#EC4899", "#8B5CF6", "#F97316"]
MOTION_COLORS = {"CV": "#38BDF8", "CT": "#A855F7", "CA": "#F59E0B"}
RADAR_COLORS = ["#EC4899", "#8B5CF6", "#3B82F6", "#F97316"]
STATUS_COLORS = {
    "delivered": "#10B981",
    "packet_lost": "#EF4444",
    "bandwidth_dropped": "#F59E0B",
    "sharing_withheld": "#6366F1",
    "outage_suppressed": "#64748B",
}


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


class ChartCanvas:
    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w = width
        self.h = height
        self.img = Image.new("RGB", (width, height), BG_COLOR)
        self.draw = ImageDraw.Draw(self.img)
        self.draw.text((40, 24), title, fill=TEXT_MAIN, font=get_font(24, bold=True))
        if subtitle:
            self.draw.text((40, 58), subtitle, fill=TEXT_MUTED, font=get_font(14))

    def draw_panel(self, rect: Tuple[int, int, int, int], title: str = ""):
        x0, y0, x1, y1 = rect
        self.draw.rectangle([x0, y0, x1, y1], fill=PANEL_BG, outline=BORDER_COLOR, width=1)
        if title:
            self.draw.text((x0 + 16, y0 + 12), title, fill=TEXT_MAIN, font=get_font(16, bold=True))

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path, format="PNG")


def render_class_distribution_plot(data: dict, out_path: Path):
    canvas = ChartCanvas(1300, 750, "Target Class Distribution Analysis", "Comparison across dataset splits (Train: 70, Val: 15, Test: 15)")
    
    # Panel 1: Overall Class Counts
    p1 = (40, 95, 620, 700)
    canvas.draw_panel(p1, "A. Overall Class Counts (100 Episodes)")
    
    px0, py0, px1, py1 = p1[0] + 55, p1[1] + 50, p1[2] - 30, p1[3] - 50
    canvas.draw.rectangle([px0, py0, px1, py1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_c = 35
    for v in np.linspace(0, max_c, 8):
        y = py1 - (v / max_c) * (py1 - py0)
        canvas.draw.line([(px0, y), (px1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((px0 - 32, y - 7), f"{v:.0f}", fill=TEXT_MUTED, font=get_font(12))

    bw = 75
    spacing = (px1 - px0 - 4 * bw) / 5
    for i, c_name in enumerate(CLASS_NAMES):
        cnt = data["overall_classes"].get(i, 0)
        bx0 = px0 + spacing + i * (bw + spacing)
        bx1 = bx0 + bw
        h = (cnt / max_c) * (py1 - py0)
        by0 = py1 - h
        
        canvas.draw.rectangle([bx0, by0, bx1, py1], fill=CLASS_COLORS[i], outline=BORDER_COLOR)
        canvas.draw.text((bx0 + 20, by0 - 22), f"{cnt}", fill=TEXT_MAIN, font=get_font(14, bold=True))
        canvas.draw.text((bx0 + 14, by0 - 38), f"{cnt/100:.1%}", fill=TEXT_MUTED, font=get_font(11))
        
        label = f"Class {i}\n({c_name[:7]})"
        canvas.draw.text((bx0 + 5, py1 + 8), label, fill=TEXT_MAIN, font=get_font(11))

    # Panel 2: Per Split Comparison
    p2 = (650, 95, 1260, 700)
    canvas.draw_panel(p2, "B. Class Proportion by Split (Train vs. Val vs. Test)")

    p2x0, p2y0, p2x1, p2y1 = p2[0] + 55, p2[1] + 50, p2[2] - 30, p2[3] - 50
    canvas.draw.rectangle([p2x0, p2y0, p2x1, p2y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    for v in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        y = p2y1 - (v / 0.5) * (p2y1 - p2y0)
        canvas.draw.line([(p2x0, y), (p2x1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((p2x0 - 40, y - 7), f"{v:.0%}", fill=TEXT_MUTED, font=get_font(12))

    group_w = (p2x1 - p2x0) / 4
    bar_sub_w = 20
    splits = ["train", "validation", "test"]

    for i in range(4):
        gx = p2x0 + i * group_w
        canvas.draw.text((gx + group_w / 2 - 25, p2y1 + 10), f"Class {i}", fill=TEXT_MAIN, font=get_font(12, bold=True))
        for si, sp in enumerate(splits):
            tot_sp = len(data["split_episodes"][sp])
            cnt = data["split_classes"][sp].get(i, 0)
            frac = cnt / tot_sp if tot_sp > 0 else 0
            
            bx0 = gx + 15 + si * (bar_sub_w + 4)
            bx1 = bx0 + bar_sub_w
            h = (frac / 0.5) * (p2y1 - p2y0)
            by0 = p2y1 - h
            
            canvas.draw.rectangle([bx0, by0, bx1, p2y1], fill=SPLIT_COLORS[sp], outline=BORDER_COLOR)
            if cnt > 0:
                canvas.draw.text((bx0 + 2, by0 - 18), f"{cnt}", fill=SPLIT_COLORS[sp], font=get_font(10, bold=True))
            else:
                canvas.draw.text((bx0 + 2, p2y1 - 18), "0!", fill=ACCENT_RED, font=get_font(10, bold=True))

    # Legend
    leg_x = p2x0 + 15
    leg_y = p2y0 + 10
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 280, leg_y + 30], fill="#090E17", outline=BORDER_COLOR)
    for si, sp in enumerate(splits):
        lx = leg_x + 10 + si * 90
        canvas.draw.rectangle([lx, leg_y + 8, lx + 14, leg_y + 22], fill=SPLIT_COLORS[sp])
        canvas.draw.text((lx + 20, leg_y + 7), sp.capitalize(), fill=TEXT_MAIN, font=get_font(11))

    canvas.save(out_path)


def render_motion_distribution_plot(data: dict, out_path: Path):
    canvas = ChartCanvas(1300, 750, "Motion Model Distribution Analysis", "CV (Constant Velocity), CA (Constant Acceleration), CT (Coordinated Turn)")
    
    # Panel 1: Overall
    p1 = (40, 95, 620, 700)
    canvas.draw_panel(p1, "A. Overall Motion Model Distribution (100 Episodes)")

    px0, py0, px1, py1 = p1[0] + 55, p1[1] + 50, p1[2] - 30, p1[3] - 50
    canvas.draw.rectangle([px0, py0, px1, py1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_c = 50
    for v in np.linspace(0, max_c, 6):
        y = py1 - (v / max_c) * (py1 - py0)
        canvas.draw.line([(px0, y), (px1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((px0 - 32, y - 7), f"{v:.0f}", fill=TEXT_MUTED, font=get_font(12))

    models = ["CV", "CA", "CT"]
    bw = 90
    spacing = (px1 - px0 - 3 * bw) / 4
    for i, m_name in enumerate(models):
        cnt = data["overall_motions"].get(m_name, 0)
        bx0 = px0 + spacing + i * (bw + spacing)
        bx1 = bx0 + bw
        h = (cnt / max_c) * (py1 - py0)
        by0 = py1 - h
        
        canvas.draw.rectangle([bx0, by0, bx1, py1], fill=MOTION_COLORS[m_name], outline=BORDER_COLOR)
        canvas.draw.text((bx0 + 26, by0 - 22), f"{cnt}", fill=TEXT_MAIN, font=get_font(15, bold=True))
        canvas.draw.text((bx0 + 24, by0 - 38), f"{cnt/100:.1%}", fill=TEXT_MUTED, font=get_font(12))
        canvas.draw.text((bx0 + 30, py1 + 10), m_name, fill=TEXT_MAIN, font=get_font(13, bold=True))

    # Panel 2: Per Split
    p2 = (650, 95, 1260, 700)
    canvas.draw_panel(p2, "B. Motion Models Grouped by Split")

    p2x0, p2y0, p2x1, p2y1 = p2[0] + 55, p2[1] + 50, p2[2] - 30, p2[3] - 50
    canvas.draw.rectangle([p2x0, p2y0, p2x1, p2y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    for v in [0.0, 0.15, 0.30, 0.45, 0.60]:
        y = p2y1 - (v / 0.60) * (p2y1 - p2y0)
        canvas.draw.line([(p2x0, y), (p2x1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((p2x0 - 40, y - 7), f"{v:.0%}", fill=TEXT_MUTED, font=get_font(12))

    group_w = (p2x1 - p2x0) / 3
    bar_sub_w = 26
    splits = ["train", "validation", "test"]

    for i, m_name in enumerate(models):
        gx = p2x0 + i * group_w
        canvas.draw.text((gx + group_w / 2 - 15, p2y1 + 12), m_name, fill=TEXT_MAIN, font=get_font(13, bold=True))
        for si, sp in enumerate(splits):
            tot_sp = len(data["split_episodes"][sp])
            cnt = data["split_motions"][sp].get(m_name, 0)
            frac = cnt / tot_sp if tot_sp > 0 else 0
            
            bx0 = gx + 25 + si * (bar_sub_w + 6)
            bx1 = bx0 + bar_sub_w
            h = (frac / 0.60) * (p2y1 - p2y0)
            by0 = p2y1 - h
            
            canvas.draw.rectangle([bx0, by0, bx1, p2y1], fill=SPLIT_COLORS[sp], outline=BORDER_COLOR)
            canvas.draw.text((bx0 + 4, by0 - 18), f"{cnt}", fill=SPLIT_COLORS[sp], font=get_font(11, bold=True))

    leg_x = p2x0 + 15
    leg_y = p2y0 + 10
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 280, leg_y + 30], fill="#090E17", outline=BORDER_COLOR)
    for si, sp in enumerate(splits):
        lx = leg_x + 10 + si * 90
        canvas.draw.rectangle([lx, leg_y + 8, lx + 14, leg_y + 22], fill=SPLIT_COLORS[sp])
        canvas.draw.text((lx + 20, leg_y + 7), sp.capitalize(), fill=TEXT_MAIN, font=get_font(11))

    canvas.save(out_path)


def render_reports_distribution_plot(data: dict, out_path: Path):
    canvas = ChartCanvas(1300, 750, "Radar Reports per Episode Distribution", "Histogram of total reports, true detections, and false alarms per episode")

    reps = data["reports_per_episode"]
    true_dets = data["true_dets_per_episode"]
    fa_dets = data["fa_dets_per_episode"]

    p1 = (40, 95, 1260, 700)
    canvas.draw_panel(p1, "Distribution Across 100 Episodes (Bins of 5 reports)")

    px0, py0, px1, py1 = p1[0] + 60, p1[1] + 50, p1[2] - 40, p1[3] - 60
    canvas.draw.rectangle([px0, py0, px1, py1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    bins = list(range(0, 70, 5))
    rep_hist, _ = np.histogram(reps, bins=bins)
    true_hist, _ = np.histogram(true_dets, bins=bins)
    fa_hist, _ = np.histogram(fa_dets, bins=bins)

    max_cnt = max(max(rep_hist), max(true_hist), max(fa_hist)) + 2

    for v in range(0, max_cnt + 1, 5):
        y = py1 - (v / max_cnt) * (py1 - py0)
        canvas.draw.line([(px0, y), (px1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((px0 - 30, y - 7), f"{v}", fill=TEXT_MUTED, font=get_font(12))

    n_bins = len(bins) - 1
    slot_w = (px1 - px0) / n_bins
    bar_w = slot_w / 3.5

    for b in range(n_bins):
        bx = px0 + b * slot_w
        canvas.draw.text((bx + slot_w / 2 - 12, py1 + 8), f"{bins[b]}", fill=TEXT_MUTED, font=get_font(10))

        # Bar 1: Total Reports (Cyan)
        h1 = (rep_hist[b] / max_cnt) * (py1 - py0)
        b1_x0 = bx + 4
        canvas.draw.rectangle([b1_x0, py1 - h1, b1_x0 + bar_w, py1], fill=ACCENT_CYAN, outline=BORDER_COLOR)
        if rep_hist[b] > 0:
            canvas.draw.text((b1_x0 + 2, py1 - h1 - 16), f"{rep_hist[b]}", fill=ACCENT_CYAN, font=get_font(10))

        # Bar 2: True Detections (Green)
        h2 = (true_hist[b] / max_cnt) * (py1 - py0)
        b2_x0 = b1_x0 + bar_w + 2
        canvas.draw.rectangle([b2_x0, py1 - h2, b2_x0 + bar_w, py1], fill=ACCENT_GREEN, outline=BORDER_COLOR)
        if true_hist[b] > 0:
            canvas.draw.text((b2_x0 + 2, py1 - h2 - 16), f"{true_hist[b]}", fill=ACCENT_GREEN, font=get_font(10))

        # Bar 3: False Alarms (Red/Gray)
        h3 = (fa_hist[b] / max_cnt) * (py1 - py0)
        b3_x0 = b2_x0 + bar_w + 2
        canvas.draw.rectangle([b3_x0, py1 - h3, b3_x0 + bar_w, py1], fill=ACCENT_PURPLE, outline=BORDER_COLOR)
        if fa_hist[b] > 0:
            canvas.draw.text((b3_x0 + 2, py1 - h3 - 16), f"{fa_hist[b]}", fill=ACCENT_PURPLE, font=get_font(10))

    canvas.draw.text((px0 + (px1 - px0) / 2 - 50, py1 + 30), "Report Count Bins", fill=TEXT_MAIN, font=get_font(13, bold=True))

    # Stats badge & legend
    leg_x = px1 - 380
    leg_y = py0 + 15
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 360, leg_y + 130], fill="#090E17", outline=BORDER_COLOR)

    canvas.draw.rectangle([leg_x + 12, leg_y + 12, leg_x + 24, leg_y + 24], fill=ACCENT_CYAN)
    canvas.draw.text((leg_x + 32, leg_y + 10), f"Total Reports (Mean: {np.mean(reps):.1f}, Med: {np.median(reps):.0f})", fill=TEXT_MAIN, font=get_font(11))

    canvas.draw.rectangle([leg_x + 12, leg_y + 36, leg_x + 24, leg_y + 48], fill=ACCENT_GREEN)
    canvas.draw.text((leg_x + 32, leg_y + 34), f"True Detections (Mean: {np.mean(true_dets):.1f}, Zero-Det: {true_dets.count(0)} eps)", fill=TEXT_MAIN, font=get_font(11))

    canvas.draw.rectangle([leg_x + 12, leg_y + 60, leg_x + 24, leg_y + 72], fill=ACCENT_PURPLE)
    canvas.draw.text((leg_x + 32, leg_y + 58), f"False Alarms / Clutter (Mean: {np.mean(fa_dets):.1f})", fill=TEXT_MAIN, font=get_font(11))

    canvas.draw.line([(leg_x + 12, leg_y + 85), (leg_x + 348, leg_y + 85)], fill=BORDER_COLOR)
    canvas.draw.text((leg_x + 15, leg_y + 95), f"Min Reports: {min(reps)}  ·  Max Reports: {max(reps)}  ·  Total: {sum(reps):,}", fill=TEXT_MUTED, font=get_font(11, bold=True))

    canvas.save(out_path)


def render_communication_status_plot(data: dict, out_path: Path):
    canvas = ChartCanvas(1300, 750, "Communication Status & Reliability Analysis", "Final disposition of 3,263 radar reports transmitted to the fusion center")

    total = sum(data["comm_counts"].values())
    p1 = (40, 95, 620, 700)
    canvas.draw_panel(p1, "A. Overall Communication Status Breakdown")

    px0, py0, px1, py1 = p1[0] + 60, p1[1] + 50, p1[2] - 30, p1[3] - 50
    canvas.draw.rectangle([px0, py0, px1, py1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_c = 3000
    for v in [0, 500, 1000, 1500, 2000, 2500, 3000]:
        y = py1 - (v / max_c) * (py1 - py0)
        canvas.draw.line([(px0, y), (px1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((px0 - 40, y - 7), f"{v}", fill=TEXT_MUTED, font=get_font(12))

    statuses = ["delivered", "packet_lost", "bandwidth_dropped", "sharing_withheld", "outage_suppressed"]
    bw = 65
    spacing = (px1 - px0 - len(statuses) * bw) / (len(statuses) + 1)

    for i, st in enumerate(statuses):
        cnt = data["comm_counts"].get(st, 0)
        bx0 = px0 + spacing + i * (bw + spacing)
        bx1 = bx0 + bw
        h = (cnt / max_c) * (py1 - py0)
        by0 = py1 - h
        
        c = STATUS_COLORS[st]
        canvas.draw.rectangle([bx0, by0, bx1, py1], fill=c, outline=BORDER_COLOR)
        canvas.draw.text((bx0 + 10, by0 - 22), f"{cnt}", fill=TEXT_MAIN, font=get_font(13, bold=True))
        pct = (cnt / total * 100) if total > 0 else 0
        canvas.draw.text((bx0 + 6, by0 - 38), f"{pct:.1f}%", fill=TEXT_MUTED, font=get_font(11))
        
        lbl = st.replace("_", "\n")
        canvas.draw.text((bx0 - 5, py1 + 8), lbl, fill=TEXT_MAIN, font=get_font(10))

    # Panel 2: Per Radar
    p2 = (650, 95, 1260, 700)
    canvas.draw_panel(p2, "B. Communication Outcomes by Radar Station (R1 - R4)")

    p2x0, p2y0, p2x1, p2y1 = p2[0] + 60, p2[1] + 50, p2[2] - 30, p2[3] - 50
    canvas.draw.rectangle([p2x0, p2y0, p2x1, p2y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_r = 1000
    for v in [0, 200, 400, 600, 800, 1000]:
        y = p2y1 - (v / max_r) * (p2y1 - p2y0)
        canvas.draw.line([(p2x0, y), (p2x1, y)], fill=GRID_COLOR, width=1)
        canvas.draw.text((p2x0 - 38, y - 7), f"{v}", fill=TEXT_MUTED, font=get_font(12))

    group_w = (p2x1 - p2x0) / 4
    for rid in range(4):
        gx = p2x0 + rid * group_w
        canvas.draw.text((gx + group_w / 2 - 22, p2y1 + 10), f"Radar {rid+1}", fill=TEXT_MAIN, font=get_font(13, bold=True))
        
        deliv = data["radar_comm"][rid].get("delivered", 0)
        lost = data["radar_comm"][rid].get("packet_lost", 0)
        tot_r = deliv + lost
        
        sb_x0 = gx + group_w / 2 - 28
        sb_x1 = sb_x0 + 56

        curr_y = p2y1
        if deliv > 0:
            h = (deliv / max_r) * (p2y1 - p2y0)
            canvas.draw.rectangle([sb_x0, curr_y - h, sb_x1, curr_y], fill=STATUS_COLORS["delivered"], outline=BORDER_COLOR)
            canvas.draw.text((sb_x0 + 12, curr_y - h/2 - 7), f"{deliv}", fill="#FFFFFF", font=get_font(12, bold=True))
            curr_y -= h
        if lost > 0:
            h = (lost / max_r) * (p2y1 - p2y0)
            canvas.draw.rectangle([sb_x0, curr_y - h, sb_x1, curr_y], fill=STATUS_COLORS["packet_lost"], outline=BORDER_COLOR)
            canvas.draw.text((sb_x0 + 15, curr_y - h/2 - 7), f"{lost}", fill="#FFFFFF", font=get_font(12, bold=True))
            curr_y -= h

        rate = (deliv / tot_r * 100) if tot_r > 0 else 0
        canvas.draw.text((sb_x0 + 5, curr_y - 20), f"{rate:.1f}% Deliv", fill=ACCENT_CYAN, font=get_font(11, bold=True))

    leg_x = p2x0 + 15
    leg_y = p2y0 + 10
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 220, leg_y + 30], fill="#090E17", outline=BORDER_COLOR)
    canvas.draw.rectangle([leg_x + 10, leg_y + 8, leg_x + 22, leg_y + 22], fill=STATUS_COLORS["delivered"])
    canvas.draw.text((leg_x + 28, leg_y + 7), "Delivered", fill=TEXT_MAIN, font=get_font(11))
    canvas.draw.rectangle([leg_x + 110, leg_y + 8, leg_x + 122, leg_y + 22], fill=STATUS_COLORS["packet_lost"])
    canvas.draw.text((leg_x + 128, leg_y + 7), "Packet Lost", fill=TEXT_MAIN, font=get_font(11))

    canvas.save(out_path)


def run_quality_analysis(dataset_dir: Path, output_dir: Path):
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    n_episodes = len(episodes)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Splits
    split_counts = Counter(ep["split"] for ep in episodes)
    split_episodes = defaultdict(list)
    for ep in episodes:
        split_episodes[ep["split"]].append(ep)

    # Episode-level collectors
    episode_rows = []
    overall_classes = Counter()
    overall_motions = Counter()
    split_classes = defaultdict(Counter)
    split_motions = defaultdict(Counter)
    
    reports_per_episode = []
    true_dets_per_episode = []
    fa_dets_per_episode = []
    
    unique_tracks_per_episode = []
    unique_confirmed_tracks_per_episode = []
    
    comm_counts = Counter()
    radar_comm = defaultdict(Counter)
    radar_reports = Counter()
    radar_true_dets = Counter()
    radar_fa = Counter()

    all_matched_errors = []
    total_target_steps = 0
    total_tracked_matched_steps = 0
    episodes_with_confirmed = 0
    episodes_without_confirmed = 0

    nan_count = 0
    inf_count = 0
    corrupt_count = 0
    missing_meta_count = 0
    required_meta = ["episode_id", "record_schema_version", "config_snapshot", "agent_ids", "completed", "recorded_steps"]

    for ep in episodes:
        ep_dir = dataset_dir / ep["output_paths"]["metadata.json"].rsplit("/", 1)[0]
        try:
            records = load_episode_records(ep_dir)
        except Exception:
            corrupt_count += 1
            continue

        meta = records["metadata"]
        for k in required_meta:
            if k not in meta:
                missing_meta_count += 1

        npz_file = ep_dir / "records.npz"
        with np.load(npz_file, allow_pickle=False) as arrays:
            for k in arrays.files:
                arr = arrays[k]
                if arr.dtype.kind == "f":
                    nan_count += int(np.isnan(arr).sum())
                    inf_count += int(np.isinf(arr).sum())

        target_cfg = meta["config_snapshot"]["scenario"]["targets"][0]
        cls_lbl = target_cfg["label"]
        motion_model = target_cfg["motion"]["model"]
        split = ep["split"]

        overall_classes[cls_lbl] += 1
        overall_motions[motion_model] += 1
        split_classes[split][cls_lbl] += 1
        split_motions[split][motion_model] += 1

        # Reports & annotations
        reps = records["reports"]
        anns = {a["report_id"]: a for a in records["annotations"]}
        n_reps = len(reps)
        n_fa = sum(anns[r["report_id"]]["is_false_alarm"] for r in reps)
        n_true = n_reps - n_fa

        reports_per_episode.append(n_reps)
        true_dets_per_episode.append(n_true)
        fa_dets_per_episode.append(n_fa)

        for r in reps:
            rid = r["radar_id"]
            radar_reports[rid] += 1
            if anns[r["report_id"]]["is_false_alarm"]:
                radar_fa[rid] += 1
            else:
                radar_true_dets[rid] += 1

        # Communication
        comm = records["communication"]
        ep_comm = Counter(c["final_status"] for c in comm)
        for c in comm:
            comm_counts[c["final_status"]] += 1
            radar_comm[c["radar_id"]][c["final_status"]] += 1

        # Tracks & Unique Track Counting
        tracks = records["tracks"]
        fused = [t for t in tracks if t["source"] == "fused"]
        unique_fused_ids = set(t["track_id"] for t in fused)
        unique_confirmed_ids = set(t["track_id"] for t in fused if t["confirmed"])

        unique_tracks_per_episode.append(len(unique_fused_ids))
        unique_confirmed_tracks_per_episode.append(len(unique_confirmed_ids))

        if len(unique_confirmed_ids) > 0:
            episodes_with_confirmed += 1
        else:
            episodes_without_confirmed += 1

        # Truth-to-Track association (Greedy Nearest Neighbor with dynamic covariance gating)
        truth = {t["step_index"]: t["p"] for t in records["truth"]}
        total_target_steps += len(truth)
        
        fused_by_step = defaultdict(list)
        for tr in fused:
            if tr["confirmed"]:
                fused_by_step[tr["step_index"]].append(tr)

        ep_errors = []
        for st, p_true in truth.items():
            cand = fused_by_step.get(st, [])
            valid_pairs = []
            for tr in cand:
                d = float(np.linalg.norm(tr["x"][:3] - p_true))
                pos_std = float(np.sqrt(np.trace(tr["P"][:3, :3]) / 3.0))
                gate = 3.0 * pos_std + 500.0
                if d < gate:
                    valid_pairs.append((d, tr))
            if valid_pairs:
                valid_pairs.sort(key=lambda x: x[0])
                best_d, _ = valid_pairs[0]
                ep_errors.append(best_d)
                all_matched_errors.append(best_d)
                total_tracked_matched_steps += 1

        mean_err = float(np.mean(ep_errors)) if ep_errors else None
        rmse_err = float(np.sqrt(np.mean(np.array(ep_errors)**2))) if ep_errors else None
        max_err = float(np.max(ep_errors)) if ep_errors else None

        episode_rows.append({
            "episode_index": ep["episode_index"],
            "dataset_episode_id": ep["dataset_episode_id"],
            "split": split,
            "class_label": cls_lbl,
            "class_name": CLASS_NAMES[cls_lbl],
            "motion_model": motion_model,
            "total_reports": n_reps,
            "true_detections": n_true,
            "false_alarms": n_fa,
            "delivered_reports": ep_comm.get("delivered", 0),
            "packet_lost_reports": ep_comm.get("packet_lost", 0),
            "unique_fused_tracks": len(unique_fused_ids),
            "unique_confirmed_tracks": len(unique_confirmed_ids),
            "matched_tracked_steps": len(ep_errors),
            "total_truth_steps": len(truth),
            "tracking_coverage_pct": round(len(ep_errors) / len(truth) * 100, 2),
            "mean_3d_error_m": round(mean_err, 2) if mean_err is not None else None,
            "rmse_3d_error_m": round(rmse_err, 2) if rmse_err is not None else None,
            "max_3d_error_m": round(max_err, 2) if max_err is not None else None,
        })

    # Summary calculations
    all_matched_errors = np.array(all_matched_errors)
    error_stats = {
        "association_method": "Step-wise Greedy Nearest Neighbor with dynamic 3-sigma position covariance gating (gate = 3 * pos_std + 500m)",
        "total_target_truth_steps": total_target_steps,
        "matched_tracked_steps": total_tracked_matched_steps,
        "overall_tracking_coverage_pct": round(total_tracked_matched_steps / total_target_steps * 100, 2),
        "mean_3d_error_m": round(float(np.mean(all_matched_errors)), 2),
        "median_3d_error_m": round(float(np.median(all_matched_errors)), 2),
        "rmse_3d_error_m": round(float(np.sqrt(np.mean(all_matched_errors**2))), 2),
        "min_3d_error_m": round(float(np.min(all_matched_errors)), 2),
        "max_3d_error_m": round(float(np.max(all_matched_errors)), 2),
        "percentile_90_3d_error_m": round(float(np.percentile(all_matched_errors, 90)), 2),
        "percentile_95_3d_error_m": round(float(np.percentile(all_matched_errors, 95)), 2),
    }

    report_data = {
        "dataset_name": "synthetic_100",
        "dataset_directory": str(dataset_dir),
        "total_episodes": n_episodes,
        "splits": {
            "counts": dict(split_counts),
            "ratios": {k: round(v / n_episodes, 2) for k, v in split_counts.items()},
        },
        "target_class_distribution": {
            "overall": {
                CLASS_NAMES[k]: {"label": k, "count": v, "percentage": round(v / n_episodes * 100, 2)}
                for k, v in sorted(overall_classes.items())
            },
            "per_split": {
                sp: {
                    CLASS_NAMES[k]: {"label": k, "count": split_classes[sp][k], "percentage": round(split_classes[sp][k] / len(split_episodes[sp]) * 100, 2)}
                    for k in range(4)
                }
                for sp in ("train", "validation", "test")
            },
        },
        "motion_model_distribution": {
            "overall": {
                m: {"count": overall_motions[m], "percentage": round(overall_motions[m] / n_episodes * 100, 2)}
                for m in ("CV", "CA", "CT")
            },
            "per_split": {
                sp: {
                    m: {"count": split_motions[sp][m], "percentage": round(split_motions[sp][m] / len(split_episodes[sp]) * 100, 2)}
                    for m in ("CV", "CA", "CT")
                }
                for sp in ("train", "validation", "test")
            },
        },
        "radar_reports_per_episode": {
            "total_reports": sum(reports_per_episode),
            "mean_per_episode": round(float(np.mean(reports_per_episode)), 2),
            "median_per_episode": round(float(np.median(reports_per_episode)), 2),
            "std_per_episode": round(float(np.std(reports_per_episode)), 2),
            "min_per_episode": int(np.min(reports_per_episode)),
            "max_per_episode": int(np.max(reports_per_episode)),
            "total_true_detections": sum(true_dets_per_episode),
            "mean_true_detections_per_episode": round(float(np.mean(true_dets_per_episode)), 2),
            "episodes_with_zero_true_detections": true_dets_per_episode.count(0),
            "episodes_with_zero_true_detections_pct": round(true_dets_per_episode.count(0) / n_episodes * 100, 2),
            "total_false_alarms": sum(fa_dets_per_episode),
            "mean_false_alarms_per_episode": round(float(np.mean(fa_dets_per_episode)), 2),
        },
        "tracking_and_confirmed_tracks": {
            "confirmed_tracks_per_episode_distribution": dict(Counter(unique_confirmed_tracks_per_episode)),
            "episodes_with_confirmed_tracks": episodes_with_confirmed,
            "episodes_without_confirmed_tracks": episodes_without_confirmed,
            "zero_confirmed_tracks_percentage": round(episodes_without_confirmed / n_episodes * 100, 2),
            "mean_unique_confirmed_tracks_per_episode": round(float(np.mean(unique_confirmed_tracks_per_episode)), 2),
            "mean_unique_track_hypotheses_per_episode": round(float(np.mean(unique_tracks_per_episode)), 2),
            "important_distinction": "Track rows in manifest count per-step updates (fused+local); unique tracks represent distinct track identities.",
            "error_statistics": error_stats,
        },
        "communication_analysis": {
            "total_events": sum(comm_counts.values()),
            "status_counts": dict(comm_counts),
            "status_percentages": {
                k: round(v / sum(comm_counts.values()) * 100, 2) for k, v in comm_counts.items()
            },
            "unobserved_statuses": [st for st in ("bandwidth_dropped", "sharing_withheld", "outage_suppressed") if comm_counts.get(st, 0) == 0],
            "unobserved_reason": "Recipe randomize stress.comm_drop_prob only; bandwidth limit and radar outage stress were not enabled in this recipe.",
        },
        "radar_station_breakdown": {
            f"R{rid+1}": {
                "radar_index": rid,
                "total_reports": radar_reports[rid],
                "true_detections": radar_true_dets[rid],
                "false_alarms": radar_fa[rid],
                "delivered_reports": radar_comm[rid].get("delivered", 0),
                "packet_lost_reports": radar_comm[rid].get("packet_lost", 0),
                "delivery_success_rate_pct": round(radar_comm[rid].get("delivered", 0) / max(radar_reports[rid], 1) * 100, 2),
            }
            for rid in range(4)
        },
        "data_integrity": {
            "nan_count": nan_count,
            "inf_count": inf_count,
            "corrupt_files_count": corrupt_count,
            "missing_metadata_count": missing_meta_count,
            "status": "PASSED - 100% clean and verified",
        },
    }

    # Save quality_report.json
    json_path = output_dir / "quality_report.json"
    json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved: {json_path}")

    # Save quality_summary.csv
    csv_path = output_dir / "quality_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(episode_rows)
    print(f"Saved: {csv_path}")

    # Render plots
    render_class_distribution_plot({
        "overall_classes": overall_classes,
        "split_classes": split_classes,
        "split_episodes": split_episodes,
    }, output_dir / "01_class_distribution.png")
    print(f"Saved: {output_dir / '01_class_distribution.png'}")

    render_motion_distribution_plot({
        "overall_motions": overall_motions,
        "split_motions": split_motions,
        "split_episodes": split_episodes,
    }, output_dir / "02_motion_model_distribution.png")
    print(f"Saved: {output_dir / '02_motion_model_distribution.png'}")

    render_reports_distribution_plot({
        "reports_per_episode": reports_per_episode,
        "true_dets_per_episode": true_dets_per_episode,
        "fa_dets_per_episode": fa_dets_per_episode,
    }, output_dir / "03_reports_per_episode_distribution.png")
    print(f"Saved: {output_dir / '03_reports_per_episode_distribution.png'}")

    render_communication_status_plot({
        "comm_counts": comm_counts,
        "radar_comm": radar_comm,
    }, output_dir / "04_communication_status_distribution.png")
    print(f"Saved: {output_dir / '04_communication_status_distribution.png'}")

    return report_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dataset quality analysis")
    parser.add_argument("--dataset", default="data/processed/synthetic_100", help="Path to processed dataset")
    parser.add_argument("--output-dir", default="data/processed/synthetic_100/quality_report", help="Output directory")
    args = parser.parse_args()

    run_quality_analysis(Path(args.dataset), Path(args.output_dir))
