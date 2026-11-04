"""Standalone visualization script for GP-MARL-RadNet synthetic dataset.

Reads Canonical Episode Records v2 using allow_pickle=False and generates
the four required diagnostic and verification plots for the first train episode:
  1. Target ground truth XY trajectory and radar measurements
  2. Target ground truth XY trajectory and centralized tracker estimates
  3. Ground truth position and tracker position error over time
  4. Distribution of radar report communication statuses (delivered, packet_lost, etc.)

Outputs PNG figures into data/processed/synthetic_100/plots/.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.simulation_service import load_episode_records

# --- Color Theme (Modern Dark Tech) ---
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

RADAR_COLORS = ["#EC4899", "#8B5CF6", "#3B82F6", "#F97316"]
STATUS_COLORS = {
    "delivered": "#10B981",
    "packet_lost": "#EF4444",
    "bandwidth_dropped": "#F59E0B",
    "sharing_withheld": "#6366F1",
    "outage_suppressed": "#64748B",
}


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Safely obtain TrueType font on Windows/Linux or fallback to default."""
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


class PlotCanvas:
    """Helper to draw multi-panel dark theme charts using Pillow."""

    def __init__(self, width: int, height: int, title: str, subtitle: str = ""):
        self.w = width
        self.h = height
        self.img = Image.new("RGB", (width, height), BG_COLOR)
        self.draw = ImageDraw.Draw(self.img)
        
        # Header
        self.draw.text((40, 24), title, fill=TEXT_MAIN, font=get_font(26, bold=True))
        if subtitle:
            self.draw.text((40, 60), subtitle, fill=TEXT_MUTED, font=get_font(15))

    def draw_panel(self, rect: Tuple[int, int, int, int], title: str = ""):
        x0, y0, x1, y1 = rect
        self.draw.rectangle([x0, y0, x1, y1], fill=PANEL_BG, outline=BORDER_COLOR, width=1)
        if title:
            self.draw.text((x0 + 16, y0 + 12), title, fill=TEXT_MAIN, font=get_font(17, bold=True))

    def draw_grid_and_axes(
        self,
        rect: Tuple[int, int, int, int],
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        xlabel: str = "",
        ylabel: str = "",
        xticks_count: int = 5,
        yticks_count: int = 5,
        equal_aspect: bool = False,
    ) -> Tuple[float, float, float, float]:
        """Draw grid, ticks, labels and return transform factors (scale_x, scale_y, ox, oy)."""
        x0, y0, x1, y1 = rect
        plot_x0 = x0 + 75
        plot_y0 = y0 + 45
        plot_x1 = x1 - 30
        plot_y1 = y1 - 45

        # Inner plot background
        self.draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

        pw = plot_x1 - plot_x0
        ph = plot_y1 - plot_y0

        xmin, xmax = xlim
        ymin, ymax = ylim
        if xmax <= xmin:
            xmax = xmin + 1.0
        if ymax <= ymin:
            ymax = ymin + 1.0

        if equal_aspect:
            dx = xmax - xmin
            dy = ymax - ymin
            span = max(dx, dy)
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            xmin = cx - 0.5 * span
            xmax = cx + 0.5 * span
            ymin = cy - 0.5 * span
            ymax = cy + 0.5 * span

        sx = pw / (xmax - xmin)
        sy = ph / (ymax - ymin)

        def to_screen(x: float, y: float) -> Tuple[float, float]:
            return plot_x0 + (x - xmin) * sx, plot_y1 - (y - ymin) * sy

        # X Ticks & Grid
        f_tick = get_font(12)
        f_label = get_font(14, bold=True)
        xticks = np.linspace(xmin, xmax, xticks_count)
        for val in xticks:
            px, _ = to_screen(val, ymin)
            if plot_x0 <= px <= plot_x1:
                self.draw.line([(px, plot_y0), (px, plot_y1)], fill=GRID_COLOR, width=1)
                self.draw.line([(px, plot_y1), (px, plot_y1 + 4)], fill=TEXT_MUTED, width=1)
                txt = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"
                self.draw.text((px - 14, plot_y1 + 8), txt, fill=TEXT_MUTED, font=f_tick)

        # Y Ticks & Grid
        yticks = np.linspace(ymin, ymax, yticks_count)
        for val in yticks:
            _, py = to_screen(xmin, val)
            if plot_y0 <= py <= plot_y1:
                self.draw.line([(plot_x0, py), (plot_x1, py)], fill=GRID_COLOR, width=1)
                self.draw.line([(plot_x0 - 4, py), (plot_x0, py)], fill=TEXT_MUTED, width=1)
                txt = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"
                self.draw.text((plot_x0 - 55, py - 7), txt, fill=TEXT_MUTED, font=f_tick)

        # Axis Labels
        if xlabel:
            self.draw.text((plot_x0 + pw / 2 - 40, plot_y1 + 25), xlabel, fill=TEXT_MAIN, font=f_label)
        if ylabel:
            self.draw.text((plot_x0 - 65, plot_y0 - 25), ylabel, fill=TEXT_MAIN, font=f_label)

        return xmin, ymin, sx, sy, plot_x0, plot_y1, plot_x1, plot_y0

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.img.save(path, format="PNG")


def plot_target_and_radar_measurements(records: dict, output_path: Path):
    """Plot 1: Ground truth target trajectory with radar measurements."""
    truth = records["truth"]
    reports = records["reports"]
    annotations = {a["report_id"]: a for a in records["annotations"]}
    radars = records["metadata"]["config_snapshot"]["scenario"]["radars"]

    truth_p = np.array([t["p"] for t in truth])
    rep_xyz = np.array([r["world_xyz"] for r in reports]) if reports else np.empty((0, 3))
    
    canvas = PlotCanvas(
        1500, 850,
        "Figure 1: Target Ground Truth XY Trajectory & Radar Detections",
        f"Episode: {records['metadata']['dataset_provenance']['dataset_episode_id']} · Class: {truth[0]['class_label']} · Duration: {truth[-1]['t_s']:.0f}s"
    )

    # Panel 1 (Left): Full Network View
    p1_rect = (40, 100, 730, 800)
    canvas.draw_panel(p1_rect, "A. Network Surveillance Area (16 × 16 km)")
    xmin1, ymin1, sx1, sy1, px0_1, py1_1, px1_1, py0_1 = canvas.draw_grid_and_axes(
        p1_rect, (-8000, 8000), (-8000, 8000), xlabel="X [m]", ylabel="Y [m]", equal_aspect=True
    )

    def s1(x, y):
        return px0_1 + (x - xmin1) * sx1, py1_1 - (y - ymin1) * sy1

    # Draw radars on panel 1
    f_radar = get_font(13, bold=True)
    for i, r in enumerate(radars):
        rx, ry = r["position"][:2]
        sx, sy = s1(rx, ry)
        c = RADAR_COLORS[i % len(RADAR_COLORS)]
        canvas.draw.rectangle([sx - 7, sy - 7, sx + 7, sy + 7], fill=c, outline=TEXT_MAIN, width=1)
        canvas.draw.text((sx + 10, sy - 8), f"{r.get('name', f'R{i+1}')}", fill=c, font=f_radar)

    # Draw false alarms (clutter) on panel 1
    for r in reports:
        ann = annotations.get(r["report_id"], {})
        is_fa = ann.get("is_false_alarm", False)
        wx, wy = r["world_xyz"][:2]
        sx, sy = s1(wx, wy)
        if px0_1 <= sx <= px1_1 and py0_1 <= sy <= py1_1:
            if is_fa:
                # Gray 'x' for clutter
                canvas.draw.line([(sx - 4, sy - 4), (sx + 4, sy + 4)], fill=ACCENT_GRAY, width=1)
                canvas.draw.line([(sx - 4, sy + 4), (sx + 4, sy - 4)], fill=ACCENT_GRAY, width=1)

    # Draw true target trajectory on panel 1
    t_coords_1 = [s1(p[0], p[1]) for p in truth_p]
    for k in range(len(t_coords_1) - 1):
        canvas.draw.line([t_coords_1[k], t_coords_1[k + 1]], fill=ACCENT_CYAN, width=3)

    # Draw target detections on panel 1
    for r in reports:
        ann = annotations.get(r["report_id"], {})
        if not ann.get("is_false_alarm", False):
            sx, sy = s1(r["world_xyz"][0], r["world_xyz"][1])
            c = RADAR_COLORS[r["radar_id"] % len(RADAR_COLORS)]
            canvas.draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=c, outline=TEXT_MAIN, width=1)

    # Panel 2 (Right): Zoomed Flight Corridor
    p2_rect = (760, 100, 1460, 800)
    canvas.draw_panel(p2_rect, "B. Zoomed Target Corridor & Detections (Local ENU)")
    
    pad = 200
    tx_min, tx_max = truth_p[:, 0].min() - pad, truth_p[:, 0].max() + pad
    ty_min, ty_max = truth_p[:, 1].min() - pad, truth_p[:, 1].max() + pad
    xmin2, ymin2, sx2, sy2, px0_2, py1_2, px1_2, py0_2 = canvas.draw_grid_and_axes(
        p2_rect, (tx_min, tx_max), (ty_min, ty_max), xlabel="X [m]", ylabel="Y [m]", equal_aspect=True
    )

    def s2(x, y):
        return px0_2 + (x - xmin2) * sx2, py1_2 - (y - ymin2) * sy2

    # Draw trajectory line
    t_coords_2 = [s2(p[0], p[1]) for p in truth_p]
    for k in range(len(t_coords_2) - 1):
        canvas.draw.line([t_coords_2[k], t_coords_2[k + 1]], fill=ACCENT_CYAN, width=3)

    # Start and End points
    s_pt = t_coords_2[0]
    e_pt = t_coords_2[-1]
    canvas.draw.ellipse([s_pt[0] - 6, s_pt[1] - 6, s_pt[0] + 6, s_pt[1] + 6], fill=ACCENT_GREEN, outline=TEXT_MAIN, width=2)
    canvas.draw.text((s_pt[0] + 10, s_pt[1] - 8), f"Start (t=0s)", fill=ACCENT_GREEN, font=get_font(12, bold=True))
    canvas.draw.ellipse([e_pt[0] - 6, e_pt[1] - 6, e_pt[0] + 6, e_pt[1] + 6], fill=ACCENT_RED, outline=TEXT_MAIN, width=2)
    canvas.draw.text((e_pt[0] + 10, e_pt[1] - 8), f"End (t=30s)", fill=ACCENT_RED, font=get_font(12, bold=True))

    # True target detections
    f_rep = get_font(11)
    for r in reports:
        ann = annotations.get(r["report_id"], {})
        if not ann.get("is_false_alarm", False):
            sx, sy = s2(r["world_xyz"][0], r["world_xyz"][1])
            c = RADAR_COLORS[r["radar_id"] % len(RADAR_COLORS)]
            canvas.draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=c, outline=TEXT_MAIN, width=1)
            canvas.draw.text((sx + 7, sy - 6), f"R{r['radar_id']+1} ({r['t_s']:.0f}s)", fill=TEXT_MUTED, font=f_rep)

    # Legend for Panel 2
    leg_x = px0_2 + 15
    leg_y = py0_2 + 15
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 220, leg_y + 115], fill="#090E17", outline=BORDER_COLOR, width=1)
    canvas.draw.line([(leg_x + 10, leg_y + 20), (leg_x + 35, leg_y + 20)], fill=ACCENT_CYAN, width=3)
    canvas.draw.text((leg_x + 45, leg_y + 12), "True Trajectory", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.ellipse([leg_x + 18, leg_y + 40, leg_x + 26, leg_y + 48], fill=ACCENT_GREEN, outline=TEXT_MAIN)
    canvas.draw.text((leg_x + 45, leg_y + 36), "Trajectory Start", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.ellipse([leg_x + 18, leg_y + 64, leg_x + 26, leg_y + 72], fill=ACCENT_RED, outline=TEXT_MAIN)
    canvas.draw.text((leg_x + 45, leg_y + 60), "Trajectory End", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.ellipse([leg_x + 18, leg_y + 88, leg_x + 26, leg_y + 96], fill=RADAR_COLORS[1], outline=TEXT_MAIN)
    canvas.draw.text((leg_x + 45, leg_y + 84), "Radar Measurement", fill=TEXT_MAIN, font=get_font(12))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def plot_target_and_fused_tracker(records: dict, output_path: Path):
    """Plot 2: Ground truth target trajectory with centralized tracker estimates."""
    truth = records["truth"]
    tracks = records["tracks"]
    fused_tracks = [t for t in tracks if t["source"] == "fused"]
    radars = records["metadata"]["config_snapshot"]["scenario"]["radars"]

    truth_p = np.array([t["p"] for t in truth])
    
    # Identify the primary confirmed target track (highest number of confirmed hits)
    confirmed_fused = [t for t in fused_tracks if t["confirmed"]]
    track_ids = Counter(t["track_id"] for t in confirmed_fused)
    primary_id = track_ids.most_common(1)[0][0] if track_ids else (fused_tracks[0]["track_id"] if fused_tracks else None)
    
    target_fused = [t for t in fused_tracks if t["track_id"] == primary_id] if primary_id else []

    canvas = PlotCanvas(
        1500, 850,
        "Figure 2: Target Ground Truth XY Trajectory vs. Centralized Tracker Estimate",
        f"Track ID: {primary_id} · Total Hits: {len(target_fused)} · Confirmed Steps: {len([t for t in target_fused if t['confirmed']])}"
    )

    # Panel 1: Full Network View
    p1_rect = (40, 100, 730, 800)
    canvas.draw_panel(p1_rect, "A. Network Surveillance Overview")
    xmin1, ymin1, sx1, sy1, px0_1, py1_1, px1_1, py0_1 = canvas.draw_grid_and_axes(
        p1_rect, (-8000, 8000), (-8000, 8000), xlabel="X [m]", ylabel="Y [m]", equal_aspect=True
    )

    def s1(x, y):
        return px0_1 + (x - xmin1) * sx1, py1_1 - (y - ymin1) * sy1

    # Draw radars
    for i, r in enumerate(radars):
        rx, ry = r["position"][:2]
        sx, sy = s1(rx, ry)
        c = RADAR_COLORS[i % len(RADAR_COLORS)]
        canvas.draw.rectangle([sx - 7, sy - 7, sx + 7, sy + 7], fill=c, outline=TEXT_MAIN, width=1)
        canvas.draw.text((sx + 10, sy - 8), f"{r.get('name', f'R{i+1}')}", fill=c, font=get_font(13, bold=True))

    # Draw truth trajectory
    t_coords_1 = [s1(p[0], p[1]) for p in truth_p]
    for k in range(len(t_coords_1) - 1):
        canvas.draw.line([t_coords_1[k], t_coords_1[k + 1]], fill=ACCENT_CYAN, width=3)

    # Draw tracker trajectory
    tr_coords_1 = [s1(t["x"][0], t["x"][1]) for t in target_fused]
    for k in range(len(tr_coords_1) - 1):
        canvas.draw.line([tr_coords_1[k], tr_coords_1[k + 1]], fill=ACCENT_AMBER, width=2)

    # Panel 2: Zoomed Tracker Detail
    p2_rect = (760, 100, 1460, 800)
    canvas.draw_panel(p2_rect, "B. Zoomed Trajectory: Ground Truth vs. Centralized Tracker")
    
    pad = 200
    tx_min, tx_max = truth_p[:, 0].min() - pad, truth_p[:, 0].max() + pad
    ty_min, ty_max = truth_p[:, 1].min() - pad, truth_p[:, 1].max() + pad
    xmin2, ymin2, sx2, sy2, px0_2, py1_2, px1_2, py0_2 = canvas.draw_grid_and_axes(
        p2_rect, (tx_min, tx_max), (ty_min, ty_max), xlabel="X [m]", ylabel="Y [m]", equal_aspect=True
    )

    def s2(x, y):
        return px0_2 + (x - xmin2) * sx2, py1_2 - (y - ymin2) * sy2

    # Draw truth trajectory
    t_coords_2 = [s2(p[0], p[1]) for p in truth_p]
    for k in range(len(t_coords_2) - 1):
        canvas.draw.line([t_coords_2[k], t_coords_2[k + 1]], fill=ACCENT_CYAN, width=3)

    # Draw tracker estimates
    tr_coords_2 = [s2(t["x"][0], t["x"][1]) for t in target_fused]
    for k in range(len(tr_coords_2) - 1):
        canvas.draw.line([tr_coords_2[k], tr_coords_2[k + 1]], fill=ACCENT_AMBER, width=2)

    # Markers for tracker points
    for t in target_fused:
        sx, sy = s2(t["x"][0], t["x"][1])
        if t["status"] == "confirmed":
            # Direct measurement update
            canvas.draw.ellipse([sx - 4, sy - 4, sx + 4, sy + 4], fill=ACCENT_AMBER, outline=TEXT_MAIN, width=1)
        elif t["status"] == "confirmed_coasting":
            # EKF prediction coasting
            canvas.draw.rectangle([sx - 3, sy - 3, sx + 3, sy + 3], fill="#78350F", outline=ACCENT_AMBER, width=1)
        else:
            # Tentative
            canvas.draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=ACCENT_GRAY, outline=TEXT_MAIN, width=1)

    # Track Initiation & Final points
    if target_fused:
        init_pt = s2(target_fused[0]["x"][0], target_fused[0]["x"][1])
        final_pt = s2(target_fused[-1]["x"][0], target_fused[-1]["x"][1])
        canvas.draw.text((init_pt[0] + 10, init_pt[1] - 8), f"Track Born (t={target_fused[0]['t_s']:.1f}s)", fill=ACCENT_AMBER, font=get_font(12, bold=True))
        canvas.draw.text((final_pt[0] + 10, final_pt[1] - 8), f"Track End (t={target_fused[-1]['t_s']:.1f}s)", fill=ACCENT_AMBER, font=get_font(12, bold=True))

    # Legend
    leg_x = px0_2 + 15
    leg_y = py0_2 + 15
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 230, leg_y + 115], fill="#090E17", outline=BORDER_COLOR, width=1)
    canvas.draw.line([(leg_x + 10, leg_y + 20), (leg_x + 35, leg_y + 20)], fill=ACCENT_CYAN, width=3)
    canvas.draw.text((leg_x + 45, leg_y + 12), "Ground Truth Trajectory", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.line([(leg_x + 10, leg_y + 44), (leg_x + 35, leg_y + 44)], fill=ACCENT_AMBER, width=2)
    canvas.draw.ellipse([leg_x + 20, leg_y + 40, leg_x + 26, leg_y + 46], fill=ACCENT_AMBER, outline=TEXT_MAIN)
    canvas.draw.text((leg_x + 45, leg_y + 36), "Tracker Confirmed (Update)", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.rectangle([leg_x + 19, leg_y + 64, leg_x + 27, leg_y + 72], fill="#78350F", outline=ACCENT_AMBER)
    canvas.draw.text((leg_x + 45, leg_y + 60), "Tracker Coasting (Prediction)", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.ellipse([leg_x + 20, leg_y + 88, leg_x + 26, leg_y + 94], fill=ACCENT_GRAY, outline=TEXT_MAIN)
    canvas.draw.text((leg_x + 45, leg_y + 84), "Tracker Tentative", fill=TEXT_MAIN, font=get_font(12))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def plot_tracking_error_vs_time(records: dict, output_path: Path):
    """Plot 3: Ground truth position and centralized tracker position error vs. time."""
    truth = records["truth"]
    tracks = records["tracks"]
    
    truth_by_step = {t["step_index"]: t for t in truth}
    fused_tracks = [t for t in tracks if t["source"] == "fused"]
    
    # Primary confirmed track
    confirmed_fused = [t for t in fused_tracks if t["confirmed"]]
    track_ids = Counter(t["track_id"] for t in confirmed_fused)
    primary_id = track_ids.most_common(1)[0][0] if track_ids else fused_tracks[0]["track_id"]
    target_fused = [t for t in fused_tracks if t["track_id"] == primary_id]

    times = []
    err_3d = []
    err_xy = []
    err_z = []
    x_truth = []
    x_est = []
    y_truth = []
    y_est = []
    z_truth = []
    z_est = []

    for t in target_fused:
        st = t["step_index"]
        if st in truth_by_step:
            p_tr = truth_by_step[st]["p"]
            p_est = t["x"][:3]
            times.append(t["t_s"])
            e3 = np.linalg.norm(p_est - p_tr)
            exy = np.linalg.norm(p_est[:2] - p_tr[:2])
            ez = abs(p_est[2] - p_tr[2])
            err_3d.append(e3)
            err_xy.append(exy)
            err_z.append(ez)
            
            x_truth.append(p_tr[0])
            x_est.append(p_est[0])
            y_truth.append(p_tr[1])
            y_est.append(p_est[1])
            z_truth.append(p_tr[2])
            z_est.append(p_est[2])

    times = np.array(times)
    err_3d = np.array(err_3d)
    err_xy = np.array(err_xy)
    err_z = np.array(err_z)

    canvas = PlotCanvas(
        1500, 900,
        "Figure 3: Ground Truth Position & Centralized Tracking Error Over Time",
        f"Episode: {records['metadata']['dataset_provenance']['dataset_episode_id']} · Mean 3D Error: {np.mean(err_3d):.2f}m · Max 3D Error: {np.max(err_3d):.2f}m"
    )

    # Panel 1 (Top): Position Coordinates X, Y, Z vs. Time
    p1_rect = (40, 100, 1460, 480)
    canvas.draw_panel(p1_rect, "A. Position Coordinates: Truth vs. Centralized Tracker Estimate")
    
    pos_min = min(min(x_truth), min(y_truth), min(z_truth)) - 100
    pos_max = max(max(x_truth), max(y_truth), max(z_truth)) + 100
    t_min, t_max = 0.0, 30.0

    xmin1, ymin1, sx1, sy1, px0_1, py1_1, px1_1, py0_1 = canvas.draw_grid_and_axes(
        p1_rect, (t_min, t_max), (pos_min, pos_max), xlabel="Time t [s]", ylabel="Position [m]", xticks_count=7, yticks_count=6
    )

    def s1(t, val):
        return px0_1 + (t - xmin1) * sx1, py1_1 - (val - ymin1) * sy1

    # Draw X, Y, Z coordinates
    coords = [
        ("X [m]", x_truth, x_est, "#38BDF8", "#0284C7"),
        ("Y [m]", y_truth, y_est, "#34D399", "#059669"),
        ("Z [m]", z_truth, z_est, "#F472B6", "#DB2777"),
    ]
    for label, tr_vals, est_vals, c_tr, c_est in coords:
        # Truth line
        for k in range(len(times) - 1):
            canvas.draw.line([s1(times[k], tr_vals[k]), s1(times[k + 1], tr_vals[k + 1])], fill=c_tr, width=3)
        # Estimated points
        for k in range(len(times)):
            pt = s1(times[k], est_vals[k])
            canvas.draw.ellipse([pt[0] - 2, pt[1] - 2, pt[0] + 2, pt[1] + 2], fill=c_est, outline=TEXT_MAIN, width=1)

    # Legend for Panel 1
    leg1_x = px0_1 + 15
    leg1_y = py0_1 + 10
    canvas.draw.rectangle([leg1_x, leg1_y, leg1_x + 360, leg1_y + 35], fill="#090E17", outline=BORDER_COLOR, width=1)
    canvas.draw.line([(leg1_x + 10, leg1_y + 18), (leg1_x + 30, leg1_y + 18)], fill="#38BDF8", width=3)
    canvas.draw.text((leg1_x + 35, leg1_y + 10), "X Truth/Est", fill=TEXT_MAIN, font=get_font(12))
    canvas.draw.line([(leg1_x + 130, leg1_y + 18), (leg1_x + 150, leg1_y + 18)], fill="#34D399", width=3)
    canvas.draw.text((leg1_x + 155, leg1_y + 10), "Y Truth/Est", fill=TEXT_MAIN, font=get_font(12))
    canvas.draw.line([(leg1_x + 250, leg1_y + 18), (leg1_x + 270, leg1_y + 18)], fill="#F472B6", width=3)
    canvas.draw.text((leg1_x + 275, leg1_y + 10), "Z Truth/Est", fill=TEXT_MAIN, font=get_font(12))

    # Panel 2 (Bottom): Tracking Position Errors vs. Time
    p2_rect = (40, 500, 1460, 860)
    canvas.draw_panel(p2_rect, "B. Tracker Position Estimation Errors Over Time")
    
    err_max = math.ceil(max(err_3d) * 1.15)
    xmin2, ymin2, sx2, sy2, px0_2, py1_2, px1_2, py0_2 = canvas.draw_grid_and_axes(
        p2_rect, (t_min, t_max), (0.0, err_max), xlabel="Time t [s]", ylabel="Error [m]", xticks_count=7, yticks_count=5
    )

    def s2(t, val):
        return px0_2 + (t - xmin2) * sx2, py1_2 - (val - ymin2) * sy2

    # Draw 3D Euclidean Error (thick Amber line)
    for k in range(len(times) - 1):
        canvas.draw.line([s2(times[k], err_3d[k]), s2(times[k + 1], err_3d[k + 1])], fill=ACCENT_AMBER, width=3)

    # Draw 2D Horizontal XY Error (Cyan line)
    for k in range(len(times) - 1):
        canvas.draw.line([s2(times[k], err_xy[k]), s2(times[k + 1], err_xy[k + 1])], fill=ACCENT_CYAN, width=2)

    # Draw Vertical Z Error (Purple line)
    for k in range(len(times) - 1):
        canvas.draw.line([s2(times[k], err_z[k]), s2(times[k + 1], err_z[k + 1])], fill=ACCENT_PURPLE, width=1)

    # Mean error reference dashed line
    mean_val = float(np.mean(err_3d))
    _, my = s2(0, mean_val)
    for px in range(int(px0_2), int(px1_2), 12):
        canvas.draw.line([(px, my), (min(px + 6, int(px1_2)), my)], fill=TEXT_MUTED, width=1)
    canvas.draw.text((px1_2 - 130, my - 16), f"Mean 3D: {mean_val:.1f} m", fill=TEXT_MUTED, font=get_font(12, bold=True))

    # Legend for Panel 2
    leg2_x = px0_2 + 15
    leg2_y = py0_2 + 10
    canvas.draw.rectangle([leg2_x, leg2_y, leg2_x + 400, leg2_y + 35], fill="#090E17", outline=BORDER_COLOR, width=1)
    canvas.draw.line([(leg2_x + 10, leg2_y + 18), (leg2_x + 30, leg2_y + 18)], fill=ACCENT_AMBER, width=3)
    canvas.draw.text((leg2_x + 35, leg2_y + 10), "3D Error ||p - p_hat||", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.line([(leg2_x + 190, leg2_y + 18), (leg2_x + 210, leg2_y + 18)], fill=ACCENT_CYAN, width=2)
    canvas.draw.text((leg2_x + 215, leg2_y + 10), "XY Error", fill=TEXT_MAIN, font=get_font(12))

    canvas.draw.line([(leg2_x + 300, leg2_y + 18), (leg2_x + 320, leg2_y + 18)], fill=ACCENT_PURPLE, width=1)
    canvas.draw.text((leg2_x + 325, leg2_y + 10), "Z Error", fill=TEXT_MAIN, font=get_font(12))

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def plot_report_communication_distribution(records: dict, output_path: Path):
    """Plot 4: Distribution of radar reports across communication final statuses."""
    communication = records["communication"]
    reports = records["reports"]
    radars = records["metadata"]["config_snapshot"]["scenario"]["radars"]

    total_reports = len(communication)
    status_counts = Counter(c["final_status"] for c in communication)
    radar_counts = Counter((c["radar_id"], c["final_status"]) for c in communication)

    canonical_statuses = [
        "delivered",
        "packet_lost",
        "bandwidth_dropped",
        "sharing_withheld",
        "outage_suppressed",
    ]

    canvas = PlotCanvas(
        1500, 900,
        "Figure 4: Distribution of Radar Reports Across Communication Statuses",
        f"Episode: {records['metadata']['dataset_provenance']['dataset_episode_id']} · Total Reports: {total_reports} · Delivered: {status_counts['delivered']} ({status_counts['delivered']/max(total_reports,1):.1%})"
    )

    # Panel 1 (Top Left): Overall Distribution Bar Chart
    p1_rect = (40, 100, 730, 520)
    canvas.draw_panel(p1_rect, "A. Communication Status Distribution (Overall)")

    plot_x0, plot_y0, plot_x1, plot_y1 = p1_rect[0] + 60, p1_rect[1] + 50, p1_rect[2] - 30, p1_rect[3] - 50
    canvas.draw.rectangle([plot_x0, plot_y0, plot_x1, plot_y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_c = max(max(status_counts.values(), default=1), 10)
    max_y = math.ceil(max_c * 1.25)
    
    # Y-axis ticks
    for val in np.linspace(0, max_y, 5):
        py = plot_y1 - (val / max_y) * (plot_y1 - plot_y0)
        canvas.draw.line([(plot_x0, py), (plot_x1, py)], fill=GRID_COLOR, width=1)
        canvas.draw.text((plot_x0 - 35, py - 7), f"{val:.0f}", fill=TEXT_MUTED, font=get_font(12))

    # Bars for 5 statuses
    bar_w = 70
    spacing = (plot_x1 - plot_x0 - len(canonical_statuses) * bar_w) / (len(canonical_statuses) + 1)
    f_bar = get_font(13, bold=True)
    f_lbl = get_font(11)

    for i, st in enumerate(canonical_statuses):
        cnt = status_counts.get(st, 0)
        bx0 = plot_x0 + spacing + i * (bar_w + spacing)
        bx1 = bx0 + bar_w
        bh = (cnt / max_y) * (plot_y1 - plot_y0)
        by0 = plot_y1 - bh
        by1 = plot_y1
        
        c = STATUS_COLORS.get(st, ACCENT_GRAY)
        canvas.draw.rectangle([bx0, by0, bx1, by1], fill=c, outline=BORDER_COLOR, width=1)

        # Count and percentage above bar
        pct = (cnt / total_reports * 100) if total_reports > 0 else 0
        canvas.draw.text((bx0 + 10, by0 - 22), f"{cnt}", fill=TEXT_MAIN, font=f_bar)
        canvas.draw.text((bx0 + 5, by0 - 38), f"{pct:.1f}%", fill=TEXT_MUTED, font=f_lbl)

        # X label
        clean_lbl = st.replace("_", "\n")
        canvas.draw.text((bx0 - 5, by1 + 8), clean_lbl, fill=TEXT_MAIN, font=f_lbl)

    # Panel 2 (Top Right): Breakdown by Radar Station
    p2_rect = (760, 100, 1460, 520)
    canvas.draw_panel(p2_rect, "B. Report Outcomes Grouped by Radar Station (R1 - R4)")

    p2_x0, p2_y0, p2_x1, p2_y1 = p2_rect[0] + 60, p2_rect[1] + 50, p2_rect[2] - 30, p2_rect[3] - 50
    canvas.draw.rectangle([p2_x0, p2_y0, p2_x1, p2_y1], fill="#0D131F", outline=BORDER_COLOR, width=1)

    max_r_cnt = 20
    for val in np.linspace(0, max_r_cnt, 5):
        py = p2_y1 - (val / max_r_cnt) * (p2_y1 - p2_y0)
        canvas.draw.line([(p2_x0, py), (p2_x1, py)], fill=GRID_COLOR, width=1)
        canvas.draw.text((p2_x0 - 35, py - 7), f"{val:.0f}", fill=TEXT_MUTED, font=get_font(12))

    r_group_w = (p2_x1 - p2_x0) / 4
    for rid in range(len(radars)):
        gx = p2_x0 + rid * r_group_w
        canvas.draw.text((gx + r_group_w / 2 - 20, p2_y1 + 10), f"Radar {rid+1}", fill=TEXT_MAIN, font=get_font(13, bold=True))
        
        # Stacked bar: delivered, packet_lost, others
        deliv = radar_counts.get((rid, "delivered"), 0)
        lost = radar_counts.get((rid, "packet_lost"), 0)
        other = sum(radar_counts.get((rid, st), 0) for st in canonical_statuses if st not in ("delivered", "packet_lost"))
        tot_r = deliv + lost + other

        sb_x0 = gx + r_group_w / 2 - 30
        sb_x1 = sb_x0 + 60

        # Draw stacked segments
        curr_y = p2_y1
        if deliv > 0:
            h = (deliv / max_r_cnt) * (p2_y1 - p2_y0)
            canvas.draw.rectangle([sb_x0, curr_y - h, sb_x1, curr_y], fill=STATUS_COLORS["delivered"], outline=BORDER_COLOR)
            canvas.draw.text((sb_x0 + 15, curr_y - h/2 - 7), f"{deliv}", fill="#FFFFFF", font=get_font(12, bold=True))
            curr_y -= h
        if lost > 0:
            h = (lost / max_r_cnt) * (p2_y1 - p2_y0)
            canvas.draw.rectangle([sb_x0, curr_y - h, sb_x1, curr_y], fill=STATUS_COLORS["packet_lost"], outline=BORDER_COLOR)
            canvas.draw.text((sb_x0 + 20, curr_y - h/2 - 7), f"{lost}", fill="#FFFFFF", font=get_font(12, bold=True))
            curr_y -= h
        if other > 0:
            h = (other / max_r_cnt) * (p2_y1 - p2_y0)
            canvas.draw.rectangle([sb_x0, curr_y - h, sb_x1, curr_y], fill=STATUS_COLORS["bandwidth_dropped"], outline=BORDER_COLOR)
            curr_y -= h

        canvas.draw.text((sb_x0 + 12, curr_y - 20), f"Total: {tot_r}", fill=TEXT_MUTED, font=get_font(11))

    # Legend for Panel 2
    leg_x = p2_x0 + 15
    leg_y = p2_y0 + 10
    canvas.draw.rectangle([leg_x, leg_y, leg_x + 280, leg_y + 35], fill="#090E17", outline=BORDER_COLOR, width=1)
    canvas.draw.rectangle([leg_x + 10, leg_y + 12, leg_x + 22, leg_y + 24], fill=STATUS_COLORS["delivered"])
    canvas.draw.text((leg_x + 28, leg_y + 10), "Delivered", fill=TEXT_MAIN, font=get_font(11))
    canvas.draw.rectangle([leg_x + 100, leg_y + 12, leg_x + 112, leg_y + 24], fill=STATUS_COLORS["packet_lost"])
    canvas.draw.text((leg_x + 118, leg_y + 10), "Packet Lost", fill=TEXT_MAIN, font=get_font(11))
    canvas.draw.rectangle([leg_x + 200, leg_y + 12, leg_x + 212, leg_y + 24], fill=STATUS_COLORS["bandwidth_dropped"])
    canvas.draw.text((leg_x + 218, leg_y + 10), "Dropped", fill=TEXT_MAIN, font=get_font(11))

    # Panel 3 (Bottom): Timeline of Transmission Events
    p3_rect = (40, 540, 1460, 860)
    canvas.draw_panel(p3_rect, "C. Report Generation and Communication Timeline (t = 0 to 30 s)")

    xmin3, ymin3, sx3, sy3, px0_3, py1_3, px1_3, py0_3 = canvas.draw_grid_and_axes(
        p3_rect, (0.0, 30.0), (-0.5, 3.5), xlabel="Simulation Time t [s]", ylabel="Radar", xticks_count=7, yticks_count=4
    )

    # Label radar rows on Y axis
    for rid in range(len(radars)):
        _, py = px0_3 + (0 - xmin3) * sx3, py1_3 - (rid - ymin3) * sy3
        canvas.draw.text((px0_3 - 60, py - 8), f"Radar {rid+1}", fill=RADAR_COLORS[rid], font=get_font(12, bold=True))

    # Plot each communication event
    for c in communication:
        t_val = c["t_s"]
        rid = c["radar_id"]
        status = c["final_status"]
        sx = px0_3 + (t_val - xmin3) * sx3
        sy = py1_3 - (rid - ymin3) * sy3
        col = STATUS_COLORS.get(status, ACCENT_GRAY)
        
        if status == "delivered":
            canvas.draw.ellipse([sx - 5, sy - 5, sx + 5, sy + 5], fill=col, outline=TEXT_MAIN, width=1)
        else:
            # Cross marker for failure
            canvas.draw.line([(sx - 5, sy - 5), (sx + 5, sy + 5)], fill=col, width=2)
            canvas.draw.line([(sx - 5, sy + 5), (sx + 5, sy - 5)], fill=col, width=2)

    canvas.save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize Canonical Episode Records v2 for GP-MARL-RadNet")
    parser.add_argument("--dataset", default="data/processed/synthetic_100", help="Path to processed dataset")
    parser.add_argument("--split", default="train", help="Dataset split to visualize")
    parser.add_argument("--episode-index", type=int, default=0, help="Index of episode within split")
    parser.add_argument("--output-dir", default=None, help="Directory to save generated PNG plots")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_eps = [ep for ep in manifest["episodes"] if ep["split"] == args.split]
    if not split_eps:
        raise ValueError(f"No episodes found for split '{args.split}'")

    if args.episode_index >= len(split_eps):
        raise ValueError(f"Episode index {args.episode_index} out of range (max {len(split_eps)-1})")

    target_ep = split_eps[args.episode_index]
    ep_id = target_ep["dataset_episode_id"]
    ep_dir = dataset_dir / args.split / ep_id

    output_dir = Path(args.output_dir) if args.output_dir else dataset_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episode records from: {ep_dir}")
    records = load_episode_records(ep_dir)

    print("Generating Figure 1: Target ground truth trajectory and radar measurements...")
    plot_target_and_radar_measurements(records, output_dir / "01_target_xy_and_radar_measurements.png")

    print("Generating Figure 2: Target ground truth trajectory and centralized tracker estimates...")
    plot_target_and_fused_tracker(records, output_dir / "02_target_xy_and_fused_tracker.png")

    print("Generating Figure 3: Tracking error vs. time...")
    plot_tracking_error_vs_time(records, output_dir / "03_tracking_error_vs_time.png")

    print("Generating Figure 4: Communication status distribution...")
    plot_report_communication_distribution(records, output_dir / "04_report_communication_distribution.png")

    print("\nVisualization generation finished successfully!")
    print(f"All figures saved to: {output_dir}")


if __name__ == "__main__":
    main()
