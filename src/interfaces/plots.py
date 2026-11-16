"""Plotly figures shared by the dashboard pages."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.core.targets import CLASS_SHORT

DARK = dict(template="plotly_dark", paper_bgcolor="#0E1417", plot_bgcolor="#151D21",
            margin=dict(l=40, r=20, t=40, b=40), font=dict(size=12))
CLASS_COLORS = ["#ff9f43", "#d980fa", "#7bed9f", "#feca57"]
MODE_COLORS = {0: "#8DA2AC", 1: "#4FD1C5", 2: "#7FB3FF", 3: "#ff5c5c", 4: "#C39BFF"}


def map_2d(payload: dict, frame_index: int, trail: int = 60) -> go.Figure:
    """Top-down view of one recorded frame (fallback when WebGL is unavailable)."""
    fr = payload["frames"][frame_index]
    fig = go.Figure()
    for r, rs in zip(payload["radars"], fr["radars"]):
        x0, y0 = r["pos"][:2]
        az = np.linspace(r["heading"] - r["fov"], r["heading"] + r["fov"], 50)
        fig.add_trace(go.Scatter(x=[x0, *(x0 + r["rmax"] * np.cos(az)), x0],
                                 y=[y0, *(y0 + r["rmax"] * np.sin(az)), y0], mode="lines",
                                 line=dict(color="#2E6F69", width=1, dash="dot"), hoverinfo="skip", showlegend=False))
        for k, (a, _) in enumerate(rs["beams"]):
            w = r["heading"] - a
            # interleaved look: the last beam is the track dwell, the others are search cells
            track_beam = rs["mode"] == 2 or (rs["mode"] == 4 and k == len(rs["beams"]) - 1)
            L = (rs.get("len") or r["rmax"]) if track_beam else r["rmax"]
            col = MODE_COLORS[1] if rs["mode"] == 4 and not track_beam else MODE_COLORS[rs["mode"]]
            fig.add_trace(go.Scatter(x=[x0, x0 + L * np.cos(w)], y=[y0, y0 + L * np.sin(w)], mode="lines",
                                     line=dict(color=col, width=1), opacity=0.5,
                                     hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=[x0], y=[y0], mode="markers+text", text=[r["name"]], textposition="top center",
                                 marker=dict(symbol="square", size=12, color=MODE_COLORS[rs["mode"]]), showlegend=False))
    frames = payload["frames"][max(0, frame_index - trail): frame_index + 1]
    for t in fr["targets"]:
        xs = [q["p"][0] for f in frames for q in f["targets"] if q["id"] == t["id"]]
        ys = [q["p"][1] for f in frames for q in f["targets"] if q["id"] == t["id"]]
        c = CLASS_COLORS[t["label"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=c, width=1), opacity=0.5, showlegend=False))
        disp_name = t.get("display_name") or payload.get("target_labels", {}).get(t["id"]) or f"truth {CLASS_SHORT[t['label']]}"
        fig.add_trace(go.Scatter(x=[t["p"][0]], y=[t["p"][1]], mode="markers", marker=dict(color=c, size=9),
                                 name=disp_name))
    for tr in fr["tracks"]:
        if not tr["conf"]:
            continue
        th = np.linspace(0, 2 * np.pi, 40)
        fig.add_trace(go.Scatter(x=tr["p"][0] + 3 * tr["sig"] * np.cos(th), y=tr["p"][1] + 3 * tr["sig"] * np.sin(th),
                                 mode="lines", line=dict(color="#E6EDEF", width=1, dash="dot"), showlegend=False))
        label = f"T{tr['id']}" + (f" {CLASS_SHORT[tr['cls']]} {tr['pc']:.0%}" if tr["cls"] >= 0 else "")
        fig.add_trace(go.Scatter(x=[tr["p"][0]], y=[tr["p"][1]], mode="markers+text", text=[label],
                                 textposition="bottom center", marker=dict(symbol="square-open", size=10, color="#E6EDEF"),
                                 showlegend=False))
    if fr["dets"]:
        d = np.array(fr["dets"])
        fig.add_trace(go.Scatter(x=d[:, 0], y=d[:, 1], mode="markers", name="detections",
                                 marker=dict(size=5, color=np.where(d[:, 3] > 0, "#ff5c5c", "#9fffe0"))))

    # Dynamic auto-zoom: compute episode-wide bounding box enclosing radars, targets, and tracks
    all_x = [float(r["pos"][0]) for r in payload.get("radars", [])]
    all_y = [float(r["pos"][1]) for r in payload.get("radars", [])]
    for f in payload.get("frames", []):
        for tg in f.get("targets", []):
            all_x.append(float(tg["p"][0]))
            all_y.append(float(tg["p"][1]))
        for tk in f.get("tracks", []):
            if tk.get("conf"):
                all_x.append(float(tk["p"][0]))
                all_y.append(float(tk["p"][1]))

    if all_x and all_y:
        xmin, xmax = min(all_x), max(all_x)
        ymin, ymax = min(all_y), max(all_y)
        pad = max(1500.0, 0.12 * max(xmax - xmin, ymax - ymin))
        xmin -= pad
        xmax += pad
        ymin -= pad
        ymax += pad
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        half_span = max(xmax - xmin, ymax - ymin) / 2.0
        x_range = [cx - half_span, cx + half_span]
        y_range = [cy - half_span, cy + half_span]
    else:
        x_range = [-20000.0, 20000.0]
        y_range = [-20000.0, 20000.0]

    fig.update_layout(**DARK, height=560, title=f"t = {fr['t']:.1f} s",
                      xaxis=dict(range=x_range, title="east (m)"),
                      yaxis=dict(range=y_range, scaleanchor="x", scaleratio=1, title="north (m)"),
                      legend=dict(orientation="h"))
    return fig


def time_series(log: dict, keys, title: str, ytitle: str = "") -> go.Figure:
    fig = go.Figure()
    for k in keys:
        fig.add_trace(go.Scatter(x=log["t"], y=log[k], mode="lines", name=k))
    fig.update_layout(**DARK, height=280, title=title, xaxis_title="time (s)", yaxis_title=ytitle)
    return fig


def line(x, ys: dict, title: str, xtitle: str, ytitle: str, log_x=False, height=330) -> go.Figure:
    fig = go.Figure()
    for name, y in ys.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
    fig.update_layout(**DARK, height=height, title=title, xaxis_title=xtitle, yaxis_title=ytitle)
    if log_x:
        fig.update_xaxes(type="log")
    return fig


def polar_pattern(angles_deg, values: dict, title: str) -> go.Figure:
    fig = go.Figure()
    for name, v in values.items():
        fig.add_trace(go.Scatterpolar(theta=angles_deg, r=v, mode="lines", name=name))
    fig.update_layout(**DARK, height=380, title=title, polar=dict(bgcolor="#151D21"))
    return fig


def confusion(cm, labels, title="Confusion matrix (rows = true)") -> go.Figure:
    cm = np.asarray(cm, float)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, colorscale="Teal", zmin=0, zmax=1,
                               text=cm.astype(int), texttemplate="%{text}", showscale=False))
    fig.update_layout(**DARK, height=360, title=title, yaxis=dict(autorange="reversed"))
    return fig


def trajectories_3d(trajs, title="Sample trajectories") -> go.Figure:
    """One 3-D panel per class, each centred on its own start point so that
    slow and fast classes are equally readable."""
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=1, cols=len(trajs), specs=[[{"type": "scene"}] * len(trajs)],
                        subplot_titles=[CLASS_SHORT[l] for l, _ in trajs])
    for k, (label, P) in enumerate(trajs, start=1):
        Q = P - P[0] * np.array([1, 1, 0])
        fig.add_trace(go.Scatter3d(x=Q[:, 0], y=Q[:, 1], z=Q[:, 2], mode="lines",
                                   line=dict(color=CLASS_COLORS[label], width=5), showlegend=False), row=1, col=k)
        fig.update_layout(**{f"scene{k if k > 1 else ''}": dict(
            xaxis_title="E (m)", yaxis_title="N (m)", zaxis_title="alt (m)", aspectmode="cube")})
    fig.update_layout(**DARK, height=420, title=title)
    return fig


def plot_action_distribution(radars_summary: dict, title: str = "Radar Action Allocation (Search vs Track vs Idle)") -> go.Figure:
    fig = go.Figure()
    r_names = list(radars_summary.keys())
    search_vals = [radars_summary[r].get("search_pct", radars_summary[r].get("search_ratio_pct", 0.0)) for r in r_names]
    track_vals = [radars_summary[r].get("track_pct", radars_summary[r].get("track_ratio_pct", 0.0)) for r in r_names]
    idle_vals = [radars_summary[r].get("idle_pct", radars_summary[r].get("idle_ratio_pct", 0.0)) for r in r_names]

    fig.add_trace(go.Bar(name="Search", x=r_names, y=search_vals, marker_color="#38BDF8"))
    fig.add_trace(go.Bar(name="Track", x=r_names, y=track_vals, marker_color="#A855F7"))
    fig.add_trace(go.Bar(name="Idle", x=r_names, y=idle_vals, marker_color="#F59E0B"))

    fig.update_layout(**DARK, barmode="stack", height=320, title=title,
                      xaxis_title="Radar", yaxis_title="Percentage of Steps (%)")
    return fig


def plot_detection_timeline(timeline_data: dict, title: str = "Detection Activity & Packet Delivery Over Time") -> go.Figure:
    fig = go.Figure()
    centers = timeline_data.get("bin_centers_s", [])
    gen = timeline_data.get("generated_counts", [])
    deliv = timeline_data.get("delivered_counts", [])
    lost = [max(0, g - d) for g, d in zip(gen, deliv)]

    fig.add_trace(go.Bar(name="Delivered to Fusion", x=centers, y=deliv, marker_color="#10B981"))
    if any(l > 0 for l in lost):
        fig.add_trace(go.Bar(name="Packet Lost", x=centers, y=lost, marker_color="#EF4444"))

    fig.update_layout(**DARK, barmode="stack", height=320, title=title,
                      xaxis_title="Time (s)", yaxis_title="Detections per Second")
    return fig


def plot_uncertainty_vs_aoi(aoi_data: dict, title: str = "Track Position Uncertainty vs Age of Information (AoI)") -> go.Figure:
    fig = go.Figure()
    aoi_s = sorted([float(k) for k in aoi_data.keys() if float(k) <= 10.0])
    stds = [float(np.mean(aoi_data[k])) if isinstance(aoi_data[k], (list, np.ndarray)) else float(aoi_data[k]) for k in aoi_s]

    fig.add_trace(go.Bar(x=aoi_s, y=stds, marker_color="#F59E0B", name="Mean Position Std"))
    fig.add_hline(y=150.0, line_dash="dash", line_color="#EF4444",
                  annotation_text="Greedy Revisit Due Threshold (150 m)", annotation_position="top left")

    fig.update_layout(**DARK, height=320, title=title,
                      xaxis_title="Age of Information (s)", yaxis_title="Position Uncertainty (m)")
    return fig
