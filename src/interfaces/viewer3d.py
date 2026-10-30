"""Embed the three.js 3-D viewer (assets/viewer3d.html) with a recorded episode."""
from __future__ import annotations

import json
from pathlib import Path

ASSET = Path(__file__).with_name("assets") / "viewer3d.html"


def build_html(payload: dict) -> str:
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return ASSET.read_text(encoding="utf-8").replace("__PAYLOAD__", data)


def export_html(payload: dict, path: str | Path) -> Path:
    """Write a stand-alone HTML replay that opens in any browser."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_html(payload), encoding="utf-8")
    return path


def render(payload: dict, height: int = 720):
    import streamlit as st
    html = build_html(payload)
    if hasattr(st, "iframe"):                 # Streamlit >= 1.5x
        st.iframe(html, height=height)
    else:                                     # older Streamlit
        import streamlit.components.v1 as components
        components.html(html, height=height, scrolling=False)
