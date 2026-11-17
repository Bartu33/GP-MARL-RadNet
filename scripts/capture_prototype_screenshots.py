"""Capture real Chrome/WebGL screenshots for Step 4D:
1. Baseline V1 episodes (basin_highlands flat dark sinkhole & ridge_multi_obstacle pyramid spike aliasing)
2. All 12 Prototype V2 episodes across the 6 purposeful scenario templates
3. Streamlit UI Replay Browser view showing Prototype V2 dataset
"""
import asyncio
import base64
import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import websockets

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interfaces.dataset_browser import load_dataset_episodes
from src.services.simulation_service import load_episode_records, records_to_viewer_payload
from src.interfaces.viewer3d import export_html

import os
import tempfile

OUT_DIR = ROOT / "reports" / "ui_verification" / "prototype"
OUT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR = Path(os.environ.get("ANTIGRAVITY_ARTIFACT_DIR")) if os.environ.get("ANTIGRAVITY_ARTIFACT_DIR") else None
if ART_DIR:
    ART_DIR.mkdir(parents=True, exist_ok=True)

CHROME_EXE = os.environ.get("CHROME_PATH", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
PROFILE_DIR = Path(tempfile.gettempdir()) / "chrome_cdp_profile_proto"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_SPECS = [
    {
        "name": "baseline_01_basin_highlands_rocky_v1.png",
        "title": "Baseline V1: Basin Highlands (Dark Flat Sinkhole)",
        "path": ROOT / "data/processed/canary_diverse_v1/dataset_single_missile_v1/train/ep-000031-1f12605e3ce8",
    },
    {
        "name": "baseline_02_ridge_multi_obstacle_dry_v1.png",
        "title": "Baseline V1: Ridge Multi-Obstacle (Pyramid Spikes & Kinks)",
        "path": ROOT / "data/processed/diverse_1000_v1/dataset_single_aircraft_v1/train/ep-000021-f9285c999148",
    },
]

req_counter = 0

async def send_cdp(ws, method, params=None):
    global req_counter
    req_counter += 1
    req_id = req_counter
    msg = {"id": req_id, "method": method}
    if params:
        msg["params"] = params
    await ws.send(json.dumps(msg))
    while True:
        raw = await ws.recv()
        data = json.loads(raw)
        if data.get("id") == req_id:
            return data

async def eval_js(ws, expr):
    res = await send_cdp(ws, "Runtime.evaluate", {"expression": expr, "returnByValue": True})
    inner = res.get("result", {})
    if "result" in inner:
        return inner["result"].get("value")
    return inner.get("value")

async def save_screenshot(ws, filename):
    res = await send_cdp(ws, "Page.captureScreenshot", {"format": "png"})
    img_b64 = res["result"]["data"]
    img_bytes = base64.b64decode(img_b64)
    out_path = OUT_DIR / filename
    out_path.write_bytes(img_bytes)
    if ART_DIR:
        art_path = ART_DIR / filename
        art_path.write_bytes(img_bytes)
    print(f"  [SAVED] {filename} ({len(img_bytes)} bytes)")
    return out_path

async def run():
    print("=================================================================")
    print("STEP 4D: CAPTURING REAL CHROME/WEBGL 3D REPLAY SCREENSHOTS")
    print("=================================================================")

    # 1. Export HTMLs for baseline episodes
    baseline_htmls = []
    for b in BASELINE_SPECS:
        print(f"Exporting baseline: {b['title']}...")
        recs = load_episode_records(b["path"])
        payload = records_to_viewer_payload(recs)
        hpath = OUT_DIR / f"{Path(b['name']).stem}.html"
        export_html(payload, hpath)
        baseline_htmls.append((b, hpath))

    # 2. Export HTMLs for 12 prototype episodes
    proto_eps = load_dataset_episodes("Prototype V2 Designs (scenario_design_prototype_v1 - 12 episodes)", ROOT)
    print(f"Found {len(proto_eps)} prototype episodes.")
    proto_htmls = []
    proto_filenames = [
        "proto_01_valley_infiltration_uav_agile.png",
        "proto_02_valley_infiltration_uav_drift.png",
        "proto_03_high_altitude_aircraft_single.png",
        "proto_04_high_altitude_aircraft_triple.png",
        "proto_05_low_altitude_missile_run_single.png",
        "proto_06_low_altitude_missile_run_chokepoint.png",
        "proto_07_coordinated_uav_approach_triple_same.png",
        "proto_08_coordinated_uav_approach_triple_2plus1.png",
        "proto_09_mixed_target_incursion_canyon.png",
        "proto_10_mixed_target_incursion_pass.png",
        "proto_11_terrain_masking_reacquisition_single.png",
        "proto_12_terrain_masking_reacquisition_triple.png",
    ]

    for ep, fname in zip(proto_eps, proto_filenames):
        print(f"Exporting prototype: {ep['display_name']} -> {fname}...")
        recs = load_episode_records(ep["full_path"])
        payload = records_to_viewer_payload(recs)
        hpath = OUT_DIR / f"{Path(fname).stem}.html"
        export_html(payload, hpath)
        proto_htmls.append((ep, fname, hpath))

    # 3. Launch Chrome with CDP
    port = 9232
    cmd = [
        CHROME_EXE,
        "--headless=new",
        f"--remote-debugging-port={port}",
        f"--user-data-dir={PROFILE_DIR}",
        "--allow-file-access-from-files",
        "--window-size=1600,1050",
        "--disable-gpu",
        "--no-first-run",
        "about:blank",
    ]
    proc = subprocess.Popen(cmd)
    try:
        await asyncio.sleep(2)
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json") as resp:
            targets = json.loads(resp.read().decode())
        ws_url = next(t for t in targets if t.get("type") == "page")["webSocketDebuggerUrl"]
        print(f"Connected to Chrome CDP WebSocket: {ws_url}")

        async with websockets.connect(ws_url, max_size=100_000_000) as ws:
            await send_cdp(ws, "Page.enable")
            await send_cdp(ws, "Runtime.enable")

            # Capture Baselines
            print("\n--- Capturing Baselines ---")
            for b, hpath in baseline_htmls:
                print(f"Loading {b['name']} ({hpath.name})...")
                await send_cdp(ws, "Page.navigate", {"url": hpath.as_uri()})
                for _ in range(25):
                    await asyncio.sleep(0.4)
                    ld = await eval_js(ws, "document.getElementById('loading') ? document.getElementById('loading').style.display : 'flex'")
                    if ld == 'none':
                        break
                await eval_js(ws, "setView('oblique'); controls.autoRotate = false;")
                await asyncio.sleep(1.0)
                await save_screenshot(ws, b["name"])

            # Capture 12 Prototypes
            print("\n--- Capturing 12 Prototype V2 Episodes ---")
            for ep, fname, hpath in proto_htmls:
                print(f"Loading {fname} ({ep['display_name']})...")
                await send_cdp(ws, "Page.navigate", {"url": hpath.as_uri()})
                for _ in range(25):
                    await asyncio.sleep(0.4)
                    ld = await eval_js(ws, "document.getElementById('loading') ? document.getElementById('loading').style.display : 'flex'")
                    if ld == 'none':
                        break
                await eval_js(ws, "setView('oblique'); controls.autoRotate = false;")
                await asyncio.sleep(1.0)
                await save_screenshot(ws, fname)

            # Capture Streamlit UI with Prototype Selected
            print("\n--- Capturing Streamlit UI with Prototype Dataset ---")
            await send_cdp(ws, "Page.navigate", {"url": "http://localhost:8501"})
            await asyncio.sleep(5)
            # Switch to Dataset Replay
            await eval_js(ws, """(() => {
                const labels = Array.from(document.querySelectorAll('label'));
                const l = labels.find(x => x.innerText && x.innerText.trim() === 'Dataset Replay');
                if (l) l.click();
            })()""")
            await asyncio.sleep(3)
            await save_screenshot(ws, "proto_ui_01_dataset_browser_selection.png")

    finally:
        proc.terminate()
        proc.wait()

    print("\n=================================================================")
    print("ALL 15 SCREENSHOTS CAPTURED SUCCESSFULLY (2 Baselines + 12 Prototypes + 1 UI)")
    print("=================================================================")

if __name__ == "__main__":
    asyncio.run(run())
