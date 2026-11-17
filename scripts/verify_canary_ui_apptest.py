"""Verify Canary dataset loading, reactive filtering, 3D replay and inspector in Streamlit AppTest."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from streamlit.testing.v1 import AppTest

print("\n=======================================================")
print("[Test 1] Verifying Canary Dataset in Dataset Catalog...")
print("=======================================================")
at1 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
assert not at1.exception, f"App startup exception: {at1.exception}"
at1.radio(key="scen_op_mode").set_value("Dataset Replay").run()
assert not at1.exception

ds_sel = at1.selectbox(key="rep_dataset_sel")
assert "All Canary Groups (canary_diverse_v1 - 70 episodes)" in ds_sel.options
assert "dataset_single_aircraft_v1 (Canary v1)" in ds_sel.options
assert "dataset_single_missile_v1 (Canary v1)" in ds_sel.options
assert "dataset_triple_all_different_v1 (Canary v1)" in ds_sel.options
print("Canary dataset options verified in UI selectbox!")

print("\n=======================================================")
print("[Test 2] Verifying 70 Episodes and Reactive Filtering...")
print("=======================================================")
at2 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at2.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at2.selectbox(key="rep_dataset_sel").set_value("All Canary Groups (canary_diverse_v1 - 70 episodes)").run()
assert not at2.exception

canary_sel_key = "rep_ep_sel_All Canary Groups (canary_diverse_v1 - 70 episodes)"
canary_ep_sel = at2.selectbox(key=canary_sel_key)
print(f"Total Canary episodes loaded in UI: {len(canary_ep_sel.options)}")
assert len(canary_ep_sel.options) == 70

# Test Difficulty Filtering: Easy
diff_sel = at2.selectbox(key="rep_f_diff")
diff_sel.set_value("easy").run()
filtered_easy = at2.selectbox(key=canary_sel_key)
print(f"Canary episodes matching Difficulty = easy: {len(filtered_easy.options)}")
assert len(filtered_easy.options) == 21

# Test Difficulty Filtering: Hard
diff_sel.set_value("hard").run()
filtered_hard = at2.selectbox(key=canary_sel_key)
print(f"Canary episodes matching Difficulty = hard: {len(filtered_hard.options)}")
assert len(filtered_hard.options) == 21

# Reset difficulty and test Target Count = 3
diff_sel.set_value("All").run()
tcount_sel = at2.selectbox(key="rep_f_tcount")
tcount_sel.set_value("3").run()
filtered_triple = at2.selectbox(key=canary_sel_key)
print(f"Canary episodes matching Target Count = 3: {len(filtered_triple.options)}")
assert len(filtered_triple.options) == 30 # 3 groups * 10 eps

print("\n=======================================================")
print("[Test 3] Loading Canary Single Aircraft Replay...")
print("=======================================================")
at3 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at3.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at3.selectbox(key="rep_dataset_sel").set_value("dataset_single_aircraft_v1 (Canary v1)").run()
at3.button(key="rep_load_btn").click().run()
assert not at3.exception
assert at3.session_state["scen_mode_active"] == "replay"
ep_name = at3.session_state["scen_ep_name"]
payload = at3.session_state["scen_payload"]
assert "EP-0001" in ep_name
assert len(payload["frames"]) == 61 # 30s at dt=0.5s is 61 frames
assert payload["terrain"]["terrain_family"] in ["valley_linear_canyon", "basin_highlands", "mountain_chokepoint", "valley_sinuous_pass", "ridge_multi_obstacle"]
print("Canary Single Aircraft episode replayed successfully:")
print("  display_name:", ep_name)
print("  frames:", len(payload["frames"]))
print("  terrain:", payload["terrain"]["terrain_family"])

print("\n=======================================================")
print("[Test 4] Loading Canary Triple All Different Replay...")
print("=======================================================")
at4 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at4.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at4.selectbox(key="rep_dataset_sel").set_value("dataset_triple_all_different_v1 (Canary v1)").run()
at4.button(key="rep_load_btn").click().run()
assert not at4.exception
assert at4.session_state["scen_mode_active"] == "replay"
ep_name_triple = at4.session_state["scen_ep_name"]
payload_triple = at4.session_state["scen_payload"]
assert len(payload_triple["frames"]) == 61
targets_f0 = payload_triple["frames"][0]["targets"]
print("Canary Triple All Different episode replayed successfully:")
print("  display_name:", ep_name_triple)
print("  target count in frame 0:", len(targets_f0))
assert len(targets_f0) == 3

print("\n=======================================================")
print("ALL CANARY UI APPTEST CHECKS PASSED WITH 100% SUCCESS!")
print("=======================================================")
