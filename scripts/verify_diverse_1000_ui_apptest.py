"""Verify Diverse 1000 dataset loading, reactive filtering, 3D replay and inspector in Streamlit AppTest."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
from streamlit.testing.v1 import AppTest

print("\n=======================================================")
print("[Test 1] Verifying Final 1000 Dataset in Dataset Catalog...")
print("=======================================================")
at1 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
assert not at1.exception, f"App startup exception: {at1.exception}"
at1.radio(key="scen_op_mode").set_value("Dataset Replay").run()
assert not at1.exception

ds_sel = at1.selectbox(key="rep_dataset_sel")
assert "All Final Groups (diverse_1000_v1 - 1000 episodes)" in ds_sel.options
assert "dataset_single_aircraft_v1 (Final 1000)" in ds_sel.options
assert "dataset_single_uav_agile_v1 (Final 1000)" in ds_sel.options
assert "dataset_single_uav_drift_v1 (Final 1000)" in ds_sel.options
assert "dataset_single_missile_v1 (Final 1000)" in ds_sel.options
assert "dataset_triple_all_same_v1 (Final 1000)" in ds_sel.options
assert "dataset_triple_all_different_v1 (Final 1000)" in ds_sel.options
assert "dataset_triple_two_plus_one_v1 (Final 1000)" in ds_sel.options
print("Final 1000 dataset options verified in UI selectbox!")

print("\n=======================================================")
print("[Test 2] Verifying 1000 Episodes and Multi-Criteria Filtering...")
print("=======================================================")
at2 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at2.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at2.selectbox(key="rep_dataset_sel").set_value("All Final Groups (diverse_1000_v1 - 1000 episodes)").run()
assert not at2.exception

final_sel_key = "rep_ep_sel_All Final Groups (diverse_1000_v1 - 1000 episodes)"
ep_sel = at2.selectbox(key=final_sel_key)
print(f"Total 1000-episode dataset loaded in UI: {len(ep_sel.options)}")
assert len(ep_sel.options) == 1000

# 1. Test Difficulty Filtering: Easy
diff_sel = at2.selectbox(key="rep_f_diff")
diff_sel.set_value("easy").run()
filtered_easy = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Difficulty = easy: {len(filtered_easy.options)}")
assert len(filtered_easy.options) == 300

# 2. Test Difficulty Filtering: Medium
diff_sel.set_value("medium").run()
filtered_med = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Difficulty = medium: {len(filtered_med.options)}")
assert len(filtered_med.options) == 450

# 3. Test Difficulty Filtering: Hard
diff_sel.set_value("hard").run()
filtered_hard = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Difficulty = hard: {len(filtered_hard.options)}")
assert len(filtered_hard.options) == 250

# Reset difficulty
diff_sel.set_value("All").run()

# 4. Test Split Filtering: Train
split_sel = at2.selectbox(key="rep_f_split")
split_sel.set_value("train").run()
filtered_train = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Split = train: {len(filtered_train.options)}")
assert len(filtered_train.options) == 700

# 5. Test Split Filtering: Validation
split_sel.set_value("validation").run()
filtered_val = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Split = validation: {len(filtered_val.options)}")
assert len(filtered_val.options) == 150

# 6. Test Split Filtering: Test
split_sel.set_value("test").run()
filtered_test = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Split = test: {len(filtered_test.options)}")
assert len(filtered_test.options) == 150

# Reset split
split_sel.set_value("All").run()

# 7. Test Target Count: 1
tcount_sel = at2.selectbox(key="rep_f_tcount")
tcount_sel.set_value("1").run()
filtered_single = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Target Count = 1: {len(filtered_single.options)}")
assert len(filtered_single.options) == 600

# 8. Test Target Count: 3
tcount_sel.set_value("3").run()
filtered_triple = at2.selectbox(key=final_sel_key)
print(f"Episodes matching Target Count = 3: {len(filtered_triple.options)}")
assert len(filtered_triple.options) == 400

print("\n=======================================================")
print("[Test 3] Loading Final 1000 Single Aircraft Replay...")
print("=======================================================")
at3 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at3.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at3.selectbox(key="rep_dataset_sel").set_value("dataset_single_aircraft_v1 (Final 1000)").run()
at3.button(key="rep_load_btn").click().run()
assert not at3.exception
assert at3.session_state["scen_mode_active"] == "replay"
ep_name = at3.session_state["scen_ep_name"]
payload = at3.session_state["scen_payload"]
assert "EP-0001" in ep_name
assert len(payload["frames"]) == 61 # 30s at dt=0.5s is 61 frames
assert payload["terrain"]["terrain_family"] in [
    "valley_linear_canyon", "basin_highlands", "mountain_chokepoint", "valley_sinuous_pass", "ridge_multi_obstacle"
]
print("Final 1000 Single Aircraft episode replayed successfully:")
print("  display_name:", ep_name)
print("  frames:", len(payload["frames"]))
print("  terrain:", payload["terrain"]["terrain_family"])

print("\n=======================================================")
print("[Test 4] Loading Final 1000 Triple All Different Replay...")
print("=======================================================")
at4 = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
at4.radio(key="scen_op_mode").set_value("Dataset Replay").run()
at4.selectbox(key="rep_dataset_sel").set_value("dataset_triple_all_different_v1 (Final 1000)").run()
at4.button(key="rep_load_btn").click().run()
assert not at4.exception
assert at4.session_state["scen_mode_active"] == "replay"
ep_name_triple = at4.session_state["scen_ep_name"]
payload_triple = at4.session_state["scen_payload"]
assert len(payload_triple["frames"]) == 61
targets_f0 = payload_triple["frames"][0]["targets"]
print("Final 1000 Triple All Different episode replayed successfully:")
print("  display_name:", ep_name_triple)
print("  target count in frame 0:", len(targets_f0))
assert len(targets_f0) == 3

print("\n=======================================================")
print("ALL FINAL 1000 UI APPTEST CHECKS PASSED WITH 100% SUCCESS!")
print("=======================================================")
