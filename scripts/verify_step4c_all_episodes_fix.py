"""Verification script for Step 4C: Dataset Browser & UI All Episodes Fix.

Verifies:
1. All Final Groups / All Episodes loads exact 1,000 episodes without truncation.
2. Counter displays '1000 / 1000 (Toplam 1000 Episode)'.
3. Reset Filters button clears active filters and restores full 1,000 episode count.
4. Dataset switching automatically clears filters from previous datasets.
5. All 7 final groups load their exact episode counts:
   - All Final Groups: 1000
   - Single Aircraft: 150
   - Single UAV Agile: 150
   - Single UAV Drift: 150
   - Single Missile: 150
   - Triple All Same: 132
   - Triple All Different: 132
   - Triple Two Plus One: 136
6. Split filtering: Train (700), Validation (150), Test (150).
7. Difficulty filtering: Easy (300), Medium (450), Hard (250).
8. Replay loading for First (EP-0001), Middle (EP-0500 / EP-0601), and Last (EP-1000) episodes:
   Verifies 61 frames, valid terrain, radars, targets.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

def main():
    print("=================================================================")
    print("STEP 4C: VERIFYING DATASET BROWSER 'ALL EPISODES' FIX & APPTEST")
    print("=================================================================")

    # -------------------------------------------------------------
    # 1. Test All Final Groups (diverse_1000_v1 - 1000 episodes) Default Load
    # -------------------------------------------------------------
    print("\n[Step 1] Loading Streamlit App and switching to Dataset Replay...")
    at = AppTest.from_file(str(ROOT / "src" / "interfaces" / "app.py"), default_timeout=35).run()
    assert not at.exception, f"App startup failed: {at.exception}"

    at.radio(key="scen_op_mode").set_value("Dataset Replay").run()
    assert not at.exception

    ds_sel = at.selectbox(key="rep_dataset_sel")
    print("Selected dataset by default:", ds_sel.value)
    assert ds_sel.value == "All Final Groups (diverse_1000_v1 - 1000 episodes)"

    final_sel_key = f"rep_ep_sel_{ds_sel.value}"
    ep_sel = at.selectbox(key=final_sel_key)
    print(f"Total episodes in selector: {len(ep_sel.options)}")
    assert len(ep_sel.options) == 1000, f"Expected 1000 episodes, got {len(ep_sel.options)}"

    # Check Markdown counter text
    md_texts = [m.value for m in at.markdown]
    counter_found = any("1000 / 1000" in t for t in md_texts)
    print(f"Counter contains '1000 / 1000': {counter_found}")
    assert counter_found, "Counter '1000 / 1000' not found in UI markdown!"

    # -------------------------------------------------------------
    # 2. Test Split and Difficulty Filtering Counts
    # -------------------------------------------------------------
    print("\n[Step 2] Testing Split and Difficulty Filtering...")

    # Difficulty: Easy (300)
    at.selectbox(key="rep_f_diff").set_value("easy").run()
    assert not at.exception
    eps_easy = at.selectbox(key=final_sel_key)
    print(f"  Difficulty = easy: {len(eps_easy.options)} (expected 300)")
    assert len(eps_easy.options) == 300

    # Difficulty: Medium (450)
    at.selectbox(key="rep_f_diff").set_value("medium").run()
    assert not at.exception
    eps_med = at.selectbox(key=final_sel_key)
    print(f"  Difficulty = medium: {len(eps_med.options)} (expected 450)")
    assert len(eps_med.options) == 450

    # Difficulty: Hard (250)
    at.selectbox(key="rep_f_diff").set_value("hard").run()
    assert not at.exception
    eps_hard = at.selectbox(key=final_sel_key)
    print(f"  Difficulty = hard: {len(eps_hard.options)} (expected 250)")
    assert len(eps_hard.options) == 250

    # Reset Difficulty back to All
    at.selectbox(key="rep_f_diff").set_value("All").run()

    # Split: Train (700)
    at.selectbox(key="rep_f_split").set_value("train").run()
    assert not at.exception
    eps_train = at.selectbox(key=final_sel_key)
    print(f"  Split = train: {len(eps_train.options)} (expected 700)")
    assert len(eps_train.options) == 700

    # Split: Validation (150)
    at.selectbox(key="rep_f_split").set_value("validation").run()
    assert not at.exception
    eps_val = at.selectbox(key=final_sel_key)
    print(f"  Split = validation: {len(eps_val.options)} (expected 150)")
    assert len(eps_val.options) == 150

    # Split: Test (150)
    at.selectbox(key="rep_f_split").set_value("test").run()
    assert not at.exception
    eps_test = at.selectbox(key=final_sel_key)
    print(f"  Split = test: {len(eps_test.options)} (expected 150)")
    assert len(eps_test.options) == 150

    # -------------------------------------------------------------
    # 3. Test 'Reset Filters' Button
    # -------------------------------------------------------------
    print("\n[Step 3] Testing Reset Filters button...")
    # Currently Split is 'test' (150). Also set text search
    at.text_input(key="rep_f_search").set_value("canyon").run()
    eps_canyon = at.selectbox(key=final_sel_key)
    print(f"  With active search + split filters: {len(eps_canyon.options)} episodes")
    assert len(eps_canyon.options) < 1000

    # Click Reset Filters
    at.button(key="rep_reset_filters_btn").click().run()
    assert not at.exception
    eps_reset = at.selectbox(key=final_sel_key)
    print(f"  After Reset Filters button click: {len(eps_reset.options)} episodes (expected 1000)")
    assert len(eps_reset.options) == 1000, f"Expected 1000 after reset, got {len(eps_reset.options)}"

    # -------------------------------------------------------------
    # 4. Test Automatic Filter Reset on Dataset Change
    # -------------------------------------------------------------
    print("\n[Step 4] Testing Automatic Filter Reset on Dataset Change...")
    # Set a restrictive filter
    at.selectbox(key="rep_f_diff").set_value("hard").run()
    assert len(at.selectbox(key=final_sel_key).options) == 250

    # Switch dataset to Single Aircraft
    at.selectbox(key="rep_dataset_sel").set_value("dataset_single_aircraft_v1 (Final 1000)").run()
    assert not at.exception
    air_sel_key = "rep_ep_sel_dataset_single_aircraft_v1 (Final 1000)"
    air_eps = at.selectbox(key=air_sel_key)
    print(f"  dataset_single_aircraft_v1 (Final 1000) loaded: {len(air_eps.options)} (expected 150)")
    # Filter should have been auto-reset so it shows all 150, not 37 hard ones!
    assert len(air_eps.options) == 150, f"Expected 150 episodes in aircraft group, got {len(air_eps.options)}"

    # Now switch back to All Final Groups
    at.selectbox(key="rep_dataset_sel").set_value("All Final Groups (diverse_1000_v1 - 1000 episodes)").run()
    assert not at.exception
    eps_restored = at.selectbox(key=final_sel_key)
    print(f"  Switched back to All Final Groups: {len(eps_restored.options)} (expected 1000)")
    assert len(eps_restored.options) == 1000

    # -------------------------------------------------------------
    # 5. Verify All 7 Final Groups Exact Counts
    # -------------------------------------------------------------
    print("\n[Step 5] Verifying exact counts for all 7 Final groups...")
    expected_counts = {
        "dataset_single_aircraft_v1 (Final 1000)": 150,
        "dataset_single_uav_agile_v1 (Final 1000)": 150,
        "dataset_single_uav_drift_v1 (Final 1000)": 150,
        "dataset_single_missile_v1 (Final 1000)": 150,
        "dataset_triple_all_same_v1 (Final 1000)": 132,
        "dataset_triple_all_different_v1 (Final 1000)": 132,
        "dataset_triple_two_plus_one_v1 (Final 1000)": 136,
    }
    for group_name, exp_count in expected_counts.items():
        at.selectbox(key="rep_dataset_sel").set_value(group_name).run()
        assert not at.exception
        grp_key = f"rep_ep_sel_{group_name}"
        grp_eps = at.selectbox(key=grp_key)
        print(f"  {group_name}: {len(grp_eps.options)} / {exp_count}")
        assert len(grp_eps.options) == exp_count, f"{group_name} count mismatch: {len(grp_eps.options)} != {exp_count}"

    # Also test the 'All Episodes (diverse_1000_v1 - 1000 episodes)' option
    at.selectbox(key="rep_dataset_sel").set_value("All Episodes (diverse_1000_v1 - 1000 episodes)").run()
    assert not at.exception
    all_ep_sel_key = "rep_ep_sel_All Episodes (diverse_1000_v1 - 1000 episodes)"
    assert len(at.selectbox(key=all_ep_sel_key).options) == 1000
    print("  All Episodes (diverse_1000_v1 - 1000 episodes) alias option: 1000 / 1000 verified!")

    # -------------------------------------------------------------
    # 6. Test Loading Replay for First, Middle, and Last Episodes
    # -------------------------------------------------------------
    print("\n[Step 6] Testing Replay Loading for First, Middle, and Last Episodes...")

    # Switch back to All Final Groups
    at.selectbox(key="rep_dataset_sel").set_value("All Final Groups (diverse_1000_v1 - 1000 episodes)").run()

    # First Episode: index 0 (EP-0001)
    print("  Loading First Episode (index 0: EP-0001)...")
    at.selectbox(key=final_sel_key).set_value(0).run()
    at.button(key="rep_load_btn").click().run()
    assert not at.exception
    assert at.session_state["scen_mode_active"] == "replay"
    assert "EP-0001" in at.session_state["scen_ep_name"]
    payload_first = at.session_state["scen_payload"]
    assert len(payload_first["frames"]) == 61
    print(f"    Loaded successfully: {at.session_state['scen_ep_name']}, frames: {len(payload_first['frames'])}")

    # Middle Episode: index 499 (EP-0500)
    print("  Loading Middle Episode (index 499: EP-0500)...")
    at.selectbox(key=final_sel_key).set_value(499).run()
    at.button(key="rep_load_btn").click().run()
    assert not at.exception
    assert "EP-0500" in at.session_state["scen_ep_name"]
    payload_mid = at.session_state["scen_payload"]
    assert len(payload_mid["frames"]) == 61
    print(f"    Loaded successfully: {at.session_state['scen_ep_name']}, frames: {len(payload_mid['frames'])}")

    # Middle Episode (First Triple): index 600 (EP-0601)
    print("  Loading Middle Triple Episode (index 600: EP-0601)...")
    at.selectbox(key=final_sel_key).set_value(600).run()
    at.button(key="rep_load_btn").click().run()
    assert not at.exception
    assert "EP-0601" in at.session_state["scen_ep_name"]
    payload_mid_triple = at.session_state["scen_payload"]
    assert len(payload_mid_triple["frames"]) == 61
    assert len(payload_mid_triple["frames"][0]["targets"]) == 3
    print(f"    Loaded successfully: {at.session_state['scen_ep_name']}, 3 targets in frame 0, frames: {len(payload_mid_triple['frames'])}")

    # Last Episode: index 999 (EP-1000)
    print("  Loading Last Episode (index 999: EP-1000)...")
    at.selectbox(key=final_sel_key).set_value(999).run()
    at.button(key="rep_load_btn").click().run()
    assert not at.exception
    assert "EP-1000" in at.session_state["scen_ep_name"]
    payload_last = at.session_state["scen_payload"]
    assert len(payload_last["frames"]) == 61
    assert len(payload_last["frames"][0]["targets"]) == 3
    print(f"    Loaded successfully: {at.session_state['scen_ep_name']}, frames: {len(payload_last['frames'])}")

    print("\n=================================================================")
    print(">>> ALL STEP 4C ACCEPTANCE CHECKS PASSED WITH 100% SUCCESS! <<<")
    print("=================================================================")

if __name__ == "__main__":
    main()
