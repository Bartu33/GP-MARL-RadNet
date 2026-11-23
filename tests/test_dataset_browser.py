"""Dataset browser: catalog, episode previews and replay payloads for generated datasets."""
from pathlib import Path

from src.interfaces.dataset_browser import DATASET_OPTIONS, load_dataset_catalog, load_dataset_episodes
from src.services.class_dataset_service import attach_previews
from src.services.missile_dive_service import plan_missile_dive_episodes
from src.services.simulation_service import load_episode_records, records_to_viewer_payload
from src.services.synthetic_dataset_service import generate_dataset

KEY = "missile_dive_pilot10 (Missile dives and evasive manoeuvres - 10 episodes)"


def test_registry_lists_only_current_datasets():
    dirs = [spec["relative_dir"] for spec in DATASET_OPTIONS.values()]
    assert not any(old in d for d in dirs for old in ("diverse", "canary", "pilot_diverse", "prototype"))


def test_catalog_preview_and_replay(tmp_path):
    root = tmp_path / "repo"
    out = root / DATASET_OPTIONS[KEY]["relative_dir"]
    plan = plan_missile_dive_episodes({"episodes": 2, "master_seed": 4})
    generate_dataset(plan, out, policy_name="central_greedy_confirm")
    attach_previews(out, plan)
    assert KEY in load_dataset_catalog(root)
    episodes = load_dataset_episodes(KEY, root)
    assert len(episodes) == 2
    for ep in episodes:
        assert ep["target_types"] == ["missile/terminal_dive"] and ep["class_labels"] == [2]
        assert ep["display_name"].startswith("EP-") and "missile/terminal_dive" in ep["display_name"]
        payload = records_to_viewer_payload(load_episode_records(Path(ep["full_path"])))
        assert payload["target_legend"] == ["T0 — Missile (terminal dive) [Ground Truth]"]
        assert {t["label"] for f in payload["frames"] for t in f["targets"]} == {2}
