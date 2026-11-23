"""Plan, generate and audit a UAV / aircraft / missile dataset.

The recipe's ``kind`` selects the planner: ``class_targets`` (mixed classes,
default) or ``missile_dive`` (missile terminal dives, missile_dive_service).

    python scripts/produce_class_targets_dataset.py \
        --recipe configs/datasets/class_targets_pilot5.yaml \
        --output data/processed/class_targets_pilot5 \
        --report reports/class_targets_pilot5

Generation resumes from the manifest when the output directory already holds
the same plan.  The report directory receives plan.json, audit.json and a
Turkish audit.md comparing realised values with the variant bands.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.services.class_dataset_service import (  # noqa: E402
    attach_previews, audit_class_dataset, audit_markdown, plan_class_episodes,
)
from src.services.missile_dive_service import plan_missile_dive_episodes  # noqa: E402
from src.services.scenario_planner import export_plan, load_recipe  # noqa: E402

PLANNERS = {"class_targets": plan_class_episodes, "missile_dive": plan_missile_dive_episodes}
from src.services.synthetic_dataset_service import generate_dataset  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--recipe", default="configs/datasets/class_targets_pilot5.yaml")
    parser.add_argument("--output", default="data/processed/class_targets_pilot5")
    parser.add_argument("--report", default="reports/class_targets_pilot5")
    parser.add_argument("--policy", default=None, help="default: the recipe's policy, else central_greedy")
    parser.add_argument("--episodes", type=int, default=None, help="override the recipe's episode count")
    args = parser.parse_args()

    recipe = load_recipe(ROOT / args.recipe)
    if args.episodes is not None:
        recipe["episodes"] = args.episodes
    t0 = time.perf_counter()
    kind = recipe.pop("kind", "class_targets")     # which planner the recipe is for
    plan = PLANNERS[kind](recipe)
    print(f"planned {len(plan['episodes'])} episodes in {time.perf_counter() - t0:.1f} s "
          f"(split {plan['split_counts']})")
    report = ROOT / args.report
    report.mkdir(parents=True, exist_ok=True)
    export_plan(plan, report / "plan.json")

    t0 = time.perf_counter()
    policy = args.policy or plan["recipe"].get("policy", "central_greedy")
    print(f"policy: {policy}")
    manifest = generate_dataset(plan, ROOT / args.output, policy_name=policy)
    print(f"generation {manifest['generation_status']} in {time.perf_counter() - t0:.1f} s")
    for entry in manifest["episodes"]:
        if entry["generation_status"] != "completed":
            print(f"  {entry['dataset_episode_id']}: {entry['generation_status']} {entry['error_reason']}")

    attach_previews(ROOT / args.output, plan)       # dataset browser listing
    audit = audit_class_dataset(ROOT / args.output)
    (report / "audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report / "audit.md").write_text(audit_markdown(audit, plan), encoding="utf-8")
    print(f"audit: {audit['targets_passing_all_checks']}/{audit['targets_checked']} targets pass all checks "
          f"-> {report / 'audit.md'}")


if __name__ == "__main__":
    main()
