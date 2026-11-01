"""Train MAPPO (or IPPO) on a scenario.

    python -m src.services.training_service --scenario nominal --steps 200000
    python -m src.services.training_service --ippo --out models/checkpoints/ippo.pt
    python -m src.services.training_service --central   # policy at the fusion centre

Training episodes are shortened (``--episode-s``) to speed up learning; the
trained policy is then evaluated on full-length episodes in the dashboard.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.env import RadarNetEnv
from src.core.scenario import load_scenario
from src.services.simulation_service import load_classifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="nominal")
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--episode-s", type=float, default=60.0)
    ap.add_argument("--ippo", action="store_true", help="decentralised critic (IPPO baseline)")
    ap.add_argument("--central", action="store_true",
                    help="policy at the fusion centre: actor sees the global state (central_ppo)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from src.models.marl.mappo import DEFAULT_CKPT, MAPPOConfig, MAPPOTrainer

    sc = load_scenario(args.scenario)
    sc["duration_s"] = args.episode_s
    clf = load_classifier()
    trainer = MAPPOTrainer(lambda: RadarNetEnv(sc, classifier=clf),
                           MAPPOConfig(centralised_critic=not args.ippo, central_actor=args.central),
                           seed=args.seed)
    name = "central_ppo" if args.central else ("ippo" if args.ippo else "mappo")
    out = Path(args.out) if args.out else DEFAULT_CKPT.with_name(f"{name}.pt")
    hist = trainer.train(args.steps, save_path=out)       # also saved every 10 updates
    trainer.save(out)
    out.with_suffix(".history.json").write_text(json.dumps(hist, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
