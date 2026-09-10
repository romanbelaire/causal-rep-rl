"""E0 lock: anti-aliased PPO (PL hinge), active modules off."""

import argparse

from src.agents.ctro import CTRO
from src.experiments.config import E0_ANTI_ALIASED_PPO, E0_TRAINING_SMOKE
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="E0 anti-aliased PPO (MiniGrid Unlock)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    train_overrides = E0_TRAINING_SMOKE if args.smoke else None
    run_experiment(
        exp_name="exp_e0_aa_ppo",
        seed=args.seed,
        agent_cls=CTRO,
        algo_overrides=E0_ANTI_ALIASED_PPO,
        train_overrides=train_overrides,
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
