"""EXP_BASELINE: vanilla PPO — representation collapse motivation."""

import argparse

from src.agents.ppo import PPO
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="EXP_BASELINE: vanilla PPO on Minigrid")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        exp_name="exp_baseline",
        seed=args.seed,
        agent_cls=PPO,
        algo_overrides={"alpha": 0.0, "beta": 0.0},
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
