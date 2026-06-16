"""EXP_PL_ONLY: PPO + PL coupling, no MICo (alpha=0)."""

import argparse

from src.agents.ctro import CTRO
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="EXP_PL_ONLY: CTRO with beta PL, alpha=0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        exp_name="exp_pl_only",
        seed=args.seed,
        agent_cls=CTRO,
        algo_overrides={"alpha": 0.0, "beta": args.beta},
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
