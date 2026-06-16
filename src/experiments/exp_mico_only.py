"""EXP_MICO_ONLY: PPO + MICo, no PL term (beta=0)."""

import argparse

from src.agents.ctro import CTRO
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="EXP_MICO_ONLY: CTRO with alpha MICo, beta=0")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    run_experiment(
        exp_name="exp_mico_only",
        seed=args.seed,
        agent_cls=CTRO,
        algo_overrides={"alpha": args.alpha, "beta": 0.0},
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
