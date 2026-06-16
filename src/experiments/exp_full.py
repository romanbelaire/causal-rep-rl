"""EXP_FULL: full CTRO objective L_PPO + alpha*L_MICo + beta*L_PL."""

import argparse

from src.agents.ctro import CTRO
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="EXP_FULL: full CTRO with alpha/beta sweep")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    exp_name = f"exp_full/alpha_{args.alpha}_beta_{args.beta}"
    run_experiment(
        exp_name=exp_name,
        seed=args.seed,
        agent_cls=CTRO,
        algo_overrides={"alpha": args.alpha, "beta": args.beta},
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
