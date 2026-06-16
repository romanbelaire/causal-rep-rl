"""Train a single Procgen or DMControl performance environment task."""

import argparse

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.evaluation.suites import EVAL_SUITES
from src.experiments.performance_runner import run_performance_train


def main():
    parser = argparse.ArgumentParser(description="Train on a performance suite task")
    parser.add_argument("--suite", type=str, required=True, choices=list(EVAL_SUITES.keys()))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp-name", type=str, default="exp_full")
    parser.add_argument("--agent", type=str, default="ctro", choices=["ctro", "ppo"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    args = parser.parse_args()

    if args.task not in EVAL_SUITES[args.suite].tasks:
        raise ValueError(f"Unknown task {args.task} for suite {args.suite}")

    agent_cls = CTRO if args.agent == "ctro" else PPO
    algo_overrides = {"alpha": args.alpha, "beta": args.beta} if args.agent == "ctro" else {}

    run_performance_train(
        suite_name=args.suite,
        task=args.task,
        seed=args.seed,
        exp_name=args.exp_name,
        agent_cls=agent_cls,
        algo_overrides=algo_overrides,
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
