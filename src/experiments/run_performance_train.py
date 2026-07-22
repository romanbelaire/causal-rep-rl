"""Train a single Procgen or DMControl performance environment task."""

import argparse

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.evaluation.suites import EVAL_SUITES
from src.experiments.performance_runner import run_performance_train


def _parse_hidden(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"empty hidden sizes: {s!r}")
    return [int(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description="Train on a performance suite task")
    parser.add_argument("--suite", type=str, required=True, choices=list(EVAL_SUITES.keys()))
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--exp-name", type=str, default="exp_full")
    parser.add_argument("--agent", type=str, default="ctro", choices=["ctro", "ppo"])
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument(
        "--policy-hidden",
        type=_parse_hidden,
        default=None,
        help="Comma-separated MLP hidden sizes, e.g. 64,64 or 256,256",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override train budget in env steps (epochs = total_steps // buffer_size).",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Override vectorized env count (default: suite config; 1 = serial rollout).",
    )
    args = parser.parse_args()

    if args.task not in EVAL_SUITES[args.suite].tasks:
        raise ValueError(f"Unknown task {args.task} for suite {args.suite}")

    agent_cls = CTRO if args.agent == "ctro" else PPO
    if args.agent == "ctro":
        algo_overrides = {}
        if args.alpha is not None:
            algo_overrides["alpha"] = args.alpha
        if args.beta is not None:
            algo_overrides["beta"] = args.beta
    else:
        algo_overrides = {"alpha": 0.0, "beta": 0.0, "vae_coef": 0.0}

    if args.learning_rate is not None:
        algo_overrides["learning_rate"] = args.learning_rate
    if args.entropy_coef is not None:
        algo_overrides["entropy_coef"] = args.entropy_coef
    if args.num_epochs is not None:
        algo_overrides["num_epochs"] = args.num_epochs

    arch_overrides = None
    if args.policy_hidden is not None:
        arch_overrides = {"policy_hidden": args.policy_hidden}

    train_overrides = None
    if args.total_steps is not None:
        train_overrides = {"total_steps": args.total_steps}

    result = run_performance_train(
        suite_name=args.suite,
        task=args.task,
        seed=args.seed,
        exp_name=args.exp_name,
        agent_cls=agent_cls,
        algo_overrides=algo_overrides,
        arch_overrides=arch_overrides,
        train_overrides=train_overrides,
        results_root=args.results_root,
        device=args.device,
        num_envs=args.num_envs,
    )
    print(
        f"TrainResult status={result.status} steps={result.total_steps} "
        f"eval_full={result.eval_full_return_mean}",
        flush=True,
    )


if __name__ == "__main__":
    main()
