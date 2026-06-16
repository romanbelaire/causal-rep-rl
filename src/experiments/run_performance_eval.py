"""Run performance evaluation suites on trained checkpoints."""

import argparse

from src.evaluation.runner import run_eval_suite_from_checkpoint
from src.evaluation.suites import EVAL_SUITES


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate policy performance on Procgen or DMControl suites (full vs test)."
    )
    parser.add_argument(
        "--suite",
        type=str,
        required=True,
        choices=list(EVAL_SUITES.keys()),
        help="Evaluation suite name",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Checkpoint file (Procgen) or directory with per-task weights (DMControl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file (Procgen) or directory with per-task config.json (DMControl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for performance_eval_metrics.csv",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="*",
        default=None,
        help="Subset of tasks (default: all tasks in suite)",
    )
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--step", type=int, default=0, help="Step label for CSV row")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    run_eval_suite_from_checkpoint(
        suite_name=args.suite,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        tasks=args.tasks,
        eval_episodes=args.eval_episodes,
        step=args.step,
    )


if __name__ == "__main__":
    main()
