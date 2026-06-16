"""Pick best EXP_FULL alpha/beta by return + stable mu_PL."""

import argparse
from pathlib import Path

from src.experiments.aggregate_ablation import _best_full_config
from src.experiments.config import DEFAULT_SEEDS


def main():
    parser = argparse.ArgumentParser(description="Select best EXP_FULL configuration")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    alpha, beta, stats = _best_full_config(Path(args.results_root), args.seeds)
    print(f"Best config: alpha={alpha}, beta={beta}")
    print(f"  return_mean={stats['return_mean']:.4f}")
    print(f"  mu_pl_q05_mean={stats['mu_pl_q05_mean']:.4f}")
    print(f"  pr_mean={stats['pr_mean']:.4f}")


if __name__ == "__main__":
    main()
