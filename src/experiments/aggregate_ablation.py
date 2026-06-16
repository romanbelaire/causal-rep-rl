"""Aggregate 2x2 ablation table and comparison plots."""

import argparse
import csv
from pathlib import Path

import numpy as np

from src.experiments.config import DEFAULT_SEEDS, FULL_SWEEP_ALPHA, FULL_SWEEP_BETA


def _read_final_row(metrics_path: Path) -> dict:
    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No metrics in {metrics_path}")
    return rows[-1]


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    val = row.get(key, "")
    if val == "" or val is None:
        return default
    return float(val)


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _aggregate_cell(results_root: Path, exp_path: str, seeds: list[int]) -> dict:
    returns = []
    mu_pl = []
    pr = []
    for seed in seeds:
        metrics_file = results_root / exp_path / f"seed_{seed}" / "metrics.csv"
        row = _read_final_row(metrics_file)
        returns.append(_float(row, "eval_return_mean", _float(row, "mean_episode_return")))
        mu_pl.append(_float(row, "mu_pl_q05"))
        pr.append(_float(row, "feature_rank_participation_ratio"))

    return {
        "return_mean": float(np.mean(returns)),
        "return_sem": _sem(returns),
        "mu_pl_q05_mean": float(np.mean(mu_pl)),
        "pr_mean": float(np.mean(pr)),
    }


def _best_full_config(results_root: Path, seeds: list[int]) -> tuple[float, float, dict]:
    best_alpha, best_beta = FULL_SWEEP_ALPHA[0], FULL_SWEEP_BETA[0]
    best_score = -float("inf")
    best_stats = {}

    for alpha in FULL_SWEEP_ALPHA:
        for beta in FULL_SWEEP_BETA:
            path = f"exp_full/alpha_{alpha}_beta_{beta}"
            stats = _aggregate_cell(results_root, path, seeds)
            score = stats["return_mean"] + 0.1 * stats["mu_pl_q05_mean"]
            if score > best_score:
                best_score = score
                best_alpha, best_beta = alpha, beta
                best_stats = stats

    return best_alpha, best_beta, best_stats


def main():
    parser = argparse.ArgumentParser(description="Aggregate CTRO 2x2 ablation table")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    root = Path(args.results_root)
    seeds = args.seeds

    best_alpha, best_beta, full_stats = _best_full_config(root, seeds)
    full_path = f"exp_full/alpha_{best_alpha}_beta_{best_beta}"

    table = {
        "BASELINE": _aggregate_cell(root, "exp_baseline", seeds),
        "MICO_ONLY": _aggregate_cell(root, "exp_mico_only", seeds),
        "PL_ONLY": _aggregate_cell(root, "exp_pl_only", seeds),
        f"FULL (alpha={best_alpha}, beta={best_beta})": full_stats,
    }

    print("\n=== CTRO 2x2 Ablation ===\n")
    print(f"{'Cell':<40} {'Return':>12} {'mu_PL (q05)':>14} {'PR':>10}")
    print("-" * 80)
    for name, stats in table.items():
        ret = f"{stats['return_mean']:.3f} ± {stats['return_sem']:.3f}"
        print(
            f"{name:<40} {ret:>12} {stats['mu_pl_q05_mean']:>14.4f} "
            f"{stats['pr_mean']:>10.4f}"
        )

    print("\n2x2 layout:")
    print(f"  No MICo  | BASELINE return={table['BASELINE']['return_mean']:.3f} "
          f"| PL_ONLY return={table['PL_ONLY']['return_mean']:.3f}")
    print(f"  With MICo| MICO_ONLY return={table['MICO_ONLY']['return_mean']:.3f} "
          f"| FULL return={full_stats['return_mean']:.3f}")


if __name__ == "__main__":
    main()
