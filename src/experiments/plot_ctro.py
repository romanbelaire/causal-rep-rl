"""Plot CTRO experiment results: learning curves and 2x2 ablation summary."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.aggregate_ablation import (
    _aggregate_cell,
    _best_full_config,
)
from src.experiments.config import DEFAULT_SEEDS

CELL_LAYOUT = [
    ("BASELINE", "exp_baseline", 0, 0),
    ("PL_ONLY", "exp_pl_only", 0, 1),
    ("MICO_ONLY", "exp_mico_only", 1, 0),
]

METRIC_SPECS = [
    ("eval_return_mean", "Eval return", "ablation_return.png"),
    ("mu_pl_q05", "mu_PL (5th percentile)", "ablation_mu_pl.png"),
    (
        "feature_rank_participation_ratio",
        "Feature rank (participation ratio)",
        "ablation_feature_rank.png",
    ),
]

COLORS = {
    "BASELINE": "#d62728",
    "MICO_ONLY": "#ff7f0e",
    "PL_ONLY": "#2ca02c",
    "FULL": "#1f77b4",
}


def load_seed_metrics(results_root: Path, exp_path: str, seed: int) -> pd.DataFrame:
    path = results_root / exp_path / f"seed_{seed}" / "metrics.csv"
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in df.columns:
        if col != "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("step").reset_index(drop=True)


def aggregate_metric_series(
    results_root: Path,
    exp_path: str,
    metric: str,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = []
    for seed in seeds:
        df = load_seed_metrics(results_root, exp_path, seed)
        if metric not in df.columns:
            raise KeyError(f"{metric} missing in {exp_path} seed {seed}")
        sub = df[["step", metric]].dropna(subset=[metric])
        series.append(sub.set_index("step")[metric])

    combined = pd.concat(series, axis=1)
    steps = combined.index.to_numpy(dtype=float)
    mean = combined.mean(axis=1).to_numpy()
    sem = combined.sem(axis=1, ddof=1).to_numpy()
    sem = np.nan_to_num(sem, nan=0.0)
    return steps, mean, sem


def plot_ablation_grid(
    results_root: Path,
    cells: list[tuple[str, str, int, int]],
    metric: str,
    ylabel: str,
    output_path: Path,
    seeds: list[int],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    fig.suptitle(ylabel, fontsize=14, fontweight="bold")

    for name, exp_path, row, col in cells:
        ax = axes[row, col]
        steps, mean, sem = aggregate_metric_series(results_root, exp_path, metric, seeds)
        color = COLORS.get(name.split()[0], COLORS.get(name, "#333333"))
        x = steps / 1e6
        ax.plot(x, mean, color=color, linewidth=2, label=name)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    for ax in axes[1, :]:
        ax.set_xlabel("Environment steps (millions)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_final_bars(table: dict[str, dict], output_path: Path) -> None:
    names = list(table.keys())
    x = np.arange(len(names))
    width = 0.25

    returns = [table[n]["return_mean"] for n in names]
    return_sems = [table[n]["return_sem"] for n in names]
    mu_pl = [table[n]["mu_pl_q05_mean"] for n in names]
    pr = [table[n]["pr_mean"] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Final metrics (mean over seeds)", fontsize=14, fontweight="bold")

    axes[0].bar(x, returns, yerr=return_sems, capsize=4, color="#1f77b4")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    axes[0].set_ylabel("Eval return")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x, mu_pl, color="#2ca02c")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    axes[1].set_ylabel("mu_PL (q05)")
    axes[1].grid(True, axis="y", alpha=0.3)

    axes[2].bar(x, pr, color="#ff7f0e")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")
    axes[2].set_ylabel("Participation ratio")
    axes[2].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_ablation_table(table: dict[str, dict], output_path: Path) -> None:
    lines = [
        "CTRO 2x2 Ablation",
        f"{'Cell':<40} {'Return':>14} {'mu_PL (q05)':>14} {'PR':>10}",
        "-" * 80,
    ]
    for name, stats in table.items():
        ret = f"{stats['return_mean']:.4f} ± {stats['return_sem']:.4f}"
        lines.append(
            f"{name:<40} {ret:>14} {stats['mu_pl_q05_mean']:>14.4f} "
            f"{stats['pr_mean']:>10.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CTRO experiment results")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots/ctro")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = args.seeds

    best_alpha, best_beta, full_stats = _best_full_config(results_root, seeds)
    full_label = f"FULL (a={best_alpha}, b={best_beta})"
    full_path = f"exp_full/alpha_{best_alpha}_beta_{best_beta}"

    cells = CELL_LAYOUT + [(full_label, full_path, 1, 1)]

    table = {
        "BASELINE": _aggregate_cell(results_root, "exp_baseline", seeds),
        "MICO_ONLY": _aggregate_cell(results_root, "exp_mico_only", seeds),
        "PL_ONLY": _aggregate_cell(results_root, "exp_pl_only", seeds),
        full_label: full_stats,
    }

    for metric, ylabel, filename in METRIC_SPECS:
        plot_ablation_grid(
            results_root,
            cells,
            metric,
            ylabel,
            output_dir / filename,
            seeds,
        )
        print(f"Wrote {output_dir / filename}")

    plot_final_bars(table, output_dir / "ablation_final_bars.png")
    print(f"Wrote {output_dir / 'ablation_final_bars.png'}")

    write_ablation_table(table, output_dir / "ablation_table.txt")
    print(f"Wrote {output_dir / 'ablation_table.txt'}")

    print(f"\nBest FULL config: alpha={best_alpha}, beta={best_beta}")


if __name__ == "__main__":
    main()
