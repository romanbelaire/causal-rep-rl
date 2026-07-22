"""Plot CTRO experiment results: learning curves and 2x2 ablation summary.

Panel A: ablation_final_bars.png (return / mu_PL / PR side-by-side)
Panel B: mu_pl_vs_return.png (mu_PL vs eval return, colored by method)
Panel C: dual_axis_baseline_vs_pl.png (return + mu_PL over training)
"""

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

DUAL_AXIS_CELLS = [
    ("BASELINE", "exp_baseline"),
    ("PL_ONLY", "exp_pl_only"),
]


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


def _method_color(name: str) -> str:
    key = name.split()[0]
    return COLORS.get(key, COLORS.get(name, "#333333"))


def collect_eval_scatter_points(
    results_root: Path,
    cells: list[tuple[str, str]],
    seeds: list[int],
) -> pd.DataFrame:
    """One row per (method, seed, eval checkpoint) with mu_PL and eval return."""
    rows = []
    for name, exp_path in cells:
        for seed in seeds:
            df = load_seed_metrics(results_root, exp_path, seed)
            if "eval_return_mean" not in df.columns or "mu_pl_q05" not in df.columns:
                raise KeyError(f"missing eval/mu_PL columns in {exp_path} seed {seed}")
            sub = df[["step", "eval_return_mean", "mu_pl_q05"]].dropna(
                subset=["eval_return_mean", "mu_pl_q05"]
            )
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "step": float(row["step"]),
                        "eval_return": float(row["eval_return_mean"]),
                        "mu_pl_q05": float(row["mu_pl_q05"]),
                    }
                )
    return pd.DataFrame(rows)


def plot_mu_pl_vs_return(
    results_root: Path,
    cells: list[tuple[str, str]],
    output_path: Path,
    seeds: list[int],
) -> None:
    """Panel B: mu_PL vs eval return across eval checkpoints, colored by method."""
    points = collect_eval_scatter_points(results_root, cells, seeds)
    if points.empty:
        raise ValueError("No eval checkpoints with mu_PL found for Panel B")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, _ in cells:
        sub = points[points["method"] == name]
        color = _method_color(name)
        # Mid-training evals: open markers; final eval per seed: filled.
        final_steps = sub.groupby("seed")["step"].transform("max")
        mid = sub[sub["step"] < final_steps]
        final = sub[sub["step"] == final_steps]
        if not mid.empty:
            ax.scatter(
                mid["mu_pl_q05"],
                mid["eval_return"],
                s=36,
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                alpha=0.7,
                label=None,
            )
        ax.scatter(
            final["mu_pl_q05"],
            final["eval_return"],
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            label=name,
            zorder=3,
        )

    # Global trend for visual mediation cue (not a causal estimate).
    x = points["mu_pl_q05"].to_numpy()
    y = points["eval_return"].to_numpy()
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label=f"trend (slope={slope:.2f})",
        )

    ax.set_xlabel(r"$\mu_{PL}$ (q05)")
    ax.set_ylabel("Eval return")
    ax.set_title(
        r"Panel B: value-geometry health ($\mu_{PL}$) vs policy return",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dual_axis_baseline_vs_pl(
    results_root: Path,
    output_path: Path,
    seeds: list[int],
    return_metric: str = "mean_episode_return",
    mu_metric: str = "mu_pl_q05",
) -> None:
    """Panel C: dual-axis return + mu_PL curves for BASELINE vs PL_ONLY."""
    series = []
    for name, exp_path in DUAL_AXIS_CELLS:
        steps_r, mean_r, sem_r = aggregate_metric_series(
            results_root, exp_path, return_metric, seeds
        )
        steps_m, mean_m, sem_m = aggregate_metric_series(
            results_root, exp_path, mu_metric, seeds
        )
        series.append((name, steps_r, mean_r, sem_r, steps_m, mean_m, sem_m))

    return_max = max(
        float((mean_r + sem_r).max()) for _, _, mean_r, sem_r, _, _, _ in series
    )
    mu_max = max(
        float((mean_m + sem_m).max()) for _, _, _, _, _, mean_m, sem_m in series
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.suptitle(
        r"Panel C: return and $\mu_{PL}$ move together under PL constraint",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (name, steps_r, mean_r, sem_r, steps_m, mean_m, sem_m) in zip(
        axes, series
    ):
        color = COLORS[name]
        x_r = steps_r / 1e6
        x_m = steps_m / 1e6

        ax.plot(x_r, mean_r, color=color, linewidth=2, label="Return")
        ax.fill_between(
            x_r, mean_r - sem_r, mean_r + sem_r, color=color, alpha=0.2
        )
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Mean episode return", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_ylim(0.0, return_max * 1.05)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(
            x_m,
            mean_m,
            color="0.25",
            linewidth=2,
            linestyle="--",
            label=r"$\mu_{PL}$ (q05)",
        )
        ax2.fill_between(
            x_m, mean_m - sem_m, mean_m + sem_m, color="0.25", alpha=0.15
        )
        ax2.set_ylabel(r"$\mu_{PL}$ (q05)", color="0.25")
        ax2.tick_params(axis="y", labelcolor="0.25")
        ax2.set_ylim(0.0, mu_max * 1.05)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


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

    scatter_cells = [
        ("BASELINE", "exp_baseline"),
        ("MICO_ONLY", "exp_mico_only"),
        ("PL_ONLY", "exp_pl_only"),
        (full_label, full_path),
    ]
    plot_mu_pl_vs_return(
        results_root,
        scatter_cells,
        output_dir / "mu_pl_vs_return.png",
        seeds,
    )
    print(f"Wrote {output_dir / 'mu_pl_vs_return.png'}")

    plot_dual_axis_baseline_vs_pl(
        results_root,
        output_dir / "dual_axis_baseline_vs_pl.png",
        seeds,
    )
    print(f"Wrote {output_dir / 'dual_axis_baseline_vs_pl.png'}")

    print(f"\nBest FULL config: alpha={best_alpha}, beta={best_beta}")


if __name__ == "__main__":
    main()
