"""Return learning curves: FULL vs MICO_ONLY (and ablation cells)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.aggregate_ablation import _best_full_config
from src.experiments.config import DEFAULT_SEEDS
from src.experiments.plot_ctro import aggregate_metric_series, load_seed_metrics

FOCUS_CELLS = [
    ("FULL", None),
    ("MICO_ONLY", "exp_mico_only"),
    ("BASELINE", "exp_baseline"),
    ("PL_ONLY", "exp_pl_only"),
]

COLORS = {
    "FULL": "#1f77b4",
    "MICO_ONLY": "#ff7f0e",
    "BASELINE": "#d62728",
    "PL_ONLY": "#2ca02c",
}


def monotonicity_stats(steps: np.ndarray, values: np.ndarray) -> dict:
    """Diagnostics for whether return is improving over training."""
    if len(values) < 2:
        return {"positive_diff_frac": float("nan"), "spearman": float("nan")}

    diffs = np.diff(values)
    positive_frac = float((diffs > 0).mean())

    rank_steps = pd.Series(values).corr(pd.Series(steps), method="spearman")
    return {
        "positive_diff_frac": positive_frac,
        "spearman": float(rank_steps),
        "final_return": float(values[-1]),
        "max_return": float(values.max()),
    }


def plot_return_curves(
    results_root: Path,
    cells: list[tuple[str, str]],
    output_path: Path,
    seeds: list[int],
    train_metric: str = "mean_episode_return",
    eval_metric: str = "eval_return_mean",
) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Return curves (mean ± SEM across seeds)", fontsize=14, fontweight="bold")

    diagnostics = {}

    for name, exp_path in cells:
        color = COLORS[name]

        train_series = []
        for seed in seeds:
            df = load_seed_metrics(results_root, exp_path, seed)
            sub = df[["step", train_metric]].dropna()
            train_series.append(sub.set_index("step")[train_metric])
        combined = pd.concat(train_series, axis=1)
        steps = combined.index.to_numpy(dtype=float)
        mean = combined.mean(axis=1).to_numpy()
        sem = combined.sem(axis=1, ddof=1).to_numpy()
        sem = np.nan_to_num(sem, nan=0.0)
        x = steps / 1e6

        axes[0].plot(x, mean, color=color, linewidth=2, label=name)
        axes[0].fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
        for seed_idx, series in enumerate(train_series):
            axes[0].plot(
                series.index.to_numpy() / 1e6,
                series.to_numpy(),
                color=color,
                alpha=0.25,
                linewidth=0.8,
            )

        diagnostics[name] = {
            "train": monotonicity_stats(steps, mean),
            "per_seed_train": {},
        }
        for seed, series in zip(seeds, train_series):
            s = series.index.to_numpy()
            v = series.to_numpy()
            diagnostics[name]["per_seed_train"][str(seed)] = monotonicity_stats(s, v)

        eval_series = []
        for seed in seeds:
            df = load_seed_metrics(results_root, exp_path, seed)
            sub = df[["step", eval_metric]].dropna(subset=[eval_metric])
            if len(sub) > 0:
                eval_series.append(sub.set_index("step")[eval_metric])
        if eval_series:
            eval_combined = pd.concat(eval_series, axis=1)
            eval_steps = eval_combined.index.to_numpy(dtype=float)
            eval_mean = eval_combined.mean(axis=1).to_numpy()
            eval_sem = eval_combined.sem(axis=1, ddof=1).to_numpy()
            eval_sem = np.nan_to_num(eval_sem, nan=0.0)
            axes[1].errorbar(
                eval_steps / 1e6,
                eval_mean,
                yerr=eval_sem,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=name,
                capsize=3,
            )
            diagnostics[name]["eval"] = monotonicity_stats(eval_steps, eval_mean)

    axes[0].set_xlabel("Environment steps (millions)")
    axes[0].set_ylabel("Training rollout return")
    axes[0].set_title("mean_episode_return (every 10k steps)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Environment steps (millions)")
    axes[1].set_ylabel("Eval return")
    axes[1].set_title("eval_return_mean (every 100 epochs)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return diagnostics


def plot_full_vs_mico(
    results_root: Path,
    full_path: str,
    output_path: Path,
    seeds: list[int],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, exp_path in [("FULL", full_path), ("MICO_ONLY", "exp_mico_only")]:
        steps, mean, sem = aggregate_metric_series(
            results_root, exp_path, "mean_episode_return", seeds
        )
        color = COLORS[name]
        x = steps / 1e6
        ax.plot(x, mean, color=color, linewidth=2.5, label=name)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Training rollout return")
    ax.set_title("FULL vs MICO_ONLY — training return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="CTRO return curve diagnostics")
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="plots/ctro")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_alpha, best_beta, _ = _best_full_config(results_root, args.seeds)
    full_path = f"exp_full/alpha_{best_alpha}_beta_{best_beta}"

    cells = [
        ("FULL", full_path),
        ("MICO_ONLY", "exp_mico_only"),
        ("BASELINE", "exp_baseline"),
        ("PL_ONLY", "exp_pl_only"),
    ]

    diagnostics = plot_return_curves(
        results_root,
        cells,
        output_dir / "return_curves.png",
        args.seeds,
    )
    plot_full_vs_mico(
        results_root,
        full_path,
        output_dir / "return_full_vs_mico.png",
        args.seeds,
    )

    report_path = output_dir / "return_curve_diagnostics.json"
    report_path.write_text(json.dumps(diagnostics, indent=2))
    print(f"Wrote {output_dir / 'return_curves.png'}")
    print(f"Wrote {output_dir / 'return_full_vs_mico.png'}")
    print(f"Wrote {report_path}")

    full_train = diagnostics["FULL"]["train"]
    mico_train = diagnostics["MICO_ONLY"]["train"]
    print(
        f"\nFULL train return: spearman={full_train['spearman']:.3f}, "
        f"positive_diff_frac={full_train['positive_diff_frac']:.3f}, "
        f"final={full_train['final_return']:.3f}"
    )
    print(
        f"MICO_ONLY train return: spearman={mico_train['spearman']:.3f}, "
        f"positive_diff_frac={mico_train['positive_diff_frac']:.3f}, "
        f"final={mico_train['final_return']:.3f}"
    )


if __name__ == "__main__":
    main()
