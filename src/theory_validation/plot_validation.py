"""
Post-hoc plots for theory validation experiments (CPU-only).

When multiple *_seed{N} metrics CSVs exist for the same experiment, plots show
mean ± std across seeds (not one curve/panel per seed).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.theory_validation.aggregate_seeds import (
    aggregate_series,
    group_metrics_by_experiment,
    metric_series,
    n_seeds,
    strip_seed_suffix,
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

CHAIN_COLS = [
    ("chain_performance_gap", "J* − J^π (proxy)"),
    ("chain_z_star_distance", "‖Z* − Z‖ (proxy)"),
    ("chain_grad_z_v", "‖∇_Z V‖"),
    ("chain_sqrt_kl", "√KL"),
]

_NON_NUMERIC_COLS = frozenset({"eval_action_distribution"})


def load_metrics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, na_values=["", "nan", "NaN", "None"])
    df = df.replace("", np.nan)
    for col in df.columns:
        if col in _NON_NUMERIC_COLS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_mean_std(ax, steps, mean, std, label, color, linestyle="-"):
    ax.plot(steps, mean, label=label, color=color, linestyle=linestyle, alpha=0.9)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.22, linewidth=0)


def _seed_label(name: str, dfs: list[pd.DataFrame]) -> str:
    ns = n_seeds(dfs)
    short = strip_seed_suffix(name).removeprefix("theory_")
    return f"{short} (n={ns})" if ns > 1 else short


def plot_mu_vs_rank(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Assumption 1: μ vs effective feature rank (mean ± std over seeds)", fontweight="bold")

    for i, (dfs, name) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        steps, mu, mu_std = aggregate_series(dfs, "convexity_mu")
        _, log_rank, lr_std = aggregate_series(dfs, "log_effective_feature_rank_pr")
        n = min(len(steps), len(mu), len(log_rank))
        if n < 3:
            continue
        steps, mu, mu_std = steps[:n], mu[:n], mu_std[:n]
        log_rank, lr_std = log_rank[:n], lr_std[:n]

        _plot_mean_std(axes[0], steps, mu, mu_std, f"{label} μ", color)
        _plot_mean_std(axes[0], steps, log_rank, lr_std, f"{label} log rank", color, linestyle="--")

        rs = []
        for df in dfs:
            _, m = metric_series(df, "convexity_mu")
            _, r = metric_series(df, "log_effective_feature_rank_pr")
            nn = min(len(m), len(r))
            if nn >= 3:
                rs.append(stats.pearsonr(m[:nn], r[:nn])[0])
        r_mean = float(np.mean(rs)) if rs else float("nan")
        axes[1].scatter(log_rank, mu, c=color, alpha=0.45, s=14, label=f"{label} (r̄={r_mean:.2f})")

    axes[0].set_xlabel("Training steps")
    axes[0].set_ylabel("Value")
    axes[0].legend(fontsize=8)
    axes[0].set_title("μ and log effective rank over time")
    axes[1].set_xlabel("log effective feature rank (participation ratio)")
    axes[1].set_ylabel("μ (min Hessian eigenvalue)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("μ vs rank (mean trajectory)")

    fig.tight_layout()
    fig.savefig(output_dir / "mu_vs_feature_rank.png", dpi=150)
    plt.close(fig)


def plot_mu_vs_return(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("μ vs policy performance (mean ± std over seeds)", fontweight="bold")

    for i, (dfs, name) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        steps, mu, mu_std = aggregate_series(dfs, "convexity_mu")
        _, ret, ret_std = aggregate_series(dfs, "mean_episode_return")
        n = min(len(steps), len(mu), len(ret))
        if n < 3:
            continue

        _plot_mean_std(axes[0], steps[:n], mu[:n], mu_std[:n], f"{label} μ", color)
        _plot_mean_std(axes[1], steps[:n], ret[:n], ret_std[:n], f"{label} return", color)

        lag_rs = []
        for df in dfs:
            _, m = metric_series(df, "convexity_mu")
            _, r = metric_series(df, "mean_episode_return")
            nn = min(len(m), len(r))
            if nn > 5:
                m_s = pd.Series(m[:nn]).rolling(5, min_periods=1).mean().values
                r_s = pd.Series(r[:nn]).rolling(5, min_periods=1).mean().values
                if len(m_s) > 2:
                    lag_rs.append(stats.pearsonr(np.diff(m_s), np.diff(r_s))[0])
        if lag_rs:
            axes[0].text(
                0.02, 0.95 - 0.08 * i,
                f"{label}: corr(Δμ,Δret)={np.mean(lag_rs):.2f}±{np.std(lag_rs):.2f}",
                transform=axes[0].transAxes,
                fontsize=8,
                color=color,
            )

    axes[0].set_ylabel("μ")
    axes[1].set_ylabel("Mean episode return")
    axes[1].set_xlabel("Training steps")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "mu_vs_return_temporal.png", dpi=150)
    plt.close(fig)


def plot_collapse_ablation(
    dfs_on: list[pd.DataFrame],
    dfs_off: list[pd.DataFrame],
    name_on: str,
    name_off: str,
    output_dir: Path,
):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Controlled collapse: repr on vs off (mean ± std over seeds)", fontweight="bold")

    for dfs, base_name, color in [
        (dfs_on, name_on, COLORS[0]),
        (dfs_off, name_off, COLORS[1]),
    ]:
        label = _seed_label(base_name, dfs)
        steps, mu, mu_std = aggregate_series(dfs, "convexity_mu")
        _, ret, ret_std = aggregate_series(dfs, "mean_episode_return")
        n = min(len(steps), len(mu), len(ret))
        if n < 2:
            continue
        _plot_mean_std(axes[0], steps[:n], mu[:n], mu_std[:n], label, color)
        _plot_mean_std(axes[1], steps[:n], ret[:n], ret_std[:n], label, color)

    axes[0].set_ylabel("μ")
    axes[1].set_ylabel("Mean episode return")
    axes[1].set_xlabel("Training steps")
    axes[0].legend()
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "collapse_lconv_ablation.png", dpi=150)
    plt.close(fig)


def _short_name(name: str) -> str:
    return strip_seed_suffix(name).removeprefix("theory_")


def _plot_chain_overlay(ax, dfs: list[pd.DataFrame], title: str):
    for i, (col, label) in enumerate(CHAIN_COLS):
        steps, mean, std = aggregate_series(dfs, col)
        if len(steps) == 0:
            continue
        _plot_mean_std(ax, steps, mean, std, label, COLORS[i % len(COLORS)])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Training steps")
    ax.legend(fontsize=7, loc="best")


def _plot_step2_proxy(ax, dfs: list[pd.DataFrame], title: str):
    steps, grad, grad_std = aggregate_series(dfs, "chain_grad_z_v")
    _, rhs, rhs_std = aggregate_series(dfs, "chain_rhs_unscaled")
    n = min(len(steps), len(grad), len(rhs))
    if n < 3:
        ax.set_visible(False)
        return
    _plot_mean_std(ax, steps[:n], grad[:n], grad_std[:n], "‖∇_Z V‖", COLORS[0])
    _plot_mean_std(ax, steps[:n], rhs[:n], rhs_std[:n], "C_Z√KL + Lδ (unscaled)", COLORS[1])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Training steps")
    ax.legend(fontsize=7, loc="best")


def plot_bounding_chain_grids(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path):
    n = len(experiments)
    ncols = 2
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.suptitle("Theorem 4 bounding chain (mean ± std over seeds)", fontweight="bold")
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (dfs, name) in zip(axes_flat, experiments):
        _plot_chain_overlay(ax, dfs, _short_name(name))
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_dir / "bounding_chain_grid.png", dpi=150)
    plt.close(fig)

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig2.suptitle("Theorem 4 step-2 proxy: ‖∇_Z V‖ vs RHS (mean ± std)", fontweight="bold")
    axes2_flat = np.atleast_1d(axes2).ravel()

    for ax, (dfs, name) in zip(axes2_flat, experiments):
        _plot_step2_proxy(ax, dfs, _short_name(name))
    for ax in axes2_flat[n:]:
        ax.set_visible(False)

    fig2.tight_layout()
    fig2.savefig(output_dir / "bounding_chain_grad_vs_rhs_grid.png", dpi=150)
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Plot theory validation metrics from CSV logs")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/theory_validation"),
        help="Directory containing *_metrics.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/theory_validation"),
        help="Where to save figures",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.log_dir.glob("*_metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics CSV in {args.log_dir}")

    by_experiment = group_metrics_by_experiment(csv_files, load_metrics)

    baseline_keys = [
        "theory_vanilla_ppo_impala_mlp",
        "theory_vae_ppo_no_repr_loss",
        "theory_vanilla_ppo_impala_mlp_v2",
        "theory_vae_ppo_no_repr_loss_v2",
    ]
    baselines = [(by_experiment[k], k) for k in baseline_keys if k in by_experiment]
    if baselines:
        plot_mu_vs_rank(baselines, args.output_dir)
        plot_mu_vs_return(baselines, args.output_dir)

    collapse_pairs = [
        ("theory_rstr_lconv_on", "theory_rstr_lconv_off"),
        ("theory_rstr_lconv_on_v2", "theory_rstr_lconv_off_v2"),
        ("exp3_rstr_repr_on", "exp3_rstr_repr_off"),
    ]
    for on_key, off_key in collapse_pairs:
        if on_key in by_experiment and off_key in by_experiment:
            plot_collapse_ablation(
                by_experiment[on_key],
                by_experiment[off_key],
                on_key,
                off_key,
                args.output_dir,
            )
            break

    chain_experiments = [
        (dfs, name)
        for name, dfs in sorted(by_experiment.items())
        if dfs and "chain_grad_z_v" in dfs[0].columns
    ]
    if chain_experiments:
        plot_bounding_chain_grids(chain_experiments, args.output_dir)

    print(f"Wrote plots to {args.output_dir} ({len(by_experiment)} experiment groups)")


if __name__ == "__main__":
    main()
