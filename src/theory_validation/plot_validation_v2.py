"""
Post-hoc plots for theory validation v2 (CPU-only).

Multiple *_seed{N} runs are aggregated as mean ± std (not one line per seed).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

from src.theory_validation.aggregate_seeds import (
    aggregate_series,
    group_metrics_by_experiment,
    metric_series,
    n_seeds,
    strip_seed_suffix,
)

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000]

# Directional κ toward Z* (step 2); fall back to global λ_min proxies in older CSVs.
CURVATURE_CONCAVE = ("convexity_kappa_concave_mean", "convexity_mu_concave_mean")
CURVATURE_PCT_BAD = ("convexity_pct_negative_kappa", "convexity_pct_concave")


def _pick_metric_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return candidates[-1]


def load_metrics(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, na_values=["", "nan", "NaN", "None"])
    except pd.errors.ParserError as e:
        raise pd.errors.ParserError(f"{csv_path}: {e}") from e
    for col in df.columns:
        if col == "eval_action_distribution":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _plot_mean_std(ax, steps, mean, std, label, color, linestyle="-"):
    ax.plot(steps, mean, label=label, color=color, linestyle=linestyle, alpha=0.9)
    ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.22, linewidth=0)


def _zscore(y: np.ndarray) -> np.ndarray:
    std = y.std()
    if std < 1e-12:
        return y - y.mean()
    return (y - y.mean()) / std


def _value_at_checkpoint(df: pd.DataFrame, checkpoint: int, metric: str) -> float:
    if metric not in df.columns:
        return np.nan
    valid = df[df[metric].notna() & df["step"].notna()]
    if valid.empty:
        return np.nan
    idx = (valid["step"] - checkpoint).abs().idxmin()
    return float(valid.loc[idx, metric])


def _seed_label(name: str, dfs: list[pd.DataFrame]) -> str:
    ns = n_seeds(dfs)
    short = strip_seed_suffix(name)
    return f"{short} (n={ns})" if ns > 1 else short


def plot_exp1_curvature_rank(
    experiments: list[tuple[list[pd.DataFrame], str]],
    output_dir: Path,
    rank_col: str = "log_effective_feature_rank_pr",
    rank_label: str = "log effective rank (PR)",
    output_name: str | None = None,
    rank_suffix: str = "rank",
    title_suffix: str = "participation ratio",
):
    concave_col = _pick_metric_column(experiments[0][0][0], CURVATURE_CONCAVE)
    if output_name is None:
        stem = "kappa_concave" if concave_col.startswith("convexity_kappa") else "mu_concave"
        output_name = f"exp1_{stem}_vs_{rank_suffix}.png"
    pct_col = _pick_metric_column(experiments[0][0][0], CURVATURE_PCT_BAD)
    using_kappa = concave_col.startswith("convexity_kappa")
    curv_label = "κ_concave" if using_kappa else "μ_concave (λ_min proxy)"
    pct_label = "pct κ<0" if using_kappa else "pct_concave"

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Exp 1: {curv_label} vs effective feature rank ({title_suffix}, mean ± std over seeds)",
        fontweight="bold",
    )

    for i, (dfs, name) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        steps, mu_c, mu_std = aggregate_series(dfs, concave_col)
        _, pct_c, pct_std = aggregate_series(dfs, pct_col)
        _, log_rank, lr_std = aggregate_series(dfs, rank_col)
        n = min(len(steps), len(mu_c), len(pct_c), len(log_rank))
        if n < 3:
            continue

        _plot_mean_std(axes[0, 0], steps[:n], mu_c[:n], mu_std[:n], label, color)
        _plot_mean_std(axes[0, 1], steps[:n], pct_c[:n], pct_std[:n], label, color)
        _plot_mean_std(axes[1, 0], steps[:n], log_rank[:n], lr_std[:n], label, color)

        rs = []
        for df in dfs:
            col_c = _pick_metric_column(df, CURVATURE_CONCAVE)
            _, m = metric_series(df, col_c)
            _, r = metric_series(df, rank_col)
            nn = min(len(m), len(r))
            if nn >= 3 and m[:nn].std() > 1e-12 and r[:nn].std() > 1e-12:
                rs.append(stats.pearsonr(m[:nn], r[:nn])[0])
        r_mean = float(np.mean(rs)) if rs else float("nan")
        axes[1, 1].scatter(log_rank[:n], mu_c[:n], c=color, alpha=0.45, s=14, label=f"{label} r̄={r_mean:.2f}")

    axes[0, 0].set_title(f"{curv_label} (batch mean)")
    axes[0, 1].set_title(pct_label)
    axes[1, 0].set_title(rank_label)
    axes[1, 1].set_title(f"{curv_label} vs rank")
    axes[1, 1].set_xlabel(rank_label)
    for ax in axes.ravel():
        if ax is not axes[1, 1]:
            ax.set_xlabel("step")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=150)
    plt.close(fig)


def plot_exp2_lagged_ccf(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path, max_lag: int = 5):
    concave_col = _pick_metric_column(experiments[0][0][0], CURVATURE_CONCAVE)
    curv_label = "κ_concave" if concave_col.startswith("convexity_kappa") else "μ_concave"
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Exp 2: lagged CCF({curv_label}, return) — mean ± std over seeds", fontweight="bold")

    for i, (dfs, name) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        ccfs = []
        lags_ref = None
        for df in dfs:
            col_c = _pick_metric_column(df, CURVATURE_CONCAVE)
            _, mu_c = metric_series(df, col_c)
            _, ret = metric_series(df, "mean_episode_return")
            nn = min(len(mu_c), len(ret))
            if nn < 8:
                continue
            mu_c = pd.Series(mu_c[:nn]).rolling(5, min_periods=1).mean().values
            ret = pd.Series(ret[:nn]).rolling(5, min_periods=1).mean().values
            mu_c = (mu_c - mu_c.mean()) / (mu_c.std() + 1e-8)
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
            ccf = signal.correlate(mu_c, ret, mode="full")
            lags = signal.correlation_lags(len(mu_c), len(ret), mode="full")
            sel = (lags >= 0) & (lags <= max_lag)
            ccfs.append(ccf[sel] / (len(mu_c) + 1e-8))
            lags_ref = lags[sel]
        if not ccfs or lags_ref is None:
            continue
        stacked = np.vstack(ccfs)
        ccf_mean = stacked.mean(axis=0)
        ccf_std = stacked.std(axis=0)
        _plot_mean_std(ax, lags_ref, ccf_mean, ccf_std, label, color)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(f"lag k ({curv_label} leads return by k metric points)")
    ax.set_ylabel("cross-correlation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "exp2_lagged_ccf.png", dpi=150)
    plt.close(fig)


def plot_exp3_checkpoints(
    dfs_on: list[pd.DataFrame],
    dfs_off: list[pd.DataFrame],
    output_dir: Path,
    metrics: list[str],
):
    rows = []
    for label, dfs in [("repr_on", dfs_on), ("repr_off", dfs_off)]:
        for df in dfs:
            for cp in CHECKPOINTS:
                row = {"condition": label, "checkpoint": cp}
                for m in metrics:
                    row[m] = _value_at_checkpoint(df, cp, m)
                rows.append(row)
    tab = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        for cond, color in zip(["repr_on", "repr_off"], COLORS[:2]):
            sub = tab[tab["condition"] == cond]
            means = [sub[sub["checkpoint"] == cp][m].mean() for cp in CHECKPOINTS]
            stds = [sub[sub["checkpoint"] == cp][m].std() for cp in CHECKPOINTS]
            x = np.arange(len(CHECKPOINTS))
            ax.bar(x + (0.2 if cond == "repr_off" else -0.2), means, width=0.35,
                   yerr=stds, label=cond, color=color, capsize=3)
        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels([f"{c/1e6:.0f}M" for c in CHECKPOINTS])
        ax.set_title(m)
        ax.legend(fontsize=8)
    fig.suptitle("Exp 3: checkpoint comparison (mean ± std over seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "exp3_checkpoints.png", dpi=150)
    plt.close(fig)
    tab.to_csv(output_dir / "exp3_checkpoint_table.csv", index=False)


def plot_exp4_chain(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path):
    n = len(experiments)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Exp 4: bounding chain with Z_ref (z-scored, mean ± std over seeds)", fontweight="bold")

    chain_cols = [
        ("chain_performance_gap", "J*−Jπ"),
        ("chain_z_star_distance", "‖Z−Z*‖"),
        ("chain_grad_z_v", "‖∇_Z V‖"),
        ("chain_sqrt_kl", "√KL"),
    ]
    for row, (dfs, name) in enumerate(experiments):
        title = _seed_label(name, dfs)
        for col_i, (col, label) in enumerate(chain_cols):
            steps, mean, std = aggregate_series(dfs, col)
            if len(steps) == 0:
                continue
            scale = mean.std() + 1e-8
            _plot_mean_std(
                axes[row, 0],
                steps,
                _zscore(mean),
                std / scale,
                label,
                COLORS[col_i % len(COLORS)],
            )
        axes[row, 0].set_title(f"{title} — chain")
        axes[row, 0].set_ylabel("z-score")
        axes[row, 0].legend(fontsize=7)

        steps, grad, grad_std = aggregate_series(dfs, "chain_grad_z_v")
        _, rhs, rhs_std = aggregate_series(dfs, "chain_rhs_scaled")
        npt = min(len(steps), len(grad), len(rhs))
        if npt >= 3:
            grad_z = _zscore(grad[:npt])
            rhs_z = _zscore(rhs[:npt])
            grad_scale = grad[:npt].std() + 1e-8
            rhs_scale = rhs[:npt].std() + 1e-8
            _plot_mean_std(axes[row, 1], steps[:npt], grad_z, grad_std[:npt] / grad_scale, "‖∇_Z V‖", COLORS[0])
            _plot_mean_std(axes[row, 1], steps[:npt], rhs_z, rhs_std[:npt] / rhs_scale, "scaled RHS", COLORS[1])
        for df in dfs:
            if "chain_bound_unreliable" not in df.columns:
                continue
            unreliable = df["chain_bound_unreliable"]
            bad = df["step"].values[unreliable > 0.5]
            for x in bad:
                axes[row, 1].axvline(x, color="red", alpha=0.03)
        axes[row, 1].set_title(f"{title} — grad vs scaled RHS")
        axes[row, 1].set_ylabel("z-score")
        axes[row, 1].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "exp4_bounding_chain.png", dpi=150)
    plt.close(fig)


def plot_exp5_transfer(experiments: list[tuple[list[pd.DataFrame], str]], output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exp 5: train vs transfer eval (z-scored, mean ± std over seeds)", fontweight="bold")
    for i, (dfs, name) in enumerate(experiments):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        steps, train_sr, train_std = aggregate_series(dfs, "eval_success_rate")
        _, transfer_sr, transfer_std = aggregate_series(dfs, "eval_transfer_success_rate")
        n = min(len(steps), len(train_sr), len(transfer_sr))
        if n < 2:
            continue
        train_z = _zscore(train_sr[:n])
        transfer_z = _zscore(transfer_sr[:n])
        train_scale = train_sr[:n].std() + 1e-8
        transfer_scale = transfer_sr[:n].std() + 1e-8
        _plot_mean_std(axes[0], steps[:n], train_z, train_std[:n] / train_scale, f"{label} train", color)
        _plot_mean_std(
            axes[0],
            steps[:n],
            transfer_z,
            transfer_std[:n] / transfer_scale,
            f"{label} transfer",
            color,
            linestyle="--",
        )
        rs = []
        for df in dfs:
            _, train = metric_series(df, "eval_success_rate")
            _, transfer = metric_series(df, "eval_transfer_success_rate")
            nn = min(len(train), len(transfer))
            if nn >= 3 and train[:nn].std() > 1e-12 and transfer[:nn].std() > 1e-12:
                rs.append(stats.pearsonr(_zscore(train[:nn]), _zscore(transfer[:nn]))[0])
        r_mean = float(np.mean(rs)) if rs else float("nan")
        axes[1].scatter(train_z, transfer_z, c=color, alpha=0.45, s=14, label=f"{label} r̄={r_mean:.2f}")

    axes[0].set_xlabel("step")
    axes[0].set_ylabel("z-score success rate")
    axes[0].set_title("train vs transfer over time")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("z-score train success rate")
    axes[1].set_ylabel("z-score transfer success rate")
    axes[1].set_title("train vs transfer correlation")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "exp5_transfer.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/theory_validation_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/theory_validation_v2"))
    parser.add_argument("--exp", type=str, default="all", choices=["all", "1", "2", "3", "4", "5"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.log_dir.glob("*_metrics.csv"))
    by_experiment = group_metrics_by_experiment(csv_files, load_metrics)

    def pick(predicate):
        return [(by_experiment[k], k) for k in sorted(by_experiment) if predicate(k)]

    if args.exp in ("all", "1"):
        exp1 = pick(lambda k: ("rstr_lconv_on" in k or "vae" in k) and "repr_" not in k and "exp4" not in k and "exp5" not in k and "expert" not in k)
        if exp1:
            plot_exp1_curvature_rank(exp1, args.output_dir)
            plot_exp1_curvature_rank(
                exp1,
                args.output_dir,
                rank_col="log_effective_feature_rank_pca",
                rank_label="log effective rank (PCA)",
                rank_suffix="pca_rank",
                title_suffix="PCA",
            )

    if args.exp in ("all", "2"):
        exp2 = pick(lambda k: ("rstr_lconv_on" in k or "vae" in k) and "repr_" not in k and "exp4" not in k and "exp5" not in k and "expert" not in k)
        if exp2:
            plot_exp2_lagged_ccf(exp2, args.output_dir)

    if args.exp in ("all", "3"):
        on_key = next((k for k in by_experiment if "repr_on" in k), None)
        off_key = next((k for k in by_experiment if "repr_off" in k), None)
        if on_key and off_key:
            plot_exp3_checkpoints(
                by_experiment[on_key],
                by_experiment[off_key],
                args.output_dir,
                [
                    "convexity_kappa_concave_mean",
                    "convexity_mu_concave_mean",
                    "log_effective_feature_rank_pr",
                    "log_effective_feature_rank_pca",
                    "mean_episode_return",
                ],
            )

    if args.exp in ("all", "4"):
        exp4 = pick(lambda k: "exp4" in k)
        if exp4:
            plot_exp4_chain(exp4, args.output_dir)

    if args.exp in ("all", "5"):
        exp5 = pick(lambda k: "exp5" in k)
        if exp5:
            plot_exp5_transfer(exp5, args.output_dir)

    print(f"Wrote plots to {args.output_dir} ({len(by_experiment)} experiment groups)")


if __name__ == "__main__":
    main()
