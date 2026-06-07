"""
Post-hoc plots for theory validation v3 (CPU-only).

Kappa arm: tv3_exp1 (curvature vs rank), tv3_exp2 (ablations), tv3_exp3 (chain).
Distill arm: tv3_exp1, tv3_exp2 (chain).
Training-loss curves from *_intervention_loss.csv.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.theory_validation.aggregate_seeds import (
    aggregate_series,
    group_metrics_by_experiment,
    metric_series,
    n_seeds,
    strip_seed_suffix,
)
from src.theory_validation.plot_validation_v2 import (
    CHECKPOINTS,
    COLORS,
    CURVATURE_CONCAVE,
    CURVATURE_PCT_BAD,
    _pick_metric_column,
    _plot_mean_std,
    _seed_label,
    _value_at_checkpoint,
    _zscore,
    load_metrics,
    plot_exp1_curvature_rank,
    plot_exp4_chain,
)

INTERVENTION_CSV_GLOB = "*_intervention_loss.csv"


def plot_tv3_checkpoints(
    dfs_a: list[pd.DataFrame],
    dfs_b: list[pd.DataFrame],
    label_a: str,
    label_b: str,
    output_dir: Path,
    output_name: str,
    metrics: list[str],
    title: str,
):
    rows = []
    for label, dfs in [(label_a, dfs_a), (label_b, dfs_b)]:
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
        for cond, color in zip([label_a, label_b], COLORS[:2]):
            sub = tab[tab["condition"] == cond]
            means = [sub[sub["checkpoint"] == cp][m].mean() for cp in CHECKPOINTS]
            stds = [sub[sub["checkpoint"] == cp][m].std() for cp in CHECKPOINTS]
            x = np.arange(len(CHECKPOINTS))
            offset = -0.2 if cond == label_a else 0.2
            ax.bar(x + offset, means, width=0.35, yerr=stds, label=cond, color=color, capsize=3)
        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels([f"{c/1e6:.0f}M" for c in CHECKPOINTS])
        ax.set_title(m)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / output_name, dpi=150)
    plt.close(fig)
    tab.to_csv(output_dir / output_name.replace(".png", "_table.csv"), index=False)


def _load_intervention_by_experiment(log_dir: Path) -> dict[str, list[pd.DataFrame]]:
    files = sorted(log_dir.glob(INTERVENTION_CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No intervention loss CSVs in {log_dir}")
    by_exp: dict[str, list[pd.DataFrame]] = {}
    for p in files:
        name = p.name.replace("_intervention_loss.csv", "")
        base = strip_seed_suffix(name)
        df = load_metrics(p)
        by_exp.setdefault(base, []).append(df)
    return by_exp


def _training_loss_predicate(arm: str, exp: str):
    """Select intervention-loss runs for the requested arm / experiment slice."""
    if arm == "kappa":
        if exp == "1":
            return lambda k: k == "theory_v3_kappa_dir_rstr"
        if exp == "2":
            return lambda k: k.startswith("theory_v3_exp3_")
        if exp == "3":
            return lambda k: k == "theory_v3_kappa_dir_chain_rstr"
        return lambda k: (
            k == "theory_v3_kappa_dir_rstr"
            or k == "theory_v3_kappa_dir_chain_rstr"
            or k.startswith("theory_v3_exp3_")
        )
    if exp == "1":
        return lambda k: k == "theory_v3_z_distill_rstr"
    if exp == "2":
        return lambda k: k == "theory_v3_z_distill_chain_rstr"
    return lambda k: k in ("theory_v3_z_distill_rstr", "theory_v3_z_distill_chain_rstr")


def plot_tv3_training_loss(
    log_dir: Path,
    output_dir: Path,
    arm: str,
    experiment_predicate,
    exp: str = "all",
):
    by_exp = _load_intervention_by_experiment(log_dir)
    selected = [(by_exp[k], k) for k in sorted(by_exp) if experiment_predicate(k)]
    if not selected:
        raise FileNotFoundError(
            f"No intervention_loss runs matched arm={arm} exp={exp}; "
            f"available={[k for k in sorted(by_exp)]}"
        )

    if arm == "kappa":
        primary_col = "train_kappa_directional_loss"
        outfile = "tv3_kappa_training_loss.png"
        title = "tv3: κ-directional training loss (mean ± std over seeds)"
        ylabel = "κ-directional loss"
    else:
        primary_col = "train_z_distill_loss"
        outfile = "tv3_distill_training_loss.png"
        title = "tv3: Z* distillation training loss (mean ± std over seeds)"
        ylabel = "Z* distillation loss"

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (dfs, name) in enumerate(selected):
        color = COLORS[i % len(COLORS)]
        label = _seed_label(name, dfs)
        steps, mean, std = aggregate_series(dfs, primary_col)
        if len(steps) < 2:
            raise ValueError(
                f"{name}: need >=2 {primary_col} points for training-loss plot, got {len(steps)}"
            )
        _plot_mean_std(ax, steps, mean, std, label, color)

    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / outfile, dpi=150)
    plt.close(fig)


def _plot_exp1_kappa(output_dir: Path, by_experiment: dict):
    exp1 = []
    if "theory_v3_kappa_dir_rstr" in by_experiment:
        exp1.append((by_experiment["theory_v3_kappa_dir_rstr"], "theory_v3_kappa_dir_rstr"))
    if "theory_v3_vae_baseline" in by_experiment:
        exp1.append((by_experiment["theory_v3_vae_baseline"], "theory_v3_vae_baseline"))
    if len(exp1) < 2:
        raise FileNotFoundError("tv3 kappa exp1 needs theory_v3_kappa_dir_rstr and theory_v3_vae_baseline")
    plot_exp1_curvature_rank(
        exp1, output_dir, output_name="tv3_exp1_kappa_concave_vs_rank.png",
        title_suffix="participation ratio (κ-directed RSTR vs VAE baseline)",
    )
    plot_exp1_curvature_rank(
        exp1,
        output_dir,
        rank_col="log_effective_feature_rank_pca",
        rank_label="log effective rank (PCA)",
        output_name="tv3_exp1_kappa_concave_vs_pca_rank.png",
        rank_suffix="pca_rank",
        title_suffix="PCA (κ-directed RSTR vs VAE baseline)",
    )


def _plot_exp1_distill(output_dir: Path, by_experiment: dict):
    exp1 = []
    for k in ("theory_v3_z_distill_rstr", "theory_v3_vae_baseline"):
        if k in by_experiment:
            exp1.append((by_experiment[k], k))
    if len(exp1) < 2:
        raise FileNotFoundError("tv3 distill exp1 needs theory_v3_z_distill_rstr and theory_v3_vae_baseline")
    plot_exp1_curvature_rank(
        exp1, output_dir, output_name="tv3_exp1_distill_kappa_concave_vs_rank.png",
        title_suffix="participation ratio (Z* distill RSTR vs VAE baseline)",
    )
    plot_exp1_curvature_rank(
        exp1,
        output_dir,
        rank_col="log_effective_feature_rank_pca",
        rank_label="log effective rank (PCA)",
        output_name="tv3_exp1_distill_kappa_concave_vs_pca_rank.png",
        rank_suffix="pca_rank",
        title_suffix="PCA (Z* distill RSTR vs VAE baseline)",
    )


def _plot_exp2_kappa(output_dir: Path, by_experiment: dict):
    metrics = [
        "convexity_kappa_concave_mean",
        "convexity_mu_concave_mean",
        "log_effective_feature_rank_pr",
        "log_effective_feature_rank_pca",
        "mean_episode_return",
    ]
    pairs = [
        ("theory_v3_exp3_kappa_all_off", "theory_v3_exp3_kappa_lconv", "all_off", "kappa_lconv",
         "tv3_exp2_checkpoints_all_off_vs_kappa_lconv.png",
         "tv3 exp2: all off vs κ+L_conv"),
        ("theory_v3_exp3_kappa_lconv", "theory_v3_exp3_lconv_only", "kappa_lconv", "lconv_only",
         "tv3_exp2_checkpoints_kappa_lconv_vs_lconv.png",
         "tv3 exp2: κ+L_conv vs L_conv only"),
    ]
    for on_key, off_key, la, lb, fname, title in pairs:
        if on_key not in by_experiment or off_key not in by_experiment:
            raise FileNotFoundError(f"Missing {on_key} or {off_key} for tv3 exp2")
        plot_tv3_checkpoints(
            by_experiment[on_key], by_experiment[off_key], la, lb,
            output_dir, fname, metrics, title,
        )


def _plot_exp3_kappa_chain(output_dir: Path, by_experiment: dict):
    exp3 = [(by_experiment[k], k) for k in sorted(by_experiment) if "kappa_dir_chain" in k]
    if not exp3:
        raise FileNotFoundError("Missing theory_v3_kappa_dir_chain_rstr metrics")
    plot_exp4_chain(exp3, output_dir)
    chain_path = output_dir / "exp4_bounding_chain.png"
    tv3_path = output_dir / "tv3_exp3_bounding_chain.png"
    if chain_path.exists():
        chain_path.rename(tv3_path)


def _plot_exp2_distill_chain(output_dir: Path, by_experiment: dict):
    exp2 = [(by_experiment[k], k) for k in sorted(by_experiment) if "z_distill_chain" in k]
    if not exp2:
        raise FileNotFoundError("Missing theory_v3_z_distill_chain_rstr metrics")
    plot_exp4_chain(exp2, output_dir)
    chain_path = output_dir / "exp4_bounding_chain.png"
    tv3_path = output_dir / "tv3_exp2_bounding_chain.png"
    if chain_path.exists():
        chain_path.rename(tv3_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/theory_validation_v3"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/theory_validation_v3"))
    parser.add_argument("--arm", type=str, choices=["kappa", "distill"], default="kappa")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "training_loss"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.log_dir.glob("*_metrics.csv"))
    by_experiment = group_metrics_by_experiment(csv_files, load_metrics)

    if args.exp in ("all", "1"):
        if args.arm == "kappa":
            _plot_exp1_kappa(args.output_dir, by_experiment)
        else:
            _plot_exp1_distill(args.output_dir, by_experiment)

    if args.exp in ("all", "2") and args.arm == "kappa":
        _plot_exp2_kappa(args.output_dir, by_experiment)

    if args.exp in ("all", "3") and args.arm == "kappa":
        _plot_exp3_kappa_chain(args.output_dir, by_experiment)

    if args.exp in ("all", "2") and args.arm == "distill":
        _plot_exp2_distill_chain(args.output_dir, by_experiment)

    if args.exp in ("all", "training_loss"):
        pred = _training_loss_predicate(args.arm, args.exp)
        plot_tv3_training_loss(args.log_dir, args.output_dir, args.arm, pred, args.exp)

    print(f"Wrote plots to {args.output_dir} (arm={args.arm}, exp={args.exp})")


if __name__ == "__main__":
    main()
