"""
Post-hoc plots for theory validation v4 (CPU-only).

MICo arm: exp1 (curvature vs rank), exp4 (bounding chain).
DBC arm: exp1, exp4 (parallel secondary comparison).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.theory_validation.aggregate_seeds import (
    aggregate_series,
    group_metrics_by_experiment,
)
from src.theory_validation.plot_validation_v2 import (
    COLORS,
    load_metrics,
    plot_exp1_curvature_rank,
    plot_exp4_chain,
)

INTERVENTION_CSV_GLOB = "*_intervention_loss.csv"

MICO_EXP1_KEYS = [
    "theory_v4_vae_baseline_mico",
    "theory_v4_mico_alpha001",
    "theory_v4_mico_alpha01",
    "theory_v4_mico_alpha05",
]

DBC_EXP1_KEYS = [
    "theory_v4_vae_baseline_dbc",
    "theory_v4_dbc_alpha001",
    "theory_v4_dbc_alpha01",
    "theory_v4_dbc_alpha05",
]


def _load_intervention_by_experiment(log_dir: Path) -> dict[str, list[pd.DataFrame]]:
    files = sorted(log_dir.glob(INTERVENTION_CSV_GLOB))
    if not files:
        raise FileNotFoundError(f"No intervention loss CSVs in {log_dir}")
    by_exp: dict[str, list[pd.DataFrame]] = {}
    for p in files:
        name = p.name.replace("_intervention_loss.csv", "")
        from src.theory_validation.aggregate_seeds import strip_seed_suffix
        base = strip_seed_suffix(name)
        df = load_metrics(p)
        by_exp.setdefault(base, []).append(df)
    return by_exp


def _plot_exp1_arm(
    output_dir: Path,
    by_experiment: dict,
    keys: list[str],
    arm: str,
):
    exp1 = []
    for k in keys:
        if k not in by_experiment:
            raise FileNotFoundError(f"Missing metrics for {k} (tv4 {arm} exp1)")
        exp1.append((by_experiment[k], k))
    plot_exp1_curvature_rank(
        exp1,
        output_dir,
        output_name=f"tv4_exp1_{arm}_kappa_concave_vs_rank.png",
        title_suffix=f"participation ratio ({arm} arm)",
    )
    plot_exp1_curvature_rank(
        exp1,
        output_dir,
        rank_col="log_effective_feature_rank_pca",
        rank_label="log effective rank (PCA)",
        output_name=f"tv4_exp1_{arm}_kappa_concave_vs_pca_rank.png",
        rank_suffix="pca_rank",
        title_suffix=f"PCA ({arm} arm)",
    )


def _plot_chain_arm(output_dir: Path, by_experiment: dict, chain_key: str, arm: str):
    if chain_key not in by_experiment:
        raise FileNotFoundError(f"Missing {chain_key} for tv4 {arm} chain")
    exp = [(by_experiment[chain_key], chain_key)]
    plot_exp4_chain(exp, output_dir)
    src = output_dir / "exp4_bounding_chain.png"
    dst = output_dir / f"tv4_exp4_{arm}_bounding_chain.png"
    if src.exists():
        src.rename(dst)


def plot_training_loss(
    log_dir: Path,
    output_dir: Path,
    arm: str,
    loss_col: str,
    name_predicate,
):
    by_exp = _load_intervention_by_experiment(log_dir)
    selected = [(by_exp[k], k) for k in sorted(by_exp) if name_predicate(k)]
    if not selected:
        raise FileNotFoundError(f"No intervention runs for {arm} training loss")

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (dfs, name) in enumerate(selected):
        color = COLORS[i % len(COLORS)]
        steps, mean, std = aggregate_series(dfs, loss_col)
        if len(steps) < 2:
            raise ValueError(f"{name}: need >=2 {loss_col} points, got {len(steps)}")
        ax.plot(steps, mean, label=name, color=color)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.2, color=color)

    ax.set_xlabel("step")
    ax.set_ylabel(loss_col)
    ax.set_title(f"tv4 {arm}: bisimulation training loss", fontweight="bold")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / f"tv4_{arm}_training_loss.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/theory_validation_v4"))
    parser.add_argument("--output-dir", type=Path, default=Path("plots/theory_validation_v4"))
    parser.add_argument("--arm", type=str, choices=["mico", "dbc"], default="mico")
    parser.add_argument(
        "--exp",
        type=str,
        default="all",
        choices=["all", "1", "4", "training_loss"],
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(args.log_dir.glob("*_metrics.csv"))
    by_experiment = group_metrics_by_experiment(csv_files, load_metrics)

    if args.exp in ("all", "1"):
        if args.arm == "mico":
            _plot_exp1_arm(args.output_dir, by_experiment, MICO_EXP1_KEYS, "mico")
        else:
            _plot_exp1_arm(args.output_dir, by_experiment, DBC_EXP1_KEYS, "dbc")

    if args.exp in ("all", "4"):
        chain_key = (
            "theory_v4_mico_chain" if args.arm == "mico" else "theory_v4_dbc_chain"
        )
        _plot_chain_arm(args.output_dir, by_experiment, chain_key, args.arm)

    if args.exp in ("all", "training_loss"):
        if args.arm == "mico":
            plot_training_loss(
                args.log_dir,
                args.output_dir,
                "mico",
                "train_mico_loss",
                lambda k: k.startswith("theory_v4_mico") or k == "theory_v4_vae_baseline_mico",
            )
        else:
            plot_training_loss(
                args.log_dir,
                args.output_dir,
                "dbc",
                "train_dbc_loss",
                lambda k: k.startswith("theory_v4_dbc") or k == "theory_v4_vae_baseline_dbc",
            )

    print(f"Wrote plots to {args.output_dir} (arm={args.arm}, exp={args.exp})")


if __name__ == "__main__":
    main()
