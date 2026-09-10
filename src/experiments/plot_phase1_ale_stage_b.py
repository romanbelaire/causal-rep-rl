"""Produce Stage B figures 1–5 after the method matrix finishes (CPU-only)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.experiments.analyze_phase1_ale_stage_b import METHODS, SEEDS, EPOCHS, analyze_task


def _load_eval_curve(root: Path, method: str, ep: int, seed: int, task: str) -> pd.DataFrame:
    run = root / f"exp_p1b_{method}_ep{ep}" / f"seed_{seed}" / task
    df = pd.read_csv(run / "metrics.csv")
    return df.dropna(subset=["eval_full_return_mean"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results/ale"))
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args()
    out = args.out_dir or (args.results_root / "figures" / args.task.replace("/", "_"))
    out.mkdir(parents=True, exist_ok=True)

    report = analyze_task(args.results_root, args.task)
    (out / "stage_b_summary.json").write_text(
        __import__("json").dumps(report, indent=2)
    )

    # Fig 1: PPO return curves standard vs stress
    fig, ax = plt.subplots(figsize=(7, 4))
    for ep, color in ((4, "C0"), (16, "C1")):
        curves = []
        for seed in SEEDS:
            df = _load_eval_curve(args.results_root, "ppo", ep, seed, args.task)
            curves.append(df[["steps", "eval_full_return_mean"]].set_index("steps"))
        # Align on union of steps via outer join mean
        joined = curves[0]
        for c in curves[1:]:
            joined = joined.join(c, how="outer", rsuffix="_x")
        mean = joined.mean(axis=1)
        ax.plot(mean.index, mean.values, color=color, label=f"PPO ep{ep}")
    ax.set_xlabel("env steps")
    ax.set_ylabel("eval return")
    ax.set_title("Fig1: PPO return standard vs stress")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig1_ppo_return.png", dpi=150)
    plt.close(fig)

    # Fig 2: cumulant NMSE
    fig, ax = plt.subplots(figsize=(7, 4))
    for ep, color in ((4, "C0"), (16, "C1")):
        ys = []
        for seed in SEEDS:
            df = _load_eval_curve(args.results_root, "ppo", ep, seed, args.task)
            ys.append(df["diag_cumulant_nmse"].to_numpy())
        # pad by min length
        m = min(len(y) for y in ys)
        arr = np.stack([y[:m] for y in ys], axis=0)
        ax.plot(arr.mean(0), color=color, label=f"ep{ep}")
        ax.fill_between(range(m), arr.min(0), arr.max(0), color=color, alpha=0.15)
    ax.set_xlabel("eval index")
    ax.set_ylabel("diag cumulant NMSE")
    ax.set_title("Fig2: cumulant NMSE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fig2_cumulant_nmse.png", dpi=150)
    plt.close(fig)

    # Fig 3: geometry panel at final window (C_0.01, dormant, stable rank)
    keys = ["diag_actor_C_0p01", "diag_actor_dormant_frac", "diag_actor_stable_rank"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, key in zip(axes, keys):
        for ep, xpos in ((4, 0), (16, 1)):
            vals = []
            for seed in SEEDS:
                df = _load_eval_curve(args.results_root, "ppo", ep, seed, args.task)
                vals.append(float(df[key].tail(max(1, len(df) // 10)).mean()))
            ax.scatter([xpos] * len(vals), vals, alpha=0.8)
            ax.plot([xpos], [float(np.mean(vals))], "ks")
        ax.set_xticks([0, 1], ["ep4", "ep16"])
        ax.set_title(key)
    fig.suptitle("Fig3: actor geometry")
    fig.tight_layout()
    fig.savefig(out / "fig3_geometry.png", dpi=150)
    plt.close(fig)

    # Fig 4: method comparison under stress
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = np.arange(len(METHODS))
    means = [report["cells"][f"{m}_ep16"]["return_mean"] for m in METHODS]
    stds = [report["cells"][f"{m}_ep16"]["return_std"] for m in METHODS]
    ax.bar(xs, means, yerr=stds, capsize=4)
    ax.set_xticks(xs, METHODS)
    ax.set_ylabel("final-window return")
    ax.set_title("Fig4: methods under stress (ep16)")
    fig.tight_layout()
    fig.savefig(out / "fig4_methods_stress.png", dpi=150)
    plt.close(fig)

    # Fig 5: interaction I
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar([0], [report["interaction_I"]], color="C2")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks([0], ["I"])
    ax.set_title("Fig5: interaction I (LTRO−PPO)")
    fig.tight_layout()
    fig.savefig(out / "fig5_interaction.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures to {out}")


if __name__ == "__main__":
    main()
