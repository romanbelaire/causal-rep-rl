"""Overlay pairwise distance histograms for D_Z three-arm runs (CPU)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ARMS = (
    ("exp_dz3_off", "off (λ=0)", "#1f77b4"),
    ("exp_dz3_fixed", "fixed η", "#d62728"),
    ("exp_dz3_adapt", "adapt η (ablation)", "#2ca02c"),
)


def _load_hist(path: Path) -> dict:
    d = np.load(path)
    return {
        "counts": d["counts"].astype(np.float64),
        "edges": d["bin_edges"].astype(np.float64),
        "p05": float(d["p05"]),
        "p50": float(d["p50"]),
        "p95": float(d["p95"]),
        "epoch": int(d["epoch"]),
        "step": int(d["step"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results/dmcontrol_pixels",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/plots/dz3_pairwise_distance_overlays.png",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default="cartpole-swingup")
    args = parser.parse_args()

    root = Path(args.results_root)
    # Collect epochs present in all arms
    per_arm: dict[str, dict[int, Path]] = {}
    for exp, _, _ in ARMS:
        hdir = root / exp / f"seed_{args.seed}" / args.task / "pairwise_distance_hist"
        if not hdir.is_dir():
            raise FileNotFoundError(hdir)
        files = sorted(hdir.glob("epoch_*.npz"))
        if not files:
            raise FileNotFoundError(f"no histograms in {hdir}")
        per_arm[exp] = {}
        for f in files:
            ep = int(f.name.split("_")[1])
            per_arm[exp][ep] = f

    epochs = sorted(set.intersection(*(set(m) for m in per_arm.values())))
    if not epochs:
        raise RuntimeError("no common checkpoint epochs across arms")

    n = len(epochs)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ep in zip(axes, epochs):
        for exp, label, color in ARMS:
            h = _load_hist(per_arm[exp][ep])
            counts = h["counts"]
            edges = h["edges"]
            widths = np.diff(edges)
            dens = counts / (counts.sum() * widths + 1e-12)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.plot(
                centers,
                dens,
                color=color,
                label=f"{label}  p05={h['p05']:.3g}",
                lw=1.8,
            )
        ax.set_title(f"epoch {ep}")
        ax.set_xlabel("pairwise ||z_i − z_j||")
        ax.set_xlim(left=0)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"D_Z three-arm pairwise distance histograms ({args.task}, seed {args.seed})",
        y=1.02,
    )
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Wrote {out}")
    for ep in epochs:
        print(f"epoch {ep}:")
        for exp, label, _ in ARMS:
            h = _load_hist(per_arm[exp][ep])
            print(
                f"  {label:22s} p05={h['p05']:.6e} p50={h['p50']:.4f} p95={h['p95']:.4f}"
            )


if __name__ == "__main__":
    main()
