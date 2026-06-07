"""
Bootstrap confidence intervals on Δmetrics across seeds at fixed checkpoints (Exp 3).
CPU-only.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CHECKPOINTS = [1_000_000, 3_000_000, 6_000_000]
METRICS = [
    "convexity_kappa_concave_mean",
    "convexity_mu_concave_mean",
    "log_effective_feature_rank_pr",
    "feature_rank_pca",
    "log_effective_feature_rank_pca",
    "mean_episode_return",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, alpha: float = 0.05) -> tuple[float, float, float]:
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    if len(values) == 1:
        v = float(values[0])
        return v, v, v
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boots.append(sample.mean())
    boots = np.array(boots)
    return float(values.mean()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def value_at_checkpoint(df: pd.DataFrame, step: int, col: str) -> float:
    if col not in df.columns:
        return float("nan")
    idx = (df["step"] - step).abs().idxmin()
    return float(df.loc[idx, col])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/theory_validation_v2"))
    parser.add_argument("--on-prefix", type=str, default="exp3_rstr_repr_on")
    parser.add_argument("--off-prefix", type=str, default="exp3_rstr_repr_off")
    parser.add_argument("--output", type=Path, default=Path("outputs/theory_validation_v2/exp3_bootstrap_ci.csv"))
    args = parser.parse_args()

    on_files = sorted(args.log_dir.glob(f"{args.on_prefix}*_metrics.csv"))
    off_files = sorted(args.log_dir.glob(f"{args.off_prefix}*_metrics.csv"))
    if not on_files or not off_files:
        raise FileNotFoundError("Need exp3 repr_on and repr_off metrics CSVs")

    rows = []
    for cp in CHECKPOINTS:
        for metric in METRICS:
            on_vals = []
            for p in on_files:
                df = pd.read_csv(p)
                on_vals.append(value_at_checkpoint(df, cp, metric))
            off_vals = []
            for p in off_files:
                df = pd.read_csv(p)
                off_vals.append(value_at_checkpoint(df, cp, metric))
            deltas = np.array(on_vals) - np.array(off_vals)
            mean_d, lo, hi = bootstrap_ci(deltas)
            rows.append({
                "checkpoint": cp,
                "metric": metric,
                "delta_mean": mean_d,
                "delta_ci_lo": lo,
                "delta_ci_hi": hi,
                "n_seeds_on": len(on_vals),
                "n_seeds_off": len(off_vals),
            })

    out = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.to_string(index=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
