"""Compute η from λ=0 calibration metrics (CPU).

Default: first 20% of logged D_Z after the first update (guide §5).
η² = p25(D_Z), η rounded to two significant digits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _round_2sig(x: float) -> float:
    if x <= 0 or not math.isfinite(x):
        raise RuntimeError(f"cannot round non-positive η={x}")
    exp = math.floor(math.log10(x))
    return round(x / (10**exp), 1) * (10**exp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Run directory containing metrics.csv",
    )
    parser.add_argument(
        "--early-frac",
        type=float,
        default=0.2,
        help="Use the first this fraction of post-first-update rows (default 0.2).",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Optional hard floor on epoch before early window (default 0).",
    )
    parser.add_argument(
        "--metric-col",
        type=str,
        default="D_Z",
        help="Column name for D_Z in metrics.csv",
    )
    args = parser.parse_args()

    if args.early_frac <= 0 or args.early_frac > 1:
        raise RuntimeError(f"early_frac must be in (0, 1], got {args.early_frac}")

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics.csv: {metrics_path}")

    df = pd.read_csv(metrics_path)
    if args.metric_col not in df.columns:
        raise KeyError(f"{args.metric_col} not in {metrics_path}; cols={list(df.columns)}")

    epoch_col = None
    for cand in ("training_epoch", "epoch", "update"):
        if cand in df.columns:
            epoch_col = cand
            break
    if epoch_col is None:
        raise KeyError(
            f"No epoch column in {metrics_path}; need one of "
            "training_epoch/epoch/update"
        )

    # Drop pre-update / empty D_Z if any; require finite.
    df = df[np.isfinite(df[args.metric_col].to_numpy(dtype=np.float64))].copy()
    if df.empty:
        raise RuntimeError("No finite D_Z rows")
    df = df[df[epoch_col] >= args.warmup_epochs]
    if df.empty:
        raise RuntimeError(
            f"No rows with {epoch_col} >= {args.warmup_epochs} "
            f"(max={pd.read_csv(metrics_path)[epoch_col].max()})"
        )

    n = len(df)
    n_early = max(1, int(math.ceil(args.early_frac * n)))
    early = df.iloc[:n_early]
    vals = early[args.metric_col].to_numpy(dtype=np.float64)
    p25 = float(np.quantile(vals, 0.25))
    if p25 < 0:
        raise RuntimeError(f"p25(D_Z)={p25} < 0")
    eta_raw = float(np.sqrt(p25))
    eta = float(_round_2sig(eta_raw))
    eta_sq = eta * eta
    print(f"n_total={n} n_early={n_early} early_frac={args.early_frac}")
    print(f"D_Z_p25={p25:.8g}")
    print(f"D_Z_p50={float(np.quantile(vals, 0.50)):.8g}")
    print(f"D_Z_mean={float(vals.mean()):.8g}")
    print(f"eta_raw={eta_raw:.8g}")
    print(f"eta_dz={eta:.8g}")
    print(f"eta_sq={eta_sq:.8g}")
    print(f"tight_eta_sq={0.5 * eta_sq:.8g} loose_eta_sq={2.0 * eta_sq:.8g}")
    out = run_dir / "eta_calib.txt"
    out.write_text(f"{eta:.8g}\n")
    (run_dir / "eta_sq_calib.txt").write_text(f"{eta_sq:.8g}\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
