"""Fit Run 4 rate models: mu_pl degradation vs eta (sqrt vs linear). CPU only."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_final_mu(run_dir: Path, min_valid_fraction: float) -> tuple[float, float]:
    path = run_dir / "metrics.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty metrics: {path}")
    last = rows[-1]
    # Prefer scale-invariant diagnostic (T1.1); fall back for older runs.
    key = "mu_pl_tilde_q05_conditional"
    if key not in last or last[key] == "":
        key = "mu_pl_q05_conditional"
    if key not in last or last[key] == "":
        raise KeyError(f"mu_pl_tilde_q05_conditional / mu_pl_q05_conditional missing in {path}")
    if "mu_pl_valid_fraction" not in last or last["mu_pl_valid_fraction"] == "":
        raise KeyError(f"mu_pl_valid_fraction missing in {path}")
    valid_fraction = float(last["mu_pl_valid_fraction"])
    if valid_fraction < min_valid_fraction:
        raise RuntimeError(
            f"Conditional mu_PL support {valid_fraction:.4f} < "
            f"{min_valid_fraction:.4f} in {path}"
        )
    return float(last[key]), valid_fraction


def poly_fit_sse(x: np.ndarray, y: np.ndarray, features: np.ndarray) -> tuple[float, float]:
    """Least squares y ~ a + b * feature; return (b, sse)."""
    A = np.column_stack([np.ones_like(features), features])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    sse = float(np.sum((y - pred) ** 2))
    return float(coef[1]), sse


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit D_Z rate test models")
    parser.add_argument(
        "--results-root",
        type=str,
        required=True,
        help="Directory containing exp_dz_run4_eta*/seed_*/task/",
    )
    parser.add_argument("--task", type=str, default="cartpole-swingup")
    parser.add_argument(
        "--etas",
        type=float,
        nargs="+",
        default=[0.01, 0.02, 0.05, 0.1, 0.2, 0.3],
    )
    parser.add_argument(
        "--min-valid-fraction",
        type=float,
        default=0.1,
        help="Minimum fraction of states with f(s)>tau required for conditional mu_PL.",
    )
    args = parser.parse_args()
    if args.min_valid_fraction <= 0.0 or args.min_valid_fraction > 1.0:
        raise ValueError(
            f"--min-valid-fraction must be in (0, 1], got {args.min_valid_fraction}"
        )
    root = Path(args.results_root)

    etas = []
    mus = []
    valid_fractions = []
    for eta in args.etas:
        exp = f"exp_dz_run4_eta{eta}"
        seed_dirs = sorted((root / exp).glob(f"seed_*/{args.task}"))
        if not seed_dirs:
            # also try suite nested path
            seed_dirs = sorted(root.glob(f"**/exp_dz_run4_eta{eta}/seed_*/{args.task}"))
        if not seed_dirs:
            raise FileNotFoundError(f"No runs for eta={eta} under {root}")
        loaded = [load_final_mu(d, args.min_valid_fraction) for d in seed_dirs]
        vals = [x[0] for x in loaded]
        coverage = [x[1] for x in loaded]
        etas.append(eta)
        mus.append(float(np.mean(vals)))
        valid_fractions.append(float(np.mean(coverage)))

    etas_a = np.asarray(etas, dtype=np.float64)
    mus_a = np.asarray(mus, dtype=np.float64)
    # degradation relative to smallest-eta (tightest constraint → expect higher mu)
    mu0 = mus_a[np.argmin(etas_a)]
    deg = mu0 - mus_a  # positive if looser eta worsens mu

    b_sqrt, sse_sqrt = poly_fit_sse(etas_a, deg, np.sqrt(etas_a))
    b_lin, sse_lin = poly_fit_sse(etas_a, deg, etas_a)

    winner = "sqrt" if sse_sqrt <= sse_lin else "linear"
    out = {
        "etas": etas,
        "mu_pl_q05_conditional_mean": mus,
        "mu_pl_valid_fraction_mean": valid_fractions,
        "min_valid_fraction_required": args.min_valid_fraction,
        "degradation_from_tightest": deg.tolist(),
        "sqrt_slope": b_sqrt,
        "sqrt_sse": sse_sqrt,
        "linear_slope": b_lin,
        "linear_sse": sse_lin,
        "preferred_model": winner,
        "theorem2_prediction": "sqrt",
        "matches_theorem2": winner == "sqrt",
    }
    print(json.dumps(out, indent=2))
    if not out["matches_theorem2"]:
        print(
            "NOTE: linear fit wins — value-drift cross-term (§5) is a likely cause of mismatch."
        )


if __name__ == "__main__":
    main()
