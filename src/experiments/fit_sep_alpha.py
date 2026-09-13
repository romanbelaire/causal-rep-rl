"""Pick α from a λ_sep=0 run so hinge-active fraction is nearest 25%.

Uses early_frac of logged alias_hinge_active_frac_a{0p25,0p5,1p0} columns.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from src.losses.separation import ALPHA_SEP_CANDIDATES


def _col(alpha: float) -> str:
    return "alias_hinge_active_frac_a" + str(alpha).replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--early-frac", type=float, default=0.2)
    parser.add_argument("--target-frac", type=float, default=0.25)
    args = parser.parse_args()

    path = Path(args.run_dir) / "metrics.csv"
    df = pd.read_csv(path)
    if len(df) < 5:
        raise RuntimeError(f"need enough metric rows in {path}, got {len(df)}")
    n = max(1, int(math.floor(len(df) * args.early_frac)))
    early = df.iloc[:n]
    scores = {}
    for a in ALPHA_SEP_CANDIDATES:
        col = _col(a)
        if col not in early.columns:
            raise RuntimeError(f"missing {col} in {path}")
        mean_frac = float(early[col].mean())
        scores[a] = mean_frac
    best = min(scores, key=lambda a: abs(scores[a] - args.target_frac))
    print(
        f"early_n={n} target={args.target_frac} "
        + " ".join(f"a{a}={scores[a]:.4f}" for a in ALPHA_SEP_CANDIDATES)
        + f" chosen_alpha={best}",
        flush=True,
    )


if __name__ == "__main__":
    main()
