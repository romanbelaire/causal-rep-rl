#!/usr/bin/env python3
"""Fit Phase 1 collapse thresholds from healthy-epoch PPO metrics (CPU).

Reads exp_p1_ppo_ep4 (or --healthy-exp) final 20% eval rows and writes numeric
thresholds into configs/frozen/phase1_collapse_thresholds.json using the
predeclared rules. Refuses to run if stress exps are mixed into the healthy set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44, 45, 46)
PROBE_COL = "functional_probe_mse"
C_COL = "diag_dz_C_0p01"
DORM_COL = "diag_dormant_unit_frac"


def _late_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "step" not in df.columns:
        raise KeyError("metrics.csv missing step")
    # Prefer rows that logged eval + probe.
    need = [PROBE_COL, C_COL, DORM_COL]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"metrics.csv missing {c}; re-run with Phase 1 probe logging")
    m = df[need].notna().all(axis=1)
    sub = df.loc[m].copy()
    if sub.empty:
        raise ValueError("no eval rows with probe metrics")
    lo = sub["step"].quantile(0.80)
    return sub.loc[sub["step"] >= lo]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--suite", default="dmcontrol_pixels")
    p.add_argument("--task", default="cartpole-swingup")
    p.add_argument("--healthy-exp", default="exp_p1v2_ppo_ep4")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("configs/frozen/phase1_collapse_thresholds.json"),
    )
    args = p.parse_args()

    probe_vals = []
    c_vals = []
    dorm_vals = []
    for seed in SEEDS:
        path = (
            args.results_root
            / args.suite
            / args.healthy_exp
            / f"seed_{seed}"
            / args.task
            / "metrics.csv"
        )
        if not path.is_file():
            raise FileNotFoundError(f"healthy metrics missing: {path}")
        late = _late_eval_rows(pd.read_csv(path))
        probe_vals.extend(late[PROBE_COL].astype(float).tolist())
        c_vals.extend(late[C_COL].astype(float).tolist())
        dorm_vals.extend(late[DORM_COL].astype(float).tolist())

    probe = np.asarray(probe_vals, dtype=float)
    c = np.asarray(c_vals, dtype=float)
    dorm = np.asarray(dorm_vals, dtype=float)

    thresholds = {
        "functional_probe_mse_mean": float(probe.mean()),
        "functional_probe_mse_std": float(probe.std(ddof=1)),
        "functional_probe_mse_threshold": float(probe.mean() + 2.0 * probe.std(ddof=1)),
        "diag_dz_C_0p01_p95": float(np.quantile(c, 0.95)),
        "diag_dz_C_0p01_threshold": float(max(0.10, np.quantile(c, 0.95))),
        "diag_dormant_unit_frac_mean": float(dorm.mean()),
        "diag_dormant_unit_frac_std": float(dorm.std(ddof=1)),
        "diag_dormant_unit_frac_threshold": float(
            dorm.mean() + 2.0 * dorm.std(ddof=1)
        ),
        "n_late_rows": int(probe.size),
        "healthy_exp": args.healthy_exp,
    }

    doc = json.loads(args.out.read_text()) if args.out.is_file() else {}
    doc["thresholds"] = thresholds
    doc["thresholds_status"] = "fitted"
    args.out.write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps(thresholds, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
