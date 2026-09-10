#!/usr/bin/env python3
"""Phase 1 collapse gate analysis for PPO epoch sweep (CPU).

Loads exp_p1_ppo_ep{4,8,16,32}, applies predeclared thresholds, reports collapse
incidence/onset and whether representation failure precedes return drop.
Refuses to mix freeze IDs when config.json records freeze_id.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = (42, 43, 44, 45, 46)
EPOCHS = (4, 8, 16, 32)
PROBE = "functional_probe_mse"
C001 = "diag_dz_C_0p01"
DORM = "diag_dormant_unit_frac"
RET = "eval_full_return_mean"


def _load_thresholds(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("thresholds_status") != "fitted" or doc.get("thresholds") is None:
        raise RuntimeError(
            f"thresholds not fitted yet: {path} (run fit_phase1_collapse_thresholds)"
        )
    return doc["thresholds"]


def _eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    need = [PROBE, C001, DORM, RET, "step"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"missing {c}")
    m = df[need].notna().all(axis=1)
    sub = df.loc[m].sort_values("step")
    if sub.empty:
        raise ValueError("no eval rows with collapse metrics")
    return sub


def _collapse_mask(df: pd.DataFrame, thr: dict) -> np.ndarray:
    func = df[PROBE].to_numpy(dtype=float) > thr["functional_probe_mse_threshold"]
    c_hi = df[C001].to_numpy(dtype=float) > thr["diag_dz_C_0p01_threshold"]
    d_hi = df[DORM].to_numpy(dtype=float) > thr["diag_dormant_unit_frac_threshold"]
    geom = c_hi | d_hi
    return func & geom


def _onset_step(steps: np.ndarray, collapse: np.ndarray) -> float | None:
    # Two consecutive True.
    for i in range(len(collapse) - 1):
        if collapse[i] and collapse[i + 1]:
            return float(steps[i])
    return None


def _return_drop_step(df: pd.DataFrame) -> float | None:
    """First step where return falls below early median − 1σ of early window."""
    steps = df["step"].to_numpy(dtype=float)
    rets = df[RET].to_numpy(dtype=float)
    n = len(rets)
    early_n = max(2, n // 5)
    early = rets[:early_n]
    thresh = float(early.mean() - early.std(ddof=1))
    for i in range(early_n, n):
        if rets[i] < thresh:
            return float(steps[i])
    return None


def analyze_run(path: Path, thr: dict) -> dict:
    df = _eval_rows(pd.read_csv(path))
    collapse = _collapse_mask(df, thr)
    steps = df["step"].to_numpy(dtype=float)
    onset = _onset_step(steps, collapse)
    ret_drop = _return_drop_step(df)
    late = df.tail(max(1, len(df) // 5))
    return {
        "collapsed": onset is not None,
        "collapse_onset_step": onset,
        "return_drop_step": ret_drop,
        "rep_before_or_with_return": (
            None
            if onset is None or ret_drop is None
            else bool(onset <= ret_drop)
        ),
        "late_mean_return": float(late[RET].mean()),
        "late_mean_probe": float(late[PROBE].mean()),
        "late_mean_C_0p01": float(late[C001].mean()),
        "late_mean_dormant": float(late[DORM].mean()),
        "n_eval": int(len(df)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=Path("results"))
    p.add_argument("--suite", default="dmcontrol_pixels")
    p.add_argument("--task", default="cartpole-swingup")
    p.add_argument(
        "--thresholds",
        type=Path,
        default=Path("configs/frozen/phase1_collapse_thresholds.json"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/dmcontrol_pixels/phase1_collapse_gate.json"),
    )
    p.add_argument("--expected-freeze-id", default="ltro_phase0_v1")
    args = p.parse_args()

    thr = _load_thresholds(args.thresholds)
    by_epoch: dict = {}
    for ep in EPOCHS:
        exp = f"exp_p1v2_ppo_ep{ep}"
        seed_results = {}
        missing = []
        for seed in SEEDS:
            run = args.results_root / args.suite / exp / f"seed_{seed}" / args.task
            mpath = run / "metrics.csv"
            cpath = run / "config.json"
            if not mpath.is_file():
                missing.append(seed)
                continue
            if cpath.is_file():
                cfg = json.loads(cpath.read_text())
                fid = cfg.get("freeze_id") or cfg.get("algorithm", {}).get("freeze_id")
                if fid is not None and fid != args.expected_freeze_id:
                    raise RuntimeError(
                        f"freeze_id mismatch in {cpath}: {fid} != {args.expected_freeze_id}"
                    )
            seed_results[str(seed)] = analyze_run(mpath, thr)
        n = len(seed_results)
        n_collapse = sum(1 for r in seed_results.values() if r["collapsed"])
        by_epoch[str(ep)] = {
            "exp": exp,
            "n_done": n,
            "missing_seeds": missing,
            "collapse_incidence": (n_collapse / n) if n else None,
            "n_collapse": n_collapse,
            "seeds": seed_results,
        }

    # Gate: at some epoch >= 8, >= half of finished seeds collapse.
    gate_pass = False
    gate_epoch = None
    for ep in EPOCHS:
        if ep < 8:
            continue
        cell = by_epoch[str(ep)]
        if cell["n_done"] < 3:
            continue
        inc = cell["collapse_incidence"]
        if inc is not None and inc >= 0.5:
            gate_pass = True
            gate_epoch = ep
            break

    ordering_ok = True
    for ep, cell in by_epoch.items():
        for r in cell["seeds"].values():
            if r["rep_before_or_with_return"] is False:
                ordering_ok = False

    payload = {
        "thresholds": thr,
        "by_epoch": by_epoch,
        "gate": {
            "pass": gate_pass and ordering_ok,
            "collapse_reproducible_epoch": gate_epoch,
            "representation_precedes_return": ordering_ok,
            "rule": (
                "pass if some epoch>=8 has collapse incidence >= 0.5 "
                "and no seed shows return drop strictly before representation collapse"
            ),
            "next_if_pass": "add PFO / CTRO / LTRO arms with frozen eta",
            "next_if_fail": "do not scale method matrix; revise stress dial or metrics",
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["gate"], indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
