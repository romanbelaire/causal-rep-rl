#!/usr/bin/env python3
"""Phase 0 P0.3 operational-radius selection from metrics.csv (CPU only).

Selection criteria follow Experiment-guide.md P0.3. Return is never used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LAMBDA_LO = 1e-4
LAMBDA_HI = 1e4
# Treat "at upper bound" as within relative/absolute tolerance of hi.
LAMBDA_HI_FRAC = 0.99
LAMBDA_LO_MULT = 1.01
LATE_FRAC = 0.2
SEEDS = (42, 43, 44, 45, 46)

DEFAULT_ARMS = {
    "dual_tight": {"exp": "exp_dzb_dual_tight", "eta": 0.05, "adapt": True},
    "dual_medium": {"exp": "exp_dzb_dual_medium", "eta": 0.07, "adapt": True},
    "dual_loose": {"exp": "exp_dzb_fixed", "eta": 0.14, "adapt": True},
    "penalty": {"exp": "exp_dzb_penalty", "eta": 0.14, "adapt": False},
    "off": {"exp": "exp_dzb_off", "eta": None, "adapt": False},
}


def _run_dir(results_root: Path, suite: str, exp: str, seed: int, task: str) -> Path:
    return results_root / suite / exp / f"seed_{seed}" / task


def _load_metrics(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty metrics: {path}")
    return df


def _late_mask(n: int) -> np.ndarray:
    start = int(np.floor((1.0 - LATE_FRAC) * n))
    m = np.zeros(n, dtype=bool)
    m[start:] = True
    return m


def score_seed(df: pd.DataFrame, adapt: bool) -> dict:
    n = len(df)
    late = _late_mask(n)
    finite_cols = ["D_Z", "lambda_dz", "dz_eta_sq", "own_encoder_grad", "own_actor_grad"]
    for c in finite_cols:
        if c not in df.columns:
            raise KeyError(f"missing column {c} in metrics")
    finite = bool(np.isfinite(df[finite_cols].to_numpy(dtype=float)).all())

    lam = df["lambda_dz"].to_numpy(dtype=float)
    upper_sat = float(np.mean(lam >= LAMBDA_HI * LAMBDA_HI_FRAC))
    above_floor = float(np.mean(lam > LAMBDA_LO * LAMBDA_LO_MULT))

    dz = df["D_Z"].to_numpy(dtype=float)
    eta_sq = df["dz_eta_sq"].to_numpy(dtype=float)
    ratio = np.where(eta_sq > 0, dz / eta_sq, np.nan)
    late_ratio = ratio[late]
    late_ratio = late_ratio[np.isfinite(late_ratio)]
    median_late_ratio = float(np.median(late_ratio)) if late_ratio.size else float("nan")

    enc = df["own_encoder_grad"].to_numpy(dtype=float)
    act = df["own_actor_grad"].to_numpy(dtype=float)
    # Geometry ownership proxy: encoder grad dominates actor+value when much larger.
    val = df["own_value_grad"].to_numpy(dtype=float)
    task = act + val
    geom_dom = float(np.mean((enc > 2.0 * task) & np.isfinite(enc) & np.isfinite(task)))

    # Clear training failure: eval returns all nan, or D_Z/lambda explode nonfinite.
    eval_col = "eval_full_return_mean"
    if eval_col in df.columns:
        ev = df[eval_col].to_numpy(dtype=float)
        # Only checkpoints that logged eval.
        ev_valid = ev[np.isfinite(ev)]
        train_ok = bool(ev_valid.size > 0)
    else:
        train_ok = True

    upper_ok = upper_sat <= 0.10
    floor_ok = (0.25 <= above_floor <= 0.75) if adapt else True
    ratio_ok = bool(np.isfinite(median_late_ratio) and 0.5 <= median_late_ratio <= 2.0)
    geom_ok = geom_dom < 0.5
    pass_seed = bool(finite and train_ok and upper_ok and floor_ok and ratio_ok and geom_ok)

    return {
        "n_rows": n,
        "finite": finite,
        "train_ok": train_ok,
        "lambda_upper_sat_rate": upper_sat,
        "lambda_above_floor_rate": above_floor,
        "median_late_dz_over_eta_sq": median_late_ratio,
        "geom_grad_dom_rate": geom_dom,
        "upper_ok": upper_ok,
        "floor_ok": floor_ok,
        "ratio_ok": ratio_ok,
        "geom_ok": geom_ok,
        "pass": pass_seed,
    }


def score_arm(
    results_root: Path,
    suite: str,
    task: str,
    arm_name: str,
    meta: dict,
) -> dict:
    exp = meta["exp"]
    adapt = bool(meta["adapt"])
    seed_scores = {}
    missing = []
    for seed in SEEDS:
        mpath = _run_dir(results_root, suite, exp, seed, task) / "metrics.csv"
        if not mpath.is_file():
            missing.append(seed)
            continue
        seed_scores[seed] = score_seed(_load_metrics(mpath), adapt=adapt)

    n_done = len(seed_scores)
    n_pass = sum(1 for s in seed_scores.values() if s["pass"])
    if n_done == 0:
        status = "missing"
        arm_pass = False
        agg = {}
    else:
        keys = [
            "lambda_upper_sat_rate",
            "lambda_above_floor_rate",
            "median_late_dz_over_eta_sq",
            "geom_grad_dom_rate",
        ]
        agg = {
            k: float(np.nanmean([seed_scores[s][k] for s in seed_scores]))
            for k in keys
        }
        # Arm passes if majority of finished seeds pass and no missing for dual arms.
        arm_pass = n_pass >= max(1, (n_done + 1) // 2) and (
            n_done == len(SEEDS) or arm_name in ("penalty", "off")
        )
        # Stricter for dual selection: require all five seeds present.
        if adapt and arm_name.startswith("dual"):
            arm_pass = n_pass >= 3 and n_done == len(SEEDS) and all(
                seed_scores[s]["finite"] and seed_scores[s]["train_ok"] for s in seed_scores
            )
            # Also require aggregate criteria.
            arm_pass = arm_pass and (
                agg["lambda_upper_sat_rate"] <= 0.10
                and (0.25 <= agg["lambda_above_floor_rate"] <= 0.75)
                and (0.5 <= agg["median_late_dz_over_eta_sq"] <= 2.0)
                and agg["geom_grad_dom_rate"] < 0.5
            )
        status = "complete" if n_done == len(SEEDS) else "partial"

    return {
        "arm": arm_name,
        "exp": exp,
        "eta": meta["eta"],
        "adapt": adapt,
        "status": status,
        "missing_seeds": missing,
        "n_done": n_done,
        "n_pass": n_pass,
        "aggregate": agg,
        "seeds": {str(k): v for k, v in seed_scores.items()},
        "pass": arm_pass,
    }


def select_radius(arm_results: list[dict]) -> dict:
    """Prefer dual arms that pass, tighter first among passers; else fixed penalty."""
    dual_order = ["dual_tight", "dual_medium", "dual_loose"]
    by_name = {r["arm"]: r for r in arm_results}
    passers = [name for name in dual_order if by_name.get(name, {}).get("pass")]
    if passers:
        # Prefer medium if both tight and medium pass (operational band often mid);
        # guide: choose using criteria only — among passers pick the one with
        # late ratio closest to 1.0 and lowest upper saturation.
        def key(name: str):
            a = by_name[name]["aggregate"]
            return (
                abs(a["median_late_dz_over_eta_sq"] - 1.0),
                a["lambda_upper_sat_rate"],
                a["geom_grad_dom_rate"],
            )

        chosen = min(passers, key=key)
        return {
            "mode": "dual",
            "arm": chosen,
            "eta": by_name[chosen]["eta"],
            "exp": by_name[chosen]["exp"],
            "reason": (
                f"dual arm {chosen} passes P0.3 among {passers}; "
                "chosen by late D_Z/η² nearest 1 then lowest upper-sat"
            ),
        }

    pending = [
        name
        for name in dual_order
        if by_name.get(name, {}).get("status") in ("missing", "partial")
    ]
    if pending:
        return {
            "mode": "pending",
            "arm": None,
            "eta": None,
            "exp": None,
            "reason": f"dual arms incomplete: {pending}; do not freeze η yet",
            "pending_arms": pending,
        }

    # All dual complete but none pass → fixed penalty fallback.
    pen = by_name.get("penalty")
    if pen and pen["n_done"] > 0:
        return {
            "mode": "fixed_penalty",
            "arm": "penalty",
            "eta": pen["eta"],
            "exp": pen["exp"],
            "lambda_dz": 1.0,
            "dz_adapt": False,
            "reason": (
                "no dual radius satisfied P0.3; prefer fixed penalty "
                "(Experiment-guide P0.3 fallback)"
            ),
        }
    raise RuntimeError("no dual passers and no penalty arm available")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        type=Path,
        default=Path("results"),
    )
    p.add_argument("--suite", default="dmcontrol_pixels")
    p.add_argument("--task", default="cartpole-swingup")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/dmcontrol_pixels/phase0_radius_selection.json"),
    )
    args = p.parse_args()

    arm_results = [
        score_arm(args.results_root, args.suite, args.task, name, meta)
        for name, meta in DEFAULT_ARMS.items()
    ]
    selection = select_radius(arm_results)
    payload = {"arms": arm_results, "selection": selection}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(selection, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
