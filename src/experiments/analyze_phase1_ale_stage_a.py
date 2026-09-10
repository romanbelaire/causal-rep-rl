"""Stage A gate: qualify ALE games under Moalla epoch stress (4 vs 16)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
EPOCHS = (4, 16)
GAMES = ("ALE/Phoenix-v5", "ALE/NameThisGame-v5")


def _run_dir(root: Path, exp: str, seed: int, task: str) -> Path:
    return root / exp / f"seed_{seed}" / task


def _load_metrics(run: Path) -> pd.DataFrame:
    path = run / "metrics.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _final_window(df: pd.DataFrame, frac: float = 0.1) -> pd.DataFrame:
    if "eval_full_return_mean" not in df.columns:
        raise RuntimeError("missing eval_full_return_mean")
    evals = df.dropna(subset=["eval_full_return_mean"])
    if len(evals) < 5:
        raise RuntimeError(f"need >=5 eval rows, got {len(evals)}")
    n = max(1, int(len(evals) * frac))
    return evals.tail(n)


def _seed_summary(df: pd.DataFrame) -> dict:
    w = _final_window(df)
    out = {
        "return_mean": float(w["eval_full_return_mean"].mean()),
        "return_std": float(w["eval_full_return_mean"].std(ddof=0)),
    }
    for key in (
        "diag_cumulant_nmse",
        "onpolicy_cumulant_nmse",
        "diag_actor_C_0p01",
        "diag_actor_dormant_frac",
        "diag_actor_stable_rank",
        "diag_actor_feature_rank_pca",
        "diag_dz_C_0p01",
    ):
        if key in w.columns:
            out[key] = float(w[key].mean())
    return out


def qualify_game(root: Path, task: str) -> dict:
    """Qualify when stress is worse on return + cumulant NMSE + ≥2 diagnostics in ≥2/3 seeds."""
    std = {}
    stress = {}
    for seed in SEEDS:
        std_df = _load_metrics(_run_dir(root, "exp_p1a_ppo_ep4", seed, task))
        st_df = _load_metrics(_run_dir(root, "exp_p1a_ppo_ep16", seed, task))
        std[seed] = _seed_summary(std_df)
        stress[seed] = _seed_summary(st_df)

    std_returns = [std[s]["return_mean"] for s in SEEDS]
    if float(np.mean(std_returns)) < 1.0:
        return {
            "task": task,
            "qualified": False,
            "reason": "standard PPO did not learn adequately (mean final return < 1)",
            "standard": std,
            "stress": stress,
        }

    worse_return = 0
    worse_nmse = 0
    worse_diag = 0
    for seed in SEEDS:
        r_ok = stress[seed]["return_mean"] < std[seed]["return_mean"]
        nmse_key = "diag_cumulant_nmse"
        if nmse_key not in stress[seed] or nmse_key not in std[seed]:
            raise RuntimeError(f"missing {nmse_key} for seed {seed}")
        n_ok = stress[seed][nmse_key] > std[seed][nmse_key]
        diag_keys = [
            "diag_actor_C_0p01",
            "diag_actor_dormant_frac",
            "diag_actor_stable_rank",
            "diag_actor_feature_rank_pca",
        ]
        support = 0
        for k in diag_keys:
            if k not in stress[seed] or k not in std[seed]:
                continue
            # Higher C_0.01 / dormant = worse; lower stable/pca rank = worse
            if k in ("diag_actor_C_0p01", "diag_actor_dormant_frac"):
                if stress[seed][k] > std[seed][k]:
                    support += 1
            else:
                if stress[seed][k] < std[seed][k]:
                    support += 1
        if r_ok:
            worse_return += 1
        if n_ok:
            worse_nmse += 1
        if support >= 2:
            worse_diag += 1

    # Pattern in ≥2 of 3 stressed seeds for joint criteria.
    seed_hits = 0
    for seed in SEEDS:
        r_ok = stress[seed]["return_mean"] < std[seed]["return_mean"]
        n_ok = stress[seed]["diag_cumulant_nmse"] > std[seed]["diag_cumulant_nmse"]
        support = 0
        for k in (
            "diag_actor_C_0p01",
            "diag_actor_dormant_frac",
            "diag_actor_stable_rank",
            "diag_actor_feature_rank_pca",
        ):
            if k not in stress[seed]:
                continue
            if k in ("diag_actor_C_0p01", "diag_actor_dormant_frac"):
                if stress[seed][k] > std[seed][k]:
                    support += 1
            else:
                if stress[seed][k] < std[seed][k]:
                    support += 1
        if r_ok and n_ok and support >= 2:
            seed_hits += 1

    qualified = seed_hits >= 2
    return {
        "task": task,
        "qualified": qualified,
        "seed_hits": seed_hits,
        "worse_return_seeds": worse_return,
        "worse_nmse_seeds": worse_nmse,
        "worse_diag_seeds": worse_diag,
        "reason": (
            "stress shows worse return, worse cumulant NMSE, and >=2 supporting "
            "diagnostics in >=2/3 seeds"
            if qualified
            else "did not meet joint stress pattern in >=2/3 seeds"
        ),
        "standard": {str(k): v for k, v in std.items()},
        "stress": {str(k): v for k, v in stress.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results/ale"))
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/ale/phase1_ale_stage_a_gate.json"),
    )
    args = p.parse_args()

    report = {
        "seeds": list(SEEDS),
        "epochs": list(EPOCHS),
        "games": [],
    }
    for task in GAMES:
        try:
            report["games"].append(qualify_game(args.results_root, task))
        except FileNotFoundError as exc:
            report["games"].append(
                {"task": task, "qualified": False, "reason": f"missing run: {exc}"}
            )

    any_q = any(g.get("qualified") for g in report["games"])
    report["any_qualified"] = any_q
    report["next_if_fail"] = [
        "impl audit vs moalla_ale_protocol.md",
        "raise stress to 32 epochs or longer train",
        "layer check on actor_preactivation",
        "add another Moalla game",
        "do not return to cartpole as primary collapse env",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not any_q:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
