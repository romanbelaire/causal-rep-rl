"""Stage B analysis: interaction I and preliminary matrix summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
EPOCHS = (4, 16)
METHODS = ("ppo", "pfo", "ctro", "ltro")


def _final_return(run: Path) -> float:
    df = pd.read_csv(run / "metrics.csv")
    evals = df.dropna(subset=["eval_full_return_mean"])
    n = max(1, int(len(evals) * 0.1))
    return float(evals.tail(n)["eval_full_return_mean"].mean())


def _final_nmse(run: Path) -> float:
    df = pd.read_csv(run / "metrics.csv")
    evals = df.dropna(subset=["diag_cumulant_nmse"])
    n = max(1, int(len(evals) * 0.1))
    return float(evals.tail(n)["diag_cumulant_nmse"].mean())


def analyze_task(root: Path, task: str) -> dict:
    table = {}
    for method in METHODS:
        for ep in EPOCHS:
            rets, nmses = [], []
            for seed in SEEDS:
                run = root / f"exp_p1b_{method}_ep{ep}" / f"seed_{seed}" / task
                if not (run / "metrics.csv").is_file():
                    raise FileNotFoundError(run)
                rets.append(_final_return(run))
                nmses.append(_final_nmse(run))
            table[f"{method}_ep{ep}"] = {
                "return_mean": float(np.mean(rets)),
                "return_std": float(np.std(rets)),
                "cumulant_nmse_mean": float(np.mean(nmses)),
                "seeds": {str(s): {"return": r, "nmse": n} for s, r, n in zip(SEEDS, rets, nmses)},
            }

    def R(method: str, ep: int) -> float:
        return table[f"{method}_ep{ep}"]["return_mean"]

    interaction = (R("ltro", 16) - R("ppo", 16)) - (R("ltro", 4) - R("ppo", 4))
    return {
        "task": task,
        "preliminary": True,
        "note": "3 seeds; stress dial 16 not 32; no superiority claim without PFO unit checks + full matrix",
        "interaction_I": interaction,
        "cells": table,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=Path, default=Path("results/ale"))
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    report = analyze_task(args.results_root, args.task)
    out = args.out or Path("results/ale") / f"phase1_ale_stage_b_{args.task.replace('/', '_')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
