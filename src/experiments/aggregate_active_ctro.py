"""Aggregate MiniGrid active-CTRO E0/E6 cells from metrics.csv (CPU)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.experiments.config import (
    ACTIVE_CTRO_MINIGRID_ARMS,
    ACTIVE_CTRO_RESULTS_MINIGRID,
    ACTIVE_CTRO_SEEDS,
)

KEYS = (
    "eval_return_mean",
    "mean_episode_return",
    "kl",
    "D_Z_inf_B",
    "pair_frac_positive",
    "pair_frac_undecided",
    "train_q_target_var_mean",
    "coverage_min_action_prob",
    "coverage_action_entropy",
    "relational_gate_accept",
)


def _read_final_row(metrics_path: Path) -> dict:
    with open(metrics_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no metrics rows in {metrics_path}")
    return rows[-1]


def _float(row: dict, key: str) -> float:
    val = row.get(key, "")
    if val is None or val == "":
        return float("nan")
    return float(val)


def _sem(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 2:
        return float("nan")
    return float(finite.std(ddof=1) / np.sqrt(finite.size))


def aggregate_arm(results_root: Path, exp_name: str, seeds: list[int]) -> dict:
    per_seed: list[dict] = []
    for seed in seeds:
        run_dir = results_root / exp_name / f"seed_{seed}"
        final = run_dir / "weights_final.pt"
        metrics_path = run_dir / "metrics.csv"
        if not final.exists():
            raise RuntimeError(f"missing finished checkpoint {final}")
        if not metrics_path.exists():
            raise RuntimeError(f"missing metrics {metrics_path}")
        row = _read_final_row(metrics_path)
        cell = {"seed": seed}
        for key in KEYS:
            cell[key] = _float(row, key)
        per_seed.append(cell)

    summary = {"exp": exp_name, "n_seeds": len(seeds), "seeds": per_seed}
    for key in KEYS:
        vals = np.array([c[key] for c in per_seed], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.nanmean(vals))
        summary[f"{key}_sem"] = _sem(vals)
    return summary


def format_table(arms: list[dict]) -> str:
    header = (
        "arm                  eval_return          train_return         "
        "kl                 D_Z_inf_B"
    )
    lines = [header]
    for arm in arms:
        lines.append(
            f"{arm['exp']:<20} "
            f"{arm['eval_return_mean_mean']:.3f}±{arm['eval_return_mean_sem']:.3f}   "
            f"{arm['mean_episode_return_mean']:.3f}±{arm['mean_episode_return_sem']:.3f}   "
            f"{arm['kl_mean']:.4f}±{arm['kl_sem']:.4f}   "
            f"{arm['D_Z_inf_B_mean']:.4f}±{arm['D_Z_inf_B_sem']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate active-CTRO MiniGrid E0/E6")
    parser.add_argument("--results-root", type=str, default=ACTIVE_CTRO_RESULTS_MINIGRID)
    parser.add_argument("--seeds", type=int, nargs="+", default=ACTIVE_CTRO_SEEDS)
    args = parser.parse_args()
    root = Path(args.results_root)
    arms = [aggregate_arm(root, name, list(args.seeds)) for name in ACTIVE_CTRO_MINIGRID_ARMS]
    table = format_table(arms)
    out_dir = root / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e0_e6_summary.json").write_text(json.dumps(arms, indent=2) + "\n")
    (out_dir / "e0_e6_summary.txt").write_text(table)
    print(table, end="")
    print(f"Wrote {out_dir / 'e0_e6_summary.txt'}")


if __name__ == "__main__":
    main()
