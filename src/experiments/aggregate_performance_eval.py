"""
Aggregate performance evaluation results with IQM + stratified bootstrap CIs.

Reads performance_eval_metrics.csv files under an eval root.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def load_eval_row(csv_path: Path) -> dict[str, float]:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, got {len(rows)}")
    return {k: float(v) for k, v in rows[0].items() if k != "step"}


def parse_task_distribution(key: str) -> tuple[str, str, str] | None:
    # eval_{task}_{distribution}_return_{mean|std}
    if not key.startswith("eval_") or "_return_" not in key:
        return None
    body, stat = key.rsplit("_return_", 1)
    parts = body[len("eval_") :].rsplit("_", 1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1], stat


def interquartile_mean(x: np.ndarray) -> float:
    """IQM: mean of observations between 25th and 75th percentiles (inclusive)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    if x.size < 4:
        return float(x.mean())
    lo, hi = np.quantile(x, [0.25, 0.75])
    mid = x[(x >= lo) & (x <= hi)]
    if mid.size == 0:
        return float(x.mean())
    return float(mid.mean())


def stratified_bootstrap_iqm_ci(
    values: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """
    Stratified bootstrap CI for IQM.

    With a single stratum (seeds), resamples with replacement over seeds.
    Returns (iqm, ci_lo, ci_hi).
    """
    rng = rng or np.random.default_rng(0)
    values = np.asarray(values, dtype=np.float64)
    point = interquartile_mean(values)
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boots = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = values[rng.integers(0, n, size=n)]
        boots[b] = interquartile_mean(sample)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def normalize_procgen_per_game(
    task_seed_returns: dict[str, list[float]],
) -> dict[str, list[float]]:
    """
    Per-game min-max normalize across seeds for that game, then keep per-task lists.
    Games with constant returns stay zero.
    """
    out: dict[str, list[float]] = {}
    for task, vals in task_seed_returns.items():
        arr = np.asarray(vals, dtype=np.float64)
        lo, hi = arr.min(), arr.max()
        if hi - lo < 1e-12:
            out[task] = [0.0] * len(vals)
        else:
            out[task] = ((arr - lo) / (hi - lo)).tolist()
    return out


def aggregate_eval_dir(
    eval_root: Path,
    n_boot: int = 2000,
    procgen_normalize: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns nested dict: task -> distribution -> {iqm, ci_lo, ci_hi, mean, std, n_seeds}.
    """
    seed_means: dict[str, dict[str, list[float]]] = {}

    csv_paths = sorted(eval_root.rglob("performance_eval_metrics.csv"))
    if not csv_paths:
        return {}

    for csv_path in csv_paths:
        row = load_eval_row(csv_path)
        for key, value in row.items():
            parsed = parse_task_distribution(key)
            if parsed is None:
                continue
            task, distribution, stat = parsed
            if stat != "mean":
                continue
            seed_means.setdefault(task, {}).setdefault(distribution, []).append(value)

    if procgen_normalize:
        for distribution in {d for t in seed_means.values() for d in t}:
            by_task = {
                task: dists[distribution]
                for task, dists in seed_means.items()
                if distribution in dists
            }
            normed = normalize_procgen_per_game(by_task)
            for task, vals in normed.items():
                seed_means[task][distribution] = vals

    rng = np.random.default_rng(0)
    results: dict[str, dict[str, dict[str, float]]] = {}
    for task, dists in seed_means.items():
        for distribution, values in dists.items():
            arr = np.asarray(values, dtype=np.float64)
            iqm, lo, hi = stratified_bootstrap_iqm_ci(arr, n_boot=n_boot, rng=rng)
            results.setdefault(task, {})[distribution] = {
                "iqm": iqm,
                "ci_lo": lo,
                "ci_hi": hi,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "n_seeds": float(len(arr)),
            }
    return results


def format_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    tasks = sorted(results.keys())
    distributions = sorted({d for task in results.values() for d in task})

    header = ["task"] + [f"{d}_return_iqm[ci]" for d in distributions]
    lines = ["\t".join(header)]

    for task in tasks:
        row = [task]
        for dist in distributions:
            stats = results[task].get(dist, {})
            iqm = stats.get("iqm", float("nan"))
            lo = stats.get("ci_lo", float("nan"))
            hi = stats.get("ci_hi", float("nan"))
            row.append(f"{iqm:.2f} [{lo:.2f}, {hi:.2f}]")
        lines.append("\t".join(row))

    if len(tasks) > 1:
        lines.append("")
        for dist in distributions:
            iqms = [results[t][dist]["iqm"] for t in tasks if dist in results[t]]
            lines.append(f"mean_iqm_{dist}: {np.mean(iqms):.2f} (across {len(iqms)} tasks)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate performance eval CSV results (IQM)")
    parser.add_argument("eval_root", type=str, help="Root directory containing eval outputs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument(
        "--procgen-normalize",
        action="store_true",
        help="Per-game normalize seed returns before IQM (Procgen protocol)",
    )
    args = parser.parse_args()
    results = aggregate_eval_dir(
        Path(args.eval_root),
        n_boot=args.n_boot,
        procgen_normalize=args.procgen_normalize,
    )
    if not results:
        print(f"No performance_eval_metrics.csv under {args.eval_root}")
        return
    print(format_table(results))


if __name__ == "__main__":
    main()
