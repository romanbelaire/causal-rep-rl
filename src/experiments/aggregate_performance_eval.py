"""
Aggregate performance evaluation results into a summary table.

Reads performance_eval_metrics.csv files and prints mean return per task,
split by full vs test distribution. Means/stds are averaged across seeds
(each seed CSV under the eval root).
"""

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


def aggregate_eval_dir(eval_root: Path) -> dict[str, dict[str, dict[str, float]]]:
    """
    Returns nested dict: task -> distribution -> {mean, std}.

    `mean` is the cross-seed mean of per-seed return means.
    `std` is the cross-seed standard deviation of those means (seed variability).
    """
    # task -> distribution -> list of per-seed means
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

    results: dict[str, dict[str, dict[str, float]]] = {}
    for task, dists in seed_means.items():
        for distribution, values in dists.items():
            arr = np.asarray(values, dtype=np.float64)
            results.setdefault(task, {})[distribution] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)),
                "n_seeds": float(len(arr)),
            }
    return results


def format_table(results: dict[str, dict[str, dict[str, float]]]) -> str:
    tasks = sorted(results.keys())
    distributions = sorted({d for task in results.values() for d in task})

    header = ["task"] + [f"{d}_return" for d in distributions]
    lines = ["\t".join(header)]

    for task in tasks:
        row = [task]
        for dist in distributions:
            stats = results[task].get(dist, {})
            mean = stats.get("mean", float("nan"))
            std = stats.get("std", float("nan"))
            row.append(f"{mean:.2f} ± {std:.2f}")
        lines.append("\t".join(row))

    if len(tasks) > 1:
        lines.append("")
        for dist in distributions:
            means = [results[t][dist]["mean"] for t in tasks if dist in results[t]]
            lines.append(f"mean_{dist}: {np.mean(means):.2f} (across {len(means)} tasks)")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Aggregate performance eval CSV results")
    parser.add_argument("eval_root", type=str, help="Root directory containing eval outputs")
    parser.add_argument("--output", type=str, default=None, help="Write table to file")
    args = parser.parse_args()

    results = aggregate_eval_dir(Path(args.eval_root))
    if not results:
        raise FileNotFoundError(f"No performance_eval_metrics.csv found under {args.eval_root}")

    table = format_table(results)
    print(table)

    if args.output:
        Path(args.output).write_text(table + "\n")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
