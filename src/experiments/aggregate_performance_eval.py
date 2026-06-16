"""
Aggregate performance evaluation results into a summary table.

Reads performance_eval_metrics.csv files and prints mean return per task,
split by full vs test distribution.
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
    """
    results: dict[str, dict[str, dict[str, float]]] = {}

    for csv_path in sorted(eval_root.rglob("performance_eval_metrics.csv")):
        row = load_eval_row(csv_path)
        for key, value in row.items():
            parsed = parse_task_distribution(key)
            if parsed is None:
                continue
            task, distribution, stat = parsed
            results.setdefault(task, {}).setdefault(distribution, {})[stat] = value

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
