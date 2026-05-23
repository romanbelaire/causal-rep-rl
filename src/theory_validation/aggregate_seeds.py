"""Aggregate metrics CSVs across random seeds (mean ± std on a common step grid)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED_SUFFIX = re.compile(r"_seed\d+$")


def strip_seed_suffix(name: str) -> str:
    return SEED_SUFFIX.sub("", name)


def group_metrics_by_experiment(
    csv_paths: list[Path],
    loader,
) -> dict[str, list[pd.DataFrame]]:
    groups: dict[str, list[pd.DataFrame]] = {}
    for path in csv_paths:
        name = path.stem.replace("_metrics", "")
        base = strip_seed_suffix(name)
        groups.setdefault(base, []).append(loader(path))
    return groups


def metric_series(df: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    if col not in df.columns:
        return np.array([]), np.array([])
    mask = df[col].notna() & df["step"].notna()
    return df.loc[mask, "step"].values, df.loc[mask, col].values


def aggregate_series(
    dfs: list[pd.DataFrame],
    col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate each seed onto the union of step values, then mean/std across seeds.

    Returns:
        steps, mean, std (std is 0 when only one seed contributes at a step)
    """
    series_list: list[tuple[np.ndarray, np.ndarray]] = []
    all_steps: set[float] = set()
    for df in dfs:
        if col not in df.columns:
            continue
        steps, vals = metric_series(df, col)
        if len(steps) == 0:
            continue
        series_list.append((steps, vals))
        all_steps.update(steps.tolist())

    if not series_list:
        return np.array([]), np.array([]), np.array([])

    common_steps = np.array(sorted(all_steps))
    stacked = []
    for steps, vals in series_list:
        interp = np.interp(common_steps, steps, vals, left=np.nan, right=np.nan)
        stacked.append(interp)
    stacked = np.vstack(stacked)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    valid = ~np.isnan(mean)
    return common_steps[valid], mean[valid], std[valid]


def n_seeds(dfs: list[pd.DataFrame]) -> int:
    return len(dfs)
