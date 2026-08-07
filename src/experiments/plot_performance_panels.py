"""Performance panels for dmcontrol_state / dmcontrol_pixels / procgen_easy.

Three-way comparison per task (or two-way for stress):
  BASELINE       plain PPO (no shared Z, no value link)
  LATENT_NOLINK  CTRO stack (shared Z) with alpha=beta=0 (no MICo/PL)
  CTRO           full CTRO (value link on)

Panel presets (--suite):
  dmcontrol_state         default default-HP three-way
  dmcontrol_state_shared  Optuna t8 matched HPs (exp_shared_*)
  dmcontrol_state_stress  on-Z stress (exp_stress_*)
  dmcontrol_pixels        same-task pixel suite
  procgen_easy            Procgen three-way

Panel A: final_bars.png       return / mu_PL / PR side-by-side
Panel B: mu_pl_vs_return.png  mu_PL vs return, colored by method
Panel C: dual_axis.png        return + mu_PL over training, one subplot per cell

Layout read: results/{results_suite}/{exp_name}/seed_{seed}/{task}/metrics.csv
Performance train CSVs log mean_episode_return (train return), not eval_return_mean,
so return is taken from mean_episode_return.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RETURN_METRIC = "mean_episode_return"
MU_METRIC = "mu_pl_q05"
PR_METRIC = "feature_rank_participation_ratio"

# Positive-control exp name differs by suite / recipe.
# Keys are panel presets; results_suite is the on-disk results/{suite}/ folder.
PANEL_PRESETS = {
    "dmcontrol_state": {
        "results_suite": "dmcontrol_state",
        "cells": [
            ("BASELINE", "exp_baseline"),
            ("LATENT_NOLINK", "exp_latent_nolink"),
            ("CTRO", "exp_ctro_mlp"),
        ],
    },
    "dmcontrol_state_shared": {
        "results_suite": "dmcontrol_state",
        "cells": [
            ("BASELINE", "exp_shared_baseline"),
            ("LATENT_NOLINK", "exp_shared_latent_nolink"),
            ("CTRO", "exp_shared_ctro"),
        ],
    },
    "dmcontrol_state_stress": {
        "results_suite": "dmcontrol_state",
        "cells": [
            ("LATENT_NOLINK", "exp_stress_latent_nolink"),
            ("CTRO", "exp_stress_ctro"),
        ],
    },
    "dmcontrol_pixels": {
        "results_suite": "dmcontrol_pixels",
        "cells": [
            ("BASELINE", "exp_baseline"),
            ("LATENT_NOLINK", "exp_latent_nolink"),
            ("CTRO", "exp_ctro"),
        ],
    },
    "procgen_easy": {
        "results_suite": "procgen_easy",
        "cells": [
            ("BASELINE", "exp_baseline"),
            ("LATENT_NOLINK", "exp_latent_nolink"),
            ("CTRO", "exp_full"),
        ],
    },
}

# Backward-compatible alias used by older call sites.
SUITE_CELLS = {k: v["cells"] for k, v in PANEL_PRESETS.items()}


COLORS = {
    "BASELINE": "#d62728",
    "LATENT_NOLINK": "#ff7f0e",
    "CTRO": "#1f77b4",
}

DEFAULT_SEEDS = [42, 43, 44]


def load_seed_metrics(
    results_root: Path, exp_path: str, seed: int, task: str
) -> pd.DataFrame:
    path = results_root / exp_path / f"seed_{seed}" / task / "metrics.csv"
    df = pd.read_csv(path)
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    for col in df.columns:
        if col != "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("step").reset_index(drop=True)


def aggregate_metric_series(
    results_root: Path,
    exp_path: str,
    metric: str,
    seeds: list[int],
    task: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = []
    for seed in seeds:
        df = load_seed_metrics(results_root, exp_path, seed, task)
        if metric not in df.columns:
            raise KeyError(f"{metric} missing in {exp_path}/seed_{seed}/{task}")
        sub = df[["step", metric]].dropna(subset=[metric])
        series.append(sub.set_index("step")[metric])

    combined = pd.concat(series, axis=1)
    steps = combined.index.to_numpy(dtype=float)
    mean = combined.mean(axis=1).to_numpy()
    sem = combined.sem(axis=1, ddof=1).to_numpy()
    sem = np.nan_to_num(sem, nan=0.0)
    return steps, mean, sem


def _final_values(
    results_root: Path, exp_path: str, seeds: list[int], task: str, metric: str
) -> list[float]:
    vals = []
    for seed in seeds:
        df = load_seed_metrics(results_root, exp_path, seed, task)
        sub = df[metric].dropna()
        if sub.empty:
            raise KeyError(f"{metric} all-NaN in {exp_path}/seed_{seed}/{task}")
        vals.append(float(sub.iloc[-1]))
    return vals


def _sem(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1) / np.sqrt(len(values)))


def build_final_table(
    results_root: Path,
    cells: list[tuple[str, str]],
    seeds: list[int],
    task: str,
) -> dict[str, dict]:
    table = {}
    for name, exp_path in cells:
        returns = _final_values(results_root, exp_path, seeds, task, RETURN_METRIC)
        mu_pl = _final_values(results_root, exp_path, seeds, task, MU_METRIC)
        pr = _final_values(results_root, exp_path, seeds, task, PR_METRIC)
        table[name] = {
            "return_mean": float(np.mean(returns)),
            "return_sem": _sem(returns),
            "mu_pl_mean": float(np.mean(mu_pl)),
            "pr_mean": float(np.mean(pr)),
        }
    return table


def plot_final_bars(table: dict[str, dict], output_path: Path, task: str) -> None:
    names = list(table.keys())
    x = np.arange(len(names))
    colors = [COLORS[n] for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(
        f"Panel A: final metrics — {task} (mean over seeds)",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].bar(
        x,
        [table[n]["return_mean"] for n in names],
        yerr=[table[n]["return_sem"] for n in names],
        capsize=4,
        color=colors,
    )
    axes[0].set_ylabel("Mean episode return")

    axes[1].bar(x, [table[n]["mu_pl_mean"] for n in names], color=colors)
    axes[1].set_ylabel(r"$\mu_{PL}$ (q05)")

    axes[2].bar(x, [table[n]["pr_mean"] for n in names], color=colors)
    axes[2].set_ylabel("Participation ratio")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def collect_scatter_points(
    results_root: Path,
    cells: list[tuple[str, str]],
    seeds: list[int],
    task: str,
) -> pd.DataFrame:
    rows = []
    for name, exp_path in cells:
        for seed in seeds:
            df = load_seed_metrics(results_root, exp_path, seed, task)
            sub = df[["step", RETURN_METRIC, MU_METRIC]].dropna(
                subset=[RETURN_METRIC, MU_METRIC]
            )
            for _, row in sub.iterrows():
                rows.append(
                    {
                        "method": name,
                        "seed": seed,
                        "step": float(row["step"]),
                        "return": float(row[RETURN_METRIC]),
                        "mu_pl_q05": float(row[MU_METRIC]),
                    }
                )
    return pd.DataFrame(rows)


def plot_mu_pl_vs_return(
    results_root: Path,
    cells: list[tuple[str, str]],
    output_path: Path,
    seeds: list[int],
    task: str,
) -> None:
    points = collect_scatter_points(results_root, cells, seeds, task)
    if points.empty:
        raise ValueError(f"No points with mu_PL for Panel B ({task})")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for name, _ in cells:
        sub = points[points["method"] == name]
        color = COLORS[name]
        final_steps = sub.groupby("seed")["step"].transform("max")
        mid = sub[sub["step"] < final_steps]
        final = sub[sub["step"] == final_steps]
        if not mid.empty:
            ax.scatter(
                mid["mu_pl_q05"],
                mid["return"],
                s=28,
                facecolors="none",
                edgecolors=color,
                linewidths=1.0,
                alpha=0.5,
            )
        ax.scatter(
            final["mu_pl_q05"],
            final["return"],
            s=70,
            color=color,
            edgecolors="black",
            linewidths=0.6,
            alpha=0.9,
            label=name,
            zorder=3,
        )

    x = points["mu_pl_q05"].to_numpy()
    y = points["return"].to_numpy()
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(x.min(), x.max(), 50)
        ax.plot(
            x_line,
            slope * x_line + intercept,
            color="0.35",
            linestyle="--",
            linewidth=1.5,
            label=f"trend (slope={slope:.2f})",
        )

    ax.set_xlabel(r"$\mu_{PL}$ (q05)")
    ax.set_ylabel("Mean episode return")
    ax.set_title(
        f"Panel B: value-geometry health vs return — {task}",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=True, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dual_axis(
    results_root: Path,
    cells: list[tuple[str, str]],
    output_path: Path,
    seeds: list[int],
    task: str,
) -> None:
    """Panel C: dual-axis return + mu_PL over training, one subplot per cell."""
    series = []
    for name, exp_path in cells:
        steps_r, mean_r, sem_r = aggregate_metric_series(
            results_root, exp_path, RETURN_METRIC, seeds, task
        )
        steps_m, mean_m, sem_m = aggregate_metric_series(
            results_root, exp_path, MU_METRIC, seeds, task
        )
        series.append((name, steps_r, mean_r, sem_r, steps_m, mean_m, sem_m))

    return_max = max(float((mr + sr).max()) for _, _, mr, sr, _, _, _ in series)
    mu_max = max(float((mm + sm).max()) for _, _, _, _, _, mm, sm in series)

    fig, axes = plt.subplots(1, len(cells), figsize=(6 * len(cells), 4.5))
    if len(cells) == 1:
        axes = [axes]
    fig.suptitle(
        f"Panel C: return and $\\mu_{{PL}}$ over training — {task}",
        fontsize=13,
        fontweight="bold",
    )

    for ax, (name, steps_r, mean_r, sem_r, steps_m, mean_m, sem_m) in zip(
        axes, series
    ):
        color = COLORS[name]
        x_r = steps_r / 1e6
        x_m = steps_m / 1e6

        ax.plot(x_r, mean_r, color=color, linewidth=2, label="Return")
        ax.fill_between(x_r, mean_r - sem_r, mean_r + sem_r, color=color, alpha=0.2)
        ax.set_xlabel("Environment steps (millions)")
        ax.set_ylabel("Mean episode return", color=color)
        ax.tick_params(axis="y", labelcolor=color)
        ax.set_ylim(min(0.0, float(mean_r.min())), return_max * 1.05)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(
            x_m, mean_m, color="0.25", linewidth=2, linestyle="--", label=r"$\mu_{PL}$"
        )
        ax2.fill_between(x_m, mean_m - sem_m, mean_m + sem_m, color="0.25", alpha=0.15)
        ax2.set_ylabel(r"$\mu_{PL}$ (q05)", color="0.25")
        ax2.tick_params(axis="y", labelcolor="0.25")
        ax2.set_ylim(0.0, mu_max * 1.05)

        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_table(table: dict[str, dict], output_path: Path, task: str) -> None:
    lines = [
        f"Negative-control comparison — {task}",
        f"{'Cell':<20} {'Return':>16} {'mu_PL (q05)':>14} {'PR':>10}",
        "-" * 62,
    ]
    for name, stats in table.items():
        ret = f"{stats['return_mean']:.4f} ± {stats['return_sem']:.4f}"
        lines.append(
            f"{name:<20} {ret:>16} {stats['mu_pl_mean']:>14.4f} {stats['pr_mean']:>10.4f}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Negative-control Panel A/B/C for a performance suite task"
    )
    parser.add_argument(
        "--suite", type=str, required=True, choices=list(PANEL_PRESETS.keys())
    )
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()

    preset = PANEL_PRESETS[args.suite]
    results_root = Path(args.results_root) / preset["results_suite"]
    output_dir = Path(
        args.output_dir or f"plots/{args.suite}/{args.task}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = preset["cells"]
    seeds = args.seeds
    task = args.task

    table = build_final_table(results_root, cells, seeds, task)
    plot_final_bars(table, output_dir / "final_bars.png", task)
    print(f"Wrote {output_dir / 'final_bars.png'}")

    plot_mu_pl_vs_return(
        results_root, cells, output_dir / "mu_pl_vs_return.png", seeds, task
    )
    print(f"Wrote {output_dir / 'mu_pl_vs_return.png'}")

    plot_dual_axis(results_root, cells, output_dir / "dual_axis.png", seeds, task)
    print(f"Wrote {output_dir / 'dual_axis.png'}")

    write_table(table, output_dir / "table.txt", task)
    print(f"Wrote {output_dir / 'table.txt'}")


if __name__ == "__main__":
    main()
