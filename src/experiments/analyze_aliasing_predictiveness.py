"""Contribution 1: does local geometry predict held-out test return better than PR?

CSV-only mode joins logged metrics (no env). --reroll freezes weights_final, collects
an on-policy buffer on CPU, and computes probe MSE / aliasing on that batch.

Failure is eval_test_return_mean. Reward density is the pre-registered interaction.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluation.suites import EVAL_SUITES
from src.experiments.diagnose_latent_geometry import load_stack
from src.experiments.performance_runner import compute_gae_vec, make_env
from src.losses.dz_trust_region import pairwise_distance_histogram
from src.metrics.aliasing import aliasing_diagnostics
from src.metrics.collapse_probes import functional_value_probe
from src.metrics.feature_rank import compute_feature_rank_metrics
from src.utils.bisimulation_utils import encode_phi
from src.utils.normalization import PerformanceNormalizer


DENSE = ("starpilot", "fruitbot", "caveflyer", "chaser")
SPARSE = ("coinrun", "maze", "miner", "leaper")
PROCGEN_EXPS = ("exp_baseline", "exp_ctro_cnn", "exp_full", "exp_latent_nolink")
DMC_STATE_EXPS = (
    "exp_baseline",
    "exp_ctro_mlp_v2",
    "exp_latent_nolink",
    "exp_shared_baseline",
    "exp_shared_ctro",
    "exp_shared_latent_nolink",
)


def _density(task: str) -> str:
    if task in DENSE:
        return "dense"
    if task in SPARSE:
        return "sparse"
    return "other"


def _float(row: dict, key: str) -> float:
    val = row[key] if key in row and row[key] not in ("", None) else ""
    if val == "":
        return float("nan")
    return float(val)


def discover_cells(results_root: Path, suite: str) -> list[dict]:
    cells = []
    if suite == "procgen_easy":
        exps = PROCGEN_EXPS
        prefix = "procgen_easy"
        tasks = EVAL_SUITES["procgen_easy"].tasks
    elif suite == "dmcontrol_state":
        exps = DMC_STATE_EXPS
        prefix = "dmcontrol_state"
        tasks = EVAL_SUITES["dmcontrol_state"].tasks
    else:
        raise RuntimeError(f"unsupported suite {suite}")
    for exp in exps:
        for seed_dir in sorted((results_root / prefix / exp).glob("seed_*")):
            seed = int(seed_dir.name.split("_")[1])
            for task in tasks:
                run = seed_dir / task
                metrics = run / "metrics.csv"
                weights = run / "weights_final.pt"
                if not metrics.is_file():
                    continue
                with metrics.open() as f:
                    rows = list(csv.DictReader(f))
                with_eval = [r for r in rows if (r.get("eval_test_return_mean") or "").strip()]
                if not with_eval:
                    continue
                row = with_eval[-1]
                cells.append(
                    {
                        "suite": suite,
                        "exp": exp,
                        "seed": seed,
                        "task": task,
                        "density": _density(task),
                        "run_dir": str(run),
                        "has_weights": weights.is_file(),
                        "eval_test_return_mean": _float(row, "eval_test_return_mean"),
                        "eval_full_return_mean": _float(row, "eval_full_return_mean"),
                        "pr": _float(row, "feature_rank_participation_ratio"),
                        "latent_pair_p05": _float(row, "latent_pair_p05"),
                        "logged_probe_mse": _float(row, "functional_probe_mse"),
                    }
                )
    return cells


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(cells: list[dict], predictor: str, outcome: str = "eval_test_return_mean") -> dict:
    x = np.array([c[predictor] for c in cells], dtype=np.float64)
    y = np.array([c[outcome] for c in cells], dtype=np.float64)
    out = {
        "predictor": predictor,
        "n": int(np.isfinite(x).sum()),
        "spearman_all": _spearman(x, y),
    }
    for dens in ("dense", "sparse", "other"):
        sub = [c for c in cells if c["density"] == dens]
        if not sub:
            continue
        xd = np.array([c[predictor] for c in sub], dtype=np.float64)
        yd = np.array([c[outcome] for c in sub], dtype=np.float64)
        out[f"spearman_{dens}"] = _spearman(xd, yd)
        out[f"n_{dens}"] = int(np.isfinite(xd).sum())
    return out


def reroll_cell(run_dir: Path, n_steps: int) -> dict:
    stack, config, _ckpt = load_stack(run_dir, device="cpu")
    stack.critic.eval()
    stack.policy.eval()
    normalizer = PerformanceNormalizer.from_state_dict(config["normalization"])
    suite = EVAL_SUITES[config["suite"]]
    env = make_env(suite, config["task"], suite.distributions[0])
    gamma = float(config["algorithm"]["gamma"])
    gae_lambda = float(config["algorithm"]["gae_lambda"])
    discrete = env.action_space_type == "discrete"

    obs_acc = []
    rew_acc = []
    term_acc = []
    trunc_acc = []
    val_acc = []
    next_acc = []
    while len(obs_acc) < n_steps:
        obs, _ = env.reset()
        done = False
        while not done and len(obs_acc) < n_steps:
            obs_b = obs.unsqueeze(0)
            norm = normalizer.normalize_obs(obs_b)
            with torch.no_grad():
                if stack.policy_on_latent:
                    phi = encode_phi(stack.critic, norm)
                    action, _logp = stack.policy.get_action(phi.squeeze(0))
                else:
                    action, _logp = stack.policy.get_action(norm.squeeze(0))
                value = stack.critic(norm).reshape(())
            step_action = action.item() if discrete else action
            next_obs, reward, terminated, truncated, _ = env.step(step_action)
            next_b = next_obs.unsqueeze(0)
            obs_acc.append(norm.squeeze(0).cpu())
            rew_acc.append(float(reward))
            term_acc.append(bool(terminated))
            trunc_acc.append(bool(truncated))
            val_acc.append(float(value.cpu()))
            next_acc.append(normalizer.normalize_obs(next_b).squeeze(0).cpu())
            obs = next_obs
            done = bool(terminated) or bool(truncated)
    env.close()

    rew_t = torch.tensor(rew_acc, dtype=torch.float32).unsqueeze(1)
    val_t = torch.tensor(val_acc, dtype=torch.float32).unsqueeze(1)
    term_t = torch.tensor(term_acc).unsqueeze(1)
    trunc_t = torch.tensor(trunc_acc).unsqueeze(1)
    next_t = torch.stack(next_acc)
    obs_t = torch.stack(obs_acc)
    with torch.no_grad():
        next_values = stack.critic(next_t).reshape(-1, 1)
    _adv, returns = compute_gae_vec(rew_t, val_t, term_t, trunc_t, next_values, gamma, gae_lambda)
    returns_flat = returns.reshape(-1)
    with torch.no_grad():
        z = encode_phi(stack.critic, obs_t)
    rank = compute_feature_rank_metrics(z)
    pair = pairwise_distance_histogram(z)
    algo = config["algorithm"]
    alpha = float(algo["alpha_sep"]) if "alpha_sep" in algo else 1.0
    alias = aliasing_diagnostics(z, returns_flat, alpha)
    probe = functional_value_probe(z, returns_flat)
    return {
        "reroll_probe_mse": probe["functional_probe_mse"],
        "reroll_probe_nmse": probe["functional_probe_nmse"],
        "reroll_pr": rank["feature_rank_participation_ratio"],
        "reroll_p05": pair["p05"],
        "reroll_alias_rate": alias["alias_rate"],
        "reroll_alias_soft": alias["alias_soft"],
        "reroll_alias_hinge_mean": alias["alias_hinge_mean"],
        "reroll_alias_value_spearman": alias["alias_value_spearman"],
        "reroll_s_z": alias["alias_s_z"],
        "reroll_skipped": alias["alias_skipped"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results/aliasing_predictiveness",
    )
    parser.add_argument("--suite", type=str, nargs="+", default=["procgen_easy", "dmcontrol_state"])
    parser.add_argument("--exps", type=str, nargs="+", default=None)
    parser.add_argument("--reroll", action="store_true", default=False)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-cells", type=int, default=None)
    args = parser.parse_args()
    if args.device != "cpu":
        raise SystemExit("analyze_aliasing_predictiveness must run on CPU")

    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    for suite in args.suite:
        found = discover_cells(root, suite)
        if args.exps is not None:
            found = [c for c in found if c["exp"] in args.exps]
        cells.extend(found)
    if args.max_cells is not None:
        cells = cells[: args.max_cells]

    if args.reroll:
        for i, cell in enumerate(cells):
            if not cell["has_weights"]:
                raise RuntimeError(f"missing weights_final.pt: {cell['run_dir']}")
            print(f"reroll {i+1}/{len(cells)} {cell['run_dir']}", flush=True)
            extra = reroll_cell(Path(cell["run_dir"]), args.n_steps)
            cell.update(extra)
            pd.DataFrame(cells).to_csv(out / "cells.csv", index=False)

    df = pd.DataFrame(cells)
    df.to_csv(out / "cells.csv", index=False)

    predictors = ["pr", "latent_pair_p05", "logged_probe_mse"]
    if args.reroll:
        predictors.extend(
            [
                "reroll_probe_mse",
                "reroll_probe_nmse",
                "reroll_p05",
                "reroll_alias_rate",
                "reroll_alias_soft",
                "reroll_alias_hinge_mean",
                "reroll_alias_value_spearman",
                "reroll_pr",
            ]
        )
    summaries = []
    for suite in args.suite:
        sub = [c for c in cells if c["suite"] == suite]
        for pred in predictors:
            summaries.append({"suite": suite, **summarize(sub, pred)})
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n")
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"wrote {out / 'cells.csv'} n={len(cells)}", flush=True)


if __name__ == "__main__":
    main()
