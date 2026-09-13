"""Leading-indicator stretch on Phoenix Stage A logs.

Not a substitute for Procgen held-out-level predictiveness.
Outcome is eval_full_return_mean (in-distribution collapse under epoch stress).
Predictors: on-policy cumulant NMSE, C_0.01, actor participation ratio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEEDS = (42, 43, 44)
EPOCHS = (4, 16)
GAMES = ("ALE/Phoenix-v5", "ALE/NameThisGame-v5")
PREDICTORS = (
    "onpolicy_cumulant_nmse",
    "diag_cumulant_nmse",
    "onpolicy_actor_C_0p01",
    "onpolicy_actor_feature_rank_pr",
    "critic_feature_rank_participation_ratio",
)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def _eval_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "eval_full_return_mean" not in df.columns:
        raise RuntimeError("missing eval_full_return_mean")
    ev = df.dropna(subset=["eval_full_return_mean"]).copy()
    if len(ev) < 5:
        raise RuntimeError(f"need >=5 eval rows, got {len(ev)}")
    return ev.reset_index(drop=True)


def _lead_spearman(pred: np.ndarray, ret: np.ndarray, lag: int) -> float:
    if lag <= 0:
        return _spearman(pred, ret)
    if len(pred) <= lag:
        return float("nan")
    return _spearman(pred[:-lag], ret[lag:])


def _onset(ev: pd.DataFrame, ret_thresh: float = 5.0) -> dict:
    """First eval where return stays below thresh; whether NMSE already exceeded 10."""
    ret = ev["eval_full_return_mean"].to_numpy()
    below = ret < ret_thresh
    onset = None
    for i in range(len(below)):
        if below[i]:
            onset = i
            break
    out = {"onset_eval_idx": onset, "n_eval": int(len(ev))}
    if onset is None:
        out["nmse_before_onset"] = float("nan")
        out["pr_before_onset"] = float("nan")
        return out
    pre = ev.iloc[: max(onset, 1)]
    if "onpolicy_cumulant_nmse" in pre.columns:
        out["nmse_before_onset"] = float(pre["onpolicy_cumulant_nmse"].iloc[-1])
    if "onpolicy_actor_feature_rank_pr" in pre.columns:
        out["pr_before_onset"] = float(pre["onpolicy_actor_feature_rank_pr"].iloc[-1])
    out["return_at_onset"] = float(ret[onset])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results/ale",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/ocean/projects/cis260223p/rbelaire/causal-rep-rl/results/aliasing_predictiveness",
    )
    args = parser.parse_args()
    root = Path(args.results_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    onsets = []
    for ep in EPOCHS:
        for game in GAMES:
            for seed in SEEDS:
                run = root / f"exp_p1a_ppo_ep{ep}" / f"seed_{seed}" / game
                path = run / "metrics.csv"
                if not path.is_file():
                    raise FileNotFoundError(path)
                ev = _eval_frame(pd.read_csv(path))
                rec = {
                    "epochs": ep,
                    "game": game,
                    "seed": seed,
                    "n_eval": int(len(ev)),
                    "final_return": float(ev["eval_full_return_mean"].iloc[-1]),
                }
                ret = ev["eval_full_return_mean"].to_numpy()
                for pred_name in PREDICTORS:
                    if pred_name not in ev.columns:
                        continue
                    x = ev[pred_name].to_numpy(dtype=np.float64)
                    rec[f"spearman_{pred_name}_lag0"] = _lead_spearman(x, ret, 0)
                    rec[f"spearman_{pred_name}_lag5"] = _lead_spearman(x, ret, 5)
                    rec[f"spearman_{pred_name}_lag10"] = _lead_spearman(x, ret, 10)
                rows.append(rec)
                onset = _onset(ev)
                onset.update({"epochs": ep, "game": game, "seed": seed})
                onsets.append(onset)

    df = pd.DataFrame(rows)
    df.to_csv(out / "phoenix_leading.csv", index=False)
    pd.DataFrame(onsets).to_csv(out / "phoenix_onset.csv", index=False)

    # Pool Phoenix ep16 (the collapse regime) for a one-line verdict.
    phoenix16 = df[(df["game"] == "ALE/Phoenix-v5") & (df["epochs"] == 16)]
    verdict = {
        "phoenix_ep16_n_seeds": int(len(phoenix16)),
        "mean_final_return": float(phoenix16["final_return"].mean()),
    }
    for pred_name in PREDICTORS:
        for lag in (0, 5, 10):
            col = f"spearman_{pred_name}_lag{lag}"
            if col in phoenix16.columns:
                verdict[f"mean_{col}"] = float(phoenix16[col].mean())
    (out / "phoenix_leading_summary.json").write_text(json.dumps(verdict, indent=2) + "\n")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {out / 'phoenix_leading.csv'}", flush=True)


if __name__ == "__main__":
    main()
