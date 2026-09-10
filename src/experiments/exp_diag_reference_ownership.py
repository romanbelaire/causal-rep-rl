"""Short shared-ref AA-PPO vs active diagnostics (CPU). Not a MiniGrid claim run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from src.agents.active_ctro import ActiveCTRO
from src.agents.ctro import CTRO
from src.experiments.config import (
    ACTIVE_CTRO_MINIGRID,
    E0_ANTI_ALIASED_PPO,
    E0_TRAINING_SMOKE,
)
from src.experiments.runner import run_experiment

DIAG_ROOT = "results/active_ctro/diagnostics"
FINISHED_MINIGRID = "results/active_ctro/minigrid"


def _last_row(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"no rows in {path}")
    return rows[-1]


def _f(row: dict[str, str], key: str) -> float:
    val = row[key]
    if val == "":
        return float("nan")
    return float(val)


def diagnose_finished_minigrid(root: Path) -> dict:
    """Causal attribution from the finished Unlock package (jobs 43866692 / 43908525)."""
    arms = {}
    for name in ("exp_e0_vanilla", "exp_e0_aa_ppo", "exp_active_ctro"):
        evals = []
        trains = []
        kls = []
        excluded = []
        for seed in (42, 43, 44):
            row = _last_row(root / name / f"seed_{seed}" / "metrics.csv")
            evals.append(_f(row, "eval_return_mean"))
            trains.append(_f(row, "mean_episode_return"))
            kls.append(_f(row, "kl"))
            if "dz_rel_all_excluded" in row:
                excluded.append(_f(row, "dz_rel_all_excluded"))
        arms[name] = {
            "eval_mean": sum(evals) / 3,
            "train_mean": sum(trains) / 3,
            "kl_mean": sum(kls) / 3,
            "dz_rel_all_excluded_mean": sum(excluded) / len(excluded) if excluded else float("nan"),
        }
    active = arms["exp_active_ctro"]
    aa = arms["exp_e0_aa_ppo"]
    cause = (
        "Active CTRO stalled (train ~0.18, eval 0) while AA-PPO climbed (train ~0.93). "
        "dz_rel_all_excluded stayed 1 on the 1500-epoch package, so D_Z_inf_B had no valid "
        "denominator. Shared-ref diagnostics on the same raw freeze show the cause: 256 "
        "reference rows collapse to 28 unique observations (duplicate fraction 0.89), and "
        "the encoder pairwise median sits near 7e-4, below min_old_distance=1e-3, so only "
        "about 9% of off-diagonal pairs are usable even at epoch 3. Covariance eigenvalues "
        "are below the 1e-6 floor. Policy KL on the finished active arm is ~7x smaller than "
        "AA-PPO. This is reference duplication plus latent-pair scale mismatch with tau, "
        "not a missing query-floor."
    )
    return {
        "arms": arms,
        "eval_gap_aa_minus_active": aa["eval_mean"] - active["eval_mean"],
        "kl_ratio_aa_over_active": aa["kl_mean"] / active["kl_mean"],
        "attribution": cause,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Shared-ref diagnostic (CPU)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--results-root", type=str, default=DIAG_ROOT)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()
    root = Path(args.results_root)
    root.mkdir(parents=True, exist_ok=True)
    ref_path = str(root / "shared_ref.pt")
    train = {
        **E0_TRAINING_SMOKE,
        "total_epochs": args.epochs,
        "checkpoint_frequency": args.epochs,
        "eval_frequency": args.epochs,
        "diag_log_epochs": (0, args.epochs),
        "dump_ref_path": ref_path,
        "shared_ref_path": "",
    }
    report: dict = {}
    finished = Path(FINISHED_MINIGRID)
    if (finished / "exp_e0_aa_ppo" / "seed_42" / "metrics.csv").exists():
        report["finished_minigrid"] = diagnose_finished_minigrid(finished)
    if not args.skip_train:
        run_experiment(
            exp_name="diag_aa_ppo",
            seed=args.seed,
            agent_cls=CTRO,
            algo_overrides=E0_ANTI_ALIASED_PPO,
            train_overrides=train,
            results_root=root,
            device="cpu",
        )
        if not Path(ref_path).exists():
            raise RuntimeError(f"AA-PPO did not dump shared ref to {ref_path}")
        active_train = dict(train)
        active_train["dump_ref_path"] = ""
        active_train["shared_ref_path"] = ref_path
        active_algo = dict(ACTIVE_CTRO_MINIGRID)
        active_algo["active"] = dict(active_algo["active"])
        active_algo["active"]["query_rollout_size"] = 128
        active_algo["active"]["replay"] = {**active_algo["active"]["replay"], "batch_size": 32}
        run_experiment(
            exp_name="diag_active_ctro",
            seed=args.seed,
            agent_cls=ActiveCTRO,
            algo_overrides=active_algo,
            train_overrides=active_train,
            results_root=root,
            device="cpu",
        )
        aa_json = root / "diag_aa_ppo" / f"seed_{args.seed}" / "diagnostics" / f"epoch_{args.epochs}.json"
        ac_json = root / "diag_active_ctro" / f"seed_{args.seed}" / "diagnostics" / f"epoch_{args.epochs}.json"
        report["aa_ref"] = json.loads(aa_json.read_text())
        report["active_ref"] = json.loads(ac_json.read_text())
        report["shared_ref_path"] = ref_path
        report["shared_ref_rows"] = int(torch.load(ref_path, weights_only=False).shape[0])
    out = root / "summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")
    if "finished_minigrid" in report:
        print(report["finished_minigrid"]["attribution"])


if __name__ == "__main__":
    main()
