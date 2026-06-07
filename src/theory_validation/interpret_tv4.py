"""
Classify theory validation v4 outcomes (A vs B) from metrics CSVs.

Outcome A: kappa meaningfully above floor, rank preserved, chain grad-vs-RHS improves.
Outcome B: kappa flat despite valid Z_ref.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.theory_validation.plot_validation_v2 import load_metrics

KAPPA_FLOOR = 1e-4
KAPPA_COL = "convexity_kappa_concave_mean"
RANK_COL = "log_effective_feature_rank_pr"
CHAIN_GRAD_COL = "chain_grad_vs_rhs_ratio"


def _latest_row(df: pd.DataFrame) -> pd.Series:
    return df.sort_values("step").iloc[-1]


def classify_arm(log_dir: Path, arm: str, experiment_keys: list[str]) -> dict:
    rows = []
    for key in experiment_keys:
        matches = sorted(log_dir.glob(f"{key}_seed*_metrics.csv"))
        if not matches:
            rows.append({"experiment": key, "status": "missing"})
            continue
        kappas = []
        ranks = []
        chains = []
        for p in matches:
            df = load_metrics(p)
            last = _latest_row(df)
            if KAPPA_COL in last:
                kappas.append(float(last[KAPPA_COL]))
            if RANK_COL in last:
                ranks.append(float(last[RANK_COL]))
            if CHAIN_GRAD_COL in last:
                chains.append(float(last[CHAIN_GRAD_COL]))
        kappa_mean = sum(kappas) / len(kappas) if kappas else float("nan")
        rank_mean = sum(ranks) / len(ranks) if ranks else float("nan")
        chain_mean = sum(chains) / len(chains) if chains else float("nan")
        kappa_ok = kappa_mean > KAPPA_FLOOR
        outcome = "A" if kappa_ok else "B"
        rows.append({
            "experiment": key,
            "arm": arm,
            "kappa_concave_mean": kappa_mean,
            "log_rank_pr": rank_mean,
            "chain_grad_vs_rhs": chain_mean,
            "outcome": outcome,
            "status": "ok",
        })
    return {"arm": arm, "classifications": rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("outputs/theory_validation_v4"))
    parser.add_argument("--output", type=Path, default=Path("outputs/theory_validation_v4/tv4_interpretation.json"))
    args = parser.parse_args()

    from src.theory_validation.plot_validation_v4 import MICO_EXP1_KEYS, DBC_EXP1_KEYS

    report = {
        "mico": classify_arm(args.log_dir, "mico", MICO_EXP1_KEYS + ["theory_v4_mico_chain"]),
        "dbc": classify_arm(args.log_dir, "dbc", DBC_EXP1_KEYS + ["theory_v4_dbc_chain"]),
        "kappa_floor": KAPPA_FLOOR,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json
    with args.output.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    for arm in ("mico", "dbc"):
        print(f"\n=== {arm.upper()} arm ===")
        for row in report[arm]["classifications"]:
            if row.get("status") == "missing":
                print(f"  {row['experiment']}: MISSING")
            else:
                print(
                    f"  {row['experiment']}: Outcome {row['outcome']} "
                    f"(kappa={row['kappa_concave_mean']:.2e}, rank={row['log_rank_pr']:.3f})"
                )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
