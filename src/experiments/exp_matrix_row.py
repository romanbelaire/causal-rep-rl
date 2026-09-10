"""Incremental MiniGrid matrix row (Stage 5). Do not treat as a claim until G0–G5 pass."""

import argparse

from src.agents.active_ctro import ActiveCTRO
from src.agents.ctro import CTRO
from src.experiments.config import E0_TRAINING_SMOKE, MATRIX_TRAINING, matrix_row_algo
from src.experiments.runner import run_experiment

ROW_NAMES = {
    1: "matrix_r1_aa_ppo",
    2: "matrix_r2_aa_ppo_target_pl",
    3: "matrix_r3_replay_q",
    4: "matrix_r4_ac_mico",
    5: "matrix_r5_lsep",
    6: "matrix_r6_query_u",
    7: "matrix_r7_relational_observe",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Active-CTRO MiniGrid matrix row")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--results-root", type=str, default="results/active_ctro/matrix")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.row not in ROW_NAMES:
        raise RuntimeError(f"--row must be 1–7, got {args.row}")
    algo = matrix_row_algo(args.row)
    agent_cls = CTRO if args.row in (1, 2) else ActiveCTRO
    if args.smoke:
        train = dict(E0_TRAINING_SMOKE)
        train["diag_log_epochs"] = (0, 3)
        if args.row >= 3:
            algo = dict(algo)
            algo["active"] = dict(algo["active"])
            algo["active"]["query_rollout_size"] = 128
            algo["active"]["replay"] = {**algo["active"]["replay"], "batch_size": 32}
    else:
        train = dict(MATRIX_TRAINING)
    run_experiment(
        exp_name=ROW_NAMES[args.row],
        seed=args.seed,
        agent_cls=agent_cls,
        algo_overrides=algo,
        train_overrides=train,
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
