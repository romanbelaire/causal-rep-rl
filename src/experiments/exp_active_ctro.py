"""Active CTRO on MiniGrid Unlock."""

import argparse

from src.agents.active_ctro import ActiveCTRO
from src.experiments.config import ACTIVE_CTRO_MINIGRID, E0_TRAINING_SMOKE, apply_algo_preset
from src.experiments.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description="Active CTRO (MiniGrid Unlock)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--results-root", type=str, default="results")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--algo-preset", type=str, default=None)
    args = parser.parse_args()
    algo = dict(ACTIVE_CTRO_MINIGRID)
    if args.algo_preset:
        preset = apply_algo_preset(args.algo_preset)
        algo.update(preset)
        if args.algo_preset == "active_ctro":
            algo["active"] = preset["active"]
    train_overrides = E0_TRAINING_SMOKE if args.smoke else None
    if args.smoke:
        algo = dict(algo)
        algo["active"] = dict(algo["active"])
        algo["active"]["query_rollout_size"] = 128
        algo["active"]["replay"] = {**algo["active"]["replay"], "batch_size": 32}
    run_experiment(
        exp_name="exp_active_ctro",
        seed=args.seed,
        agent_cls=ActiveCTRO,
        algo_overrides=algo,
        train_overrides=train_overrides,
        results_root=args.results_root,
        device=args.device,
    )


if __name__ == "__main__":
    main()
