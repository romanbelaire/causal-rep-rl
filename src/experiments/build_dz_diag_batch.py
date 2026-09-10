"""Build a frozen cartpole-pixels diagnostic observation batch (CPU-safe env steps).

Collects random-policy observations, stores fixed pair indices. Encoding and
C_0.01 evaluation happen later with checkpoint encoders.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.evaluation.suites import EVAL_SUITES
from src.experiments.performance_runner import make_train_env
from src.losses.dz_trust_region import sample_pair_index_mask


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=str, default="dmcontrol_pixels")
    parser.add_argument("--task", type=str, default="cartpole-swingup")
    parser.add_argument("--n-obs", type=int, default=512)
    parser.add_argument("--n-pairs", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=str,
        default="results/dmcontrol_pixels/diag/cartpole-swingup/dz_diag.pt",
    )
    args = parser.parse_args()

    if args.suite not in EVAL_SUITES:
        raise ValueError(f"Unknown suite {args.suite}")
    torch.manual_seed(args.seed)
    np_rng = __import__("numpy").random.default_rng(args.seed)
    env = make_train_env(args.suite, args.task)
    obs_list = []
    obs, _ = env.reset(seed=args.seed)
    while len(obs_list) < args.n_obs:
        obs_list.append(torch.as_tensor(obs).float())
        if hasattr(env, "action_space"):
            action = env.action_space.sample()
        else:
            # DMControl wrappers expose action_low/high, not gymnasium action_space.
            action = np_rng.uniform(env.action_low, env.action_high).astype("float32")
            action = torch.as_tensor(action)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    obs_t = torch.stack(obs_list[: args.n_obs], dim=0)
    pair_i, pair_j = sample_pair_index_mask(
        obs_t.shape[0], args.n_pairs, lambda_loc=0.0, device=torch.device("cpu")
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "obs": obs_t,
        "pair_i": pair_i.cpu(),
        "pair_j": pair_j.cpu(),
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "n_obs": int(obs_t.shape[0]),
        "n_pairs": int(pair_i.shape[0]),
    }
    torch.save(payload, out)
    print(f"wrote {out} obs={tuple(obs_t.shape)} pairs={pair_i.shape[0]}")


if __name__ == "__main__":
    main()
