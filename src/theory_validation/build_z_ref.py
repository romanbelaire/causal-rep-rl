"""
Build Z_ref lookup tables from a frozen expert checkpoint (Exp 0).

CPU/GPU: uses cuda if available in config; rollouts are lightweight.
"""

import argparse
import json
from pathlib import Path

import torch

from src.environments.minigrid_wrapper import MinigridWrapper
from src.main import (
    create_critic,
    create_policy,
    set_seed,
    _policy_action_from_obs,
)
from src.algorithms.ppo import PPO
from src.utils.config import Config
from src.theory_validation.z_ref_store import ZRefStore, build_table_from_rollout
from src.theory_validation.z_ref_paths import resolve_expert_weights_file


def _setup_from_config(config_path: str, weights_path: Path, device: str, seed: int):
    config = Config(config_path)
    set_seed(seed)
    task = config["environment"]["task"]
    env = MinigridWrapper(task, seed=seed, keep_image_format=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    critic_type = config["architecture"]["critic"]["type"]
    repr_net = None
    if critic_type == "vae":
        latent_dim = config["architecture"]["critic"].get("latent_dim", 8)
        repr_dim = latent_dim
    else:
        repr_dim = config["architecture"].get("representation", {}).get("repr_dim", 512)
    policy = create_policy(repr_dim, action_dim, config.config, device, use_repr_input=True)
    critic_input_dim = obs_dim if critic_type == "vae" else repr_dim
    critic = create_critic(critic_input_dim, config.config, device, repr_net=repr_net)
    algorithm = PPO(policy, critic, config["algorithm"], device, repr_net=repr_net)
    algorithm.load(str(weights_path))
    policy.eval()
    critic.eval()
    return env, policy, critic, repr_net, device


def roll_expert_trajectories(
    env,
    policy,
    critic,
    repr_net,
    device: str,
    n_episodes: int,
    deterministic: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    gt_list = []
    z_list = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            gt = env.get_ground_truth_representation(obs)
            with torch.no_grad():
                obs_t = obs.unsqueeze(0).to(device) if obs.dim() == 1 else obs.to(device)
                if hasattr(critic, "encode"):
                    mu, _ = critic.encode(obs_t)
                    z = mu.squeeze(0).cpu()
                elif hasattr(critic, "get_latent_representation"):
                    z = critic.get_latent_representation(obs_t).squeeze(0).cpu()
                else:
                    raise ValueError("Expert critic must be VAE with encoder for Z_ref")
            gt_list.append(gt.cpu() if isinstance(gt, torch.Tensor) else gt)
            z_list.append(z)
            action_val = _policy_action_from_obs(
                obs, policy, critic, repr_net, device, deterministic=deterministic
            )
            obs, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
    return gt_list, z_list


def main():
    parser = argparse.ArgumentParser(description="Build Z_ref table from expert checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Expert run dir or path to weights_expert.pt / weights_final.pt",
    )
    parser.add_argument(
        "--use-weights-final",
        action="store_true",
        help="Load weights_final.pt from the expert run dir (e.g. 92% success without 95% gate).",
    )
    parser.add_argument("--output", type=str, required=True, help="Output .pt path")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    weights_path, checkpoint_kind = resolve_expert_weights_file(
        args.weights, use_weights_final=args.use_weights_final
    )

    config = Config(args.config)
    device = config.get("experiment.device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env, policy, critic, repr_net, device = _setup_from_config(
        args.config, weights_path, device, args.seed
    )
    gt_list, z_list = roll_expert_trajectories(
        env, policy, critic, repr_net, device, args.n_episodes
    )
    table = build_table_from_rollout(gt_list, z_list)
    store = ZRefStore(table)
    metadata = {
        "config": args.config,
        "weights": str(weights_path),
        "checkpoint_kind": checkpoint_kind,
        "use_weights_final": args.use_weights_final,
        "n_episodes": args.n_episodes,
        "n_states": len(table),
        "seed": args.seed,
    }
    store.save(args.output, metadata=metadata)
    print(f"Saved Z_ref with {len(table)} keys to {args.output}")
    env.close()


if __name__ == "__main__":
    main()
