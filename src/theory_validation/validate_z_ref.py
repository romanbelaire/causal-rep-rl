"""
Offline validation of Z_ref quality (Exp 0): gradient ratio + color augmentation robustness.
CPU-only by default.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.environments.minigrid_wrapper import MinigridColorAugWrapper, MinigridWrapper
from src.metrics.gradients import compute_value_gradient_z_magnitude
from src.main import create_critic, create_policy, set_seed, _policy_action_from_obs
from src.algorithms.ppo import PPO
from src.utils.config import Config
from src.theory_validation.z_ref_store import ZRefStore


def _encode(critic, obs, device):
    obs_t = obs.unsqueeze(0).to(device)
    mu, _ = critic.encode(obs_t)
    return mu.squeeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--expert-weights", type=str, required=True)
    parser.add_argument("--z-ref", type=str, required=True)
    parser.add_argument("--subopt-weights", type=str, default=None,
                        help="Optional suboptimal checkpoint for grad comparison")
    parser.add_argument("--output", type=str, default="./artifacts/theory_validation/exp0_validation.json")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cpu"
    config = Config(args.config)
    set_seed(args.seed)
    task = config["environment"]["task"]
    env = MinigridWrapper(task, seed=args.seed, keep_image_format=False)
    aug_env = MinigridColorAugWrapper(
        MinigridWrapper(task, seed=args.seed + 1, keep_image_format=False),
        color_perm_seed=1234,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    latent_dim = config["architecture"]["critic"]["latent_dim"]
    policy = create_policy(latent_dim, action_dim, config.config, device, use_repr_input=True)
    critic = create_critic(obs_dim, config.config, device)
    algo = PPO(policy, critic, config["algorithm"], device)
    algo.load(args.expert_weights)
    policy.eval()
    critic.eval()

    store = ZRefStore.load(args.z_ref)

    z_ref_grads = []
    obs_by_key = {}
    for _ in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            gt = env.get_ground_truth_representation(obs)
            key = tuple(round(float(x), 4) for x in gt.tolist())
            z = _encode(critic, obs, device)
            z_ref_grads.append(
                compute_value_gradient_z_magnitude(critic, z.unsqueeze(0))
            )
            obs_by_key[key] = obs
            action_val = _policy_action_from_obs(obs, policy, critic, None, device, True)
            obs, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated

    subopt_grads = []
    if args.subopt_weights:
        critic_sub = create_critic(obs_dim, config.config, device)
        algo_sub = PPO(policy, critic_sub, config["algorithm"], device)
        algo_sub.load(args.subopt_weights)
        critic_sub.eval()
        for _ in range(args.n_episodes):
            obs, _ = env.reset()
            done = False
            while not done:
                z = _encode(critic_sub, obs, device)
                subopt_grads.append(
                    compute_value_gradient_z_magnitude(critic_sub, z.unsqueeze(0))
                )
                action_val = _policy_action_from_obs(obs, policy, critic_sub, None, device, False)
                obs, _, terminated, truncated, _ = env.step(action_val)
                done = terminated or truncated

    aug_dists = []
    for key, obs_orig in list(obs_by_key.items())[:200]:
        obs_aug, _ = aug_env.reset()
        # align env state by replaying is hard; compare same obs through aug wrapper on tensor
        obs_flat = obs_orig
        img = obs_flat.view(*env.obs_shape)
        perm = aug_env.color_perm
        obs_perm = img[..., perm].reshape(-1)
        z1 = _encode(critic, obs_flat, device)
        z2 = _encode(critic, obs_perm, device)
        aug_dists.append(torch.norm(z1 - z2).item())

    report = {
        "n_z_ref_grad_samples": len(z_ref_grads),
        "grad_z_ref_mean": float(np.mean(z_ref_grads)),
        "grad_z_ref_max": float(np.max(z_ref_grads)),
        "grad_subopt_mean": float(np.mean(subopt_grads)) if subopt_grads else None,
        "grad_ratio_subopt_over_expert": (
            float(np.mean(subopt_grads) / (np.mean(z_ref_grads) + 1e-8))
            if subopt_grads else None
        ),
        "color_aug_z_dist_mean": float(np.mean(aug_dists)) if aug_dists else None,
        "n_z_ref_keys": len(store),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    env.close()
    aug_env.close()


if __name__ == "__main__":
    main()
