"""
Sanity checks for bisimulation Z_ref (MICo / DBC experts).

CPU-only, deterministic. Fail fast if embeddings do not collapse by behavioral equivalence.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.environments.minigrid_wrapper import MinigridColorAugWrapper, MinigridWrapper
from src.main import _policy_action_from_obs, create_critic, create_policy, set_seed
from src.algorithms.ppo import PPO
from src.theory_validation.z_ref_expert import encode_z_ref_batch, load_expert_critic
from src.theory_validation.z_ref_paths import resolve_z_ref_expert
from src.utils.config import Config


def _collect_states(env, policy, critic, repr_net, device, n_episodes: int):
    """Roll out policy; return list of (obs, gt_repr_tuple)."""
    states = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            gt = env.get_ground_truth_representation(obs)
            key = tuple(round(float(x), 4) for x in gt.tolist())
            states.append((obs.clone(), key))
            action_val = _policy_action_from_obs(obs, policy, critic, repr_net, device, True)
            obs, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
    return states


def _intra_inter_cluster_distances(
    critic,
    states: list,
    device: str,
) -> tuple[float, float]:
    """Mean pairwise L2 within gt_repr clusters vs between clusters."""
    by_key: dict[tuple, list[torch.Tensor]] = {}
    for obs, key in states:
        z = encode_z_ref_batch(critic, obs.unsqueeze(0).to(device)).squeeze(0).cpu()
        by_key.setdefault(key, []).append(z)

    intra_dists = []
    for zs in by_key.values():
        if len(zs) < 2:
            continue
        for i in range(len(zs)):
            for j in range(i + 1, len(zs)):
                intra_dists.append(torch.norm(zs[i] - zs[j]).item())

    keys = list(by_key.keys())
    inter_dists = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            zi = torch.stack(by_key[keys[i]]).mean(dim=0)
            zj = torch.stack(by_key[keys[j]]).mean(dim=0)
            inter_dists.append(torch.norm(zi - zj).item())

    intra_mean = float(np.mean(intra_dists)) if intra_dists else float("nan")
    inter_mean = float(np.mean(inter_dists)) if inter_dists else float("nan")
    return intra_mean, inter_mean


def _color_invariance_distance(critic, device: str, n_samples: int = 50) -> float:
    """Same grid state with permuted RGB should map to nearby embeddings."""
    base = MinigridWrapper("MiniGrid-Unlock-v0", seed=99, keep_image_format=False)
    aug = MinigridColorAugWrapper(
        MinigridWrapper("MiniGrid-Unlock-v0", seed=99, keep_image_format=False),
        color_perm_seed=1234,
    )
    obs, _ = base.reset(seed=99)
    dists = []
    for _ in range(n_samples):
        z = encode_z_ref_batch(critic, obs.unsqueeze(0).to(device))
        obs_aug = aug._permute_colors(obs)
        z_aug = encode_z_ref_batch(critic, obs_aug.unsqueeze(0).to(device))
        dists.append(torch.norm(z - z_aug).item())
        obs, _, term, trunc, _ = base.step(0)
        if term or trunc:
            obs, _ = base.reset()
    base.close()
    aug.close()
    return float(np.mean(dists))


def _eval_success_rate(config_path: str, weights_path: str, seed: int, n_episodes: int) -> float:
    config = Config(config_path)
    device = "cpu"
    set_seed(seed)
    task = config["environment"]["task"]
    env = MinigridWrapper(task, seed=seed, keep_image_format=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    latent_dim = config["architecture"]["critic"]["latent_dim"]
    policy = create_policy(latent_dim, action_dim, config.config, device, use_repr_input=True)
    critic = create_critic(obs_dim, config.config, device)
    algo = PPO(policy, critic, config["algorithm"], device, action_dim=action_dim)
    algo.load(weights_path)
    policy.eval()
    critic.eval()

    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        while not done:
            action_val = _policy_action_from_obs(obs, policy, critic, None, device, True)
            obs, reward, terminated, truncated, _ = env.step(action_val)
            ep_return += reward
            done = terminated or truncated
        if ep_return > 0:
            successes += 1
    env.close()
    return successes / n_episodes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", type=str, choices=["mico", "dbc"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    device = "cpu"
    expert_config, expert_weights, _ = resolve_z_ref_expert(args.family, args.seed)
    critic = load_expert_critic(expert_config, expert_weights, device)

    config = Config(expert_config)
    set_seed(args.seed)
    task = config["environment"]["task"]
    env = MinigridWrapper(task, seed=args.seed, keep_image_format=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    latent_dim = config["architecture"]["critic"]["latent_dim"]
    policy = create_policy(latent_dim, action_dim, config.config, device, use_repr_input=True)
    loaded_critic = create_critic(obs_dim, config.config, device)
    algo = PPO(policy, loaded_critic, config["algorithm"], device, action_dim=action_dim)
    algo.load(expert_weights)
    policy.eval()

    states = _collect_states(env, policy, critic, None, device, args.n_episodes)
    intra, inter = _intra_inter_cluster_distances(critic, states, device)
    color_dist = _color_invariance_distance(critic, device)
    success_rate = _eval_success_rate(expert_config, expert_weights, args.seed, args.n_episodes)

    env.close()

    if intra >= inter:
        raise ValueError(
            f"Z_ref sanity failed ({args.family} seed {args.seed}): "
            f"intra-cluster mean dist {intra:.4f} >= inter-cluster {inter:.4f}"
        )
    if success_rate < 0.95:
        raise ValueError(
            f"Z_ref sanity failed ({args.family} seed {args.seed}): "
            f"eval success rate {success_rate:.2%} < 95%"
        )

    report = {
        "family": args.family,
        "seed": args.seed,
        "intra_cluster_mean_l2": intra,
        "inter_cluster_mean_l2": inter,
        "color_aug_mean_l2": color_dist,
        "eval_success_rate": success_rate,
        "passed": True,
    }

    out_path = args.output or (
        f"outputs/theory_validation_v4/validate_{args.family}_seed{args.seed}.json"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
