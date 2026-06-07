"""Deep Bisimulation for Control (DBC) metric matching loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.latent_dynamics import LatentDynamicsModel
from src.utils.bisimulation_utils import (
    encode_phi,
    permute_batch_pairs,
    reward_pair_dispersion,
)


def compute_dbc_loss(
    critic: nn.Module,
    dynamics: LatentDynamicsModel,
    dynamics_target: LatentDynamicsModel,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    coef: float,
    gamma: float,
    huber_delta: float = 1.0,
    embed_ball_radius: float | None = None,
    repr_net: nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    DBC on-policy bisimulation loss with Gaussian W2 transition metric.

    T = |r_x - r_y| + gamma * W2(P_bar(.|z_x,a_x), P_bar(.|z_y,a_y))
    d = ||z_x - z_y||_2
    L = Huber(T - d)
    """
    batch_size = obs.shape[0]
    perm = permute_batch_pairs(batch_size, obs.device)

    obs_y = obs[perm]
    actions_y = actions[perm]
    rewards_y = rewards[perm]

    z_x = encode_phi(critic, obs, repr_net=repr_net, embed_ball_radius=embed_ball_radius)
    z_y = encode_phi(critic, obs_y, repr_net=repr_net, embed_ball_radius=embed_ball_radius)

    d_rep = torch.norm(z_x - z_y, dim=1)

    with torch.no_grad():
        mean_x, log_std_x = dynamics_target(z_x, actions)
        mean_y, log_std_y = dynamics_target(z_y, actions_y)
        w2 = dynamics_target.gaussian_w2(mean_x, log_std_x, mean_y, log_std_y)
        target = torch.abs(rewards - rewards_y) + gamma * w2

    huber = F.huber_loss(d_rep, target, reduction="mean", delta=huber_delta)

    stats = {
        "train_dbc_loss": huber.item(),
        "train_dbc_w2_mean": w2.mean().item(),
        "train_dbc_target_mean": target.mean().item(),
        "train_dbc_reward_pair_dispersion": reward_pair_dispersion(rewards, rewards_y),
        "train_dbc_latent_norm_mean": z_x.norm(dim=1).mean().item(),
        "dbc_loss_coef_effective": coef,
    }
    return huber, stats
