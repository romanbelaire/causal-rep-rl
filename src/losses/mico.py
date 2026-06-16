"""MICo bisimulation metric loss (angular diffuse metric, Huber target)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.bisimulation_utils import (
    VAEEncoderTarget,
    encode_phi,
    encode_phi_target,
    permute_batch_pairs,
)


def angular_distance(phi_x: torch.Tensor, phi_y: torch.Tensor) -> torch.Tensor:
    """Angle theta between embedding rows (MICo diffuse metric)."""
    dot = (phi_x * phi_y).sum(dim=1)
    norms = torch.norm(phi_x, dim=1) * torch.norm(phi_y, dim=1)
    cos_theta = dot / norms.clamp(min=1e-8)
    cos_theta = cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    return torch.acos(cos_theta)


def mico_u_omega(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """U_omega(x, y) = (||phi(x)||^2 + ||phi(y)||^2)/2 + beta * theta(phi(x), phi(y))."""
    norm_term = (phi_x.pow(2).sum(dim=1) + phi_y.pow(2).sum(dim=1)) * 0.5
    theta = angular_distance(phi_x, phi_y)
    return norm_term + beta * theta


def compute_mico_loss(
    critic: nn.Module,
    encoder_target: VAEEncoderTarget,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    beta: float = 0.1,
    huber_delta: float = 1.0,
    embed_ball_radius: float | None = None,
    repr_net: nn.Module | None = None,
) -> tuple[torch.Tensor, dict]:
    """
    MICo loss: Huber(U - target) with target = |r_x-r_y| + gamma * U_bar(x', y').

    Returns raw Huber mean (caller scales by alpha).
    """
    batch_size = obs.shape[0]
    perm = permute_batch_pairs(batch_size, obs.device)

    obs_y = obs[perm]
    next_obs_y = next_obs[perm]
    rewards_y = rewards[perm]

    phi_x = encode_phi(critic, obs, repr_net=repr_net, embed_ball_radius=embed_ball_radius)
    phi_y = encode_phi(critic, obs_y, repr_net=repr_net, embed_ball_radius=embed_ball_radius)

    with torch.no_grad():
        phi_x_next = encode_phi_target(
            encoder_target, next_obs, embed_ball_radius=embed_ball_radius
        )
        phi_y_next = encode_phi_target(
            encoder_target, next_obs_y, embed_ball_radius=embed_ball_radius
        )

    u_omega = mico_u_omega(phi_x, phi_y, beta)
    u_bar = mico_u_omega(phi_x_next, phi_y_next, beta)
    target = torch.abs(rewards - rewards_y) + gamma * u_bar

    huber = F.huber_loss(u_omega, target, reduction="mean", delta=huber_delta)

    stats = {
        "train_mico_loss": huber.item(),
        "train_mico_u_omega_mean": u_omega.mean().item(),
        "train_mico_target_mean": target.mean().item(),
        "train_embedding_norm_mean": phi_x.norm(dim=1).mean().item(),
    }
    return huber, stats


def reward_dispersion(rewards: torch.Tensor) -> float:
    """std(r) over batch — guard for sparse-reward MICo collapse."""
    return rewards.std(unbiased=False).item()
