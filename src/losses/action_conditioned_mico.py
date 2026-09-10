"""Same-action MICo surrogate. Refuses unequal-action pairs."""

import torch
import torch.nn.functional as F

from src.losses.mico import mico_u_omega


def pair_metric(
    phi_x: torch.Tensor,
    phi_y: torch.Tensor,
    metric: str,
    beta_mico: float,
) -> torch.Tensor:
    if metric == "angular_diffuse":
        return mico_u_omega(phi_x, phi_y, beta_mico)
    if metric == "euclidean":
        return (phi_x - phi_y).pow(2).sum(dim=1).sqrt()
    raise RuntimeError(f"unknown pair_metric {metric!r}")


def action_conditioned_mico_loss(
    phi: torch.Tensor,
    phi_next_target: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    actions: torch.Tensor,
    pair_i: torch.Tensor,
    pair_j: torch.Tensor,
    gamma: float,
    metric: str,
    beta_mico: float,
    huber_delta: float,
    pair_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    if (actions[pair_i] != actions[pair_j]).any():
        raise RuntimeError("action-conditioned MICo received unequal-action pairs")
    u = pair_metric(phi[pair_i], phi[pair_j], metric, beta_mico)
    with torch.no_grad():
        u_bar = pair_metric(
            phi_next_target[pair_i], phi_next_target[pair_j], metric, beta_mico
        )
        live = (~terminated[pair_i]).float() * (~terminated[pair_j]).float()
        target = (rewards[pair_i] - rewards[pair_j]).abs() + gamma * live * u_bar
    residual = F.huber_loss(u, target, reduction="none", delta=huber_delta)
    if pair_weight is None:
        loss = residual.mean()
    else:
        w = pair_weight / pair_weight.sum().clamp_min(1e-8)
        loss = (residual * w).sum()
    stats = {
        "train_mico_ac_loss": loss.detach(),
        "train_mico_ac_u_mean": u.mean().detach(),
        "train_mico_ac_target_mean": target.mean().detach(),
        "train_mico_ac_n_pairs": torch.tensor(float(pair_i.numel()), device=phi.device),
    }
    return loss, stats
