"""PL coupling hinge loss: penalize when value-head PL ratio falls below mu_0."""

import torch
import torch.nn as nn

from src.utils.batched_grad import batched_value_head_grad_z


def _value_head_output(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    return critic.value_head(z).squeeze(-1)


def _grad_v_wrt_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Per-sample dV/dZ rows [N, d]; Z retains connection to encoder."""
    return batched_value_head_grad_z(critic, z)


def compute_pl_coupling_loss(
    critic: nn.Module,
    z: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    mu_0: float = 0.1,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """
  L_PL = mean(max(0, mu_0 - pl_ratio))

  pl_ratio = ||grad_Z V(Z(s))||^2 / (2 * max(value_gap, eps))
  value_gap = stop_grad(V_target) - V(Z(s)),  V_target = r + gamma * V(Z(s'))
    """
    mu_next, _ = critic.encode(next_obs)
    z_next = mu_next.detach()

    v = _value_head_output(critic, z)
    with torch.no_grad():
        v_next = _value_head_output(critic, z_next)
    v_target = rewards + gamma * v_next
    value_gap = (v_target - v).detach()

    grad_z = _grad_v_wrt_z(critic, z)
    grad_sq = grad_z.pow(2).sum(dim=1)
    denom = 2.0 * value_gap.clamp(min=eps)
    pl_ratio = grad_sq / denom

    hinge = torch.relu(mu_0 - pl_ratio)
    loss = hinge.mean()

    stats = {
        "train_pl_loss": loss.item(),
        "train_pl_ratio_mean": pl_ratio.mean().item(),
        "train_pl_ratio_q05": torch.quantile(pl_ratio.detach(), 0.05).item(),
        "train_value_gap_mean": value_gap.mean().item(),
    }
    return loss, stats
