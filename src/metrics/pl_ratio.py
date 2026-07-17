"""mu_PL diagnostic: bootstrap PL ratio at 5th percentile over batch."""

import torch
import torch.nn as nn

from src.utils.batched_grad import batched_value_head_grad_z


def compute_mu_pl_bootstrap(
    critic: nn.Module,
    z: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    eps: float = 1e-4,
    max_samples: int | None = None,
) -> dict[str, float]:
    """
    mu_PL = 5th percentile of pl_ratio over batch.

    V_target = r + gamma * V(Z(s'))  [stop-gradient on bootstrap]
    value_gap = V_target - V(Z(s))
    pl_ratio = ||grad_Z V||^2 / (2 * max(value_gap, eps))
    """
    n = z.shape[0]
    if max_samples is not None and max_samples < n:
        idx = torch.randperm(n, device=z.device)[:max_samples]
        z = z[idx]
        rewards = rewards[idx]
        next_obs = next_obs[idx]

    with torch.no_grad():
        mu_next, _ = critic.encode(next_obs)
        z_next = mu_next
        v = critic.value_head(z).squeeze(-1)
        v_next = critic.value_head(z_next).squeeze(-1)
        v_target = rewards + gamma * v_next
        value_gap = v_target - v

    z_grad = z.detach().requires_grad_(True)
    grad_z = batched_value_head_grad_z(critic, z_grad)
    grad_sq = grad_z.pow(2).sum(dim=1)

    with torch.no_grad():
        denom = 2.0 * value_gap.clamp(min=eps)
        pl_ratio = grad_sq / denom
        q05 = torch.quantile(pl_ratio, 0.05).item()

    return {
        "mu_pl_q05": q05,
        "mu_pl_mean": pl_ratio.mean().item(),
        "mu_pl_median": pl_ratio.median().item(),
        "mu_pl_inf": pl_ratio.min().item(),
    }
