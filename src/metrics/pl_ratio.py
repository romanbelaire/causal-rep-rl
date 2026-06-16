"""mu_PL diagnostic: bootstrap PL ratio at 5th percentile over batch."""

import torch
import torch.nn as nn


def compute_mu_pl_bootstrap(
    critic: nn.Module,
    z: torch.Tensor,
    rewards: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    eps: float = 1e-4,
) -> dict[str, float]:
    """
    mu_PL = 5th percentile of pl_ratio over batch.

    V_target = r + gamma * V(Z(s'))  [stop-gradient on bootstrap]
    value_gap = V_target - V(Z(s))
    pl_ratio = ||grad_Z V||^2 / (2 * max(value_gap, eps))
    """
    with torch.no_grad():
        mu_next, _ = critic.encode(next_obs)
        z_next = mu_next
        v = critic.value_head(z).squeeze(-1)
        v_next = critic.value_head(z_next).squeeze(-1)
        v_target = rewards + gamma * v_next
        value_gap = v_target - v

    rows = []
    for i in range(z.shape[0]):
        zi = z[i : i + 1].detach().requires_grad_(True)
        vi = critic.value_head(zi).squeeze()
        gi = torch.autograd.grad(vi, zi, retain_graph=False)[0]
        rows.append(gi.squeeze(0))
    grad_z = torch.stack(rows, dim=0)
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
