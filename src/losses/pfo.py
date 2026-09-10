"""Policy Feature Orthogonality (PFO) / feature trust loss (Moalla et al.).

L_PFO = || phi_new(s) - phi_old(s) ||_2^2 on actor_preactivation; old detached.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


def actor_phi(policy: nn.Module, obs: torch.Tensor) -> torch.Tensor:
    return policy.actor_preactivation(obs)


def compute_pfo_loss(
    policy: nn.Module,
    old_policy: nn.Module,
    obs: torch.Tensor,
) -> torch.Tensor:
    """Mean squared L2 distance between new and detached old actor pre-activations."""
    phi_new = actor_phi(policy, obs)
    with torch.no_grad():
        phi_old = actor_phi(old_policy, obs)
    phi_old = phi_old.detach()
    diff = phi_new - phi_old
    return (diff * diff).sum(dim=-1).mean()


def snapshot_policy(policy: nn.Module) -> nn.Module:
    """Frozen deepcopy used as phi_old for one PPO update cycle."""
    old = copy.deepcopy(policy)
    old.eval()
    for p in old.parameters():
        p.requires_grad_(False)
    return old


def verify_pfo_unit_checks(device: str = "cpu") -> None:
    """Fail-fast checks: identical → 0; old params no grad; perturbation raises loss."""
    from src.architectures.policies.nature_policy import NaturePolicy

    torch.manual_seed(0)
    obs_shape = (4, 84, 84)
    policy = NaturePolicy(obs_shape, action_dim=6).to(device)
    old = snapshot_policy(policy)
    obs = torch.randn(8, *obs_shape, device=device)

    loss0 = compute_pfo_loss(policy, old, obs)
    if float(loss0.item()) > 1e-10:
        raise RuntimeError(f"identical nets must give PFO≈0, got {loss0.item()}")

    for p in old.parameters():
        if p.requires_grad:
            raise RuntimeError("old policy params must have requires_grad=False")

    loss0.backward()
    for p in old.parameters():
        if p.grad is not None:
            raise RuntimeError("old policy must receive no gradients")
    policy.zero_grad(set_to_none=True)

    with torch.no_grad():
        for p in policy.parameters():
            p.add_(0.1)
    loss1 = compute_pfo_loss(policy, old, obs)
    if float(loss1.item()) <= float(loss0.item()) + 1e-8:
        raise RuntimeError(
            f"perturbation must raise PFO loss ({loss0.item()} -> {loss1.item()})"
        )
    print("PFO unit checks passed", flush=True)


if __name__ == "__main__":
    verify_pfo_unit_checks("cpu")
