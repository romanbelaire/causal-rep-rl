"""PL-2 regret, mu_PL, and restricted curvature on subspace bases."""

import copy

import torch
import torch.nn as nn

from src.metrics.gradients import compute_value_jacobian_z


def mc_return_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    episode_ids: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Monte Carlo return-to-go G_t for each transition (gauge-invariant).
    """
    n = rewards.shape[0]
    returns = torch.zeros(n, device=rewards.device, dtype=rewards.dtype)
    for ep in episode_ids.unique():
        mask = episode_ids == ep
        idx = torch.where(mask)[0]
        g = 0.0
        for pos in range(len(idx) - 1, -1, -1):
            t = idx[pos]
            g = rewards[t] + gamma * g * (1.0 - dones[t])
            returns[t] = g
    return returns


def nstep_bootstrap_return_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    episode_ids: torch.Tensor,
    v_continuation: torch.Tensor,
    gamma: float,
    horizon: int = 20,
) -> torch.Tensor:
    """
    n-step return-to-go with critic bootstrap at horizon (gauge-invariant).
    """
    n = rewards.shape[0]
    returns = torch.zeros(n, device=rewards.device, dtype=rewards.dtype)
    for i in range(n):
        g = 0.0
        disc = 1.0
        ep = episode_ids[i]
        for j in range(horizon):
            t = i + j
            if t >= n or episode_ids[t] != ep:
                break
            g += disc * rewards[t]
            if dones[t] > 0:
                disc = 0.0
                break
            disc *= gamma
        else:
            t_horizon = i + horizon
            if t_horizon < n and episode_ids[t_horizon] == ep:
                g += disc * v_continuation[t_horizon]
        returns[i] = g
    return returns


def critic_values_on_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """V(z) per sample, shape [N]."""
    with torch.no_grad():
        return critic.value_head(z).squeeze(-1)


def continuation_values_on_z(
    critic: nn.Module,
    z: torch.Tensor,
) -> torch.Tensor:
    """Bootstrap continuation V(z) per sample, shape [N]."""
    return critic_values_on_z(critic, z)


def compute_pl2_regret(
    critic: nn.Module,
    z: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    episode_ids: torch.Tensor,
    gamma: float,
    bootstrap_horizon: int = 20,
    target_critic: nn.Module | None = None,
) -> dict[str, float]:
    """
    PL-2 regret proxy: V*(s) - V(Z).

    V* = max(MC return, n-step bootstrap with online critic, n-step with target critic).
    When target_critic is provided (double estimator), both bootstrap paths are included
    in the max to reduce optimistic bias from a single critic bootstrap.
    """
    v_mc = mc_return_to_go(rewards, dones, episode_ids, gamma)
    v_z = critic_values_on_z(critic, z)

    v_nstep_online = nstep_bootstrap_return_to_go(
        rewards,
        dones,
        episode_ids,
        continuation_values_on_z(critic, z),
        gamma,
        horizon=bootstrap_horizon,
    )

    v_star = torch.maximum(v_mc, v_nstep_online)

    if target_critic is not None:
        with torch.no_grad():
            v_nstep_target = nstep_bootstrap_return_to_go(
                rewards,
                dones,
                episode_ids,
                continuation_values_on_z(target_critic, z),
                gamma,
                horizon=bootstrap_horizon,
            )
        v_star = torch.maximum(v_star, v_nstep_target)
    else:
        v_nstep_target = None

    regret = v_star - v_z
    regret_mc = v_mc - v_z
    regret_nstep = v_nstep_online - v_z

    negative_frac = (regret < 0).float().mean().item()

    out = {
        "chain_regret_pl2": regret.mean().item(),
        "chain_regret_pl2_median": regret.median().item(),
        "chain_regret_pl2_min": regret.min().item(),
        "chain_regret_pl2_max": regret.max().item(),
        "regret_pl2_per_sample": regret,
        "v_star_mean": v_star.mean().item(),
        "v_z_mean": v_z.mean().item(),
        "v_mc_mean": v_mc.mean().item(),
        "v_nstep_mean": v_nstep_online.mean().item(),
        "chain_regret_pl2_mc": regret_mc.mean().item(),
        "chain_regret_pl2_nstep": regret_nstep.mean().item(),
        "regret_negative_frac": negative_frac,
    }
    if v_nstep_target is not None:
        out["v_nstep_target_mean"] = v_nstep_target.mean().item()
        out["chain_regret_pl2_nstep_target"] = (v_nstep_target - v_z).mean().item()
    return out


def frozen_critic_copy(critic: nn.Module) -> nn.Module:
    """Deep copy of critic for target-network bootstrap (eval, no grad)."""
    target = copy.deepcopy(critic)
    target.eval()
    for p in target.parameters():
        p.requires_grad_(False)
    return target


def compute_mu_pl(
    critic: nn.Module,
    z: torch.Tensor,
    regret_per_sample: torch.Tensor,
    eps_floor: float = 1e-4,
) -> dict[str, float]:
    """
    mu_PL = ||grad V||^2 / (2 * max(V* - V, eps_floor)) per sample.
    """
    jacobian = compute_value_jacobian_z(critic, z)
    grad_sq = (jacobian.pow(2).sum(dim=1))
    denom = 2.0 * torch.clamp(regret_per_sample, min=eps_floor)
    mu_per = grad_sq / denom
    q05 = torch.quantile(mu_per, 0.05).item()

    positive_mask = regret_per_sample > 0
    mu_positive = mu_per[positive_mask]
    mu_pl_q05_positive = (
        torch.quantile(mu_positive, 0.05).item()
        if mu_positive.numel() > 0
        else float("nan")
    )

    return {
        "mu_pl_inf": mu_per.min().item(),
        "mu_pl_q05": q05,
        "mu_pl_median": mu_per.median().item(),
        "mu_pl_mean": mu_per.mean().item(),
        "mu_pl_q05_positive_regret": mu_pl_q05_positive,
        "mu_pl_per_sample": mu_per,
    }


def hvp_value_z(critic: nn.Module, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Hessian-vector product of sum_i V(z_i) w.r.t. z, direction v per row."""
    z_h = z.detach().requires_grad_(True)
    values = critic.value_head(z_h).sum()
    grad = torch.autograd.grad(values, z_h, create_graph=True)[0]
    gv = (grad * v).sum()
    hvp = torch.autograd.grad(gv, z_h, retain_graph=False)[0]
    return hvp


def restricted_curvature_min(
    critic: nn.Module,
    z: torch.Tensor,
    basis: torch.Tensor,
) -> float:
    """
    Minimum v^T (-H) v over unit v in column span of basis (restricted lambda_min+).
    """
    curvatures = []
    for j in range(basis.shape[1]):
        v = basis[:, j]
        v = v / (v.norm() + 1e-8)
        v_batch = v.unsqueeze(0).expand(z.shape[0], -1)
        hvp = hvp_value_z(critic, z, v_batch)
        v_hvp = (hvp * v_batch).sum(dim=1).mean()
        curvatures.append((-v_hvp).item())
    return min(curvatures)


def compute_pl_and_curvature_bundle(
    critic: nn.Module,
    z: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    episode_ids: torch.Tensor,
    subspaces: dict[str, torch.Tensor],
    gamma: float,
    eps_floor: float = 1e-4,
    target_critic: nn.Module | None = None,
    use_target_bootstrap: bool = False,
) -> dict[str, float]:
    """PL-2 regret, mu_PL, and restricted curvature on A/B/C."""
    bootstrap_target = target_critic
    if use_target_bootstrap and bootstrap_target is None:
        bootstrap_target = frozen_critic_copy(critic)

    pl2 = compute_pl2_regret(
        critic,
        z,
        rewards,
        dones,
        episode_ids,
        gamma,
        target_critic=bootstrap_target,
    )
    mu = compute_mu_pl(critic, z, pl2["regret_pl2_per_sample"], eps_floor=eps_floor)
    del pl2["regret_pl2_per_sample"]
    del mu["mu_pl_per_sample"]

    metrics = {**pl2, **mu}
    for name, basis in subspaces.items():
        key = name.lower()
        metrics[f"lambda_min_plus_{key}"] = restricted_curvature_min(critic, z, basis)
    return metrics
