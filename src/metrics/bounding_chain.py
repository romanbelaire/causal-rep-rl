"""
Theorem 4 bounding-chain diagnostics (causal-policy chain).

Measures proxies for:
  J* − J^π, ‖Z* − Z‖, ‖∇_Z V(Z)‖, √ε_KL

and checks directional consistency of the inequality chain.
"""

import torch
import torch.nn as nn

from src.metrics.gradients import compute_value_gradient_z_magnitude


def compute_z_star_distance(
    z: torch.Tensor,
    z_star: torch.Tensor | None = None,
) -> dict[str, float]:
    """
    Per-sample L2 distance to Z*.

    If z_star is [N, d], uses per-row pairing. If [1, d] or [d], broadcasts.
    If z_star is None, uses batch centroid (legacy proxy).
    """
    if z_star is None:
        z_star = z.mean(dim=0, keepdim=True).expand_as(z)
    elif z_star.dim() == 1:
        z_star = z_star.unsqueeze(0).expand_as(z)
    elif z_star.shape[0] == 1 and z.shape[0] > 1:
        z_star = z_star.expand_as(z)

    distances = torch.norm(z - z_star, dim=1)
    return {
        "z_star_distance_mean": distances.mean().item(),
        "z_star_distance_max": distances.max().item(),
    }


def compute_bounding_chain_metrics(
    critic: nn.Module,
    z: torch.Tensor,
    kl_divergence: float,
    mean_episode_return: float,
    running_best_return: float,
    smoothness_L: float | None = None,
    z_ref: torch.Tensor | None = None,
    mu_concave: float | None = None,
    pct_concave: float | None = None,
    c_z: float = 1.0,
    c1: float = 1.0,
    bound_unreliable_pct_threshold: float = 0.10,
) -> dict[str, float]:
    """
    Assemble all quantities in Theorem 4's bounding chain.

    Args:
        critic: Value critic (operates on z)
        z: Latent batch [N, d]
        kl_divergence: Per-state KL proxy from policy comparison
        mean_episode_return: Recent mean episodic return
        running_best_return: Running maximum return as J* proxy
        smoothness_L: Top Hessian eigenvalue proxy for L-smoothness
        z_ref: Per-sample Z*(s) from expert table [N, d]
        mu_concave: -min(λ_min, 0) batch max for scaled RHS
        pct_concave: Fraction of batch with λ_min < -ε
        c_z: Lemma zgrad constant scale
        c1: Causal theorem constant scale
        bound_unreliable_pct_threshold: Flag bound when pct_concave exceeds this
    """
    grad_z = compute_value_gradient_z_magnitude(critic, z)
    sqrt_kl = kl_divergence ** 0.5

    z_dist = compute_z_star_distance(z, z_star=z_ref)
    z_star_distance = z_dist["z_star_distance_mean"]

    performance_gap = max(running_best_return - mean_episode_return, 0.0)

    if smoothness_L is None:
        smoothness_L = 1.0

    chain_kl_term = c_z * sqrt_kl
    chain_proximity_term = smoothness_L * z_star_distance
    chain_rhs_unscaled = chain_kl_term + chain_proximity_term

    mu_c = max(mu_concave, 1e-6) if mu_concave is not None else 1.0
    chain_rhs_scaled = (c1 / mu_c) * chain_rhs_unscaled

    bound_unreliable = 0.0
    if pct_concave is not None and pct_concave > bound_unreliable_pct_threshold:
        bound_unreliable = 1.0

    return {
        "chain_performance_gap": performance_gap,
        "chain_z_star_distance": z_star_distance,
        "chain_grad_z_v": grad_z,
        "chain_sqrt_kl": sqrt_kl,
        "chain_kl_term": chain_kl_term,
        "chain_proximity_term": chain_proximity_term,
        "chain_rhs_unscaled": chain_rhs_unscaled,
        "chain_rhs_scaled": chain_rhs_scaled,
        "chain_bound_unreliable": bound_unreliable,
        "chain_running_best_return": running_best_return,
        "chain_mean_return": mean_episode_return,
    }


def check_chain_directionality(
    performance_gap: float,
    grad_z_v: float,
    mu: float,
    chain_rhs_unscaled: float,
    chain_rhs_scaled: float | None = None,
    c1: float = 1.0,
) -> dict[str, float]:
    """
    Directional checks (inequalities need not be tight).

    Step 1 proxy: performance_gap ≲ (c1/μ) * grad_z_v
    Step 2 proxy: grad_z_v ≲ chain_rhs (unscaled or scaled)
    """
    mu_floor = max(mu, 1e-6)
    repr_bound_lhs = performance_gap
    repr_bound_rhs = (c1 / mu_floor) * grad_z_v
    repr_bound_ratio = repr_bound_lhs / (repr_bound_rhs + 1e-8)

    grad_bound_ratio = grad_z_v / (chain_rhs_unscaled + 1e-8)

    rhs_scaled = chain_rhs_scaled if chain_rhs_scaled is not None else chain_rhs_unscaled
    grad_bound_scaled_ratio = grad_z_v / (rhs_scaled + 1e-8)

    return {
        "chain_repr_bound_ratio": repr_bound_ratio,
        "chain_grad_vs_rhs_ratio": grad_bound_ratio,
        "chain_grad_vs_rhs_scaled_ratio": grad_bound_scaled_ratio,
        "chain_repr_bound_holds_directional": float(repr_bound_ratio <= 5.0),
        "chain_grad_bound_holds_directional": float(grad_bound_ratio <= 5.0),
        "chain_grad_bound_scaled_holds_directional": float(grad_bound_scaled_ratio <= 5.0),
    }
