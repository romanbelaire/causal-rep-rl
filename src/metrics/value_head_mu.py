"""
Analytic μ tracking for structured VAE value heads (μ w.r.t. encoder latent Z).
"""

import torch
import torch.nn as nn

from src.metrics.value_head_types import (
    is_affine_vae_value_head,
    is_quadratic_latent_value_head,
    is_squared_norm_value_head,
)
def get_analytic_mu_latent(critic: nn.Module) -> torch.Tensor | None:
    """μ = 2 σ_min(A)² for quadratic PSD on Z (constant ∇²_Z V)."""
    if is_quadratic_latent_value_head(critic):
        return critic.value_head.analytic_mu_latent()
    return None


def compute_mu_latent_autodiff(critic: nn.Module, Z: torch.Tensor, num_samples: int = 3) -> torch.Tensor:
    """Smallest eigenvalue of ∇²_Z V via autodiff on a batch of latents."""
    from src.utils.representation_loss import compute_smallest_eigenvalue_hessian_z

    Z_grad = Z.detach().requires_grad_(True)
    V = critic.value_head(Z_grad)
    return compute_smallest_eigenvalue_hessian_z(V, Z_grad, critic, num_samples=num_samples)


def get_mu_for_repr_loss(
    critic: nn.Module,
    Z: torch.Tensor,
    num_jacobian_samples: int = 3,
) -> torch.Tensor | None:
    """
    Cheap μ for representation-loss weighting when available.

    - Quadratic latent: analytic μ w.r.t. Z (= 2σ_min(A)²).
    - Squared norm: min-batch Jacobian proxy 2 λ_min(J_f^T J_f) w.r.t. Z.
    - Otherwise: None → full Hessian autodiff in caller.
    """
    mu_latent = get_analytic_mu_latent(critic)
    if mu_latent is not None:
        return mu_latent

    if is_squared_norm_value_head(critic):
        batch_size = Z.shape[0]
        n = min(num_jacobian_samples, batch_size)
        idx = torch.randperm(batch_size, device=Z.device)[:n]
        return critic.value_head.mu_jacobian_proxy(Z[idx]).min()

    return None


def value_head_mu_stats(
    critic: nn.Module,
    Z: torch.Tensor,
    num_autodiff_samples: int = 3,
    enforce_quadratic_agreement: bool = False,
    rtol: float = 1e-2,
    atol: float = 1e-5,
) -> dict[str, float]:
    """Scalars to log each training / metric step."""
    stats: dict[str, float] = {}

    mu_a = get_analytic_mu_latent(critic)
    if mu_a is not None:
        stats["mu_latent_analytic"] = mu_a.item()
        stats["mu_analytic"] = mu_a.item()

    if is_quadratic_latent_value_head(critic) or is_squared_norm_value_head(critic):
        mu_h = compute_mu_latent_autodiff(critic, Z, num_samples=num_autodiff_samples)
        stats["mu_latent_autodiff"] = mu_h.item()
        if mu_a is not None:
            denom = max(abs(mu_h.item()), atol)
            stats["mu_latent_rel_err"] = abs(mu_a.item() - mu_h.item()) / denom
            if enforce_quadratic_agreement and stats["mu_latent_rel_err"] > rtol:
                raise RuntimeError(
                    f"μ_latent analytic ({mu_a.item():.6e}) != autodiff ({mu_h.item():.6e}); "
                    f"rel_err={stats['mu_latent_rel_err']:.6e} > rtol={rtol}. "
                    "Nonlinear layer between Z and quadratic head?"
                )

    if is_squared_norm_value_head(critic):
        mu_j = critic.value_head.mu_jacobian_proxy_batch_min(Z)
        stats["mu_jacobian_proxy"] = mu_j.item()
        if "mu_analytic" not in stats:
            stats["mu_analytic"] = mu_j.item()

    if is_affine_vae_value_head(critic):
        stats["mu_analytic"] = 0.0
        stats["mu_latent_analytic"] = 0.0
        stats["affine_by_construction"] = 1.0

    return stats
