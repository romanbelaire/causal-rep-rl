"""
Verify μ_latent_analytic ≈ μ_latent_autodiff for quadratic-on-Z value heads.
"""

import torch
import torch.nn as nn

from src.metrics.value_head_mu import (
    compute_mu_latent_autodiff,
    get_analytic_mu_latent,
)
from src.metrics.value_head_types import is_quadratic_latent_value_head, is_squared_norm_value_head


def verify_mu_latent_agreement(
    critic: nn.Module,
    Z: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-5,
    num_samples: int = 3,
    raise_on_mismatch: bool | None = None,
) -> dict[str, float]:
    """
    Compare analytic and autodiff μ w.r.t. encoder latent Z.

    raise_on_mismatch: default True for quadratic_latent*, False for squared_norm.
    """
    if raise_on_mismatch is None:
        raise_on_mismatch = is_quadratic_latent_value_head(critic)

    mu_analytic = get_analytic_mu_latent(critic)
    if mu_analytic is None:
        if is_squared_norm_value_head(critic):
            mu_autodiff = compute_mu_latent_autodiff(critic, Z, num_samples=num_samples)
            return {
                "mu_latent_analytic": float("nan"),
                "mu_latent_autodiff": mu_autodiff.item(),
                "mu_latent_rel_err": float("nan"),
            }
        return {}

    mu_autodiff = compute_mu_latent_autodiff(critic, Z, num_samples=num_samples)
    mu_a = mu_analytic.item()
    mu_h = mu_autodiff.item()
    denom = max(abs(mu_h), atol)
    rel_err = abs(mu_a - mu_h) / denom

    result = {
        "mu_latent_analytic": mu_a,
        "mu_latent_autodiff": mu_h,
        "mu_latent_rel_err": rel_err,
    }

    if raise_on_mismatch and rel_err > rtol:
        raise RuntimeError(
            f"μ_latent agreement failed: analytic={mu_a:.6e}, autodiff={mu_h:.6e}, "
            f"rel_err={rel_err:.6e} > rtol={rtol}"
        )

    return result
