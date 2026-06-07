"""
Training losses toward live expert Z* (theory validation v3).

- L_kappa: maximize directional κ_concave along Z → Z*
- L_distill: ‖Z_online − Z_ref‖²
"""

import torch
import torch.nn as nn

from src.metrics.convexity_validation import _value_on_z_batch
from src.metrics.value_head_types import is_affine_vae_value_head


def effective_intervention_loss_coef(
    base_coef: float,
    warmup_epochs: int,
    training_epoch: int | None,
) -> float:
    """Linear warmup of κ / distill loss coefficients (same pattern as repr loss)."""
    if base_coef <= 0.0:
        return 0.0
    if warmup_epochs <= 0 or training_epoch is None:
        return base_coef
    scale = min(1.0, float(training_epoch) / float(warmup_epochs))
    return base_coef * scale


def compute_z_distill_loss(
    z_online: torch.Tensor,
    z_ref: torch.Tensor,
    coef: float,
) -> tuple[torch.Tensor, dict]:
    """
    L_distill = coef * mean(||Z_online − Z_ref||²).

    z_ref must be detached (expert encode, no grad).
    """
    if z_online.shape != z_ref.shape:
        raise ValueError(f"Shape mismatch: z_online {z_online.shape} vs z_ref {z_ref.shape}")

    mse = ((z_online - z_ref) ** 2).sum(dim=1).mean()
    loss = coef * mse
    return loss, {
        "train_z_distill_mse": mse.item(),
        "train_z_distill_loss": loss.item(),
    }


def compute_kappa_directional_loss(
    critic: nn.Module,
    z_online: torch.Tensor,
    z_ref: torch.Tensor,
    coef: float,
    epsilon: float = 0.01,
    min_distance: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """
    Maximize mean κ_concave toward Z* along Δ = (Z − Z*) / ‖Z − Z*‖.

    Loss = −coef * mean(κ_concave) on valid batch rows.
    """
    if is_affine_vae_value_head(critic):
        raise ValueError(
            "kappa_directional_loss requires non-affine value head; got affine VAE head"
        )

    if z_online.dim() == 1:
        z_online = z_online.unsqueeze(0)
    if z_ref.dim() == 1:
        z_ref = z_ref.unsqueeze(0)
    if z_online.shape != z_ref.shape:
        raise ValueError(f"Shape mismatch: z {z_online.shape} vs z_ref {z_ref.shape}")

    delta_vec = z_online - z_ref
    distances = torch.norm(delta_vec, dim=1)
    valid = distances > min_distance
    if not valid.any():
        raise ValueError(
            "No samples with ‖Z − Z*‖ > min_distance; cannot compute kappa directional loss"
        )

    z_v = z_online[valid]
    z_s = z_ref[valid]
    delta = delta_vec[valid] / distances[valid].unsqueeze(1)

    z_s_grad = z_s.detach().requires_grad_(True)
    v0 = _value_on_z_batch(critic, z_s_grad)
    grad_v = torch.autograd.grad(v0.sum(), z_s_grad, create_graph=True)[0]
    dir_deriv = (grad_v * delta).sum(dim=1)

    z_pert = z_s_grad + epsilon * delta
    v_pert = _value_on_z_batch(critic, z_pert)
    denom = 0.5 * epsilon ** 2
    kappa_batch = (v_pert - v0 - epsilon * dir_deriv) / denom
    kappa_concave_batch = (-kappa_batch).clamp(min=0.0)
    kappa_concave_mean = kappa_concave_batch.mean()

    loss = -coef * kappa_concave_mean
    return loss, {
        "train_kappa_concave_mean": kappa_concave_mean.item(),
        "train_kappa_directional_loss": loss.item(),
    }
