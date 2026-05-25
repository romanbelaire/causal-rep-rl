"""Type detection for structured VAE value heads."""

import torch.nn as nn

from src.architectures.value_heads.quadratic_psd import (
    QuadraticBottleneckValueHead,
    QuadraticLatentPSDValueHead,
)
from src.architectures.value_heads.squared_norm import SquaredNormValueHead


def is_affine_vae_value_head(critic: nn.Module) -> bool:
    if not (hasattr(critic, "value_head") and hasattr(critic, "latent_dim")):
        return False
    if isinstance(critic.value_head, nn.Linear):
        return True
    if not isinstance(critic.value_head, nn.Sequential):
        return False
    linear_layers = [m for m in critic.value_head if isinstance(m, nn.Linear)]
    nonlinear_layers = [
        m
        for m in critic.value_head
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Tanh, nn.ELU))
    ]
    return len(nonlinear_layers) == 0 and len(linear_layers) == 1


def is_quadratic_latent_value_head(critic: nn.Module) -> bool:
    return hasattr(critic, "value_head") and isinstance(
        critic.value_head, QuadraticLatentPSDValueHead
    )


def is_quadratic_bottleneck_value_head(critic: nn.Module) -> bool:
    return hasattr(critic, "value_head") and isinstance(
        critic.value_head, QuadraticBottleneckValueHead
    )


def is_quadratic_psd_value_head(critic: nn.Module) -> bool:
    """Quadratic PSD head with analytic μ (latent-direct only for theory)."""
    return is_quadratic_latent_value_head(critic)


def is_squared_norm_value_head(critic: nn.Module) -> bool:
    return hasattr(critic, "value_head") and isinstance(
        critic.value_head, SquaredNormValueHead
    )
