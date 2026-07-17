"""Batched per-sample value-head gradients dV/dZ."""

import torch
import torch.nn as nn
from torch.func import grad, vmap


def batched_value_head_grad_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Per-sample rows of dV/dZ with shape [batch, latent_dim].

    Assumes V(z)_i depends only on z_i (standard row-wise MLP / affine heads).
    """

    def value_row(z_row: torch.Tensor) -> torch.Tensor:
        return critic.value_head(z_row.unsqueeze(0)).reshape(())

    return vmap(grad(value_row))(z)
