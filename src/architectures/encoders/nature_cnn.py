"""NatureCNN trunk (CleanRL / Moalla Atari)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Linear | nn.Conv2d, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Conv stack ending at flattened features (pre-512). Obs scaled by /255 inside."""

    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        c, h, w = obs_shape
        if (h, w) != (84, 84):
            raise ValueError(f"NatureCNN expects 84x84 frames, got {obs_shape}")
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            self.flat_dim = int(self.conv(dummy / 255.0).shape[1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return flattened conv features (actor_preactivation / critic pre-512)."""
        return self.conv(obs / 255.0)
