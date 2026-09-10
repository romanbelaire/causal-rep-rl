"""NatureCNN value critic (separate trunk from the actor)."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.architectures.encoders.nature_cnn import NatureCNN, layer_init


class NatureValueCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, int, int]):
        super().__init__()
        self.cnn = NatureCNN(obs_shape)
        self.encoder = nn.Sequential(
            self.cnn,
            layer_init(nn.Linear(self.cnn.flat_dim, 512)),
            nn.ReLU(),
        )
        self.latent_dim = 512
        self.value_head = layer_init(nn.Linear(512, 1), std=1.0)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        return z, torch.zeros_like(z)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(obs)
        return self.value_head(z)
