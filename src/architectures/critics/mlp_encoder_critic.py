"""Deterministic MLP encoder critic: obs -> Z -> value_head."""

import torch
import torch.nn as nn

from src.architectures.activation import activation_module


class MLPEncoderCritic(nn.Module):
    """
    Shared causal representation without VAE recon/KL.

    encode(obs) -> (z, dummy_log_std) for CTRO/MICo/PL API compatibility.
    forward(obs) -> value_head(z).
    """

    def __init__(
        self,
        obs_dim: int,
        encoder_hidden: list[int] = [256, 256],
        activation: str = "tanh",
    ):
        if len(encoder_hidden) == 0:
            raise ValueError("encoder_hidden must be non-empty")

        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = encoder_hidden[-1]

        act_module = activation_module(activation)
        layers: list[nn.Module] = []
        input_dim = obs_dim
        for hidden_size in encoder_hidden:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(act_module)
            input_dim = hidden_size
        self.encoder = nn.Sequential(*layers)

        self.value_head = nn.Linear(self.latent_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(obs)
        dummy_log_std = torch.zeros_like(z)
        return z, dummy_log_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(obs)
        return self.value_head(z)
