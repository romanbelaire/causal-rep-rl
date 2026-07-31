"""Deterministic CNN encoder critic: pixels -> Impala CNN -> Z -> value_head."""

import torch
import torch.nn as nn

from src.architectures.activation import activation_module
from src.architectures.encoders.impala_cnn import ProcgenImpalaEncoder


class CNNEncoderCritic(nn.Module):
    """
    Shared causal representation for Procgen without VAE recon/KL.

    encode(obs) -> (z, dummy_log_std) for CTRO/MICo/PL API compatibility.
    forward(obs) -> value_head(z).
    """

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        latent_dim: int = 128,
        emb_size: int = 256,
        depths: tuple[int, ...] = (16, 32, 32),
        activation: str = "gelu",
        value_hidden: list[int] | None = None,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        value_hidden = value_hidden or [128, 128]

        self.encoder = ProcgenImpalaEncoder(obs_shape, depths=depths, emb_size=emb_size)
        self.fc_z = nn.Linear(emb_size, latent_dim)
        nn.init.orthogonal_(self.fc_z.weight, gain=1.0)
        nn.init.zeros_(self.fc_z.bias)

        act_module = activation_module(activation)
        layers: list[nn.Module] = []
        input_dim = latent_dim
        for hidden_size in value_hidden:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(act_module)
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        nn.init.orthogonal_(layers[-1].weight, gain=0.01)
        nn.init.zeros_(layers[-1].bias)
        self.value_head = nn.Sequential(*layers)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batched = obs.dim() == 4
        if not batched:
            obs = obs.unsqueeze(0)
        h = self.encoder(obs)
        z = self.fc_z(h)
        if not batched:
            z = z.squeeze(0)
        dummy_log_std = torch.zeros_like(z)
        return z, dummy_log_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(obs)
        return self.value_head(z)
