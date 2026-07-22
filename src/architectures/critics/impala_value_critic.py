"""Impala CNN value critic for Procgen PPO baseline."""

import torch
import torch.nn as nn

from src.architectures.encoders.impala_cnn import ProcgenImpalaEncoder


class ImpalaValueCritic(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        emb_size: int = 256,
        depths: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__()
        self.encoder = ProcgenImpalaEncoder(obs_shape, depths=depths, emb_size=emb_size)
        self.latent_dim = emb_size
        self.value_head = nn.Linear(emb_size, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Z, dummy_log_std) for metric-evaluator API compatibility."""
        z = self.encoder(obs)
        dummy_log_std = torch.zeros_like(z)
        return z, dummy_log_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(obs)
        return self.value_head(z)
