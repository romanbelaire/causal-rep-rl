"""Information-value critic U(z, a) in R^{|A|}."""

import torch
import torch.nn as nn


class InformationCritic(nn.Module):
    def __init__(self, latent_dim: int, n_actions: int, hidden_sizes: list[int]):
        super().__init__()
        self.n_actions = n_actions
        layers: list[nn.Module] = []
        dim = latent_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(dim, hidden))
            layers.append(nn.ReLU())
            dim = hidden
        layers.append(nn.Linear(dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, D] -> U [B, A]."""
        return self.net(z)
