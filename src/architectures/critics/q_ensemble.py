"""Discrete action-value ensemble Q_j(z) in R^{|A|}."""

import torch
import torch.nn as nn


class QEnsemble(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        n_actions: int,
        n_members: int,
        hidden_sizes: list[int],
    ):
        super().__init__()
        if n_members < 2:
            raise RuntimeError(f"Q ensemble needs >= 2 members, got {n_members}")
        self.n_members = n_members
        self.n_actions = n_actions
        heads = []
        for _ in range(n_members):
            layers: list[nn.Module] = []
            dim = latent_dim
            for hidden in hidden_sizes:
                layers.append(nn.Linear(dim, hidden))
                layers.append(nn.ReLU())
                dim = hidden
            layers.append(nn.Linear(dim, n_actions))
            heads.append(nn.Sequential(*layers))
        self.heads = nn.ModuleList(heads)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z [B, D] -> Q [B, J, A]."""
        stacked = [head(z) for head in self.heads]
        return torch.stack(stacked, dim=1)
