"""Gaussian latent dynamics model for DBC bisimulation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.activation import activation_module


class LatentDynamicsModel(nn.Module):
    """
    Diagonal Gaussian p(z' | z, a).

    Outputs mean and log-std for next latent given current latent and discrete action.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_sizes: list[int],
        activation: str = "gelu",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        act = activation_module(activation)
        input_dim = latent_dim + action_dim
        layers: list[nn.Module] = []
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(act)
            input_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.fc_mean = nn.Linear(input_dim, latent_dim)
        self.fc_log_std = nn.Linear(input_dim, latent_dim)

    def _action_one_hot(self, actions: torch.Tensor) -> torch.Tensor:
        return F.one_hot(actions.long(), num_classes=self.action_dim).float()

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_oh = self._action_one_hot(actions)
        h = self.trunk(torch.cat([z, a_oh], dim=1))
        mean = self.fc_mean(h)
        log_std = self.fc_log_std(h).clamp(min=-5.0, max=2.0)
        return mean, log_std

    def gaussian_w2(
        self,
        mean1: torch.Tensor,
        log_std1: torch.Tensor,
        mean2: torch.Tensor,
        log_std2: torch.Tensor,
    ) -> torch.Tensor:
        """Closed-form W2 distance between diagonal Gaussians."""
        std1 = torch.exp(log_std1)
        std2 = torch.exp(log_std2)
        mean_diff_sq = (mean1 - mean2).pow(2).sum(dim=1)
        std_diff_sq = (std1 - std2).pow(2).sum(dim=1)
        return torch.sqrt(mean_diff_sq + std_diff_sq + 1e-8)
