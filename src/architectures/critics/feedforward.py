"""
Standard Feedforward Neural Network Critic.
"""

import torch
import torch.nn as nn

from src.architectures.activation import activation_module


class FeedforwardCritic(nn.Module):
    """
    Standard multilayer perceptron value function critic.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
    ):
        """
        Initialize feedforward critic.
        
        Args:
            obs_dim: Observation dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name ("relu", "tanh", etc.)
        """
        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be non-empty")

        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = hidden_sizes[-1]

        self.activation = activation_module(activation)

        # Torso: obs -> penultimate features Z. The value function is affine in Z,
        # so these features are the representation whose geometry (feature rank,
        # PL ratio) we probe for plain PPO.
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size
        self.encoder = nn.Sequential(*layers)

        self.value_head = nn.Linear(self.latent_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=0.01)
        nn.init.zeros_(self.value_head.bias)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (Z, dummy_log_std) for metric-evaluator API compatibility."""
        z = self.encoder(obs)
        dummy_log_std = torch.zeros_like(z)
        return z, dummy_log_std

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]
            
        Returns:
            Value estimates [batch_size, 1] or [1]
        """
        z, _ = self.encode(obs)
        return self.value_head(z)

