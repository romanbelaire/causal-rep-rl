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
        super().__init__()
        self.obs_dim = obs_dim
        
        self.activation = activation_module(activation)
        
        # Build network
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size
        
        # Output layer (single value)
        layers.append(nn.Linear(input_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer to output small values
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]
            
        Returns:
            Value estimates [batch_size, 1] or [1]
        """
        return self.network(obs)

