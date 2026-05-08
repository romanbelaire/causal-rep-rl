"""
Forward model for temporal contrastive learning.
Predicts next representation Z_{t+1} from current representation Z_t and action a_t.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ForwardModel(nn.Module):
    """
    Forward model: predicts next representation from current representation and action.
    
    Architecture: MLP that takes (Z_t, action) and outputs Z_{t+1}
    """
    
    def __init__(
        self,
        repr_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
    ):
        """
        Initialize forward model.
        
        Args:
            repr_dim: Representation dimension (Z dimension)
            action_dim: Action dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network: (Z_t, action) -> Z_{t+1}
        layers = []
        input_dim = repr_dim + action_dim  # Concatenate Z_t and action
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size
        
        # Output layer: predict next representation
        layers.append(nn.Linear(input_dim, repr_dim))
        self.network = nn.Sequential(*layers)
        
        # Initialize output layer
        nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, z_t: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next representation from current representation and action.
        
        Args:
            z_t: Current representation [batch_size, repr_dim] or [repr_dim]
            action: Action [batch_size, action_dim] or [action_dim]
                    For discrete actions, should be one-hot encoded or integer indices
        
        Returns:
            Predicted next representation Z_{t+1} [batch_size, repr_dim] or [repr_dim]
        """
        # Handle single sample case
        if z_t.dim() == 1:
            z_t = z_t.unsqueeze(0)
            action = action.unsqueeze(0) if isinstance(action, torch.Tensor) else torch.tensor([action]).unsqueeze(0)
            single_sample = True
        else:
            single_sample = False
        
        # Handle discrete actions: convert to one-hot if needed
        if action.dtype == torch.long or (isinstance(action, torch.Tensor) and action.dim() == 1 and action.dtype != torch.float32):
            # Discrete action indices: convert to one-hot
            action_onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        elif action.dim() == 1:
            # Single discrete action
            action_onehot = F.one_hot(action.long().unsqueeze(0), num_classes=self.action_dim).float()
        else:
            # Already one-hot or continuous
            action_onehot = action.float()
        
        # Ensure action_onehot has correct shape
        if action_onehot.dim() == 1:
            action_onehot = action_onehot.unsqueeze(0)
        
        # Concatenate Z_t and action
        z_action = torch.cat([z_t, action_onehot], dim=-1)  # [batch_size, repr_dim + action_dim]
        
        # Predict next representation
        z_next_pred = self.network(z_action)  # [batch_size, repr_dim]
        
        # Handle single sample case
        if single_sample:
            z_next_pred = z_next_pred.squeeze(0)
        
        return z_next_pred

