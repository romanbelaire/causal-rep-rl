"""
IMPALA policy architecture for scalable RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IMPALAPolicy(nn.Module):
    """
    IMPALA-style policy network with residual blocks.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        action_space_type: str = "discrete",
        num_residual_blocks: int = 2,
    ):
        """
        Initialize IMPALA policy.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
            activation: Activation function
            action_space_type: "discrete" or "continuous"
            num_residual_blocks: Number of residual blocks
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space_type = action_space_type
        
        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Input layer
        self.input_layer = nn.Linear(obs_dim, hidden_sizes[0])
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(hidden_sizes[0], activation))
        
        # Output heads
        if action_space_type == "discrete":
            self.action_head = nn.Linear(hidden_sizes[0], action_dim)
        else:
            self.action_mean = nn.Linear(hidden_sizes[0], action_dim)
            self.action_log_std = nn.Linear(hidden_sizes[0], action_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        x = self.activation(self.input_layer(obs))
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        if self.action_space_type == "discrete":
            logits = self.action_head(x)
            return logits
        else:
            mean = self.action_mean(x)
            log_std = self.action_log_std(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            if obs.dim() == 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            
            return action, log_prob
        else:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            if obs.dim() == 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            
            return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate actions under current policy."""
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy


class ResidualBlock(nn.Module):
    """Residual block for IMPALA."""
    
    def __init__(self, hidden_dim: int, activation: str = "relu"):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        out = out + residual  # Residual connection
        out = self.activation(out)
        return out

