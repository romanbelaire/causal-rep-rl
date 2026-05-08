"""
Standard MLP Policy for discrete and continuous action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network.
    
    Supports both discrete and continuous action spaces.
    For discrete: outputs logits for categorical distribution
    For continuous: outputs mean and log_std for Gaussian distribution
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        action_space_type: str = "discrete",  # "discrete" or "continuous"
    ):
        """
        Initialize MLP policy.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: List of hidden layer sizes
            activation: Activation function name ("relu", "tanh", etc.)
            action_space_type: "discrete" or "continuous"
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space_type = action_space_type
        
        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build network
        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Output heads
        if action_space_type == "discrete":
            self.action_head = nn.Linear(input_dim, action_dim)
        else:  # continuous
            self.action_mean = nn.Linear(input_dim, action_dim)
            self.action_log_std = nn.Linear(input_dim, action_dim)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim]
            
        Returns:
            For discrete: logits [batch_size, action_dim]
            For continuous: (mean, log_std) both [batch_size, action_dim]
        """
        features = self.feature_extractor(obs)
        
        if self.action_space_type == "discrete":
            logits = self.action_head(features)
            return logits
        else:
            mean = self.action_mean(features)
            log_std = self.action_log_std(features)
            # Clamp log_std for numerical stability
            log_std = torch.clamp(log_std, min=-20, max=2)
            return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor [batch_size, obs_dim] or [obs_dim]
            deterministic: If True, return deterministic action
            
        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
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
        else:  # continuous
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            
            if deterministic:
                action = mean
            else:
                action = dist.sample()
            
            # For continuous, sum log probs over action dimensions
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
            
            if obs.dim() == 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            
            return action, log_prob
    
    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions under current policy.
        
        Args:
            obs: Observations [batch_size, obs_dim]
            actions: Actions [batch_size, action_dim] or [batch_size] for discrete
            
        Returns:
            log_probs: Log probabilities [batch_size]
            entropy: Entropy [batch_size]
            value: Value estimates (if available, otherwise None)
        """
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
        else:  # continuous
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        
        return log_probs, entropy

