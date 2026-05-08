"""
CNN-IMPALA policy architecture for image observations.
Processes images in HWC format using CNN layers, then IMPALA-style residual blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CNNIMPALAPolicy(nn.Module):
    """
    CNN-IMPALA policy network for image observations.
    
    Architecture:
    - CNN feature extractor: [H, W, C] -> CNN -> flattened features
    - IMPALA-style residual blocks
    - Action head
    """
    
    def __init__(
        self,
        obs_shape: tuple,  # (H, W, C) for images
        action_dim: int,
        cnn_channels: list[int] = [32, 64, 64],
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        action_space_type: str = "discrete",
        num_residual_blocks: int = 2,
    ):
        """
        Initialize CNN-IMPALA policy.
        
        Args:
            obs_shape: Observation shape (H, W, C) for images
            action_dim: Action dimension
            cnn_channels: List of CNN output channels [ch1, ch2, ch3, ...]
            hidden_sizes: Hidden layer sizes for MLP after CNN
            activation: Activation function
            action_space_type: "discrete" or "continuous"
            num_residual_blocks: Number of residual blocks
        """
        super().__init__()
        self.obs_shape = obs_shape
        H, W, C = obs_shape
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
        
        # CNN feature extractor
        # Input: [batch_size, C, H, W] (will permute from HWC)
        cnn_layers = []
        in_channels = C
        current_h, current_w = H, W
        
        for out_channels in cnn_channels:
            cnn_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            cnn_layers.append(self.activation)
            # Optional: add stride=2 for downsampling (reduce spatial size)
            if current_h > 4 and current_w > 4:  # Only downsample if large enough
                cnn_layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))
                cnn_layers.append(self.activation)
                current_h = math.ceil(current_h / 2)
                current_w = math.ceil(current_w / 2)
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Compute CNN output dimension
        # Test with dummy input to get output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            dummy_output = self.cnn(dummy_input)
            cnn_output_dim = dummy_output.view(1, -1).shape[1]
        
        # MLP layers after CNN (IMPALA-style)
        # Input layer
        self.input_layer = nn.Linear(cnn_output_dim, hidden_sizes[0])
        
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
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
                - [batch_size, H, W, C] (HWC format)
                - [H, W, C] (single observation)
        """
        # Handle input format
        if obs.dim() == 3:
            # Single observation: [H, W, C] -> [1, H, W, C]
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Convert from HWC to CHW for Conv2d: [batch_size, H, W, C] -> [batch_size, C, H, W]
        if obs.shape[-1] == self.obs_shape[2]:  # Last dim is channels
            obs = obs.permute(0, 3, 1, 2)  # [batch_size, H, W, C] -> [batch_size, C, H, W]
        
        # CNN feature extraction
        cnn_features = self.cnn(obs)  # [batch_size, channels, H', W']
        
        # Flatten CNN features
        batch_size = cnn_features.shape[0]
        cnn_features_flat = cnn_features.view(batch_size, -1)  # [batch_size, cnn_output_dim]
        
        # MLP layers (IMPALA-style)
        x = self.activation(self.input_layer(cnn_features_flat))
        
        # Apply residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        if self.action_space_type == "discrete":
            logits = self.action_head(x)
            if squeeze_output:
                logits = logits.squeeze(0)
            return logits
        else:
            mean = self.action_mean(x)
            log_std = self.action_log_std(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            if squeeze_output:
                mean = mean.squeeze(0)
                log_std = log_std.squeeze(0)
            return mean, log_std
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        if obs.dim() == 3:
            obs = obs.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            if squeeze_output:
                logits = logits.unsqueeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
            
            log_prob = dist.log_prob(action)
            
            if squeeze_output:
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
            
            if squeeze_output:
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

