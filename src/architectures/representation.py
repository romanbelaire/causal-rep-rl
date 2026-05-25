"""
Representation network: maps observations s -> representation z.
This is a separate network that can be shared between critic and policy.

For image observations (Minigrid, Procgen): Uses CNN layers followed by linear layers.
For vector observations: Uses MLP layers.
"""

import torch
import torch.nn as nn

from src.architectures.activation import activation_module
import torch.nn.functional as F
import math


class RepresentationNetwork(nn.Module):
    """
    Representation network that maps observations to representations.
    
    This network learns a causal representation z from observations s.
    Can be used by both critic (z -> v) and policy (z -> actions).
    
    For image observations: CNN -> ReLU -> Linear layers -> z
    For vector observations: Linear layers -> z
    """
    
    def __init__(
        self,
        obs_dim: int,
        repr_dim: int = 512,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        obs_shape: tuple = None,  # (H, W, C) for images, None for vectors
        use_cnn: bool = None,  # Auto-detect from obs_shape if None
    ):
        """
        Initialize representation network.
        
        Args:
            obs_dim: Observation dimension (flattened)
            repr_dim: Representation dimension (output), default 512
            hidden_sizes: List of hidden layer sizes for MLP/linear layers
            activation: Activation function name ("relu", "tanh", etc.)
            obs_shape: Observation shape (H, W, C) for images, None for vectors
            use_cnn: Whether to use CNN (auto-detect from obs_shape if None)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.repr_dim = repr_dim
        self.obs_shape = obs_shape
        
        # Determine if we should use CNN
        if use_cnn is None:
            use_cnn = obs_shape is not None and len(obs_shape) == 3
        
        self.use_cnn = use_cnn
        
        # Activation function
        self.activation = activation_module(activation)
        
        if use_cnn:
            # CNN-based representation network for image observations
            # Architecture: CNN layers -> ReLU -> Linear layers -> z
            H, W, C = obs_shape
            
            # CNN feature extractor
            # Standard architecture: 3x3 conv -> ReLU -> 3x3 conv -> ReLU -> flatten
            self.cnn = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=3, stride=1, padding=1),
                self.activation,
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
                self.activation,
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                self.activation,
            )
            
            # Compute CNN output size
            # After conv layers: H/2 x W/2 x 64
            cnn_h = math.ceil(H / 2)
            cnn_w = math.ceil(W / 2)
            cnn_output_dim = cnn_h * cnn_w * 64
            
            # Linear layers after CNN
            linear_layers = []
            input_dim = cnn_output_dim
            for hidden_size in hidden_sizes:
                linear_layers.append(nn.Linear(input_dim, hidden_size))
                linear_layers.append(self.activation)
                input_dim = hidden_size
            
            # Output layer (representation)
            linear_layers.append(nn.Linear(input_dim, repr_dim))
            
            self.linear = nn.Sequential(*linear_layers)
            
            # Initialize output layer
            nn.init.orthogonal_(self.linear[-1].weight, gain=0.01)
            nn.init.zeros_(self.linear[-1].bias)
        else:
            # MLP-based representation network for vector observations
            layers = []
            input_dim = obs_dim
            for hidden_size in hidden_sizes:
                layers.append(nn.Linear(input_dim, hidden_size))
                layers.append(self.activation)
                input_dim = hidden_size
            
            # Output layer (representation)
            layers.append(nn.Linear(input_dim, repr_dim))
            
            self.network = nn.Sequential(*layers)
            
            # Initialize output layer
            nn.init.orthogonal_(self.network[-1].weight, gain=0.01)
            nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: s -> z.
        
        Args:
            obs: Observation tensor
                - For images: [batch_size, H, W, C] or [H, W, C] (will be reshaped)
                - For vectors: [batch_size, obs_dim] or [obs_dim]
            
        Returns:
            Representation z [batch_size, repr_dim] or [repr_dim]
        """
        if self.use_cnn:
            # Handle image observations
            # Input is flattened from MinigridWrapper: [batch_size, H*W*C] or [H*W*C]
            H, W, C = self.obs_shape
            
            # Track original input dimension to handle single observation case
            original_dim = obs.dim()
            was_single = (original_dim == 1)
            
            # Reshape flattened observations back to image format
            if obs.dim() == 2:
                # Batch: [batch_size, H*W*C] -> [batch_size, H, W, C]
                batch_size = obs.shape[0]
                obs = obs.reshape(batch_size, H, W, C)
            elif obs.dim() == 1:
                # Single: [H*W*C] -> [1, H, W, C]
                obs = obs.reshape(1, H, W, C)
            
            # Convert to CHW format for Conv2d: [batch_size, H, W, C] -> [batch_size, C, H, W]
            # PyTorch Conv2d expects (N, C, H, W) format
            if obs.dim() == 4:
                obs = obs.permute(0, 3, 1, 2).contiguous()  # [batch_size, H, W, C] -> [batch_size, C, H, W]
            elif obs.dim() == 3:
                obs = obs.permute(2, 0, 1).unsqueeze(0).contiguous()  # [H, W, C] -> [1, C, H, W]
            
            # CNN feature extraction
            cnn_features = self.cnn(obs)  # [batch_size, 64, H/2, W/2]
            
            # Flatten CNN features
            batch_size = cnn_features.shape[0]
            cnn_features_flat = cnn_features.reshape(batch_size, -1)  # [batch_size, cnn_output_dim]
            
            # Linear layers
            z = self.linear(cnn_features_flat)  # [batch_size, repr_dim]
            
            # Handle single observation case: squeeze batch dimension if input was single
            if was_single and z.shape[0] == 1:
                z = z.squeeze(0)  # [1, repr_dim] -> [repr_dim]
            
            return z
        else:
            # MLP for vector observations
            return self.network(obs)

