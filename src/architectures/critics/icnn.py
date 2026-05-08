"""
Input Convex Neural Network (ICNN) Critic.
Uses convex-init library for ICNN implementation.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add convex-init to path
_convex_init_path = Path(__file__).parent.parent.parent.parent / "convex-init"
if _convex_init_path.exists():
    sys.path.insert(0, str(_convex_init_path))
    from convex_modules import ConvexLinear, ExponentialPositivity, LazyClippedPositivity
    from convex_init import ConvexInitialiser
else:
    raise ImportError(f"convex-init directory not found at {_convex_init_path}")


class ICNNCritic(nn.Module):
    """
    Input Convex Neural Network value function critic.
    
    Takes representation z as input and outputs value v.
    Enforces convexity in the representation-value mapping using positive weights
    and proper initialization. Supports strong convexity via quadratic term.
    
    Architecture: z -> v (where z comes from representation network s -> z)
    """
    
    def __init__(
        self,
        repr_dim: int,  # Changed from obs_dim to repr_dim
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        positivity: str = "exp",  # "exp" or "clip"
        mu: float = 0.0,  # Strong convexity parameter (0 = just convex)
        use_convex_init: bool = True,
    ):
        """
        Initialize ICNN critic.
        
        Args:
            repr_dim: Representation dimension (input z dimension)
            hidden_sizes: List of hidden layer sizes
            activation: Activation function ("relu" or "softplus")
            positivity: Positivity function ("exp" or "clip")
            mu: Strong convexity parameter (adds mu/2 * ||z||^2 term)
            use_convex_init: Use convex initialization
        """
        super().__init__()
        self.repr_dim = repr_dim
        self.mu = mu
        
        # Positivity function
        # Note: "clip" (LazyClippedPositivity) is generally faster than "exp" (ExponentialPositivity)
        # because it uses clamp() instead of exp(). However, "exp" may provide better gradients.
        # For better efficiency, consider using "clip" if performance is critical.
        if positivity == "exp":
            self.positivity = ExponentialPositivity()
        elif positivity == "clip":
            self.positivity = LazyClippedPositivity()
        else:
            raise ValueError(f"Unknown positivity: {positivity}")
        
        # Activation function (must preserve convexity)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "softplus":
            self.activation = nn.Softplus()
        else:
            raise ValueError(f"ICNN activation must be 'relu' or 'softplus', got {activation}")
        
        # First layer (regular linear, can have negative weights)
        layers = []
        if len(hidden_sizes) == 0:
            raise ValueError("ICNN requires at least one hidden layer")
        
        first_hidden = hidden_sizes[0]
        self.first_layer = nn.Linear(repr_dim, first_hidden)  # Changed from obs_dim to repr_dim
        layers.append(self.first_layer)
        layers.append(self.activation)
        
        # Convex layers (must have positive weights)
        input_dim = first_hidden
        convex_layers = []
        for hidden_size in hidden_sizes[1:]:
            convex_layer = ConvexLinear(input_dim, hidden_size, positivity=self.positivity)
            convex_layers.append(convex_layer)
            layers.append(convex_layer)
            layers.append(self.activation)
            input_dim = hidden_size
        
        # Output layer (convex, single value)
        output_layer = ConvexLinear(input_dim, 1, positivity=self.positivity)
        convex_layers.append(output_layer)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
        self.convex_layers = convex_layers
        
        # Store original network before compilation (needed for Hessian computation)
        # torch.compile() doesn't work with second-order gradients (create_graph=True)
        self._original_network = self.network
        
        # Initialize
        # First layer: standard initialization
        nn.init.kaiming_uniform_(self.first_layer.weight, nonlinearity="linear")
        nn.init.zeros_(self.first_layer.bias)
        
        # Convex layers: use convex initialization
        if use_convex_init:
            convex_init = ConvexInitialiser()
            for layer in self.convex_layers:
                convex_init(layer.weight, layer.bias)
        else:
            # Standard initialization (will be made positive by positivity function)
            for layer in self.convex_layers:
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="linear")
                nn.init.zeros_(layer.bias)
        
        # Compile the network for better performance (PyTorch 2.0+)
        # This can significantly speed up the forward pass
        # Note: Compiled networks don't work with second-order gradients (Hessian computation)
        # We store the original network and use it when needed
        self._is_compiled = False
        try:
            if hasattr(torch, 'compile') and torch.__version__ >= "2.0.0":
                self.network = torch.compile(self.network, mode="reduce-overhead")
                self._is_compiled = True
        except Exception:
            # If compilation fails, continue without it
            pass
    
    def forward(self, z: torch.Tensor, use_original: bool = False) -> torch.Tensor:
        """
        Forward pass: z -> v.
        
        Args:
            z: Representation tensor [batch_size, repr_dim] or [repr_dim]
            use_original: If True, use original (uncompiled) network. 
                         Needed for second-order gradients (Hessian computation).
            
        Returns:
            Value estimates [batch_size, 1] or [1]
        """
        # Use original network if requested (for Hessian computation) or if not compiled
        network_to_use = self._original_network if (use_original or not self._is_compiled) else self.network
        value = network_to_use(z)
        
        # Add strong convexity term if mu > 0
        # Optimized: use in-place operations and efficient norm computation
        if self.mu > 0:
            # More efficient: use norm squared instead of sum of squares
            # torch.norm(z, dim=-1, keepdim=True) ** 2 is equivalent but can be slower
            # Direct computation is faster: z.pow(2).sum(dim=-1, keepdim=True)
            # But for numerical stability and efficiency, we can use:
            if z.dim() == 1:
                # Single sample: [repr_dim] -> [1]
                z_norm_sq = torch.dot(z, z).unsqueeze(0)
            else:
                # Batch: [batch_size, repr_dim] -> [batch_size, 1]
                z_norm_sq = torch.sum(z * z, dim=-1, keepdim=True)
            quadratic_term = (self.mu * 0.5) * z_norm_sq
            value = value + quadratic_term
        
        return value

