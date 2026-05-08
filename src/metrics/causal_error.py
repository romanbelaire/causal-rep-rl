"""
Causal prediction error: ||Z*(s) - Z(s)||
"""

import torch
import torch.nn as nn


def compute_causal_prediction_error(
    critic: nn.Module,
    ground_truth_repr_fn: callable,
    obs: torch.Tensor,
    repr_net: nn.Module = None,
) -> dict:
    """
    Compute causal prediction error ||Z*(s) - Z(s)||.
    
    Args:
        critic: Value function critic (or representation extractor)
        ground_truth_repr_fn: Function that takes obs and returns ground-truth Z*(s)
        obs: Observations [N, obs_dim] or representations [N, repr_dim] if already encoded
        repr_net: Optional representation network to extract z from obs (if obs is not already encoded)
        
    Returns:
        Dictionary with:
            - error: Mean L2 error
            - errors: Per-sample errors [N]
            - max_error: Maximum error
            - min_error: Minimum error
    """
    # Get device from observations
    device = obs.device
    
    # Note: obs might already be encoded (z) if called from metric evaluator with repr_net
    # We need raw observations for ground truth, but if obs is already z, we use it directly
    # For ground truth, we'll try to use obs as-is (assuming it's raw observations)
    # If ground_truth_repr_fn needs raw obs, caller should pass raw obs separately
    
    # Get ground-truth representation
    z_star_list = []
    for i in range(len(obs)):
        z_star = ground_truth_repr_fn(obs[i])
        # Ensure ground-truth representation is on same device as observations
        if isinstance(z_star, torch.Tensor) and z_star.device != device:
            z_star = z_star.to(device)
        z_star_list.append(z_star)
    z_star = torch.stack(z_star_list)  # [N, repr_dim]
    
    # Ensure z_star is on correct device
    z_star = z_star.to(device)
    
    # Get learned representation
    # If repr_net is provided, use it (obs might be raw or already encoded)
    # If obs is already encoded (z), repr_net will just re-encode it (should be same)
    if repr_net is not None:
        # Use representation network to extract z from observations
        with torch.no_grad():
            z = repr_net(obs)  # s -> z (or z -> z if already encoded)
    elif hasattr(critic, 'get_latent_representation'):
        z = critic.get_latent_representation(obs)  # [N, latent_dim]
    elif hasattr(critic, 'encode'):
        mu, _ = critic.encode(obs)
        z = mu  # [N, latent_dim]
    else:
        # For other critics, use intermediate features
        # Extract from last hidden layer
        with torch.no_grad():
            # Try to get features from network
            if hasattr(critic, 'network'):
                # Handle compiled networks (OptimizedModule) and regular Sequential
                try:
                    # Try to access as list/sequence first
                    if isinstance(critic.network, (list, nn.Sequential)):
                        network_layers = critic.network
                    elif hasattr(critic, '_original_network'):
                        # Use original network if available (for compiled networks)
                        network_layers = critic._original_network
                    else:
                        # For OptimizedModule, try to convert to list
                        # This may not work, so we'll fall back
                        network_layers = list(critic.network)
                    
                    # Extract features from all but last layer
                    features = obs
                    # Handle slicing: if it's a Sequential or list, we can slice
                    if isinstance(network_layers, nn.Sequential):
                        # Sequential supports slicing in newer PyTorch versions
                        try:
                            for layer in network_layers[:-1]:
                                features = layer(features)
                        except (TypeError, AttributeError):
                            # Fallback: iterate manually
                            layers_list = list(network_layers.children())
                            for layer in layers_list[:-1]:
                                features = layer(features)
                    elif isinstance(network_layers, list):
                        for layer in network_layers[:-1]:
                            features = layer(features)
                    else:
                        # Can't extract features, use fallback
                        raise AttributeError("Cannot extract features from network")
                    
                    z = features
                except (TypeError, AttributeError, IndexError):
                    # Fallback: use observation itself if we can't extract features
                    # This happens for ICNN (which takes z as input, not obs) or compiled networks
                    z = obs
            else:
                # Fallback: use observation itself
                z = obs
    
    # Ensure same dimension (pad or project if needed)
    if z.shape[1] != z_star.shape[1]:
        min_dim = min(z.shape[1], z_star.shape[1])
        z = z[:, :min_dim]
        z_star = z_star[:, :min_dim]
    
    # Ensure both tensors are on same device before computing error
    z_star = z_star.to(z.device)
    
    # Compute L2 error
    errors = torch.norm(z - z_star, dim=1)  # [N]
    
    return {
        "error": errors.mean().item(),
        "errors": errors,
        "max_error": errors.max().item(),
        "min_error": errors.min().item(),
        "std_error": errors.std().item(),
    }

