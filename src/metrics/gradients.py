"""
Gradient magnitude and difference metrics.
"""

import torch
import torch.nn as nn


def compute_gradient_magnitude(
    model: nn.Module,
    loss_fn: callable,
    inputs: torch.Tensor,
) -> float:
    """
    Compute gradient magnitude ||∇_θ L(θ)||.
    
    Args:
        model: Model to compute gradients for
        loss_fn: Loss function that takes model and inputs
        inputs: Input tensor
        
    Returns:
        Gradient magnitude (L2 norm)
    """
    model.zero_grad()
    loss = loss_fn(model, inputs)
    loss.backward()
    
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def compute_value_gradient_magnitude(
    critic: nn.Module,
    obs: torch.Tensor,
) -> float:
    """
    Compute ||∇_θ V^π(s)|| for value function.
    
    Args:
        critic: Value function critic
        obs: Observations [N, obs_dim] or encoded representations [N, latent_dim]
              For VAE critics, obs is already encoded (latent z)
        
    Returns:
        Average gradient magnitude
    """
    # Create a dummy loss that depends on critic parameters
    # For VAE critics, if obs is already encoded (latent z), use value_head directly
    # Otherwise, use critic.forward() which handles encoding
    if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
        # VAE critic: check if obs is already encoded (matches latent_dim)
        if obs.shape[-1] == critic.latent_dim:
            # obs is already encoded (latent z): use value_head directly
            values = critic.value_head(obs).sum()  # Sum to get scalar
        else:
            # obs is raw: use critic.forward() which handles encoding
            values = critic(obs).sum()  # Sum to get scalar
    else:
        # ICNN/Feedforward: obs might be raw or encoded, critic handles it
        values = critic(obs).sum()  # Sum to get scalar
    
    # Compute gradients w.r.t. parameters
    critic.zero_grad()
    values.backward()
    
    # Compute L2 norm of gradients
    total_norm = 0.0
    for param in critic.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def compute_value_gradient_difference(
    critic1: nn.Module,
    critic2: nn.Module,
    obs: torch.Tensor,
) -> float:
    """
    Compute ||∇ V^{π'}(s) - ∇ V^{π}(s)||.
    
    Args:
        critic1: First value function (e.g., V^{π'})
        critic2: Second value function (e.g., V^{π})
        obs: Observations [N, obs_dim] or encoded representations [N, latent_dim]
              For VAE critics, obs is already encoded (latent z)
        
    Returns:
        Average gradient difference magnitude
    """
    # Compute gradients for both critics
    # For VAE critics, if obs is already encoded (latent z), use value_head directly
    if hasattr(critic1, 'value_head') and hasattr(critic1, 'latent_dim'):
        if obs.shape[-1] == critic1.latent_dim:
            values1 = critic1.value_head(obs).sum()
        else:
            values1 = critic1(obs).sum()
    else:
        values1 = critic1(obs).sum()
    
    if hasattr(critic2, 'value_head') and hasattr(critic2, 'latent_dim'):
        if obs.shape[-1] == critic2.latent_dim:
            values2 = critic2.value_head(obs).sum()
        else:
            values2 = critic2(obs).sum()
    else:
        values2 = critic2(obs).sum()
    
    critic1.zero_grad()
    critic2.zero_grad()
    values1.backward()
    values2.backward()
    
    # Compute difference
    diff_norm = 0.0
    for p1, p2 in zip(critic1.parameters(), critic2.parameters()):
        if p1.grad is not None and p2.grad is not None:
            diff = p1.grad - p2.grad
            diff_norm += diff.norm(2).item() ** 2
    
    diff_norm = diff_norm ** (1.0 / 2)
    return diff_norm


def _critic_value_on_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Scalar sum of V(z) for autograd through z."""
    if hasattr(critic, "value_head"):
        return critic.value_head(z).sum()
    return critic(z).sum()


def compute_value_jacobian_z(critic: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """
    Per-sample value-head Jacobian rows J_phi = dV/dZ, shape [N, d].

    For scalar V(z), row i is grad V(z_i) w.r.t. z_i.
    """
    rows = []
    for i in range(z.shape[0]):
        zi = z[i : i + 1].detach().requires_grad_(True)
        vi = _critic_value_on_z(critic, zi)
        gi = torch.autograd.grad(vi, zi, retain_graph=False)[0]
        rows.append(gi.squeeze(0))
    return torch.stack(rows, dim=0)


def compute_value_gradient_z_magnitude(critic: nn.Module, z: torch.Tensor) -> float:
    """
    Compute mean ||∇_Z V(Z)|| over a batch of latent representations.

    This is the quantity in Theorem reprbound and the bounding chain (Theorem 4).
    """
    grad_v = compute_value_jacobian_z(critic, z)
    return torch.norm(grad_v, dim=1).mean().item()

