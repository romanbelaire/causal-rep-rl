"""
Representation loss with convexity weighting.

Implements L_rep = α * (1/μ) * ||∇_Z V(Z)||²

where:
- Z is the encoded representation (from encoder or repr_net)
- V is the value function (critic output)
- μ is the smallest eigenvalue of Hessian(V) w.r.t. Z
- α is the regularization coefficient
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_representation_loss_with_convexity(
    encoder: nn.Module,
    critic: nn.Module,
    states: torch.Tensor,
    alpha: float = 0.1,
    use_convexity_weighting: bool = True,
    convexity_coef: float = 1.0,
    grad_norm_power: float = 1.0,
    hessian_compute_freq: int = 1,
    step: int = 0,
) -> tuple[torch.Tensor, dict]:
    """
    Compute representation loss with optional convexity weighting.
    
    L_rep = α * (-μ) * ||∇_Z V(Z)||²
    
    where μ is estimated as the smallest eigenvalue of the Hessian of V w.r.t. Z.
    
    Note: Hessian is computed once per batch update (on the current batch of states).
    
    Args:
        encoder: Encoder network (repr_net or VAE encoder) that maps s -> z
                  For VAE critics, pass the critic itself (it has encode method)
                  For ICNN/Feedforward, pass repr_net
        critic: Value function critic that maps z -> v
        states: Observations [batch_size, obs_dim]
        alpha: Regularization coefficient
        use_convexity_weighting: If True, weight by -μ. If False, just use gradient norm.
        convexity_coef: Coefficient to weight the μ term (default 1.0)
        grad_norm_power: Power to raise gradient norm to (default 1.0 = L2 norm, 2.0 = squared)
                         The theoretical bound uses ||∇_Z V|| (power=1.0), but squared (power=2.0) is smoother
        hessian_compute_freq: Compute Hessian every N batch updates (1 = every batch update)
        step: Current batch update number (for frequency control)
        
    Returns:
        (L_rep, stats_dict): Tuple of:
            - L_rep: Representation loss [scalar]
            - stats_dict: Dictionary with grad_norm, mu, etc.
    """
    # Get representation Z with gradients enabled
    # For VAE critics, encoder is the VAE encoder (critic itself)
    # For ICNN/Feedforward, encoder is repr_net
    if hasattr(encoder, 'encode'):
        # VAE encoder: encode() returns (mu, log_std)
        mu, log_std = encoder.encode(states)
        Z = mu  # Use mean for deterministic representation
    else:
        # repr_net: direct forward pass
        Z = encoder(states)
    
    # Ensure Z requires gradients
    Z.requires_grad_(True)
    
    # Compute value V = critic(Z)
    # For VAE critics, we need to use value_head directly since Z is already encoded
    if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
        # VAE critic: Z is already the latent, use value_head
        V = critic.value_head(Z)  # [batch_size, 1]
    else:
        # ICNN/Feedforward: Z is representation, use critic directly
        V = critic(Z)  # [batch_size, 1]
    
    # Estimate μ (smallest eigenvalue of Hessian) if requested
    # Hessian is computed once per batch update on the current batch
    # MUST compute Hessian BEFORE computing first gradient to avoid graph reuse issues
    mu_estimate = None
    if use_convexity_weighting:
        if step % hessian_compute_freq == 0:
            mu_estimate = compute_smallest_eigenvalue_hessian_z(V, Z, critic)
            if torch.isnan(mu_estimate) or torch.isinf(mu_estimate):
                raise ValueError(f"Hessian computation returned invalid μ: {mu_estimate}")
        else:
            raise ValueError(f"use_convexity_weighting=True requires hessian_compute_freq=1 (compute every batch update), got {hessian_compute_freq}")
    
    # Compute gradient ∇_Z V
    # V is [batch_size, 1], so we sum over batch to get scalar for gradient
    # Note: If we computed Hessian first, the graph may have been consumed
    # So we need to recompute V if Hessian was computed
    if use_convexity_weighting and mu_estimate is not None:
        # Graph was consumed by Hessian computation, need to recompute V
        if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
            V = critic.value_head(Z)
        else:
            V = critic(Z)
    
    V_sum = V.sum()  # Scalar
    grad_V = torch.autograd.grad(V_sum, Z, create_graph=False, retain_graph=False)[0]
    # grad_V is [batch_size, z_dim]
    
    # Compute gradient norm per sample
    # Options: power=1.0 gives L2 norm ||∇V|| (matches theoretical bound)
    #          power=2.0 gives squared norm ||∇V||² (smoother, but smaller for small gradients)
    grad_norm = torch.norm(grad_V, p=2, dim=1, keepdim=True)  # [batch_size, 1] - L2 norm
    grad_norm_powered = grad_norm ** grad_norm_power  # [batch_size, 1]
    
    # Compute weighted loss
    if use_convexity_weighting and mu_estimate is not None:
        # Weight by -convexity_coef * μ: penalize negative μ (non-convex) more strongly
        # Higher convexity_coef increases the importance of the μ term
        weight = -convexity_coef * mu_estimate
        L_rep = alpha * weight * grad_norm_powered.mean()
    else:
        # Basic version: just gradient norm
        L_rep = alpha * grad_norm_powered.mean()
        mu_estimate = torch.tensor(0.0, device=Z.device, dtype=Z.dtype)
    
    stats = {
        "representation_loss": L_rep.item(),
        "grad_norm": grad_norm.mean().item(),
        "mu_estimate": mu_estimate.item() if isinstance(mu_estimate, torch.Tensor) else mu_estimate,
        "grad_norm_powered": grad_norm_powered.mean().item(),
        "grad_norm_sq_mean": (grad_norm ** 2).mean().item(),  # Keep for backward compatibility
    }
    
    return L_rep, stats


def compute_smallest_eigenvalue_hessian_z(V: torch.Tensor, Z: torch.Tensor, critic: nn.Module, num_samples: int = 3) -> torch.Tensor:
    """
    Compute smallest eigenvalue of Hessian of V w.r.t. Z.
    
    Computed once per batch update on the current batch.
    For efficiency, we compute Hessian for a subset of samples and return mean.
    
    For each sample i:
        H[i] = ∇²_Z V[i]  [z_dim, z_dim]
        μ[i] = smallest eigenvalue of H[i]
    
    Args:
        V: Value function output [batch_size, 1]
        Z: Representation [batch_size, z_dim] (must have requires_grad=True)
        num_samples: Number of samples to compute Hessian for (for efficiency)
        
    Returns:
        Mean smallest eigenvalue [scalar]
    """
    batch_size, z_dim = Z.shape
    
    # Sample a subset for efficiency (Hessian computation is expensive)
    if num_samples >= batch_size:
        sample_indices = list(range(batch_size))
    else:
        # Randomly sample indices
        sample_indices = torch.randperm(batch_size, device=Z.device)[:num_samples].tolist()
    
    eigenvalues_list = []
    
    for idx, i in enumerate(sample_indices):
        # Extract single sample value and representation
        V_i = V[i, 0]  # Scalar
        Z_i = Z[i].clone().detach().requires_grad_(True)  # [z_dim] - fresh computation graph
        
        # Recompute V_i from Z_i to get a fresh computation graph
        # This avoids graph reuse issues
        if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
            V_i_recomputed = critic.value_head(Z_i.unsqueeze(0))[0, 0]  # Scalar
        else:
            V_i_recomputed = critic(Z_i.unsqueeze(0))[0, 0]  # Scalar
        
        # Compute Hessian using functional API
        from torch.autograd.functional import hessian
        
        def value_fn(z):
            z_batch = z.unsqueeze(0)  # [1, z_dim]
            if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
                return critic.value_head(z_batch)[0, 0]
            else:
                return critic(z_batch)[0, 0]
        
        # Compute Hessian: H[i] = ∇²_Z V[i]
        H_i = hessian(value_fn, Z_i, create_graph=False)  # [z_dim, z_dim]
        
        # Make Hessian symmetric (should be symmetric, but numerical issues can occur)
        H_i_sym = (H_i + H_i.T) / 2.0
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvalsh(H_i_sym)  # [z_dim]
        smallest_eigenval = eigenvals[0]  # Smallest eigenvalue
        eigenvalues_list.append(smallest_eigenval)
    
    if len(eigenvalues_list) == 0:
        raise RuntimeError("No eigenvalues computed - all samples failed")
    
    # Return minimum eigenvalue across sampled batch
    mu_batch = torch.stack(eigenvalues_list)  # [num_samples]
    return mu_batch.min()  # Scalar - minimum eigenvalue

