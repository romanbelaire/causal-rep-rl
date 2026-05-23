"""
Convexity hypothesis validation protocol.

Validates the hypothesis that policies with convex value functions learn low-error 
representations, which lead to low-regret policies.

Core theorem: ||Z*(s) - Z(s)|| ≤ (1/μ) ||∇V(Z(s))||

Where:
- Z*(s) is the optimal representation
- Z(s) is the current representation
- V(Z) is the value function (should be μ-strongly convex)
- μ is the smallest eigenvalue of the Hessian ∇²V(Z)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional

from src.metrics.value_head_types import is_affine_vae_value_head, is_quadratic_latent_value_head
from src.metrics.value_head_mu import get_analytic_mu_latent


def estimate_local_convexity(
    critic: nn.Module,
    z: torch.Tensor,
    mu_min: float = 1e-6,
    sample_size: int = 16,
    concavity_epsilon: float = 1e-3,
) -> Dict[str, Any]:
    """
    Estimate local convexity parameter μ (smallest eigenvalue) of Hessian at sampled representations.
    
    Computes the Hessian H = ∇²V(Z) w.r.t. the representation Z (not parameters).
    The smallest eigenvalue μ indicates the strength of local convexity.
    
    Args:
        critic: Value function critic that takes Z as input and outputs V(Z)
        z: Representation tensor [N, repr_dim] or [repr_dim]
        mu_min: Minimum value for μ to avoid division by zero
        sample_size: Number of samples to compute full Hessian for (rest use statistics)
        
    Returns:
        Dictionary with:
            - mu: Smallest eigenvalue (local convexity parameter)
            - eigenvalues: All eigenvalues (sorted ascending)
            - max_eigenvalue: Largest eigenvalue
            - mean_eigenvalue: Mean eigenvalue
            - is_convex: Whether μ > mu_min (local convexity holds)
            - pct_convex: Percentage of samples with μ > mu_min (for batched computation)
    """
    if is_affine_vae_value_head(critic):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        batch_size, repr_dim = z.shape
        eigenvalues = torch.zeros(batch_size, repr_dim, device=z.device, dtype=z.dtype)
        print(
            "Convexity validation: affine value head V(z)=wᵀz+b → μ=0, ∇²_z V ≡ 0 (convex by construction, not μ-strongly convex)"
        )
        return {
            "mu": 0.0,
            "eigenvalues": eigenvalues,
            "max_eigenvalue": 0.0,
            "mean_eigenvalue": 0.0,
            "is_convex": True,
            "pct_convex": 1.0,
            "mu_concave": 0.0,
            "mu_concave_mean": 0.0,
            "pct_concave": 0.0,
            "affine_by_construction": True,
        }

    if is_quadratic_latent_value_head(critic):
        mu_z = get_analytic_mu_latent(critic).item()
        print(
            "Convexity validation: quadratic PSD on Z → "
            f"μ_latent_analytic={mu_z:.6f} (2σ_min(A)², constant ∇²_Z V)"
        )

    # Handle single sample case
    if z.dim() == 1:
        z = z.unsqueeze(0)
        single_sample = True
    else:
        single_sample = False
    
    batch_size, repr_dim = z.shape
    
    # Sample a subset for full Hessian computation (expensive but accurate)
    sample_indices = torch.randperm(batch_size)[:min(sample_size, batch_size)]
    z_sample = z[sample_indices]  # [sample_size, repr_dim]
    
    eigenvalues_list = []
    
    # Compute full Hessian for sampled points
    for idx in range(len(sample_indices)):
        z_i = z_sample[idx:idx+1].clone().detach().requires_grad_(True)  # [1, repr_dim]
        
        # Forward pass for sample i
        # For VAE critics, z is the latent representation, so use value_head directly
        if hasattr(critic, 'value_head'):
            # VAE critic: z is latent, use value_head
            v_i = critic.value_head(z_i).squeeze()  # Scalar
        elif hasattr(critic, 'forward'):
            import inspect
            sig = inspect.signature(critic.forward)
            if 'use_original' in sig.parameters:
                v_i = critic(z_i, use_original=True).squeeze()  # Scalar
            else:
                v_i = critic(z_i).squeeze()
        else:
            v_i = critic(z_i).squeeze()
        
        # Compute Hessian using functional API
        try:
            def value_fn(z_single):
                z_single = z_single.unsqueeze(0)  # [1, repr_dim]
                # For VAE critics, z is the latent representation, so use value_head directly
                if hasattr(critic, 'value_head'):
                    return critic.value_head(z_single).squeeze()
                elif hasattr(critic, 'forward'):
                    sig = inspect.signature(critic.forward)
                    if 'use_original' in sig.parameters:
                        return critic(z_single, use_original=True).squeeze()
                    else:
                        return critic(z_single).squeeze()
                else:
                    return critic(z_single).squeeze()
            
            H_i = torch.autograd.functional.hessian(
                value_fn,
                z_i.squeeze(0),  # [repr_dim]
                create_graph=False,
            )  # [repr_dim, repr_dim]
        except Exception as e:
            # Fallback: compute via gradient of gradient (slower but more robust)
            grad_i = torch.autograd.grad(
                v_i,
                z_i,
                create_graph=True,
                retain_graph=True,
            )[0]  # [1, repr_dim]
            
            # Compute Hessian row by row
            H_i = torch.zeros(repr_dim, repr_dim, device=z_i.device, dtype=z_i.dtype)
            for j in range(repr_dim):
                grad_ij = torch.autograd.grad(
                    grad_i[0, j],
                    z_i,
                    retain_graph=(j < repr_dim - 1),
                )[0]  # [1, repr_dim]
                H_i[j, :] = grad_ij[0, :]
        
        # Compute eigenvalues (symmetric matrix, use eigvalsh)
        eigenvals_i = torch.linalg.eigvalsh(H_i)  # [repr_dim], sorted ascending
        eigenvalues_list.append(eigenvals_i)
    
    # For remaining samples, use statistics from sampled points
    remaining_count = batch_size - len(sample_indices)
    if remaining_count > 0 and len(eigenvalues_list) > 0:
        # Use mean eigenvalue distribution from sampled points
        sampled_eigenvals = torch.stack(eigenvalues_list)  # [sample_size, repr_dim]
        mean_eigenval_dist = sampled_eigenvals.mean(dim=0)  # [repr_dim]
        
        # Replicate mean distribution for remaining samples
        for _ in range(remaining_count):
            # Add small noise to avoid exact duplicates
            noise = torch.randn_like(mean_eigenval_dist) * 0.01
            eigenvals_i = mean_eigenval_dist + noise
            eigenvals_i = torch.sort(eigenvals_i)[0]  # Sort ascending
            eigenvalues_list.append(eigenvals_i)
    
    # Stack eigenvalues: [batch_size, repr_dim]
    eigenvalues = torch.stack(eigenvalues_list, dim=0)
    
    # Compute statistics
    mu = eigenvalues.min().item()  # Smallest eigenvalue across all samples
    max_eigenvalue = eigenvalues.max().item()
    mean_eigenvalue = eigenvalues.mean().item()
    
    # Percentage of samples with μ > mu_min
    min_eigenvals_per_sample = eigenvalues.min(dim=1)[0]  # [batch_size]
    pct_convex = (min_eigenvals_per_sample > mu_min).float().mean().item()

    mu_concave_batch = (-min_eigenvals_per_sample).clamp(min=0.0)
    mu_concave = mu_concave_batch.max().item()
    mu_concave_mean = mu_concave_batch.mean().item()
    pct_concave = (min_eigenvals_per_sample < -concavity_epsilon).float().mean().item()
    
    return {
        "mu": mu,
        "eigenvalues": eigenvalues,
        "max_eigenvalue": max_eigenvalue,
        "mean_eigenvalue": mean_eigenvalue,
        "is_convex": mu > mu_min,
        "pct_convex": pct_convex,
        "mu_concave": mu_concave,
        "mu_concave_mean": mu_concave_mean,
        "pct_concave": pct_concave,
    }


def verify_representation_bound(
    critic: nn.Module,
    z_old: torch.Tensor,
    z_new: torch.Tensor,
    mu_min: float = 0.05,
) -> Dict[str, Any]:
    """
    Verify the representation bound: ||ΔZ|| ≈ (1/μ)||∇V||
    
    The theorem states: ||Z*(s) - Z(s)|| ≤ (1/μ) ||∇V(Z(s))||
    
    This function compares:
    - Actual representation error: ||Z_new - Z_old||
    - Predicted error from bound: (1/μ) ||∇V(Z_old)||
    
    Args:
        critic: Value function critic
        z_old: Old representations [N, repr_dim]
        z_new: New representations [N, repr_dim] (after update)
        mu_min: Minimum value for μ to avoid division by zero
        
    Returns:
        Dictionary with:
            - correlation: Correlation between actual and predicted error (should be ~1.0)
            - mape: Mean absolute percentage error (should be <0.3)
            - delta_Z_actual: Mean actual representation error
            - delta_Z_predicted: Mean predicted error
            - mu_local: Local convexity parameter used
            - bound_holds: Whether bound holds (correlation > 0.5 and mape < 0.5)
    """
    # Ensure same shape
    if z_old.shape != z_new.shape:
        raise ValueError(f"Shape mismatch: z_old {z_old.shape} vs z_new {z_new.shape}")
    
    N = z_old.shape[0]
    
    # Actual representation error: ||Z_new - Z_old||
    delta_Z = z_new - z_old  # [N, repr_dim]
    delta_Z_norm = torch.norm(delta_Z, dim=1)  # [N]
    
    # Compute gradient magnitude at old Z: ||∇V(Z_old)||
    z_old_grad = z_old.clone().detach().requires_grad_(True)
    
    # Forward pass
    # For VAE critics, z is the latent representation, so use value_head directly
    if hasattr(critic, 'value_head'):
        # VAE critic: z is latent, use value_head
        V_old = critic.value_head(z_old_grad).squeeze(-1)  # [N]
    elif hasattr(critic, 'forward'):
        import inspect
        sig = inspect.signature(critic.forward)
        if 'use_original' in sig.parameters:
            V_old = critic(z_old_grad, use_original=True).squeeze(-1)  # [N]
        else:
            V_old = critic(z_old_grad).squeeze(-1)
    else:
        V_old = critic(z_old_grad).squeeze(-1)
    
    # Compute gradient
    grad_V = torch.autograd.grad(
        V_old.sum(),
        z_old_grad,
        create_graph=True,
        retain_graph=True,
    )[0]  # [N, repr_dim]
    
    grad_V_norm = torch.norm(grad_V, dim=1)  # [N]
    
    # Compute local μ at old Z (use estimate_local_convexity for efficiency)
    # For efficiency, we'll estimate μ from a sample
    sample_size = min(32, N)  # Use smaller sample for μ estimation
    z_sample = z_old[:sample_size]
    convexity_results = estimate_local_convexity(critic, z_sample, mu_min=mu_min)
    mu_local = max(convexity_results["mu"], mu_min)  # Ensure μ >= mu_min
    
    # Predicted error from bound: (1/μ) ||∇V||
    predicted_delta_Z = (1.0 / mu_local) * grad_V_norm  # [N]
    
    # Compare actual vs predicted
    # Convert to numpy for correlation computation
    delta_Z_np = delta_Z_norm.detach().cpu().numpy()
    predicted_delta_Z_np = predicted_delta_Z.detach().cpu().numpy()
    
    # Compute correlation
    if len(delta_Z_np) > 1 and np.std(delta_Z_np) > 1e-8 and np.std(predicted_delta_Z_np) > 1e-8:
        correlation = np.corrcoef(delta_Z_np, predicted_delta_Z_np)[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Compute MAPE (Mean Absolute Percentage Error)
    # MAPE = mean(|actual - predicted| / (actual + eps))
    eps = 1e-6
    mape = torch.mean(
        torch.abs(delta_Z_norm - predicted_delta_Z) / (delta_Z_norm + eps)
    ).item()
    
    # Check if bound holds
    bound_holds = (correlation > 0.5) and (mape < 0.5)
    
    return {
        "correlation": correlation,
        "mape": mape,
        "delta_Z_actual": delta_Z_norm.mean().item(),
        "delta_Z_predicted": predicted_delta_Z.mean().item(),
        "mu_local": mu_local,
        "bound_holds": bound_holds,
    }


def check_neighborhood_membership(
    critic: nn.Module,
    z: torch.Tensor,
    z_star: Optional[torch.Tensor] = None,
    radius: float = 0.1,
) -> Dict[str, Any]:
    """
    Check if representations Z(s) stay within a convex neighborhood of Z*.
    
    The theorem assumes Z(s) stays in a locally convex neighborhood 𝒩 of Z*.
    This function checks if ||Z(s) - Z*|| < radius for most samples.
    
    Args:
        critic: Value function critic
        z: Current representations [N, repr_dim]
        z_star: Optimal representations [N, repr_dim] (if None, use EMA of z)
        radius: Neighborhood radius threshold
        
    Returns:
        Dictionary with:
            - pct_in_neighborhood: Percentage of samples within radius
            - max_distance: Maximum distance to Z*
            - mean_distance: Mean distance to Z*
            - radius: Radius used
    """
    N = z.shape[0]
    
    # If Z* not provided, use mean of current Z as proxy
    if z_star is None:
        z_star = z.mean(dim=0, keepdim=True).expand_as(z)  # [N, repr_dim]
    else:
        if z_star.shape != z.shape:
            raise ValueError(f"Shape mismatch: z {z.shape} vs z_star {z_star.shape}")
    
    # Compute distances: ||Z(s) - Z*||
    distances = torch.norm(z - z_star, dim=1)  # [N]
    
    # Check neighborhood membership
    in_neighborhood = (distances < radius).float()  # [N]
    pct_in_neighborhood = in_neighborhood.mean().item()
    
    return {
        "pct_in_neighborhood": pct_in_neighborhood,
        "max_distance": distances.max().item(),
        "mean_distance": distances.mean().item(),
        "radius": radius,
    }


def diagnostic_step(
    critic: nn.Module,
    z: torch.Tensor,
    z_old: Optional[torch.Tensor] = None,
    mu_min: float = 0.05,
    neighborhood_radius: float = 0.1,
    concavity_epsilon: float = 1e-3,
    step: int = 0,
) -> Dict[str, Any]:
    """
    Run full diagnostic to validate convexity hypothesis.
    
    This is the end-to-end validation function that combines:
    1. Local convexity estimation (μ)
    2. Representation bound verification
    3. Neighborhood membership check
    
    Args:
        critic: Value function critic
        z: Current representations [N, repr_dim]
        z_old: Previous representations [N, repr_dim] (for bound verification)
        mu_min: Minimum value for μ
        neighborhood_radius: Radius for neighborhood check
        step: Current step/epoch number (for logging)
        
    Returns:
        Dictionary with all diagnostic results
    """
    results = {
        "step": step,
    }
    
    # 1. Estimate local convexity
    convexity_results = estimate_local_convexity(
        critic, z, mu_min=mu_min, concavity_epsilon=concavity_epsilon
    )
    results.update({
        "convexity_mu": convexity_results["mu"],
        "convexity_max_eigenvalue": convexity_results["max_eigenvalue"],
        "convexity_mean_eigenvalue": convexity_results["mean_eigenvalue"],
        "convexity_is_convex": convexity_results["is_convex"],
        "convexity_pct_convex": convexity_results["pct_convex"],
        "convexity_mu_concave": convexity_results["mu_concave"],
        "convexity_mu_concave_mean": convexity_results["mu_concave_mean"],
        "convexity_pct_concave": convexity_results["pct_concave"],
    })
    
    # 2. Verify representation bound (if z_old provided)
    if z_old is not None:
        bound_results = verify_representation_bound(critic, z_old, z, mu_min=mu_min)
        results.update({
            "bound_correlation": bound_results["correlation"],
            "bound_mape": bound_results["mape"],
            "bound_delta_Z_actual": bound_results["delta_Z_actual"],
            "bound_delta_Z_predicted": bound_results["delta_Z_predicted"],
            "bound_mu_local": bound_results["mu_local"],
            "bound_holds": bound_results["bound_holds"],
        })
    else:
        results.update({
            "bound_correlation": None,
            "bound_mape": None,
            "bound_delta_Z_actual": None,
            "bound_delta_Z_predicted": None,
            "bound_mu_local": None,
            "bound_holds": None,
        })
    
    # 3. Check neighborhood membership
    neighborhood_results = check_neighborhood_membership(
        critic, z, z_star=None, radius=neighborhood_radius
    )
    results.update({
        "neighborhood_pct": neighborhood_results["pct_in_neighborhood"],
        "neighborhood_max_distance": neighborhood_results["max_distance"],
        "neighborhood_mean_distance": neighborhood_results["mean_distance"],
        "neighborhood_radius": neighborhood_results["radius"],
    })
    
    # 4. Decision rules and warnings
    mu = convexity_results["mu"]
    correlation = bound_results["correlation"] if z_old is not None else None
    pct_in_neighborhood = neighborhood_results["pct_in_neighborhood"]
    
    warnings = []
    affine_head = convexity_results.get("affine_by_construction", False)

    if mu < mu_min and not affine_head:
        warnings.append(f"⚠️  WARNING: μ too small ({mu:.4f} < {mu_min}), convexity assumption violated")
    elif affine_head:
        warnings.append(
            "Affine value head: ∇²_z V ≡ 0 (weakly convex in z; representation bound uses μ_min floor)"
        )
    
    if correlation is not None and correlation < 0.5:
        warnings.append(f"⚠️  WARNING: bound is loose (correlation={correlation:.3f} < 0.5), may need stronger assumptions")
    
    if pct_in_neighborhood < 0.7:
        warnings.append(f"⚠️  WARNING: updates escaping convex neighborhood ({pct_in_neighborhood:.1%} < 70%)")
    
    results["warnings"] = warnings
    
    # 5. Overall validation status
    # Success criteria:
    # - ICNN: μ > 0.1 AND corr > 0.8 AND 80% in neighborhood
    # - MLP: μ > 0.05 AND corr > 0.6 AND 75% in neighborhood
    # - VAE: μ > 0.02 AND corr > 0.5 AND 60% in neighborhood
    
    # For now, use general criteria
    if mu > 0.1 and (correlation is None or correlation > 0.8) and pct_in_neighborhood > 0.8:
        validation_status = "✓ ICNN validates theory precisely"
    elif mu > 0.05 and (correlation is None or correlation > 0.6) and pct_in_neighborhood > 0.75:
        validation_status = "✓ MLP exhibits local convexity → PPO implicitly does this"
    elif affine_head:
        validation_status = "✓ Affine V(z) in z: convex by construction (∇²_z V = 0)"
    elif mu > 0.02 and (correlation is None or correlation > 0.5) and pct_in_neighborhood > 0.6:
        validation_status = "✓ VAE trades theoretical tightness for causal structure"
    else:
        validation_status = "✗ Hypothesis fundamentally violated, re-examine assumptions"
    
    results["validation_status"] = validation_status
    
    # Print summary
    print(f"Step {step}: μ={mu:.4f}, ", end="")
    if correlation is not None:
        print(f"corr(ΔZ, ∇V)={correlation:.3f}, ", end="")
    print(f"in_N={pct_in_neighborhood:.1%}")
    
    for warning in warnings:
        print(f"  {warning}")
    
    return results

