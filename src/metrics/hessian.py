"""
Hessian spectrum computation for value functions.

Mathematical Background:
------------------------
The Hessian H = ∇²L(θ) is the second derivative of the loss L with respect to parameters θ.
We compute Hv (Hessian-vector product) efficiently using:
    Hv = ∇_θ (∇_θ L(θ) · v)

When Parameters Are Unused (Mathematically Correct):
---------------------------------------------------
A parameter θ_i is "unused" if ∂L/∂θ_i = 0, meaning it doesn't affect the loss. This happens when:

1. **Dead Neurons**: ReLU neurons with always-negative inputs output 0, so their parameters don't affect loss
2. **Zero Weights**: If weight w_ij = 0 (or effectively 0 after positivity transformation), it doesn't contribute
3. **ICNN Positivity Functions**:
   - ExponentialPositivity: exp(w) ≈ 0 if w << 0, making the parameter effectively unused
   - LazyClippedPositivity: max(0, w) = 0 if w < 0, making negative weights unused
4. **Frozen Parameters**: Parameters with requires_grad=False are excluded (not in computation graph)

Mathematical Correctness:
-------------------------
- If ∂L/∂θ_i = 0, then ∂²L/∂θ_i∂θ_j = 0 for all j (Hessian row/column is zero)
- Excluding unused parameters is mathematically correct - they don't affect the Hessian
- We must be consistent: if θ_i is excluded from first_grad, it must be excluded from Hv computation
- The Hessian is computed only over the subspace of parameters that affect the loss

Implementation:
---------------
We filter parameters consistently:
1. Compute first_grad = ∇L(θ) for all parameters
2. Identify used parameters: those with non-None gradients
3. Compute Hv only over used parameters to ensure size consistency
"""

import torch
import torch.nn as nn
import numpy as np


def compute_hessian_spectrum(
    critic: nn.Module,
    obs: torch.Tensor,
    top_k: int = 10,
) -> dict:
    """
    Compute eigenvalues of Hessian ∇²V(z) on critic inputs.
    
    Uses Lanczos method for efficient eigenvalue computation on large Hessians.
    
    Args:
        critic: Value function critic
        obs: Observations [N, obs_dim] (sample of states)
        top_k: Number of top eigenvalues to compute
        
    Returns:
        Dictionary with:
            - eigenvalues: Top-k eigenvalues [top_k]
            - eigenvectors: Corresponding eigenvectors [top_k, param_dim]
            - min_eigenvalue: Minimum eigenvalue
            - max_eigenvalue: Maximum eigenvalue
            - mean_eigenvalue: Mean eigenvalue
    """
    obs.requires_grad_(True)
    
    # Compute value sum (scalar)
    # For VAE critics, if obs is already encoded (latent z), use value_head directly
    # Otherwise, use critic.forward() which handles encoding
    if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
        # VAE critic: check if obs is already encoded (matches latent_dim)
        if obs.shape[-1] == critic.latent_dim:
            # obs is already encoded (latent z): use value_head directly
            values = critic.value_head(obs).sum()
        else:
            # obs is raw: use critic.forward() which handles encoding
            # Use original (uncompiled) network for Hessian computation if available
            if hasattr(critic, 'forward'):
                import inspect
                sig = inspect.signature(critic.forward)
                if 'use_original' in sig.parameters:
                    values = critic(obs, use_original=True).sum()
                else:
                    values = critic(obs).sum()
            else:
                values = critic(obs).sum()
    else:
        # ICNN/Feedforward: use original (uncompiled) network for Hessian computation
        # torch.compile() doesn't work with create_graph=True and retain_graph=True
        if hasattr(critic, 'forward'):
            import inspect
            sig = inspect.signature(critic.forward)
            if 'use_original' in sig.parameters:
                values = critic(obs, use_original=True).sum()
            else:
                values = critic(obs).sum()
        else:
            values = critic(obs).sum()
    
    # Compute first-order gradients
    # Filter out parameters that don't require grad (frozen parameters)
    params = [p for p in critic.parameters() if p.requires_grad]
    first_grad = torch.autograd.grad(values, params, create_graph=True, retain_graph=True, allow_unused=True)
    
    # Identify unused parameters (those with None gradients)
    # This is mathematically correct: if ∂L/∂θ_i = None, the parameter doesn't affect the loss
    unused_param_indices = [i for i, g in enumerate(first_grad) if g is None]
    used_param_indices = [i for i, g in enumerate(first_grad) if g is not None]
    
    # Diagnostic: Log unused parameters to understand why they're unused
    if unused_param_indices:
        param_names = [name for name, p in critic.named_parameters() if p.requires_grad]
        unused_names = [param_names[i] for i in unused_param_indices if i < len(param_names)]
        unused_count = len(unused_names)
        total_count = len(params)
        print(
            f"Hessian computation: {unused_count}/{total_count} parameters unused "
            f"({100*unused_count/total_count:.1f}%). "
            f"Examples: {unused_names[:3] if unused_names else 'N/A'}. "
            f"This is normal for ICNN with positivity functions (dead neurons, zero weights)."
        )
    
    # Filter to only used parameters (mathematically correct - only compute Hessian over active subspace)
    first_grad = [first_grad[i] for i in used_param_indices]
    used_params = [params[i] for i in used_param_indices]
    
    if len(first_grad) == 0:
        raise RuntimeError(
            "No gradients computed - all parameters appear unused. "
            "This suggests the critic output doesn't depend on any parameters, which is likely a bug."
        )
    
    # Pre-check: Test Hv computation to identify parameters that will return None in second gradient
    # This filters out parameters that are unused in the second-order computation (e.g., dead neurons, zero biases)
    flat_grad = torch.cat([g.view(-1) for g in first_grad])
    test_v = torch.randn(flat_grad.shape[0], device=flat_grad.device, dtype=flat_grad.dtype)
    test_grad_dot_v = (flat_grad * test_v).sum()
    test_Hv = torch.autograd.grad(test_grad_dot_v, used_params, retain_graph=True, allow_unused=True)
    
    # Filter out parameters that return None in Hv computation
    hv_none_indices = [i for i, h in enumerate(test_Hv) if h is None]
    if hv_none_indices:
        # These parameters are unused in second-order computation - filter them out proactively
        param_names = [name for name, p in critic.named_parameters() if p.requires_grad]
        used_param_names = [param_names[i] for i in used_param_indices if i < len(param_names)]
        none_names = [used_param_names[i] for i in hv_none_indices if i < len(used_param_names)]
        
        print(
            f"Hessian computation: Filtering {len(hv_none_indices)} parameter(s) that are unused in second-order computation: "
            f"{', '.join(none_names)}. "
            f"This is normal for ICNN with positivity functions (e.g., zero-initialized biases that never update)."
        )
        
        # Filter to only parameters that work in both first and second gradient
        final_used_indices = [i for i in range(len(used_params)) if i not in hv_none_indices]
        first_grad = [first_grad[i] for i in final_used_indices]
        used_params = [used_params[i] for i in final_used_indices]
        
        if len(first_grad) == 0:
            raise RuntimeError(
                "No parameters remain after filtering unused parameters in second-order computation. "
                "This suggests a bug in the network architecture or computation graph."
            )
        
        flat_grad = torch.cat([g.view(-1) for g in first_grad])
    
    # Compute Hessian-vector products for top eigenvalues
    # Use power iteration / Lanczos approximation
    eigenvalues = []
    eigenvectors = []
    
    # Initialize random vector matching the filtered parameter size
    v = torch.randn(flat_grad.shape[0], device=flat_grad.device, dtype=flat_grad.dtype)
    v = v / v.norm()
    
    for i in range(top_k):
        # Compute Hv using the SAME used parameter set and SAME computation graph to ensure consistency
        # Pass the pre-computed first_grad to avoid recreating the computation graph
        # Returns (Hv, v_adjusted, used_params_adjusted) in case v/params were adjusted due to None gradients
        Hv, v_adjusted, used_params_adjusted = compute_hessian_vector_product(
            critic, obs, v, used_params=used_params, first_grad=first_grad, values=values
        )
        
        # Update v and used_params to the adjusted versions (in case some parameters were filtered)
        v = v_adjusted
        used_params = used_params_adjusted
        
        # Rayleigh quotient
        eigenvalue = (v * Hv).sum()
        eigenvalues.append(eigenvalue.item())
        eigenvectors.append(v.clone())
        
        # Deflate for next eigenvector
        if i < top_k - 1:
            # Orthogonalize against previous eigenvectors
            for prev_v in eigenvectors[:-1]:
                v = v - (v * prev_v).sum() * prev_v
            v = v / (v.norm() + 1e-8)
    
    eigenvalues = np.array(eigenvalues)
    
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": torch.stack(eigenvectors),
        "min_eigenvalue": float(eigenvalues.min()),
        "max_eigenvalue": float(eigenvalues.max()),
        "mean_eigenvalue": float(eigenvalues.mean()),
    }


def compute_hessian_vector_product(
    critic: nn.Module,
    obs: torch.Tensor,
    v: torch.Tensor,
    used_params: list[torch.nn.Parameter] = None,
    first_grad: list[torch.Tensor] = None,
    values: torch.Tensor = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Hessian-vector product Hv efficiently.
    
    Mathematical formulation:
        Hv = ∇_θ (∇_θ L(θ) · v)
    
    We compute this efficiently by:
        1. Compute g = ∇_θ L(θ) (first-order gradient)
        2. Compute g · v (dot product)
        3. Compute ∇_θ (g · v) (gradient of dot product)
    
    Args:
        critic: Value function critic
        obs: Observations [N, obs_dim]
        v: Vector [param_dim] - must match the size of filtered (used) parameters
        used_params: Optional list of parameters to use. If None, will determine automatically.
                    This ensures consistency when called multiple times.
        first_grad: Optional pre-computed first-order gradients. If provided, reuse the same computation graph.
        values: Optional pre-computed values tensor. If provided with first_grad, reuse the same computation graph.
        
    Returns:
        (Hv, v_adjusted, used_params_adjusted): Tuple of:
            - Hv: Hessian-vector product [param_dim] - same size as v_adjusted
            - v_adjusted: Adjusted v if some parameters were filtered out, otherwise original v
            - used_params_adjusted: Adjusted parameter list if some were filtered out, otherwise original used_params
        
    Mathematical Correctness:
        - We only compute Hv over parameters that affect the loss (used parameters)
        - This is correct: if ∂L/∂θ_i = 0, then H_ij = 0 for all j
        - We must use the SAME parameter set for both first_grad and Hv to ensure consistency
    """
    # Reuse computation graph if provided, otherwise create new one
    if first_grad is not None and values is not None:
        # Reuse the same computation graph from pre-check
        # This ensures mathematical consistency: if a parameter has first-order gradient,
        # it must have second-order gradient in the same graph
        first_grad_filtered = first_grad
        if used_params is None:
            raise ValueError("used_params must be provided when first_grad is provided")
    else:
        # Create new computation graph (for standalone calls)
        obs.requires_grad_(True)
        # For VAE critics, if obs is already encoded (latent z), use value_head directly
        if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
            if obs.shape[-1] == critic.latent_dim:
                values = critic.value_head(obs).sum()
            else:
                # Use original (uncompiled) network for Hessian computation
                if hasattr(critic, 'forward'):
                    import inspect
                    sig = inspect.signature(critic.forward)
                    if 'use_original' in sig.parameters:
                        values = critic(obs, use_original=True).sum()
                    else:
                        values = critic(obs).sum()
                else:
                    values = critic(obs).sum()
        else:
            # Use original (uncompiled) network for Hessian computation
            if hasattr(critic, 'forward'):
                import inspect
                sig = inspect.signature(critic.forward)
                if 'use_original' in sig.parameters:
                    values = critic(obs, use_original=True).sum()
                else:
                    values = critic(obs).sum()
            else:
                values = critic(obs).sum()
        
        # Use provided used_params if available, otherwise determine them
        if used_params is None:
            # First gradient - identify used vs unused parameters
            params = [p for p in critic.parameters() if p.requires_grad]
            # For VAE critics with encoded obs, we need to compute values first
            # (values was already computed above, so we can use it)
            if values is None:
                # This shouldn't happen, but handle it just in case
                if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim') and obs.shape[-1] == critic.latent_dim:
                    values = critic.value_head(obs).sum()
                else:
                    if hasattr(critic, 'forward'):
                        import inspect
                        sig = inspect.signature(critic.forward)
                        if 'use_original' in sig.parameters:
                            values = critic(obs, use_original=True).sum()
                        else:
                            values = critic(obs).sum()
                    else:
                        values = critic(obs).sum()
            first_grad = torch.autograd.grad(values, params, create_graph=True, retain_graph=True, allow_unused=True)
            
            # Track which parameters are used (non-None gradients)
            used_param_indices = [i for i, g in enumerate(first_grad) if g is not None]
            used_params = [params[i] for i in used_param_indices]
            first_grad_filtered = [first_grad[i] for i in used_param_indices]
        else:
            # Use the provided parameter set (ensures consistency)
            # When used_params is provided, ALL parameters should have gradients
            # If any return None, that indicates an inconsistency (computation graph changed)
            first_grad_all = torch.autograd.grad(values, used_params, create_graph=True, retain_graph=True, allow_unused=True)
            
            # Check for None gradients (shouldn't happen if used_params is correct)
            none_indices = [i for i, g in enumerate(first_grad_all) if g is None]
            if none_indices:
                # This indicates the computation graph changed - parameters that were used are now unused
                # This is a bug - we should use the same computation graph
                raise RuntimeError(
                    f"Computation graph inconsistency: {len(none_indices)} parameters in used_params "
                    f"returned None gradients. This suggests the computation graph changed between calls. "
                    f"This should not happen - ensure obs and critic state are the same."
                )
            
            first_grad_filtered = first_grad_all
    
    if len(first_grad_filtered) == 0:
        raise RuntimeError(
            "No gradients computed - all parameters appear unused. "
            "This suggests the critic output doesn't depend on any parameters."
        )
    
    flat_grad = torch.cat([g.view(-1) for g in first_grad_filtered])
    
    # Verify that used_params matches the size of v
    expected_size = sum(p.numel() for p in used_params)
    if flat_grad.shape[0] != expected_size:
        raise RuntimeError(
            f"Size mismatch: flat_grad has size {flat_grad.shape[0]}, but used_params has total size {expected_size}. "
            f"This indicates some parameters in used_params are actually unused."
        )
    
    # Ensure v matches flat_grad size (consistency check)
    if v.shape[0] != flat_grad.shape[0]:
        raise RuntimeError(
            f"Vector size mismatch: v has size {v.shape[0]}, but filtered gradients have size {flat_grad.shape[0]}. "
            f"This indicates inconsistent parameter filtering. "
            f"v should be created from the same filtered parameter set as first_grad."
        )
    
    # Compute gradient of (grad^T · v) with respect to the SAME used parameters
    # This gives us Hv where H is the Hessian over the active parameter subspace
    # Mathematical correctness: If a parameter has first-order gradient (∂V/∂θ_i exists),
    # it MUST have second-order gradient (∂²V/∂θ_i∂θ_j exists) in the same computation graph.
    # We use allow_unused=False because we've pre-filtered to only include parameters with gradients.
    grad_dot_v = (flat_grad * v).sum()
    Hv = torch.autograd.grad(grad_dot_v, used_params, retain_graph=True, allow_unused=False)
    
    # Check for None gradients - this should NOT happen since we pre-filtered used_params
    # If it does, that indicates a computation graph inconsistency (bug)
    none_indices = [i for i, h in enumerate(Hv) if h is None]
    if none_indices:
        # This should not happen since we pre-filtered used_params to exclude parameters
        # that return None in second-order computation. If it does, that's a bug.
        param_names = []
        for idx in none_indices:
            param = used_params[idx]
            # Find the name of this parameter
            param_name = None
            for name, p in critic.named_parameters():
                if p is param:  # Use 'is' for identity check
                    param_name = name
                    break
            if param_name:
                param_names.append(f"{param_name} (shape={tuple(param.shape)}, numel={param.numel()})")
            else:
                param_names.append(f"param[{idx}] (shape={tuple(param.shape)}, numel={param.numel()})")
        
        raise RuntimeError(
            f"Computation graph inconsistency: {len(none_indices)} parameter(s) in used_params "
            f"returned None gradients during Hv computation, even though they passed the pre-filter. "
            f"This suggests the computation graph changed between the pre-check and actual computation. "
            f"None parameters: {', '.join(param_names)}. "
            f"This is a bug - the pre-filter should have caught these."
        )
    else:
        # All parameters have gradients - normal case
        Hv_flat = torch.cat([h.view(-1) for h in Hv])
        
        # Final size check
        if Hv_flat.shape[0] != v.shape[0]:
            raise RuntimeError(
                f"Hessian-vector product size mismatch: Hv has size {Hv_flat.shape[0]}, but v has size {v.shape[0]}. "
                f"This indicates inconsistent parameter filtering between first and second gradient computation. "
                f"This is a bug - the same parameter set should be used for both."
            )
        
        return Hv_flat, v, used_params


def compute_hessian_trace(
    critic: nn.Module,
    obs: torch.Tensor,
    num_samples: int = 10,
) -> float:
    """
    Estimate trace of Hessian using Hutchinson's trace estimator.
    
    Args:
        critic: Value function critic
        obs: Observations [N, obs_dim]
        num_samples: Number of random vectors for estimation
        
    Returns:
        Estimated trace
    """
    # Determine used parameters once (consistent with compute_hessian_spectrum)
    obs_test = obs.clone().detach().requires_grad_(True)
    # For VAE critics, if obs is already encoded (latent z), use value_head directly
    if hasattr(critic, 'value_head') and hasattr(critic, 'latent_dim'):
        if obs_test.shape[-1] == critic.latent_dim:
            values_test = critic.value_head(obs_test).sum()
        else:
            # Use original (uncompiled) network for Hessian computation
            if hasattr(critic, 'forward'):
                import inspect
                sig = inspect.signature(critic.forward)
                if 'use_original' in sig.parameters:
                    values_test = critic(obs_test, use_original=True).sum()
                else:
                    values_test = critic(obs_test).sum()
            else:
                values_test = critic(obs_test).sum()
    else:
        # Use original (uncompiled) network for Hessian computation
        if hasattr(critic, 'forward'):
            import inspect
            sig = inspect.signature(critic.forward)
            if 'use_original' in sig.parameters:
                values_test = critic(obs_test, use_original=True).sum()
            else:
                values_test = critic(obs_test).sum()
        else:
            values_test = critic(obs_test).sum()
    
    params = [p for p in critic.parameters() if p.requires_grad]
    if len(params) == 0:
        return 0.0
    
    test_grad = torch.autograd.grad(values_test, params, create_graph=True, retain_graph=True, allow_unused=True)
    used_param_indices = [i for i, g in enumerate(test_grad) if g is not None]
    used_params = [params[i] for i in used_param_indices]
    
    if len(used_params) == 0:
        return 0.0
    
    # Pre-check: Test Hv computation to identify parameters that will return None in second gradient
    # This filters out parameters that are unused in the second-order computation (e.g., dead neurons, zero biases)
    # Mathematically correct: if ∂²V/∂θ_i∂θ_j doesn't exist (parameter not in second-order graph), exclude it
    test_grad_filtered = [test_grad[i] for i in used_param_indices]
    flat_test_grad = torch.cat([g.view(-1) for g in test_grad_filtered])
    test_v = torch.randn(flat_test_grad.shape[0], device=obs_test.device, dtype=obs_test.dtype)
    test_grad_dot_v = (flat_test_grad * test_v).sum()
    test_Hv = torch.autograd.grad(test_grad_dot_v, used_params, retain_graph=True, allow_unused=True)
    
    # Filter out parameters that return None in Hv computation
    hv_none_indices = [i for i, h in enumerate(test_Hv) if h is None]
    if hv_none_indices:
        # These parameters are unused in second-order computation - filter them out proactively
        param_names = [name for name, p in critic.named_parameters() if p.requires_grad]
        used_param_names = [param_names[i] for i in used_param_indices if i < len(param_names)]
        none_names = [used_param_names[i] for i in hv_none_indices if i < len(used_param_names)]
        
        print(
            f"Hessian trace computation: Filtering {len(hv_none_indices)} parameter(s) that are unused in second-order computation: "
            f"{', '.join(none_names)}. "
            f"This is normal for ICNN with positivity functions (e.g., zero-initialized biases that never update)."
        )
        
        # Filter to only parameters that work in both first and second gradient
        final_used_indices = [i for i in range(len(used_params)) if i not in hv_none_indices]
        used_params = [used_params[i] for i in final_used_indices]
        
        if len(used_params) == 0:
            return 0.0
    
    trace_estimate = 0.0
    
    for _ in range(num_samples):
        # Get current parameter size from used_params (may have been adjusted in previous iteration)
        param_size = sum(p.numel() for p in used_params)
        
        # Random vector matching current filtered parameter size
        v = torch.randn(param_size, device=obs.device, dtype=obs.dtype)
        v = v / v.norm()
        
        # Compute Hv using the SAME used parameter set (returns adjusted v and params if needed)
        Hv, v_adjusted, used_params_adjusted = compute_hessian_vector_product(critic, obs, v, used_params=used_params)
        
        # Use adjusted v and params (in case some parameters were filtered)
        v = v_adjusted
        used_params = used_params_adjusted
        
        # Sizes should match now
        if v.shape[0] != Hv.shape[0]:
            raise RuntimeError(
                f"Size mismatch in trace computation: v has size {v.shape[0]}, Hv has size {Hv.shape[0]}. "
                f"This indicates inconsistent parameter filtering."
            )
        trace_estimate += (v * Hv).sum().item()
    
    return trace_estimate / num_samples

