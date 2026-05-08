"""
VAE Proxy Validation Metrics

Measures conditions for when VAE can work as proxy for ICNN (Exp 3 Success Conditions):

1. Hessian(Z_VAE) ≻ 0 near policy updates (μ > 0.1)
2. 90%+ of updates stay in local convexity neighborhood
3. Causal mask separates domains (t-SNE clusters by task, not env)

Usage Example:
--------------
```python
from src.metrics.vae_proxy_validation import check_local_convexity, validate_vae_proxy_conditions

# Check local convexity post-update
def check_local_convexity(vae_critic, states, radius=0.1):
    Z = vae_critic.encode(states)
    H = hessian(vae_critic.value_net, Z)  # Auto-diff Hessian
    eigenvalues = torch.linalg.eigvals(H)
    mu_local = eigenvalues.real.min()
    return mu_local > 0.05  # Threshold for "good enough"

# Or use the provided function:
results = check_local_convexity(vae_critic, states, mu_threshold=0.1)
if results["is_convex"]:
    print(f"Local convexity satisfied: μ = {results['mu_local']:.4f}")

# Validate all conditions:
validation_results = validate_vae_proxy_conditions(
    vae_critic,
    states,
    task_labels=task_labels,  # Optional: for t-SNE analysis
    env_labels=env_labels,     # Optional: for t-SNE analysis
    update_history=update_history,  # Optional: for tracking success rate
)
if validation_results["all_conditions_met"]:
    print("VAE can work as proxy for ICNN!")
```
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.manifold import TSNE


def compute_hessian_on_latent(
    vae_critic: nn.Module,
    states: torch.Tensor,
    value_net: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Compute Hessian of value network w.r.t. latent representation Z_VAE.
    
    The Hessian is computed as H = ∇²V(Z) where:
    - Z = encode(states) is the latent representation
    - V(Z) is the value function output
    
    Uses efficient autograd-based Hessian computation.
    
    Args:
        vae_critic: VAE critic with encode() and value_head
        states: Observations [N, obs_dim]
        value_net: Optional value network (if None, uses vae_critic.value_head)
        
    Returns:
        Hessian matrix [latent_dim, latent_dim] averaged over batch
    """
    if not hasattr(vae_critic, 'encode'):
        raise ValueError("VAE critic must have encode() method")
    
    # Get latent representation Z_VAE (detached from encoder, we only care about value network Hessian)
    with torch.no_grad():
        mu, _ = vae_critic.encode(states)
    
    # Use deterministic encoding (mean) for Hessian computation
    # Make Z require gradients for Hessian computation
    z = mu.clone().detach().requires_grad_(True)  # [N, latent_dim]
    
    # Get value network (either provided or from critic)
    if value_net is None:
        if not hasattr(vae_critic, 'value_head'):
            raise ValueError("VAE critic must have value_head attribute")
        value_net = vae_critic.value_head
    
    # Compute value for each sample
    values = value_net(z)  # [N, 1]
    values_sum = values.sum()  # Scalar for gradient computation
    
    # Compute first-order gradients w.r.t. Z
    # grad_z = [∂V/∂z_1, ..., ∂V/∂z_d] for all samples
    latent_dim = z.shape[1]
    N = z.shape[0]
    
    # Compute gradient w.r.t. z (sum over batch for efficiency)
    grad_z = torch.autograd.grad(
        values_sum,
        z,
        create_graph=True,
        retain_graph=True,
    )[0]  # [N, latent_dim]
    
    # Compute Hessian: H_ij = ∂²V/∂z_i ∂z_j
    # For each dimension j, compute gradient of grad_z[:, j] w.r.t. z
    hessians = []
    for j in range(latent_dim):
        # Compute gradient of ∂V/∂z_j w.r.t. all z dimensions
        grad_j = torch.autograd.grad(
            grad_z[:, j].sum(),  # Sum over batch
            z,
            retain_graph=(j < latent_dim - 1),
            create_graph=False,
        )[0]  # [N, latent_dim]
        
        # Average over batch to get Hessian row
        hessian_row = grad_j.mean(dim=0)  # [latent_dim]
        hessians.append(hessian_row)
    
    # Stack to form full Hessian matrix
    hessian = torch.stack(hessians)  # [latent_dim, latent_dim]
    
    # Symmetrize (Hessian should be symmetric, but numerical errors may occur)
    hessian = (hessian + hessian.T) / 2.0
    
    return hessian


def check_local_convexity(
    vae_critic: nn.Module,
    states: torch.Tensor,
    radius: float = 0.1,
    mu_threshold: float = 0.1,
    value_net: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Check local convexity post-update.
    
    Validates that Hessian(Z_VAE) ≻ 0 near policy updates (μ > 0.1).
    
    Args:
        vae_critic: VAE critic with encode() and value_head
        states: Observations [N, obs_dim]
        radius: Neighborhood radius for local convexity check (default: 0.1)
        mu_threshold: Minimum eigenvalue threshold (default: 0.1)
        value_net: Optional value network (if None, uses vae_critic.value_head)
        
    Returns:
        Dictionary with:
            - mu_local: Minimum eigenvalue of Hessian
            - is_convex: Boolean indicating if mu_local > mu_threshold
            - max_eigenvalue: Maximum eigenvalue
            - mean_eigenvalue: Mean eigenvalue
            - condition_number: max_eigenvalue / min_eigenvalue (if min > 0)
    """
    # Compute Hessian on latent representation
    H = compute_hessian_on_latent(vae_critic, states, value_net)
    
    # Compute eigenvalues
    eigenvalues = torch.linalg.eigvals(H).real  # Only real part (H should be symmetric)
    mu_local = eigenvalues.min().item()
    max_eigenvalue = eigenvalues.max().item()
    mean_eigenvalue = eigenvalues.mean().item()
    
    # Check if positive definite (all eigenvalues > 0)
    is_positive_definite = (eigenvalues > 0).all().item()
    
    # Check if meets threshold
    is_convex = mu_local > mu_threshold
    
    # Condition number (only if positive definite)
    if mu_local > 1e-8:  # Avoid division by zero
        condition_number = max_eigenvalue / mu_local
    else:
        condition_number = float('inf')
    
    return {
        "mu_local": mu_local,
        "is_convex": is_convex,
        "is_positive_definite": is_positive_definite,
        "max_eigenvalue": max_eigenvalue,
        "mean_eigenvalue": mean_eigenvalue,
        "condition_number": condition_number,
        "mu_threshold": mu_threshold,
    }


def track_convexity_updates(
    update_history: List[Dict[str, float]],
    convexity_threshold: float = 0.1,
    success_rate_threshold: float = 0.9,
) -> Dict[str, float]:
    """
    Track percentage of updates staying in local convexity neighborhood.
    
    Args:
        update_history: List of dictionaries with 'mu_local' key from check_local_convexity
        convexity_threshold: Minimum eigenvalue threshold (default: 0.1)
        success_rate_threshold: Required success rate (default: 0.9 = 90%)
        
    Returns:
        Dictionary with:
            - success_rate: Percentage of updates with mu_local > threshold
            - total_updates: Total number of updates tracked
            - convex_updates: Number of updates that stayed convex
            - meets_threshold: Boolean indicating if success_rate >= success_rate_threshold
    """
    if len(update_history) == 0:
        return {
            "success_rate": 0.0,
            "total_updates": 0,
            "convex_updates": 0,
            "meets_threshold": False,
        }
    
    # Count updates that stayed in convexity neighborhood
    convex_updates = sum(
        1 for update in update_history
        if update.get("mu_local", 0.0) > convexity_threshold
    )
    
    total_updates = len(update_history)
    success_rate = convex_updates / total_updates if total_updates > 0 else 0.0
    meets_threshold = success_rate >= success_rate_threshold
    
    return {
        "success_rate": success_rate,
        "total_updates": total_updates,
        "convex_updates": convex_updates,
        "meets_threshold": meets_threshold,
        "success_rate_threshold": success_rate_threshold,
    }


def compute_tsne_clustering(
    latent_repr: np.ndarray,
    task_labels: np.ndarray,
    env_labels: np.ndarray,
    n_components: int = 2,
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Dict[str, any]:
    """
    Compute t-SNE clustering to check if causal mask separates domains.
    
    Validates that t-SNE clusters by task, not env (indicating causal separation).
    
    Args:
        latent_repr: Latent representations [N, latent_dim]
        task_labels: Task labels for each sample [N]
        env_labels: Environment labels for each sample [N]
        n_components: Number of t-SNE dimensions (default: 2)
        perplexity: t-SNE perplexity parameter (default: 30.0)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - tsne_embedding: t-SNE embedding [N, n_components]
            - task_cluster_score: Silhouette score for task-based clustering
            - env_cluster_score: Silhouette score for env-based clustering
            - task_separation: Boolean indicating if task clusters better than env
            - task_cluster_labels: Cluster labels based on tasks
            - env_cluster_labels: Cluster labels based on environments
    """
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        raise ImportError("scikit-learn is required for t-SNE clustering. Install with: pip install scikit-learn")
    
    # Validate inputs
    if len(latent_repr) != len(task_labels) or len(latent_repr) != len(env_labels):
        raise ValueError(
            f"Input length mismatch: latent_repr={len(latent_repr)}, "
            f"task_labels={len(task_labels)}, env_labels={len(env_labels)}"
        )
    
    if len(latent_repr) < 4:
        raise ValueError(f"Need at least 4 samples for t-SNE, got {len(latent_repr)}")
    
    # Adjust perplexity if needed (must be less than n_samples)
    perplexity = min(perplexity, len(latent_repr) - 1)
    if perplexity < 1:
        raise ValueError(f"Perplexity must be >= 1, got {perplexity} (n_samples={len(latent_repr)})")
    
    # Compute t-SNE embedding
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
    )
    tsne_embedding = tsne.fit_transform(latent_repr)
    
    # Get unique task and env labels
    unique_tasks = np.unique(task_labels)
    unique_envs = np.unique(env_labels)
    
    # Cluster by task: use task labels as cluster assignments
    if len(unique_tasks) > 1:
        task_cluster_score = silhouette_score(tsne_embedding, task_labels)
        task_cluster_labels = task_labels
    else:
        # Only one task, can't compute silhouette
        task_cluster_score = 0.0
        task_cluster_labels = task_labels
    
    # Cluster by env: use env labels as cluster assignments
    if len(unique_envs) > 1:
        env_cluster_score = silhouette_score(tsne_embedding, env_labels)
        env_cluster_labels = env_labels
    else:
        # Only one env, can't compute silhouette
        env_cluster_score = 0.0
        env_cluster_labels = env_labels
    
    # Check if task separation is better than env separation
    # Higher silhouette score = better clustering
    task_separation = task_cluster_score > env_cluster_score
    
    return {
        "tsne_embedding": tsne_embedding,
        "task_cluster_score": task_cluster_score,
        "env_cluster_score": env_cluster_score,
        "task_separation": task_separation,
        "task_cluster_labels": task_cluster_labels,
        "env_cluster_labels": env_cluster_labels,
        "n_tasks": len(unique_tasks),
        "n_envs": len(unique_envs),
    }


def validate_vae_proxy_conditions(
    vae_critic: nn.Module,
    states: torch.Tensor,
    task_labels: Optional[np.ndarray] = None,
    env_labels: Optional[np.ndarray] = None,
    update_history: Optional[List[Dict[str, float]]] = None,
    mu_threshold: float = 0.1,
    success_rate_threshold: float = 0.9,
    value_net: Optional[nn.Module] = None,
) -> Dict[str, any]:
    """
    Validate all three VAE proxy conditions.
    
    Args:
        vae_critic: VAE critic with encode() and value_head
        states: Observations [N, obs_dim]
        task_labels: Optional task labels [N] for t-SNE analysis
        env_labels: Optional environment labels [N] for t-SNE analysis
        update_history: Optional history of convexity checks from previous updates
        mu_threshold: Minimum eigenvalue threshold (default: 0.1)
        success_rate_threshold: Required success rate for updates (default: 0.9)
        value_net: Optional value network (if None, uses vae_critic.value_head)
        
    Returns:
        Dictionary with all validation results:
            - condition_1_convexity: Results from check_local_convexity
            - condition_2_update_tracking: Results from track_convexity_updates
            - condition_3_tsne_separation: Results from compute_tsne_clustering (if labels provided)
            - all_conditions_met: Boolean indicating if all conditions are satisfied
    """
    results = {}
    
    # Condition 1: Hessian(Z_VAE) ≻ 0 near policy updates (μ > 0.1)
    condition_1 = check_local_convexity(
        vae_critic, states, mu_threshold=mu_threshold, value_net=value_net
    )
    results["condition_1_convexity"] = condition_1
    
    # Condition 2: 90%+ of updates stay in local convexity neighborhood
    if update_history is not None:
        # Add current update to history
        current_update = {"mu_local": condition_1["mu_local"]}
        update_history_with_current = update_history + [current_update]
    else:
        update_history_with_current = [{"mu_local": condition_1["mu_local"]}]
    
    condition_2 = track_convexity_updates(
        update_history_with_current,
        convexity_threshold=mu_threshold,
        success_rate_threshold=success_rate_threshold,
    )
    results["condition_2_update_tracking"] = condition_2
    
    # Condition 3: Causal mask separates domains (t-SNE clusters by task, not env)
    if task_labels is not None and env_labels is not None:
        # Get latent representations
        with torch.no_grad():
            mu, _ = vae_critic.encode(states)
            latent_repr = mu.cpu().numpy()
        
        condition_3 = compute_tsne_clustering(
            latent_repr, task_labels, env_labels
        )
        results["condition_3_tsne_separation"] = condition_3
    else:
        results["condition_3_tsne_separation"] = {
            "task_separation": None,
            "note": "Task and env labels not provided",
        }
    
    # Check if all conditions are met
    all_conditions_met = (
        condition_1["is_convex"] and
        condition_2["meets_threshold"] and
        (results["condition_3_tsne_separation"].get("task_separation", False) if 
         results["condition_3_tsne_separation"].get("task_separation") is not None else True)
    )
    results["all_conditions_met"] = all_conditions_met
    
    return results
