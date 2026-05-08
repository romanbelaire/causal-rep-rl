"""
Metric evaluation and extraction module.
Handles compute-intensive metrics that are evaluated periodically.
"""

import torch
from typing import Dict, Any, Callable

# Import all metric computation functions at module level
# This ensures import errors are caught early, before experiments run
from src.metrics.hessian import compute_hessian_spectrum, compute_hessian_trace
from src.metrics.fisher import compute_fisher_information_index
from src.metrics.causal_error import compute_causal_prediction_error
from src.metrics.regret import compute_policy_regret
from src.metrics.occupancy import compute_occupancy_measure, compute_occupancy_kl
from src.metrics.kl_divergence import compute_policy_kl
from src.metrics.gradients import compute_value_gradient_magnitude, compute_value_gradient_difference
from src.metrics.vae_proxy_validation import (
    check_local_convexity,
    track_convexity_updates,
    compute_tsne_clustering,
    validate_vae_proxy_conditions,
)
from src.metrics.convexity_validation import (
    estimate_local_convexity,
    verify_representation_bound,
    check_neighborhood_membership,
    diagnostic_step,
)


class MetricEvaluator:
    """
    Evaluates metrics periodically (e.g., every N epochs) to reduce computational overhead.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        ground_truth_repr_fn: Callable = None,
        repr_net: torch.nn.Module = None,
    ):
        """
        Initialize metric evaluator.
        
        Args:
            config: Metrics configuration dict
            ground_truth_repr_fn: Function to extract ground-truth representation
            repr_net: Optional representation network to encode observations before metric computation
        """
        self.config = config
        self.ground_truth_repr_fn = ground_truth_repr_fn
        self.repr_net = repr_net
        
        # Track VAE proxy validation history
        self.vae_proxy_update_history = []
        
        # Track previous representations for convexity validation
        self.prev_z = None
        self.convexity_step = 0
    
    def evaluate_all(
        self,
        policy: torch.nn.Module,
        critic: torch.nn.Module,
        obs_buffer: torch.Tensor,
        old_policy: torch.nn.Module = None,
        old_critic: torch.nn.Module = None,
        episode_returns: list[float] = None,
        old_occupancy: Dict = None,
    ) -> Dict[str, float]:
        """
        Evaluate all enabled metrics.
        
        Args:
            policy: Current policy
            critic: Current critic
            obs_buffer: Buffer of observations [N, obs_dim]
            old_policy: Previous policy (for KL/gradient difference)
            old_critic: Previous critic (for gradient difference)
            episode_returns: List of episode returns (for regret)
            old_occupancy: Previous occupancy measure (for stability)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {}
        
        # Use subset of observations for efficiency
        sample_size = min(128, len(obs_buffer))
        obs_sample_raw = obs_buffer[:sample_size]  # Keep raw observations for causal error
        obs_sample = obs_sample_raw  # Will be encoded if needed
        
        # Encode observations for metrics that use critic/policy
        # Priority: VAE critic encoder > repr_net > raw observations
        with torch.no_grad():
            if hasattr(critic, 'get_latent_representation'):
                # VAE critic: use VAE encoder to get latent z (e.g., 32-dim)
                obs_sample = critic.get_latent_representation(obs_sample_raw)
            elif hasattr(critic, 'encode'):
                # VAE with encode method
                mu, _ = critic.encode(obs_sample_raw)
                obs_sample = mu
            elif self.repr_net is not None:
                # ICNN or other critics with separate representation network
                obs_sample = self.repr_net(obs_sample_raw)  # s -> z
            # else: obs_sample remains as raw observations (for feedforward critics)
        
        # Hessian spectrum (expensive)
        if self.config.get("collect_hessian", False):
            hessian_results = compute_hessian_spectrum(critic, obs_sample, top_k=10)
            metrics["hessian_min_eigenvalue"] = hessian_results["min_eigenvalue"]
            metrics["hessian_max_eigenvalue"] = hessian_results["max_eigenvalue"]
            metrics["hessian_mean_eigenvalue"] = hessian_results["mean_eigenvalue"]
            
            # Also compute trace
            trace = compute_hessian_trace(critic, obs_sample, num_samples=5)
            metrics["hessian_trace"] = trace
        
        # Fisher information
        if self.config.get("collect_fisher", False):
            fisher_idx = compute_fisher_information_index(policy, obs_sample)
            metrics["fisher_information_index"] = fisher_idx
        
        # Causal prediction error
        # Note: ground_truth_repr_fn expects raw observations, so pass obs_sample_raw
        if self.config.get("collect_causal_error", False) and self.ground_truth_repr_fn is not None:
            causal_results = compute_causal_prediction_error(critic, self.ground_truth_repr_fn, obs_sample_raw, repr_net=self.repr_net)
            metrics["causal_prediction_error"] = causal_results["error"]
            metrics["causal_error_max"] = causal_results["max_error"]
            metrics["causal_error_std"] = causal_results["std_error"]
        
        # Policy regret
        if self.config.get("collect_regret", False) and episode_returns is not None:
            regret_results = compute_policy_regret(episode_returns)
            metrics["mean_return"] = regret_results["mean_return"]
            metrics["std_return"] = regret_results["std_return"]
            if "regret_vs_optimal" in regret_results:
                metrics["regret_vs_optimal"] = regret_results["regret_vs_optimal"]
        
        # Occupancy measure stability
        if self.config.get("collect_occupancy", False):
            current_occupancy = compute_occupancy_measure(obs_buffer, discretize=True, grid_size=10)
            metrics["occupancy_entropy"] = current_occupancy["entropy"]
            metrics["occupancy_unique_states"] = current_occupancy["unique_states"]
            
            if old_occupancy is not None:
                kl = compute_occupancy_kl(current_occupancy, old_occupancy)
                metrics["occupancy_kl"] = kl
        
        # KL divergence (if old policy available)
        if self.config.get("collect_kl", False) and old_policy is not None:
            # Sample actions from old policy
            with torch.no_grad():
                old_actions, old_log_probs = old_policy.get_action(obs_sample)
            
            kl = compute_policy_kl(policy, obs_sample, old_log_probs)
            metrics["kl_divergence"] = kl
        
        # Gradient magnitude
        if self.config.get("collect_gradients", False):
            grad_mag = compute_value_gradient_magnitude(critic, obs_sample)
            metrics["value_gradient_magnitude"] = grad_mag
            
            # Gradient difference (if old critic available)
            if old_critic is not None:
                grad_diff = compute_value_gradient_difference(critic, old_critic, obs_sample)
                metrics["value_gradient_difference"] = grad_diff
        
        # VAE Proxy Validation (Exp 3 Success Conditions)
        if self.config.get("collect_vae_proxy_validation", False):
            # Check if critic is VAE-based
            if hasattr(critic, 'encode') and hasattr(critic, 'value_head'):
                # Condition 1: Check local convexity
                convexity_results = check_local_convexity(
                    critic,
                    obs_sample,
                    mu_threshold=self.config.get("vae_proxy_mu_threshold", 0.1),
                )
                metrics["vae_proxy_mu_local"] = convexity_results["mu_local"]
                metrics["vae_proxy_is_convex"] = convexity_results["is_convex"]
                metrics["vae_proxy_max_eigenvalue"] = convexity_results["max_eigenvalue"]
                metrics["vae_proxy_mean_eigenvalue"] = convexity_results["mean_eigenvalue"]
                
                # Condition 2: Track update history
                # Add current update to history
                self.vae_proxy_update_history.append({
                    "mu_local": convexity_results["mu_local"]
                })
                
                # Keep only recent history (last 100 updates)
                if len(self.vae_proxy_update_history) > 100:
                    self.vae_proxy_update_history = self.vae_proxy_update_history[-100:]
                
                # Compute success rate
                update_tracking = track_convexity_updates(
                    self.vae_proxy_update_history,
                    convexity_threshold=self.config.get("vae_proxy_mu_threshold", 0.1),
                    success_rate_threshold=self.config.get("vae_proxy_success_rate_threshold", 0.9),
                )
                metrics["vae_proxy_success_rate"] = update_tracking["success_rate"]
                metrics["vae_proxy_meets_threshold"] = update_tracking["meets_threshold"]
                metrics["vae_proxy_total_updates"] = update_tracking["total_updates"]
                metrics["vae_proxy_convex_updates"] = update_tracking["convex_updates"]
                
                # Condition 3: t-SNE clustering (requires task/env labels)
                # This would need to be provided separately or extracted from environment
                # For now, we skip this if labels aren't available
                if hasattr(self, 'task_labels') and hasattr(self, 'env_labels'):
                    try:
                        with torch.no_grad():
                            mu, _ = critic.encode(obs_sample)
                            latent_repr = mu.cpu().numpy()
                        
                        tsne_results = compute_tsne_clustering(
                            latent_repr,
                            self.task_labels[:len(latent_repr)],
                            self.env_labels[:len(latent_repr)],
                        )
                        metrics["vae_proxy_task_cluster_score"] = tsne_results["task_cluster_score"]
                        metrics["vae_proxy_env_cluster_score"] = tsne_results["env_cluster_score"]
                        metrics["vae_proxy_task_separation"] = tsne_results["task_separation"]
                    except Exception as e:
                        # Fail loudly for research codebase
                        raise RuntimeError(f"t-SNE clustering failed: {e}")
        
        # Convexity Hypothesis Validation
        if self.config.get("collect_convexity_validation", False):
            # Get current representations Z(s)
            # For VAE critics, always use VAE's own encoder (latent_dim, e.g., 32)
            # For ICNN critics, use repr_net if available (repr_dim, e.g., 512)
            # For Feedforward critics, use obs directly or repr_net if available
            
            # Priority 1: VAE critics - always use VAE encoder (not repr_net)
            if hasattr(critic, 'encode') or hasattr(critic, 'get_latent_representation'):
                # VAE critic: extract latent Z using VAE's own encoder
                # This gives us the correct latent_dim (e.g., 32), not repr_net's repr_dim (e.g., 512)
                with torch.no_grad():
                    if hasattr(critic, 'get_latent_representation'):
                        z_current = critic.get_latent_representation(obs_sample_raw)
                    else:
                        mu, _ = critic.encode(obs_sample_raw)
                        z_current = mu
            elif self.repr_net is not None:
                # ICNN or other critics with repr_net: use repr_net encoding
                # obs_sample is already encoded if repr_net is available
                z_current = obs_sample  # [N, repr_dim]
            else:
                # Feedforward critic without repr_net: use obs as "representation"
                z_current = obs_sample_raw  # [N, obs_dim]
            
            # Run diagnostic step
            try:
                diagnostic_results = diagnostic_step(
                    critic=critic,
                    z=z_current,
                    z_old=self.prev_z,
                    mu_min=self.config.get("convexity_mu_min", 0.05),
                    neighborhood_radius=self.config.get("convexity_neighborhood_radius", 0.1),
                    step=self.convexity_step,
                )
                
                # Add all diagnostic results to metrics
                metrics.update({
                    "convexity_mu": diagnostic_results["convexity_mu"],
                    "convexity_max_eigenvalue": diagnostic_results["convexity_max_eigenvalue"],
                    "convexity_mean_eigenvalue": diagnostic_results["convexity_mean_eigenvalue"],
                    "convexity_is_convex": float(diagnostic_results["convexity_is_convex"]),
                    "convexity_pct_convex": diagnostic_results["convexity_pct_convex"],
                    "neighborhood_pct": diagnostic_results["neighborhood_pct"],
                    "neighborhood_max_distance": diagnostic_results["neighborhood_max_distance"],
                    "neighborhood_mean_distance": diagnostic_results["neighborhood_mean_distance"],
                })
                
                # Add bound verification results if available
                if diagnostic_results["bound_correlation"] is not None:
                    metrics.update({
                        "bound_correlation": diagnostic_results["bound_correlation"],
                        "bound_mape": diagnostic_results["bound_mape"],
                        "bound_delta_Z_actual": diagnostic_results["bound_delta_Z_actual"],
                        "bound_delta_Z_predicted": diagnostic_results["bound_delta_Z_predicted"],
                        "bound_mu_local": diagnostic_results["bound_mu_local"],
                        "bound_holds": float(diagnostic_results["bound_holds"]),
                    })
                
                # Store current Z for next comparison
                self.prev_z = z_current.clone().detach()
                self.convexity_step += 1
                
            except Exception as e:
                # Fail loudly for research codebase
                raise RuntimeError(f"Convexity validation failed: {e}")
        
        return metrics
    
    def set_task_env_labels(self, task_labels, env_labels):
        """
        Set task and environment labels for t-SNE clustering analysis.
        
        Args:
            task_labels: Task labels array [N]
            env_labels: Environment labels array [N]
        """
        self.task_labels = task_labels
        self.env_labels = env_labels

