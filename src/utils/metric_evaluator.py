"""
Metric evaluation and extraction module.
Handles compute-intensive metrics that are evaluated periodically.
"""

import torch
from typing import Dict, Any, Callable, Optional

from src.metrics.hessian import (
    compute_hessian_spectrum,
    compute_hessian_trace,
    is_affine_vae_value_head,
)
from src.metrics.value_head_types import is_quadratic_latent_value_head, is_quadratic_psd_value_head
from src.metrics.value_head_mu import value_head_mu_stats
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
)
from src.metrics.convexity_validation import diagnostic_step
from src.metrics.feature_rank import compute_feature_rank_metrics
from src.metrics.bounding_chain import (
    compute_bounding_chain_metrics,
    check_chain_directionality,
)
from src.theory_validation.z_ref_store import ZRefStore
from src.theory_validation.z_ref_expert import load_expert_critic, encode_z_ref_batch


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
        self.config = config
        self.ground_truth_repr_fn = ground_truth_repr_fn
        self.repr_net = repr_net
        
        self.vae_proxy_update_history = []
        self.prev_z = None
        self.convexity_step = 0
        self.running_best_return = float("-inf")

        self.z_ref_store: Optional[ZRefStore] = None
        self.z_ref_critic: Optional[torch.nn.Module] = None
        self.z_ref_device: Optional[str] = None

        z_ref_expert_weights = config.get("z_ref_expert_weights")
        z_ref_expert_config = config.get("z_ref_expert_config")
        if z_ref_expert_weights and z_ref_expert_config:
            device = config.get("z_ref_expert_device", "cuda")
            if device == "cuda" and not torch.cuda.is_available():
                device = "cpu"
            self.z_ref_critic = load_expert_critic(
                z_ref_expert_config, z_ref_expert_weights, device
            )
            self.z_ref_device = device
        else:
            z_ref_path = config.get("z_ref_path")
            if z_ref_path:
                self.z_ref_store = ZRefStore.load(z_ref_path)
    
    def evaluate_all(
        self,
        policy: torch.nn.Module,
        critic: torch.nn.Module,
        obs_buffer: torch.Tensor,
        old_policy: torch.nn.Module = None,
        old_critic: torch.nn.Module = None,
        episode_returns: list[float] = None,
        old_occupancy: Dict = None,
        gt_repr_buffer: torch.Tensor = None,
    ) -> Dict[str, float]:
        metrics = {}
        
        sample_size = min(128, len(obs_buffer))
        obs_sample_raw = obs_buffer[:sample_size]
        obs_sample = obs_sample_raw

        gt_sample = None
        if gt_repr_buffer is not None:
            gt_sample = gt_repr_buffer[:sample_size]
        
        with torch.no_grad():
            if hasattr(critic, 'get_latent_representation'):
                obs_sample = critic.get_latent_representation(obs_sample_raw)
            elif hasattr(critic, 'encode'):
                mu, _ = critic.encode(obs_sample_raw)
                obs_sample = mu
            elif self.repr_net is not None:
                obs_sample = self.repr_net(obs_sample_raw)
        
        if self.config.get("collect_hessian", False):
            use_latent_hessian = is_affine_vae_value_head(critic) or is_quadratic_psd_value_head(critic)
            hessian_obs = obs_sample_raw if use_latent_hessian else obs_sample
            hessian_results = compute_hessian_spectrum(critic, hessian_obs, top_k=10)
            metrics["hessian_min_eigenvalue"] = hessian_results["min_eigenvalue"]
            metrics["hessian_max_eigenvalue"] = hessian_results["max_eigenvalue"]
            metrics["hessian_mean_eigenvalue"] = hessian_results["mean_eigenvalue"]
            if hessian_results.get("affine_value_head"):
                metrics["hessian_affine_by_construction"] = 1.0
            if hessian_results.get("quadratic_psd_value_head"):
                metrics["hessian_quadratic_psd"] = 1.0
                metrics["mu_latent_analytic"] = hessian_results["mu_latent_analytic"]

            trace = compute_hessian_trace(critic, hessian_obs, num_samples=5)
            metrics["hessian_trace"] = trace

        if hasattr(critic, "value_head") and hasattr(critic, "latent_dim"):
            metrics.update(
                value_head_mu_stats(
                    critic,
                    obs_sample,
                    enforce_quadratic_agreement=is_quadratic_latent_value_head(critic),
                )
            )
        
        if self.config.get("collect_fisher", False):
            fisher_idx = compute_fisher_information_index(policy, obs_sample)
            metrics["fisher_information_index"] = fisher_idx
        
        if self.config.get("collect_causal_error", False) and gt_sample is not None:
            z_star = gt_sample.to(obs_sample.device)
            with torch.no_grad():
                z = self._get_latent_batch(critic, obs_sample_raw, obs_sample)
            errors = torch.norm(z - z_star, dim=1)
            metrics["causal_prediction_error"] = errors.mean().item()
            metrics["causal_error_max"] = errors.max().item()
            metrics["causal_error_std"] = errors.std().item()
        elif self.config.get("collect_causal_error", False) and self.ground_truth_repr_fn is not None:
            causal_results = compute_causal_prediction_error(
                critic, self.ground_truth_repr_fn, obs_sample_raw, repr_net=self.repr_net
            )
            metrics["causal_prediction_error"] = causal_results["error"]
            metrics["causal_error_max"] = causal_results["max_error"]
            metrics["causal_error_std"] = causal_results["std_error"]
        
        if self.config.get("collect_regret", False) and episode_returns is not None:
            regret_results = compute_policy_regret(episode_returns)
            metrics["mean_return"] = regret_results["mean_return"]
            metrics["std_return"] = regret_results["std_return"]
            if "regret_vs_optimal" in regret_results:
                metrics["regret_vs_optimal"] = regret_results["regret_vs_optimal"]
        
        if self.config.get("collect_occupancy", False):
            current_occupancy = compute_occupancy_measure(obs_buffer, discretize=True, grid_size=10)
            metrics["occupancy_entropy"] = current_occupancy["entropy"]
            metrics["occupancy_unique_states"] = current_occupancy["unique_states"]
            
            if old_occupancy is not None:
                kl = compute_occupancy_kl(current_occupancy, old_occupancy)
                metrics["occupancy_kl"] = kl
        
        if self.config.get("collect_kl", False) and old_policy is not None:
            with torch.no_grad():
                old_actions, old_log_probs = old_policy.get_action(obs_sample)
            kl = compute_policy_kl(policy, obs_sample, old_log_probs)
            metrics["kl_divergence"] = kl
        
        if self.config.get("collect_gradients", False):
            grad_mag = compute_value_gradient_magnitude(critic, obs_sample)
            metrics["value_gradient_magnitude"] = grad_mag
            
            if old_critic is not None:
                grad_diff = compute_value_gradient_difference(critic, old_critic, obs_sample)
                metrics["value_gradient_difference"] = grad_diff
        
        if self.config.get("collect_vae_proxy_validation", False):
            if hasattr(critic, 'encode') and hasattr(critic, 'value_head'):
                convexity_results = check_local_convexity(
                    critic,
                    obs_sample,
                    mu_threshold=self.config.get("vae_proxy_mu_threshold", 0.1),
                )
                metrics["vae_proxy_mu_local"] = convexity_results["mu_local"]
                metrics["vae_proxy_is_convex"] = convexity_results["is_convex"]
                metrics["vae_proxy_max_eigenvalue"] = convexity_results["max_eigenvalue"]
                metrics["vae_proxy_mean_eigenvalue"] = convexity_results["mean_eigenvalue"]
                
                self.vae_proxy_update_history.append({"mu_local": convexity_results["mu_local"]})
                if len(self.vae_proxy_update_history) > 100:
                    self.vae_proxy_update_history = self.vae_proxy_update_history[-100:]
                
                update_tracking = track_convexity_updates(
                    self.vae_proxy_update_history,
                    convexity_threshold=self.config.get("vae_proxy_mu_threshold", 0.1),
                    success_rate_threshold=self.config.get("vae_proxy_success_rate_threshold", 0.9),
                )
                metrics["vae_proxy_success_rate"] = update_tracking["success_rate"]
                metrics["vae_proxy_meets_threshold"] = update_tracking["meets_threshold"]
                metrics["vae_proxy_total_updates"] = update_tracking["total_updates"]
                metrics["vae_proxy_convex_updates"] = update_tracking["convex_updates"]
        
        if self.config.get("collect_convexity_validation", False):
            if hasattr(critic, 'encode') or hasattr(critic, 'get_latent_representation'):
                with torch.no_grad():
                    if hasattr(critic, 'get_latent_representation'):
                        z_current = critic.get_latent_representation(obs_sample_raw)
                    else:
                        mu, _ = critic.encode(obs_sample_raw)
                        z_current = mu
            elif self.repr_net is not None:
                z_current = obs_sample
            else:
                z_current = obs_sample_raw

            z_star_batch = self._lookup_z_ref_batch(
                obs_sample_raw, gt_sample, z_current.device
            )

            diagnostic_results = diagnostic_step(
                critic=critic,
                z=z_current,
                z_old=self.prev_z,
                z_star=z_star_batch,
                mu_min=self.config.get("convexity_mu_min", 0.05),
                neighborhood_radius=self.config.get("convexity_neighborhood_radius", 0.1),
                concavity_epsilon=self.config.get("concavity_epsilon", 1e-3),
                directional_epsilon=self.config.get("convexity_directional_epsilon", 1e-2),
                step=self.convexity_step,
            )
            
            metrics.update({
                "convexity_mu": diagnostic_results["convexity_mu"],
                "convexity_max_eigenvalue": diagnostic_results["convexity_max_eigenvalue"],
                "convexity_mean_eigenvalue": diagnostic_results["convexity_mean_eigenvalue"],
                "convexity_is_convex": float(diagnostic_results["convexity_is_convex"]),
                "convexity_pct_convex": diagnostic_results["convexity_pct_convex"],
                "convexity_mu_concave": diagnostic_results["convexity_mu_concave"],
                "convexity_mu_concave_mean": diagnostic_results["convexity_mu_concave_mean"],
                "convexity_pct_concave": diagnostic_results["convexity_pct_concave"],
                "neighborhood_pct": diagnostic_results["neighborhood_pct"],
                "neighborhood_max_distance": diagnostic_results["neighborhood_max_distance"],
                "neighborhood_mean_distance": diagnostic_results["neighborhood_mean_distance"],
            })
            if diagnostic_results["convexity_kappa_mean"] is not None:
                metrics.update({
                    "convexity_kappa": diagnostic_results["convexity_kappa"],
                    "convexity_kappa_min": diagnostic_results["convexity_kappa_min"],
                    "convexity_kappa_max": diagnostic_results["convexity_kappa_max"],
                    "convexity_kappa_mean": diagnostic_results["convexity_kappa_mean"],
                    "convexity_kappa_concave": diagnostic_results["convexity_kappa_concave"],
                    "convexity_kappa_concave_mean": diagnostic_results["convexity_kappa_concave_mean"],
                    "convexity_pct_negative_kappa": diagnostic_results["convexity_pct_negative_kappa"],
                    "convexity_kappa_n_valid": diagnostic_results["convexity_kappa_n_valid"],
                })
            
            if diagnostic_results["bound_correlation"] is not None:
                metrics.update({
                    "bound_correlation": diagnostic_results["bound_correlation"],
                    "bound_mape": diagnostic_results["bound_mape"],
                    "bound_delta_Z_actual": diagnostic_results["bound_delta_Z_actual"],
                    "bound_delta_Z_predicted": diagnostic_results["bound_delta_Z_predicted"],
                    "bound_mu_local": diagnostic_results["bound_mu_local"],
                    "bound_holds": float(diagnostic_results["bound_holds"]),
                })
            
            self.prev_z = z_current.clone().detach()
            self.convexity_step += 1

        if self.config.get("collect_theory_validation", False):
            z_theory = self._get_latent_batch(critic, obs_sample_raw, obs_sample)
            rank_metrics = compute_feature_rank_metrics(z_theory)
            metrics.update(rank_metrics)

            mean_return = 0.0
            if episode_returns and len(episode_returns) > 0:
                mean_return = float(sum(episode_returns) / len(episode_returns))
                self.running_best_return = max(self.running_best_return, mean_return)

            kl_val = metrics.get("kl_divergence", 0.0)
            smoothness_L = metrics.get("convexity_max_eigenvalue", None)
            mu_concave = metrics.get("convexity_mu_concave", 0.0)
            pct_concave = metrics.get("convexity_pct_concave", 0.0)
            kappa_concave = metrics.get("convexity_kappa_concave")
            pct_negative_kappa = metrics.get("convexity_pct_negative_kappa")
            curvature_concave = kappa_concave if kappa_concave is not None else mu_concave
            curvature_pct_bad = (
                pct_negative_kappa if pct_negative_kappa is not None else pct_concave
            )

            z_ref_batch = self._lookup_z_ref_batch(
                obs_sample_raw, gt_sample, z_theory.device
            )
            if self.z_ref_critic is not None:
                metrics["z_ref_live_expert"] = 1.0

            chain_metrics = compute_bounding_chain_metrics(
                critic=critic,
                z=z_theory,
                kl_divergence=kl_val,
                mean_episode_return=mean_return,
                running_best_return=max(self.running_best_return, mean_return),
                smoothness_L=smoothness_L,
                z_ref=z_ref_batch,
                mu_concave=curvature_concave,
                pct_concave=curvature_pct_bad,
                c_z=self.config.get("chain_c_z", 1.0),
                c1=self.config.get("chain_c1", 1.0),
                bound_unreliable_pct_threshold=self.config.get(
                    "bound_unreliable_pct_threshold", 0.10
                ),
            )
            metrics.update(chain_metrics)

            mu_val = metrics.get("convexity_kappa_min")
            if mu_val is None:
                mu_val = metrics.get("convexity_mu", metrics.get("hessian_min_eigenvalue", 0.05))
            direction_metrics = check_chain_directionality(
                performance_gap=chain_metrics["chain_performance_gap"],
                grad_z_v=chain_metrics["chain_grad_z_v"],
                mu=mu_val,
                chain_rhs_unscaled=chain_metrics["chain_rhs_unscaled"],
                chain_rhs_scaled=chain_metrics["chain_rhs_scaled"],
            )
            metrics.update(direction_metrics)

            metrics["theory_mu_vs_rank_product"] = mu_val * rank_metrics["log_effective_feature_rank_pr"]
            if metrics.get("convexity_kappa_mean") is not None:
                metrics["theory_kappa_vs_rank_product"] = (
                    metrics["convexity_kappa_mean"] * rank_metrics["log_effective_feature_rank_pr"]
                )
        
        return metrics

    def _lookup_z_ref_batch(
        self,
        obs_sample_raw: torch.Tensor,
        gt_sample: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.z_ref_critic is not None:
            obs_for_ref = obs_sample_raw.to(self.z_ref_device)
            return encode_z_ref_batch(self.z_ref_critic, obs_for_ref).to(device)
        if self.z_ref_store is not None:
            if gt_sample is None:
                raise ValueError(
                    "z_ref_path is set but gt_repr_buffer was not passed to evaluate_all"
                )
            return self.z_ref_store.lookup_batch(gt_sample).to(device)
        return None

    def _get_latent_batch(
        self,
        critic: torch.nn.Module,
        obs_raw: torch.Tensor,
        obs_encoded: torch.Tensor,
    ) -> torch.Tensor:
        if hasattr(critic, "get_latent_representation"):
            with torch.no_grad():
                return critic.get_latent_representation(obs_raw)
        if hasattr(critic, "encode"):
            with torch.no_grad():
                mu, _ = critic.encode(obs_raw)
                return mu
        if self.repr_net is not None:
            return obs_encoded
        return obs_raw
    
    def set_task_env_labels(self, task_labels, env_labels):
        self.task_labels = task_labels
        self.env_labels = env_labels
