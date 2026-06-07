"""
Proximal Policy Optimization (PPO) algorithm implementation.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.architectures.latent_dynamics import LatentDynamicsModel
from src.theory_validation.z_ref_expert import encode_z_ref_batch
from src.utils.bisimulation_utils import VAEEncoderTarget, soft_update_module
from src.utils.dbc_loss import compute_dbc_loss
from src.utils.mico_loss import compute_mico_loss
from src.utils.representation_loss import (
    compute_representation_loss_with_convexity,
    effective_representation_loss_coef,
)
from src.utils.z_star_training_losses import (
    compute_kappa_directional_loss,
    compute_z_distill_loss,
    effective_intervention_loss_coef,
)


class PPO:
    """
    PPO algorithm with standard policy KL trust region.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config: dict,
        device: str = "cuda",
        repr_net: nn.Module = None,
        action_dim: int | None = None,
    ):
        """
        Initialize PPO.
        
        Args:
            policy: Policy network (takes Z as input)
            critic: Value function critic (takes Z as input)
            config: Algorithm configuration dict
            device: Device to run on
            repr_net: Shared representation network (s -> z), optional
        """
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.repr_net = repr_net.to(device) if repr_net is not None else None
        self.device = device
        
        # Hyperparameters
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vae_coef = config.get("vae_coef", 0.1)  # VAE loss coefficient (reconstruction + KL)
        self.representation_loss_coef = config.get("representation_loss_coef", 0.0)  # Representation loss coefficient
        self.representation_loss_coef_warmup_epochs = config.get(
            "representation_loss_coef_warmup_epochs", 0
        )  # 0 = no warmup
        self.use_convexity_weighting = config.get("use_convexity_weighting", True)  # Weight by -μ
        self.convexity_coef = config.get("convexity_coef", 1.0)  # Coefficient for μ term in loss
        self.grad_norm_power = config.get("grad_norm_power", 1.0)  # Power for gradient norm (1.0 = L2 norm, 2.0 = squared)
        self.hessian_compute_freq = config.get("hessian_compute_freq", 10)  # Compute Hessian every N steps
        self.kappa_directional_loss_coef = config.get("kappa_directional_loss_coef", 0.0)
        self.z_distill_loss_coef = config.get("z_distill_loss_coef", 0.0)
        self.kappa_directional_loss_coef_warmup_epochs = config.get(
            "kappa_directional_loss_coef_warmup_epochs", 0
        )
        self.z_distill_loss_coef_warmup_epochs = config.get(
            "z_distill_loss_coef_warmup_epochs", 0
        )
        self.kappa_directional_epsilon = config.get("kappa_directional_epsilon", 0.01)
        self.kappa_directional_min_distance = config.get("kappa_directional_min_distance", 1e-6)
        self.mico_loss_coef = config.get("mico_loss_coef", 0.0)
        self.mico_loss_coef_warmup_epochs = config.get("mico_loss_coef_warmup_epochs", 0)
        self.mico_beta = config.get("mico_beta", 0.1)
        self.mico_huber_delta = config.get("mico_huber_delta", 1.0)
        self.mico_target_update_tau = config.get("mico_target_update_tau", 0.005)
        self.mico_embed_ball_radius = config.get("mico_embed_ball_radius", None)
        self.dbc_loss_coef = config.get("dbc_loss_coef", 0.0)
        self.dbc_loss_coef_warmup_epochs = config.get("dbc_loss_coef_warmup_epochs", 0)
        self.dbc_gamma = config.get("dbc_gamma", None)
        self.dbc_huber_delta = config.get("dbc_huber_delta", 1.0)
        self.dbc_target_update_tau = config.get("dbc_target_update_tau", 0.005)
        self.dbc_embed_ball_radius = config.get("dbc_embed_ball_radius", None)
        self.dbc_dynamics_hidden = config.get("dbc_dynamics_hidden", [64, 64])
        self.dbc_dynamics_activation = config.get("dbc_dynamics_activation", "gelu")
        if self.mico_loss_coef > 0 and self.dbc_loss_coef > 0:
            raise ValueError("mico_loss_coef and dbc_loss_coef are mutually exclusive")
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        self.z_ref_expert = None
        self.encoder_target = None
        self.latent_dynamics = None
        self.latent_dynamics_target = None

        if self.mico_loss_coef > 0:
            if not hasattr(self.critic, "encode"):
                raise ValueError("MICo loss requires VAE critic with encode()")
            self.encoder_target = VAEEncoderTarget(self.critic).to(device)

        if self.dbc_loss_coef > 0:
            if action_dim is None:
                raise ValueError("dbc_loss_coef > 0 requires action_dim")
            if not hasattr(self.critic, "encode"):
                raise ValueError("DBC loss requires VAE critic with encode()")
            latent_dim = self.critic.latent_dim
            self.latent_dynamics = LatentDynamicsModel(
                latent_dim,
                action_dim,
                self.dbc_dynamics_hidden,
                self.dbc_dynamics_activation,
            ).to(device)
            self.latent_dynamics_target = copy.deepcopy(self.latent_dynamics)
            for param in self.latent_dynamics_target.parameters():
                param.requires_grad = False
        
        # Optimizers - always use unified optimizer for all models
        # Unified optimizer includes all parameters: policy + critic + repr_net (if exists)
        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
        if self.latent_dynamics is not None:
            all_params = list(self.latent_dynamics.parameters()) + all_params
        self.unified_optimizer = optim.Adam(all_params, lr=self.lr)
        self.policy_optimizer = None
        self.critic_optimizer = None
        self.repr_optimizer = None
        
        # Store old policy for KL computation
        self.old_policy = None
        
        # Track training step for Hessian computation frequency
        self._step = 0
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            rewards: Rewards [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value of next state after trajectory
            
        Returns:
            advantages: Advantages [T]
            returns: Returns [T]
        """
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
            next_value = values[t]
        
        returns = advantages + values
        return advantages, returns
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        rewards: torch.Tensor | None = None,
        next_obs: torch.Tensor | None = None,
        training_epoch: int | None = None,
    ) -> dict:
        """
        Update policy and critic networks.
        
        Args:
            obs: Observations [N, obs_dim]
            actions: Actions [N]
            old_log_probs: Old log probabilities [N]
            advantages: Advantages [N]
            returns: Returns [N]
            
        Returns:
            Dictionary with training statistics
        """
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        effective_repr_coef = effective_representation_loss_coef(
            self.representation_loss_coef,
            self.representation_loss_coef_warmup_epochs,
            training_epoch,
        )
        effective_kappa_coef = effective_intervention_loss_coef(
            self.kappa_directional_loss_coef,
            self.kappa_directional_loss_coef_warmup_epochs,
            training_epoch,
        )
        effective_distill_coef = effective_intervention_loss_coef(
            self.z_distill_loss_coef,
            self.z_distill_loss_coef_warmup_epochs,
            training_epoch,
        )
        effective_mico_coef = effective_intervention_loss_coef(
            self.mico_loss_coef,
            self.mico_loss_coef_warmup_epochs,
            training_epoch,
        )
        effective_dbc_coef = effective_intervention_loss_coef(
            self.dbc_loss_coef,
            self.dbc_loss_coef_warmup_epochs,
            training_epoch,
        )

        use_bisim = effective_mico_coef > 0 or effective_dbc_coef > 0
        if use_bisim:
            if rewards is None or next_obs is None:
                raise ValueError("MICo/DBC loss requires rewards and next_obs in update()")

        # Create dataset
        if use_bisim:
            dataset = TensorDataset(
                obs, actions, old_log_probs, advantages, returns, rewards, next_obs
            )
        else:
            dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl = 0
        repr_grad_norms = []  # Track encoder gradient norms
        
        # Track raw loss components (before coefficients) for detailed printing
        raw_loss_components = {
            'value_loss_raw': [],
            'vae_recon_loss_raw': [],
            'vae_kl_loss_raw': [],
            'vae_loss_raw': [],
            'representation_grad_norm_sq': [],
            'representation_mu': [],
            'representation_loss_raw': [],  # -mu * grad_norm_sq (before alpha)
            'policy_loss_raw': [],
            'entropy_raw': [],
        }
        
        for epoch in range(self.num_epochs):
            for batch in dataloader:
                if use_bisim:
                    (
                        batch_obs,
                        batch_actions,
                        batch_old_log_probs,
                        batch_advantages,
                        batch_returns,
                        batch_rewards,
                        batch_next_obs,
                    ) = batch
                    batch_rewards = batch_rewards.to(self.device)
                    batch_next_obs = batch_next_obs.to(self.device)
                else:
                    batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch

                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_advantages = batch_advantages.to(self.device)
                batch_returns = batch_returns.to(self.device)
                
                # Get representation z for policy
                # Priority: repr_net > VAE encoder > raw observations
                # For VAE critics, we need to compute z with gradients enabled so policy loss flows back
                vae_info = None
                if self.repr_net is not None:
                    # Encode observations: s -> z
                    # Ensure batch_obs is on the correct device
                    batch_obs = batch_obs.to(self.device)
                    z = self.repr_net(batch_obs)
                    # Verify z has correct shape (should be [batch_size, repr_dim])
                    if z.shape[-1] != self.repr_net.repr_dim:
                        raise RuntimeError(
                            f"Representation network output shape mismatch: "
                            f"expected last dim={self.repr_net.repr_dim}, got {z.shape[-1]}. "
                            f"Input shape: {batch_obs.shape}, Output shape: {z.shape}"
                        )
                elif hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
                    # VAE critic: encode with gradients enabled so policy loss flows back to encoder
                    # Use encode() directly (not get_latent_representation which might detach)
                    if hasattr(self.critic, 'encode'):
                        mu, log_std = self.critic.encode(batch_obs)
                        z = mu  # Use mean for deterministic representation
                    else:
                        # Fallback to get_latent_representation if encode() not available
                        z = self.critic.get_latent_representation(batch_obs)
                else:
                    # No encoder: flatten image observations if needed
                    if batch_obs.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = batch_obs.shape
                        z = batch_obs.view(N, H * W * C)
                    else:
                        z = batch_obs
                
                # Get current policy outputs (policy takes z as input)
                log_probs, entropy = self.policy.evaluate_actions(z, batch_actions)
                
                # Policy loss (PPO clip)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Track raw policy loss (before entropy coefficient)
                raw_loss_components['policy_loss_raw'].append(policy_loss.item())
                
                # Value loss: For VAE, get VAE losses too; for others, just value
                if hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
                    # VAE critic: get both value and VAE losses (reconstruction + KL)
                    # This ensures encoder gets gradients from value prediction
                    values, vae_info = self.critic(batch_obs, return_latent=True)
                    values = values.squeeze(-1)
                    # Extract VAE losses
                    recon_loss = vae_info["recon_loss"]
                    kl_loss = vae_info["kl_loss"]
                    vae_loss = vae_info["vae_loss"]  # recon_loss + beta * kl_loss
                    
                    # Track raw VAE loss components
                    raw_loss_components['vae_recon_loss_raw'].append(recon_loss.item())
                    raw_loss_components['vae_kl_loss_raw'].append(kl_loss.item())
                    raw_loss_components['vae_loss_raw'].append(vae_loss.item())
                else:
                    # ICNN/Feedforward: pass representation z
                    values = self.critic(z).squeeze(-1)
                    vae_loss = torch.tensor(0.0, device=values.device)
                    recon_loss = torch.tensor(0.0, device=values.device)
                    kl_loss = torch.tensor(0.0, device=values.device)
                
                value_loss = nn.functional.mse_loss(values, batch_returns)
                
                # Track raw value loss (before any coefficients)
                raw_loss_components['value_loss_raw'].append(value_loss.item())
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                raw_loss_components['entropy_raw'].append(entropy.mean().item())
                
                # Representation loss: L_rep = α * (-convexity_coef * μ) * ||∇_Z V(Z)||²
                representation_loss = torch.tensor(0.0, device=self.device)
                representation_loss_stats = {}
                if effective_repr_coef > 0:
                    # Determine encoder: repr_net > VAE encoder
                    encoder = None
                    if self.repr_net is not None:
                        encoder = self.repr_net
                    elif hasattr(self.critic, 'encode'):
                        encoder = self.critic
                    
                    if encoder is not None:
                        representation_loss, representation_loss_stats = compute_representation_loss_with_convexity(
                            encoder=encoder,
                            critic=self.critic,
                            states=batch_obs,
                            alpha=effective_repr_coef,
                            use_convexity_weighting=self.use_convexity_weighting,
                            convexity_coef=self.convexity_coef,
                            grad_norm_power=self.grad_norm_power,
                            hessian_compute_freq=self.hessian_compute_freq,
                            step=self._step,
                        )
                        
                        # Track raw representation loss components
                        if representation_loss_stats:
                            grad_norm = representation_loss_stats.get('grad_norm', 0.0)
                            grad_norm_powered = representation_loss_stats.get('grad_norm_powered', 0.0)
                            mu_est = representation_loss_stats.get('mu_estimate', 0.0)
                            # Raw loss = -convexity_coef * mu * grad_norm^power (before alpha coefficient)
                            raw_repr_loss = -self.convexity_coef * mu_est * grad_norm_powered
                            raw_loss_components['representation_grad_norm_sq'].append(grad_norm_powered)  # Keep name for compatibility
                            raw_loss_components['representation_mu'].append(mu_est)
                            raw_loss_components['representation_loss_raw'].append(raw_repr_loss)

                kappa_loss = torch.tensor(0.0, device=self.device)
                distill_loss = torch.tensor(0.0, device=self.device)
                intervention_stats = {}
                use_z_ref = (
                    self.z_ref_expert is not None
                    and (effective_kappa_coef > 0 or effective_distill_coef > 0)
                )
                if use_z_ref:
                    z_ref = encode_z_ref_batch(self.z_ref_expert, batch_obs)
                    if effective_kappa_coef > 0:
                        kappa_loss, kappa_stats = compute_kappa_directional_loss(
                            self.critic,
                            z,
                            z_ref,
                            effective_kappa_coef,
                            epsilon=self.kappa_directional_epsilon,
                            min_distance=self.kappa_directional_min_distance,
                        )
                        intervention_stats.update(kappa_stats)
                    if effective_distill_coef > 0:
                        distill_loss, distill_stats = compute_z_distill_loss(
                            z, z_ref, effective_distill_coef
                        )
                        intervention_stats.update(distill_stats)

                bisim_loss = torch.tensor(0.0, device=self.device)
                if effective_mico_coef > 0:
                    mico_raw, mico_stats = compute_mico_loss(
                        self.critic,
                        self.encoder_target,
                        batch_obs,
                        batch_next_obs,
                        batch_rewards,
                        effective_mico_coef,
                        self.gamma,
                        beta=self.mico_beta,
                        huber_delta=self.mico_huber_delta,
                        embed_ball_radius=self.mico_embed_ball_radius,
                        repr_net=self.repr_net,
                    )
                    bisim_loss = mico_raw
                    intervention_stats.update(mico_stats)
                elif effective_dbc_coef > 0:
                    dbc_gamma = self.dbc_gamma if self.dbc_gamma is not None else self.gamma
                    dbc_raw, dbc_stats = compute_dbc_loss(
                        self.critic,
                        self.latent_dynamics,
                        self.latent_dynamics_target,
                        batch_obs,
                        batch_next_obs,
                        batch_actions,
                        batch_rewards,
                        effective_dbc_coef,
                        dbc_gamma,
                        huber_delta=self.dbc_huber_delta,
                        embed_ball_radius=self.dbc_embed_ball_radius,
                        repr_net=self.repr_net,
                    )
                    bisim_loss = dbc_raw
                    intervention_stats.update(dbc_stats)

                # Zero gradients
                self.unified_optimizer.zero_grad()
                
                # Total critic loss: value loss + VAE loss + representation loss
                bisim_coef = effective_mico_coef if effective_mico_coef > 0 else effective_dbc_coef
                if bisim_coef > 0:
                    value_term = (1.0 - bisim_coef) * value_loss
                    bisim_term = bisim_coef * bisim_loss
                else:
                    value_term = value_loss
                    bisim_term = torch.tensor(0.0, device=self.device)

                critic_loss = (
                    value_term
                    + bisim_term
                    + self.vae_coef * vae_loss
                    + representation_loss
                    + kappa_loss
                    + distill_loss
                )
                
                # Backward pass: gradients flow to policy, critic, and repr_net
                # Policy loss flows back through z to encoder (for VAE) or repr_net (for ICNN)
                # Use retain_graph=True so we can compute gradients for both losses
                policy_loss_with_entropy = policy_loss + self.entropy_coef * entropy_loss
                policy_loss_with_entropy.backward(retain_graph=True)
                
                # Now backward pass for critic loss (value + VAE)
                # This ensures encoder gets gradients from both value prediction and VAE reconstruction
                critic_loss.backward()
                
                # Clip gradients for all components
                all_params = list(self.policy.parameters()) + list(self.critic.parameters())
                if self.repr_net is not None:
                    all_params = list(self.repr_net.parameters()) + all_params
                if self.latent_dynamics is not None:
                    all_params = list(self.latent_dynamics.parameters()) + all_params
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                
                # Track encoder gradient norms before clipping (for verification)
                repr_grad_norm_before_clip = None
                if self.repr_net is not None:
                    # Compute gradient norm before clipping (for stats only)
                    repr_grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.repr_net.parameters(), float('inf')  # Don't clip, just get norm
                    )
                
                # Update all networks with unified optimizer
                self.unified_optimizer.step()

                if self.encoder_target is not None:
                    self.encoder_target.soft_update_from(self.critic, self.mico_target_update_tau)
                if self.latent_dynamics_target is not None:
                    soft_update_module(
                        self.latent_dynamics_target,
                        self.latent_dynamics,
                        self.dbc_target_update_tau,
                    )
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Track representation loss stats
                if intervention_stats:
                    if not hasattr(self, "_intervention_stats"):
                        self._intervention_stats = []
                    self._intervention_stats.append(intervention_stats)

                if effective_repr_coef > 0 and representation_loss_stats:
                    if not hasattr(self, '_repr_loss_stats'):
                        self._repr_loss_stats = {
                            'representation_loss': [],
                            'grad_norm': [],
                            'mu_estimate': [],
                        }
                    self._repr_loss_stats['representation_loss'].append(representation_loss_stats.get('representation_loss', 0.0))
                    self._repr_loss_stats['grad_norm'].append(representation_loss_stats.get('grad_norm', 0.0))
                    self._repr_loss_stats['mu_estimate'].append(representation_loss_stats.get('mu_estimate', 0.0))
                
                # Track VAE losses if available
                if vae_info is not None:
                    if not hasattr(self, '_vae_recon_loss'):
                        self._vae_recon_loss = []
                        self._vae_kl_loss = []
                        self._vae_loss = []
                    self._vae_recon_loss.append(recon_loss.item())
                    self._vae_kl_loss.append(kl_loss.item())
                    self._vae_loss.append(vae_loss.item())
                
                # KL divergence (approximate)
                with torch.no_grad():
                    kl = (batch_old_log_probs - log_probs).mean().item()
                    total_kl += abs(kl)
                
                # Track encoder gradient norm (only on first batch of first epoch to avoid spam)
                if epoch == 0 and self.repr_net is not None and repr_grad_norm_before_clip is not None:
                    repr_grad_norms.append(repr_grad_norm_before_clip.item())
        
        # Increment step counter for Hessian computation frequency
        self._step += 1
        
        num_updates = self.num_epochs * len(dataloader)
        
        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / num_updates,
        }
        if self.representation_loss_coef > 0:
            stats["representation_loss_coef_effective"] = effective_repr_coef
        if self.kappa_directional_loss_coef > 0:
            stats["kappa_directional_loss_coef_effective"] = effective_kappa_coef
        if self.z_distill_loss_coef > 0:
            stats["z_distill_loss_coef_effective"] = effective_distill_coef
        if self.mico_loss_coef > 0:
            stats["mico_loss_coef_effective"] = effective_mico_coef
        if self.dbc_loss_coef > 0:
            stats["dbc_loss_coef_effective"] = effective_dbc_coef

        if hasattr(self, "_intervention_stats") and self._intervention_stats:
            keys = self._intervention_stats[0].keys()
            for key in keys:
                vals = [s[key] for s in self._intervention_stats if key in s]
                stats[key] = sum(vals) / len(vals)
            delattr(self, "_intervention_stats")

        # Add representation loss stats if available
        if hasattr(self, '_repr_loss_stats') and len(self._repr_loss_stats['representation_loss']) > 0:
            stats["representation_loss"] = sum(self._repr_loss_stats['representation_loss']) / len(self._repr_loss_stats['representation_loss'])
            stats["repr_grad_norm"] = sum(self._repr_loss_stats['grad_norm']) / len(self._repr_loss_stats['grad_norm'])
            stats["repr_mu_estimate"] = sum(self._repr_loss_stats['mu_estimate']) / len(self._repr_loss_stats['mu_estimate'])
            # Clear for next update
            delattr(self, '_repr_loss_stats')
        
        # Add encoder gradient norm if available
        if self.repr_net is not None and len(repr_grad_norms) > 0:
            stats["repr_grad_norm"] = sum(repr_grad_norms) / len(repr_grad_norms)
        
        # Add VAE losses if available
        if hasattr(self, '_vae_recon_loss') and len(self._vae_recon_loss) > 0:
            stats["vae_recon_loss"] = sum(self._vae_recon_loss) / len(self._vae_recon_loss)
            stats["vae_kl_loss"] = sum(self._vae_kl_loss) / len(self._vae_kl_loss)
            stats["vae_loss"] = sum(self._vae_loss) / len(self._vae_loss)
            # Clear for next update
            delattr(self, '_vae_recon_loss')
            delattr(self, '_vae_kl_loss')
            delattr(self, '_vae_loss')
        
        # Add raw loss components for detailed printing
        if len(raw_loss_components['value_loss_raw']) > 0:
            stats['raw_loss_components'] = {
                k: sum(v) / len(v) if len(v) > 0 else 0.0 
                for k, v in raw_loss_components.items()
            }
        
        return stats
    
    def save(self, path: str):
        """Save policy and critic weights."""
        save_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "unified_optimizer": self.unified_optimizer.state_dict(),
        }
        if self.repr_net is not None:
            save_dict["repr_net"] = self.repr_net.state_dict()
        if self.latent_dynamics is not None:
            save_dict["latent_dynamics"] = self.latent_dynamics.state_dict()
        if self.encoder_target is not None:
            save_dict["encoder_target"] = self.encoder_target.state_dict()
        if self.latent_dynamics_target is not None:
            save_dict["latent_dynamics_target"] = self.latent_dynamics_target.state_dict()
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load policy and critic weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.unified_optimizer.load_state_dict(checkpoint["unified_optimizer"])
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])
        if self.latent_dynamics is not None and "latent_dynamics" in checkpoint:
            self.latent_dynamics.load_state_dict(checkpoint["latent_dynamics"])
        if self.encoder_target is not None and "encoder_target" in checkpoint:
            self.encoder_target.load_state_dict(checkpoint["encoder_target"])
        if self.latent_dynamics_target is not None and "latent_dynamics_target" in checkpoint:
            self.latent_dynamics_target.load_state_dict(checkpoint["latent_dynamics_target"])

