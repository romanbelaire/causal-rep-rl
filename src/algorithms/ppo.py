"""
Proximal Policy Optimization (PPO) algorithm implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.utils.representation_loss import compute_representation_loss_with_convexity


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
        self.use_convexity_weighting = config.get("use_convexity_weighting", True)  # Weight by -μ
        self.convexity_coef = config.get("convexity_coef", 1.0)  # Coefficient for μ term in loss
        self.grad_norm_power = config.get("grad_norm_power", 1.0)  # Power for gradient norm (1.0 = L2 norm, 2.0 = squared)
        self.hessian_compute_freq = config.get("hessian_compute_freq", 10)  # Compute Hessian every N steps
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        
        # Optimizers - always use unified optimizer for all models
        # Unified optimizer includes all parameters: policy + critic + repr_net (if exists)
        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
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
        
        # Create dataset
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
            for batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns in dataloader:
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
                if self.representation_loss_coef > 0:
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
                            alpha=self.representation_loss_coef,
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
                
                # Zero gradients
                self.unified_optimizer.zero_grad()
                
                # Total critic loss: value loss + VAE loss + representation loss
                # VAE loss ensures encoder learns good representations
                # Representation loss shrinks representation error via V-gradients
                critic_loss = value_loss + self.vae_coef * vae_loss + representation_loss
                
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
                
                # Statistics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                
                # Track representation loss stats
                if self.representation_loss_coef > 0 and representation_loss_stats:
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
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load policy and critic weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.unified_optimizer.load_state_dict(checkpoint["unified_optimizer"])
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])

