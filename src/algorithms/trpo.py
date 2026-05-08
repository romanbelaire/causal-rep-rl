"""
Trust Region Policy Optimization (TRPO) algorithm implementation.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.optimize import minimize

from src.utils.representation_loss import compute_representation_loss_with_convexity


class TRPO:
    """
    TRPO algorithm with policy KL trust region.
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
        Initialize TRPO.
        
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
        self.max_kl = config.get("max_kl", 0.01)  # Trust region size
        self.damping = config.get("damping", 0.1)  # For conjugate gradient
        self.cg_iters = config.get("cg_iters", 10)  # Conjugate gradient iterations
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vae_coef = config.get("vae_coef", 0.1)  # VAE loss coefficient (reconstruction + KL)
        self.representation_loss_coef = config.get("representation_loss_coef", 0.0)  # Representation loss coefficient
        self.use_convexity_weighting = config.get("use_convexity_weighting", True)  # Weight by 1/μ
        self.hessian_compute_freq = config.get("hessian_compute_freq", 10)  # Compute Hessian every N steps
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        
        # Optimizers - always use unified optimizer for all models
        # Unified optimizer includes all parameters: policy + critic + repr_net (if exists)
        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
        self.unified_optimizer = optim.Adam(all_params, lr=self.lr)
        self.critic_optimizer = None
        
        # Track training step for Hessian computation frequency
        self._step = 0
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation."""
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
    
    def flat_grad(self, loss: torch.Tensor, parameters: list) -> torch.Tensor:
        """Flatten gradients."""
        grads = torch.autograd.grad(loss, parameters, create_graph=True)
        return torch.cat([g.view(-1) for g in grads])
    
    def flat_params(self, model: nn.Module) -> torch.Tensor:
        """Flatten parameters."""
        return torch.cat([p.view(-1) for p in model.parameters()])
    
    def set_flat_params(self, model: nn.Module, flat_params: torch.Tensor):
        """Set parameters from flat tensor."""
        prev_idx = 0
        for param in model.parameters():
            param_size = param.numel()
            param.data.copy_(flat_params[prev_idx:prev_idx + param_size].view(param.size()))
            prev_idx += param_size
    
    def conjugate_gradient(self, Ax_fn, b: torch.Tensor, nsteps: int = 10) -> torch.Tensor:
        """Conjugate gradient algorithm."""
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        
        for _ in range(nsteps):
            Ap = Ax_fn(p)
            alpha = rdotr / torch.dot(p, Ap)
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            p = r + (new_rdotr / rdotr) * p
            rdotr = new_rdotr
        
        return x
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict:
        """
        Update policy and critic using TRPO.
        
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
        
        # Get representation z for policy
        # Priority: repr_net > VAE encoder > raw observations
        if self.repr_net is not None:
            # Encode observations: s -> z
            z = self.repr_net(obs)
        elif hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
            # VAE critic: encode with gradients enabled so policy loss flows back to encoder
            # Use encode() directly (not get_latent_representation which might detach)
            if hasattr(self.critic, 'encode'):
                mu, log_std = self.critic.encode(obs)
                z = mu  # Use mean for deterministic representation
            else:
                # Fallback to get_latent_representation if encode() not available
                z = self.critic.get_latent_representation(obs)
        else:
            # No encoder: use raw observations
            z = obs
        
        # Update critic: For VAE, get VAE losses too; for others, just value
        vae_info = None
        if hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
            # VAE critic: get both value and VAE losses (reconstruction + KL)
            values, vae_info = self.critic(obs, return_latent=True)
            values = values.squeeze(-1)
        else:
            # ICNN/Feedforward: pass representation z
            values = self.critic(z).squeeze(-1)
        
        value_loss = nn.functional.mse_loss(values, returns)
        
        # Add VAE loss for VAE critics
        if vae_info is not None:
            vae_loss = vae_info["vae_loss"]  # recon_loss + beta * kl_loss
        else:
            vae_loss = torch.tensor(0.0, device=self.device)
        
        # Representation loss: L_rep = α * (1/μ) * ||∇_Z V(Z)||²
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
                    states=obs,
                    alpha=self.representation_loss_coef,
                    use_convexity_weighting=self.use_convexity_weighting,
                    hessian_compute_freq=self.hessian_compute_freq,
                    step=self._step,
                )
        
        # Total critic loss: value loss + VAE loss + representation loss
        critic_loss = value_loss + self.vae_coef * vae_loss + representation_loss
        
        self.unified_optimizer.zero_grad()
        critic_loss.backward()
        
        # Clip gradients for all components
        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
        torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
        self.unified_optimizer.step()
        
        # Get current policy outputs (policy takes z as input)
        log_probs, entropy = self.policy.evaluate_actions(z, actions)
        
        # Policy gradient
        policy_loss = -(log_probs * advantages).mean()
        
        # Compute KL divergence
        kl = (old_log_probs - log_probs).mean()
        
        # Compute gradient of policy loss
        # Include repr_net or VAE encoder parameters in policy gradient so it gets updated
        # This ensures policy loss flows back through the encoder
        policy_params = list(self.policy.parameters())
        if self.repr_net is not None:
            policy_params = list(self.repr_net.parameters()) + policy_params
        elif hasattr(self.critic, 'encoder'):
            # VAE critic: include encoder parameters so policy loss flows back
            policy_params = list(self.critic.encoder.parameters()) + list(self.critic.fc_mu.parameters()) + list(self.critic.fc_log_std.parameters()) + policy_params
        
        policy_grad = self.flat_grad(policy_loss, policy_params)
        
        # Fisher-vector product function
        def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
            kl_grad = self.flat_grad(kl, policy_params)
            kl_v = (kl_grad * v).sum()
            kl_grad_grad = self.flat_grad(kl_v, policy_params)
            return kl_grad_grad + self.damping * v
        
        # Natural gradient using conjugate gradient
        stepdir = self.conjugate_gradient(fisher_vector_product, -policy_grad, self.cg_iters)
        
        # Line search
        shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum()
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = stepdir / lm
        
        # Try full step first
        # Get old params for both policy and repr_net
        if self.repr_net is not None:
            old_params = torch.cat([self.flat_params(self.policy), self.flat_params(self.repr_net)])
        else:
            old_params = self.flat_params(self.policy)
        
        def get_loss_kl(params: torch.Tensor) -> tuple[float, float]:
            # Set params for both policy and repr_net
            if self.repr_net is not None:
                policy_size = sum(p.numel() for p in self.policy.parameters())
                policy_params = params[:policy_size]
                repr_params = params[policy_size:]
                self.set_flat_params(self.policy, policy_params)
                self.set_flat_params(self.repr_net, repr_params)
                # Recompute z with updated repr_net
                z = self.repr_net(obs)
            else:
                self.set_flat_params(self.policy, params)
            
            with torch.no_grad():
                # Policy takes z as input
                new_log_probs, _ = self.policy.evaluate_actions(z, actions)
                new_loss = -(new_log_probs * advantages).mean()
                new_kl = (old_log_probs - new_log_probs).mean()
            return new_loss.item(), new_kl.item()
        
        # Line search
        expected_improve = (policy_grad * fullstep).sum()
        success = False
        for fraction in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            new_params = old_params + fraction * fullstep
            new_loss, new_kl = get_loss_kl(new_params)
            if new_kl <= self.max_kl and new_loss < policy_loss.item():
                success = True
                break
        
        if not success:
            # If line search failed, use small step
            new_params = old_params + 0.1 * fullstep
            new_loss, new_kl = get_loss_kl(new_params)
        
        stats = {
            "policy_loss": new_loss,
            "value_loss": value_loss.item(),
            "entropy": entropy.mean().item(),
            "kl": new_kl,
        }
        
        # Add representation loss stats if available
        if self.representation_loss_coef > 0 and representation_loss_stats:
            stats["representation_loss"] = representation_loss_stats.get('representation_loss', 0.0)
            stats["repr_grad_norm"] = representation_loss_stats.get('grad_norm', 0.0)
            stats["repr_mu_estimate"] = representation_loss_stats.get('mu_estimate', 0.0)
        
        # Increment step counter for Hessian computation frequency
        self._step += 1
        
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

