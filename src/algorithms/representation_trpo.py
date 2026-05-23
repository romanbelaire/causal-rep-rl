"""
Representation-Space Trust Region (RSTR) algorithm implementation.

Based on representation_space_trust_region.md specification.
Uses representation-space Fisher metric F_Z instead of KL-Fisher.
"""

import copy
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.architectures.forward_model import ForwardModel
from src.utils.representation_loss import (
    compute_representation_loss_with_convexity,
    effective_representation_loss_coef,
)


class RepresentationTRPO:
    """
    Representation-Space Trust Region algorithm.
    
    Constrains policy updates using representation-space distance:
    D_Z(θ', θ) = E[||Z_θ'(s) - Z_θ(s)||²] ≤ δ_Z
    
    Uses representation-space Fisher metric F_Z = E[J_Z(s)^T J_Z(s)]
    instead of KL-Fisher metric.
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
        Initialize Representation-Space TRPO.
        
        Args:
            policy: Policy network (maps s -> actions)
            critic: Critic network (maps z -> v for ICNN, or s -> v for others)
            config: Algorithm configuration dict
            device: Device to run on
            repr_net: Representation network (maps s -> z), required for ICNN critic
        """
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.repr_net = repr_net.to(device) if repr_net is not None else None
        self.device = device
        
        # Hyperparameters
        self.lr = config.get("learning_rate", 3e-4)
        self.critic_lr = config.get("critic_lr", None)  # Separate learning rate for critic (if None, use self.lr)
        if self.critic_lr is None:
            self.critic_lr = self.lr * 3.0  # Default: 3x higher learning rate for critic (not constrained by trust region)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.delta_z = config.get("delta_z", 0.01)  # Representation trust region size
        self.damping = config.get("damping", 0.1)  # For conjugate gradient
        self.cg_iters = config.get("cg_iters", 10)  # Conjugate gradient iterations
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.max_step_norm = config.get("max_step_norm", 1.0)  # Maximum step norm for policy updates (safety check)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vae_coef = config.get("vae_coef", 0.1)  # VAE loss coefficient (reconstruction + KL)
        self.representation_loss_coef = config.get("representation_loss_coef", 0.0)  # Representation loss coefficient
        self.representation_loss_coef_warmup_epochs = config.get(
            "representation_loss_coef_warmup_epochs", 0
        )
        self.use_convexity_weighting = config.get("use_convexity_weighting", True)  # Weight by 1/μ
        self.hessian_compute_freq = config.get("hessian_compute_freq", 10)  # Compute Hessian every N steps
        self.huber_delta = config.get("huber_delta", 1.0)  # Threshold for Huber loss (switches from quadratic to linear)
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        
        # Return normalization: standardize returns to mean=0, std=1
        # This ensures value loss is on similar scale to other losses (O(1) instead of O(1000))
        self.normalize_returns = config.get("normalize_returns", True)
        self.return_stats_ema = config.get("return_stats_ema", 0.99)  # EMA for return statistics (for monitoring)
        
        # Running statistics for return normalization (for monitoring only)
        self.return_mean = None
        self.return_std = None
        
        # Gradient/Hessian thresholding
        self.grad_clip = config.get("grad_clip", 10.0)  # c_g in spec
        self.hessian_clip = config.get("hessian_clip", 100.0)  # c_H in spec
        self.use_second_order = config.get("use_second_order", False)  # Use H in update
        
        # Unified backpropagation mode: update all components together vs phasic training
        self.use_unified_backprop = config.get("use_unified_backprop", False)
        
        # Store config for lazy initialization
        self.config = config
        
        # Track training step for Hessian computation frequency
        self._step = 0
        
        # Contrastive loss (causal loss) hyperparameters
        self.use_contrastive_loss = config.get("use_contrastive_loss", True)
        self.contrastive_coef = config.get("contrastive_coef", 1.0)  # Weight for contrastive loss
        
        # Diversity regularization: encourages representation diversity to prevent collapse
        self.diversity_coef = config.get("diversity_coef", 0.0)  # Weight for diversity regularization
        
        # Diversity regularization: encourages representation diversity to prevent collapse
        self.diversity_coef = config.get("diversity_coef", 0.0)  # Weight for diversity regularization
        
        # Smoothing for contrastive loss to reduce choppiness
        # EMA smoothing reduces variance in contrastive loss updates, leading to smoother training
        self.contrastive_ema_alpha = config.get("contrastive_ema_alpha", 0.99)  # EMA smoothing factor (0.99 = very smooth)
        self.contrastive_loss_ema = None  # Exponential moving average of contrastive loss
        
        # Target network for representation: provides stable targets for contrastive loss
        # Similar to DQN's target Q-network, this reduces variance by using slowly-updating targets
        self.use_target_network = config.get("use_target_network", True)
        self.target_update_tau = config.get("target_update_tau", 0.005)  # Soft update coefficient (0.005 = very slow)
        self.repr_net_target = None  # Target representation network (updated slowly)
        if self.repr_net is not None and self.use_target_network:
            # Create target network as a copy of the main network
            self.repr_net_target = copy.deepcopy(self.repr_net).to(device)
            # Freeze target network parameters (they'll be updated via soft updates)
            for param in self.repr_net_target.parameters():
                param.requires_grad = False
        
        # Forward model for temporal contrastive learning
        self.forward_model = None
        if self.repr_net is not None and self.use_contrastive_loss:
            # Get action dimension from policy
            # For discrete actions, we need the number of action classes
            if hasattr(policy, 'action_dim'):
                action_dim = policy.action_dim
            elif hasattr(policy, 'action_head'):
                # MLP policy with action_head
                action_dim = policy.action_head.out_features
            elif hasattr(policy, 'action_space_type'):
                # Try to find action head in network
                for name, module in policy.named_modules():
                    if 'action' in name.lower() and isinstance(module, nn.Linear):
                        action_dim = module.out_features
                        break
                else:
                    # Fallback: assume discrete actions, need to get from environment
                    # We'll set this later when we have access to actions
                    action_dim = None  # Will be inferred from first batch
            else:
                action_dim = None  # Will be inferred from first batch
            
            forward_hidden = config.get("forward_model_hidden", [256, 256])
            forward_activation = config.get("forward_model_activation", "relu")
            if action_dim is not None:
                self.forward_model = ForwardModel(
                    repr_dim=self.repr_net.repr_dim,
                    action_dim=action_dim,
                    hidden_sizes=forward_hidden,
                    activation=forward_activation,
                ).to(device)
            # If action_dim is None, we'll create forward_model lazily on first update
        
        # Optimizers - always use unified optimizer for all models
        # Unified optimizer includes all parameters: policy + critic + repr_net (if exists)
        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
        self.unified_optimizer = optim.Adam(all_params, lr=self.lr)
        self.policy_optimizer = None
        # Separate optimizer for critic with higher learning rate (critic not constrained by trust region)
        self.critic_optimizer = optim.Adam(list(self.critic.parameters()), lr=self.critic_lr)
        self.repr_optimizer = None
        
        if self.forward_model is not None:
            self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=self.lr)
    
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
    
    def get_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Extract representation Z(s) from observation.
        
        Architecture:
        - Representation network: s -> z (if available)
        - ICNN critic: z -> v (uses representation network)
        - VAE critic: s -> z (via encoder), then z -> v
        - Feedforward: s -> v (no separate representation)
        
        Args:
            obs: Observations 
                - For CNN-IMPALA: [N, H, W, C] (image format)
                - For MLP/IMPALA: [N, obs_dim] (flattened)
            
        Returns:
            Representation Z(s) [N, repr_dim]
        """
        if self.repr_net is not None:
            # Use separate representation network: s -> z
            # Representation network expects flattened input, so flatten if needed
            if obs.dim() == 4:  # [N, H, W, C] from CNN-IMPALA
                # Flatten image observations for representation network
                N, H, W, C = obs.shape
                obs_flat = obs.view(N, H * W * C)
            else:
                obs_flat = obs
            return self.repr_net(obs_flat)
        elif hasattr(self.critic, 'get_latent_representation'):
            # VAE critic: extract latent representation
            if obs.dim() == 4:  # [N, H, W, C]
                N, H, W, C = obs.shape
                obs_flat = obs.view(N, H * W * C)
            else:
                obs_flat = obs
            return self.critic.get_latent_representation(obs_flat)
        elif hasattr(self.critic, 'encode'):
            # VAE with encode method
            if obs.dim() == 4:  # [N, H, W, C]
                N, H, W, C = obs.shape
                obs_flat = obs.view(N, H * W * C)
            else:
                obs_flat = obs
            mu, _ = self.critic.encode(obs_flat)
            return mu
        else:
            # For Feedforward critic, use observation as representation
            # (no separate representation network)
            if obs.dim() == 4:  # [N, H, W, C]
                N, H, W, C = obs.shape
                obs_flat = obs.view(N, H * W * C)
            else:
                obs_flat = obs
            return obs_flat
    
    def compute_representation_jacobian(
        self,
        obs: torch.Tensor,
        params: list,
    ) -> torch.Tensor:
        """
        Compute representation Jacobian J_Z(s) = ∂Z_θ(s)/∂θ.
        
        For RSTR, we compute how the representation Z(s) changes w.r.t.
        the parameters θ. In our architecture, Z comes from the critic,
        so we compute J_Z w.r.t. critic parameters (since the critic is
        the encoder that produces Z).
        
        Args:
            obs: Observations [N, obs_dim] (must have requires_grad=True)
            params: List of parameters (critic parameters for F_Z computation)
            
        Returns:
            Jacobian [N, repr_dim, num_params]
        """
        # Ensure obs requires grad for gradient computation
        obs = obs.detach().requires_grad_(True)
        
        # Get representation (must be part of computation graph)
        z = self.get_representation(obs)  # [N, repr_dim]
        
        # Ensure z requires grad
        if not z.requires_grad:
            # If z doesn't require grad, it means it's not connected to params
            # This happens when Z = obs for ICNN. In that case, we need to
            # compute representation through critic output or intermediate features
            # For now, use critic output as representation
            z = self.critic(obs)
            if z.dim() > 1:
                z = z.squeeze(-1) if z.shape[-1] == 1 else z
        
        # Compute Jacobian: J_Z(s) = ∂Z(s)/∂θ for each sample
        jacobian_list = []
        
        for i in range(len(obs)):
            z_i = z[i]  # [repr_dim]
            jac_i = []
            
            for j in range(z_i.shape[0]):  # For each repr dimension
                z_ij = z_i[j]  # Scalar
                
                # Compute gradient w.r.t. all parameters at once (more efficient)
                grad_list = torch.autograd.grad(
                    z_ij,
                    params,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )
                
                # Flatten gradients
                grad_flat = []
                for grad in grad_list:
                    if grad is not None:
                        grad_flat.append(grad.view(-1))
                    else:
                        # If gradient is None, param doesn't affect z_ij
                        # Find which param this corresponds to
                        for param in params:
                            if param.requires_grad:
                                grad_flat.append(torch.zeros(param.numel(), device=z.device))
                                break
                
                if grad_flat:
                    jac_i.append(torch.cat(grad_flat))
            
            if jac_i:
                # Stack: [repr_dim, num_params]
                jac_i_tensor = torch.stack(jac_i)  # [repr_dim, num_params]
                jacobian_list.append(jac_i_tensor)
        
        if jacobian_list:
            # Stack: [N, repr_dim, num_params]
            return torch.stack(jacobian_list)
        
        # Fallback: return zeros
        num_params = sum(p.numel() for p in params if p.requires_grad)
        repr_dim = z.shape[1] if z.dim() > 1 else 1
        return torch.zeros(len(obs), repr_dim, num_params, device=z.device)
    
    def compute_representation_fisher(
        self,
        obs: torch.Tensor,
        params: list = None,
    ) -> torch.Tensor:
        """
        Compute representation-space Fisher metric F_Z = E[J_Z(s)^T J_Z(s)].
        
        According to the spec: F_Z = E[J_Z(s)^T J_Z(s)] where J_Z(s) = ∂Z_θ(s)/∂θ.
        
        In our architecture, Z comes from the critic, so we compute F_Z w.r.t.
        critic parameters. This Fisher metric is then used to constrain policy
        updates, ensuring they respect the representation-space geometry.
        
        Args:
            obs: Observations [N, obs_dim]
            params: List of parameters (default: critic parameters)
            
        Returns:
            Fisher matrix [num_params, num_params]
        """
        if params is None:
            # Use critic parameters (since critic produces Z)
            params = list(self.critic.parameters())
        
        # Get representation Jacobian
        J_Z = self.compute_representation_jacobian(obs, params)  # [N, repr_dim, num_params]
        
        # Compute F_Z = (1/N) * sum_i J_Z(s_i)^T J_Z(s_i)
        # For each sample: J_Z^T J_Z gives [num_params, num_params]
        N = J_Z.shape[0]
        F_Z = torch.zeros(J_Z.shape[2], J_Z.shape[2], device=J_Z.device)
        
        for i in range(N):
            J_i = J_Z[i]  # [repr_dim, num_params]
            F_Z += J_i.T @ J_i  # [num_params, num_params]
        
        F_Z = F_Z / N
        
        return F_Z
    
    def flat_grad(self, loss: torch.Tensor, parameters: list, create_graph: bool = True) -> torch.Tensor:
        """Flatten gradients."""
        grads = torch.autograd.grad(loss, parameters, create_graph=create_graph, retain_graph=create_graph)
        return torch.cat([g.view(-1) for g in grads if g is not None])
    
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
        
        # Check for NaN/Inf in initial residual
        if torch.isnan(rdotr) or torch.isinf(rdotr) or rdotr < 1e-10:
            return x  # Return zero step if initial residual is invalid
        
        for step in range(nsteps):
            Ap = Ax_fn(p)
            
            # Check for NaN/Inf in Ap
            if torch.isnan(Ap).any() or torch.isinf(Ap).any():
                break  # Stop if Ap contains invalid values
            
            pAp = torch.dot(p, Ap)
            if torch.isnan(pAp) or torch.isinf(pAp) or abs(pAp) < 1e-10:
                break  # Stop if denominator is invalid
            
            alpha = rdotr / pAp
            
            # Check for NaN/Inf in alpha
            if torch.isnan(alpha) or torch.isinf(alpha):
                break
            
            x += alpha * p
            r -= alpha * Ap
            new_rdotr = torch.dot(r, r)
            
            # Check for NaN/Inf in new residual
            if torch.isnan(new_rdotr) or torch.isinf(new_rdotr) or new_rdotr < 1e-10:
                break
            
            beta = new_rdotr / rdotr
            if torch.isnan(beta) or torch.isinf(beta):
                break
            
            p = r + beta * p
            rdotr = new_rdotr
            
            # Cleanup after each CG iteration to prevent memory accumulation
            # fisher_vector_product creates computation graphs, so we need to free them
            # OPTIMIZED: Only cleanup every 5 iterations to reduce overhead
            if step % 5 == 0:  # Every 5 iterations to balance performance vs overhead
                del Ap
                gc.collect()
                torch.cuda.empty_cache()
                # Removed synchronize() - too expensive, only needed for debugging
        
        # Final cleanup - Ap may not exist if loop broke early
        try:
            del Ap
        except NameError:
            pass
        del r, p
        # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
        # Removed expensive gc/sync calls
        
        # Check for NaN/Inf in final result
        if torch.isnan(x).any() or torch.isinf(x).any():
            return torch.zeros_like(b)  # Return zero step if result is invalid
        
        return x
    
    def clip_gradient(self, g: torch.Tensor) -> torch.Tensor:
        """Apply gradient clipping: g ← g * min(1, c_g / ||g||)."""
        g_norm = torch.norm(g)
        if g_norm > self.grad_clip:
            return g * (self.grad_clip / g_norm)
        return g
    
    def clip_hessian(self, H: torch.Tensor) -> torch.Tensor:
        """Apply Hessian spectral clipping: H ← U diag(min(λ_i, c_H)) U^T."""
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(H)
            eigenvalues_clipped = torch.clamp(eigenvalues, max=self.hessian_clip)
            H_clipped = eigenvectors @ torch.diag(eigenvalues_clipped) @ eigenvectors.T
            return H_clipped
        except Exception:
            # If eigendecomposition fails, return original
            return H
    
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        phase: str = "all",  # "representation", "critic", "policy", or "all"
        next_obs: torch.Tensor = None,  # Next observations for contrastive loss
        training_epoch: int | None = None,
    ) -> dict:
        """
        Update policy and critic using Representation-Space Trust Region.
        
        Args:
            obs: Observations [N, obs_dim]
            actions: Actions [N]
            old_log_probs: Old log probabilities [N]
            advantages: Advantages [N]
            returns: Returns [N]
            phase: Training phase - "representation", "critic", "policy", or "all"
            
        Returns:
            Dictionary with training statistics
        """
        # Initialize representation loss stats (will be populated if representation loss is computed)
        representation_loss_stats = {}
        
        warning_flag = False
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Normalize returns to have mean=0, std=1 (principled way to balance value loss scale)
        # This ensures value loss is on a similar scale to other losses (typically O(1) instead of O(1000))
        returns_mean = returns.mean()
        returns_std = returns.std() + 1e-8  # Add small epsilon to avoid division by zero
        returns_variance = returns.var() + 1e-8  # Variance for loss normalization
        
        if self.normalize_returns:
            returns = (returns - returns_mean) / returns_std
            
            # Update running statistics (optional, for monitoring)
            if self.return_mean is None:
                self.return_mean = returns_mean.item()
                self.return_std = returns_std.item()
            else:
                self.return_mean = self.return_stats_ema * self.return_mean + (1 - self.return_stats_ema) * returns_mean.item()
                self.return_std = self.return_stats_ema * self.return_std + (1 - self.return_stats_ema) * returns_std.item()
        
        # Store return statistics for loss normalization (even if we don't normalize returns)
        self._current_returns_mean = returns_mean
        self._current_returns_std = returns_std
        self._current_returns_variance = returns_variance

        effective_repr_coef = effective_representation_loss_coef(
            self.representation_loss_coef,
            self.representation_loss_coef_warmup_epochs,
            training_epoch,
        )
        
        # Phasic training: only update specified component
        value_loss = torch.tensor(0.0, device=obs.device)
        policy_loss = torch.tensor(0.0, device=obs.device)
        entropy = torch.tensor(0.0, device=obs.device)
        repr_dist = 0.0
        contrastive_loss_val = 0.0
        
        # Handle observation format: CNN-IMPALA uses [N, H, W, C], others use [N, obs_dim]
        # Representation network always expects flattened input
        if obs.dim() == 4:  # [N, H, W, C] from CNN-IMPALA policy
            N, H, W, C = obs.shape
            obs_flat = obs.view(N, H * W * C)
        else:
            obs_flat = obs
        
        # Cache representation z to avoid recomputing it multiple times
        # This is especially important for ICNN which can be expensive
        z_cached = None
        if self.repr_net is not None and phase in ["critic", "policy", "all"]:
            # Pre-compute representation once if it will be used in multiple phases
            z_cached = self.repr_net(obs_flat)
        
        # Zero gradients at start of update
        self.unified_optimizer.zero_grad()
        
        # Unified backprop mode: update all components together with gradients flowing through CNN
        if self.use_unified_backprop and self.repr_net is not None:
            # Forward pass: everything flows from obs through CNN
            z_t = self.repr_net(obs_flat)  # [N, repr_dim] - current representation
            
            # Get next representation if available
            z_next = None
            if next_obs is not None:
                if next_obs.dim() == 4:  # [N, H, W, C]
                    N_next, H_next, W_next, C_next = next_obs.shape
                    next_obs_flat = next_obs.view(N_next, H_next * W_next * C_next)
                else:
                    next_obs_flat = next_obs
                # Ensure next_obs_flat is on the correct device
                next_obs_flat = next_obs_flat.to(self.device)
                # Use target network for stable targets (if enabled), otherwise use current network
                if self.use_target_network and self.repr_net_target is not None:
                    with torch.no_grad():
                        z_next = self.repr_net_target(next_obs_flat)  # Stable target from target network
                else:
                    z_next = self.repr_net(next_obs_flat)  # Next representation [N, repr_dim]
            
            # Value head: z -> v
            values = self.critic(z_t).squeeze(-1)
            # Use Huber loss for robustness to outliers (more stable than MSE)
            value_loss_raw = nn.functional.huber_loss(values, returns, reduction='mean', delta=self.huber_delta)
            # Normalize value loss by variance of returns to make it scale-invariant
            value_loss = value_loss_raw / self._current_returns_variance
            
            # Policy head: z -> actions
            log_probs, entropy = self.policy.evaluate_actions(z_t, actions)
            policy_loss = -(log_probs * advantages).mean()
            
            # Auxiliary: forward model (causal coherence)
            contrastive_loss = torch.tensor(0.0, device=obs.device)
            contrastive_loss_val = 0.0
            if self.use_contrastive_loss and z_next is not None:
                # Lazy initialization of forward model if needed
                if self.forward_model is None:
                    if actions.dtype == torch.long:
                        action_dim = int(actions.max().item() + 1)
                    else:
                        action_dim = actions.shape[-1] if actions.dim() > 1 else 7
                    forward_hidden = self.config.get("forward_model_hidden", [256, 256]) if hasattr(self, 'config') else [256, 256]
                    forward_activation = self.config.get("forward_model_activation", "relu") if hasattr(self, 'config') else "relu"
                    self.forward_model = ForwardModel(
                        repr_dim=self.repr_net.repr_dim,
                        action_dim=action_dim,
                        hidden_sizes=forward_hidden,
                        activation=forward_activation,
                    ).to(self.device)
                    # Recreate unified optimizer with forward model params
                    all_params = list(self.repr_net.parameters()) + list(self.critic.parameters()) + list(self.forward_model.parameters())
                    self.unified_optimizer = optim.Adam(all_params, lr=self.lr)
                
                z_pred = self.forward_model(z_t, actions)  # Predict next representation
                contrastive_loss_raw = nn.functional.mse_loss(z_pred, z_next)
                contrastive_loss_val = contrastive_loss_raw.item()
                
                # Apply EMA smoothing to reduce choppiness
                if self.contrastive_loss_ema is None:
                    self.contrastive_loss_ema = contrastive_loss_val
                else:
                    self.contrastive_loss_ema = (
                        self.contrastive_ema_alpha * self.contrastive_loss_ema + 
                        (1 - self.contrastive_ema_alpha) * contrastive_loss_val
                    )
                
                # Use raw loss for gradient computation (EMA is only for monitoring)
                contrastive_loss = contrastive_loss_raw
            else:
                contrastive_loss = torch.tensor(0.0, device=obs.device)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # UNIFIED loss: all gradients flow through CNN
            # Scale value_loss to balance with other losses
            total_loss = (
                self.value_coef * value_loss + 
                policy_loss + 
                self.contrastive_coef * contrastive_loss +
                self.entropy_coef * entropy_loss
            )
            
            # REPRESENTATION-SPACE TRUST REGION CONSTRAINT ENFORCEMENT
            # Compute representation BEFORE update (for monitoring distance)
            with torch.no_grad():
                if self.repr_net is not None:
                    z_old = self.repr_net(obs_flat)
                else:
                    z_old = obs_flat
            
            # Parameters that affect representation: repr_net + policy (since policy uses z)
            # We constrain updates to these parameters using representation-space Fisher metric
            repr_params = []
            if self.repr_net is not None:
                repr_params.extend(list(self.repr_net.parameters()))
            # Policy parameters also affect representation indirectly (policy takes z as input)
            # But we primarily constrain repr_net since that's what produces Z(s)
            
            if len(repr_params) > 0:
                # Compute representation-space Fisher metric F_Z
                # F_Z = E[J_Z(s)^T J_Z(s)] where J_Z(s) = ∂Z_θ(s)/∂θ
                # We compute this w.r.t. repr_net parameters
                try:
                    # Don't compute F_Z explicitly - we compute Fisher-vector products implicitly
                    # This saves memory (F_Z would be [num_params, num_params] which is huge)
                    
                    # Compute policy gradient w.r.t. representation parameters
                    # We want to maximize policy objective while constraining representation change
                    self.unified_optimizer.zero_grad()
                    policy_loss.backward(retain_graph=True)
                    
                    # Get flattened gradient for representation parameters
                    repr_grad = self.flat_grad(policy_loss, repr_params)
                    
                    # Also include gradients from value and contrastive losses (they also affect representation)
                    if value_loss.item() > 0:
                        value_loss.backward(retain_graph=True)
                    if contrastive_loss.item() > 0:
                        contrastive_loss.backward(retain_graph=True)
                    
                    # Combine gradients: policy + value + contrastive (all affect representation)
                    combined_grad = self.flat_grad(total_loss, repr_params)
                    
                    # Clip gradient before computing natural gradient step (reduces step size naturally)
                    # This prevents extremely large steps that would need heavy clipping later
                    combined_grad_norm = torch.norm(combined_grad)
                    if combined_grad_norm > self.max_grad_norm:
                        combined_grad = combined_grad * (self.max_grad_norm / combined_grad_norm)
                    
                    # Fisher-vector product function for conjugate gradient
                    # F_Z * v computes the representation-space metric applied to vector v
                    # Optimized using R-operator (Pearlmutter's trick) for efficiency
                    def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
                        # F_Z @ v = (1/N) * sum_i J_Z(s_i)^T @ (J_Z(s_i) @ v)
                        # Use subset of samples and R-operator for efficiency
                        
                        # OPTIMIZED: Use 1 sample for speed (was 2)
                        max_samples_for_fisher = min(1, len(obs_flat))
                        indices = torch.randperm(len(obs_flat), device=obs_flat.device)[:max_samples_for_fisher]
                        obs_subset = obs_flat[indices]
                        
                        N = len(obs_subset)
                        result = torch.zeros_like(v)
                        
                        # Process samples one at a time to minimize peak memory
                        for i in range(N):
                            obs_i = obs_subset[i:i+1].detach().requires_grad_(True)  # [1, obs_dim]
                            
                            # Get representation for this sample
                            z_i = self.get_representation(obs_i)  # [1, repr_dim]
                            if not z_i.requires_grad:
                                z_i = self.critic(obs_i)
                                if z_i.dim() > 1:
                                    z_i = z_i.squeeze(-1) if z_i.shape[-1] == 1 else z_i
                            
                            z_i_flat = z_i.squeeze(0)  # [repr_dim]
                            
                            # Compute J_i @ v: [repr_dim]
                            # For each dimension: (J_i @ v)[j] = ∂z_i[j]/∂θ @ v
                            # This is the theoretically correct computation
                            # OPTIMIZED: Process dimensions in batches to prevent graph accumulation
                            # Recompute z_i for each batch to get fresh graph (prevents memory leak)
                            # Network state is the same, so numerical consistency is maintained
                            batch_size = 64  # Increased from 32 to reduce iterations (was optimized for memory, now optimizing for speed)
                            repr_dim = z_i_flat.shape[0]
                            Jv_i = torch.zeros(repr_dim, device=v.device, requires_grad=False)
                            
                            for batch_start in range(0, repr_dim, batch_size):
                                batch_end = min(batch_start + batch_size, repr_dim)
                                
                                # Recompute z_i for this batch to get fresh computation graph
                                # This prevents graph accumulation while maintaining numerical consistency
                                # (network parameters haven't changed, so z_i will be the same)
                                obs_i_batch = obs_subset[i:i+1].detach().requires_grad_(True)
                                z_i_batch = self.get_representation(obs_i_batch)
                                if not z_i_batch.requires_grad:
                                    z_i_batch = self.critic(obs_i_batch)
                                    if z_i_batch.dim() > 1:
                                        z_i_batch = z_i_batch.squeeze(-1) if z_i_batch.shape[-1] == 1 else z_i_batch
                                z_i_batch_flat = z_i_batch.squeeze(0)
                                
                                # Process dimensions in this batch
                                for j in range(batch_start, batch_end):
                                    z_ij = z_i_batch_flat[j - batch_start]
                                    # Only retain graph within this batch (not across batches)
                                    retain = (j < batch_end - 1)
                                    grad_ij_list = torch.autograd.grad(
                                        z_ij,
                                        repr_params,
                                        create_graph=True,
                                        retain_graph=retain,  # Only retain within batch
                                        allow_unused=True,
                                    )
                                    grad_ij_flat = torch.cat([g.view(-1) for g in grad_ij_list if g is not None])
                                    if len(grad_ij_flat) == len(v):
                                        Jv_i[j] = (grad_ij_flat * v).sum()
                                    
                                    # Clear gradient tensors immediately
                                    del grad_ij_list, grad_ij_flat
                                
                                # Free batch computation graph
                                del z_i_batch, z_i_batch_flat, obs_i_batch
                                # OPTIMIZED: Only cleanup periodically to reduce overhead
                                # Removed per-batch gc/sync - too expensive
                            
                            # Compute J_i^T @ (J_i @ v) = grad of (z_i^T @ Jv_i) w.r.t. params
                            # Need to recompute z_i with fresh computation graph for this
                            obs_i_fresh = obs_subset[i:i+1].detach().requires_grad_(True)
                            z_i_fresh = self.get_representation(obs_i_fresh)
                            if not z_i_fresh.requires_grad:
                                z_i_fresh = self.critic(obs_i_fresh)
                                if z_i_fresh.dim() > 1:
                                    z_i_fresh = z_i_fresh.squeeze(-1) if z_i_fresh.shape[-1] == 1 else z_i_fresh
                            z_i_fresh_flat = z_i_fresh.squeeze(0)
                            
                            z_dot_Jv = (z_i_fresh_flat * Jv_i.detach()).sum()
                            grad_result_list = torch.autograd.grad(
                                z_dot_Jv,
                                repr_params,
                                create_graph=False,
                                retain_graph=False,  # Don't retain - we're done with this sample
                                allow_unused=True,
                            )
                            grad_result_flat = torch.cat([g.view(-1) for g in grad_result_list if g is not None])
                            if len(grad_result_flat) == len(v):
                                result += grad_result_flat
                            
                            # Clear all intermediate tensors and computation graphs
                            # Note: We don't need to explicitly detach since we're not retaining graphs
                            del z_i, z_i_flat, z_i_fresh, z_i_fresh_flat, Jv_i, z_dot_Jv, grad_result_list, grad_result_flat, obs_i_fresh
                            
                            # OPTIMIZED: Only cleanup periodically to reduce overhead
                            # Removed per-sample gc/sync - too expensive
                        
                        result = result / N + self.damping * v
                        
                        # Check for NaN/Inf in result - fail loudly if detected
                        if torch.isnan(result).any() or torch.isinf(result).any():
                            raise ValueError(f"Fisher-vector product produced NaN/Inf. N={N}, result stats: mean={result.mean().item() if not torch.isnan(result).all() else 'NaN'}, std={result.std().item() if not torch.isnan(result).all() else 'NaN'}, max={result.abs().max().item() if not torch.isnan(result).all() else 'NaN'}")
                        
                        return result
                    
                    # Natural gradient using conjugate gradient: F_Z^{-1} * g
                    stepdir = self.conjugate_gradient(fisher_vector_product, -combined_grad, self.cg_iters)
                    
                    # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                    # Removed expensive gc/sync calls
                    
                    # Scale step to satisfy trust region constraint: (step^T F_Z step) <= delta_z
                    # Compute step size: lambda = sqrt(delta_z / (stepdir^T F_Z stepdir))
                    shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum()
                    
                    # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                    # Removed expensive gc/sync calls
                    if shs > 0:
                        lm = torch.sqrt(shs / self.delta_z)
                        fullstep = stepdir / lm
                    else:
                        # If shs is too small, use unscaled step (shouldn't happen)
                        fullstep = stepdir
                    
                    # Get old parameters
                    old_params = self.flat_params(self.repr_net)
                    
                    # Check for NaN/Inf in old parameters (should never happen, but defensive check)
                    if torch.isnan(old_params).any() or torch.isinf(old_params).any():
                        raise ValueError(f"Old parameters already contain NaN/Inf in unified mode! This indicates parameter corruption before trust region update. old_params stats: mean={old_params.mean().item() if not torch.isnan(old_params).all() else 'NaN'}, std={old_params.std().item() if not torch.isnan(old_params).all() else 'NaN'}, max={old_params.abs().max().item() if not torch.isnan(old_params).all() else 'NaN'}")
                    
                    # Apply constrained update to representation network
                    new_params = old_params + fullstep
                    
                    # Check for NaN/Inf in new parameters before setting
                    if torch.isnan(new_params).any() or torch.isinf(new_params).any():
                        raise ValueError(f"New parameters contain NaN/Inf in unified mode. old_params stats: mean={old_params.mean().item():.6f}, std={old_params.std().item():.6f}, fullstep stats: mean={fullstep.mean().item():.6f}, std={fullstep.std().item():.6f}")
                    
                    self.set_flat_params(self.repr_net, new_params)
                    
                    # Verify parameters were set correctly (defensive check)
                    verify_params = self.flat_params(self.repr_net)
                    if torch.isnan(verify_params).any() or torch.isinf(verify_params).any():
                        raise ValueError(f"Parameters contain NaN/Inf after set_flat_params in unified mode. This indicates a bug in set_flat_params or parameter corruption.")
                    
                    # Now update other components (critic, policy, forward_model) with standard gradient descent
                    # Zero gradients and recompute for other components only
                    self.unified_optimizer.zero_grad()
                    
                    # Recompute forward pass with updated representation
                    z_t_new = self.repr_net(obs_flat)
                    values_new = self.critic(z_t_new).squeeze(-1)
                    # Use Huber loss for robustness to outliers
                    value_loss_new = nn.functional.huber_loss(values_new, returns, reduction='mean', delta=self.huber_delta)
                    
                    log_probs_new, entropy_new = self.policy.evaluate_actions(z_t_new, actions)
                    policy_loss_new = -(log_probs_new * advantages).mean()
                    
                    # Contrastive loss with updated representation
                    contrastive_loss_new = torch.tensor(0.0, device=obs.device)
                    if self.use_contrastive_loss and z_next is not None:
                        z_pred_new = self.forward_model(z_t_new, actions)
                        contrastive_loss_new = nn.functional.mse_loss(z_pred_new, z_next)
                    
                    entropy_loss_new = -entropy_new.mean()
                    
                    # Loss for other components (critic, policy, forward_model)
                    other_loss = (
                        self.value_coef * value_loss_new +
                        policy_loss_new +
                        self.contrastive_coef * contrastive_loss_new +
                        self.entropy_coef * entropy_loss_new
                    )
                    other_loss.backward()
                    
                    # Zero gradients for repr_net (already updated via trust region)
                    if self.repr_net is not None:
                        for param in self.repr_net.parameters():
                            if param.grad is not None:
                                param.grad.zero_()
                    
                    # Clip gradients for other components
                    torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()), self.max_grad_norm)
                    if self.forward_model is not None:
                        torch.nn.utils.clip_grad_norm_(list(self.forward_model.parameters()), self.max_grad_norm)
                    
                    # Update other components (critic, policy, forward_model)
                    self.unified_optimizer.step()
                    
                    # Compute actual representation distance after update
                    with torch.no_grad():
                        z_new = self.repr_net(obs_flat)
                        repr_dist = torch.mean((z_new - z_old) ** 2).item()
                    
                except Exception as e:
                    # Fallback to standard gradient descent if trust region computation fails
                    raise ValueError(f"Trust region computation failed in unified mode:\n{e}")

            else:
                # No representation network: standard gradient descent
                self.unified_optimizer.zero_grad()
                total_loss.backward()
                
                all_params = list(self.policy.parameters()) + list(self.critic.parameters())
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                if self.forward_model is not None:
                    torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), self.max_grad_norm)
                
                self.unified_optimizer.step()
                
                with torch.no_grad():
                    if self.repr_net is not None:
                        z_new = self.repr_net(obs_flat)
                        repr_dist = torch.mean((z_new - z_old) ** 2).item()
                    else:
                        repr_dist = 0.0
            
            # Compute gradient magnitudes per component (for diagnostics)
            with torch.no_grad():
                value_grad_norm = 0.0
                policy_grad_norm = 0.0
                repr_grad_norm = 0.0
                contrastive_grad_norm = 0.0
                
                # Value gradients (critic parameters)
                for param in self.critic.parameters():
                    if param.grad is not None:
                        value_grad_norm += param.grad.data.norm(2).item() ** 2
                value_grad_norm = value_grad_norm ** 0.5
                
                # Policy gradients
                for param in self.policy.parameters():
                    if param.grad is not None:
                        policy_grad_norm += param.grad.data.norm(2).item() ** 2
                policy_grad_norm = policy_grad_norm ** 0.5
                
                # Representation gradients (shared CNN)
                if self.repr_net is not None:
                    for param in self.repr_net.parameters():
                        if param.grad is not None:
                            repr_grad_norm += param.grad.data.norm(2).item() ** 2
                    repr_grad_norm = repr_grad_norm ** 0.5
                
                # Contrastive gradients (forward model)
                if self.forward_model is not None:
                    for param in self.forward_model.parameters():
                        if param.grad is not None:
                            contrastive_grad_norm += param.grad.data.norm(2).item() ** 2
                    contrastive_grad_norm = contrastive_grad_norm ** 0.5
            
            # Soft update target network (if enabled)
            if self.use_target_network and self.repr_net_target is not None:
                # Update target network parameters: θ_target = τ * θ + (1 - τ) * θ_target
                # This slowly moves target network towards current network
                with torch.no_grad():
                    for target_param, param in zip(self.repr_net_target.parameters(), self.repr_net.parameters()):
                        target_param.data.mul_(1 - self.target_update_tau).add_(
                            param.data, alpha=self.target_update_tau
                        )
            
            # Return stats
            stats = {
                "policy_loss": policy_loss.item(),
                "value_loss": value_loss.item(),  # Already normalized by returns variance
                "value_loss_raw": (value_loss * self._current_returns_variance).item() if hasattr(self, '_current_returns_variance') else value_loss.item(),  # Raw MSE for reference
                "entropy": entropy.mean().item(),
                "representation_distance": repr_dist,
                "delta_z": self.delta_z,
                "phase": "unified",
                "contrastive_loss": contrastive_loss_val,
                "value_grad_norm": value_grad_norm,
                "policy_grad_norm": policy_grad_norm,
                "repr_grad_norm": repr_grad_norm,
                "contrastive_grad_norm": contrastive_grad_norm,
            }
            
            # Add representation loss stats if available (for unified mode, we'd need to compute it separately)
            # For now, unified mode doesn't compute representation loss (can be added if needed)
            
            # Increment step counter for Hessian computation frequency
            self._step += 1
            
            return stats
        
        # Phase 1: Update representation network (if applicable) - PHASIC MODE
        if phase in ["representation", "all"] and self.repr_net is not None:
            # For representation network, we train it with:
            # 1. Value prediction error (encourages representation useful for value estimation)
            # 2. Contrastive loss (temporal contrastive learning for causal representation)
            
            # Freeze critic parameters during representation training
            for param in self.critic.parameters():
                param.requires_grad = False
            
            # Encode observations: s -> z
            z_t = self.repr_net(obs_flat)  # [N, repr_dim] - current representation
            
            # 1. Value prediction loss
            values = self.critic(z_t).squeeze(-1)  # z -> v
            # Use Huber loss for robustness to outliers (more stable than MSE)
            value_loss_raw = nn.functional.huber_loss(values, returns, reduction='mean', delta=self.huber_delta)
            
            # Normalize value loss by variance of returns to make it scale-invariant
            # This ensures value loss is on a similar scale to other losses (O(1) instead of O(1000))
            # Principled approach: Huber loss / Var(returns) = relative error, which is scale-invariant
            value_loss = value_loss_raw / self._current_returns_variance
            
            # 2. Contrastive loss (temporal contrastive learning)
            contrastive_loss = torch.tensor(0.0, device=obs.device)
            contrastive_loss_val = 0.0  # Initialize for phasic mode
            if self.use_contrastive_loss and next_obs is not None:
                # Lazy initialization of forward model if action_dim wasn't known at init
                if self.forward_model is None:
                    # Infer action_dim from actions tensor
                    if actions.dtype == torch.long:
                        action_dim = int(actions.max().item() + 1)  # Number of action classes
                    else:
                        action_dim = actions.shape[-1] if actions.dim() > 1 else 7  # Fallback
                    
                    forward_hidden = self.config.get("forward_model_hidden", [256, 256]) if hasattr(self, 'config') else [256, 256]
                    forward_activation = self.config.get("forward_model_activation", "relu") if hasattr(self, 'config') else "relu"
                    self.forward_model = ForwardModel(
                        repr_dim=self.repr_net.repr_dim,
                        action_dim=action_dim,
                        hidden_sizes=forward_hidden,
                        activation=forward_activation,
                    ).to(self.device)
                    self.forward_optimizer = optim.Adam(self.forward_model.parameters(), lr=self.lr)
                
                # Temporal contrastive loss: predict Z_{t+1} from Z_t and action
                # Flatten next observations if needed
                if next_obs.dim() == 4:  # [N, H, W, C]
                    N, H, W, C = next_obs.shape
                    next_obs_flat = next_obs.view(N, H * W * C)
                else:
                    next_obs_flat = next_obs
                
                # Ensure next_obs_flat is on the correct device
                next_obs_flat = next_obs_flat.to(self.device)
                # Use target network for stable targets (if enabled), otherwise use current network
                if self.use_target_network and self.repr_net_target is not None:
                    with torch.no_grad():
                        z_next = self.repr_net_target(next_obs_flat)  # Stable target from target network
                else:
                    z_next = self.repr_net(next_obs_flat)  # Next representation [N, repr_dim]
                
                # Forward model: predict next representation from current + action
                z_pred = self.forward_model(z_t, actions)  # [N, repr_dim]
                
                # L2 contrastive loss: || Z_pred - Z_next ||^2
                contrastive_loss_raw = nn.functional.mse_loss(z_pred, z_next)
                contrastive_loss_val = contrastive_loss_raw.item()
                
                # Apply EMA smoothing to reduce choppiness
                if self.contrastive_loss_ema is None:
                    self.contrastive_loss_ema = contrastive_loss_val
                else:
                    self.contrastive_loss_ema = (
                        self.contrastive_ema_alpha * self.contrastive_loss_ema + 
                        (1 - self.contrastive_ema_alpha) * contrastive_loss_val
                    )
                
                contrastive_loss = contrastive_loss_raw
            
            # Diversity regularization: penalize low variance in representations
            # This encourages the representation to differentiate between different states
            diversity_loss = torch.tensor(0.0, device=obs.device)
            if self.diversity_coef > 0 and self.repr_net is not None:
                # Compute variance across batch: encourage high variance to prevent collapse
                # Use negative variance as loss (we want to maximize variance)
                z_batch = z_t  # Current batch representations [N, repr_dim]
                z_var = z_batch.var(dim=0).mean()  # Mean variance across representation dimensions
                # Negative variance loss: we want to maximize variance, so minimize negative variance
                diversity_loss = -z_var  # Negative because we want to maximize variance
            
            # Total loss for representation network
            # Scale value_loss to balance with contrastive loss and diversity
            repr_loss = self.value_coef * value_loss + self.contrastive_coef * contrastive_loss + self.diversity_coef * diversity_loss
            
            # Check for NaN/Inf in loss components before proceeding
            if torch.isnan(repr_loss) or torch.isinf(repr_loss):
                raise ValueError(
                    f"repr_loss is NaN/Inf: repr_loss={repr_loss.item()}, "
                    f"value_loss={value_loss.item()}, contrastive_loss={contrastive_loss.item()}, "
                    f"diversity_loss={diversity_loss.item()}, "
                    f"value_coef={self.value_coef}, contrastive_coef={self.contrastive_coef}, diversity_coef={self.diversity_coef}"
                )
            if torch.isnan(value_loss) or torch.isinf(value_loss):
                raise ValueError(
                    f"value_loss is NaN/Inf: value_loss={value_loss.item()}, "
                    f"values stats: mean={values.mean().item() if not torch.isnan(values).all() else 'NaN'}, "
                    f"std={values.std().item() if not torch.isnan(values).all() else 'NaN'}, "
                    f"returns stats: mean={returns.mean().item() if not torch.isnan(returns).all() else 'NaN'}, "
                    f"std={returns.std().item() if not torch.isnan(returns).all() else 'NaN'}"
                )
            if torch.isnan(contrastive_loss) or torch.isinf(contrastive_loss):
                raise ValueError(
                    f"contrastive_loss is NaN/Inf: contrastive_loss={contrastive_loss.item()}"
                )
            
            # REPRESENTATION-SPACE TRUST REGION CONSTRAINT ENFORCEMENT
            # Compute representation BEFORE update (for monitoring distance)
            with torch.no_grad():
                z_old = self.repr_net(obs_flat)
            
            # Parameters that affect representation: repr_net
            repr_params = list(self.repr_net.parameters())
            
            try:
                # Don't compute F_Z explicitly - we compute Fisher-vector products implicitly
                # This saves memory (F_Z would be [num_params, num_params] which is huge)
                
                # Get flattened gradient for representation parameters
                # Note: flat_grad uses torch.autograd.grad() which doesn't require .backward() first
                # CRITICAL: Use create_graph=False to prevent memory accumulation
                # We don't need second-order derivatives for the gradient itself, only for Fisher-vector products
                self.unified_optimizer.zero_grad()
                
                # Save repr_loss value for error messages before freeing computation graph
                repr_loss_value = repr_loss.item()
                
                # Save contrastive_loss value for forward model update (will recompute later with updated params)
                contrastive_loss_value = contrastive_loss.item() if hasattr(contrastive_loss, 'item') else 0.0
                
                repr_grad = self.flat_grad(repr_loss, repr_params, create_graph=False)
                
                # Free the computation graph from repr_loss by explicitly deleting references
                # This prevents memory accumulation over epochs
                # NOTE: contrastive_loss is part of repr_loss graph, so we'll recompute it later
                del repr_loss, contrastive_loss
                # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                
                # Check for NaN/Inf in gradient before proceeding
                if torch.isnan(repr_grad).any() or torch.isinf(repr_grad).any():
                    raise ValueError(
                        f"repr_grad contains NaN/Inf. repr_loss={repr_loss_value}, "
                        f"repr_grad stats: mean={repr_grad.mean().item() if not torch.isnan(repr_grad).all() else 'NaN'}, "
                        f"std={repr_grad.std().item() if not torch.isnan(repr_grad).all() else 'NaN'}, "
                        f"max={repr_grad.abs().max().item() if not torch.isnan(repr_grad).all() else 'NaN'}"
                    )
                
                # Clip gradient to prevent extreme values that cause numerical instability
                grad_norm = torch.norm(repr_grad)
                max_grad_norm = 100.0  # Limit gradient norm to prevent extreme steps
                if grad_norm > max_grad_norm:
                    repr_grad = repr_grad * (max_grad_norm / grad_norm)
                    # print(f"Warning: Clipped repr_grad norm from {grad_norm.item():.2e} to {max_grad_norm:.2e}")
                
                # Fisher-vector product function for conjugate gradient
                # Use subset of samples to reduce memory requirements
                def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
                    # Use subset of samples for Fisher computation to reduce memory and time
                    # CRITICAL: Use only 1 sample to minimize memory usage and prevent OOM
                    max_samples_for_fisher = min(1, len(obs_flat))  # Use only 1 sample to prevent OOM
                    indices = torch.randperm(len(obs_flat), device=obs_flat.device)[:max_samples_for_fisher]
                    obs_subset = obs_flat[indices].detach()  # Detach to prevent graph accumulation
                    
                    N = len(obs_subset)
                    result = torch.zeros_like(v)
                    
                    # Process samples one at a time to minimize peak memory
                    for i in range(N):
                        obs_i = obs_subset[i:i+1].detach().requires_grad_(True)  # [1, obs_dim]
                        
                        # Compute J_i @ v: [repr_dim] using autograd
                        # OPTIMIZED: Process dimensions in batches to prevent graph accumulation
                        # Recompute z_i for each batch to get fresh graph (prevents memory leak)
                        # Network state is the same, so numerical consistency is maintained
                        # First, get repr_dim by doing a single forward pass (then delete it)
                        z_i_temp = self.get_representation(obs_i)
                        if not z_i_temp.requires_grad:
                            z_i_temp = self.critic(obs_i)
                            if z_i_temp.dim() > 1:
                                z_i_temp = z_i_temp.squeeze(-1) if z_i_temp.shape[-1] == 1 else z_i_temp
                        repr_dim = z_i_temp.squeeze(0).shape[0]
                        del z_i_temp  # Delete immediately - we only needed the dimension
                        
                        batch_size = 32  # Increased from 8 to reduce iterations (optimizing for speed)
                        Jv_i = torch.zeros(repr_dim, device=v.device, requires_grad=False)
                        
                        for batch_start in range(0, repr_dim, batch_size):
                            batch_end = min(batch_start + batch_size, repr_dim)
                            
                            # Recompute z_i for this batch to get fresh computation graph
                            # This prevents graph accumulation while maintaining numerical consistency
                            # (network parameters haven't changed, so z_i will be the same)
                            obs_i_batch = obs_subset[i:i+1].detach().requires_grad_(True)
                            z_i_batch = self.get_representation(obs_i_batch)
                            if not z_i_batch.requires_grad:
                                z_i_batch = self.critic(obs_i_batch)
                                if z_i_batch.dim() > 1:
                                    z_i_batch = z_i_batch.squeeze(-1) if z_i_batch.shape[-1] == 1 else z_i_batch
                            z_i_batch_flat = z_i_batch.squeeze(0)
                            
                            # Process dimensions in this batch
                            for j in range(batch_start, batch_end):
                                z_ij = z_i_batch_flat[j - batch_start]
                                # Only retain graph within this batch (not across batches)
                                retain = (j < batch_end - 1)
                                grad_ij_list = torch.autograd.grad(
                                    z_ij,
                                    repr_params,
                                    create_graph=True,
                                    retain_graph=retain,  # Only retain within batch
                                    allow_unused=True,
                                )
                                grad_ij_flat = torch.cat([g.view(-1) for g in grad_ij_list if g is not None])
                                if len(grad_ij_flat) == len(v):
                                    # Compute Jv_i[j] and immediately detach to free computation graph
                                    Jv_i[j] = (grad_ij_flat * v).sum().detach()
                                
                                # Clear gradient tensors immediately - this should free the computation graph
                                del grad_ij_list, grad_ij_flat
                                
                                # OPTIMIZED: Removed per-dimension cleanup - too expensive
                            
                            # Free batch computation graph
                            del z_i_batch, z_i_batch_flat, obs_i_batch
                            # OPTIMIZED: Removed per-batch cleanup - too expensive
                        
                        # Compute J_i^T @ (J_i @ v) = grad of (z_i^T @ Jv_i) w.r.t. params
                        # Need to recompute z_i with fresh computation graph for this
                        obs_i_fresh = obs_subset[i:i+1].detach().requires_grad_(True)
                        z_i_fresh = self.get_representation(obs_i_fresh)
                        if not z_i_fresh.requires_grad:
                            z_i_fresh = self.critic(obs_i_fresh)
                            if z_i_fresh.dim() > 1:
                                z_i_fresh = z_i_fresh.squeeze(-1) if z_i_fresh.shape[-1] == 1 else z_i_fresh
                        z_i_fresh_flat = z_i_fresh.squeeze(0)
                        
                        z_dot_Jv = (z_i_fresh_flat * Jv_i.detach()).sum()
                        grad_result_list = torch.autograd.grad(
                            z_dot_Jv,
                            repr_params,
                            create_graph=False,
                            retain_graph=False,  # Don't retain - we're done with this sample
                            allow_unused=True,
                        )
                        grad_result_flat = torch.cat([g.view(-1) for g in grad_result_list if g is not None])
                        if len(grad_result_flat) == len(v):
                            result += grad_result_flat
                        
                        # Clear all intermediate tensors and computation graphs immediately
                        del z_i_fresh, z_i_fresh_flat, Jv_i, z_dot_Jv, grad_result_list, grad_result_flat, obs_i_fresh, obs_i
                        
                        # OPTIMIZED: Removed per-sample cleanup - too expensive
                    
                    result = result / N + self.damping * v
                    
                    # Final cleanup before returning
                    del obs_subset, indices
                    # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                    # Removed expensive gc/sync calls
                    
                    # Check for NaN/Inf in result - fail loudly if detected
                    if torch.isnan(result).any() or torch.isinf(result).any():
                        raise ValueError(f"Fisher-vector product produced NaN/Inf. N={N}, result stats: mean={result.mean().item() if not torch.isnan(result).all() else 'NaN'}, std={result.std().item() if not torch.isnan(result).all() else 'NaN'}, max={result.abs().max().item() if not torch.isnan(result).all() else 'NaN'}")
                    
                    return result
                
                # Natural gradient using conjugate gradient: F_Z^{-1} * g
                stepdir = self.conjugate_gradient(fisher_vector_product, -repr_grad, self.cg_iters)
                
                # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                # Removed expensive gc/sync calls
                
                # Check for NaN/Inf in stepdir before proceeding
                if torch.isnan(stepdir).any() or torch.isinf(stepdir).any():
                    raise ValueError(f"Conjugate gradient produced NaN/Inf stepdir. repr_grad stats: mean={repr_grad.mean().item():.6f}, std={repr_grad.std().item():.6f}, max={repr_grad.abs().max().item():.6f}")
                
                # Scale step to satisfy trust region constraint: (step^T F_Z step) <= delta_z
                # Note: fisher_vector_product creates computation graphs, so cleanup after this call
                shs = 0.5 * (stepdir * fisher_vector_product(stepdir)).sum()
                
                # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                # Removed expensive gc/sync calls
                
                # Check for NaN/Inf in shs
                if torch.isnan(shs) or torch.isinf(shs):
                    raise ValueError(f"Fisher-vector product produced NaN/Inf shs: {shs.item()}")
                
                # Initialize lm to None - will be set if shs > 0
                lm = None
                if shs > 0:
                    lm = torch.sqrt(shs / self.delta_z)
                    if torch.isnan(lm) or torch.isinf(lm):
                        raise ValueError(f"Step scaling produced NaN/Inf lm: {lm.item()}, shs={shs.item()}, delta_z={self.delta_z}")
                    fullstep = stepdir / lm
                else:
                    # If shs <= 0, use unscaled step (but check for validity)
                    fullstep = stepdir
                
                # Check for NaN/Inf in fullstep before applying
                if torch.isnan(fullstep).any() or torch.isinf(fullstep).any():
                    raise ValueError(f"Fullstep contains NaN/Inf. stepdir stats: mean={stepdir.mean().item():.6f}, std={stepdir.std().item():.6f}, max={stepdir.abs().max().item():.6f}, shs={shs.item()}")
                
                # Limit step size to prevent extreme parameter values
                step_norm = torch.norm(fullstep)
                # This is a safety check for numerical stability
                # The trust region constraint (delta_z) already limits representation change
                if step_norm > self.max_step_norm:
                    fullstep = fullstep * (self.max_step_norm / step_norm)
                    lm_str = f"{lm.item():.2e}" if lm is not None else "N/A (shs<=0)"
                    # print(f"Warning: Clipped fullstep norm from {step_norm.item():.2e} to {self.max_step_norm:.2e} (delta_z={self.delta_z}, shs={shs.item():.2e}, lm={lm_str})")
                # Get old parameters
                old_params = self.flat_params(self.repr_net)
                
                # Check for NaN/Inf in old parameters (should never happen, but defensive check)
                if torch.isnan(old_params).any() or torch.isinf(old_params).any():
                    raise ValueError(f"Old parameters already contain NaN/Inf! This indicates parameter corruption before trust region update. old_params stats: mean={old_params.mean().item() if not torch.isnan(old_params).all() else 'NaN'}, std={old_params.std().item() if not torch.isnan(old_params).all() else 'NaN'}, max={old_params.abs().max().item() if not torch.isnan(old_params).all() else 'NaN'}")
                
                # Apply constrained update to representation network
                new_params = old_params + fullstep
                
                # Check for NaN/Inf in new parameters before setting
                if torch.isnan(new_params).any() or torch.isinf(new_params).any():
                    raise ValueError(f"New parameters contain NaN/Inf. old_params stats: mean={old_params.mean().item():.6f}, std={old_params.std().item():.6f}, fullstep stats: mean={fullstep.mean().item():.6f}, std={fullstep.std().item():.6f}")
                
                self.set_flat_params(self.repr_net, new_params)
                
                # Verify parameters were set correctly (defensive check)
                verify_params = self.flat_params(self.repr_net)
                if torch.isnan(verify_params).any() or torch.isinf(verify_params).any():
                    raise ValueError(f"Parameters contain NaN/Inf after set_flat_params. This indicates a bug in set_flat_params or parameter corruption.")
                
                # Update forward model with standard gradient descent (not constrained)
                # CRITICAL: Recompute contrastive_loss with updated representation network parameters
                # The old contrastive_loss was part of repr_loss graph which we deleted
                if self.forward_model is not None and self.use_contrastive_loss and next_obs is not None:
                    self.forward_optimizer.zero_grad()
                    
                    # Recompute contrastive loss with updated representation network
                    with torch.no_grad():
                        z_t_new = self.repr_net(obs_flat)
                        z_next_new = self.repr_net(next_obs_flat)
                    
                    # Forward pass through forward model (requires gradients)
                    z_pred_new = self.forward_model(z_t_new, actions)
                    contrastive_loss_new = nn.functional.mse_loss(z_pred_new, z_next_new)
                    
                    if contrastive_loss_new.item() > 0:
                        contrastive_loss_new.backward()
                        torch.nn.utils.clip_grad_norm_(self.forward_model.parameters(), self.max_grad_norm)
                        self.forward_optimizer.step()
                    
                    # Clean up
                    del z_t_new, z_next_new, z_pred_new, contrastive_loss_new
                    # OPTIMIZED: Minimal cleanup - let PyTorch handle memory automatically
                
                # Final verification: check all network parameters for NaN/Inf after all updates
                final_repr_params = self.flat_params(self.repr_net)
                if torch.isnan(final_repr_params).any() or torch.isinf(final_repr_params).any():
                    raise ValueError(f"repr_net parameters contain NaN/Inf after all updates! This indicates parameter corruption during update.")
                
                # Compute actual representation distance after update
                with torch.no_grad():
                    # Check parameters one more time before forward pass
                    final_check_params = self.flat_params(self.repr_net)
                    if torch.isnan(final_check_params).any() or torch.isinf(final_check_params).any():
                        raise ValueError(
                            f"repr_net parameters contain NaN/Inf right before forward pass! "
                            f"old_params stats: mean={old_params.mean().item():.6f}, std={old_params.std().item():.6f}, "
                            f"fullstep stats: mean={fullstep.mean().item():.6f}, std={fullstep.std().item():.6f}, "
                            f"new_params stats: mean={new_params.mean().item():.6f}, std={new_params.std().item():.6f}"
                        )
                    
                    # Check for extreme parameter values that might cause numerical instability
                    param_max = final_check_params.abs().max().item()
                    if param_max > 1e6:
                        raise ValueError(
                            f"repr_net parameters have extreme values (max={param_max:.2e}) that may cause numerical instability! "
                            f"This suggests the trust region step was too large or the gradient was too large."
                        )
                    
                    z_new = self.repr_net(obs_flat)
                    # Check if z_new contains NaN (indicates corrupted network)
                    if torch.isnan(z_new).any() or torch.isinf(z_new).any():
                        raise ValueError(
                            f"Representation z_new contains NaN/Inf after update! "
                            f"Parameters are valid (mean={final_check_params.mean().item():.6f}, std={final_check_params.std().item():.6f}, max={param_max:.2e}), "
                            f"but forward pass produces NaN. This indicates numerical instability in the network computation. "
                            f"Consider: 1) Reducing delta_z (current: {self.delta_z}), 2) Adding gradient clipping, 3) Checking for division by zero or log(0) in network."
                        )
                    repr_dist = torch.mean((z_new - z_old) ** 2).item()
                    
            except Exception as e:
                # Fail loudly if trust region computation fails - no fallback
                raise ValueError(f"Trust region computation failed in phasic mode:\n{e}")
            
            # Soft update target network (if enabled)
            if self.use_target_network and self.repr_net_target is not None:
                # Update target network parameters: θ_target = τ * θ + (1 - τ) * θ_target
                with torch.no_grad():
                    for target_param, param in zip(self.repr_net_target.parameters(), self.repr_net.parameters()):
                        target_param.data.mul_(1 - self.target_update_tau).add_(
                            param.data, alpha=self.target_update_tau
                        )
            
            # Re-enable critic gradients for next phase
            for param in self.critic.parameters():
                param.requires_grad = True
        
        # Phase 2: Update critic
        if phase in ["critic", "all"]:
            # Standard Actor-Critic: Compute value loss, backprop (gradients accumulate in CNN)
            vae_info = None
            if self.repr_net is not None:
                # Use cached representation if available, otherwise compute it
                if z_cached is not None:
                    z = z_cached  # Reuse cached representation
                else:
                    z = self.repr_net(obs_flat)  # s -> z (gradients enabled, no detaching)
                values = self.critic(z).squeeze(-1)  # z -> v
            elif hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
                # VAE critic: get both value and VAE losses (reconstruction + KL)
                values, vae_info = self.critic(obs_flat, return_latent=True)
                values = values.squeeze(-1)
            else:
                values = self.critic(obs_flat).squeeze(-1)  # s -> v
            
            # Use Huber loss for robustness to outliers (more stable than MSE)
            value_loss_raw = nn.functional.huber_loss(values, returns, reduction='mean', delta=self.huber_delta)
            # Normalize value loss by variance of returns to make it scale-invariant
            value_loss = value_loss_raw / self._current_returns_variance
            
            # Add VAE loss for VAE critics (reconstruction + KL)
            # This ensures encoder learns good representations
            if vae_info is not None:
                vae_loss = vae_info["vae_loss"]  # recon_loss + beta * kl_loss
            else:
                vae_loss = torch.tensor(0.0, device=self.device)
            
            # Representation loss: L_rep = α * (1/μ) * ||∇_Z V(Z)||²
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
                        states=obs_flat,
                        alpha=effective_repr_coef,
                        use_convexity_weighting=self.use_convexity_weighting,
                        hessian_compute_freq=self.hessian_compute_freq,
                        step=self._step,
                    )
            
            # Total critic loss: value loss + VAE loss + representation loss
            # Representation loss shrinks representation error via V-gradients
            critic_loss = value_loss + self.vae_coef * vae_loss + representation_loss
            
            # Use separate critic optimizer with higher learning rate (critic not constrained by trust region)
            # Note: Convexity constraints for ICNN are automatically enforced during forward pass
            # via ConvexLinear layers (LazyClippedPositivity clamps weights, ExponentialPositivity uses exp)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()  # Backprop value + VAE loss through critic
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_optimizer.step()
        
        # Phase 3: Update policy
        if phase in ["policy", "all"]:
            # Standard Actor-Critic: Compute policy loss, backprop (gradients accumulate in CNN)
            # Get representation z for policy: Priority: repr_net > VAE encoder > raw obs
            if self.repr_net is not None:
                # If phase="all", we need to recompute z because Phase 2's backward() freed the graph
                # Otherwise, reuse cached representation if available
                if phase == "all" or z_cached is None:
                    z = self.repr_net(obs_flat)  # Recompute to get fresh computation graph
                else:
                    z = z_cached  # Reuse cached representation (only safe when phase != "all")
            elif hasattr(self.critic, 'get_latent_representation') or hasattr(self.critic, 'encode'):
                # VAE critic: encode with gradients enabled so policy loss flows back to encoder
                # Use encode() directly (not get_latent_representation which might detach)
                if hasattr(self.critic, 'encode'):
                    mu, log_std = self.critic.encode(obs_flat)
                    z = mu  # Use mean for deterministic representation
                else:
                    # Fallback to get_latent_representation if encode() not available
                    z = self.critic.get_latent_representation(obs_flat)
            else:
                z = obs_flat
            
            # Get current policy outputs (policy takes z as input)
            log_probs, entropy = self.policy.evaluate_actions(z, actions)
            
            # Policy gradient
            policy_loss = -(log_probs * advantages).mean()
            
            # Compute representation BEFORE update (for monitoring distance)
            with torch.no_grad():
                if self.repr_net is not None:
                    z_old = self.repr_net(obs_flat)
                elif hasattr(self.critic, 'get_latent_representation'):
                    z_old = self.critic.get_latent_representation(obs_flat)
                elif hasattr(self.critic, 'encode'):
                    mu, _ = self.critic.encode(obs_flat)
                    z_old = mu
                else:
                    z_old = obs_flat
            
            # Backprop policy loss (gradients accumulate in unified optimizer)
            # Gradients from value_loss (if computed) are already accumulated
            policy_loss.backward()  # Accumulates with value loss gradients
            
            # Single optimizer step updates all parameters from accumulated gradients
            # Clip gradients for all components
            all_params = list(self.policy.parameters()) + list(self.critic.parameters())
            if self.repr_net is not None:
                all_params = list(self.repr_net.parameters()) + all_params
            torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
            
            # Check for NaN/Inf in gradients before optimizer step
            for name, param in [("policy", self.policy), ("critic", self.critic)]:
                for p in param.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            raise ValueError(f"{name} gradients contain NaN/Inf before optimizer step!")
            if self.repr_net is not None:
                for p in self.repr_net.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            raise ValueError(f"repr_net gradients contain NaN/Inf before optimizer step!")
            
            self.unified_optimizer.step()  # Updates all parameters together
            
            # Check for NaN/Inf in parameters after optimizer step
            if self.repr_net is not None:
                final_repr_params = self.flat_params(self.repr_net)
                if torch.isnan(final_repr_params).any() or torch.isinf(final_repr_params).any():
                    raise ValueError(f"repr_net parameters contain NaN/Inf after policy phase optimizer step!")
            
            # Soft update target network (if enabled)
            if self.use_target_network and self.repr_net_target is not None:
                # Update target network parameters: θ_target = τ * θ + (1 - τ) * θ_target
                with torch.no_grad():
                    for target_param, param in zip(self.repr_net_target.parameters(), self.repr_net.parameters()):
                        target_param.data.mul_(1 - self.target_update_tau).add_(
                            param.data, alpha=self.target_update_tau
                        )
            
            # Compute representation AFTER update (for monitoring distance)
            with torch.no_grad():
                if self.repr_net is not None:
                    z_new = self.repr_net(obs_flat)
                    # Check if z_new contains NaN (indicates corrupted network)
                    if torch.isnan(z_new).any() or torch.isinf(z_new).any():
                        raise ValueError(f"Representation z_new contains NaN/Inf after policy phase! This indicates the repr_net is producing invalid outputs.")
                    repr_dist = torch.mean((z_new - z_old) ** 2).item()
                else:
                    repr_dist = 0.0
            
            policy_loss_val = policy_loss.item()
            entropy_val = entropy.mean().item()
        else:
            # Policy not being trained this phase
            policy_loss_val = 0.0
            entropy_val = 0.0
        
        stats = {
            "policy_loss": policy_loss_val,
            "value_loss": value_loss.item(),  # Already normalized by returns variance
            "value_loss_raw": (value_loss * self._current_returns_variance).item() if hasattr(self, '_current_returns_variance') else value_loss.item(),  # Raw MSE for reference
            "entropy": entropy_val,
            "representation_distance": repr_dist,
            "delta_z": self.delta_z,
            "phase": phase,
            "contrastive_loss": contrastive_loss_val,
        }
        if self.representation_loss_coef > 0:
            stats["representation_loss_coef_effective"] = effective_repr_coef
        
        # Add representation loss stats if available
        if effective_repr_coef > 0 and 'representation_loss_stats' in locals() and representation_loss_stats:
            stats["representation_loss"] = representation_loss_stats.get('representation_loss', 0.0)
            stats["repr_grad_norm"] = representation_loss_stats.get('grad_norm', 0.0)
            stats["repr_mu_estimate"] = representation_loss_stats.get('mu_estimate', 0.0)
        
        # Increment step counter for Hessian computation frequency
        self._step += 1
        
        return stats
    
    def save(self, path: str):
        """Save policy, critic, and representation network weights."""
        save_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "unified_optimizer": self.unified_optimizer.state_dict(),
        }
        if self.critic_optimizer is not None:
            save_dict["critic_optimizer"] = self.critic_optimizer.state_dict()
        if self.repr_net is not None:
            save_dict["repr_net"] = self.repr_net.state_dict()
        if self.repr_net_target is not None:
            save_dict["repr_net_target"] = self.repr_net_target.state_dict()
        if self.forward_model is not None:
            save_dict["forward_model"] = self.forward_model.state_dict()
            if hasattr(self, 'forward_optimizer'):
                save_dict["forward_optimizer"] = self.forward_optimizer.state_dict()
        torch.save(save_dict, path)
    
    def load(self, path: str):
        """Load policy, critic, and representation network weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.unified_optimizer.load_state_dict(checkpoint["unified_optimizer"])
        if self.critic_optimizer is not None and "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])
        if self.repr_net_target is not None and "repr_net_target" in checkpoint:
            self.repr_net_target.load_state_dict(checkpoint["repr_net_target"])
        elif self.repr_net_target is not None and "repr_net" in checkpoint:
            # If target network wasn't saved, initialize it from main network
            self.repr_net_target.load_state_dict(checkpoint["repr_net"])
        if self.forward_model is not None and "forward_model" in checkpoint:
            self.forward_model.load_state_dict(checkpoint["forward_model"])
            if hasattr(self, 'forward_optimizer') and "forward_optimizer" in checkpoint:
                self.forward_optimizer.load_state_dict(checkpoint["forward_optimizer"])

