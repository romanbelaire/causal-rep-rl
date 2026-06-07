"""
Main training loop for bounding chain experiments.
Buffer-based epoch training with periodic metric evaluation.
"""

import argparse
import random
import copy
from pathlib import Path

import numpy as np
import torch

from src.algorithms.ppo import PPO
from src.algorithms.trpo import TRPO
from src.algorithms.representation_trpo import RepresentationTRPO
from src.architectures.critics.feedforward import FeedforwardCritic
from src.architectures.critics.icnn import ICNNCritic
from src.architectures.critics.vae_critic import VAECritic
from src.architectures.policies.mlp_policy import MLPPolicy
from src.architectures.policies.impala import IMPALAPolicy
from src.architectures.policies.cnn_impala import CNNIMPALAPolicy
from src.architectures.representation import RepresentationNetwork
from src.environments.minigrid_wrapper import MinigridWrapper
from src.environments.frozenlake_wrapper import FrozenLakeWrapper
from src.utils.config import Config
from src.utils.logging import CSVLogger
from src.utils.metric_evaluator import MetricEvaluator
from src.metrics.mu_latent_validation import verify_mu_latent_agreement
from src.metrics.value_head_types import is_quadratic_latent_value_head, is_squared_norm_value_head

INTERVENTION_LOG_KEYS = frozenset({
    "train_kappa_directional_loss",
    "train_kappa_concave_mean",
    "train_z_distill_loss",
    "train_z_distill_mse",
    "kappa_directional_loss_coef_effective",
    "z_distill_loss_coef_effective",
    "train_mico_loss",
    "train_mico_u_omega_mean",
    "train_mico_target_mean",
    "train_mico_reward_pair_dispersion",
    "train_embedding_norm_mean",
    "mico_loss_coef_effective",
    "train_dbc_loss",
    "train_dbc_w2_mean",
    "train_dbc_target_mean",
    "train_dbc_reward_pair_dispersion",
    "train_dbc_latent_norm_mean",
    "dbc_loss_coef_effective",
})


def _intervention_metrics_subset(metrics: dict) -> dict:
    return {
        k: metrics[k]
        for k in INTERVENTION_LOG_KEYS
        if k in metrics and isinstance(metrics[k], (int, float))
    }


def _log_cycle_metrics(logger: CSVLogger, step: int, final_metrics: dict) -> None:
    logger.log_metrics(step, final_metrics)
    intervention = _intervention_metrics_subset(final_metrics)
    logger.log_intervention_loss(step, intervention)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_representation_network(obs_dim: int, config: dict, device: str = "cuda", obs_shape: tuple = None):
    """
    Create representation network based on config.
    
    Args:
        obs_dim: Observation dimension (flattened)
        config: Configuration dict
        device: Device to place network on
        obs_shape: Observation shape (H, W, C) for images, None for vectors
    """
    repr_config = config["architecture"].get("representation", {})
    repr_dim = repr_config.get("repr_dim", 512)  # Default representation dimension (512 as mentioned)
    hidden_sizes = repr_config.get("hidden_sizes", [256, 256])
    activation = repr_config.get("activation", "relu")
    use_cnn = repr_config.get("use_cnn", None)  # Auto-detect if None
    
    repr_net = RepresentationNetwork(
        obs_dim, 
        repr_dim, 
        hidden_sizes, 
        activation,
        obs_shape=obs_shape,
        use_cnn=use_cnn,
    )
    return repr_net.to(device)


def create_critic(obs_dim: int, config: dict, device: str = "cuda", repr_net: RepresentationNetwork = None):
    """
    Create critic network based on config.
    
    Args:
        obs_dim: Observation dimension
        config: Configuration dict
        device: Device to place network on
        repr_net: Optional representation network (required for ICNN)
    """
    critic_type = config["architecture"]["critic"]["type"]
    activation = config["architecture"]["critic"]["activation"]
    
    if critic_type == "feedforward":
        hidden_sizes = config["architecture"]["critic"]["hidden_sizes"]
        critic = FeedforwardCritic(obs_dim, hidden_sizes, activation)
    elif critic_type == "icnn":
        # ICNN requires representation network: z -> v
        if repr_net is None:
            raise ValueError("ICNN critic requires a representation network. Create it first with create_representation_network()")
        repr_dim = repr_net.repr_dim
        hidden_sizes = config["architecture"]["critic"]["hidden_sizes"]
        mu = config["architecture"]["critic"].get("mu", 0.0)
        positivity = config["architecture"]["critic"].get("positivity", "exp")
        use_convex_init = config["architecture"]["critic"].get("use_convex_init", True)
        critic = ICNNCritic(repr_dim, hidden_sizes, activation, positivity, mu, use_convex_init)
    elif critic_type == "vae":
        latent_dim = config["architecture"]["critic"].get("latent_dim", 32)
        encoder_hidden = config["architecture"]["critic"].get("encoder_hidden", [256, 256])
        decoder_hidden = config["architecture"]["critic"].get("decoder_hidden", [256, 256])
        value_hidden = config["architecture"]["critic"].get("value_hidden", [128, 128])
        beta = config["architecture"]["critic"].get("beta", 1.0)
        value_head_type = config["architecture"]["critic"].get("value_head_type")
        value_mlp_hidden = config["architecture"]["critic"].get("value_mlp_hidden", 64)
        value_bottleneck_dim = config["architecture"]["critic"].get("value_bottleneck_dim", 8)
        value_mu_min_floor = config["architecture"]["critic"].get("value_mu_min_floor", 0.02)
        value_feature_dim = config["architecture"]["critic"].get("value_feature_dim", 8)
        critic = VAECritic(
            obs_dim,
            latent_dim,
            encoder_hidden,
            decoder_hidden,
            value_hidden,
            activation,
            beta,
            value_head_type=value_head_type,
            value_mlp_hidden=value_mlp_hidden,
            value_bottleneck_dim=value_bottleneck_dim,
            value_mu_min_floor=value_mu_min_floor,
            value_feature_dim=value_feature_dim,
        )
    else:
        raise ValueError(f"Unknown critic type: {critic_type}")
    
    return critic.to(device)


def create_policy(repr_dim: int, action_dim: int, config: dict, device: str = "cuda", obs_shape: tuple = None, use_repr_input: bool = False):
    """
    Create policy network based on config.
    
    Args:
        repr_dim: Input dimension (representation Z dimension, or obs_dim if no encoder)
        action_dim: Action dimension
        config: Configuration dict
        device: Device to place network on
        obs_shape: Observation shape (H, W, C) for images, None for vectors (unused, kept for compatibility)
        use_repr_input: If True, policy takes Z (representation) as input. If False, takes raw observations.
    """
    policy_type = config["architecture"]["policy"]["type"]
    hidden_sizes = config["architecture"]["policy"]["hidden_sizes"]
    activation = config["architecture"]["policy"]["activation"]
    action_space_type = "discrete"  # Minigrid uses discrete actions
    
    if policy_type == "mlp":
        # MLP policy: takes Z (representation) as input
        policy = MLPPolicy(repr_dim, action_dim, hidden_sizes, activation, action_space_type)
    elif policy_type == "impala":
        # IMPALA policy: takes Z (representation) as input
        num_residual = config["architecture"]["policy"].get("num_residual_blocks", 2)
        policy = IMPALAPolicy(repr_dim, action_dim, hidden_sizes, activation, action_space_type, num_residual)
    elif policy_type == "cnn_impala":
        # CNN-IMPALA policy: deprecated - use MLP/IMPALA with shared CNN encoder instead
        # This was for direct image processing, but standard is shared encoder
        raise ValueError("CNN-IMPALA policy type is deprecated. Use 'mlp' or 'impala' with shared representation network instead.")
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
    
    return policy.to(device)


def collect_rollout_buffer(
    env: MinigridWrapper,
    policy: MLPPolicy | IMPALAPolicy,
    critic: torch.nn.Module,
    buffer_size: int,
    device: str = "cuda",
    repr_net: torch.nn.Module = None,
) -> dict:
    """
    Collect rollout buffer by running episodes until buffer is full.
    
    Args:
        env: Environment
        policy: Policy network
        critic: Critic network
        buffer_size: Target buffer size (in steps)
        device: Device
        
    Returns:
        Dictionary with:
            - obs: Observations [buffer_size, obs_dim]
            - actions: Actions [buffer_size]
            - rewards: Rewards [buffer_size]
            - dones: Done flags [buffer_size]
            - log_probs: Log probabilities [buffer_size]
            - values: Value estimates [buffer_size]
            - episode_returns: List of episode returns
            - episode_lengths: List of episode lengths
    """
    buffer = {
        "obs": [],
        "actions": [],
        "rewards": [],
        "dones": [],
        "log_probs": [],
        "values": [],
    }
    
    episode_returns = []
    episode_lengths = []
    current_episode_return = 0
    current_episode_length = 0
    
    while len(buffer["obs"]) < buffer_size:
        obs, info = env.reset()
        done = False
        
        while not done and len(buffer["obs"]) < buffer_size:
            # Standard architecture: Shared CNN encoder s -> z, then both policy and critic use z
            obs_tensor = obs.unsqueeze(0).to(device) if not isinstance(obs, torch.Tensor) else obs.unsqueeze(0).to(device)
            
            with torch.no_grad():
                # Get representation z for policy and critic
                # Priority: repr_net > VAE encoder > raw observations
                if repr_net is not None:
                    # Image observations: use shared CNN encoder s -> z
                    # Flatten image observation for encoder
                    if obs_tensor.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = obs_tensor.shape
                        obs_flat = obs_tensor.view(N, H * W * C)
                    else:
                        obs_flat = obs_tensor
                    z = repr_net(obs_flat)  # s -> z [512-dim]
                elif hasattr(critic, 'get_latent_representation'):
                    # VAE critic: use VAE encoder to get latent z
                    if obs_tensor.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = obs_tensor.shape
                        obs_flat = obs_tensor.view(N, H * W * C)
                    else:
                        obs_flat = obs_tensor
                    z = critic.get_latent_representation(obs_flat)  # s -> z [32-dim]
                elif hasattr(critic, 'encode'):
                    # VAE with encode method
                    if obs_tensor.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = obs_tensor.shape
                        obs_flat = obs_tensor.view(N, H * W * C)
                    else:
                        obs_flat = obs_tensor
                    mu, _ = critic.encode(obs_flat)
                    z = mu  # s -> z [32-dim]
                else:
                    # No encoder: flatten image observations if needed
                    if obs_tensor.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = obs_tensor.shape
                        z = obs_tensor.view(N, H * W * C)  # Raw obs
                    else:
                        z = obs_tensor
                
                # Policy: z -> actions (z is always representation, not raw obs)
                action, log_prob = policy.get_action(z)
                
                # Critic: For VAE, pass raw obs; for others, pass z
                if hasattr(critic, 'get_latent_representation') or hasattr(critic, 'encode'):
                    # VAE critic: pass raw observations (critic has its own encoder)
                    if obs_tensor.dim() == 4:
                        N, H, W, C = obs_tensor.shape
                        obs_flat = obs_tensor.view(N, H * W * C)
                    else:
                        obs_flat = obs_tensor
                    value = critic(obs_flat).squeeze(-1)
                else:
                    # ICNN/Feedforward: pass representation z
                    value = critic(z).squeeze(-1)
                
                action = action.squeeze(0) if action.dim() > 0 else action
                log_prob = log_prob.squeeze(0) if log_prob.dim() > 0 else log_prob
                value = value.squeeze(0) if value.dim() > 0 else value
            
            next_obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            buffer["obs"].append(obs.cpu() if isinstance(obs, torch.Tensor) else obs)
            if hasattr(env, "get_ground_truth_representation"):
                gt = env.get_ground_truth_representation(obs)
                buffer.setdefault("gt_repr", []).append(
                    gt.cpu() if isinstance(gt, torch.Tensor) else gt
                )
            buffer["actions"].append(action.cpu())
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)
            buffer["log_probs"].append(log_prob.cpu())
            buffer["values"].append(value.cpu())
            
            # Store next observation for contrastive loss (if needed)
            # We'll store it as the "next_obs" for the previous step
            if "next_obs" not in buffer:
                buffer["next_obs"] = []
            buffer["next_obs"].append(next_obs.cpu() if isinstance(next_obs, torch.Tensor) else next_obs)
            
            current_episode_return += reward
            current_episode_length += 1
            
            obs = next_obs
            
            if done:
                episode_returns.append(current_episode_return)
                episode_lengths.append(current_episode_length)
                current_episode_return = 0
                current_episode_length = 0
    
    # Trim to exact buffer size if needed
    actual_size = min(len(buffer["obs"]), buffer_size)
    
    # Convert to tensors
    # Observations are stored as flattened [obs_dim] (will be encoded to z during training via shared encoder)
    buffer["obs"] = torch.stack(buffer["obs"][:actual_size])
    
    buffer["actions"] = torch.stack(buffer["actions"][:actual_size])
    buffer["rewards"] = torch.tensor(buffer["rewards"][:actual_size], dtype=torch.float32)
    buffer["dones"] = torch.tensor(buffer["dones"][:actual_size], dtype=torch.bool)
    buffer["log_probs"] = torch.stack(buffer["log_probs"][:actual_size])
    buffer["values"] = torch.stack(buffer["values"][:actual_size])
    
    # Next observations for contrastive loss
    # Note: next_obs[i] corresponds to the observation after obs[i]
    # For the last step, next_obs will be the terminal state (or next episode start)
    if "next_obs" in buffer and len(buffer["next_obs"]) > 0:
        buffer["next_obs"] = torch.stack(buffer["next_obs"][:actual_size])
    else:
        buffer["next_obs"] = None
    
    if "gt_repr" in buffer and len(buffer["gt_repr"]) > 0:
        buffer["gt_repr"] = torch.stack(buffer["gt_repr"][:actual_size])

    buffer["episode_returns"] = episode_returns
    buffer["episode_lengths"] = episode_lengths
    
    return buffer


def _policy_action_from_obs(
    obs,
    policy,
    critic,
    repr_net,
    device: str,
    deterministic: bool = True,
):
    """Shared obs -> action for eval and transfer rollouts."""
    if not isinstance(obs, torch.Tensor):
        obs_tensor = obs.unsqueeze(0).to(device)
    else:
        obs_tensor = obs.to(device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

    with torch.no_grad():
        if repr_net is not None:
            if obs_tensor.dim() == 4:
                N, H, W, C = obs_tensor.shape
                obs_flat = obs_tensor.view(N, H * W * C)
            else:
                obs_flat = obs_tensor
            z = repr_net(obs_flat)
        elif hasattr(critic, "get_latent_representation"):
            if obs_tensor.dim() == 4:
                N, H, W, C = obs_tensor.shape
                obs_flat = obs_tensor.view(N, H * W * C)
            else:
                obs_flat = obs_tensor
            z = critic.get_latent_representation(obs_flat)
        elif hasattr(critic, "encode"):
            if obs_tensor.dim() == 4:
                N, H, W, C = obs_tensor.shape
                obs_flat = obs_tensor.view(N, H * W * C)
            else:
                obs_flat = obs_tensor
            mu, _ = critic.encode(obs_flat)
            z = mu
        else:
            z = obs_tensor

        action, _ = policy.get_action(z, deterministic=deterministic)
        if isinstance(action, torch.Tensor):
            action = action.squeeze(0) if action.dim() > 0 and action.shape[0] == 1 else action
            action_val = action.item() if action.numel() == 1 else int(action)
        else:
            action_val = int(action)
    return action_val


def run_eval_rollout(
    env,
    policy,
    critic,
    repr_net,
    device: str,
    n_episodes: int,
    deterministic: bool = True,
) -> dict:
    """Run eval episodes; success = total_reward > 0 (Unlock unlock reward)."""
    rewards = []
    lengths = []
    successes = 0
    policy.eval()
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        steps = 0
        while not done:
            action_val = _policy_action_from_obs(
                obs, policy, critic, repr_net, device, deterministic=deterministic
            )
            obs, reward, terminated, truncated, _ = env.step(action_val)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated
        rewards.append(total_reward)
        lengths.append(steps)
        if total_reward > 0:
            successes += 1
    policy.train()
    return {
        "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
        "reward_std": float(np.std(rewards)) if rewards else 0.0,
        "reward_max": float(np.max(rewards)) if rewards else 0.0,
        "episode_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": successes / n_episodes if n_episodes > 0 else 0.0,
    }


def train(config_path: str, seed_override: int | None = None):
    """
    Main training function with buffer-based epochs.
    
    Information Flow:
    1. Main -> train() -> Setup (config, env, networks, algorithm, logger)
    2. train() -> collect_rollout_buffer() -> Collect episodes until buffer full
    3. train() -> algorithm.update() -> Update policy and critic on buffer
    4. train() -> metric_evaluator.evaluate_all() -> Compute metrics (periodically)
    5. train() -> logger.log_metrics() -> Log to CSV
    6. Repeat steps 2-5 for each epoch
    """
    # Load config
    config = Config(config_path)
    
    # Set seed
    seed = seed_override if seed_override is not None else config.get("experiment.seed", 42)
    set_seed(seed)
    
    # Device
    device = config.get("experiment.device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Create environment
    env_name = config["environment"]["name"]
    obs_shape = None  # For image observations
    if env_name == "minigrid":
        task = config["environment"]["task"]
        # Always flatten observations - shared encoder will handle encoding
        env = MinigridWrapper(task, seed=seed, keep_image_format=False)
        transfer_env = None
        eval_cfg = config.config.get("evaluation", {})
        if eval_cfg.get("transfer_augmentation") == "color_perm_v1":
            perm_seed = eval_cfg.get("transfer_color_perm_seed", seed + 1000)
            from src.environments.minigrid_wrapper import MinigridColorAugWrapper
            transfer_base = MinigridWrapper(
                eval_cfg.get("transfer_task", task), seed=seed, keep_image_format=False
            )
            transfer_env = MinigridColorAugWrapper(transfer_base, color_perm_seed=perm_seed)
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        ground_truth_repr_fn = env.get_ground_truth_representation
        
        # Get observation shape for CNN (Minigrid images are typically 7x7x3)
        # Use the shape stored in wrapper
        if hasattr(env, 'obs_shape'):
            obs_shape = env.obs_shape  # (H, W, C)
            print(f"Detected image observation shape: {obs_shape}")
    elif env_name == "frozenlake":
        task = config["environment"]["task"]  # e.g. "8x8" (map_name)
        is_slippery = config["environment"].get("is_slippery", False)

        env = FrozenLakeWrapper(task, is_slippery=is_slippery, seed=seed)
        obs_dim = env.obs_dim
        action_dim = env.action_dim
        ground_truth_repr_fn = env.get_ground_truth_representation
        obs_shape = env.obs_shape
        transfer_env = None
    else:
        raise ValueError(f"Unknown environment: {env_name}")
    
    # Create networks
    # Standard architecture: Shared CNN encoder (representation network) s -> z
    # Then both policy and critic use z as input
    # - Policy: z -> actions
    # - Critic: z -> v
    
    # Determine if we need a representation network
    # Only create repr_net for ICNN and Feedforward critics (not VAE - VAE has its own encoder)
    critic_type = config["architecture"]["critic"]["type"]
    needs_repr_net = (critic_type == "icnn" or critic_type == "feedforward")
    
    repr_net = None
    if needs_repr_net:
        # Vector observations (e.g. FrozenLake) use an MLP representation network.
        # Image observations (e.g. Minigrid) use the CNN representation network path.
        repr_net = create_representation_network(obs_dim, config.config, device, obs_shape=obs_shape)
        repr_dim = config["architecture"].get("representation", {}).get("repr_dim", 512)
        if obs_shape is not None:
            print(f"Created shared CNN encoder (representation network): s -> z (dim={repr_dim})")
        else:
            print(f"Created shared MLP encoder (representation network): s -> z (dim={repr_dim})")
    elif critic_type == "vae":
        # VAE critic has its own encoder, no separate repr_net needed
        # Policy will use VAE encoder's latent representation z
        latent_dim = config["architecture"]["critic"].get("latent_dim", 32)
        repr_dim = latent_dim  # Policy input dimension is VAE latent dimension
        print(f"VAE critic detected: Policy uses VAE encoder's latent z (dim={latent_dim})")
    else:
        # For vector observations, no separate encoder needed
        repr_dim = obs_dim
    
    # Create policy: takes Z as input (not raw observations s)
    # For image observations: policy uses z from CNN encoder
    # For vector observations: policy uses s directly (no encoder)
    policy = create_policy(repr_dim, action_dim, config.config, device, obs_shape=obs_shape, use_repr_input=True)
    
    # Create critic: 
    # - For ICNN: takes repr_dim (representation z) as input
    # - For VAE: takes obs_dim (raw observations) as input (VAE has its own encoder)
    # - For Feedforward: takes repr_dim if repr_net exists, else obs_dim
    critic_input_dim = obs_dim if critic_type == "vae" else repr_dim
    critic = create_critic(critic_input_dim, config.config, device, repr_net=repr_net)

    if critic_type == "vae" and (
        is_quadratic_latent_value_head(critic) or is_squared_norm_value_head(critic)
    ):
        probe_obs = torch.randn(4, critic_input_dim, device=device)
        mu_probe, _ = critic.encode(probe_obs)
        mu_result = verify_mu_latent_agreement(critic, mu_probe)
        print(f"μ_latent startup check: {mu_result}")

    # Create algorithm
    algorithm_type = config["algorithm"]["type"]
    if algorithm_type == "ppo":
        algorithm = PPO(
            policy, critic, config["algorithm"], device,
            repr_net=repr_net, action_dim=action_dim,
        )
    elif algorithm_type == "trpo":
        algorithm = TRPO(policy, critic, config["algorithm"], device, repr_net=repr_net)
    elif algorithm_type == "rstr" or algorithm_type == "representation_trpo":
        algorithm = RepresentationTRPO(policy, critic, config["algorithm"], device, repr_net=repr_net)
    else:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    # Create logger
    experiment_name = config["experiment"]["name"]
    if seed_override is not None:
        experiment_name = f"{experiment_name}_seed{seed}"
    log_dir = Path(config["logging"]["log_dir"])
    logger = CSVLogger(log_dir, experiment_name)
    logger.save_config(config.config)
    
    # Checkpoints/final weights live under experiment_name (avoids collisions when
    # policy/critic types match across different configs, e.g. RSTR vs VAE PPO experts).
    exp_storage_dir = log_dir / experiment_name
    exp_storage_dir.mkdir(parents=True, exist_ok=True)
    
    # Training configuration
    buffer_size = config["training"].get("buffer_size", 2048)  # Steps per epoch
    total_epochs = config["training"].get("total_epochs", None)
    total_steps = config["training"].get("total_steps", None)
    metric_eval_freq = config["training"].get("metric_evaluation_frequency", 10)  # Epochs between metric evaluation
    checkpoint_freq = config["training"].get("checkpoint_frequency", 100)  # Epochs between checkpoints
    eval_freq = config["training"].get("eval_frequency", 10)  # Epochs between policy evaluation
    eval_episodes = config["training"].get("eval_episodes", 10)
    eval_deterministic = config["training"].get("eval_deterministic", True)  # Use deterministic policy for eval (default: True, can set False for stochastic)
    
    # Phasic training configuration
    phase_size = config["training"].get("phase_size", None)  # Epochs per phase (None = no phasic training)
    phases = ["representation", "critic", "policy"]  # Order of phases
    
    # Seed-matched live expert Z*(s) for κ and bounding chain (avoids gt_repr table gaps)
    metrics_cfg = config["metrics"]
    z_ref_family = metrics_cfg.get("z_ref_expert_family") or metrics_cfg.get("z_ref_family")
    if z_ref_family:
        from src.theory_validation.z_ref_paths import resolve_z_ref_expert

        expert_config, expert_weights, z_ref_ckpt_kind = resolve_z_ref_expert(z_ref_family, seed)
        metrics_cfg["z_ref_expert_config"] = expert_config
        metrics_cfg["z_ref_expert_weights"] = expert_weights
        metrics_cfg["z_ref_expert_checkpoint_kind"] = z_ref_ckpt_kind
        metrics_cfg.pop("z_ref_path", None)
        print(
            f"Z_ref live expert ({z_ref_family}, seed {seed}): {expert_weights} [{z_ref_ckpt_kind}]",
            flush=True,
        )

    # Create metric evaluator (pass repr_net so it can encode observations for metrics)
    metric_evaluator = MetricEvaluator(metrics_cfg, ground_truth_repr_fn, repr_net=repr_net)

    algo_cfg = config["algorithm"]
    needs_training_z_ref = (
        algo_cfg.get("kappa_directional_loss_coef", 0) > 0
        or algo_cfg.get("z_distill_loss_coef", 0) > 0
    )
    if needs_training_z_ref:
        if not z_ref_family:
            raise ValueError(
                "kappa_directional_loss_coef or z_distill_loss_coef > 0 requires "
                "metrics.z_ref_expert_family"
            )
        if algorithm_type != "ppo":
            raise ValueError(
                f"Z* training losses require algorithm.type ppo, got {algorithm_type}"
            )
        from src.theory_validation.z_ref_expert import load_expert_critic

        algorithm.z_ref_expert = load_expert_critic(
            metrics_cfg["z_ref_expert_config"],
            metrics_cfg["z_ref_expert_weights"],
            device,
        )
        print("Z_ref expert attached for κ / distill training losses", flush=True)
    
    # Training state
    epoch = 0
    total_steps_collected = 0
    old_policy = None
    old_critic = None
    old_occupancy = None
    
    print(f"Starting training: {experiment_name}")
    print(f"Environment: {task}")
    print(f"Policy: {config['architecture']['policy']['type']}, Critic: {config['architecture']['critic']['type']}")
    print(f"Algorithm: {algorithm_type}")
    print(f"Buffer size: {buffer_size} steps per epoch")
    print(f"Metric evaluation frequency: Every {metric_eval_freq} epochs")
    if phase_size is not None:
        print(f"Phasic training: {phase_size} epochs per phase (order: {phases})")
    else:
        print("Phasic training: Disabled (training all components together)")
    
    # Phasic training state
    current_phase_idx = 0
    phase_epoch_count = 0
    cycle_count = 0  # Track complete cycles (N*3 phases)
    
    # Accumulate metrics across a cycle for phasic training
    cycle_metrics = None
    cycle_buffer = None
    
    # For non-phasic training: accumulate metrics over cycles to match RSTR logging frequency
    # RSTR uses phase_size=5 with 3 phases = 15 epochs per cycle
    non_phasic_cycle_size = 15  # Match RSTR's cycle length
    non_phasic_cycle_metrics = None
    non_phasic_cycle_count = 0
    
    # GPU memory tracking
    max_gpu_memory_mb = 0.0
    gpu_memory_log_freq = 100  # Log every 100 epochs
    
    expert_cfg = config.config.get("expert", {})
    training_complete = False

    # Training loop: epoch-based
    while True:
        # Check stopping condition
        if training_complete:
            break
        if total_epochs is not None and epoch >= total_epochs:
            break
        if total_steps is not None and total_steps_collected >= total_steps:
            break
        
        epoch += 1
        
        # Track GPU memory usage
        if torch.cuda.is_available():
            current_gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # Convert to MB
            max_gpu_memory_mb = max(max_gpu_memory_mb, current_gpu_memory_mb)
            
            # Log GPU memory every 100 epochs
            if epoch % gpu_memory_log_freq == 0:
                current_memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_memory_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                print(f"Epoch {epoch}: GPU Memory - Current: {current_memory_mb:.1f} MB, "
                      f"Reserved: {reserved_memory_mb:.1f} MB, Max (so far): {max_gpu_memory_mb:.1f} MB")
        
        # Determine current phase for phasic training
        cycle_completed = False
        if phase_size is not None:
            # Check if we need to switch phases
            if phase_epoch_count >= phase_size:
                # Check if we're completing a full cycle (just finished policy phase)
                if current_phase_idx == len(phases) - 1:  # Just finished policy phase
                    cycle_completed = True
                    cycle_count += 1
                
                current_phase_idx = (current_phase_idx + 1) % len(phases)
                phase_epoch_count = 0
            
            current_phase = phases[current_phase_idx]
            phase_epoch_count += 1
        else:
            current_phase = "all"
        
        # Collect rollout buffer (one epoch = buffer_size steps)
        buffer = collect_rollout_buffer(env, policy, critic, buffer_size, device=device, repr_net=repr_net)
        
        # Move buffer to device
        buffer["obs"] = buffer["obs"].to(device)
        buffer["actions"] = buffer["actions"].to(device)
        buffer["rewards"] = buffer["rewards"].to(device)
        buffer["dones"] = buffer["dones"].to(device)
        buffer["log_probs"] = buffer["log_probs"].to(device)
        buffer["values"] = buffer["values"].to(device)
        if buffer.get("next_obs") is not None:
            buffer["next_obs"] = buffer["next_obs"].to(device)
        
        total_steps_collected += len(buffer["obs"])
        
        # Compute advantages and returns
        next_value = 0.0  # Terminal state
        advantages, returns = algorithm.compute_gae(
            buffer["rewards"],
            buffer["values"],
            buffer["dones"],
            next_value,
        )
        
        # Update networks (one training epoch)
        # Only pass phase and next_obs to RepresentationTRPO (RSTR/RRSTR)
        # Vanilla PPO/TRPO don't accept these parameters
        if algorithm_type == "rstr" or algorithm_type == "representation_trpo":
            update_stats = algorithm.update(
                buffer["obs"],
                buffer["actions"],
                buffer["log_probs"],
                advantages,
                returns,
                phase=current_phase,
                next_obs=buffer.get("next_obs"),  # Pass next_obs for contrastive loss (RSTR only)
                training_epoch=epoch,
            )
        else:
            update_kwargs = dict(
                obs=buffer["obs"],
                actions=buffer["actions"],
                old_log_probs=buffer["log_probs"],
                advantages=advantages,
                returns=returns,
                training_epoch=epoch,
            )
            if algorithm_type == "ppo":
                algo_cfg = config["algorithm"]
                if (
                    algo_cfg.get("mico_loss_coef", 0) > 0
                    or algo_cfg.get("dbc_loss_coef", 0) > 0
                ):
                    update_kwargs["rewards"] = buffer["rewards"]
                    update_kwargs["next_obs"] = buffer["next_obs"]
            update_stats = algorithm.update(**update_kwargs)
        
        # Accumulate metrics for phasic training
        if phase_size is not None:
            # Initialize cycle metrics on first epoch of cycle
            if cycle_metrics is None:
                cycle_metrics = {
                    "cycle": cycle_count + 1,
                    "epoch_start": epoch - phase_epoch_count + 1,
                    "epoch_end": epoch,
                    "total_steps": total_steps_collected,
                    "episodes": [],
                    "episode_returns": [],
                    "episode_lengths": [],
                    "update_stats": [],
                }
                cycle_buffer = buffer
            
            # Accumulate metrics across the cycle
            cycle_metrics["epoch_end"] = epoch
            cycle_metrics["total_steps"] = total_steps_collected
            cycle_metrics["episodes"].append(len(buffer["episode_returns"]))
            cycle_metrics["episode_returns"].extend(buffer["episode_returns"])
            cycle_metrics["episode_lengths"].extend(buffer["episode_lengths"])
            cycle_metrics["update_stats"].append(update_stats)
            
            # Use the buffer from the policy phase (last phase) for expensive metrics
            if current_phase == "policy" and phase_epoch_count == phase_size:
                # This is the last epoch of the policy phase, use this buffer for metrics
                cycle_buffer = buffer
            
            # Only log at end of cycle (after policy phase completes)
            # Logging happens above when we detect cycle completion
        else:
            # Non-phasic training: accumulate metrics over cycles (15 epochs) to match RSTR logging frequency
            # Initialize cycle metrics on first epoch of cycle
            if non_phasic_cycle_metrics is None:
                non_phasic_cycle_metrics = {
                    "cycle": non_phasic_cycle_count + 1,
                    "epoch_start": epoch,
                    "epoch_end": epoch,
                    "total_steps": total_steps_collected,
                    "episodes": [],
                    "episode_returns": [],
                    "episode_lengths": [],
                    "update_stats": [],
                }
            
            # Accumulate metrics across the cycle
            non_phasic_cycle_metrics["epoch_end"] = epoch
            non_phasic_cycle_metrics["total_steps"] = total_steps_collected
            non_phasic_cycle_metrics["episodes"].append(len(buffer["episode_returns"]))
            non_phasic_cycle_metrics["episode_returns"].extend(buffer["episode_returns"])
            non_phasic_cycle_metrics["episode_lengths"].extend(buffer["episode_lengths"])
            non_phasic_cycle_metrics["update_stats"].append(update_stats)
            
            # Print training returns periodically for debugging (every epoch, but only log every cycle)
            mean_train_return = np.mean(buffer["episode_returns"]) if buffer["episode_returns"] else 0.0
            max_train_return = np.max(buffer["episode_returns"]) if buffer["episode_returns"] else 0.0
            if epoch % 50 == 0 or epoch <= 10:
                print(f"Epoch {epoch}: Training returns - mean: {mean_train_return:.3f}, max: {max_train_return:.3f}, episodes: {len(buffer['episode_returns'])}")
                # Print encoder gradient norm if available
                if "repr_grad_norm" in update_stats:
                    grad_norm = update_stats["repr_grad_norm"]
                    if grad_norm < 1e-8:
                        print(f"  WARNING: Encoder gradient norm is near zero ({grad_norm:.2e}) - encoder may not be learning!")
                    else:
                        print(f"  Encoder gradient norm: {grad_norm:.6f}")
            
            # Check encoding diversity on training buffer (periodically)
            if repr_net is not None and (epoch % 10 == 0 or epoch <= 5):
                with torch.no_grad():
                    # Sample a subset of observations from buffer
                    sample_size = min(100, buffer["obs"].shape[0])
                    sample_indices = torch.randperm(buffer["obs"].shape[0])[:sample_size]
                    sample_obs = buffer["obs"][sample_indices].to(device)
                    
                    # Get encodings
                    z_sample = repr_net(sample_obs)  # [sample_size, repr_dim]
                    
                    # Compute diversity metrics
                    z_mean = z_sample.mean(dim=0)
                    z_std = z_sample.std(dim=0)
                    z_var = z_sample.var(dim=0)
                    
                    # Pairwise distances
                    pairwise_dists = torch.cdist(z_sample, z_sample)
                    mask = torch.triu(torch.ones_like(pairwise_dists), diagonal=1).bool()
                    pairwise_dists_flat = pairwise_dists[mask]
                    
                    mean_std = z_std.mean().item()
                    mean_var = z_var.mean().item()
                    mean_pairwise_dist = pairwise_dists_flat.mean().item()
                    min_pairwise_dist = pairwise_dists_flat.min().item()
                    
                    if epoch <= 5 or epoch % 50 == 0:
                        print(f"  Training buffer encoding diversity ({sample_size} samples):")
                        print(f"    Mean std: {mean_std:.6f}, Mean var: {mean_var:.6f}, Mean pairwise dist: {mean_pairwise_dist:.6f}, Min pairwise dist: {min_pairwise_dist:.6f}")
                    
                    # Check for identical encodings
                    if mean_pairwise_dist < 1e-6:
                        print(f"    WARNING: Encodings appear identical in training buffer!")
                    elif min_pairwise_dist < 1e-6:
                        print(f"    WARNING: Some training encodings are identical (min pairwise dist: {min_pairwise_dist:.6e})")
            
            # Check if we've completed a cycle (15 epochs)
            epochs_in_cycle = epoch - non_phasic_cycle_metrics["epoch_start"] + 1
            if epochs_in_cycle >= non_phasic_cycle_size:
                # End of cycle: log averaged metrics (matching RSTR behavior)
                non_phasic_cycle_count += 1
                
                # Compute averaged metrics
                final_metrics = {
                    "epoch": epoch,
                    "total_steps": total_steps_collected,
                    "episodes": sum(non_phasic_cycle_metrics["episodes"]),
                    "mean_episode_return": np.mean(non_phasic_cycle_metrics["episode_returns"]) if non_phasic_cycle_metrics["episode_returns"] else 0.0,
                    "mean_episode_length": np.mean(non_phasic_cycle_metrics["episode_lengths"]) if non_phasic_cycle_metrics["episode_lengths"] else 0.0,
                }
                
                # Add GPU memory metrics if we're at a logging epoch
                if epoch % gpu_memory_log_freq == 0 and torch.cuda.is_available():
                    current_memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                    reserved_memory_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                    final_metrics["gpu_memory_mb"] = current_memory_mb
                    final_metrics["gpu_memory_reserved_mb"] = reserved_memory_mb
                    final_metrics["gpu_memory_max_mb"] = max_gpu_memory_mb
                
                # Aggregate update stats (take mean across epochs in cycle)
                if non_phasic_cycle_metrics["update_stats"]:
                    for key in non_phasic_cycle_metrics["update_stats"][0].keys():
                        values = [stats[key] for stats in non_phasic_cycle_metrics["update_stats"] if key in stats and isinstance(stats[key], (int, float))]
                        if values:
                            final_metrics[key] = np.mean(values)
                
                # Periodic metric evaluation (expensive metrics) - only at end of cycle
                if epoch % metric_eval_freq == 0:
                    print(f"Cycle {non_phasic_cycle_count} complete (epochs {non_phasic_cycle_metrics['epoch_start']}-{epoch}): Evaluating metrics...")
                    metric_results = metric_evaluator.evaluate_all(
                        policy=policy,
                        critic=critic,
                        obs_buffer=buffer["obs"],
                        old_policy=old_policy,
                        old_critic=old_critic,
                        episode_returns=non_phasic_cycle_metrics["episode_returns"],
                        old_occupancy=old_occupancy,
                        gt_repr_buffer=buffer.get("gt_repr"),
                    )
                    final_metrics.update(metric_results)
                    
                    # Update old models/occupancy for next comparison
                    old_policy = copy.deepcopy(policy)
                    old_critic = copy.deepcopy(critic)
                    if config["metrics"].get("collect_occupancy", False):
                        from src.metrics.occupancy import compute_occupancy_measure
                        old_occupancy = compute_occupancy_measure(buffer["obs"].cpu(), discretize=True)
                
                # Log accumulated metrics for the completed cycle
                _log_cycle_metrics(logger, total_steps_collected, final_metrics)
                
                # Reset for next cycle
                non_phasic_cycle_metrics = None
        
        # Periodic metric evaluation for phasic training (at end of cycle)
        if phase_size is not None and current_phase == "policy" and phase_epoch_count == phase_size:
            # End of policy phase = end of cycle, evaluate expensive metrics
            if epoch % metric_eval_freq == 0:
                obs_buf = cycle_buffer["obs"] if cycle_buffer is not None else buffer["obs"]
                gt_buf = (
                    cycle_buffer.get("gt_repr")
                    if cycle_buffer is not None
                    else buffer.get("gt_repr")
                )
                metric_results = metric_evaluator.evaluate_all(
                    policy=policy,
                    critic=critic,
                    obs_buffer=obs_buf,
                    old_policy=old_policy,
                    old_critic=old_critic,
                    episode_returns=cycle_metrics["episode_returns"] if cycle_metrics else buffer["episode_returns"],
                    old_occupancy=old_occupancy,
                    gt_repr_buffer=gt_buf,
                )
                if cycle_metrics is not None:
                    cycle_metrics.update(metric_results)
                
                # Update old models/occupancy for next comparison
                old_policy = copy.deepcopy(policy)
                old_critic = copy.deepcopy(critic)
                if config["metrics"].get("collect_occupancy", False):
                    from src.metrics.occupancy import compute_occupancy_measure
                    old_occupancy = compute_occupancy_measure(
                        (cycle_buffer["obs"] if cycle_buffer is not None else buffer["obs"]).cpu(),
                        discretize=True
                    )
        
        # Log cycle metrics when cycle completes
        if cycle_completed and cycle_metrics is not None:
            # Aggregate accumulated metrics
            final_metrics = {
                "cycle": cycle_metrics["cycle"],
                "epoch": cycle_metrics["epoch_end"],  # Log with final epoch number
                "epoch_start": cycle_metrics["epoch_start"],
                "epoch_end": cycle_metrics["epoch_end"],
                "total_steps": cycle_metrics["total_steps"],
                "total_episodes": sum(cycle_metrics["episodes"]),
                "mean_episode_return": np.mean(cycle_metrics["episode_returns"]) if cycle_metrics["episode_returns"] else 0.0,
                "mean_episode_length": np.mean(cycle_metrics["episode_lengths"]) if cycle_metrics["episode_lengths"] else 0.0,
            }
            
            # Add GPU memory metrics if we're at a logging epoch
            if epoch % gpu_memory_log_freq == 0 and torch.cuda.is_available():
                current_memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
                reserved_memory_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)
                final_metrics["gpu_memory_mb"] = current_memory_mb
                final_metrics["gpu_memory_reserved_mb"] = reserved_memory_mb
                final_metrics["gpu_memory_max_mb"] = max_gpu_memory_mb
            
            # Aggregate update stats (take mean across epochs in cycle)
            if cycle_metrics["update_stats"]:
                for key in cycle_metrics["update_stats"][0].keys():
                    values = [stats[key] for stats in cycle_metrics["update_stats"] if key in stats and isinstance(stats[key], (int, float))]
                    if values:
                        final_metrics[key] = np.mean(values)
            
            # Log accumulated metrics for the completed cycle
            _log_cycle_metrics(logger, total_steps_collected, final_metrics)
            
            # Reset for next cycle
            cycle_metrics = None
            cycle_buffer = None
        
        # Policy evaluation (rollout evaluation)
        # Evaluate based on eval_frequency, regardless of phase
        # For phasic training, we still evaluate at the specified frequency, but don't require end of phase
        if epoch % eval_freq == 0:
            print(f"Epoch {epoch}: Evaluating policy...")
            policy.eval()
            eval_rewards = []
            eval_episode_lengths = []
            action_counts = {}  # Track action distribution
            for ep_idx in range(eval_episodes):
                obs, _ = env.reset()
                total_reward = 0
                done = False
                step_count = 0
                episode_actions = []
                while not done:
                    # Process observation same way as in training
                    if not isinstance(obs, torch.Tensor):
                        obs_tensor = obs.unsqueeze(0).to(device)
                    else:
                        # Ensure it's on device and has batch dimension
                        obs_tensor = obs.to(device)
                        if obs_tensor.dim() == 1:
                            obs_tensor = obs_tensor.unsqueeze(0)
                    
                    with torch.no_grad():
                        # Get representation z for policy: Priority: repr_net > VAE encoder > raw obs
                        if repr_net is not None:
                            # Flatten image observation for encoder (same as training)
                            if obs_tensor.dim() == 4:  # [N, H, W, C]
                                N, H, W, C = obs_tensor.shape
                                obs_flat = obs_tensor.view(N, H * W * C)
                            else:
                                # Already flattened: [N, H*W*C] or [H*W*C]
                                obs_flat = obs_tensor
                            z = repr_net(obs_flat)  # s -> z [512-dim]
                            
                            # Track encodings for diversity check
                            if ep_idx == 0:
                                if "eval_encodings" not in locals():
                                    eval_encodings = []
                                    eval_obs_samples = []  # Track observations to verify if identical encodings = identical states
                                eval_encodings.append(z.cpu().clone())
                                # Store observations to check if identical encodings correspond to identical observations
                                # (For discrete state spaces, identical states should produce identical encodings)
                                eval_obs_samples.append(obs_flat.cpu().clone() if isinstance(obs_flat, torch.Tensor) else torch.tensor(obs_flat))
                        elif hasattr(critic, 'get_latent_representation'):
                            # VAE critic: use VAE encoder to get latent z for policy
                            if obs_tensor.dim() == 4:  # [N, H, W, C]
                                N, H, W, C = obs_tensor.shape
                                obs_flat = obs_tensor.view(N, H * W * C)
                            else:
                                obs_flat = obs_tensor
                            z = critic.get_latent_representation(obs_flat)  # s -> z [32-dim]
                        elif hasattr(critic, 'encode'):
                            # VAE with encode method
                            if obs_tensor.dim() == 4:  # [N, H, W, C]
                                N, H, W, C = obs_tensor.shape
                                obs_flat = obs_tensor.view(N, H * W * C)
                            else:
                                obs_flat = obs_tensor
                            mu, _ = critic.encode(obs_flat)
                            z = mu  # s -> z [32-dim]
                        else:
                            z = obs_tensor
                        
                        # Debug: check observation and representation shapes
                        if ep_idx == 0 and step_count == 0:
                            print(f"  Eval: obs shape={obs.shape if isinstance(obs, torch.Tensor) else 'numpy'}, obs_tensor shape={obs_tensor.shape}, z shape={z.shape}")
                        
                        action, _ = policy.get_action(z, deterministic=eval_deterministic)
                        
                        # Debug: check policy logits for first episode
                        if ep_idx == 0 and step_count < 3:
                            with torch.no_grad():
                                logits = policy.forward(z)
                                if isinstance(logits, tuple):
                                    logits = logits[0]  # For continuous, get mean
                                # Handle both batched and unbatched logits
                                if logits.dim() == 1:
                                    logits_batch = logits.unsqueeze(0)
                                else:
                                    logits_batch = logits
                                probs = torch.softmax(logits_batch, dim=-1)
                                if probs.dim() > 1 and probs.shape[0] == 1:
                                    probs = probs.squeeze(0)
                                z_norm = z.norm().item()
                                print(f"  Eval step {step_count}: logits={logits.cpu().numpy()}, probs={probs.cpu().numpy()}, z_norm={z_norm:.3f}")
                                
                                # Check if encoding changed from previous step
                                if step_count > 0 and repr_net is not None and len(eval_encodings) > 1:
                                    z_prev = eval_encodings[-2]
                                    z_curr = eval_encodings[-1]
                                    z_diff = (z_curr - z_prev).norm().item()
                                    print(f"    Encoding change from prev step: {z_diff:.6f}")
                    
                    # Extract action value - ensure proper squeezing like in training
                    if isinstance(action, torch.Tensor):
                        action = action.squeeze(0) if action.dim() > 0 and action.shape[0] == 1 else action
                        action_val = action.item() if action.numel() == 1 else int(action)
                    else:
                        action_val = int(action)
                    
                    episode_actions.append(action_val)
                    obs, reward, terminated, truncated, _ = env.step(action_val)
                    
                    # Verify reward is a number (debug check)
                    if not isinstance(reward, (int, float, np.number)):
                        print(f"Warning: Reward type is {type(reward)}, value: {reward}")
                    total_reward += float(reward)  # Ensure reward is converted to float
                    step_count += 1
                    done = terminated or truncated
                    
                    # Debug: print first episode's first few steps
                    if ep_idx == 0 and step_count <= 5:
                        print(f"  Eval step {step_count}: action={action_val}, reward={reward:.3f}, done={done}")
                
                eval_rewards.append(total_reward)
                eval_episode_lengths.append(step_count)
                
                # Track action distribution
                for a in episode_actions:
                    action_counts[a] = action_counts.get(a, 0) + 1
            
            policy.train()
            
            # Check encoding diversity if encoder is used
            if repr_net is not None and "eval_encodings" in locals() and len(eval_encodings) > 1:
                # Stack all encodings
                all_encodings = torch.stack(eval_encodings).squeeze(1)  # [num_steps, repr_dim]
                
                # Compute statistics
                encoding_mean = all_encodings.mean(dim=0)  # [repr_dim]
                encoding_std = all_encodings.std(dim=0)  # [repr_dim]
                encoding_var = all_encodings.var(dim=0)  # [repr_dim]
                
                # Compute pairwise distances
                pairwise_dists = torch.cdist(all_encodings, all_encodings)  # [num_steps, num_steps]
                # Get upper triangle (excluding diagonal)
                mask = torch.triu(torch.ones_like(pairwise_dists), diagonal=1).bool()
                pairwise_dists_flat = pairwise_dists[mask]
                
                # Summary statistics
                mean_encoding = encoding_mean.mean().item()  # Mean value across all dimensions
                mean_std = encoding_std.mean().item()
                mean_var = encoding_var.mean().item()
                mean_pairwise_dist = pairwise_dists_flat.mean().item()
                min_pairwise_dist = pairwise_dists_flat.min().item()
                max_pairwise_dist = pairwise_dists_flat.max().item()
                
                # Count identical pairs (pairs with distance < 1e-6)
                identical_pairs = (pairwise_dists_flat < 1e-6).sum().item()
                
                # Count unique encodings that are identical to at least one other encoding
                # Find all pairs with zero distance
                if min_pairwise_dist < 1e-6:
                    # Get indices of identical pairs from the upper triangle mask
                    identical_mask = pairwise_dists < 1e-6
                    # Find which encodings are involved in identical pairs
                    identical_encoding_indices = torch.any(identical_mask, dim=1) | torch.any(identical_mask, dim=0)
                    num_identical_encodings = identical_encoding_indices.sum().item()
                else:
                    num_identical_encodings = 0
                
                print(f"  Encoding diversity (first episode, {len(eval_encodings)} steps):")
                print(f"    Mean value across dims: {mean_encoding:.6f}")
                print(f"    Mean std across dims: {mean_std:.6f}")
                print(f"    Mean variance: {mean_var:.6f}")
                print(f"    Mean pairwise distance: {mean_pairwise_dist:.6f}")
                print(f"    Min pairwise distance: {min_pairwise_dist:.6f}")
                print(f"    Max pairwise distance: {max_pairwise_dist:.6f}")
                
                # Check if encodings are identical (all pairwise distances near zero)
                if mean_pairwise_dist < 1e-6:
                    print(f"    WARNING: All encodings appear identical! Mean pairwise distance is near zero.")
                    # Check if observations are actually different
                    if "eval_obs_samples" in locals() and len(eval_obs_samples) > 1:
                        obs_samples_tensor = torch.stack(eval_obs_samples)
                        obs_pairwise = torch.cdist(obs_samples_tensor, obs_samples_tensor)
                        obs_mask = torch.triu(torch.ones_like(obs_pairwise), diagonal=1).bool()
                        obs_pairwise_flat = obs_pairwise[obs_mask]
                        obs_min_dist = obs_pairwise_flat.min().item() if len(obs_pairwise_flat) > 0 else 0.0
                        print(f"    DEBUG: Min observation distance (first 10): {obs_min_dist:.6e}")
                        if obs_min_dist > 1e-6:
                            print(f"    ERROR: Observations are different but encodings are identical!")
                            print(f"    This indicates representation network collapse - network outputs constant encoding.")
                            print(f"    Encoding statistics: mean={all_encodings.mean().item():.6f}, std={all_encodings.std().item():.6f}")
                            print(f"    Encoding range: [{all_encodings.min().item():.6f}, {all_encodings.max().item():.6f}]")
                elif min_pairwise_dist < 1e-6:
                    # Check if this is due to discrete state space (expected) or representation collapse (problem)
                    # For discrete environments like Minigrid, identical states should produce identical encodings
                    if "eval_obs_samples" in locals() and len(eval_obs_samples) > 1:
                        obs_samples_tensor = torch.stack(eval_obs_samples)
                        obs_pairwise = torch.cdist(obs_samples_tensor, obs_samples_tensor)
                        obs_mask = torch.triu(torch.ones_like(obs_pairwise), diagonal=1).bool()
                        obs_pairwise_flat = obs_pairwise[obs_mask]
                        obs_min_dist = obs_pairwise_flat.min().item() if len(obs_pairwise_flat) > 0 else float('inf')
                        
                        # If observations are also identical, this is expected for discrete state spaces
                        if obs_min_dist < 1e-6:
                            print(f"    NOTE: {num_identical_encodings} out of {len(all_encodings)} encodings are identical")
                            print(f"            ({identical_pairs} identical pairs found). This is EXPECTED for discrete state spaces")
                            print(f"            (Minigrid has discrete states, so identical states → identical encodings)")
                        else:
                            # Different observations but identical encodings = representation collapse
                            print(f"    WARNING: {num_identical_encodings} out of {len(all_encodings)} encodings are identical to at least one other encoding")
                            print(f"            ({identical_pairs} identical pairs found, min pairwise distance: {min_pairwise_dist:.6e})")
                            print(f"            Different observations (min obs distance: {obs_min_dist:.6e}) produce identical encodings!")
                            print(f"            This indicates representation collapse - network outputs constant encoding for different states.")
                    else:
                        # Can't verify, but likely discrete state space
                        print(f"    NOTE: {num_identical_encodings} out of {len(all_encodings)} encodings are identical")
                        print(f"            ({identical_pairs} identical pairs found). This may be EXPECTED for discrete state spaces")
                        print(f"            (Minigrid has discrete states, so identical states → identical encodings)")
            
            avg_reward = np.mean(eval_rewards)
            std_reward = np.std(eval_rewards)
            avg_length = np.mean(eval_episode_lengths)
            max_reward = np.max(eval_rewards)
            eval_success_rate = (
                sum(1 for r in eval_rewards if r > 0) / len(eval_rewards) if eval_rewards else 0.0
            )
            eval_mode = "deterministic" if eval_deterministic else "stochastic"
            
            # Format action distribution as string for CSV
            action_dist_str = ""
            if len(action_counts) > 0:
                action_dist_str = str(dict(sorted(action_counts.items())))
            
            # Get recent loss values from update_stats (if available)
            # For phasic training, find the most recent stats from the appropriate phase
            recent_value_loss = None
            recent_policy_loss = None
            recent_contrastive_loss = None
            recent_entropy = None
            
            if phase_size is not None and cycle_metrics is not None and len(cycle_metrics["update_stats"]) > 0:
                # Search backwards through update_stats to find the most recent stats from each phase
                # value_loss is computed in all phases, so use the most recent
                # policy_loss is only computed in policy phase
                # contrastive_loss is only computed in representation phase
                for stats in reversed(cycle_metrics["update_stats"]):
                    phase = stats.get("phase", "unknown")
                    
                    # Get value_loss from most recent update (any phase)
                    if recent_value_loss is None and stats.get("value_loss") is not None:
                        value_loss_val = stats.get("value_loss")
                        if value_loss_val != 0.0:  # Only use non-zero value loss
                            recent_value_loss = value_loss_val
                    
                    # Get policy_loss from most recent policy phase update
                    if recent_policy_loss is None and phase in ["policy", "all"] and stats.get("policy_loss") is not None:
                        policy_loss_val = stats.get("policy_loss")
                        if policy_loss_val != 0.0:  # Only use non-zero policy loss
                            recent_policy_loss = policy_loss_val
                    
                    # Get contrastive_loss from most recent representation phase update
                    if recent_contrastive_loss is None and phase in ["representation", "all"] and stats.get("contrastive_loss") is not None:
                        contrastive_loss_val = stats.get("contrastive_loss")
                        if contrastive_loss_val != 0.0:  # Only use non-zero contrastive loss
                            recent_contrastive_loss = contrastive_loss_val
                    
                    # Stop if we've found all metrics
                    if recent_value_loss is not None and recent_policy_loss is not None and recent_contrastive_loss is not None:
                        break
            elif non_phasic_cycle_metrics is not None and len(non_phasic_cycle_metrics["update_stats"]) > 0:
                # For non-phasic training, get the most recent update stats
                recent_stats = non_phasic_cycle_metrics["update_stats"][-1]
                recent_value_loss = recent_stats.get("value_loss")
                recent_policy_loss = recent_stats.get("policy_loss")
                recent_contrastive_loss = recent_stats.get("contrastive_loss")
                # Don't use entropy from update_stats - compute it directly from policy
            
            # Compute entropy directly from current policy using a sample of observations
            # This is a property of the policy, not a training metric
            policy.eval()
            with torch.no_grad():
                # Use a sample of observations from the current buffer
                sample_size = min(256, len(buffer["obs"]))
                obs_sample = buffer["obs"][:sample_size]
                
                # Policy is always fed latent z (same path as rollout). Map buffer obs -> z.
                if repr_net is not None:
                    # Flatten image observations if needed
                    if obs_sample.dim() == 4:  # [N, H, W, C]
                        N, H, W, C = obs_sample.shape
                        obs_flat = obs_sample.view(N, H * W * C)
                    else:
                        obs_flat = obs_sample
                    z_sample = repr_net(obs_flat)
                elif hasattr(critic, 'get_latent_representation'):
                    z_sample = critic.get_latent_representation(obs_sample)
                elif hasattr(critic, 'encode'):
                    mu, _ = critic.encode(obs_sample)
                    z_sample = mu
                else:
                    z_sample = obs_sample
                
                # Compute entropy directly from policy forward pass
                try:
                    if hasattr(policy, 'action_space_type') and policy.action_space_type == "discrete":
                        logits = policy(z_sample)
                        dist = torch.distributions.Categorical(logits=logits)
                        entropy_vals = dist.entropy()
                    else:
                        # Continuous action space or fallback
                        mean, log_std = policy(z_sample)
                        std = torch.exp(log_std)
                        dist = torch.distributions.Normal(mean, std)
                        entropy_vals = dist.entropy().sum(dim=-1)
                    computed_entropy = entropy_vals.mean().item()
                except Exception as e:
                    print(f"Warning: Could not compute entropy: {e}")
                    computed_entropy = None
            policy.train()
            
            # Use computed entropy if available, otherwise fall back to update_stats
            if computed_entropy is not None:
                recent_entropy = computed_entropy
            
            # Print clean evaluation block
            print("\n" + "=" * 60)
            print(f"EVALUATION - Epoch {epoch} ({eval_mode})")
            print("=" * 60)
            print(f"reward_mean:     {avg_reward:.4f}")
            print(f"reward_std:      {std_reward:.4f}")
            print(f"reward_max:      {max_reward:.4f}")
            print(f"episode_length:   {avg_length:.1f}")
            print(f"success_rate:     {eval_success_rate:.2%}")
            if len(action_counts) > 0:
                print(f"action_dist:     {action_dist_str}")
            if recent_value_loss is not None:
                print(f"value_loss:      {recent_value_loss:.6f}")
            if recent_policy_loss is not None:
                print(f"policy_loss:     {recent_policy_loss:.6f}")
            if recent_contrastive_loss is not None:
                print(f"contrastive_loss: {recent_contrastive_loss:.6f}")
            if recent_entropy is not None:
                print(f"entropy:         {recent_entropy:.6f}")
            
            # Print detailed loss breakdown with and without coefficients
            if non_phasic_cycle_metrics is not None and len(non_phasic_cycle_metrics["update_stats"]) > 0:
                recent_stats = non_phasic_cycle_metrics["update_stats"][-1]
                raw_components = recent_stats.get("raw_loss_components", {})
                
                if raw_components:
                    print("\n" + "-" * 60)
                    print("DETAILED LOSS BREAKDOWN (with and without coefficients)")
                    print("-" * 60)
                    
                    # Get coefficients from config
                    value_coef = config.get("algorithm", {}).get("value_coef", 0.5)
                    vae_coef = config.get("algorithm", {}).get("vae_coef", 0.1)
                    representation_loss_coef = config.get("algorithm", {}).get("representation_loss_coef", 0.0)
                    convexity_coef = config.get("algorithm", {}).get("convexity_coef", 1.0)
                    entropy_coef = config.get("algorithm", {}).get("entropy_coef", 0.01)
                    
                    # Value loss
                    if raw_components.get('value_loss_raw') is not None:
                        val_raw = raw_components['value_loss_raw']
                        val_weighted = val_raw * value_coef
                        print(f"Value Loss:")
                        print(f"  Raw:           {val_raw:.6f}")
                        print(f"  Coef:          {value_coef:.4f}")
                        print(f"  Weighted:      {val_weighted:.6f}")
                    
                    # VAE losses
                    if raw_components.get('vae_recon_loss_raw') is not None:
                        recon_raw = raw_components['vae_recon_loss_raw']
                        kl_raw = raw_components.get('vae_kl_loss_raw', 0.0)
                        vae_raw = raw_components.get('vae_loss_raw', 0.0)
                        vae_weighted = vae_raw * vae_coef
                        print(f"\nVAE Loss:")
                        print(f"  Recon (raw):   {recon_raw:.6f}")
                        print(f"  KL (raw):      {kl_raw:.6f}")
                        print(f"  Total (raw):   {vae_raw:.6f}")
                        print(f"  Coef:          {vae_coef:.4f}")
                        print(f"  Weighted:      {vae_weighted:.6f}")
                    
                    # Representation loss
                    if raw_components.get('representation_loss_raw') is not None:
                        repr_grad_norm_powered = raw_components.get('representation_grad_norm_sq', 0.0)  # Actually grad_norm^power
                        repr_mu = raw_components.get('representation_mu', 0.0)
                        repr_raw = raw_components['representation_loss_raw']
                        repr_weighted = repr_raw * representation_loss_coef
                        grad_norm_power = config.get("algorithm", {}).get("grad_norm_power", 1.0)
                        print(f"\nRepresentation Loss:")
                        print(f"  Grad norm^{grad_norm_power:.1f}:  {repr_grad_norm_powered:.6f}")
                        print(f"  μ (min eig):   {repr_mu:.6f}")
                        print(f"  -μ * grad^{grad_norm_power:.1f}:  {-repr_mu * repr_grad_norm_powered:.6f} (before convexity_coef)")
                        print(f"  Convexity coef: {convexity_coef:.4f}")
                        print(f"  Raw loss:      {repr_raw:.6f} (= -{convexity_coef:.4f} * {repr_mu:.6f} * {repr_grad_norm_powered:.6f})")
                        print(f"  Repr coef:     {representation_loss_coef:.4f}")
                        print(f"  Weighted:      {repr_weighted:.6f}")
                    
                    # Policy loss
                    if raw_components.get('policy_loss_raw') is not None:
                        policy_raw = raw_components['policy_loss_raw']
                        entropy_raw = raw_components.get('entropy_raw', 0.0)
                        policy_with_entropy = policy_raw + entropy_coef * (-entropy_raw)
                        print(f"\nPolicy Loss:")
                        print(f"  Policy (raw):  {policy_raw:.6f}")
                        print(f"  Entropy (raw): {entropy_raw:.6f}")
                        print(f"  Entropy coef:  {entropy_coef:.4f}")
                        print(f"  Total:         {policy_with_entropy:.6f} (= {policy_raw:.6f} + {entropy_coef:.4f} * {-entropy_raw:.6f})")
                    
                    # Total critic loss
                    val_weighted = raw_components.get('value_loss_raw', 0.0) * value_coef
                    vae_weighted = raw_components.get('vae_loss_raw', 0.0) * vae_coef
                    repr_weighted = raw_components.get('representation_loss_raw', 0.0) * representation_loss_coef
                    total_critic = val_weighted + vae_weighted + repr_weighted
                    print(f"\nTotal Critic Loss: {total_critic:.6f}")
                    print(f"  = Value ({val_weighted:.6f}) + VAE ({vae_weighted:.6f}) + Repr ({repr_weighted:.6f})")
                    print("-" * 60)
            
            print("=" * 60 + "\n")
            
            # Log all metrics to CSV
            # Ensure we use the current step count for evaluation metrics
            # This ensures evaluation rows have the same step value as training rows at that point
            eval_metrics = {
                "eval_reward_mean": avg_reward,
                "eval_reward_std": std_reward,
                "eval_reward_max": max_reward,
                "eval_episode_length": avg_length,
                "eval_success_rate": eval_success_rate,
                "eval_mode": eval_mode,
            }
            if action_dist_str:
                eval_metrics["eval_action_distribution"] = action_dist_str

            if transfer_env is not None:
                transfer_results = run_eval_rollout(
                    transfer_env,
                    policy,
                    critic,
                    repr_net,
                    device,
                    eval_episodes,
                    deterministic=eval_deterministic,
                )
                eval_metrics["eval_transfer_reward_mean"] = transfer_results["reward_mean"]
                eval_metrics["eval_transfer_success_rate"] = transfer_results["success_rate"]
                print(
                    f"transfer success: {transfer_results['success_rate']:.2%}, "
                    f"transfer reward: {transfer_results['reward_mean']:.4f}"
                )
            
            # Use total_steps_collected to ensure alignment with training metrics
            logger.log_metrics(total_steps_collected, eval_metrics)

            min_sr = expert_cfg.get("min_success_rate")
            if min_sr is not None and eval_success_rate >= min_sr:
                expert_dir = log_dir / experiment_name
                expert_dir.mkdir(parents=True, exist_ok=True)
                expert_path = expert_dir / "weights_expert.pt"
                algorithm.save(str(expert_path))
                print(
                    f"Expert checkpoint saved to {expert_path} "
                    f"(success_rate {eval_success_rate:.2%} >= {min_sr:.2%})"
                )
                if expert_cfg.get("stop_when_ready", False):
                    training_complete = True
                    print("Expert success threshold met — stopping training.")
        
        # Checkpointing
        if epoch % checkpoint_freq == 0:
            latest_path = exp_storage_dir / "weights_latest.pt"
            algorithm.save(str(latest_path))
            print(f"Epoch {epoch}: Checkpoint saved")
    
    # Log final incomplete cycle for non-phasic training (if any)
    if phase_size is None and non_phasic_cycle_metrics is not None:
        # Log the remaining metrics from the incomplete cycle
        final_metrics = {
            "epoch": epoch,
            "total_steps": total_steps_collected,
            "episodes": sum(non_phasic_cycle_metrics["episodes"]),
            "mean_episode_return": np.mean(non_phasic_cycle_metrics["episode_returns"]) if non_phasic_cycle_metrics["episode_returns"] else 0.0,
            "mean_episode_length": np.mean(non_phasic_cycle_metrics["episode_lengths"]) if non_phasic_cycle_metrics["episode_lengths"] else 0.0,
        }
        
        # Aggregate update stats
        if non_phasic_cycle_metrics["update_stats"]:
            for key in non_phasic_cycle_metrics["update_stats"][0].keys():
                values = [stats[key] for stats in non_phasic_cycle_metrics["update_stats"] if key in stats and isinstance(stats[key], (int, float))]
                if values:
                    final_metrics[key] = np.mean(values)
        
        _log_cycle_metrics(logger, total_steps_collected, final_metrics)
        epochs_in_final_cycle = non_phasic_cycle_metrics["epoch_end"] - non_phasic_cycle_metrics["epoch_start"] + 1
        print(f"Final cycle logged: epochs {non_phasic_cycle_metrics['epoch_start']}-{non_phasic_cycle_metrics['epoch_end']} ({epochs_in_final_cycle} epochs)")
    
    # Save final weights
    final_path = exp_storage_dir / "weights_final.pt"
    algorithm.save(str(final_path))
    logger.close()
    
    print(f"Training completed! Final weights saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON file")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed")
    args = parser.parse_args()
    
    train(args.config, seed_override=args.seed)
