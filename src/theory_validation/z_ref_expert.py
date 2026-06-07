"""Frozen expert critic for live Z_ref(s) = expert_encoder(s) during metric evaluation."""

from pathlib import Path

import torch
import torch.nn as nn

from src.environments.minigrid_wrapper import MinigridWrapper
from src.utils.config import Config


def load_expert_critic(config_path: str, weights_path: str, device: str) -> nn.Module:
    """Load expert checkpoint; return VAE critic used for Z_ref encoding."""
    from src.algorithms.ppo import PPO
    from src.main import create_critic, create_policy

    config = Config(config_path)
    task = config["environment"]["task"]
    env = MinigridWrapper(task, seed=42, keep_image_format=False)
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    critic_type = config["architecture"]["critic"]["type"]
    repr_net = None
    if critic_type == "vae":
        repr_dim = config["architecture"]["critic"].get("latent_dim", 8)
    else:
        repr_dim = config["architecture"].get("representation", {}).get("repr_dim", 512)
    policy = create_policy(repr_dim, action_dim, config.config, device, use_repr_input=True)
    critic_input_dim = obs_dim if critic_type == "vae" else repr_dim
    critic = create_critic(critic_input_dim, config.config, device, repr_net=repr_net)
    algo_cfg = dict(config["algorithm"])
    algo_cfg["mico_loss_coef"] = 0
    algo_cfg["dbc_loss_coef"] = 0
    algorithm = PPO(
        policy, critic, algo_cfg, device,
        repr_net=repr_net, action_dim=action_dim,
    )
    algorithm.load(str(weights_path))
    critic.eval()
    env.close()
    return critic


@torch.no_grad()
def encode_z_ref_batch(critic: nn.Module, obs: torch.Tensor) -> torch.Tensor:
    """Map observations to expert latents [N, d]."""
    if hasattr(critic, "encode"):
        mu, _ = critic.encode(obs)
        return mu
    if hasattr(critic, "get_latent_representation"):
        return critic.get_latent_representation(obs)
    raise ValueError("Expert critic must provide encode() for Z_ref")
