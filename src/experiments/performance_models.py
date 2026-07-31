"""Model factories for Procgen / DMControl performance experiments."""

from dataclasses import dataclass

import torch.nn as nn

from src.agents.ctro import CTRO
from src.agents.ppo import PPO
from src.architectures.critics.cnn_encoder_critic import CNNEncoderCritic
from src.architectures.critics.cnn_vae_critic import CNNVAECritic
from src.architectures.critics.feedforward import FeedforwardCritic
from src.architectures.critics.impala_value_critic import ImpalaValueCritic
from src.architectures.critics.mlp_encoder_critic import MLPEncoderCritic
from src.architectures.policies.impala import IMPALAPolicy
from src.architectures.policies.impala_policy import ImpalaPolicy
from src.architectures.policies.mlp_policy import MLPPolicy


@dataclass(frozen=True)
class PerformanceStack:
    policy: nn.Module
    critic: nn.Module
    policy_on_latent: bool
    pixel_obs: bool
    stack_type: str


def _create_latent_policy(
    repr_dim: int,
    action_dim: int,
    arch: dict,
    device: str,
    action_space_type: str,
) -> IMPALAPolicy:
    p = arch["policy"]
    return IMPALAPolicy(
        repr_dim,
        action_dim,
        p["hidden_sizes"],
        p["activation"],
        action_space_type,
        p["num_residual_blocks"],
    ).to(device)


def _create_mlp_encoder_critic(obs_dim: int, arch_cfg: dict, device: str) -> MLPEncoderCritic:
    c = arch_cfg["critic"]
    return MLPEncoderCritic(
        obs_dim=obs_dim,
        encoder_hidden=c["encoder_hidden"],
        activation=c.get("activation", "tanh"),
    ).to(device)


def _create_cnn_encoder_critic(
    obs_shape: tuple[int, int, int],
    arch_cfg: dict,
    device: str,
) -> CNNEncoderCritic:
    c = arch_cfg["critic"]
    cnn = arch_cfg.get("cnn", {})
    return CNNEncoderCritic(
        obs_shape=obs_shape,
        latent_dim=c["latent_dim"],
        emb_size=cnn.get("emb_size", 256),
        depths=tuple(cnn.get("depths", [16, 32, 32])),
        activation=c.get("activation", "gelu"),
        value_hidden=c.get("value_hidden", [128, 128]),
    ).to(device)


def _create_cnn_vae_critic(
    obs_shape: tuple[int, int, int],
    arch_cfg: dict,
    device: str,
) -> CNNVAECritic:
    c = arch_cfg["critic"]
    cnn = arch_cfg.get("cnn", {})
    return CNNVAECritic(
        obs_shape=obs_shape,
        latent_dim=c["latent_dim"],
        emb_size=cnn.get("emb_size", 256),
        depths=tuple(cnn.get("depths", [16, 32, 32])),
        activation=c.get("activation", "gelu"),
        beta=c.get("beta", 1.0),
        value_hidden=c.get("value_hidden", [128, 128]),
    ).to(device)


def _pixel_ctro_stack(
    obs_shape: tuple[int, int, int],
    action_dim: int,
    action_space_type: str,
    arch_cfg: dict,
    device: str,
    stack_type_override: str | None = None,
) -> PerformanceStack:
    """Procgen CTRO: encoder (no VAE) by default; VAE when critic.type==vae or override."""
    c = arch_cfg["critic"]
    critic_type = c.get("type", "encoder")
    use_vae = stack_type_override == "ctro_cnn_vae" or (
        stack_type_override is None and critic_type == "vae"
    )
    if use_vae:
        critic = _create_cnn_vae_critic(obs_shape, arch_cfg, device)
        stack_type = "ctro_cnn_vae"
    else:
        critic = _create_cnn_encoder_critic(obs_shape, arch_cfg, device)
        stack_type = "ctro_cnn"
    policy = _create_latent_policy(
        c["latent_dim"],
        action_dim,
        arch_cfg,
        device,
        action_space_type,
    )
    return PerformanceStack(
        policy=policy,
        critic=critic,
        policy_on_latent=True,
        pixel_obs=True,
        stack_type=stack_type,
    )


def _create_mlp_latent_policy(
    latent_dim: int,
    action_dim: int,
    arch_cfg: dict,
    device: str,
    action_space_type: str,
) -> MLPPolicy:
    p = arch_cfg["policy"]
    return MLPPolicy(
        latent_dim,
        action_dim,
        hidden_sizes=p["hidden_sizes"],
        activation=p["activation"],
        action_space_type=action_space_type,
    ).to(device)


def build_performance_stack(
    arch_cfg: dict,
    env,
    agent_cls: type,
    device: str,
) -> PerformanceStack:
    is_ctro = agent_cls is CTRO
    pixel_obs = env.obs_shape is not None
    c = arch_cfg["critic"]
    p = arch_cfg["policy"]

    if pixel_obs:
        obs_shape = tuple(env.obs_shape)
        cnn = arch_cfg.get("cnn", {})
        emb_size = cnn.get("emb_size", 256)
        depths = tuple(cnn.get("depths", [16, 32, 32]))
        if is_ctro:
            return _pixel_ctro_stack(
                obs_shape,
                env.action_dim,
                env.action_space_type,
                arch_cfg,
                device,
            )

        critic = ImpalaValueCritic(obs_shape, emb_size=emb_size, depths=depths).to(device)
        policy = ImpalaPolicy(
            obs_shape,
            env.action_dim,
            action_space_type=env.action_space_type,
            emb_size=emb_size,
            depths=depths,
        ).to(device)
        return PerformanceStack(
            policy=policy,
            critic=critic,
            policy_on_latent=False,
            pixel_obs=True,
            stack_type="ppo_impala",
        )

    if is_ctro:
        critic = _create_mlp_encoder_critic(env.obs_dim, arch_cfg, device)
        policy = _create_mlp_latent_policy(
            critic.latent_dim,
            env.action_dim,
            arch_cfg,
            device,
            env.action_space_type,
        )
        return PerformanceStack(
            policy=policy,
            critic=critic,
            policy_on_latent=True,
            pixel_obs=False,
            stack_type="ctro_mlp",
        )

    critic = FeedforwardCritic(
        env.obs_dim,
        hidden_sizes=p.get("hidden_sizes", [256, 256]),
        activation=p.get("activation", "tanh"),
    ).to(device)
    policy = MLPPolicy(
        env.obs_dim,
        env.action_dim,
        hidden_sizes=p.get("hidden_sizes", [256, 256]),
        activation=p.get("activation", "tanh"),
        action_space_type=env.action_space_type,
    ).to(device)
    return PerformanceStack(
        policy=policy,
        critic=critic,
        policy_on_latent=False,
        pixel_obs=False,
        stack_type="ppo_mlp",
    )


def build_performance_stack_from_config(
    config: dict,
    obs_dim: int,
    action_dim: int,
    action_space_type: str,
    obs_shape: tuple[int, ...] | None,
    device: str,
) -> PerformanceStack:
    arch_cfg = config["architecture"]
    agent_cls = CTRO if config["agent_class"] == "CTRO" else PPO
    is_ctro = agent_cls is CTRO
    pixel_obs = obs_shape is not None
    c = arch_cfg["critic"]
    p = arch_cfg["policy"]

    if pixel_obs:
        obs_shape_t = tuple(obs_shape)
        cnn = arch_cfg.get("cnn", {})
        emb_size = cnn.get("emb_size", 256)
        depths = tuple(cnn.get("depths", [16, 32, 32]))
        if is_ctro:
            return _pixel_ctro_stack(
                obs_shape_t,
                action_dim,
                action_space_type,
                arch_cfg,
                device,
                stack_type_override=config.get("stack_type"),
            )

        critic = ImpalaValueCritic(obs_shape_t, emb_size=emb_size, depths=depths).to(device)
        policy = ImpalaPolicy(
            obs_shape_t,
            action_dim,
            action_space_type=action_space_type,
            emb_size=emb_size,
            depths=depths,
        ).to(device)
        return PerformanceStack(
            policy=policy,
            critic=critic,
            policy_on_latent=False,
            pixel_obs=True,
            stack_type="ppo_impala",
        )

    if is_ctro:
        # Backward-compatible reload for older VAE CTRO checkpoints.
        if config.get("stack_type") == "ctro_mlp_vae":
            from src.experiments.runner import create_critic as create_mlp_vae_critic

            critic = create_mlp_vae_critic(obs_dim, arch_cfg, device)
            policy = IMPALAPolicy(
                c["latent_dim"],
                action_dim,
                p["hidden_sizes"],
                p["activation"],
                action_space_type,
                p["num_residual_blocks"],
            ).to(device)
            return PerformanceStack(
                policy=policy,
                critic=critic,
                policy_on_latent=True,
                pixel_obs=False,
                stack_type="ctro_mlp_vae",
            )

        critic = _create_mlp_encoder_critic(obs_dim, arch_cfg, device)
        policy = _create_mlp_latent_policy(
            critic.latent_dim,
            action_dim,
            arch_cfg,
            device,
            action_space_type,
        )
        return PerformanceStack(
            policy=policy,
            critic=critic,
            policy_on_latent=True,
            pixel_obs=False,
            stack_type="ctro_mlp",
        )

    critic = FeedforwardCritic(
        obs_dim,
        hidden_sizes=p.get("hidden_sizes", [256, 256]),
        activation=p.get("activation", "tanh"),
    ).to(device)
    policy = MLPPolicy(
        obs_dim,
        action_dim,
        hidden_sizes=p.get("hidden_sizes", [256, 256]),
        activation=p.get("activation", "tanh"),
        action_space_type=action_space_type,
    ).to(device)
    return PerformanceStack(
        policy=policy,
        critic=critic,
        policy_on_latent=False,
        pixel_obs=False,
        stack_type="ppo_mlp",
    )
