"""
VAE-based Critic: Uses variational autoencoder for causal feature encoding,
then value function over latent space.
"""

import torch
import torch.nn as nn

from src.architectures.activation import activation_module
import torch.nn.functional as F

from src.architectures.value_heads.quadratic_psd import (
    QuadraticBottleneckValueHead,
    QuadraticLatentPSDValueHead,
)
from src.architectures.value_heads.squared_norm import SquaredNormValueHead


def build_vae_value_head(
    value_head_type: str,
    latent_dim: int,
    value_hidden: list[int],
    activation: nn.Module,
    value_mlp_hidden: int,
    value_bottleneck_dim: int,
    value_mu_min_floor: float,
    value_feature_dim: int,
) -> nn.Module:
    """Construct value head from type string and hyperparameters."""
    if value_head_type == "affine":
        return nn.Linear(latent_dim, 1)

    if value_head_type == "mlp":
        value_layers: list[nn.Module] = []
        input_dim = latent_dim
        for hidden_size in value_hidden:
            value_layers.append(nn.Linear(input_dim, hidden_size))
            value_layers.append(activation)
            input_dim = hidden_size
        value_layers.append(nn.Linear(input_dim, 1))
        head = nn.Sequential(*value_layers)
        nn.init.orthogonal_(head[-1].weight, gain=0.01)
        nn.init.zeros_(head[-1].bias)
        return head

    if value_head_type in ("quadratic_latent", "quadratic_latent_mumin"):
        floor = value_mu_min_floor if value_head_type == "quadratic_latent_mumin" else 0.0
        return QuadraticLatentPSDValueHead(latent_dim=latent_dim, mu_min_floor=floor)

    if value_head_type in ("quadratic_bottleneck", "quadratic_bottleneck_mumin"):
        floor = value_mu_min_floor if value_head_type == "quadratic_bottleneck_mumin" else 0.0
        return QuadraticBottleneckValueHead(
            latent_dim=latent_dim,
            hidden_dim=value_mlp_hidden,
            bottleneck_dim=value_bottleneck_dim,
            mu_min_floor=floor,
        )

    if value_head_type == "squared_norm":
        return SquaredNormValueHead(
            latent_dim=latent_dim,
            hidden_dim=value_mlp_hidden,
            feature_dim=value_feature_dim,
        )

    raise ValueError(
        f"Unknown value_head_type: {value_head_type}. "
        "Use affine, mlp, quadratic_latent, quadratic_latent_mumin, "
        "quadratic_bottleneck, quadratic_bottleneck_mumin, or squared_norm."
    )


def resolve_value_head_type(value_head_type: str | None, value_hidden: list[int]) -> str:
    if value_head_type is not None:
        return value_head_type
    if len(value_hidden) == 0:
        return "affine"
    return "mlp"


class VAECritic(nn.Module):
    """
    VAE-based critic that:
    1. Encodes observations to latent causal representation
    2. Predicts value from latent representation
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int = 32,
        encoder_hidden: list[int] = [256, 256],
        decoder_hidden: list[int] = [256, 256],
        value_hidden: list[int] = [128, 128],
        activation: str = "relu",
        beta: float = 1.0,
        value_head_type: str | None = None,
        value_mlp_hidden: int = 64,
        value_bottleneck_dim: int = 8,
        value_mu_min_floor: float = 0.02,
        value_feature_dim: int = 8,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.beta = beta
        self.value_head_type = resolve_value_head_type(value_head_type, value_hidden)

        act_module = activation_module(activation)

        encoder_layers = []
        input_dim = obs_dim
        for hidden_size in encoder_hidden:
            encoder_layers.append(nn.Linear(input_dim, hidden_size))
            encoder_layers.append(act_module)
            input_dim = hidden_size

        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(input_dim, latent_dim)
        self.fc_log_std = nn.Linear(input_dim, latent_dim)

        decoder_layers = []
        input_dim = latent_dim
        for hidden_size in decoder_hidden:
            decoder_layers.append(nn.Linear(input_dim, hidden_size))
            decoder_layers.append(act_module)
            input_dim = hidden_size
        decoder_layers.append(nn.Linear(input_dim, obs_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        self.value_head = build_vae_value_head(
            self.value_head_type,
            latent_dim,
            value_hidden,
            act_module,
            value_mlp_hidden,
            value_bottleneck_dim,
            value_mu_min_floor,
            value_feature_dim,
        )

        if self.value_head_type == "affine":
            nn.init.orthogonal_(self.value_head.weight, gain=0.01)
            nn.init.zeros_(self.value_head.bias)

    def encode(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        mu = self.fc_mu(h)
        log_std = self.fc_log_std(h)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mu, log_std

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, obs: torch.Tensor, return_latent: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        mu, log_std = self.encode(obs)
        z = self.reparameterize(mu, log_std)
        values = self.value_head(z)

        if return_latent:
            recon = self.decode(z)
            recon_loss = F.mse_loss(recon, obs, reduction="mean")
            kl_loss = -0.5 * torch.sum(
                1 + 2 * log_std - mu.pow(2) - log_std.exp().pow(2), dim=1
            ).mean()

            return values, {
                "mu": mu,
                "log_std": log_std,
                "z": z,
                "recon_loss": recon_loss,
                "kl_loss": kl_loss,
                "vae_loss": recon_loss + self.beta * kl_loss,
            }

        return values

    def get_latent_representation(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(obs)
        return mu
