"""VAE critic with Impala CNN encoder for Procgen CTRO."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.activation import activation_module
from src.architectures.critics.vae_critic import build_vae_value_head, resolve_value_head_type
from src.architectures.encoders.impala_cnn import ProcgenImpalaEncoder


class CNNVAEDecoder(nn.Module):
    def __init__(self, latent_dim: int, obs_shape: tuple[int, int, int], activation: str = "relu"):
        super().__init__()
        self.obs_shape = obs_shape
        h, w, _ = obs_shape
        self.activation = activation_module(activation)

        self.fc = nn.Linear(latent_dim, 32 * 8 * 8)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            self.activation,
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            self.activation,
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
        )

        with torch.no_grad():
            dummy = self.deconvs(self.activation(self.fc(torch.zeros(1, latent_dim))).view(1, 32, 8, 8))
            self.out_h, self.out_w = dummy.shape[2], dummy.shape[3]

        self.adapt = None
        if (self.out_h, self.out_w) != (h, w):
            self.adapt = nn.Upsample(size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc(z)).view(z.shape[0], 32, 8, 8)
        x = self.deconvs(x)
        if self.adapt is not None:
            x = self.adapt(x)
        return x.permute(0, 2, 3, 1)


class CNNVAECritic(nn.Module):
    """CNN encoder + VAE latent + pixel decoder + value head."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        latent_dim: int = 128,
        emb_size: int = 256,
        depths: tuple[int, ...] = (16, 32, 32),
        activation: str = "gelu",
        beta: float = 1.0,
        value_hidden: list[int] | None = None,
        value_head_type: str | None = None,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.latent_dim = latent_dim
        self.beta = beta
        value_hidden = value_hidden or [128, 128]
        self.value_head_type = resolve_value_head_type(value_head_type, value_hidden)

        self.encoder = ProcgenImpalaEncoder(obs_shape, depths=depths, emb_size=emb_size)
        self.fc_mu = nn.Linear(emb_size, latent_dim)
        self.fc_log_std = nn.Linear(emb_size, latent_dim)
        self.decoder = CNNVAEDecoder(latent_dim, obs_shape, activation=activation)
        act_module = activation_module(activation)
        self.value_head = build_vae_value_head(
            self.value_head_type,
            latent_dim,
            value_hidden,
            act_module,
            value_mlp_hidden=64,
            value_bottleneck_dim=8,
            value_mu_min_floor=0.02,
            value_feature_dim=8,
        )

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
        batched = obs.dim() == 4
        if not batched:
            obs = obs.unsqueeze(0)
        mu, log_std = self.encode(obs)
        z = self.reparameterize(mu, log_std)
        values = self.value_head(z)
        if not return_latent:
            return values.squeeze(0) if not batched else values
        recon = self.decode(z)
        recon_loss = F.mse_loss(recon, obs, reduction="mean")
        kl_loss = -0.5 * torch.sum(
            1 + 2 * log_std - mu.pow(2) - log_std.exp().pow(2), dim=1
        ).mean()
        payload = {
            "mu": mu,
            "log_std": log_std,
            "z": z,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "vae_loss": recon_loss + self.beta * kl_loss,
        }
        if not batched:
            values = values.squeeze(0)
        return values, payload

    def get_latent_representation(self, obs: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(obs)
        return mu
