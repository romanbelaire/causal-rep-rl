"""Shared helpers for MICo / DBC bisimulation losses."""

import copy

import torch
import torch.nn as nn


class VAEEncoderTarget(nn.Module):
    """Target copy of VAE encoder (mu only) for stable bisimulation targets."""

    def __init__(self, critic: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(critic.encoder)
        self.fc_mu = copy.deepcopy(critic.fc_mu)
        for param in self.parameters():
            param.requires_grad = False

    def encode_mu(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.encoder(obs)
        return self.fc_mu(h)

    @torch.no_grad()
    def soft_update_from(self, critic: nn.Module, tau: float) -> None:
        for target_param, param in zip(self.encoder.parameters(), critic.encoder.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
        for target_param, param in zip(self.fc_mu.parameters(), critic.fc_mu.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)


def encode_phi(
    critic: nn.Module,
    obs: torch.Tensor,
    repr_net: nn.Module | None = None,
    embed_ball_radius: float | None = None,
) -> torch.Tensor:
    """Online representation phi(s)."""
    if repr_net is not None:
        z = repr_net(obs)
    elif hasattr(critic, "encode"):
        mu, _ = critic.encode(obs)
        z = mu
    else:
        raise ValueError("Bisimulation loss requires repr_net or critic.encode()")

    return project_latent_ball(z, embed_ball_radius)


def encode_phi_target(
    encoder_target: VAEEncoderTarget,
    obs: torch.Tensor,
    embed_ball_radius: float | None = None,
) -> torch.Tensor:
    """Target representation phi_bar(s), stop-grad by caller."""
    z = encoder_target.encode_mu(obs)
    return project_latent_ball(z, embed_ball_radius)


def project_latent_ball(z: torch.Tensor, radius: float | None) -> torch.Tensor:
    """Project each row of z onto L2 ball of given radius."""
    if radius is None:
        return z
    norms = torch.norm(z, dim=1, keepdim=True)
    scale = torch.clamp(radius / norms, max=1.0)
    return z * scale


def permute_batch_pairs(batch_size: int, device: torch.device) -> torch.Tensor:
    """Random permutation for (x, y) pair sampling (DBC Algorithm 1)."""
    return torch.randperm(batch_size, device=device)


def reward_pair_dispersion(rewards_x: torch.Tensor, rewards_y: torch.Tensor) -> float:
    """Variance of |r_x - r_y| over batch pairs."""
    diff = torch.abs(rewards_x - rewards_y)
    return diff.var(unbiased=False).item()


def soft_update_module(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak average target <- (1-tau)*target + tau*source."""
    with torch.no_grad():
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
