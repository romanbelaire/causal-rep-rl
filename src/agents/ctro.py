"""
Causal Trust Region Optimization (CTRO) agent.

L_CTRO = L_PPO + alpha * L_MICo + beta * L_PL
"""

import torch
import torch.nn as nn

from src.agents.ppo import PPO
from src.losses.mico import compute_mico_loss, reward_dispersion
from src.losses.pl_coupling import compute_pl_coupling_loss
from src.utils.bisimulation_utils import VAEEncoderTarget


class CTRO(PPO):
    """PPO extended with additive MICo and PL coupling losses."""

    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config: dict,
        device: str = "cuda",
        repr_net: nn.Module = None,
    ):
        super().__init__(policy, critic, config, device, repr_net=repr_net)

        self.alpha = config.get("alpha", 0.0)
        self.beta = config.get("beta", 0.0)
        self.mu_0 = config.get("mu_0", 0.1)
        self.beta_mico = config.get("beta_mico", 0.1)
        self.pl_eps = config.get("pl_eps", 1e-4)
        self.mico_huber_delta = config.get("mico_huber_delta", 1.0)
        self.mico_target_update_tau = config.get("mico_target_update_tau", 0.005)
        self.mico_embed_ball_radius = config.get("mico_embed_ball_radius", None)

        self.encoder_target = None
        if self.alpha > 0:
            if not hasattr(self.critic, "encode"):
                raise ValueError("MICo loss requires VAE critic with encode()")
            self.encoder_target = VAEEncoderTarget(self.critic).to(device)

    @property
    def needs_transition_batch(self) -> bool:
        return self.alpha > 0 or self.beta > 0

    def _extra_critic_terms(
        self,
        batch_obs: torch.Tensor,
        z: torch.Tensor,
        batch_rewards: torch.Tensor | None,
        batch_next_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        extra = torch.tensor(0.0, device=self.device)
        stats: dict[str, float] = {}

        if batch_rewards is not None:
            stats["reward_dispersion"] = reward_dispersion(batch_rewards)

        if self.alpha > 0:
            mico_raw, mico_stats = compute_mico_loss(
                self.critic,
                self.encoder_target,
                batch_obs,
                batch_next_obs,
                batch_rewards,
                self.gamma,
                beta=self.beta_mico,
                huber_delta=self.mico_huber_delta,
                embed_ball_radius=self.mico_embed_ball_radius,
                repr_net=self.repr_net,
            )
            extra = extra + self.alpha * mico_raw
            stats.update(mico_stats)

        if self.beta > 0:
            pl_raw, pl_stats = compute_pl_coupling_loss(
                self.critic,
                z,
                batch_rewards,
                batch_next_obs,
                self.gamma,
                mu_0=self.mu_0,
                eps=self.pl_eps,
            )
            extra = extra + self.beta * pl_raw
            stats.update(pl_stats)

        return extra, stats

    def _after_optimizer_step(self) -> None:
        if self.encoder_target is not None:
            self.encoder_target.soft_update_from(self.critic, self.mico_target_update_tau)

    def save(self, path: str) -> None:
        import torch

        save_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "unified_optimizer": self.unified_optimizer.state_dict(),
        }
        if self.repr_net is not None:
            save_dict["repr_net"] = self.repr_net.state_dict()
        if self.encoder_target is not None:
            save_dict["encoder_target"] = self.encoder_target.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str, weights_only: bool = False) -> None:
        import torch

        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        if not weights_only:
            self.unified_optimizer.load_state_dict(checkpoint["unified_optimizer"])
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])
        if self.encoder_target is not None and "encoder_target" in checkpoint:
            self.encoder_target.load_state_dict(checkpoint["encoder_target"])
