"""
Causal Trust Region Optimization (CTRO) agent.

L_CTRO = L_PPO + alpha * L_MICo + beta * L_PL
"""

import torch
import torch.nn as nn

from src.agents.ppo import PPO
from src.losses.mico import compute_mico_loss, reward_dispersion
from src.losses.pl_coupling import compute_pl_coupling_loss
from src.utils.bisimulation_utils import VAEEncoderTarget, encode_phi


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

        # Target coefficients; effective values ramp via warmup_epochs (0 = always on).
        self.alpha = config.get("alpha", 0.0)
        self.beta = config.get("beta", 0.0)
        self.alpha_warmup_epochs = int(config.get("alpha_warmup_epochs", 0))
        self.beta_warmup_epochs = int(config.get("beta_warmup_epochs", 0))
        self.mu_0 = config.get("mu_0", 0.1)
        self.beta_mico = config.get("beta_mico", 0.1)
        self.pl_eps = config.get("pl_eps", 1e-4)
        self.mico_huber_delta = config.get("mico_huber_delta", 1.0)
        self.mico_target_update_tau = config.get("mico_target_update_tau", 0.005)
        self.mico_embed_ball_radius = config.get("mico_embed_ball_radius", None)

        self.encoder_target = None
        if self.alpha > 0:
            if not hasattr(self.critic, "encode"):
                raise ValueError("MICo loss requires critic with encode()")
            self.encoder_target = VAEEncoderTarget(self.critic).to(device)

    def _encode_batch(self, batch_obs: torch.Tensor) -> torch.Tensor:
        if self.repr_net is not None:
            return self.repr_net(batch_obs)
        return encode_phi(
            self.critic,
            batch_obs,
            embed_ball_radius=self.mico_embed_ball_radius,
        )

    @staticmethod
    def _warmup_scale(warmup_epochs: int, training_epoch: int | None) -> float:
        if warmup_epochs <= 0:
            return 1.0
        if training_epoch is None:
            raise ValueError("training_epoch required when alpha/beta warmup_epochs > 0")
        return min(1.0, float(training_epoch) / float(warmup_epochs))

    @property
    def needs_transition_batch(self) -> bool:
        # Use target coeffs so transition batches are available once warmup engages.
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
        epoch = self._training_epoch
        alpha_eff = self.alpha * self._warmup_scale(self.alpha_warmup_epochs, epoch)
        beta_eff = self.beta * self._warmup_scale(self.beta_warmup_epochs, epoch)
        stats["alpha_eff"] = alpha_eff
        stats["beta_eff"] = beta_eff

        if batch_rewards is not None:
            stats["reward_dispersion"] = reward_dispersion(batch_rewards)

        if alpha_eff > 0:
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
            extra = extra + alpha_eff * mico_raw
            stats.update(mico_stats)

        if beta_eff > 0:
            pl_raw, pl_stats = compute_pl_coupling_loss(
                self.critic,
                z,
                batch_rewards,
                batch_next_obs,
                self.gamma,
                mu_0=self.mu_0,
                eps=self.pl_eps,
            )
            extra = extra + beta_eff * pl_raw
            stats.update(pl_stats)

        return extra, stats

    def _after_optimizer_step(self) -> None:
        if self.encoder_target is not None:
            self.encoder_target.soft_update_from(self.critic, self.mico_target_update_tau)

    def checkpoint_dict(self) -> dict:
        save_dict = super().checkpoint_dict()
        if self.encoder_target is not None:
            save_dict["encoder_target"] = self.encoder_target.state_dict()
        return save_dict

    def _load_checkpoint(self, checkpoint: dict, weights_only: bool = False) -> None:
        super()._load_checkpoint(checkpoint, weights_only=weights_only)
        if self.encoder_target is not None:
            self.encoder_target.load_state_dict(checkpoint["encoder_target"])
