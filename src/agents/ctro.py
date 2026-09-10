"""
Causal Trust Region Optimization (CTRO) agent.

Legacy: L_CTRO = L_PPO + alpha * L_MICo + beta * L_PL + lambda_dz * D_Z

Live anti-aliased PPO (preset anti_aliased_ppo): L = L_PPO + beta * L_PL
with scale-corrected hinge on a frozen reference batch; alpha=0, no D_Z.
See docs/METHOD_CHANGELOG.md.
"""

import torch
import torch.nn as nn

from src.agents.ppo import PPO
from src.losses.dz_trust_region import (
    DEFAULT_DZ_DELTA,
    adaptive_eta_dz,
    adaptive_lambda_dz,
    compute_dz,
    compute_s_ref,
    mean_nearest_buffer_distance,
    sample_pair_index_mask,
    update_dz_ema,
)
from src.losses.mico import compute_mico_loss, reward_dispersion
from src.losses.pl_coupling import compute_pl_coupling_loss
from src.utils.bisimulation_utils import VAEEncoderTarget, encode_phi, project_latent_ball


class CTRO(PPO):
    """PPO extended with additive MICo, PL coupling, and optional D_Z trust region."""

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
        self.pl_eps = config.get("pl_eps", 1e-4)  # legacy; PL uses f_floor
        self.pl_scale_invariant = bool(config.get("pl_scale_invariant", False))
        self.pl_value_normalize = bool(config.get("pl_value_normalize", False))
        self.pl_f_mode = config.get("pl_f_mode", "exclude")
        self.pl_f_min = float(config.get("pl_f_min", 1e-3))
        self.pl_on_ref_buffer = bool(config.get("pl_on_ref_buffer", False))
        self.mico_huber_delta = config.get("mico_huber_delta", 1.0)
        self.mico_target_update_tau = config.get("mico_target_update_tau", 0.005)
        self.mico_embed_ball_radius = config.get("mico_embed_ball_radius", None)
        if config.get("value_spectral_norm", False):
            from src.utils.spectral_value_head import apply_value_head_spectral_norm

            n_sn = apply_value_head_spectral_norm(self.critic)
            print(f"Applied spectral_norm to {n_sn} value_head Linear layer(s)", flush=True)

        if self.pl_on_ref_buffer:
            if self.alpha != 0.0:
                raise RuntimeError(
                    "pl_on_ref_buffer (anti-aliased PPO) requires alpha=0; "
                    "use --algo-preset ctro_full_legacy for MICo"
                )
            if config.get("value_spectral_norm", False):
                raise RuntimeError(
                    "pl_on_ref_buffer (anti-aliased PPO) forbids value_spectral_norm"
                )
            if self.pl_f_mode != "clamp":
                raise RuntimeError("pl_on_ref_buffer requires pl_f_mode='clamp'")
            if not self.head_phasing:
                raise RuntimeError("pl_on_ref_buffer requires head_phasing=True")
            if config.get("dz_enabled", False):
                raise RuntimeError(
                    "pl_on_ref_buffer (anti-aliased PPO) requires dz_enabled=False"
                )

        # D_Z reference-scaled log-geometry trust region
        self.dz_enabled = config.get("dz_enabled", False)
        self.lambda_dz = float(config.get("lambda_dz", 1.0))
        self.eta_dz = float(config.get("eta_dz", 0.05))
        self.dz_adapt = bool(config.get("dz_adapt", True))
        # Adaptive radius: η_t = max(η_min, c * D̄_Z^(t-1)); never below η_min.
        self.dz_eta_adapt = bool(config.get("dz_eta_adapt", False))
        _eta_min = config.get("eta_dz_min", None)
        self.eta_dz_min = float(self.eta_dz if _eta_min is None else _eta_min)
        self.eta_dz_c = float(config.get("eta_dz_c", 1.5))
        self.dz_ema_tau = float(config.get("dz_ema_tau", 0.05))
        # Seed EMA at formation-scale D_Z so the first η is permissive; contracts to η_min.
        self.dz_ema_init = float(config.get("dz_ema_init", 0.5))
        self._dz_ema: float | None = None
        self.lambda_loc = float(config.get("lambda_loc", 0.7))
        self.dz_delta = float(config.get("dz_delta", DEFAULT_DZ_DELTA))
        self.dz_collapse_target_thresh = float(
            config.get("dz_collapse_target_thresh", 0.1)
        )
        self.dz_n_pairs = int(config.get("dz_n_pairs", 2048))
        self.ref_n = int(config.get("ref_buffer_size", 2048))
        self.ref_refresh_factor = float(config.get("ref_refresh_factor", 2.0))
        self._initial_buffer_distance: float | None = None
        self._pending_refresh = False
        # Frozen target latents for the current PPO update (rollout obs order).
        self._dz_z_old: torch.Tensor | None = None
        self._dz_z_next_old: torch.Tensor | None = None
        # Frozen reference scale; set once on first ref-buffer fill.
        self.s_ref: float | None = None

        self.encoder_target = None
        need_target = self.alpha > 0 or self.dz_enabled
        if need_target:
            if not hasattr(self.critic, "encode"):
                raise ValueError("MICo/D_Z require critic with encode()")
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
        # MICo needs (s, r, s'); D_Z collapse split needs rewards / next_obs.
        return self.alpha > 0 or self.dz_enabled

    def maybe_init_or_refresh_ref_buffer(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor | None,
        rewards: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """Fill frozen ref buffer once; refresh if on-policy→buffer distance grows.

        When pl_on_ref_buffer is set, the buffer is owned by the PL hinge and is
        never refreshed (see maybe_init_frozen_pl_ref_buffer on PPO.update).
        """
        stats: dict[str, float] = {"ref_buffer_refresh": 0.0}
        if self.pl_on_ref_buffer:
            return stats
        if not self.dz_enabled:
            return stats

        def _fill(*, init_s_ref: bool) -> None:
            if rewards is None or next_obs is None:
                raise RuntimeError(
                    "D_Z ref buffer requires rewards and next_obs "
                    "(MICo collapse split + pair targets)"
                )
            n = min(self.ref_n, obs.shape[0])
            self.set_reference_buffer(obs[:n], next_obs[:n], rewards[:n])
            if self.encoder_target is not None:
                self.encoder_target = VAEEncoderTarget(self.critic).to(self.device)
            self._initial_buffer_distance = None
            if init_s_ref:
                if self.s_ref is not None:
                    raise RuntimeError("s_ref already set; must freeze only once")
                with torch.no_grad():
                    z_ref = self._project_latent(
                        self.encoder_target.encode_mu(self.ref_obs)
                    )
                    pair_i, pair_j = sample_pair_index_mask(
                        z_ref.shape[0], self.dz_n_pairs, self.lambda_loc, self.device
                    )
                    self.s_ref = compute_s_ref(z_ref, pair_i, pair_j)
                stats["dz_s_ref"] = self.s_ref
            stats["ref_buffer_refresh"] = 1.0

        if self.ref_obs is None:
            _fill(init_s_ref=True)
            self._pending_refresh = False
            return stats
        if self._pending_refresh:
            _fill(init_s_ref=False)
            self._pending_refresh = False
            return stats

        with torch.no_grad():
            z_on = self._encode_batch(obs)
            z_buf = self._encode_batch(self.ref_obs)
            dist = mean_nearest_buffer_distance(z_on, z_buf)
        stats["on_policy_to_buffer_dist"] = dist
        if self._initial_buffer_distance is None:
            self._initial_buffer_distance = dist
        elif dist > self.ref_refresh_factor * max(self._initial_buffer_distance, 1e-8):
            _fill(init_s_ref=False)
            stats["on_policy_to_buffer_dist"] = dist
        return stats

    def _project_latent(self, z: torch.Tensor) -> torch.Tensor:
        if self.mico_embed_ball_radius is None:
            return z
        return project_latent_ball(z, self.mico_embed_ball_radius)

    def _cache_dz_latents(self, obs: torch.Tensor, next_obs: torch.Tensor) -> None:
        """Encode full rollout once with frozen target; index into it each minibatch."""
        if self.encoder_target is None:
            raise RuntimeError("D_Z requires encoder_target")
        with torch.no_grad():
            z_old = self._project_latent(
                self.encoder_target.encode_mu(obs.to(self.device))
            )
            z_next_old = self._project_latent(
                self.encoder_target.encode_mu(next_obs.to(self.device))
            )
            self._dz_z_old = z_old.detach()
            self._dz_z_next_old = z_next_old.detach()

    def _compute_dz_term(
        self,
        z_new: torch.Tensor,
        batch_indices: torch.Tensor,
        batch_rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        if self._dz_z_old is None or self._dz_z_next_old is None:
            raise RuntimeError("D_Z enabled but target latents were not cached")

        if self.s_ref is None:
            raise RuntimeError("D_Z enabled but s_ref was not initialized")
        z_old = self._dz_z_old[batch_indices]
        z_next_old = self._dz_z_next_old[batch_indices]
        n = z_new.shape[0]
        pair_i, pair_j = sample_pair_index_mask(
            n, self.dz_n_pairs, self.lambda_loc, self.device
        )
        dz, dz_stats = compute_dz(
            z_new,
            z_old,
            pair_i,
            pair_j,
            s_ref=self.s_ref,
            delta=self.dz_delta,
            rewards=batch_rewards,
            z_next_old=z_next_old,
            gamma=self.gamma,
            collapse_target_thresh=self.dz_collapse_target_thresh,
        )
        dz_stats["lambda_dz"] = self.lambda_dz
        dz_stats["eta_dz"] = self.eta_dz
        dz_stats["dz_eta_sq"] = self.eta_dz * self.eta_dz
        return dz, dz_stats

    def _extra_critic_terms(
        self,
        batch_obs: torch.Tensor,
        z: torch.Tensor,
        batch_rewards: torch.Tensor | None,
        batch_next_obs: torch.Tensor | None,
        phase: str = "joint",
        batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        extra = torch.tensor(0.0, device=self.device)
        stats: dict = {}
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
            if self.v_ref is None:
                raise RuntimeError("PL loss requires v_ref to be set before update")
            if self.pl_on_ref_buffer:
                if self.ref_obs is None:
                    raise RuntimeError(
                        "pl_on_ref_buffer=True but ref_obs is unset; "
                        "frozen PL buffer must be initialized before the hinge"
                    )
                z_pl = self._encode_batch(self.ref_obs)
            else:
                z_pl = z
            pl_raw, pl_stats = compute_pl_coupling_loss(
                self.critic,
                z_pl,
                v_ref=self.v_ref,
                mu_0=self.mu_0,
                f_floor=self.f_floor,
                scale_invariant=self.pl_scale_invariant,
                value_normalize=self.pl_value_normalize,
                f_mode=self.pl_f_mode,
                f_min=self.pl_f_min,
            )
            extra = extra + beta_eff * pl_raw
            stats.update(pl_stats)

        if self.dz_enabled and phase in ("joint", "enc"):
            if batch_indices is None:
                raise RuntimeError("D_Z requires batch_indices into cached z_old")
            if batch_rewards is None:
                raise RuntimeError("D_Z requires batch_rewards for collapse split")
            dz_raw, dz_stats = self._compute_dz_term(z, batch_indices, batch_rewards)
            # Additive term: λ (D_Z^log − η²), λ ≥ 0. Not a hinge.
            # With fixed λ (dz_adapt=False), −η² is constant → no gradient through η;
            # the optimized geometry contribution is λ D_Z (LTRO-fixed freeze: λ=1).
            extra = extra + self.lambda_dz * (dz_raw - self.eta_dz * self.eta_dz)
            stats.update(dz_stats)

        return extra, stats

    def _after_optimizer_step(self, phase: str = "joint") -> None:
        if self.encoder_target is not None and phase in ("joint", "enc"):
            self.encoder_target.soft_update_from(self.critic, self.mico_target_update_tau)

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        rewards: torch.Tensor | None = None,
        next_obs: torch.Tensor | None = None,
        training_epoch: int | None = None,
    ) -> dict:
        buf_stats = self.maybe_init_or_refresh_ref_buffer(obs, next_obs, rewards)
        if self.dz_enabled:
            if next_obs is None:
                raise RuntimeError("D_Z requires next_obs to cache target latents")
            self._cache_dz_latents(obs, next_obs)
            if self.dz_eta_adapt:
                if self._dz_ema is None:
                    self._dz_ema = self.dz_ema_init
                # Pre-update EMA only — no within-step feedback into η_t.
                self.eta_dz = adaptive_eta_dz(
                    self._dz_ema, self.eta_dz_min, c=self.eta_dz_c
                )
        try:
            stats = super().update(
                obs,
                actions,
                old_log_probs,
                advantages,
                returns,
                rewards=rewards,
                next_obs=next_obs,
                training_epoch=training_epoch,
            )
        finally:
            self._dz_z_old = None
            self._dz_z_next_old = None

        if self.dz_enabled and "D_Z" in stats:
            dz_val = float(stats["D_Z"])
            if self.dz_eta_adapt:
                self._dz_ema = update_dz_ema(self._dz_ema, dz_val, self.dz_ema_tau)
                stats["dz_ema"] = self._dz_ema
            if self.dz_adapt:
                self.lambda_dz = adaptive_lambda_dz(
                    self.lambda_dz, dz_val, self.eta_dz
                )
                stats["lambda_dz"] = self.lambda_dz
            stats["eta_dz"] = self.eta_dz
        stats.update(buf_stats)
        return stats

    def checkpoint_dict(
        self,
        training_epoch: int | None = None,
        total_env_steps: int | None = None,
    ) -> dict:
        save_dict = super().checkpoint_dict(training_epoch, total_env_steps)
        if self.encoder_target is not None:
            save_dict["encoder_target"] = self.encoder_target.state_dict()
        save_dict["lambda_dz"] = self.lambda_dz
        save_dict["eta_dz"] = self.eta_dz
        if self.s_ref is not None:
            save_dict["s_ref"] = self.s_ref
        if self._dz_ema is not None:
            save_dict["dz_ema"] = self._dz_ema
        return save_dict

    def _load_checkpoint(self, checkpoint: dict, weights_only: bool = False) -> None:
        super()._load_checkpoint(checkpoint, weights_only=weights_only)
        if self.encoder_target is not None and "encoder_target" in checkpoint:
            self.encoder_target.load_state_dict(checkpoint["encoder_target"])
        if "lambda_dz" in checkpoint:
            self.lambda_dz = float(checkpoint["lambda_dz"])
        if "eta_dz" in checkpoint:
            self.eta_dz = float(checkpoint["eta_dz"])
        if "s_ref" in checkpoint:
            self.s_ref = float(checkpoint["s_ref"])
        if "dz_ema" in checkpoint:
            self._dz_ema = float(checkpoint["dz_ema"])
