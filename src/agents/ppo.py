"""
Proximal Policy Optimization (PPO) — vanilla policy + value + VAE.
"""

import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.metrics.pl_ratio import target_landscape_pl_stats
from src.utils.ownership import delta_l2, grad_l2, param_l2
from src.utils.target_network import TargetNetwork


def _mean_stat(vals: list) -> float:
    """Average minibatch stats; tensors stay on device until a single sync here."""
    first = vals[0]
    if torch.is_tensor(first):
        return float(torch.stack(vals).mean().item())
    return sum(vals) / len(vals)


class PPO:
    """PPO with clipped surrogate, GAE value loss, and optional VAE reconstruction."""

    def __init__(
        self,
        policy: nn.Module,
        critic: nn.Module,
        config: dict,
        device: str = "cuda",
        repr_net: nn.Module = None,
    ):
        self.policy = policy.to(device)
        self.critic = critic.to(device)
        self.repr_net = repr_net.to(device) if repr_net is not None else None
        self.device = device

        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.use_policy_clip = config.get("use_policy_clip", True)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        if not self.use_policy_clip and self.clip_epsilon is not None:
            # Explicit: clip is off, not "large ε".
            pass
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vae_coef = config.get("vae_coef", 0.1)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        self.policy_on_latent = config.get("policy_on_latent", True)

        # Head phasing: encoder+policy epochs then value-head epochs.
        self.head_phasing = config.get("head_phasing", False)
        self.enc_epochs = config.get("enc_epochs", 4)
        self.val_epochs = config.get("val_epochs", 4)

        actor_params = list(self.policy.parameters())
        repr_params = list(self.critic.parameters())
        if self.repr_net is not None:
            repr_params = list(self.repr_net.parameters()) + repr_params
        self.actor_optimizer = optim.Adam(actor_params, lr=self.lr)
        self.representation_optimizer = optim.Adam(repr_params, lr=self.lr)
        self.actor_updates_encoder = bool(config.get("actor_updates_encoder", True))
        target_kl = config.get("target_kl", None)
        self.target_kl = None if target_kl is None else float(target_kl)
        self.policy_version = 0
        self.encoder_version = 0

        self.old_policy = None
        self._step = 0
        self._training_epoch = None
        self.v_ref: float | None = None
        self.tau_ref = config.get("tau_ref", 0.01)
        self.f_floor = config.get("f_floor", 1e-3)
        self.v_ref_quantile = config.get("v_ref_quantile", 0.99)
        self.pl_f_mode = config.get("pl_f_mode", "exclude")
        self.pl_f_min = float(config.get("pl_f_min", 1e-3))
        self.pl_on_ref_buffer = bool(config.get("pl_on_ref_buffer", False))
        self.ref_n = int(config.get("ref_buffer_size", 2048))
        self.pfo_coef = float(config.get("pfo_coef", 0.0))
        self._pfo_old_policy: nn.Module | None = None
        self.anneal_lr = bool(config.get("anneal_lr", False))
        self.norm_adv = bool(config.get("norm_adv", True))
        self.total_update_epochs = int(config.get("total_update_epochs", 0))
        self._lr_update_count = 0

        # Reference buffer for D_Z / PL frozen probe / geometry (optional)
        self.ref_obs: torch.Tensor | None = None
        self.ref_obs_next: torch.Tensor | None = None
        self.ref_rewards: torch.Tensor | None = None
        self.ref_buffer_refresh_count = 0
        self._pl_ref_frozen = False
        self.ref_freeze_epoch = 0
        self.pl_target_critic = TargetNetwork(self.critic)
        self.v_ref_target: float | None = None
        self.own_nan_count = 0
        self.own_rollback_count = 0

    @property
    def needs_transition_batch(self) -> bool:
        return False

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0,
        terminations: torch.Tensor | None = None,
        next_values: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """GAE with Gymnasium-style terminated vs truncated handling.

        - `dones` = terminated | truncated (resets GAE carry across episodes)
        - `terminations` = true episode failures only (no value bootstrap)
        - If `next_values` is provided (V(s_{t+1}) per step), use those for bootstraps;
          otherwise fall back to the reverse `next_value` chain (legacy Minigrid path).
        """
        if terminations is None:
            terminations = dones

        advantages = torch.zeros_like(rewards)
        last_gae = 0.0
        T = len(rewards)

        if next_values is not None:
            for t in reversed(range(T)):
                non_terminal = (~terminations[t]).float()
                non_done = (~dones[t]).float()
                delta = rewards[t] + self.gamma * next_values[t] * non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * non_done * last_gae
                advantages[t] = last_gae
        else:
            for t in reversed(range(T)):
                non_terminal = (~terminations[t]).float()
                non_done = (~dones[t]).float()
                delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * non_done * last_gae
                advantages[t] = last_gae
                next_value = values[t]

        returns = advantages + values
        return advantages, returns

    def _encode_batch(self, batch_obs: torch.Tensor) -> torch.Tensor:
        if self.repr_net is not None:
            return self.repr_net(batch_obs)
        if hasattr(self.critic, "encode"):
            mu, _ = self.critic.encode(batch_obs)
            return mu
        return batch_obs

    def _encoder_params(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        if self.repr_net is not None:
            params.extend(list(self.repr_net.parameters()))
        if hasattr(self.critic, "encoder"):
            params.extend(list(self.critic.encoder.parameters()))
        if hasattr(self.critic, "fc_mu") and self.critic.fc_mu is not None:
            params.extend(list(self.critic.fc_mu.parameters()))
        if hasattr(self.critic, "fc_z") and self.critic.fc_z is not None:
            params.extend(list(self.critic.fc_z.parameters()))
        if hasattr(self.critic, "fc_log_std") and self.critic.fc_log_std is not None:
            params.extend(list(self.critic.fc_log_std.parameters()))
        return params

    def _value_head_params(self) -> list[nn.Parameter]:
        return list(self.critic.value_head.parameters())

    def _set_phase_requires_grad(self, phase: str) -> None:
        """phase in {'enc', 'val', 'joint'}."""
        enc_params = self._encoder_params()
        val_params = self._value_head_params()
        pol_params = list(self.policy.parameters())

        if phase == "joint":
            for p in enc_params + val_params + pol_params:
                p.requires_grad = True
            return
        if phase == "enc":
            for p in val_params:
                p.requires_grad = False
            for p in enc_params + pol_params:
                p.requires_grad = True
            return
        if phase == "val":
            for p in enc_params + pol_params:
                p.requires_grad = False
            for p in val_params:
                p.requires_grad = True
            return
        raise ValueError(f"Unknown phase {phase}")

    def _assert_frozen_grads(self, phase: str) -> None:
        if phase == "enc":
            frozen = self._value_head_params()
            name = "value_head"
        elif phase == "val":
            frozen = self._encoder_params() + list(self.policy.parameters())
            name = "encoder/policy"
        else:
            return
        for p in frozen:
            if p.grad is not None and p.grad.abs().sum().item() != 0.0:
                raise RuntimeError(
                    f"Head phasing phase={phase}: non-zero grad leaked into frozen {name}"
                )

    def _forward_batch(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_returns: torch.Tensor,
        compute_policy: bool = True,
        compute_value: bool = True,
    ) -> dict:
        z = self._encode_batch(batch_obs)
        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"Non-finite latent z in PPO update (nan={torch.isnan(z).any().item()}, "
                f"inf={torch.isinf(z).any().item()})"
            )
        policy_in = z if self.policy_on_latent else batch_obs
        if compute_policy and not self.actor_updates_encoder:
            policy_in = policy_in.detach()

        if compute_policy:
            log_probs, entropy = self.policy.evaluate_actions(policy_in, batch_actions)
            if not torch.isfinite(log_probs).all():
                raise RuntimeError("Non-finite policy log_probs in PPO update")

            # Clamp log-ratio before exp so negative-advantage / huge-ratio cases
            # cannot produce Inf surrogates (common continuous-PPO NaN path).
            log_ratio = torch.clamp(log_probs - batch_old_log_probs, min=-20.0, max=2.0)
            ratio = torch.exp(log_ratio)
            surr1 = ratio * batch_advantages
            if self.use_policy_clip:
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon)
                    * batch_advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                clip_frac = (
                    (ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)
                ).float().mean()
            else:
                policy_loss = -surr1.mean()
                clip_frac = torch.tensor(0.0, device=self.device)
            entropy_loss = -entropy.mean()
        else:
            log_probs = batch_old_log_probs
            policy_loss = torch.tensor(0.0, device=self.device)
            entropy_loss = torch.tensor(0.0, device=self.device)
            clip_frac = torch.tensor(0.0, device=self.device)

        vae_loss = torch.tensor(0.0, device=self.device)
        recon_loss = torch.tensor(0.0, device=self.device)
        kl_loss = torch.tensor(0.0, device=self.device)

        if compute_value:
            if hasattr(self.critic, "encode") and self.vae_coef > 0:
                values, vae_info = self.critic(batch_obs, return_latent=True)
                values = values.squeeze(-1)
                recon_loss = vae_info["recon_loss"]
                kl_loss = vae_info["kl_loss"]
                vae_loss = vae_info["vae_loss"]
            else:
                values = self.critic(batch_obs).squeeze(-1)
            value_loss = nn.functional.mse_loss(values, batch_returns)
        else:
            value_loss = torch.tensor(0.0, device=self.device)

        return {
            "z": z,
            "log_probs": log_probs,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "clip_frac": clip_frac,
        }

    def _extra_critic_terms(
        self,
        batch_obs: torch.Tensor,
        z: torch.Tensor,
        batch_rewards: torch.Tensor | None,
        batch_next_obs: torch.Tensor | None,
        phase: str = "joint",
        batch_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        return torch.tensor(0.0, device=self.device), {}

    def _after_optimizer_step(self, phase: str = "joint") -> None:
        pass

    def set_reference_buffer(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor | None = None,
        rewards: torch.Tensor | None = None,
    ) -> None:
        self.ref_obs = obs.detach().to(self.device)
        self.ref_obs_next = None if next_obs is None else next_obs.detach().to(self.device)
        self.ref_rewards = None if rewards is None else rewards.detach().to(self.device)
        self.ref_buffer_refresh_count += 1

    def maybe_init_frozen_pl_ref_buffer(self, obs: torch.Tensor) -> dict[str, float]:
        """Sample N states once for the PL hinge; never resample when frozen."""
        stats = {"pl_ref_buffer_init": 0.0}
        if not self.pl_on_ref_buffer:
            return stats
        if self._pl_ref_frozen and self.ref_obs is not None:
            return stats
        n = min(self.ref_n, obs.shape[0])
        if n < 2:
            raise RuntimeError(
                f"Frozen PL ref buffer needs at least 2 states (got n={n})"
            )
        self.set_reference_buffer(obs[:n])
        self._pl_ref_frozen = True
        if self._training_epoch is not None:
            self.ref_freeze_epoch = int(self._training_epoch)
        stats["pl_ref_buffer_init"] = 1.0
        stats["pl_ref_buffer_n"] = float(n)
        return stats

    def update_v_ref_from_obs(self, obs: torch.Tensor) -> float:
        from src.metrics.pl_ratio import update_v_ref

        with torch.no_grad():
            z = self._encode_batch(obs)
            v = self.critic.value_head(z).squeeze(-1)
            self.v_ref = update_v_ref(
                self.v_ref, v, tau_ref=self.tau_ref, quantile=self.v_ref_quantile
            )
        return self.v_ref

    def _run_minibatch(
        self,
        batch,
        phase: str,
        extra_stats_acc: dict[str, list],
        totals: dict[str, float],
    ) -> None:
        transition = self.needs_transition_batch
        if transition:
            (
                batch_obs,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
                batch_rewards,
                batch_next_obs,
                batch_indices,
            ) = batch
            batch_rewards = batch_rewards.to(self.device)
            batch_next_obs = batch_next_obs.to(self.device)
        else:
            (
                batch_obs,
                batch_actions,
                batch_old_log_probs,
                batch_advantages,
                batch_returns,
                batch_indices,
            ) = batch
            batch_rewards = None
            batch_next_obs = None

        batch_obs = batch_obs.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_old_log_probs = batch_old_log_probs.to(self.device)
        batch_advantages = batch_advantages.to(self.device)
        batch_returns = batch_returns.to(self.device)
        batch_indices = batch_indices.to(self.device)

        compute_policy = phase in ("joint", "enc")
        compute_value = phase in ("joint", "val")
        # Value loss still needs forward for GAE targets in enc phase only if we
        # are not updating value — skip value loss in enc phase under phasing.
        if self.head_phasing and phase == "enc":
            compute_value = False
        if self.head_phasing and phase == "val":
            compute_policy = False

        fwd = self._forward_batch(
            batch_obs,
            batch_actions,
            batch_old_log_probs,
            batch_advantages,
            batch_returns,
            compute_policy=compute_policy,
            compute_value=compute_value,
        )

        # D_Z / MICo / PL only when encoder is allowed to move (or joint).
        extra_loss = torch.tensor(0.0, device=self.device)
        extra_stats: dict = {}
        if phase in ("joint", "enc"):
            extra_loss, extra_stats = self._extra_critic_terms(
                batch_obs,
                fwd["z"],
                batch_rewards,
                batch_next_obs,
                phase=phase,
                batch_indices=batch_indices,
            )

        critic_loss = (
            self.value_coef * fwd["value_loss"]
            + self.vae_coef * fwd["vae_loss"]
            + extra_loss
        )

        pfo_loss = torch.tensor(0.0, device=self.device)
        if self.pfo_coef > 0.0 and compute_policy:
            if self._pfo_old_policy is None:
                raise RuntimeError("pfo_coef>0 requires PFO snapshot before minibatches")
            from src.losses.pfo import compute_pfo_loss

            pfo_loss = compute_pfo_loss(self.policy, self._pfo_old_policy, batch_obs)
            extra_stats["pfo_loss"] = pfo_loss.detach()

        self.actor_optimizer.zero_grad()
        self.representation_optimizer.zero_grad()
        policy_total = (
            fwd["policy_loss"]
            + self.entropy_coef * fwd["entropy_loss"]
            + self.pfo_coef * pfo_loss
        )
        total = policy_total + critic_loss
        total.backward()
        if self.head_phasing and phase in ("enc", "val"):
            self._assert_frozen_grads(phase)
        actor_params = list(self.policy.parameters())
        repr_params = list(self.critic.parameters())
        if self.repr_net is not None:
            repr_params = list(self.repr_net.parameters()) + repr_params
        enc_params = self._encoder_params()
        val_params = self._value_head_params()
        mu_params = self._encoder_params()
        own = {}
        if phase in ("joint", "enc"):
            own["own_actor_grad"] = grad_l2(actor_params)
            own["own_encoder_grad"] = grad_l2(mu_params)
            actor_before = param_l2(actor_params)
            enc_before = param_l2(mu_params)
        if phase in ("joint", "val"):
            own["own_value_grad"] = grad_l2(val_params)
            val_before = param_l2(val_params)
        torch.nn.utils.clip_grad_norm_(actor_params, self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(repr_params, self.max_grad_norm)
        self.actor_optimizer.step()
        self.representation_optimizer.step()
        if phase in ("joint", "enc"):
            own["own_actor_delta"] = delta_l2(actor_before, actor_params)
            own["own_encoder_delta"] = delta_l2(enc_before, mu_params)
            own["own_actor_step"] = 1.0
            own["own_encoder_step"] = 1.0
        else:
            own["own_actor_step"] = 0.0
            own["own_encoder_step"] = 0.0
        if phase in ("joint", "val"):
            own["own_value_delta"] = delta_l2(val_before, val_params)
            own["own_value_step"] = 1.0
        else:
            own["own_value_step"] = 0.0
        extra_stats.update(own)
        extra_stats["own_clip_frac"] = fwd["clip_frac"].detach()
        extra_stats["own_policy_loss"] = fwd["policy_loss"].detach()
        extra_stats["own_value_loss"] = fwd["value_loss"].detach()
        extra_stats["own_policy_coef"] = torch.tensor(1.0, device=self.device)
        extra_stats["own_value_coef"] = torch.tensor(self.value_coef, device=self.device)
        self.policy_version += 1
        self.encoder_version += 1
        self._after_optimizer_step(phase=phase)

        with torch.no_grad():
            for name, param in [("policy", self.policy), ("critic", self.critic)]:
                for p in param.parameters():
                    if not torch.isfinite(p).all():
                        raise RuntimeError(
                            f"Non-finite {name} weights after optimizer step"
                        )

        for key, val in extra_stats.items():
            extra_stats_acc.setdefault(key, []).append(val)

        totals["policy_loss"] += fwd["policy_loss"].item()
        totals["value_loss"] += fwd["value_loss"].item()
        totals["entropy"] += (-fwd["entropy_loss"]).item() if compute_policy else 0.0
        totals["vae_loss"] += fwd["vae_loss"].item()
        totals["recon_loss"] += fwd["recon_loss"].item()
        totals["vae_kl_loss"] += fwd["kl_loss"].item()
        totals["clip_frac"] += float(fwd["clip_frac"].item()) if compute_policy else 0.0
        with torch.no_grad():
            if compute_policy:
                kl = (batch_old_log_probs - fwd["log_probs"]).mean().item()
                totals["kl"] += abs(kl)

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
        self._training_epoch = training_epoch
        if self.norm_adv:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.anneal_lr and self.total_update_epochs > 0:
            self._lr_update_count += 1
            frac = 1.0 - (self._lr_update_count - 1) / float(self.total_update_epochs)
            lr_now = self.lr * max(0.0, frac)
            for opt in (self.actor_optimizer, self.representation_optimizer):
                for g in opt.param_groups:
                    g["lr"] = lr_now

        if self.pfo_coef > 0.0:
            from src.losses.pfo import snapshot_policy

            self._pfo_old_policy = snapshot_policy(self.policy)

        pl_ref_stats = self.maybe_init_frozen_pl_ref_buffer(obs)

        # v_ref from frozen PL ref buffer when enabled; else on-policy / D_Z buffer
        ref_for_v = self.ref_obs if self.ref_obs is not None else obs
        self.update_v_ref_from_obs(ref_for_v)

        indices = torch.arange(obs.shape[0], dtype=torch.long)
        if self.needs_transition_batch:
            if rewards is None or next_obs is None:
                raise ValueError("CTRO losses require rewards and next_obs in update()")
            dataset = TensorDataset(
                obs, actions, old_log_probs, advantages, returns, rewards, next_obs, indices
            )
        else:
            dataset = TensorDataset(
                obs, actions, old_log_probs, advantages, returns, indices
            )

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "vae_loss": 0.0,
            "recon_loss": 0.0,
            "vae_kl_loss": 0.0,
            "clip_frac": 0.0,
        }
        extra_stats_acc: dict[str, list] = {}
        self._kl_stop = False

        if self.head_phasing:
            phase_schedule = (["enc"] * self.enc_epochs) + (["val"] * self.val_epochs)
        else:
            phase_schedule = ["joint"] * self.num_epochs

        num_updates = 0
        for phase in phase_schedule:
            if self._kl_stop and phase in ("enc", "joint"):
                continue
            self._set_phase_requires_grad(phase)
            for batch in dataloader:
                self._run_minibatch(batch, phase, extra_stats_acc, totals)
                num_updates += 1
                if (
                    self.target_kl is not None
                    and phase in ("enc", "joint")
                    and totals["kl"] / num_updates > self.target_kl
                ):
                    self._kl_stop = True
                    break

        # Restore full grads for next external use / eval.
        self._set_phase_requires_grad("joint")

        self._step += 1
        if num_updates == 0:
            raise RuntimeError("PPO update ran zero minibatches")

        stats = {
            "policy_loss": totals["policy_loss"] / num_updates,
            "value_loss": totals["value_loss"] / num_updates,
            "entropy": totals["entropy"] / max(1, num_updates),
            "kl": totals["kl"] / max(1, num_updates),
            "clip_frac": totals["clip_frac"] / max(1, num_updates),
            "advantage_var": float(advantages.var(unbiased=False).item()),
            "vae_loss": totals["vae_loss"] / num_updates,
            "recon_loss": totals["recon_loss"] / num_updates,
            "vae_kl_loss": totals["vae_kl_loss"] / num_updates,
            "v_ref": float(self.v_ref) if self.v_ref is not None else float("nan"),
            "use_policy_clip": float(self.use_policy_clip),
            "own_nan_count": float(self.own_nan_count),
            "own_rollback_count": float(self.own_rollback_count),
        }
        for key, vals in extra_stats_acc.items():
            stats[key] = _mean_stat(vals)

        stats.update(pl_ref_stats)
        stats.update(self.target_landscape_pl_stats())
        return stats

    def refresh_pl_target(self) -> None:
        self.pl_target_critic.hard_copy_from(self.critic)
        if self.v_ref is not None:
            self.v_ref_target = float(self.v_ref)

    def target_landscape_pl_stats(self) -> dict[str, float]:
        if self.ref_obs is None:
            return {}
        if self.v_ref_target is None:
            return {}
        with torch.no_grad():
            mu, _ = self.pl_target_critic.module.encode(self.ref_obs)
        return target_landscape_pl_stats(
            self.pl_target_critic.module,
            mu,
            self.v_ref_target,
            f_floor=self.f_floor,
        )

    def reference_validity_report(self, min_old_distance: float) -> dict:
        from src.metrics.reference_validity import covariance_floor_stats, reference_validity_stats

        if self.ref_obs is None:
            raise RuntimeError("reference validity requires a frozen ref batch")
        z = self._encode_batch(self.ref_obs)
        stats = reference_validity_stats(
            self.ref_obs, z, min_old_distance, self.ref_freeze_epoch, "mu"
        )
        stats.update(covariance_floor_stats(z, 1e-6))
        return stats

    def snapshot_representation(self) -> dict:
        snap = {
            "critic": copy.deepcopy(self.critic.state_dict()),
            "representation_optimizer": copy.deepcopy(
                self.representation_optimizer.state_dict()
            ),
            "encoder_version": self.encoder_version,
        }
        if self.repr_net is not None:
            snap["repr_net"] = copy.deepcopy(self.repr_net.state_dict())
        return snap

    def restore_representation(self, snap: dict) -> None:
        self.critic.load_state_dict(snap["critic"])
        self.representation_optimizer.load_state_dict(snap["representation_optimizer"])
        self.encoder_version = snap["encoder_version"]
        if self.repr_net is not None:
            self.repr_net.load_state_dict(snap["repr_net"])
        self._assert_representation_matches(snap)

    def _assert_representation_matches(self, snap: dict) -> None:
        for key, tensor in self.critic.state_dict().items():
            ref = snap["critic"][key]
            if not torch.equal(tensor.cpu(), ref.cpu()):
                raise RuntimeError(
                    f"hard-gate restore failed: critic.{key} does not match snapshot"
                )
        opt_now = self.representation_optimizer.state_dict()
        opt_ref = snap["representation_optimizer"]
        if opt_now.keys() != opt_ref.keys():
            raise RuntimeError("hard-gate restore failed: optimizer key mismatch")
        for group_now, group_ref in zip(opt_now["param_groups"], opt_ref["param_groups"]):
            if group_now != group_ref:
                raise RuntimeError(
                    "hard-gate restore failed: optimizer param_groups mismatch"
                )

    def checkpoint_dict(
        self,
        training_epoch: int | None = None,
        total_env_steps: int | None = None,
    ) -> dict:
        save_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "representation_optimizer": self.representation_optimizer.state_dict(),
            "v_ref": self.v_ref,
            "pl_ref_frozen": self._pl_ref_frozen,
            "policy_version": self.policy_version,
            "encoder_version": self.encoder_version,
        }
        if self.repr_net is not None:
            save_dict["repr_net"] = self.repr_net.state_dict()
        if self.ref_obs is not None:
            save_dict["ref_obs"] = self.ref_obs.detach().cpu()
        if training_epoch is not None:
            save_dict["training_epoch"] = int(training_epoch)
            save_dict["total_env_steps"] = int(total_env_steps)
            save_dict["lr_update_count"] = int(self._lr_update_count)
        return save_dict

    def save(
        self,
        path: str,
        training_epoch: int | None = None,
        total_env_steps: int | None = None,
    ) -> None:
        torch.save(
            self.checkpoint_dict(training_epoch, total_env_steps),
            path,
        )

    def load(self, path: str, weights_only: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self._load_checkpoint(checkpoint, weights_only=weights_only)

    def _load_checkpoint(self, checkpoint: dict, weights_only: bool = False) -> None:
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        if not weights_only:
            if "actor_optimizer" not in checkpoint:
                raise RuntimeError(
                    "checkpoint missing actor_optimizer; re-run after the optimizer split"
                )
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.representation_optimizer.load_state_dict(
                checkpoint["representation_optimizer"]
            )
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])
        if "v_ref" in checkpoint:
            self.v_ref = checkpoint["v_ref"]
        if "policy_version" in checkpoint:
            self.policy_version = int(checkpoint["policy_version"])
        if "encoder_version" in checkpoint:
            self.encoder_version = int(checkpoint["encoder_version"])
        if "ref_obs" in checkpoint:
            self.ref_obs = checkpoint["ref_obs"].to(self.device)
            self._pl_ref_frozen = bool(checkpoint.get("pl_ref_frozen", True))
        elif "pl_ref_frozen" in checkpoint:
            self._pl_ref_frozen = bool(checkpoint["pl_ref_frozen"])
        if "lr_update_count" in checkpoint:
            self._lr_update_count = int(checkpoint["lr_update_count"])
