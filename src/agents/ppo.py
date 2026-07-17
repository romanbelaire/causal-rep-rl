"""
Proximal Policy Optimization (PPO) — vanilla policy + value + VAE.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


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
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vae_coef = config.get("vae_coef", 0.1)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.batch_size = config.get("batch_size", 64)
        self.num_epochs = config.get("num_epochs", 4)
        self.policy_on_latent = config.get("policy_on_latent", True)

        all_params = list(self.policy.parameters()) + list(self.critic.parameters())
        if self.repr_net is not None:
            all_params = list(self.repr_net.parameters()) + all_params
        self.unified_optimizer = optim.Adam(all_params, lr=self.lr)

        self.old_policy = None
        self._step = 0

    @property
    def needs_transition_batch(self) -> bool:
        return False

    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * next_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae

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

    def _forward_batch(
        self,
        batch_obs: torch.Tensor,
        batch_actions: torch.Tensor,
        batch_old_log_probs: torch.Tensor,
        batch_advantages: torch.Tensor,
        batch_returns: torch.Tensor,
    ) -> dict:
        z = self._encode_batch(batch_obs)
        if not torch.isfinite(z).all():
            raise RuntimeError(
                f"Non-finite latent z in PPO update (nan={torch.isnan(z).any().item()}, "
                f"inf={torch.isinf(z).any().item()})"
            )
        policy_in = z if self.policy_on_latent else batch_obs

        log_probs, entropy = self.policy.evaluate_actions(policy_in, batch_actions)
        if not torch.isfinite(log_probs).all():
            raise RuntimeError("Non-finite policy log_probs in PPO update")

        # Clamp log-ratio before exp so negative-advantage / huge-ratio cases
        # cannot produce Inf surrogates (common continuous-PPO NaN path).
        log_ratio = torch.clamp(log_probs - batch_old_log_probs, min=-20.0, max=2.0)
        ratio = torch.exp(log_ratio)
        surr1 = ratio * batch_advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()

        vae_loss = torch.tensor(0.0, device=self.device)
        recon_loss = torch.tensor(0.0, device=self.device)
        kl_loss = torch.tensor(0.0, device=self.device)

        if hasattr(self.critic, "encode") and self.vae_coef > 0:
            values, vae_info = self.critic(batch_obs, return_latent=True)
            values = values.squeeze(-1)
            recon_loss = vae_info["recon_loss"]
            kl_loss = vae_info["kl_loss"]
            vae_loss = vae_info["vae_loss"]
        else:
            values = self.critic(batch_obs).squeeze(-1)

        value_loss = nn.functional.mse_loss(values, batch_returns)

        return {
            "z": z,
            "log_probs": log_probs,
            "policy_loss": policy_loss,
            "entropy_loss": entropy_loss,
            "value_loss": value_loss,
            "vae_loss": vae_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def _extra_critic_terms(
        self,
        batch_obs: torch.Tensor,
        z: torch.Tensor,
        batch_rewards: torch.Tensor | None,
        batch_next_obs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        return torch.tensor(0.0, device=self.device), {}

    def _after_optimizer_step(self) -> None:
        pass

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
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if self.needs_transition_batch:
            if rewards is None or next_obs is None:
                raise ValueError("CTRO losses require rewards and next_obs in update()")
            dataset = TensorDataset(
                obs, actions, old_log_probs, advantages, returns, rewards, next_obs
            )
        else:
            dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        extra_stats_acc: dict[str, list[float]] = {}

        for _epoch in range(self.num_epochs):
            for batch in dataloader:
                if self.needs_transition_batch:
                    (
                        batch_obs,
                        batch_actions,
                        batch_old_log_probs,
                        batch_advantages,
                        batch_returns,
                        batch_rewards,
                        batch_next_obs,
                    ) = batch
                    batch_rewards = batch_rewards.to(self.device)
                    batch_next_obs = batch_next_obs.to(self.device)
                else:
                    batch_obs, batch_actions, batch_old_log_probs, batch_advantages, batch_returns = batch
                    batch_rewards = None
                    batch_next_obs = None

                batch_obs = batch_obs.to(self.device)
                batch_actions = batch_actions.to(self.device)
                batch_old_log_probs = batch_old_log_probs.to(self.device)
                batch_advantages = batch_advantages.to(self.device)
                batch_returns = batch_returns.to(self.device)

                fwd = self._forward_batch(
                    batch_obs,
                    batch_actions,
                    batch_old_log_probs,
                    batch_advantages,
                    batch_returns,
                )

                extra_loss, extra_stats = self._extra_critic_terms(
                    batch_obs,
                    fwd["z"],
                    batch_rewards,
                    batch_next_obs,
                )

                critic_loss = (
                    self.value_coef * fwd["value_loss"]
                    + self.vae_coef * fwd["vae_loss"]
                    + extra_loss
                )

                self.unified_optimizer.zero_grad()
                policy_total = fwd["policy_loss"] + self.entropy_coef * fwd["entropy_loss"]
                policy_total.backward(retain_graph=True)
                critic_loss.backward()
                all_params = list(self.policy.parameters()) + list(self.critic.parameters())
                if self.repr_net is not None:
                    all_params = list(self.repr_net.parameters()) + all_params
                torch.nn.utils.clip_grad_norm_(all_params, self.max_grad_norm)
                self.unified_optimizer.step()
                self._after_optimizer_step()

                with torch.no_grad():
                    for name, param in [("policy", self.policy), ("critic", self.critic)]:
                        for p in param.parameters():
                            if not torch.isfinite(p).all():
                                raise RuntimeError(
                                    f"Non-finite {name} weights after optimizer step"
                                )

                for key, val in extra_stats.items():
                    extra_stats_acc.setdefault(key, []).append(val)

                total_policy_loss += fwd["policy_loss"].item()
                total_value_loss += fwd["value_loss"].item()
                total_entropy += (-fwd["entropy_loss"]).item()

                with torch.no_grad():
                    kl = (batch_old_log_probs - fwd["log_probs"]).mean().item()
                    total_kl += abs(kl)

        self._step += 1
        num_updates = self.num_epochs * len(dataloader)

        stats = {
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "kl": total_kl / num_updates,
        }
        for key, vals in extra_stats_acc.items():
            stats[key] = sum(vals) / len(vals)

        return stats

    def save(self, path: str) -> None:
        save_dict = {
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "unified_optimizer": self.unified_optimizer.state_dict(),
        }
        if self.repr_net is not None:
            save_dict["repr_net"] = self.repr_net.state_dict()
        torch.save(save_dict, path)

    def load(self, path: str, weights_only: bool = False) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        if not weights_only:
            self.unified_optimizer.load_state_dict(checkpoint["unified_optimizer"])
        if self.repr_net is not None and "repr_net" in checkpoint:
            self.repr_net.load_state_dict(checkpoint["repr_net"])
