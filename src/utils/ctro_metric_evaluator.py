"""CTRO-focused metric evaluation: return, feature rank, mu_PL."""

import torch
import torch.nn as nn

from src.metrics.feature_rank import compute_feature_rank_metrics
from src.metrics.pl_ratio import compute_mu_pl_bootstrap


class CTROMetricEvaluator:
    """Lightweight evaluator for CTRO experiments."""

    def __init__(self, gamma: float = 0.99, repr_net: nn.Module | None = None):
        self.gamma = gamma
        self.repr_net = repr_net

    def _encode_obs(self, critic: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        if self.repr_net is not None:
            return self.repr_net(obs)
        mu, _ = critic.encode(obs)
        return mu

    def evaluate(
        self,
        critic: nn.Module,
        obs_buffer: torch.Tensor,
        next_obs_buffer: torch.Tensor | None,
        rewards_buffer: torch.Tensor | None,
    ) -> dict[str, float]:
        with torch.no_grad():
            z = self._encode_obs(critic, obs_buffer)

        metrics = compute_feature_rank_metrics(z)

        if next_obs_buffer is not None and rewards_buffer is not None:
            pl = compute_mu_pl_bootstrap(
                critic,
                z,
                rewards_buffer,
                next_obs_buffer,
                self.gamma,
            )
            metrics.update(pl)

        return metrics
