"""CTRO-focused metric evaluation: return, feature rank, mu_PL."""

import torch
import torch.nn as nn

from src.metrics.feature_rank import compute_feature_rank_metrics
from src.metrics.pl_ratio import compute_mu_pl_bootstrap, update_v_ref
from src.utils.bisimulation_utils import encode_phi


class CTROMetricEvaluator:
    """Lightweight evaluator for CTRO experiments."""

    def __init__(
        self,
        gamma: float = 0.99,
        repr_net: nn.Module | None = None,
        mu_pl_max_samples: int | None = None,
        mico_embed_ball_radius: float | None = None,
        tau_ref: float = 0.01,
        f_floor: float = 1e-3,
        v_ref_quantile: float = 0.99,
    ):
        self.gamma = gamma
        self.repr_net = repr_net
        self.mu_pl_max_samples = mu_pl_max_samples
        self.mico_embed_ball_radius = mico_embed_ball_radius
        self.tau_ref = tau_ref
        self.f_floor = f_floor
        self.v_ref_quantile = v_ref_quantile
        self.v_ref: float | None = None

    def _encode_obs(self, critic: nn.Module, obs: torch.Tensor) -> torch.Tensor:
        return encode_phi(
            critic,
            obs,
            repr_net=self.repr_net,
            embed_ball_radius=self.mico_embed_ball_radius,
        )

    def evaluate(
        self,
        critic: nn.Module,
        obs_buffer: torch.Tensor,
        next_obs_buffer: torch.Tensor | None = None,
        rewards_buffer: torch.Tensor | None = None,
        v_ref: float | None = None,
    ) -> dict[str, float]:
        del next_obs_buffer, rewards_buffer  # denom no longer uses Bellman residual

        with torch.no_grad():
            z = self._encode_obs(critic, obs_buffer)
            v = critic.value_head(z).squeeze(-1)

        metrics = compute_feature_rank_metrics(z)

        if v_ref is None:
            self.v_ref = update_v_ref(
                self.v_ref, v, tau_ref=self.tau_ref, quantile=self.v_ref_quantile
            )
            v_ref_use = self.v_ref
        else:
            v_ref_use = v_ref
            self.v_ref = v_ref

        pl = compute_mu_pl_bootstrap(
            critic,
            z,
            v_ref=v_ref_use,
            f_floor=self.f_floor,
            max_samples=self.mu_pl_max_samples,
        )
        metrics.update(pl)
        return metrics
