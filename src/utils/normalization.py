"""Running observation and reward normalization for performance training."""

from __future__ import annotations

import numpy as np
import torch

ALLOWED_OBS_NORM = frozenset({"running_mean_std"})
ALLOWED_REWARD_NORM = frozenset({"return_var_scale"})


class RunningMeanStd:
    """Welford running mean/variance for vector or tensor observations."""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        self.shape = shape
        self.epsilon = epsilon
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, batch: np.ndarray) -> None:
        arr = np.asarray(batch, dtype=np.float64)
        if arr.shape == self.shape:
            arr = arr[np.newaxis, ...]
        batch_mean = arr.mean(axis=0)
        batch_var = arr.var(axis=0)
        batch_count = arr.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: float
    ) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / total_count
        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, obs: torch.Tensor, clip: float = 10.0) -> torch.Tensor:
        mean = torch.as_tensor(self.mean, dtype=obs.dtype, device=obs.device)
        var = torch.as_tensor(self.var, dtype=obs.dtype, device=obs.device)
        normalized = (obs - mean) / torch.sqrt(var + self.epsilon)
        return torch.clamp(normalized, -clip, clip)

    def state_dict(self) -> dict:
        return {
            "shape": self.shape,
            "epsilon": self.epsilon,
            "mean": self.mean.tolist(),
            "var": self.var.tolist(),
            "count": float(self.count),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> RunningMeanStd:
        obj = cls(tuple(state["shape"]), epsilon=state["epsilon"])
        obj.mean = np.array(state["mean"], dtype=np.float64)
        obj.var = np.array(state["var"], dtype=np.float64)
        obj.count = float(state["count"])
        return obj


class RewardNormalizer:
    """Discounted-return variance normalization (CleanRL-style).

    Supports scalar (eval) and vectorized (num_envs>1) rewards.
    """

    def __init__(self, gamma: float, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(())
        self.returns = np.zeros((), dtype=np.float64)

    def reset_episode(self) -> None:
        self.returns = np.zeros_like(self.returns)

    def normalize(
        self, reward: float | np.ndarray, done: bool | np.ndarray
    ) -> float | np.ndarray:
        reward_arr = np.asarray(reward, dtype=np.float64)
        done_arr = np.asarray(done, dtype=np.float64)
        if self.returns.shape != reward_arr.shape:
            self.returns = np.zeros(reward_arr.shape, dtype=np.float64)
        self.returns = self.returns * self.gamma + reward_arr
        self.return_rms.update(self.returns.reshape(-1))
        normalized = reward_arr / np.sqrt(self.return_rms.var + self.epsilon)
        self.returns = self.returns * (1.0 - done_arr)
        if normalized.ndim == 0:
            return float(normalized)
        return normalized.astype(np.float32)

    def normalize_batch(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        episode_returns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized version for N parallel envs sharing one return-variance estimate.

        `episode_returns` is the caller-held per-env discounted-return accumulator
        (shape [N]); it is advanced here and the post-reset copy is returned so the
        rollout can carry it to the next step.
        """
        episode_returns = episode_returns * self.gamma + rewards
        self.return_rms.update(episode_returns.astype(np.float64))
        normalized = rewards / np.sqrt(self.return_rms.var + self.epsilon)
        episode_returns = np.where(dones, 0.0, episode_returns)
        return normalized.astype(np.float32), episode_returns

    def state_dict(self) -> dict:
        return {
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "return_rms": self.return_rms.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> RewardNormalizer:
        obj = cls(state["gamma"], epsilon=state["epsilon"])
        obj.return_rms = RunningMeanStd.from_state_dict(state["return_rms"])
        return obj


class PerformanceNormalizer:
    """Observation + reward normalization for a single training run.

    Modes (fail fast on unknown):
      obs_norm="running_mean_std"  — (x-mean)/std with ±obs_norm_clip
      reward_norm_mode="return_var_scale" — CleanRL return-std scale, no reward clip
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        gamma: float,
        obs_norm: str = "running_mean_std",
        obs_norm_clip: float = 10.0,
        reward_norm_mode: str = "return_var_scale",
    ):
        if obs_norm not in ALLOWED_OBS_NORM:
            raise ValueError(f"Unknown obs_norm={obs_norm!r}; allowed={sorted(ALLOWED_OBS_NORM)}")
        if reward_norm_mode not in ALLOWED_REWARD_NORM:
            raise ValueError(
                f"Unknown reward_norm_mode={reward_norm_mode!r}; "
                f"allowed={sorted(ALLOWED_REWARD_NORM)}"
            )
        self.obs_norm = obs_norm
        self.obs_norm_clip = float(obs_norm_clip)
        self.reward_norm_mode = reward_norm_mode
        self.obs_rms = RunningMeanStd(obs_shape)
        self.reward_norm = RewardNormalizer(gamma)

    def observe(self, obs: torch.Tensor) -> torch.Tensor:
        self.obs_rms.update(obs.detach().cpu().numpy())
        return self.obs_rms.normalize(obs, clip=self.obs_norm_clip)

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Z-score without updating running stats (e.g. next_obs in vec rollouts)."""
        return self.obs_rms.normalize(obs, clip=self.obs_norm_clip)

    def reward(self, reward: float, done: bool) -> float:
        return self.reward_norm.normalize(reward, done)

    def state_dict(self) -> dict:
        return {
            "obs_norm": self.obs_norm,
            "obs_norm_clip": self.obs_norm_clip,
            "reward_norm_mode": self.reward_norm_mode,
            "obs_rms": self.obs_rms.state_dict(),
            "reward_norm": self.reward_norm.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> PerformanceNormalizer:
        obs_shape = tuple(state["obs_rms"]["shape"])
        gamma = state["reward_norm"]["gamma"]
        obj = cls(
            obs_shape,
            gamma,
            obs_norm=state.get("obs_norm", "running_mean_std"),
            obs_norm_clip=float(state.get("obs_norm_clip", 10.0)),
            reward_norm_mode=state.get("reward_norm_mode", "return_var_scale"),
        )
        obj.obs_rms = RunningMeanStd.from_state_dict(state["obs_rms"])
        obj.reward_norm = RewardNormalizer.from_state_dict(state["reward_norm"])
        return obj
