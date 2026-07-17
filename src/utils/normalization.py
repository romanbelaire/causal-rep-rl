"""Running observation and reward normalization for performance training."""

from __future__ import annotations

import numpy as np
import torch


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
    """Discounted-return variance normalization (CleanRL-style)."""

    def __init__(self, gamma: float, epsilon: float = 1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.return_rms = RunningMeanStd(())
        self.episode_return = 0.0

    def reset_episode(self) -> None:
        self.episode_return = 0.0

    def normalize(self, reward: float, done: bool) -> float:
        self.episode_return = self.episode_return * self.gamma + reward
        self.return_rms.update(np.array([self.episode_return], dtype=np.float64))
        normalized = reward / np.sqrt(self.return_rms.var + self.epsilon)
        if done:
            self.reset_episode()
        return float(normalized)

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
    """Observation + reward normalization for a single training run."""

    def __init__(self, obs_shape: tuple[int, ...], gamma: float):
        self.obs_rms = RunningMeanStd(obs_shape)
        self.reward_norm = RewardNormalizer(gamma)

    def observe(self, obs: torch.Tensor) -> torch.Tensor:
        self.obs_rms.update(obs.detach().cpu().numpy())
        return self.obs_rms.normalize(obs)

    def reward(self, reward: float, done: bool) -> float:
        return self.reward_norm.normalize(reward, done)

    def state_dict(self) -> dict:
        return {
            "obs_rms": self.obs_rms.state_dict(),
            "reward_norm": self.reward_norm.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> PerformanceNormalizer:
        obs_shape = tuple(state["obs_rms"]["shape"])
        gamma = state["reward_norm"]["gamma"]
        obj = cls(obs_shape, gamma)
        obj.obs_rms = RunningMeanStd.from_state_dict(state["obs_rms"])
        obj.reward_norm = RewardNormalizer.from_state_dict(state["reward_norm"])
        return obj
