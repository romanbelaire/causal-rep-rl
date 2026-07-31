"""
Procgen environment wrapper for performance evaluation.

Uses the Procgen gym3 interface. Level sets are controlled via start_level and
num_levels (num_levels=0 means the full procedural distribution).

Supports vectorized environments via num_envs>1 (ProcgenGym3Env(num=...)).
"""

import numpy as np
import torch
from procgen import ProcgenGym3Env


class ProcgenWrapper:
    """
    Procgen adapter matching the MinigridWrapper interface.

    Observations are RGB images normalized to [0, 1].
    With num_envs==1, reset/step use scalar obs/action (eval path).
    With num_envs>1, reset/step use batched tensors/arrays (train path).
    """

    def __init__(
        self,
        env_name: str,
        distribution_mode: str = "easy",
        num_levels: int = 0,
        start_level: int = 0,
        keep_image_format: bool = False,
        num_envs: int = 1,
    ):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.env_name = env_name
        self.distribution_mode = distribution_mode
        self.num_levels = num_levels
        self.start_level = start_level
        self.keep_image_format = keep_image_format
        self.num_envs = num_envs
        self.action_space_type = "discrete"

        self.env = ProcgenGym3Env(
            num=num_envs,
            env_name=env_name,
            distribution_mode=distribution_mode,
            num_levels=num_levels,
            start_level=start_level,
        )

        _, obs, _ = self.env.observe()
        sample = obs["rgb"][0]
        self.obs_shape = tuple(sample.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(self.env.ac_space.eltype.n)
        self._obs = self._process_obs_batch(obs["rgb"])

    def _process_obs_batch(self, rgb: np.ndarray) -> torch.Tensor:
        obs_normalized = rgb.astype(np.float32) / 255.0
        if self.keep_image_format:
            return torch.from_numpy(obs_normalized)
        return torch.from_numpy(obs_normalized.reshape(rgb.shape[0], -1))

    def current_obs(self) -> torch.Tensor:
        if self.num_envs == 1:
            return self._obs[0]
        return self._obs

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            raise ValueError(
                "Procgen level sets are fixed at construction; recreate the wrapper "
                f"to change start_level/num_levels (got reset seed={seed})."
            )
        _, obs, _ = self.env.observe()
        self._obs = self._process_obs_batch(obs["rgb"])
        return self.current_obs(), {}

    def step(
        self, action
    ) -> tuple[torch.Tensor, float | np.ndarray, bool | np.ndarray, bool | np.ndarray, dict | list]:
        if self.num_envs == 1:
            if isinstance(action, torch.Tensor):
                action = int(action.item())
            actions = np.array([action], dtype=np.int32)
        else:
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            actions = np.asarray(action, dtype=np.int32)
            if actions.shape != (self.num_envs,):
                raise ValueError(
                    f"Expected actions shape ({self.num_envs},), got {actions.shape}"
                )

        self.env.act(actions)
        reward, obs, first = self.env.observe()
        self._obs = self._process_obs_batch(obs["rgb"])
        infos = self.env.get_info()
        # gym3 `first` is True on the observation that starts a new episode
        # (i.e. the previous transition was terminal / auto-reset).
        dones = first.astype(bool)

        if self.num_envs == 1:
            return (
                self._obs[0],
                float(reward[0]),
                bool(dones[0]),
                False,
                infos[0],
            )
        return (
            self._obs,
            reward.astype(np.float32),
            dones,
            np.zeros(self.num_envs, dtype=bool),
            infos,
        )

    def close(self) -> None:
        self.env.close()

    def render(self, mode: str = "human"):
        raise NotImplementedError("ProcgenWrapper does not support render()")
