"""
Procgen environment wrapper for performance evaluation.

Uses the Procgen gym3 interface. Level sets are controlled via start_level and
num_levels (num_levels=0 means the full procedural distribution).
"""

import numpy as np
import torch
from procgen import ProcgenGym3Env


class ProcgenWrapper:
    """
    Single-env Procgen adapter matching the MinigridWrapper interface.

    Observations are RGB images normalized to [0, 1], flattened to [H*W*C] by default.
    """

    def __init__(
        self,
        env_name: str,
        distribution_mode: str = "easy",
        num_levels: int = 0,
        start_level: int = 0,
        keep_image_format: bool = False,
    ):
        self.env_name = env_name
        self.distribution_mode = distribution_mode
        self.num_levels = num_levels
        self.start_level = start_level
        self.keep_image_format = keep_image_format
        self.action_space_type = "discrete"

        self.env = ProcgenGym3Env(
            num=1,
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

    def _process_obs(self, obs: np.ndarray) -> torch.Tensor:
        obs_normalized = obs.astype(np.float32) / 255.0
        if self.keep_image_format:
            return torch.from_numpy(obs_normalized)
        return torch.from_numpy(obs_normalized.reshape(-1))

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None:
            raise ValueError(
                "Procgen level sets are fixed at construction; recreate the wrapper "
                f"to change start_level/num_levels (got reset seed={seed})."
            )
        _, obs, _ = self.env.observe()
        return self._process_obs(obs["rgb"][0]), {}

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        self.env.act(np.array([action], dtype=np.int32))
        reward, obs, first = self.env.observe()
        info = self.env.get_info()[0]
        done = bool(first[0])
        return self._process_obs(obs["rgb"][0]), float(reward[0]), done, False, info

    def close(self) -> None:
        self.env.close()

    def render(self, mode: str = "human"):
        raise NotImplementedError("ProcgenWrapper does not support render()")
