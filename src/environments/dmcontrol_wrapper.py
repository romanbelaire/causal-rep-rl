"""
DeepMind Control Suite wrapper with flattened state observations.
"""

import numpy as np
import torch
from dm_control import suite

FLAT_OBS_KEY = "observations"


class DMControlWrapper:
    """
    DMControl task adapter with vector (state) observations.

    task_kwargs['random'] controls the episode's initial conditions. Use different
    random seeds to define full vs held-out test distributions.
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        random_seed: int = 0,
    ):
        self.domain_name = domain_name
        self.task_name = task_name
        self.random_seed = random_seed
        self.action_space_type = "continuous"
        self.keep_image_format = False
        self.obs_shape = None

        self.env = suite.load(
            domain_name,
            task_name,
            task_kwargs={"random": random_seed},
            environment_kwargs={"flat_observation": True},
        )

        time_step = self.env.reset()
        sample = time_step.observation[FLAT_OBS_KEY]
        self.obs_dim = int(sample.shape[0])
        self.action_dim = int(self.env.action_spec().shape[0])

    def _process_obs(self, observation: dict) -> torch.Tensor:
        return torch.from_numpy(observation[FLAT_OBS_KEY].astype(np.float32))

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None and seed != self.random_seed:
            self.random_seed = seed
            self.env = suite.load(
                self.domain_name,
                self.task_name,
                task_kwargs={"random": seed},
                environment_kwargs={"flat_observation": True},
            )
        time_step = self.env.reset()
        return self._process_obs(time_step.observation), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        action = action.detach().cpu().numpy().astype(np.float64)
        time_step = self.env.step(action)
        obs = self._process_obs(time_step.observation)
        reward = float(time_step.reward)
        terminated = bool(time_step.last())
        return obs, reward, terminated, False, {}

    def close(self) -> None:
        pass

    def render(self, mode: str = "human"):
        raise NotImplementedError("DMControlWrapper does not support render()")
