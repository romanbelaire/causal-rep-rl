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

    Actions are clipped to the env action_spec (CleanRL ClipAction equivalent).
    Time-limit ends are reported as truncated (discount > 0); true fails as terminated.
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
        spec = self.env.action_spec()
        self.action_dim = int(spec.shape[0])
        self.action_low = np.asarray(spec.minimum, dtype=np.float32)
        self.action_high = np.asarray(spec.maximum, dtype=np.float32)

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
            spec = self.env.action_spec()
            self.action_low = np.asarray(spec.minimum, dtype=np.float32)
            self.action_high = np.asarray(spec.maximum, dtype=np.float32)
        time_step = self.env.reset()
        return self._process_obs(time_step.observation), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        action_np = np.asarray(action.detach().cpu().numpy(), dtype=np.float64)
        action_np = np.clip(action_np, self.action_low, self.action_high)
        time_step = self.env.step(action_np)
        obs = self._process_obs(time_step.observation)
        reward = float(time_step.reward)
        if time_step.last():
            # dm_control: discount==0 => true termination; discount>0 => time-limit truncate.
            if float(time_step.discount) == 0.0:
                terminated, truncated = True, False
            else:
                terminated, truncated = False, True
        else:
            terminated, truncated = False, False
        return obs, reward, terminated, truncated, {}

    def close(self) -> None:
        pass

    def render(self, mode: str = "human"):
        raise NotImplementedError("DMControlWrapper does not support render()")
