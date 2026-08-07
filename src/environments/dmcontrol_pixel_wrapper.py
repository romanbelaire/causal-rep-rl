"""
DeepMind Control Suite wrapper with pixel (RGB) observations.
"""

import numpy as np
import torch
from dm_control import suite


class DMControlPixelWrapper:
    """
    DMControl task adapter with rendered RGB observations.

    Renders via physics.render(H, W), normalizes to [0, 1], shape (H, W, 3).
    task_kwargs['random'] controls episode initial conditions (full vs test seeds).
    Actions are clipped to action_spec; time limits are truncated (discount > 0).
    """

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        random_seed: int = 0,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
    ):
        self.domain_name = domain_name
        self.task_name = task_name
        self.random_seed = random_seed
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.action_space_type = "continuous"
        self.keep_image_format = True
        self.obs_shape = (height, width, 3)
        self.obs_dim = int(np.prod(self.obs_shape))

        self.env = suite.load(
            domain_name,
            task_name,
            task_kwargs={"random": random_seed},
        )
        spec = self.env.action_spec()
        self.action_dim = int(spec.shape[0])
        self.action_low = np.asarray(spec.minimum, dtype=np.float32)
        self.action_high = np.asarray(spec.maximum, dtype=np.float32)

    def _process_obs(self) -> torch.Tensor:
        rgb = self.env.physics.render(
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
        )
        return torch.from_numpy(rgb.astype(np.float32) / 255.0)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        if seed is not None and seed != self.random_seed:
            self.random_seed = seed
            self.env = suite.load(
                self.domain_name,
                self.task_name,
                task_kwargs={"random": seed},
            )
            spec = self.env.action_spec()
            self.action_low = np.asarray(spec.minimum, dtype=np.float32)
            self.action_high = np.asarray(spec.maximum, dtype=np.float32)
        self.env.reset()
        return self._process_obs(), {}

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, float, bool, bool, dict]:
        action_np = np.asarray(action.detach().cpu().numpy(), dtype=np.float64)
        action_np = np.clip(action_np, self.action_low, self.action_high)
        time_step = self.env.step(action_np)
        obs = self._process_obs()
        reward = float(time_step.reward)
        if time_step.last():
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
        raise NotImplementedError("DMControlPixelWrapper does not support render()")
