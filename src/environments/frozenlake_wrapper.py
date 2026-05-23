"""
FrozenLake environment wrapper with one-hot observations and
ground-truth causal representation extraction.
"""

from __future__ import annotations

import gymnasium as gym
import torch


class FrozenLakeWrapper:
    """
    Adapter around Gymnasium's FrozenLake that matches the interface
    your training code expects from `MinigridWrapper`.

    - reset(seed=None) -> (obs_tensor, info)
    - step(action) -> (obs_tensor, reward, terminated, truncated, info)
    - attributes: obs_dim, action_dim, obs_shape
    - get_ground_truth_representation(obs) -> torch.Tensor

    Observations are returned as one-hot vectors over FrozenLake states.
    """

    def __init__(
        self,
        map_name: str = "8x8",
        is_slippery: bool = False,
        seed: int | None = None,
    ):
        self.map_name = map_name
        self.is_slippery = is_slippery

        self.env = gym.make("FrozenLake-v1", map_name=map_name, is_slippery=is_slippery)

        if seed is not None:
            self.env.reset(seed=seed)

        self.obs_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.obs_dim = self.obs_space.n  # FrozenLake observation is discrete state index
        self.action_dim = self.action_space.n
        self.obs_shape = None  # Vector observations

        # Precompute ground-truth features from the fixed FrozenLake layout.
        unwrapped = self.env.unwrapped
        desc = unwrapped.desc  # (nrow, ncol) grid of characters
        nrow, ncol = desc.shape

        self._nrow = nrow
        self._ncol = ncol
        self._state_x_norm = torch.empty(self.obs_dim, dtype=torch.float32)
        self._state_y_norm = torch.empty(self.obs_dim, dtype=torch.float32)

        goal_mask = torch.zeros(self.obs_dim, dtype=torch.bool)
        hole_mask = torch.zeros(self.obs_dim, dtype=torch.bool)

        goal_state = None
        goal_row = 0
        goal_col = 0

        for state in range(self.obs_dim):
            r = state // ncol
            c = state % ncol
            self._state_x_norm[state] = torch.tensor(c / (ncol - 1), dtype=torch.float32)
            self._state_y_norm[state] = torch.tensor(r / (nrow - 1), dtype=torch.float32)

            cell = desc[r, c]
            is_goal = (cell == "G") or (cell == b"G")
            is_hole = (cell == "H") or (cell == b"H")

            goal_mask[state] = is_goal
            hole_mask[state] = is_hole

            if is_goal and goal_state is None:
                goal_state = state
                goal_row = r
                goal_col = c

        if goal_state is None:
            raise ValueError("FrozenLake map has no goal 'G' cell")

        self._goal_x_norm = goal_col / (ncol - 1)
        self._goal_y_norm = goal_row / (nrow - 1)
        self._goal_mask = goal_mask
        self._hole_mask = hole_mask

    def _obs_to_onehot(self, state_idx: int) -> torch.Tensor:
        onehot = torch.zeros(self.obs_dim, dtype=torch.float32)
        onehot[state_idx] = 1.0
        return onehot

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed)
        return self._obs_to_onehot(int(obs)), info

    def step(self, action: int) -> tuple[torch.Tensor, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._obs_to_onehot(int(obs)), float(reward), bool(terminated), bool(truncated), info

    def get_ground_truth_representation(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute a fixed-size ground-truth feature vector Z*(s) from the
        one-hot observation encoding of FrozenLake's state.

        Returns:
            Tensor [6] = [
              x_norm, y_norm,
              goal_x_norm, goal_y_norm,
              is_goal, is_hole
            ]
        """
        device = obs.device

        state_idx = int(torch.argmax(obs, dim=-1).item())

        x_norm = self._state_x_norm[state_idx].to(device)
        y_norm = self._state_y_norm[state_idx].to(device)
        goal_x_norm = torch.tensor(self._goal_x_norm, dtype=torch.float32, device=device)
        goal_y_norm = torch.tensor(self._goal_y_norm, dtype=torch.float32, device=device)

        is_goal = self._goal_mask[state_idx].to(device).to(torch.float32)
        is_hole = self._hole_mask[state_idx].to(device).to(torch.float32)

        return torch.stack([x_norm, y_norm, goal_x_norm, goal_y_norm, is_goal, is_hole])

    def render(self, mode: str = "human"):
        return self.env.render()

    def close(self):
        self.env.close()

