"""Gymnasium ALE (Atari) wrapper matching the performance-suite env API."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, TransformReward


def _make_ale_gym(env_id: str, seed: int | None) -> gym.Env:
    import ale_py

    gym.register_envs(ale_py)
    env = gym.make(env_id, frameskip=1, repeat_action_probability=0.25)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=False,
    )
    env = FrameStackObservation(env, stack_size=4)
    env = TransformReward(env, np.sign)
    if seed is not None:
        env.reset(seed=seed)
    return env


class ALEWrapper:
    """Single ALE env. Observations are uint8 stacks cast to float32 [0, 255] (CNN /255)."""

    def __init__(self, env_id: str, seed: int | None = None):
        self.env_id = env_id
        self.action_space_type = "discrete"
        self.env = _make_ale_gym(env_id, seed)
        self.action_dim = int(self.env.action_space.n)
        obs, _ = self.env.reset(seed=seed)
        # FrameStackObservation yields (4, 84, 84)
        self.obs_shape = tuple(np.asarray(obs).shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self._obs = self._to_tensor(obs)

    def _to_tensor(self, obs) -> torch.Tensor:
        arr = np.asarray(obs, dtype=np.float32)
        return torch.from_numpy(arr)

    def reset(self, seed: int | None = None) -> tuple[torch.Tensor, dict]:
        obs, info = self.env.reset(seed=seed)
        self._obs = self._to_tensor(obs)
        return self._obs, info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = int(action.item())
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._obs = self._to_tensor(obs)
        return self._obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.env.close()


class ALEVectorEnv:
    """SyncVectorEnv of ALE wrappers for Moalla-style 8-env rollouts."""

    def __init__(self, env_id: str, num_envs: int, base_seed: int):
        if num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {num_envs}")
        self.num_envs = num_envs
        self.action_space_type = "discrete"
        self.env_id = env_id

        def _thunk(rank: int):
            def _make():
                return _make_ale_gym(env_id, base_seed + rank)

            return _make

        self.env = gym.vector.SyncVectorEnv([_thunk(i) for i in range(num_envs)])
        obs, _ = self.env.reset(seed=base_seed)
        sample = np.asarray(obs[0])
        self.obs_shape = tuple(sample.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(self.env.single_action_space.n)
        self._obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))

    def reset(self) -> torch.Tensor:
        obs, _ = self.env.reset()
        self._obs = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        return self._obs

    def step(self, actions: np.ndarray):
        from src.environments.vec_env import VecStepResult

        obs, rewards, terminations, truncations, infos = self.env.step(actions.astype(np.int64))
        dones = np.logical_or(terminations, truncations)
        feed = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        # SyncVectorEnv auto-resets; terminal obs is in infos["final_obs"] when present.
        terminals = []
        for i in range(self.num_envs):
            if dones[i] and "final_obs" in infos:
                # gymnasium vector stores final_obs per-env in infos
                final = infos["final_obs"][i]
                if final is None:
                    terminals.append(feed[i].numpy())
                else:
                    terminals.append(np.asarray(final, dtype=np.float32))
            else:
                terminals.append(feed[i].numpy())
        terminal_t = torch.from_numpy(np.stack(terminals, axis=0).astype(np.float32))
        self._obs = feed
        return VecStepResult(
            obs=feed,
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            terminations=np.asarray(terminations, dtype=bool),
            truncations=np.asarray(truncations, dtype=bool),
            next_obs=terminal_t,
        )

    def close(self) -> None:
        self.env.close()
