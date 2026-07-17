"""Track and persist the highest-return pixel episode as uint8 RGB frames."""

from pathlib import Path

import numpy as np
import torch


class BestEpisodeFrameRecorder:
    """Keep one in-flight episode buffer and retain the best completed trajectory."""

    def __init__(self, obs_shape: tuple[int, ...]):
        self.obs_shape = obs_shape
        self.best_return = float("-inf")
        self.best_frames: np.ndarray | None = None
        self._frames: list[np.ndarray] = []
        self._return = 0.0

    def start_episode(self) -> None:
        self._frames = []
        self._return = 0.0

    def append_frame(self, obs: torch.Tensor) -> None:
        self._frames.append(self._obs_to_uint8_frame(obs))

    def add_reward(self, reward: float) -> None:
        self._return += float(reward)

    def finish_episode(self) -> None:
        if self._return > self.best_return and self._frames:
            self.best_return = self._return
            self.best_frames = np.stack(self._frames, axis=0)

    def save(self, path: Path) -> None:
        if self.best_frames is None:
            return
        np.savez_compressed(
            path,
            frames=self.best_frames,
            episode_return=np.array(self.best_return),
            obs_shape=np.array(self.obs_shape),
        )

    def _obs_to_uint8_frame(self, obs: torch.Tensor) -> np.ndarray:
        flat = obs.detach().cpu().numpy()
        frame = flat.reshape(self.obs_shape)
        return np.clip(frame * 255.0, 0, 255).astype(np.uint8)


def make_best_episode_frame_recorder(env) -> BestEpisodeFrameRecorder | None:
    """Return a recorder for pixel envs; state-vector envs have obs_shape=None."""
    if env.obs_shape is None:
        return None
    return BestEpisodeFrameRecorder(env.obs_shape)
