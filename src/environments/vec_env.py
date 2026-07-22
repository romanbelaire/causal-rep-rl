"""
Vectorized environments for performance-suite training.

Two backends behind one interface so the rollout code is env-agnostic:

  * ProcgenVectorEnv  -- Procgen is natively batched in C (ProcgenGym3Env(num=N)),
    so N environments step together on multiple cores with a single call and
    auto-reset internally. This is the correct way to vectorize Procgen.

  * SubprocVectorEnv  -- DMControl (MuJoCo) has no native batching, so we run one
    env per worker process and step them in lockstep. This is what turns the
    single-env CPU-bound rollout into a multi-core one.

Both return batched CPU float32 tensors shaped [num_envs, *obs] and apply
auto-reset: after an episode ends, `obs` is already the first observation of the
next episode, while `next_obs` carries the terminal observation used for value
bootstrapping (so GAE/critic targets see the true last state, not the reset).
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np
import torch

from src.evaluation.suites import EVAL_SUITES, parse_dmcontrol_task


class VecStepResult:
    """One vectorized transition: batched next-step feed obs + bootstrap obs."""

    __slots__ = ("obs", "rewards", "dones", "next_obs")

    def __init__(self, obs, rewards, dones, next_obs):
        self.obs = obs
        self.rewards = rewards
        self.dones = dones
        self.next_obs = next_obs


class ProcgenVectorEnv:
    """Native batched Procgen. Auto-resets internally; no terminal obs available."""

    def __init__(self, env_name: str, distribution_mode: str, num_levels: int, start_level: int, num_envs: int):
        from procgen import ProcgenGym3Env

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

    def _process(self, rgb: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(rgb.astype(np.float32) / 255.0)

    def reset(self) -> torch.Tensor:
        _, obs, _ = self.env.observe()
        return self._process(obs["rgb"])

    def step(self, actions: np.ndarray) -> VecStepResult:
        self.env.act(actions.astype(np.int32))
        reward, obs, first = self.env.observe()
        feed = self._process(obs["rgb"])
        return VecStepResult(
            obs=feed,
            rewards=reward.astype(np.float32),
            dones=first.astype(bool),
            # Procgen auto-resets in-C without exposing the terminal frame, so the
            # reset frame is used for bootstrapping (matches the single-env wrapper).
            next_obs=feed,
        )

    def close(self) -> None:
        self.env.close()


def _dmcontrol_worker(remote, domain: str, task_name: str, seed: int):
    from src.environments.dmcontrol_wrapper import DMControlWrapper

    env = DMControlWrapper(domain_name=domain, task_name=task_name, random_seed=seed)
    obs, _ = env.reset()
    remote.send(obs.numpy())
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            action = torch.from_numpy(data)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            terminal = next_obs.numpy()
            if done:
                feed, _ = env.reset()
                feed = feed.numpy()
            else:
                feed = terminal
            remote.send((feed, reward, done, terminal))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        else:
            raise ValueError(f"Unknown command: {cmd}")


class SubprocVectorEnv:
    """One DMControl env per worker process, stepped in lockstep across cores."""

    def __init__(self, domain: str, task_name: str, num_envs: int, base_seed: int):
        self.num_envs = num_envs
        self.action_space_type = "continuous"
        self.obs_shape = None

        ctx = mp.get_context("spawn")
        self._remotes = []
        self._procs = []
        for i in range(num_envs):
            parent, child = ctx.Pipe()
            proc = ctx.Process(
                target=_dmcontrol_worker,
                args=(child, domain, task_name, base_seed + i),
                daemon=True,
            )
            proc.start()
            child.close()
            self._remotes.append(parent)
            self._procs.append(proc)

        self._initial_obs = np.stack([r.recv() for r in self._remotes], axis=0)
        self.obs_dim = int(self._initial_obs.shape[1])
        probe = DMControlProbe(domain, task_name)
        self.action_dim = probe.action_dim

    def reset(self) -> torch.Tensor:
        return torch.from_numpy(self._initial_obs.astype(np.float32))

    def step(self, actions: np.ndarray) -> VecStepResult:
        for remote, action in zip(self._remotes, actions):
            remote.send(("step", np.ascontiguousarray(action, dtype=np.float32)))
        feeds, rewards, dones, terminals = [], [], [], []
        for remote in self._remotes:
            feed, reward, done, terminal = remote.recv()
            feeds.append(feed)
            rewards.append(reward)
            dones.append(done)
            terminals.append(terminal)
        return VecStepResult(
            obs=torch.from_numpy(np.stack(feeds, axis=0).astype(np.float32)),
            rewards=np.asarray(rewards, dtype=np.float32),
            dones=np.asarray(dones, dtype=bool),
            next_obs=torch.from_numpy(np.stack(terminals, axis=0).astype(np.float32)),
        )

    def close(self) -> None:
        for remote in self._remotes:
            remote.send(("close", None))
        for proc in self._procs:
            proc.join()


class DMControlProbe:
    """Read action_dim without keeping a MuJoCo env alive in the main process."""

    def __init__(self, domain: str, task_name: str):
        from src.environments.dmcontrol_wrapper import DMControlWrapper

        env = DMControlWrapper(domain_name=domain, task_name=task_name, random_seed=0)
        self.action_dim = env.action_dim
        env.close()


def make_train_vec_env(suite_name: str, task: str, num_envs: int, base_seed: int):
    """Vectorized training env matching make_train_env's distribution."""
    suite = EVAL_SUITES[suite_name]
    if suite.env_type == "procgen":
        return ProcgenVectorEnv(
            env_name=task,
            distribution_mode=suite.distribution_mode,
            num_levels=suite.train_num_levels,
            start_level=0,
            num_envs=num_envs,
        )
    domain, task_name = parse_dmcontrol_task(task)
    return SubprocVectorEnv(domain, task_name, num_envs=num_envs, base_seed=base_seed)
