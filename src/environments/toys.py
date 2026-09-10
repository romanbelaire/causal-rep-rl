"""Synthetic discrete environments for E1–E5 (CPU, deterministic)."""

import torch


class ActionAliasToy:
    """Same obs features, different reward under action 1 vs action 0."""

    action_dim = 2
    obs_dim = 4

    def __init__(self, seed: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self._t = 0
        self._alias = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        self._t = 0
        self._alias = int(torch.randint(0, 2, (1,), generator=self.rng).item())
        obs = torch.zeros(self.obs_dim)
        obs[0] = 1.0
        obs[1 + self._alias] = 1.0
        return obs, {}

    def step(self, action: int):
        self._t += 1
        obs = torch.zeros(self.obs_dim)
        obs[0] = 1.0
        obs[1 + self._alias] = 1.0
        if action == 0:
            reward = 1.0
        else:
            reward = 10.0 if self._alias == 0 else 0.0
        terminated = self._t >= 1
        truncated = False
        return obs, reward, terminated, truncated, {"alias": self._alias}


class NuisanceToy:
    """Visually distant states, identical reward futures for all actions."""

    action_dim = 2
    obs_dim = 8
    n_histories = 2

    def __init__(self, seed: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self._variant = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        self._variant = int(torch.randint(0, 2, (1,), generator=self.rng).item())
        obs = torch.zeros(self.obs_dim)
        obs[self._variant * 4] = 1.0
        return obs, {}

    def step(self, action: int):
        obs = torch.zeros(self.obs_dim)
        obs[self._variant * 4] = 1.0
        reward = 1.0
        return obs, reward, True, False, {}


class CoverageToy:
    """Discriminating action 1 only matters on alias=1; alias=1 is rare."""

    action_dim = 2
    obs_dim = 4
    alias_rate = 0.05

    def __init__(self, seed: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self._alias = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        u = torch.rand(1, generator=self.rng).item()
        self._alias = 1 if u < self.alias_rate else 0
        obs = torch.zeros(self.obs_dim)
        obs[self._alias] = 1.0
        return obs, {}

    def step(self, action: int):
        obs = torch.zeros(self.obs_dim)
        obs[self._alias] = 1.0
        if self._alias == 0:
            reward = 1.0
        else:
            reward = 10.0 if action == 1 else 0.0
        return obs, reward, True, False, {"alias": self._alias}


def kl_null_shear_z(z: torch.Tensor, c: float) -> torch.Tensor:
    """Scale latent coordinate 1 by c (gauge that can leave logits fixed if heads compensate)."""
    out = z.clone()
    out[:, 1] = out[:, 1] * c
    return out


class EqualValueAliasToy:
    """Two states, equal V^pi under action 0, different Q for the unused action 1."""

    action_dim = 2
    obs_dim = 4

    def __init__(self, seed: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self._alias = 0

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        self._alias = int(torch.randint(0, 2, (1,), generator=self.rng).item())
        return self._obs(), {}

    def _obs(self):
        obs = torch.zeros(self.obs_dim)
        obs[self._alias] = 1.0
        return obs

    def step(self, action: int):
        if action == 0:
            reward = 1.0
        else:
            reward = 10.0 if self._alias == 0 else 0.0
        return self._obs(), reward, True, False, {"alias": self._alias}


class FiniteRewardTestToy:
    """Three resettable histories. Histories 0 and 1 share reward tests; 2 does not."""

    n_histories = 3
    action_dim = 2
    obs_dim = 3
    _rewards = {
        (0, 0): 1.0,
        (0, 1): 0.0,
        (1, 0): 1.0,
        (1, 1): 0.0,
        (2, 0): 0.0,
        (2, 1): 1.0,
    }

    def __init__(self, seed: int = 0):
        self.rng = torch.Generator().manual_seed(seed)
        self._h = 0

    def set_history(self, h: int) -> torch.Tensor:
        self._h = int(h)
        return self._obs()

    def _obs(self):
        obs = torch.zeros(self.obs_dim)
        obs[self._h] = 1.0
        return obs

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = torch.Generator().manual_seed(seed)
        self._h = int(torch.randint(0, self.n_histories, (1,), generator=self.rng).item())
        return self._obs(), {}

    def step(self, action: int):
        reward = self._rewards[(self._h, int(action))]
        return self._obs(), reward, True, False, {"history": self._h}
