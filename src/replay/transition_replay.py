"""Ring buffer of transitions. Preserves episode order via episode_id/step_id."""

import torch

SOURCE_PPO = 0
SOURCE_QUERY = 1
SOURCE_FROM_STR = {"ppo": SOURCE_PPO, "query": SOURCE_QUERY}
SOURCE_TO_STR = {SOURCE_PPO: "ppo", SOURCE_QUERY: "query"}


class TransitionReplay:
    """Fixed-capacity ring buffer. Every add must include source and log beta."""

    def __init__(self, capacity: int):
        if capacity < 2:
            raise RuntimeError(f"replay capacity must be >= 2, got {capacity}")
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self._obs = None
        self._next_obs = None
        self._action = None
        self._reward = None
        self._terminated = None
        self._truncated = None
        self._episode_id = None
        self._step_id = None
        self._behavior_log_prob = None
        self._control_log_prob = None
        self._source = None
        self._policy_version = None
        self._encoder_version = None
        self._priority_ig = None
        self._priority_gap = None
        self._priority_pair = None
        self._priority_geom = None

    def _allocate(self, obs: torch.Tensor, action: torch.Tensor) -> None:
        cap = self.capacity
        obs_shape = obs.shape[1:]
        act_shape = action.shape[1:]
        device = torch.device("cpu")
        self._obs = torch.zeros((cap, *obs_shape), dtype=obs.dtype, device=device)
        self._next_obs = torch.zeros((cap, *obs_shape), dtype=obs.dtype, device=device)
        self._action = torch.zeros((cap, *act_shape), dtype=action.dtype, device=device)
        self._reward = torch.zeros(cap, dtype=torch.float32, device=device)
        self._terminated = torch.zeros(cap, dtype=torch.bool, device=device)
        self._truncated = torch.zeros(cap, dtype=torch.bool, device=device)
        self._episode_id = torch.zeros(cap, dtype=torch.long, device=device)
        self._step_id = torch.zeros(cap, dtype=torch.long, device=device)
        self._behavior_log_prob = torch.zeros(cap, dtype=torch.float32, device=device)
        self._control_log_prob = torch.zeros(cap, dtype=torch.float32, device=device)
        self._source = torch.zeros(cap, dtype=torch.long, device=device)
        self._policy_version = torch.zeros(cap, dtype=torch.long, device=device)
        self._encoder_version = torch.zeros(cap, dtype=torch.long, device=device)
        self._priority_ig = torch.zeros(cap, dtype=torch.float32, device=device)
        self._priority_gap = torch.zeros(cap, dtype=torch.float32, device=device)
        self._priority_pair = torch.zeros(cap, dtype=torch.float32, device=device)
        self._priority_geom = torch.zeros(cap, dtype=torch.float32, device=device)

    def add_batch(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        episode_id: torch.Tensor,
        step_id: torch.Tensor,
        behavior_log_prob: torch.Tensor,
        control_log_prob: torch.Tensor,
        source: str,
        policy_version: int,
        encoder_version: int,
    ) -> None:
        if source not in SOURCE_FROM_STR:
            raise RuntimeError(f"unknown transition source {source!r}")
        n = obs.shape[0]
        if n == 0:
            raise RuntimeError("add_batch received empty tensors")
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        obs_c = obs.detach().cpu()
        next_obs_c = next_obs.detach().cpu()
        action_c = action.detach().cpu()
        reward_c = reward.detach().cpu().reshape(-1)
        terminated_c = terminated.detach().cpu().reshape(-1)
        truncated_c = truncated.detach().cpu().reshape(-1)
        episode_c = episode_id.detach().cpu().reshape(-1)
        step_c = step_id.detach().cpu().reshape(-1)
        blog_c = behavior_log_prob.detach().cpu().reshape(-1)
        clog_c = control_log_prob.detach().cpu().reshape(-1)
        if self._obs is None:
            self._allocate(obs_c, action_c)
        src_code = SOURCE_FROM_STR[source]
        for i in range(n):
            p = self.ptr
            self._obs[p] = obs_c[i]
            self._next_obs[p] = next_obs_c[i]
            self._action[p] = action_c[i]
            self._reward[p] = reward_c[i]
            self._terminated[p] = terminated_c[i]
            self._truncated[p] = truncated_c[i]
            self._episode_id[p] = episode_c[i]
            self._step_id[p] = step_c[i]
            self._behavior_log_prob[p] = blog_c[i]
            self._control_log_prob[p] = clog_c[i]
            self._source[p] = src_code
            self._policy_version[p] = policy_version
            self._encoder_version[p] = encoder_version
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample_indices(
        self,
        batch_size: int,
        sample_p: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.size == 0:
            raise RuntimeError("cannot sample from empty replay")
        if self.size < batch_size:
            raise RuntimeError(
                f"replay size {self.size} < batch_size {batch_size}"
            )
        if sample_p.shape[0] != self.size:
            raise RuntimeError(
                f"sample_p length {sample_p.shape[0]} != replay size {self.size}"
            )
        if (sample_p <= 0).any():
            raise RuntimeError("every stored item must have strictly positive sample probability")
        return torch.multinomial(sample_p, batch_size, replacement=False, generator=generator)

    def gather(self, indices: torch.Tensor) -> dict[str, torch.Tensor]:
        idx = indices.cpu()
        action = self._action[idx]
        if action.shape[-1] == 1:
            action = action.squeeze(-1)
        return {
            "obs": self._obs[idx].clone(),
            "next_obs": self._next_obs[idx].clone(),
            "action": action.clone(),
            "reward": self._reward[idx].clone(),
            "terminated": self._terminated[idx].clone(),
            "truncated": self._truncated[idx].clone(),
            "episode_id": self._episode_id[idx].clone(),
            "step_id": self._step_id[idx].clone(),
            "behavior_log_prob": self._behavior_log_prob[idx].clone(),
            "control_log_prob": self._control_log_prob[idx].clone(),
            "source": self._source[idx].clone(),
            "policy_version": self._policy_version[idx].clone(),
            "encoder_version": self._encoder_version[idx].clone(),
            "priority_ig": self._priority_ig[idx].clone(),
            "priority_gap": self._priority_gap[idx].clone(),
            "priority_pair": self._priority_pair[idx].clone(),
            "priority_geom": self._priority_geom[idx].clone(),
            "indices": idx.clone(),
        }

    def set_priorities(
        self,
        indices: torch.Tensor,
        ig: torch.Tensor,
        gap: torch.Tensor,
        pair: torch.Tensor,
        geom: torch.Tensor,
    ) -> None:
        idx = indices.cpu()
        self._priority_ig[idx] = ig.detach().cpu()
        self._priority_gap[idx] = gap.detach().cpu()
        self._priority_pair[idx] = pair.detach().cpu()
        self._priority_geom[idx] = geom.detach().cpu()

    def priority_components(self) -> dict[str, torch.Tensor]:
        n = self.size
        return {
            "ig": self._priority_ig[:n].clone(),
            "gap": self._priority_gap[:n].clone(),
            "pair": self._priority_pair[:n].clone(),
            "geom": self._priority_geom[:n].clone(),
        }

    def all_actions(self) -> torch.Tensor:
        action = self._action[: self.size]
        if action.shape[-1] == 1:
            action = action.squeeze(-1)
        return action.clone()

    def all_obs(self) -> torch.Tensor:
        return self._obs[: self.size].clone()

    def all_rewards(self) -> torch.Tensor:
        return self._reward[: self.size].clone()

    def all_sources(self) -> torch.Tensor:
        return self._source[: self.size].clone()
