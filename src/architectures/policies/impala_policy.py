"""Impala CNN policy for Procgen PPO baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.architectures.encoders.impala_cnn import ProcgenImpalaEncoder


class ImpalaPolicy(nn.Module):
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        action_space_type: str = "discrete",
        emb_size: int = 256,
        depths: tuple[int, ...] = (16, 32, 32),
    ):
        super().__init__()
        self.action_space_type = action_space_type
        self.encoder = ProcgenImpalaEncoder(obs_shape, depths=depths, emb_size=emb_size)
        if action_space_type == "discrete":
            self.action_head = nn.Linear(emb_size, action_dim)
        else:
            self.action_mean = nn.Linear(emb_size, action_dim)
            self.action_log_std = nn.Linear(emb_size, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        if self.action_space_type == "discrete":
            return self.action_head(features)
        mean = self.action_mean(features)
        log_std = self.action_log_std(features)
        return mean, torch.clamp(log_std, min=-20, max=2)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        else:
            mean, log_std = self.forward(obs)
            dist = torch.distributions.Normal(mean, torch.exp(log_std))
            action = mean if deterministic else dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            return dist.log_prob(actions), dist.entropy()
        mean, log_std = self.forward(obs)
        dist = torch.distributions.Normal(mean, torch.exp(log_std))
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)
