"""NatureCNN discrete policy with locked actor_preactivation hook."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.architectures.encoders.nature_cnn import NatureCNN, layer_init


class NaturePolicy(nn.Module):
    """Separate actor NatureCNN. Headline φ = flattened conv features before 512 Linear."""

    def __init__(self, obs_shape: tuple[int, int, int], action_dim: int):
        super().__init__()
        self.action_space_type = "discrete"
        self.cnn = NatureCNN(obs_shape)
        self.fc = nn.Sequential(
            layer_init(nn.Linear(self.cnn.flat_dim, 512)),
            nn.ReLU(),
        )
        self.action_head = layer_init(nn.Linear(512, action_dim), std=0.01)

    def actor_preactivation(self, obs: torch.Tensor) -> torch.Tensor:
        """Locked headline representation (Moalla penultimate pre-activation)."""
        return self.cnn(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        pre = self.actor_preactivation(obs)
        return self.action_head(self.fc(pre))

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        single = obs.dim() == 3
        if single:
            obs = obs.unsqueeze(0)
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        if single:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)
        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy()
