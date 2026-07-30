"""
Standard MLP Policy for discrete and continuous action spaces.
"""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network.

    Supports both discrete and continuous action spaces.
    For discrete: outputs logits for categorical distribution
    For continuous: Gaussian with state-independent log_std (CleanRL-style)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: list[int] = [256, 256],
        activation: str = "relu",
        action_space_type: str = "discrete",  # "discrete" or "continuous"
        action_low: float = -1.0,
        action_high: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space_type = action_space_type
        self.action_low = float(action_low)
        self.action_high = float(action_high)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        layers = []
        input_dim = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation)
            input_dim = hidden_size

        self.feature_extractor = nn.Sequential(*layers)

        if action_space_type == "discrete":
            self.action_head = nn.Linear(input_dim, action_dim)
            nn.init.orthogonal_(self.action_head.weight, gain=0.01)
            nn.init.zeros_(self.action_head.bias)
        else:
            self.action_mean = nn.Linear(input_dim, action_dim)
            # CleanRL: state-independent log std (one vector shared across states).
            self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
            nn.init.orthogonal_(self.action_mean.weight, gain=0.01)
            nn.init.zeros_(self.action_mean.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(obs)

        if self.action_space_type == "discrete":
            return self.action_head(features)

        mean = self.action_mean(features)
        log_std = self.action_log_std.expand_as(mean)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        return mean, log_std

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action. Accepts [obs_dim] or [batch, obs_dim]; returns matching rank."""
        squeeze_batch = obs.dim() == 1
        if squeeze_batch:
            obs = obs.unsqueeze(0)

        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
        else:
            mean, log_std = self.forward(obs)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = mean if deterministic else dist.sample()
            # Log-prob of the pre-clip sample (CleanRL ClipAction pattern).
            log_prob = dist.log_prob(action).sum(dim=-1)

        if squeeze_batch:
            action = action.squeeze(0)
            log_prob = log_prob.squeeze(0)

        return action, log_prob

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.action_space_type == "discrete":
            logits = self.forward(obs)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy

        if actions.dim() != 2:
            raise ValueError(
                f"continuous actions must be [batch, action_dim], got shape {tuple(actions.shape)}"
            )
        mean, log_std = self.forward(obs)
        if not torch.isfinite(mean).all():
            raise RuntimeError(
                f"Non-finite continuous policy mean "
                f"(nan={torch.isnan(mean).any().item()}, inf={torch.isinf(mean).any().item()})"
            )
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_probs, entropy
