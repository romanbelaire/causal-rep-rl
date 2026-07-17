"""OpenAI Procgen-style Impala CNN encoder."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImpalaResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv0(x))
        out = self.conv1(out)
        return F.relu(x + out)


class ImpalaConvSequence(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ImpalaResidualBlock(out_channels),
            ImpalaResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def obs_to_chw(obs: torch.Tensor) -> torch.Tensor:
    if obs.dim() == 3:
        obs = obs.unsqueeze(0)
    if obs.shape[-1] in (1, 3, 4):
        return obs.permute(0, 3, 1, 2).contiguous()
    return obs


class ProcgenImpalaEncoder(nn.Module):
    """Impala CNN trunk: HWC obs -> feature vector."""

    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        depths: tuple[int, ...] = (16, 32, 32),
        emb_size: int = 256,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        _, _, channels = obs_shape

        convs = []
        in_channels = channels
        for depth in depths:
            convs.append(ImpalaConvSequence(in_channels, depth))
            in_channels = depth
        self.convs = nn.Sequential(*convs)

        with torch.no_grad():
            dummy = torch.zeros(1, channels, obs_shape[0], obs_shape[1])
            flat_dim = self.convs(dummy).view(1, -1).shape[1]
        self.fc = nn.Linear(flat_dim, emb_size)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = obs_to_chw(obs)
        x = self.convs(x)
        return self.fc(x.view(x.shape[0], -1))
