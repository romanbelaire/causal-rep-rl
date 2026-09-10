"""Optimizer-ownership diagnostics: grad norms and parameter deltas."""

import torch
import torch.nn as nn


def grad_l2(params: list[nn.Parameter]) -> float:
    total = torch.zeros((), device=params[0].device)
    for p in params:
        total = total + p.grad.detach().pow(2).sum()
    return float(total.sqrt().item())


def param_l2(params: list[nn.Parameter]) -> list[torch.Tensor]:
    return [p.detach().clone() for p in params]


def delta_l2(before: list[torch.Tensor], params: list[nn.Parameter]) -> float:
    total = torch.zeros((), device=params[0].device)
    for b, p in zip(before, params):
        total = total + (p.detach() - b).pow(2).sum()
    return float(total.sqrt().item())
