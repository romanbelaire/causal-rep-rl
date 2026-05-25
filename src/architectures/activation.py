"""Named activation modules for architecture builders."""

import torch.nn as nn


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    raise ValueError(
        f"Unknown activation: {name}. Use relu, gelu, tanh, or elu."
    )
