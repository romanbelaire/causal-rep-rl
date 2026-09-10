"""Apply spectral normalization to critic value heads."""

from __future__ import annotations

import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


def apply_value_head_spectral_norm(critic: nn.Module) -> int:
    """
    Wrap every nn.Linear inside critic.value_head with spectral_norm.

    Returns the number of Linear layers parametrized. Raises if value_head is
    missing or has no Linear modules.
    """
    if not hasattr(critic, "value_head"):
        raise RuntimeError("critic has no value_head for spectral_norm")
    head = critic.value_head
    n = 0
    if isinstance(head, nn.Linear):
        critic.value_head = spectral_norm(head)
        return 1
    if isinstance(head, nn.Sequential):
        for i, mod in enumerate(head):
            if isinstance(mod, nn.Linear):
                head[i] = spectral_norm(mod)
                n += 1
        if n == 0:
            raise RuntimeError("value_head Sequential has no Linear layers")
        return n
    raise RuntimeError(f"unsupported value_head type for spectral_norm: {type(head)}")
