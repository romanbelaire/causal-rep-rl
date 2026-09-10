"""Frozen target copies. Hard copy at phase boundaries; EMA is an ablation."""

import copy

import torch
import torch.nn as nn


class TargetNetwork:
    """Stop-gradient clone. Raises if updated while a collection phase is active."""

    def __init__(self, module: nn.Module):
        self.module = copy.deepcopy(module)
        self.module.eval()
        for param in self.module.parameters():
            param.requires_grad = False
        self.collection_frozen = False

    def begin_collection(self) -> None:
        self.collection_frozen = True

    def end_collection(self) -> None:
        self.collection_frozen = False

    def _assert_not_collecting(self) -> None:
        if self.collection_frozen:
            raise RuntimeError("target networks are frozen inside a collection phase")

    def hard_copy_from(self, source: nn.Module) -> None:
        self._assert_not_collecting()
        self.module.load_state_dict(source.state_dict())

    def ema_copy_from(self, source: nn.Module, tau: float) -> None:
        self._assert_not_collecting()
        with torch.no_grad():
            for target_param, param in zip(self.module.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)
            for target_buf, buf in zip(self.module.buffers(), source.buffers()):
                target_buf.copy_(buf)
