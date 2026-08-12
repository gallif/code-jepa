"""EMA target encoder utilities (V-JEPA-style)."""

from __future__ import annotations

import copy
import math

import torch.nn as nn


class EMATarget(nn.Module):
    """Exponential moving average copy of an online module; always eval, no grads."""

    def __init__(self, online: nn.Module, momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum
        self.module = copy.deepcopy(online)
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, online: nn.Module) -> None:
        m = self.momentum
        for p_t, p_o in zip(self.module.parameters(), online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)
        for b_t, b_o in zip(self.module.buffers(), online.buffers()):
            b_t.data.copy_(b_o.data)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def ema_momentum_cosine(step: int, total_steps: int, m_start: float = 0.996, m_end: float = 1.0) -> float:
    """Cosine schedule from m_start → m_end (V-JEPA-style)."""
    if total_steps <= 0:
        return m_end
    t = min(max(step / total_steps, 0.0), 1.0)
    return m_end - (m_end - m_start) * (1 + math.cos(math.pi * t)) / 2
