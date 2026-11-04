from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """Sketched isotropic-Gaussian regularizer (LeJEPA / H-JEPA)."""

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / max(knots - 1, 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.numel() == 0 or embeddings.shape[0] < 2:
            return embeddings.new_zeros(())
        proj = embeddings.float()
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device, dtype=proj.dtype)
        a = a / a.norm(p=2, dim=0).clamp_min(1e-8)
        x_t = (proj @ a).unsqueeze(-1) * self.t
        cos = x_t.cos().mean(dim=0)
        sin = x_t.sin().mean(dim=0)
        err = (cos - self.phi).square() + sin.square()
        statistic = (err @ self.weights) * proj.size(0)
        return statistic.mean()


def pairwise_distance(pred: torch.Tensor, target: torch.Tensor, kind: str = "cosine") -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    if kind == "cosine":
        return 1.0 - F.cosine_similarity(pred, target, dim=-1).mean()
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "l2":
        return torch.linalg.norm(pred - target, ord=2, dim=-1).mean()
    raise ValueError(f"unknown distance {kind}")
