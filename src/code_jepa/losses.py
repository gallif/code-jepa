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


class EmbeddingBank:
    """FIFO of detached embeddings so SIGReg sees more than one micro-batch."""

    def __init__(self, size: int):
        self.size = size
        self.chunks: list[torch.Tensor] = []

    def _count(self) -> int:
        return sum(chunk.size(0) for chunk in self.chunks)

    def regularize(self, current: torch.Tensor, sigreg: SIGReg, training: bool) -> torch.Tensor:
        if current.numel() == 0:
            return current.new_zeros(())
        pieces = [current]
        if self.chunks:
            pieces.append(torch.cat(self.chunks, dim=0).detach())
        pooled = torch.cat(pieces, dim=0)
        if pooled.size(0) < 2:
            loss = current.new_zeros(())
        else:
            # Keep the gradient scale close to a current-batch-only SIGReg term.
            loss = sigreg(pooled) * (current.size(0) / pooled.size(0))
        if training and self.size > 0:
            self.chunks.append(current.detach())
            while self._count() > self.size and self.chunks:
                self.chunks.pop(0)
        return loss


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
