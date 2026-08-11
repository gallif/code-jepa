from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        h, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.ln1(x + h)
        x = self.ln2(x + self.ff(x))
        return x


class SigPredictor(nn.Module):
    """Small student: token hidden states (or embeddings) → predicted body latent."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(d_in)
        self.proj_in = nn.Linear(d_in, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.proj_out = nn.Linear(d_model, d_out)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B, T, D], attention_mask: [B, T] with 1=keep
        x = self.proj_in(self.in_norm(hidden.float()))
        pad = attention_mask == 0
        for block in self.blocks:
            x = block(x, key_padding_mask=pad)
        mask = attention_mask.float().unsqueeze(-1)
        pooled = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj_out(pooled)

    def forward(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(hidden, attention_mask)


def info_nce(pred: torch.Tensor, target: torch.Tensor, logit_scale: torch.Tensor) -> torch.Tensor:
    """In-batch InfoNCE; pred/target [B, D]."""
    pred = F.normalize(pred.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    scale = logit_scale.exp().clamp(max=100.0)
    logits = scale * pred @ target.t()
    labels = torch.arange(pred.size(0), device=pred.device)
    return F.cross_entropy(logits, labels)
