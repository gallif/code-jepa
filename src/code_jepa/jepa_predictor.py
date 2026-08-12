"""Small JEPA predictor: visible token features + mask tokens → masked targets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from code_jepa.student import TransformerBlock


class JEPAPredictor(nn.Module):
    """
    V-JEPA-style predictor over a 1D token sequence.

    Builds a full-length sequence: visible positions keep projected encoder
    features; masked positions get a learned mask token + position embedding.
    A small Transformer then predicts target features at masked indices.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        max_positions: int = 2048,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.proj_in = nn.Linear(d_in, d_model)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed = nn.Embedding(max_positions, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj_out = nn.Linear(d_model, d_in)

    def forward(
        self,
        context_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        masked: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            context_hidden: [B, T, D] encoder features (visible positions meaningful)
            attention_mask: [B, T] 1 = real token
            masked: [B, T] bool, True = positions to predict

        Returns:
            pred_full: [B, T, D] predictions (only masked positions are trained)
        """
        bsz, tmax, _ = context_hidden.shape
        device = context_hidden.device
        pos = torch.arange(tmax, device=device).unsqueeze(0).expand(bsz, -1)
        pos = pos.clamp(max=self.pos_embed.num_embeddings - 1)

        x_vis = self.proj_in(context_hidden.float())
        x_mask = self.mask_token.expand(bsz, tmax, -1)
        m = masked.unsqueeze(-1)
        x = torch.where(m, x_mask, x_vis)
        x = x + self.pos_embed(pos)

        # Padding should not attend
        pad = attention_mask == 0
        for block in self.blocks:
            x = block(x, key_padding_mask=pad)
        x = self.norm(x)
        return self.proj_out(x)


def masked_smooth_l1(
    pred: torch.Tensor,
    target: torch.Tensor,
    masked: torch.Tensor,
    attention_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """Mean SmoothL1 over masked non-padding positions."""
    keep = masked & attention_mask.bool()
    if not keep.any():
        return pred.new_zeros(())
    p = pred[keep].float()
    t = target[keep].float()
    return F.smooth_l1_loss(p, t, beta=beta, reduction="mean")
