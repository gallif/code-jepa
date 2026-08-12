"""Multiblock token masking for Program JEPA (V-JEPA-style spans)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class TokenMask:
    """Boolean masks over non-padding tokens. ``masked`` True = predict this position."""

    masked: torch.Tensor  # [B, T] bool
    visible: torch.Tensor  # [B, T] bool  (attention_mask & ~masked)


def sample_multiblock_mask(
    attention_mask: torch.Tensor,
    *,
    mask_ratio: float = 0.4,
    min_block: int = 8,
    max_block: int = 64,
    generator: torch.Generator | None = None,
) -> TokenMask:
    """
    Sample contiguous token blocks to mask (like V-JEPA multiblock, 1D).

    ``attention_mask``: [B, T] with 1 = real token.
    """
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in (0,1), got {mask_ratio}")
    if min_block < 1 or max_block < min_block:
        raise ValueError("need 1 <= min_block <= max_block")

    bsz, tmax = attention_mask.shape
    device = attention_mask.device
    masked = torch.zeros(bsz, tmax, dtype=torch.bool, device=device)

    for b in range(bsz):
        lengths = int(attention_mask[b].sum().item())
        if lengths <= 1:
            continue
        target = max(1, int(math.ceil(mask_ratio * lengths)))
        covered = 0
        hi = min(max_block, lengths)
        lo = min(min_block, hi)
        # Cap attempts so tiny sequences still terminate
        for _ in range(lengths * 4):
            if covered >= target:
                break
            blk = int(torch.randint(lo, hi + 1, (1,), generator=generator).item())
            start_max = lengths - blk
            start = int(torch.randint(0, start_max + 1, (1,), generator=generator).item())
            end = start + blk
            newly = int((~masked[b, start:end]).sum().item())
            masked[b, start:end] = True
            covered += newly

        # Ensure at least one visible and one masked token when possible
        if masked[b, :lengths].all():
            masked[b, 0] = False
        if not masked[b, :lengths].any() and lengths > 1:
            masked[b, lengths // 2] = True

    visible = attention_mask.bool() & ~masked
    # Never mask padding
    masked = masked & attention_mask.bool()
    return TokenMask(masked=masked, visible=visible)
