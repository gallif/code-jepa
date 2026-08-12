"""Helpers to hide masked tokens from the online encoder."""

from __future__ import annotations

import torch


def hide_masked_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    masked: torch.Tensor,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Replace masked token ids with pad and drop them from attention so the
    online encoder cannot see masked content (JEPA context view).
    """
    ids = input_ids.clone()
    ids = ids.masked_fill(masked, pad_token_id)
    ctx_mask = attention_mask.clone()
    ctx_mask = ctx_mask.masked_fill(masked, 0)
    return ids, ctx_mask
