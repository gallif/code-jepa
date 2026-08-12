"""Unit tests for multiblock masking."""

import torch

from code_jepa.masking import sample_multiblock_mask


def test_mask_respects_padding_and_ratio():
    attn = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        ]
    )
    mask = sample_multiblock_mask(attn, mask_ratio=0.4, min_block=2, max_block=4)
    assert mask.masked.shape == attn.shape
    # no padding masked
    assert not (mask.masked & ~attn.bool()).any()
    # visible is complement on real tokens
    real = attn.bool()
    assert torch.equal(mask.visible, real & ~mask.masked)
    for b in range(attn.size(0)):
        n = int(attn[b].sum())
        m = int(mask.masked[b, :n].sum())
        assert 1 <= m < n
