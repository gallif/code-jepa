from pathlib import Path

import torch

from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed
from code_jepa.model import HierarchicalCodeJEPA, build_tiny_backbone

FIXTURE = Path(__file__).resolve().parent / "fixtures/sample.jsonl"


def test_tiny_forward_finite(cfg, tokenizer):
    cfg.objectives.jepa_dropout = 0.0
    cfg.objectives.sigreg.weight = 0.1
    backbone = build_tiny_backbone(len(tokenizer))
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone)
    model.eval()
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
    batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
    losses = model.compute_losses(batch)
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["view"])
    assert torch.isfinite(losses["lm"])
    losses["total"].backward()
    grad = next(p for p in model.heads["unit"].parameters() if p.grad is not None)
    assert torch.isfinite(grad).all()


def test_e1_skips_view_loss(cfg, tokenizer):
    original = (cfg.objectives.view.weight, cfg.objectives.sigreg.weight)
    cfg.objectives.view.weight = 0.0
    cfg.objectives.sigreg.weight = 0.0
    try:
        backbone = build_tiny_backbone(len(tokenizer))
        model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone)
        model.train()
        dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
        batch = collate_packed([dataset[0]], tokenizer.pad_token_id)
        losses = model.compute_losses(batch)
        assert losses["view"].item() == 0.0
        assert "pred" not in losses
    finally:
        cfg.objectives.view.weight, cfg.objectives.sigreg.weight = original


def test_view_shapes(cfg, tokenizer):
    backbone = build_tiny_backbone(len(tokenizer))
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone)
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
    batch = collate_packed([dataset[0]], tokenizer.pad_token_id)
    pred, target = model.view_forward(batch)
    assert pred.shape[-1] == cfg.projection_head.dim
    assert pred.shape == target.shape
    assert batch["valid_nodes"].sum().item() == 14
