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


def test_sigreg_bank_accepts_single_vector():
    from code_jepa.losses import EmbeddingBank, SIGReg

    sig = SIGReg(num_proj=8, knots=5)
    bank = EmbeddingBank(size=8)
    first = bank.regularize(torch.randn(1, 4), sig, training=True)
    assert first.item() == 0.0
    second = bank.regularize(torch.randn(1, 4), sig, training=True)
    assert torch.isfinite(second)
    assert bank._count() == 2


def test_view_shapes(cfg, tokenizer):
    backbone = build_tiny_backbone(len(tokenizer))
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone)
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
    batch = collate_packed([dataset[0]], tokenizer.pad_token_id)
    pred, target = model.view_forward(batch)
    assert pred.shape[-1] == cfg.projection_head.dim
    assert pred.shape == target.shape
    assert batch["valid_nodes"].sum().item() == 14


def _special_row_grads(cfg, tokenizer, **objectives):
    """Backward one batch; return |grad| summed over node-token and pred-token embedding rows."""
    import copy

    from code_jepa.tokenize_utils import SPECIAL_NODE_TOKENS, SPECIAL_PRED_TOKENS

    cfg = copy.deepcopy(cfg)
    obj = cfg.objectives
    obj.jepa_dropout = 0.0
    obj.lm.weight = 0.0
    obj.view.weight = objectives.get("view_weight", 0.0)
    obj.view.stop_grad_target = objectives.get("stop_grad", True)
    obj.view.direction = objectives.get("direction", "text_to_code")
    obj.sigreg.weight = objectives.get("sigreg_weight", 0.0)
    obj.sigreg.side = objectives.get("sigreg_on", "code")
    obj.sigreg.bank_size = 0
    torch.manual_seed(0)
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
    model.train()
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
    batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
    losses = model.compute_losses(batch)
    emb = model.backbone.get_input_embeddings().weight
    if losses["total"].requires_grad:
        losses["total"].backward()
    grad = emb.grad if emb.grad is not None else torch.zeros_like(emb)
    node_ids = tokenizer.convert_tokens_to_ids(list(SPECIAL_NODE_TOKENS.values()))
    pred_ids = tokenizer.convert_tokens_to_ids(list(SPECIAL_PRED_TOKENS.values()))
    return float(grad[node_ids].abs().sum()), float(grad[pred_ids].abs().sum())


def test_sigreg_on_target_is_inert_under_stop_grad(cfg, tokenizer):
    node_g, pred_g = _special_row_grads(cfg, tokenizer, sigreg_weight=1.0, sigreg_on="target")
    assert node_g == 0.0 and pred_g == 0.0


def test_sigreg_on_code_reaches_node_tokens_both_directions(cfg, tokenizer):
    for direction in ("text_to_code", "code_to_text"):
        node_g, pred_g = _special_row_grads(
            cfg, tokenizer, sigreg_weight=1.0, sigreg_on="code", direction=direction
        )
        assert node_g > 0.0, direction
        assert pred_g == 0.0, direction


def test_code_to_text_stop_grad_trains_code_side_only(cfg, tokenizer):
    node_g, pred_g = _special_row_grads(cfg, tokenizer, view_weight=1.0, direction="code_to_text")
    assert node_g > 0.0
    assert pred_g == 0.0
    node_g, pred_g = _special_row_grads(cfg, tokenizer, view_weight=1.0, direction="text_to_code")
    assert node_g == 0.0
    assert pred_g > 0.0


def test_multi_unit_split_survives_level_filtering(cfg, tokenizer):
    import copy

    from code_jepa.metrics import representation_metrics

    cfg = copy.deepcopy(cfg)
    cfg.hierarchy.node_levels = ["approach", "intent"]
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, cfg, split=None)
    batch = collate_packed([dataset[0]], tokenizer.pad_token_id)
    assert int(batch["n_units"][0]) >= 2  # fixture is a two-unit program
    assert not (batch["level_ids"] == 1).any()
    emb = torch.randn(*batch["valid_nodes"].shape, 8)
    m = representation_metrics(emb, emb, batch["valid_nodes"], batch["level_ids"], n_units=batch["n_units"])
    assert m["n_approach_multi"] == 1 and m["n_approach_single"] == 0
    assert abs(m["cos_approach_multi"] - 1.0) < 1e-5
