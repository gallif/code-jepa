from pathlib import Path

import torch

from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed
from code_jepa.model import HierarchicalCodeJEPA, build_tiny_backbone

FIXTURE = Path(__file__).resolve().parent / "fixtures/sample.jsonl"


def cfg_default():
    from code_jepa.config import ExperimentConfig

    return ExperimentConfig()


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
    obj.view.stop_grad = objectives.get("stop_grad", "none")
    obj.view.stop_grad_target = False
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


def test_sigreg_on_target_is_inert_when_target_is_detached_text(cfg, tokenizer):
    node_g, pred_g = _special_row_grads(
        cfg, tokenizer, sigreg_weight=1.0, sigreg_on="target", stop_grad="text", direction="code_to_text"
    )
    assert node_g == 0.0 and pred_g == 0.0


def test_sigreg_on_code_reaches_node_tokens_both_directions(cfg, tokenizer):
    for direction in ("text_to_code", "code_to_text"):
        node_g, pred_g = _special_row_grads(
            cfg, tokenizer, sigreg_weight=1.0, sigreg_on="code", direction=direction
        )
        assert node_g > 0.0, direction
        assert pred_g == 0.0, direction


def test_view_loss_always_trains_code_side(cfg, tokenizer):
    """The trainee (code-node embeddings) must get view-loss gradient in every allowed config."""
    for stop_grad in ("none", "text"):
        for direction in ("text_to_code", "code_to_text"):
            node_g, pred_g = _special_row_grads(
                cfg, tokenizer, view_weight=1.0, stop_grad=stop_grad, direction=direction
            )
            assert node_g > 0.0, (stop_grad, direction)
            if stop_grad == "none":
                assert pred_g > 0.0, (stop_grad, direction)
            else:
                assert pred_g == 0.0, (stop_grad, direction)


def test_stop_grad_on_code_is_rejected(cfg, tokenizer):
    import copy

    import pytest

    from code_jepa.config import resolve_stop_grad

    view = copy.deepcopy(cfg.objectives.view)
    view.stop_grad, view.stop_grad_target, view.direction = "code", False, "text_to_code"
    with pytest.raises(ValueError):
        resolve_stop_grad(view)
    view.stop_grad, view.stop_grad_target = "none", True  # legacy sgcode / stopgrad configs
    with pytest.raises(ValueError):
        resolve_stop_grad(view)
    view.direction = "code_to_text"  # legacy c2t maps to detaching text
    assert resolve_stop_grad(view) == "text"
    bad = copy.deepcopy(cfg)
    bad.objectives.view.stop_grad, bad.objectives.view.stop_grad_target = "none", True
    bad.objectives.view.direction = "text_to_code"
    with pytest.raises(ValueError):
        HierarchicalCodeJEPA(bad, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))


def test_bank_does_not_dilute_sigreg_gradient():
    from code_jepa.losses import EmbeddingBank, SIGReg

    torch.manual_seed(0)
    sig = SIGReg(num_proj=256, knots=17)
    collapsed = lambda n: torch.randn(n, 32) * 0.05 + 3.0  # off-centre, low-variance: far from N(0, I)
    bank = EmbeddingBank(size=256)
    for _ in range(64):
        bank.regularize(collapsed(4), sig, training=True)
    cur = collapsed(4).requires_grad_(True)
    torch.autograd.grad(sig(cur), cur)  # warm-up, same RNG use pattern
    torch.manual_seed(1)
    g_batch = torch.autograd.grad(sig(cur), cur)[0].norm(dim=-1).mean()
    torch.manual_seed(1)
    g_bank = torch.autograd.grad(bank.regularize(cur, sig, training=False), cur)[0].norm(dim=-1).mean()
    ratio = float(g_bank / g_batch)
    assert 0.3 < ratio < 3.0, ratio


def test_separate_text_heads(cfg, tokenizer):
    import copy

    c = copy.deepcopy(cfg)
    c.projection_head.shared = False
    c.objectives.jepa_dropout = 0.0
    model = HierarchicalCodeJEPA(c, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
    assert model.text_heads is not None
    dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)
    batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
    losses = model.compute_losses(batch)
    losses["total"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.heads.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.text_heads.parameters())


def test_grad_diag_reports_contract(cfg, tokenizer):
    import copy

    for stop_grad in ("none", "text"):
        c = copy.deepcopy(cfg)
        c.objectives.jepa_dropout = 0.0
        c.objectives.view.weight = 1.0
        c.objectives.view.stop_grad = stop_grad
        c.objectives.sigreg.weight = 0.3
        c.objectives.sigreg.side = "both"
        c.objectives.sigreg.bank_size = 16
        model = HierarchicalCodeJEPA(c, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
        model.train()
        dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)
        batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
        d = model.compute_losses(batch, grad_diag=True)["grad_diag"]
        for level in ("step", "unit", "approach", "intent"):
            assert d[f"gn_view_code_{level}"] > 0.0, (stop_grad, level)
            assert d[f"gn_sigreg_code_{level}"] > 0.0, (stop_grad, level)
            assert d[f"gn_sigreg_text_{level}"] > 0.0, (stop_grad, level)
            if stop_grad == "text":
                assert d[f"gn_view_text_{level}"] == 0.0, level
            else:
                assert d[f"gn_view_text_{level}"] > 0.0, level


def test_level_sep_ignores_shared_level_offset():
    from code_jepa.metrics import representation_metrics

    torch.manual_seed(0)
    n_prog = 300
    levels = torch.tensor([0, 0, 0, 1, 1, 2, 3]).repeat(n_prog, 1)
    valid = torch.ones_like(levels, dtype=torch.bool)
    offset = torch.randn(4, 16) * 10
    # Level-only signal: identical rows collapse to ~0 after centring -> no hits.
    lvl = offset[levels]
    m = representation_metrics(lvl, lvl, valid, levels, code=lvl, text=lvl)
    assert m["level_sep_acc"] == 0.0
    # Level offset + independent code/text noise: no node information -> about chance.
    code = lvl + torch.randn(n_prog, 7, 16)
    text = lvl + torch.randn(n_prog, 7, 16)
    m = representation_metrics(text, code, valid, levels, code=code, text=text)
    assert abs(m["level_sep_acc"] - m["level_sep_chance"]) < 0.06, (m["level_sep_acc"], m["level_sep_chance"])
    # Shared node signal on both sides -> near-perfect.
    node = torch.randn(n_prog, 7, 16) * 3
    m = representation_metrics(lvl + node, lvl + node + 0.1 * torch.randn_like(node), valid, levels,
                               code=lvl + node, text=lvl + node + 0.1 * torch.randn_like(node))
    assert m["level_sep_acc"] > 0.95


def test_centered_view_loss_ignores_shared_offset():
    import copy

    c = copy.deepcopy(cfg_default())
    c.objectives.view.center = "ema"
    c.objectives.view.center_momentum = 0.0  # mean = current batch mean
    model = HierarchicalCodeJEPA.__new__(HierarchicalCodeJEPA)
    torch.nn.Module.__init__(model)
    model.cfg, model.centers = c, {}
    model.train()
    torch.manual_seed(0)
    p, t = torch.randn(6, 8), torch.randn(6, 8)
    offset = torch.randn(8) * 50
    from code_jepa.losses import pairwise_distance

    base = pairwise_distance(model._center(p, ("pred", "x")), model._center(t, ("target", "x")))
    model.centers = {}
    shifted = pairwise_distance(model._center(p + offset, ("pred", "x")), model._center(t + offset, ("target", "x")))
    assert torch.allclose(base, shifted, atol=1e-5)
    assert pairwise_distance(p + offset, t + offset) < 0.05  # raw cosine is fooled by the offset


def test_infonce_and_centered_train_code_side(cfg, tokenizer):
    import copy

    for distance, center in (("infonce", "none"), ("cosine", "ema")):
        c = copy.deepcopy(cfg)
        c.objectives.jepa_dropout = 0.0
        c.objectives.lm.weight = 0.0
        c.objectives.sigreg.weight = 0.0
        c.objectives.view.weight = 1.0
        c.objectives.view.distance = distance
        c.objectives.view.center = center
        c.objectives.view.infonce_bank = 8
        model = HierarchicalCodeJEPA(c, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
        model.train()
        dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)
        batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
        model.compute_losses(batch)  # fills the bank / centres
        if center == "ema":  # the batch repeats one program, so seed means that do not zero the rows
            for side in ("pred", "target"):
                for level in ("step", "unit", "approach", "intent"):
                    model.centers[(side, level)] = torch.randn(c.projection_head.dim) * 0.1
        d = model.compute_losses(batch, grad_diag=True)
        assert torch.isfinite(d["total"]), distance
        for level in ("step", "unit", "approach", "intent"):
            assert d["grad_diag"][f"gn_view_code_{level}"] > 0.0, (distance, level)


def test_sigreg_level_weights(cfg, tokenizer):
    import copy

    def run(level_weights, weight=0.3):
        c = copy.deepcopy(cfg)
        c.objectives.jepa_dropout = 0.0
        c.objectives.lm.weight = 0.0
        c.objectives.view.weight = 0.0
        c.objectives.sigreg.weight = weight
        c.objectives.sigreg.side = "both"
        c.objectives.sigreg.bank_size = 0
        c.objectives.sigreg.level_weights = level_weights
        torch.manual_seed(0)
        model = HierarchicalCodeJEPA(c, tokenizer, backbone=build_tiny_backbone(len(tokenizer)))
        model.train()
        dataset = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)
        batch = collate_packed([dataset[0], dataset[0]], tokenizer.pad_token_id)
        torch.manual_seed(1)
        return model.compute_losses(batch, grad_diag=True)

    base = run(None)
    same = run({"step": 0.3, "unit": 0.3, "approach": 0.3, "intent": 0.3})
    assert torch.allclose(base["total"], same["total"], rtol=1e-5)
    only_prog = run({"step": 0.0, "unit": 0.0, "approach": 1.0, "intent": 1.0})
    d = only_prog["grad_diag"]
    assert d["gn_sigreg_code_step"] == 0.0 and d["gn_sigreg_code_unit"] == 0.0
    assert d["gn_sigreg_code_approach"] > 0.0 and d["gn_sigreg_code_intent"] > 0.0


def test_program_tokens_all_code_mask(cfg, tokenizer):
    import copy

    seen = {}
    for mode in ("tree", "all_code"):
        c = copy.deepcopy(cfg)
        c.hierarchy.program_tokens = mode
        ex = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)[0]
        n_code = min(ex.node_seq_pos)
        seen[mode] = {
            (lv, i): int((ex.view_mask[pos, :n_code] == 0).sum())
            for i, (lv, pos) in enumerate(zip(ex.node_levels, ex.node_seq_pos))
        }
    for (lv, i), n in seen["all_code"].items():
        if lv in ("approach", "intent"):
            assert seen["tree"][(lv, i)] < n  # tree: header lines only; all_code: every code token
        else:
            assert seen["tree"][(lv, i)] == n  # steps/units unchanged


def test_shuffle_descriptions_permutes_within_level_only(cfg, tokenizer):
    import copy

    c = copy.deepcopy(cfg)
    c.data.shuffle_seed = 3
    true = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)[0]
    shuf = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None, shuffle_descriptions=True)[0]
    by_level = lambda ex: {lv: sorted(n.text for n in ex.nodes if n.level == lv) for lv in ("step", "unit")}
    assert by_level(true) == by_level(shuf)  # same texts per level
    assert [n.text for n in true.nodes] != [n.text for n in shuf.nodes]  # but re-paired
    assert true.lm_input_ids == shuf.lm_input_ids  # LM prompt keeps the true texts
    c.data.shuffle_descriptions = True  # the config flag only applies to the training split
    c.data.split = None
    assert HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None).shuffled
    c.data.split = "train"
    assert not HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None).shuffled


def test_ground_program_targets_approach_token(cfg, tokenizer):
    import copy

    c = copy.deepcopy(cfg)
    span = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)[0]
    c.hierarchy.ground = "program"
    prog = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)[0]
    assert prog.node_levels == span.node_levels  # same descriptions
    assert len(prog.pred_seq_pos) == len(span.pred_seq_pos)
    appr = [p for lv, p in zip(prog.node_levels, prog.node_seq_pos) if lv == "approach"][0]
    for lv, p in zip(prog.node_levels, prog.node_seq_pos):
        if lv in ("step", "unit"):
            assert p == appr
    n_code = min(prog.node_seq_pos)
    assert int((prog.view_mask[appr, :n_code] == 0).sum()) == n_code  # program token reads all code


def test_program_tokens_hybrid_is_union(cfg, tokenizer):
    import copy

    rows = {}
    for mode in ("tree", "all_code", "units_and_code"):
        c = copy.deepcopy(cfg)
        c.hierarchy.program_tokens = mode
        ex = HierarchicalJsonlDataset(FIXTURE, tokenizer, c, split=None)[0]
        rows[mode] = {lv: ex.view_mask[p] == 0 for lv, p in zip(ex.node_levels, ex.node_seq_pos) if lv in ("approach", "intent")}
    for lv in ("approach", "intent"):
        assert torch.equal(rows["units_and_code"][lv], rows["tree"][lv] | rows["all_code"][lv])
        assert rows["units_and_code"][lv].sum() > rows["all_code"][lv].sum()  # also sees the unit tokens


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
