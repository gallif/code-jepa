from __future__ import annotations

import torch
import torch.nn.functional as F

LEVEL_NAMES = ("step", "unit", "approach", "intent")


def effective_rank(embeddings: torch.Tensor) -> torch.Tensor:
    if embeddings.ndim != 2 or embeddings.size(0) < 2:
        return embeddings.new_zeros(())
    centered = embeddings.float() - embeddings.float().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered).clamp_min(0)
    denom = singular.square().sum().clamp_min(1e-8)
    return (singular.sum() ** 2) / denom


def representation_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    level_ids: torch.Tensor,
    code: torch.Tensor | None = None,
    text: torch.Tensor | None = None,
    n_units: torch.Tensor | None = None,
) -> dict[str, float]:
    """Cosine, spread, rank, and within-program level separation.

    ``std_*``/``erank_*`` are on the view target (flips with view.direction). When ``code`` and
    ``text`` are given, ``erank_code_*``/``erank_text_*`` pin the side explicitly. ``cos_centered_*``
    subtracts each side's per-level mean first, so a shared offset direction cannot inflate it.
    ``n_units`` (one count per program, from the full tree) adds ``*_{level}_single``/``*_multi``
    splits of cos, cos_centered and n; multi-unit means >= 2 units. Centring uses the all-program
    level mean so the two subsets are on the same scale.
    """
    metrics: dict[str, float] = {}
    pred = pred.float()
    target = target.float()
    for i, name in enumerate(LEVEL_NAMES):
        sel = valid & (level_ids == i)
        n = int(sel.sum().item())
        metrics[f"n_{name}"] = n
        if n == 0:
            continue
        p, t = pred[sel], target[sel]
        metrics[f"cos_{name}"] = float(F.cosine_similarity(p, t, dim=-1).mean().item())
        metrics[f"std_{name}"] = float(t.std(unbiased=False).item()) if n > 1 else 0.0
        metrics[f"erank_{name}"] = float(effective_rank(t).item()) if n > 1 else 0.0
        if n > 1:
            pc, tc = p - p.mean(dim=0, keepdim=True), t - t.mean(dim=0, keepdim=True)
            metrics[f"cos_centered_{name}"] = float(F.cosine_similarity(pc, tc, dim=-1).mean().item())
        if n_units is not None:
            multi_rows = (n_units >= 2).to(valid.device)
            row_of = multi_rows.unsqueeze(1).expand_as(valid)[sel]
            cos_all = F.cosine_similarity(p, t, dim=-1)
            cos_c = F.cosine_similarity(pc, tc, dim=-1) if n > 1 else None
            for tag, m in (("multi", row_of), ("single", ~row_of)):
                k = int(m.sum().item())
                metrics[f"n_{name}_{tag}"] = k
                if k == 0:
                    continue
                metrics[f"cos_{name}_{tag}"] = float(cos_all[m].mean().item())
                if cos_c is not None:
                    metrics[f"cos_centered_{name}_{tag}"] = float(cos_c[m].mean().item())
        if code is not None and text is not None and n > 1:
            for side, emb in (("code", code.float()[sel]), ("text", text.float()[sel])):
                metrics[f"std_{side}_{name}"] = float(emb.std(dim=0, unbiased=False).mean().item())
                metrics[f"erank_{side}_{name}"] = float(effective_rank(emb).item())

    # Level separation (replaces confusion_acc, which counted the matched pair and read 1.0 at
    # step 0). Query = centred code embedding of node v; candidates = centred text embedding of v
    # and of every node of the same program at a *different* level. Hit if v's own description
    # scores strictly highest. Centring is per level over all eval nodes, so a shared per-level
    # offset cannot produce hits. Chance = mean of 1 / (1 + #other-level candidates).
    q_all = (code if code is not None else target).float()
    k_all = (text if text is not None else pred).float()
    qc = torch.zeros_like(q_all)
    kc = torch.zeros_like(k_all)
    for i in range(len(LEVEL_NAMES)):
        sel = valid & (level_ids == i)
        if int(sel.sum().item()) > 1:
            qc[sel] = q_all[sel] - q_all[sel].mean(dim=0, keepdim=True)
            kc[sel] = k_all[sel] - k_all[sel].mean(dim=0, keepdim=True)
    # Rows that centring reduced to float residue carry no signal; zero them so normalising
    # cannot turn rounding noise into a direction.
    for x, ref in ((qc, q_all), (kc, k_all)):
        tiny = x.norm(dim=-1) <= 1e-4 * ref.norm(dim=-1).clamp_min(1e-12)
        x[tiny] = 0.0
    acc = {"all": [0.0, 0.0, 0.0], "multi": [0.0, 0.0, 0.0], "single": [0.0, 0.0, 0.0]}
    bsz = valid.size(0)
    for b in range(bsz):
        idx = valid[b]
        lv = level_ids[b, idx]
        if int(lv.unique().numel()) < 2:
            continue
        qb = F.normalize(qc[b, idx], dim=-1)
        kb = F.normalize(kc[b, idx], dim=-1)
        sims = qb @ kb.T
        own = sims.diagonal()
        other = lv.unsqueeze(1) != lv.unsqueeze(0)
        best_other = sims.masked_fill(~other, float("-inf")).max(dim=-1).values
        hit = (own > best_other).float()
        chance = 1.0 / (1.0 + other.sum(dim=-1).float())
        units_b = int(n_units[b].item()) if n_units is not None else int((lv == 1).sum().item())
        for tag in ("all", "multi" if units_b >= 2 else "single"):
            acc[tag][0] += float(hit.sum())
            acc[tag][1] += float(hit.numel())
            acc[tag][2] += float(chance.sum())
    for tag, suffix in (("all", ""), ("multi", "_multi_unit"), ("single", "_single_unit")):
        hits, n, ch = acc[tag]
        if n:
            metrics[f"level_sep_acc{suffix}"] = hits / n
            metrics[f"level_sep_chance{suffix}"] = ch / n
    metrics["n_programs"] = float(bsz)
    if n_units is not None:
        metrics["n_programs_multi_unit"] = float((n_units >= 2).sum().item())
    return metrics
