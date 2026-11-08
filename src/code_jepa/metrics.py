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
    """Cosine, spread, rank, and within-program level confusion.

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

    hits_all = 0.0
    n_all = 0.0
    hits_multi = 0.0
    n_multi = 0.0
    hits_single = 0.0
    n_single = 0.0
    bsz = valid.size(0)
    for b in range(bsz):
        idx = valid[b]
        if int(idx.sum().item()) < 2:
            continue
        c = F.normalize(target[b, idx], dim=-1)
        p = F.normalize(pred[b, idx], dim=-1)
        lv = level_ids[b, idx]
        assigned = (c @ p.T).argmax(dim=-1)
        hit = (lv[assigned] == lv).float()
        units_b = int(n_units[b].item()) if n_units is not None else int((lv == 1).sum().item())
        n = float(hit.numel())
        s = float(hit.sum().item())
        hits_all += s
        n_all += n
        if units_b >= 2:
            hits_multi += s
            n_multi += n
        else:
            hits_single += s
            n_single += n
    if n_all:
        metrics["confusion_acc"] = hits_all / n_all
    if n_multi:
        metrics["confusion_acc_multi_unit"] = hits_multi / n_multi
    if n_single:
        metrics["confusion_acc_single_unit"] = hits_single / n_single
    metrics["n_programs"] = float(bsz)
    if n_units is not None:
        metrics["n_programs_multi_unit"] = float((n_units >= 2).sum().item())
    return metrics
