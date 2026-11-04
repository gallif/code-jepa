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
) -> dict[str, float]:
    """Cosine, spread, rank, and within-program level confusion."""
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
        n_units = int((lv == 1).sum().item())
        n = float(hit.numel())
        s = float(hit.sum().item())
        hits_all += s
        n_all += n
        if n_units >= 2:
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
    return metrics
