#!/usr/bin/env python
"""Frozen-backbone linear probes: a fair per-level comparison of E0 / E1 / E2 / E4.

For each model, take last-layer hidden states (no projection heads) at the node tokens (code side)
and at the description predictor tokens (text side). Fit a ridge map text -> code per level on train
programs, choosing lambda on a held-out 20%, and score on val:
  - cos_centered: mean cosine(predicted code state, true code state), both centred with train means
  - R@1 / R@5: retrieve the true node among all val nodes of the same level
  - *_multi: the same on multi-unit programs only
Also reports eval_lm on the val loader (mean of per-batch losses, like train.evaluate).

  python scripts/probe_eval.py --config configs/tzag_p3_e4_s82.yaml --out probes.json \
      --model E0=pretrained --model E4_s82=/path/last.pt ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from torch.utils.data import DataLoader  # noqa: E402

from code_jepa.config import ExperimentConfig  # noqa: E402
from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed  # noqa: E402
from code_jepa.model import LEVEL_NAMES, HierarchicalCodeJEPA  # noqa: E402
from code_jepa.tokenize_utils import load_tokenizer  # noqa: E402

LAMBDAS = [1e-2, 1e-1, 1, 10, 100, 1000]


@torch.no_grad()
def extract(model, loader, device):
    out = {lv: {"code": [], "text": [], "multi": []} for lv in LEVEL_NAMES}
    lm = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        hidden = model._forward_hidden(
            batch["view_input_ids"], {"full_attention": batch["view_attention_mask"]}, batch["view_position_ids"]
        )
        code = model._gather(hidden, batch["node_pos"]).float()
        text = model._gather(hidden, batch["pred_pos"]).float()
        valid, levels = batch["valid_nodes"], batch["level_ids"]
        multi = (batch["n_units"] >= 2).unsqueeze(1).expand_as(valid)
        for i, lv in enumerate(LEVEL_NAMES):
            sel = valid & (levels == i)
            if sel.any():
                out[lv]["code"].append(code[sel].cpu())
                out[lv]["text"].append(text[sel].cpu())
                out[lv]["multi"].append(multi[sel].cpu())
        lm.append(float(model.lm_forward(batch)))
    res = {}
    for lv, d in out.items():
        if d["code"]:
            res[lv] = {k: torch.cat(v) for k, v in d.items()}
    return res, sum(lm) / max(len(lm), 1)


def ridge(x, y, lam):
    xtx = x.T @ x
    return torch.linalg.solve(xtx + lam * torch.eye(xtx.size(0)) * xtx.diagonal().mean(), x.T @ y)


def scores(pred, true):
    p, t = F.normalize(pred, dim=-1), F.normalize(true, dim=-1)
    cos = (p * t).sum(-1)
    rank = ((p @ t.T) > cos.unsqueeze(1)).sum(-1)  # nodes scored above the true one
    return cos, rank


def probe(train, val):
    tx, ty = train["text"].double(), train["code"].double()
    mx, my = tx.mean(0), ty.mean(0)
    tx, ty = tx - mx, ty - my
    n = tx.size(0)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    cut = int(0.8 * n)
    a, b = perm[:cut], perm[cut:]
    best = max(LAMBDAS, key=lambda lam: scores(tx[b] @ ridge(tx[a], ty[a], lam), ty[b])[0].mean().item())
    w = ridge(tx, ty, best)
    vx, vy = val["text"].double() - mx, val["code"].double() - my
    cos, rank = scores(vx @ w, vy)
    m = val["multi"]
    row = {"n": int(vy.size(0)), "lambda": best, "cos_centered": cos.mean().item(),
           "R@1": (rank < 1).double().mean().item(), "R@5": (rank < 5).double().mean().item()}
    if m.any():
        cm, rm = scores(vx[m] @ w, vy[m])  # retrieval among multi-unit nodes only
        row.update({"n_multi": int(m.sum()), "cos_centered_multi": cm.mean().item(),
                    "R@1_multi": (rm < 1).double().mean().item()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="packing config (e.g. the E4 config)")
    ap.add_argument("--model", action="append", required=True, help="NAME=pretrained|/path/to/last.pt")
    ap.add_argument("--train-samples", type=int, default=4000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    cfg.backbone.grad_checkpointing = False
    device = torch.device("cuda")
    tok = load_tokenizer(cfg.backbone)
    mk = lambda split, n: DataLoader(
        HierarchicalJsonlDataset(cfg.data.jsonl, tok, cfg, split=split, max_samples=n),
        batch_size=cfg.train.batch_size, shuffle=False, collate_fn=lambda b: collate_packed(b, tok.pad_token_id))
    train_loader, val_loader = mk(cfg.data.split, args.train_samples), mk(cfg.data.eval_split, None)
    print(f"train n={len(train_loader.dataset)} val n={len(val_loader.dataset)}", flush=True)

    results = {"config": args.config, "train_samples": len(train_loader.dataset),
               "val_samples": len(val_loader.dataset), "models": {}}
    for spec in args.model:
        name, path = spec.split("=", 1)
        torch.manual_seed(0)
        model = HierarchicalCodeJEPA(cfg, tok).to(device).eval()
        if path != "pretrained":
            state = torch.load(path, map_location="cpu")["model"]
            model.load_state_dict(state, strict=True)
        tr, _ = extract(model, train_loader, device)
        va, lm = extract(model, val_loader, device)
        row = {"eval_lm": lm}
        for lv in LEVEL_NAMES:
            if lv in tr and lv in va:
                row[lv] = probe(tr[lv], va[lv])
        results["models"][name] = row
        print(name, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in row.items()}), flush=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
