#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_jepa.config import ExperimentConfig
from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed
from code_jepa.model import HierarchicalCodeJEPA, build_tiny_backbone
from code_jepa.tokenize_utils import load_tokenizer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def param_groups(model: HierarchicalCodeJEPA, cfg: ExperimentConfig):
    new_ids = {id(p) for p in model.heads.parameters()} | {id(p) for p in model.sigreg.parameters()}
    emb = model.backbone.get_input_embeddings().weight
    new_ids.add(id(emb))
    backbone, new_params = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (new_params if id(param) in new_ids else backbone).append(param)
    return [
        {"params": backbone, "lr": cfg.train.lr_backbone},
        {"params": new_params, "lr": cfg.train.lr_new_params},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical Code-JEPA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="random 2-layer Qwen2, for smoke tests")
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.data:
        cfg.data.jsonl = args.data
    if args.max_samples is not None:
        cfg.data.max_samples = args.max_samples
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    if not cfg.data.jsonl:
        raise SystemExit("set data.jsonl in the config or pass --data")

    set_seed(cfg.train.seed)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cpu" and cfg.backbone.dtype in {"bf16", "fp16"}:
        cfg.backbone.dtype = "fp32"
    tokenizer = load_tokenizer(cfg.backbone)
    dataset = HierarchicalJsonlDataset(
        cfg.data.jsonl,
        tokenizer,
        cfg,
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=lambda b: collate_packed(b, tokenizer.pad_token_id),
    )
    backbone = build_tiny_backbone(len(tokenizer)) if args.tiny else None
    if args.tiny:
        cfg.backbone.grad_checkpointing = False
        cfg.backbone.dtype = "fp32"
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone).to(device)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=0.01)
    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))

    model.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = {
                k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
            }
            losses = model.compute_losses(batch)
            loss = losses["total"] / cfg.train.grad_accum
            loss.backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.train.log_every == 0:
                pbar.set_postfix(
                    {
                        "total": float(losses["total"]),
                        "lm": float(losses["lm"]),
                        "view": float(losses["view"]),
                        "sigreg": float(losses["sigreg"]),
                    }
                )
            if step % cfg.train.save_every == 0:
                ckpt = output_dir / f"step_{step}.pt"
                torch.save({"model": model.state_dict(), "step": step}, ckpt)
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                break
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break

    torch.save({"model": model.state_dict(), "step": step}, output_dir / "last.pt")
    print(f"finished {step} steps, dropped {dataset.dropped} records, saved {output_dir}")


if __name__ == "__main__":
    main()
