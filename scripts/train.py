#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from code_jepa.config import ExperimentConfig
from code_jepa.data.dataset import HierarchicalJsonlDataset, collate_packed
from code_jepa.metrics import representation_metrics
from code_jepa.model import HierarchicalCodeJEPA, build_tiny_backbone
from code_jepa.tokenize_utils import load_tokenizer

LOG_KEYS = ("total", "lm", "view", "sigreg", "view_step", "view_unit", "view_approach", "view_intent")


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


def make_loader(dataset, cfg: ExperimentConfig, tokenizer, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        shuffle=shuffle,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        collate_fn=lambda b: collate_packed(b, tokenizer.pad_token_id),
    )


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def _cat_pad(tensors: list[torch.Tensor], pad_value=0) -> torch.Tensor:
    width = max(t.size(1) for t in tensors)
    padded = []
    for tensor in tensors:
        if tensor.size(1) < width:
            shape = (tensor.size(0), width - tensor.size(1), *tensor.shape[2:])
            tensor = torch.cat([tensor, tensor.new_full(shape, pad_value)], dim=1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)


@torch.no_grad()
def evaluate(model: HierarchicalCodeJEPA, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    loss_sums: dict[str, float] = {}
    loss_count = 0
    preds, targets, valids, levels, codes, texts, n_units = [], [], [], [], [], [], []
    for batch in loader:
        batch = to_device(batch, device)
        losses = model.compute_losses(batch, with_embeddings=True)
        for key in LOG_KEYS:
            if key in losses and torch.is_tensor(losses[key]):
                loss_sums[key] = loss_sums.get(key, 0.0) + float(losses[key].detach())
        loss_count += 1
        if "pred" in losses:
            preds.append(losses["pred"].detach().cpu())
            targets.append(losses["target"].detach().cpu())
            valids.append(batch["valid_nodes"].detach().cpu())
            levels.append(batch["level_ids"].detach().cpu())
            codes.append(losses["code_emb"].detach().cpu())
            texts.append(losses["text_emb"].detach().cpu())
            n_units.append(batch["n_units"].detach().cpu())
    row = {f"eval_{k}": v / max(loss_count, 1) for k, v in loss_sums.items()}
    if preds:
        metrics = representation_metrics(
            _cat_pad(preds),
            _cat_pad(targets),
            _cat_pad(valids, pad_value=False),
            _cat_pad(levels, pad_value=-1),
            code=_cat_pad(codes),
            text=_cat_pad(texts),
            n_units=torch.cat(n_units),
        )
        row.update({f"eval_{k}": v for k, v in metrics.items()})
    if device.type == "cuda":
        row["eval_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
    model.train()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Train hierarchical Code-JEPA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--tiny", action="store_true", help="random 2-layer Qwen2, for smoke tests")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="dotted key=value overrides, e.g. objectives.sigreg.weight=1.0",
    )
    args = parser.parse_args()

    cfg = ExperimentConfig.load(args.config)
    if args.override:
        cfg.apply_dot_overrides(args.override)
        print("overrides:", args.override)
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
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.reset_peak_memory_stats()
    if device.type == "cpu" and cfg.backbone.dtype in {"bf16", "fp16"}:
        cfg.backbone.dtype = "fp32"
    tokenizer = load_tokenizer(cfg.backbone)

    eval_loader = None
    if cfg.data.eval_split:
        eval_ds = HierarchicalJsonlDataset(
            cfg.data.jsonl,
            tokenizer,
            cfg,
            split=cfg.data.eval_split,
            max_samples=cfg.data.eval_max_samples,
        )
        eval_loader = make_loader(eval_ds, cfg, tokenizer, shuffle=False)
        print(f"eval split={cfg.data.eval_split} n={len(eval_ds)} dropped={eval_ds.dropped}")

    output_dir = Path(cfg.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, default=str))
    metrics_path = output_dir / "metrics.jsonl"

    backbone = build_tiny_backbone(len(tokenizer)) if args.tiny else None
    if args.tiny:
        cfg.backbone.grad_checkpointing = False
        cfg.backbone.dtype = "fp32"
    model = HierarchicalCodeJEPA(cfg, tokenizer, backbone=backbone).to(device)
    if device.type == "cuda":
        print(
            f"device={device} gpu={torch.cuda.get_device_name(0)} "
            f"mem_gb={torch.cuda.memory_allocated() / 1024**3:.2f}"
        )

    if eval_loader is not None:
        row = {"split": "eval", "step": 0, "epoch": 0, **evaluate(model, eval_loader, device)}
        append_jsonl(metrics_path, row)
        print("eval@0", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k.startswith("eval_")})

    if args.eval_only:
        print(f"eval-only done, wrote {metrics_path}")
        return

    dataset = HierarchicalJsonlDataset(
        cfg.data.jsonl,
        tokenizer,
        cfg,
        split=cfg.data.split,
        max_samples=cfg.data.max_samples,
    )
    loader = make_loader(dataset, cfg, tokenizer, shuffle=True)
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=0.01)
    print(f"train n={len(dataset)} dropped={dataset.dropped} batch={cfg.train.batch_size}")

    model.train()
    step = 0
    opt.zero_grad(set_to_none=True)
    t0 = time.time()
    for epoch in range(cfg.train.epochs):
        pbar = tqdm(loader, desc=f"epoch {epoch}")
        for batch in pbar:
            batch = to_device(batch, device)
            losses = model.compute_losses(batch)
            loss = losses["total"] / cfg.train.grad_accum
            loss.backward()
            if (step + 1) % cfg.train.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
            step += 1
            if step % cfg.train.log_every == 0:
                row = {
                    "split": "train",
                    "step": step,
                    "epoch": epoch,
                    "sec": time.time() - t0,
                }
                for key in LOG_KEYS:
                    if key in losses:
                        row[key] = float(losses[key].detach())
                if device.type == "cuda":
                    row["mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
                append_jsonl(metrics_path, row)
                pbar.set_postfix({k: f"{row[k]:.3f}" for k in ("total", "lm", "view") if k in row})
            if eval_loader is not None and cfg.train.eval_every and step % cfg.train.eval_every == 0:
                row = {"split": "eval", "step": step, "epoch": epoch, **evaluate(model, eval_loader, device)}
                append_jsonl(metrics_path, row)
                print(f"eval@{step}", {k: round(v, 4) for k, v in row.items() if k in ("eval_lm", "eval_view", "eval_cos_intent", "eval_cos_step", "eval_confusion_acc")})
            if step % cfg.train.save_every == 0:
                torch.save({"model": model.state_dict(), "step": step}, output_dir / f"step_{step}.pt")
            if cfg.train.max_steps and step >= cfg.train.max_steps:
                break
        if cfg.train.max_steps and step >= cfg.train.max_steps:
            break

    if eval_loader is not None:
        row = {"split": "eval", "step": step, "epoch": cfg.train.epochs, **evaluate(model, eval_loader, device)}
        append_jsonl(metrics_path, row)
        print("eval@final", {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items() if k.startswith("eval_")})
    torch.save({"model": model.state_dict(), "step": step}, output_dir / "last.pt")
    print(f"finished {step} steps, dropped {dataset.dropped} records, saved {output_dir}")


if __name__ == "__main__":
    main()
