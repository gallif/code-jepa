#!/usr/bin/env python3
"""Exp 1: JEPA mask prediction — train E + P_mask with EMA targets on full_text."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from code_jepa.config import load_config
from code_jepa.ema import EMATarget, ema_momentum_cosine
from code_jepa.jepa_encoder import ProgramEncoder
from code_jepa.jepa_predictor import JEPAPredictor, masked_smooth_l1
from code_jepa.jepa_views import hide_masked_tokens
from code_jepa.masking import sample_multiblock_mask


class FullTextDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        if "full_text" not in df.columns:
            raise KeyError("expected full_text column in functions parquet")
        self.texts = df["full_text"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, i: int) -> str:
        return self.texts[i]


def collate_texts(batch: list[str]) -> list[str]:
    return list(batch)


def build_optimizer(encoder, predictor, cfg):
    enc_params = [p for p in encoder.parameters() if p.requires_grad]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": float(cfg["train"]["lr_encoder"])})
    groups.append(
        {"params": predictor.parameters(), "lr": float(cfg["train"]["lr_predictor"])}
    )
    return torch.optim.AdamW(groups, weight_decay=float(cfg["train"]["weight_decay"]))


def lr_multiplier(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


@torch.no_grad()
def eval_loss(encoder, target_ema, predictor, loader, cfg, device, max_batches: int) -> float:
    encoder.eval()
    predictor.eval()
    losses = []
    for i, texts in enumerate(loader):
        if i >= max_batches:
            break
        tok = encoder.tokenize(texts, max_length=int(cfg["encoder"]["max_tokens"]), device=device)
        input_ids, attention_mask = tok["input_ids"], tok["attention_mask"]
        mask = sample_multiblock_mask(
            attention_mask,
            mask_ratio=float(cfg["mask"]["mask_ratio"]),
            min_block=int(cfg["mask"]["min_block"]),
            max_block=int(cfg["mask"]["max_block"]),
        )
        ctx_ids, ctx_attn = hide_masked_tokens(
            input_ids, attention_mask, mask.masked, encoder.tokenizer.pad_token_id
        )
        target = target_ema(input_ids, attention_mask).token_hidden
        ctx = encoder(ctx_ids, ctx_attn).token_hidden
        pred = predictor(ctx, attention_mask, mask.masked)
        losses.append(
            masked_smooth_l1(pred, target, mask.masked, attention_mask).item()
        )
    encoder.train()
    predictor.train()
    return float(sum(losses) / max(len(losses), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jepa_exp1.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--max-steps", type=int, default=0, help="optional early stop for smoke tests")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    runs_dir = Path(cfg["runs_dir"])
    run_name = args.run_name or time.strftime("jepa_mask_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.json").open("w") as f:
        json.dump(cfg, f, indent=2, default=str)

    torch.manual_seed(int(cfg["train"]["seed"]))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_df = pd.read_parquet(data_dir / "functions_train.parquet")
    val_df = pd.read_parquet(data_dir / "functions_val.parquet")
    train_loader = DataLoader(
        FullTextDataset(train_df),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate_texts,
    )
    val_loader = DataLoader(
        FullTextDataset(val_df),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=collate_texts,
    )

    layer = cfg["encoder"].get("layer", None)
    encoder = ProgramEncoder(cfg["encoder"]["model_name"], layer=layer).to(device)
    if cfg["train"].get("freeze_encoder"):
        for p in encoder.parameters():
            p.requires_grad_(False)

    predictor = JEPAPredictor(
        d_in=encoder.hidden_size,
        d_model=int(cfg["predictor"]["d_model"]),
        nhead=int(cfg["predictor"]["nhead"]),
        num_layers=int(cfg["predictor"]["num_layers"]),
        dropout=float(cfg["predictor"]["dropout"]),
        max_positions=int(cfg["encoder"]["max_tokens"]),
    ).to(device)

    target_ema = EMATarget(encoder, momentum=float(cfg["ema"]["momentum_start"]))
    # EMA module on device
    target_ema.module.to(device)

    opt = build_optimizer(encoder, predictor, cfg)
    epochs = int(cfg["train"]["epochs"])
    steps_per_epoch = max(len(train_loader), 1)
    total_steps = epochs * steps_per_epoch
    warmup = int(cfg["train"]["warmup_steps"])
    history = []
    global_step = 0
    best_val = float("inf")

    for epoch in range(epochs):
        encoder.train()
        predictor.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        running = 0.0
        n_seen = 0
        for texts in pbar:
            tok = encoder.tokenize(
                texts, max_length=int(cfg["encoder"]["max_tokens"]), device=device
            )
            input_ids, attention_mask = tok["input_ids"], tok["attention_mask"]
            mask = sample_multiblock_mask(
                attention_mask,
                mask_ratio=float(cfg["mask"]["mask_ratio"]),
                min_block=int(cfg["mask"]["min_block"]),
                max_block=int(cfg["mask"]["max_block"]),
            )
            ctx_ids, ctx_attn = hide_masked_tokens(
                input_ids,
                attention_mask,
                mask.masked,
                encoder.tokenizer.pad_token_id,
            )

            with torch.no_grad():
                target = target_ema(input_ids, attention_mask).token_hidden

            ctx = encoder(ctx_ids, ctx_attn).token_hidden
            pred = predictor(ctx, attention_mask, mask.masked)
            loss = masked_smooth_l1(pred, target, mask.masked, attention_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in list(encoder.parameters()) + list(predictor.parameters()) if p.requires_grad],
                float(cfg["train"]["grad_clip"]),
            )
            if global_step == 0:
                for g in opt.param_groups:
                    g["initial_lr"] = g["lr"]
            mult = lr_multiplier(global_step, warmup, total_steps)
            for g in opt.param_groups:
                g["lr"] = g["initial_lr"] * mult

            opt.step()

            m = ema_momentum_cosine(
                global_step,
                total_steps,
                float(cfg["ema"]["momentum_start"]),
                float(cfg["ema"]["momentum_end"]),
            )
            target_ema.momentum = m
            target_ema.update(encoder)

            running += loss.item()
            n_seen += 1
            global_step += 1
            pbar.set_postfix(loss=running / n_seen, m=f"{m:.4f}")

            if args.max_steps and global_step >= args.max_steps:
                break

        val = eval_loss(
            encoder,
            target_ema,
            predictor,
            val_loader,
            cfg,
            device,
            max_batches=int(cfg["eval"]["max_batches"]),
        )
        row = {"epoch": epoch, "train_loss": running / max(n_seen, 1), "val_loss": val}
        history.append(row)
        print(row)
        ckpt = {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "ema": target_ema.module.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
            "val_loss": val,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if val < best_val:
            best_val = val
            torch.save(ckpt, run_dir / "best.pt")

        if args.max_steps and global_step >= args.max_steps:
            break

    with (run_dir / "history.json").open("w") as f:
        json.dump(history, f, indent=2)
    print(f"done → {run_dir} best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
