#!/usr/bin/env python3
"""Train SigPredictor with InfoNCE: sig teacher tokens → body embedding."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from code_jepa.config import load_config
from code_jepa.student import SigPredictor, info_nce
from code_jepa.teacher import FrozenTeacher


class SigBodyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, body_emb: np.ndarray):
        self.sigs = df["sig_text"].tolist()
        self.body_emb = body_emb.astype(np.float32)

    def __len__(self):
        return len(self.sigs)

    def __getitem__(self, i):
        return self.sigs[i], self.body_emb[i]


def collate(batch):
    sigs, embs = zip(*batch)
    return list(sigs), torch.tensor(np.stack(embs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--run-name", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    cache_dir = Path(cfg["cache_dir"])
    runs_dir = Path(cfg["runs_dir"])
    run_name = args.run_name or time.strftime("student_%Y%m%d_%H%M%S")
    run_dir = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    stem = f"teacher_{cfg['teacher_name'].replace('/', '_')}_L{cfg['teacher_layer']}"
    # Prefer full cached tensors with split column in meta
    meta_path = cache_dir / f"{stem}_meta.parquet"
    body_path = cache_dir / f"{stem}_body.npy"
    if not meta_path.exists():
        # fallback: per-split files concatenated expectation — require all
        raise FileNotFoundError(
            f"Missing {meta_path}. Run: python scripts/cache_teacher.py --split all"
        )

    meta = pd.read_parquet(meta_path)
    body = np.load(body_path)
    assert len(meta) == len(body)

    def subset(split: str):
        m = meta["split"] == split
        return meta.loc[m].reset_index(drop=True), body[m.to_numpy()]

    train_df, train_body = subset("train")
    val_df, val_body = subset("val")

    teacher = FrozenTeacher(cfg["teacher_name"], int(cfg["teacher_layer"]), device=args.device)
    student = SigPredictor(
        d_in=teacher.hidden_size,
        d_out=teacher.hidden_size,
        d_model=int(cfg["student"]["d_model"]),
        nhead=int(cfg["student"]["nhead"]),
        num_layers=int(cfg["student"]["num_layers"]),
        dropout=float(cfg["student"]["dropout"]),
    ).to(args.device)

    train_loader = DataLoader(
        SigBodyDataset(train_df, train_body),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=2,
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = DataLoader(
        SigBodyDataset(val_df, val_body),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=2,
        collate_fn=collate,
    )

    opt = torch.optim.AdamW(
        student.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    total_steps = len(train_loader) * int(cfg["train"]["epochs"])
    warmup = int(cfg["train"]["warmup_steps"])

    def lr_at(step: int) -> float:
        if step < warmup:
            return float(cfg["train"]["lr"]) * (step + 1) / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return float(cfg["train"]["lr"]) * 0.5 * (1 + math.cos(math.pi * progress))

    best_val = float("inf")
    history = []
    global_step = 0

    for epoch in range(int(cfg["train"]["epochs"])):
        student.train()
        losses = []
        for sigs, body_t in tqdm(train_loader, desc=f"epoch {epoch}"):
            for pg in opt.param_groups:
                pg["lr"] = lr_at(global_step)
            with torch.no_grad():
                tout = teacher.encode(sigs, max_length=int(cfg["max_sig_tokens"]))
            pred = student(tout.token_hidden, tout.attention_mask)
            loss = info_nce(pred, body_t.to(args.device), student.logit_scale)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), float(cfg["train"]["grad_clip"]))
            opt.step()
            losses.append(loss.item())
            global_step += 1

        student.eval()
        val_losses = []
        with torch.no_grad():
            for sigs, body_t in val_loader:
                tout = teacher.encode(sigs, max_length=int(cfg["max_sig_tokens"]))
                pred = student(tout.token_hidden, tout.attention_mask)
                val_losses.append(
                    info_nce(pred, body_t.to(args.device), student.logit_scale).item()
                )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_loss": float(np.mean(val_losses)) if val_losses else None,
            "lr": lr_at(global_step),
        }
        history.append(row)
        print(row)
        ckpt = {
            "student": student.state_dict(),
            "cfg": cfg,
            "epoch": epoch,
        }
        torch.save(ckpt, run_dir / "last.pt")
        if row["val_loss"] is not None and row["val_loss"] < best_val:
            best_val = row["val_loss"]
            torch.save(ckpt, run_dir / "best.pt")

    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
