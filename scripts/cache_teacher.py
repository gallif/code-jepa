#!/usr/bin/env python3
"""Cache frozen-teacher body embeddings (+ optional sig hidden states). Exp 0 sanity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from code_jepa.config import load_config
from code_jepa.teacher import FrozenTeacher


def pooled_stats(emb: np.ndarray) -> dict:
    # emb [N, D]
    x = emb.astype(np.float64)
    norms = np.linalg.norm(x, axis=1)
    # pairwise on a sample
    n = min(512, len(x))
    idx = np.linspace(0, len(x) - 1, n, dtype=int)
    s = x[idx]
    s = s / np.linalg.norm(s, axis=1, keepdims=True).clip(1e-8)
    sim = s @ s.T
    iu = np.triu_indices(n, k=1)
    off = sim[iu]
    return {
        "n": int(len(emb)),
        "dim": int(emb.shape[1]),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "offdiag_cosine_mean": float(off.mean()),
        "offdiag_cosine_std": float(off.std()),
        "feature_var_mean": float(x.var(axis=0).mean()),
    }


@torch.no_grad()
def encode_column(teacher: FrozenTeacher, texts: list[str], max_len: int, bs: int) -> np.ndarray:
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="encode"):
        batch = texts[i : i + bs]
        out = teacher.encode(batch, max_length=max_len)
        outs.append(out.pooled.cpu().float().numpy())
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--split", default="train", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    frames = []
    for sp in splits:
        path = data_dir / f"functions_{sp}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run prepare_data.py first")
        df = pd.read_parquet(path)
        df["split"] = sp
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if args.max_rows:
        df = df.head(args.max_rows).reset_index(drop=True)

    teacher = FrozenTeacher(cfg["teacher_name"], int(cfg["teacher_layer"]), device=args.device)
    print(
        f"Teacher {cfg['teacher_name']} layer={cfg['teacher_layer']} "
        f"hidden={teacher.hidden_size} n_layers={teacher.num_layers}"
    )

    body_emb = encode_column(
        teacher, df["body_text"].tolist(), int(cfg["max_body_tokens"]), args.batch_size
    )
    sig_emb = encode_column(
        teacher, df["sig_text"].tolist(), int(cfg["max_sig_tokens"]), args.batch_size
    )

    stem = f"teacher_{cfg['teacher_name'].replace('/', '_')}_L{cfg['teacher_layer']}"
    if args.split != "all":
        stem = f"{stem}_{args.split}"
    if args.max_rows:
        stem = f"{stem}_n{args.max_rows}"

    body_path = cache_dir / f"{stem}_body.npy"
    sig_path = cache_dir / f"{stem}_sig.npy"
    meta_path = cache_dir / f"{stem}_meta.parquet"
    np.save(body_path, body_emb)
    np.save(sig_path, sig_emb)
    df.to_parquet(meta_path, index=False)

    stats = {
        "body": pooled_stats(body_emb),
        "sig": pooled_stats(sig_emb),
        "body_path": str(body_path),
        "sig_path": str(sig_path),
        "meta_path": str(meta_path),
        "teacher": cfg["teacher_name"],
        "layer": cfg["teacher_layer"],
    }
    # Exp 0 gates: non-trivial variance + off-diag cosine not ~1
    body_ok = stats["body"]["feature_var_mean"] > 1e-4 and stats["body"]["offdiag_cosine_mean"] < 0.95
    stats["exp0_pass"] = bool(body_ok)
    stats_path = cache_dir / f"{stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    if not body_ok:
        raise SystemExit("Exp 0 FAILED: body embeddings look collapsed")
    print("Exp 0 PASS: body embedding stats look healthy")


if __name__ == "__main__":
    main()
