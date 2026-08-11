#!/usr/bin/env python3
"""Evaluate Rank@k retrieval: student(sig) vs FAISS body corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from code_jepa.config import load_config
from code_jepa.retrieval import build_ip_index, ranks_of_targets, recall_at_k
from code_jepa.student import SigPredictor
from code_jepa.teacher import FrozenTeacher


@torch.no_grad()
def predict_all(student, teacher, sigs, max_len, device, bs=16):
    preds = []
    for i in tqdm(range(0, len(sigs), bs), desc="predict"):
        batch = sigs[i : i + bs]
        tout = teacher.encode(batch, max_length=max_len)
        pred = student(tout.token_hidden, tout.attention_mask)
        preds.append(pred.cpu().float().numpy())
    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corpus", default="all", choices=["all", "split"],
                    help="all = train+val+test bodies; split = eval split only")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache_dir = Path(cfg["cache_dir"])
    stem = f"teacher_{cfg['teacher_name'].replace('/', '_')}_L{cfg['teacher_layer']}"
    meta = pd.read_parquet(cache_dir / f"{stem}_meta.parquet")
    body = np.load(cache_dir / f"{stem}_body.npy")
    sig_emb = np.load(cache_dir / f"{stem}_sig.npy")

    teacher = FrozenTeacher(cfg["teacher_name"], int(cfg["teacher_layer"]), device=args.device)
    student = SigPredictor(
        d_in=teacher.hidden_size,
        d_out=teacher.hidden_size,
        d_model=int(cfg["student"]["d_model"]),
        nhead=int(cfg["student"]["nhead"]),
        num_layers=int(cfg["student"]["num_layers"]),
        dropout=float(cfg["student"]["dropout"]),
    ).to(args.device)
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    student.load_state_dict(ckpt["student"])
    student.eval()

    eval_mask = (meta["split"] == args.split).to_numpy()
    eval_df = meta.loc[eval_mask].reset_index(drop=True)
    eval_idx = np.where(eval_mask)[0]

    if args.corpus == "all":
        corpus_emb = body
        # target id in full corpus = original row index
        target_ids = eval_idx.tolist()
    else:
        corpus_emb = body[eval_mask]
        target_ids = list(range(len(eval_df)))

    index = build_ip_index(corpus_emb)

    # Student predictions
    pred = predict_all(
        student, teacher, eval_df["sig_text"].tolist(),
        int(cfg["max_sig_tokens"]), args.device,
    )
    ranks_student = ranks_of_targets(pred, index, target_ids)

    # Baseline: teacher mean-pool sig embedding (already cached)
    ranks_sig = ranks_of_targets(sig_emb[eval_mask], index, target_ids)

    ks = list(cfg["eval"]["rank_ks"])
    results = {
        "split": args.split,
        "corpus": args.corpus,
        "n_queries": int(len(eval_df)),
        "n_corpus": int(corpus_emb.shape[0]),
        "student": recall_at_k(ranks_student, ks),
        "teacher_sig_pool": recall_at_k(ranks_sig, ks),
        "ckpt": args.ckpt,
    }
    # stringify keys for json
    results["student"] = {str(k): v for k, v in results["student"].items()}
    results["teacher_sig_pool"] = {str(k): v for k, v in results["teacher_sig_pool"].items()}

    r10 = results["student"]["10"]
    r1 = results["student"]["1"]
    results["pass_rank10"] = r10 >= float(cfg["eval"]["success_rank10"])
    results["pass_rank1"] = r1 >= float(cfg["eval"]["success_rank1"])
    results["beats_sig_pool"] = r10 > results["teacher_sig_pool"]["10"]

    out = Path(args.ckpt).parent / f"eval_{args.split}_{args.corpus}.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
