#!/usr/bin/env python3
"""Teacher mean-pool query ladder: Rank@k vs query tokens.

Compares increasingly rich query text (name → sig → sig+k body lines → full)
against the cached body FAISS index, and optionally overlays a JEPA student
eval JSON to answer: how many teacher-pool tokens to match JEPA Rank@k?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from code_jepa.config import load_config
from code_jepa.retrieval import build_ip_index, ranks_of_targets, recall_at_k
from code_jepa.teacher import FrozenTeacher


def strip_docstring_header(sig: str) -> str:
    """Remove a leading docstring block from sig_text if present."""
    lines = sig.splitlines()
    # Keep def / decorator lines until a docstring opener
    out = []
    i = 0
    while i < len(lines):
        s = lines[i]
        if '"""' in s or "'''" in s:
            # drop remainder of sig (docstring + anything after)
            break
        out.append(s)
        i += 1
    return "\n".join(out).rstrip() if out else sig.splitlines()[0] if lines else sig


def first_k_body_lines(body: str, k: int) -> str:
    if k <= 0:
        return ""
    lines = body.splitlines()
    return "\n".join(lines[:k]).rstrip()


def build_ladder_queries(df: pd.DataFrame, body_ks: list[int]) -> list[tuple[str, list[str]]]:
    """Return (rung_name, list[query_text]) in increasing richness order."""
    names = df["func_name"].astype(str).tolist()
    sigs = df["sig_text"].astype(str).tolist()
    bodies = df["body_text"].astype(str).tolist()
    sig_nodeoc = [strip_docstring_header(s) for s in sigs]

    rungs: list[tuple[str, list[str]]] = [
        ("name", names),
        ("sig_no_doc", sig_nodeoc),
        ("sig", sigs),
    ]
    for k in body_ks:
        rungs.append(
            (
                f"sig+body_l{k}",
                [f"{s}\n{first_k_body_lines(b, k)}".rstrip() for s, b in zip(sigs, bodies)],
            )
        )
    rungs.append(
        (
            "full_sig+body",
            [f"{s}\n{b}".rstrip() for s, b in zip(sigs, bodies)],
        )
    )
    return rungs


def mean_token_count(tokenizer, texts: list[str], max_length: int) -> dict:
    lengths = []
    for t in texts:
        ids = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        lengths.append(len(ids))
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


@torch.no_grad()
def encode_pool(teacher: FrozenTeacher, texts: list[str], max_len: int, bs: int) -> np.ndarray:
    outs = []
    for i in tqdm(range(0, len(texts), bs), desc="encode"):
        batch = texts[i : i + bs]
        out = teacher.encode(batch, max_length=max_len)
        outs.append(out.pooled.cpu().float().numpy())
    return np.concatenate(outs, axis=0)


def tokens_to_match(
    ladder: list[dict],
    target_recall: float,
    metric_key: str = "10",
) -> dict | None:
    """Smallest rung (by mean tokens) whose Rank@k >= target_recall."""
    ordered = sorted(ladder, key=lambda r: r["tokens"]["mean"])
    for row in ordered:
        if row["recall"].get(metric_key, 0.0) >= target_recall:
            return {
                "rung": row["rung"],
                "tokens_mean": row["tokens"]["mean"],
                "recall": row["recall"][metric_key],
                "metric": f"Rank@{metric_key}",
                "target": target_recall,
            }
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--corpus", default="all", choices=["all", "split"])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument(
        "--body-ks",
        default="1,2,4,8,16,32",
        help="Comma-separated # of leading body lines to append after sig",
    )
    ap.add_argument(
        "--jepa-eval",
        default="",
        help="Optional eval_retrieval JSON to overlay (student Rank@k + ckpt)",
    )
    ap.add_argument("--max-query-tokens", type=int, default=0, help="0 = sig+body caps from config")
    args = ap.parse_args()
    cfg = load_config(args.config)

    cache_dir = Path(cfg["cache_dir"])
    runs_dir = Path(cfg["runs_dir"])
    stem = f"teacher_{cfg['teacher_name'].replace('/', '_')}_L{cfg['teacher_layer']}"
    meta = pd.read_parquet(cache_dir / f"{stem}_meta.parquet")
    body = np.load(cache_dir / f"{stem}_body.npy")

    eval_mask = (meta["split"] == args.split).to_numpy()
    eval_df = meta.loc[eval_mask].reset_index(drop=True)
    eval_idx = np.where(eval_mask)[0]

    if args.corpus == "all":
        corpus_emb = body
        target_ids = eval_idx.tolist()
    else:
        corpus_emb = body[eval_mask]
        target_ids = list(range(len(eval_df)))

    index = build_ip_index(corpus_emb)
    body_ks = [int(x) for x in args.body_ks.split(",") if x.strip()]
    rungs = build_ladder_queries(eval_df, body_ks)

    max_len = args.max_query_tokens or (
        int(cfg["max_sig_tokens"]) + int(cfg["max_body_tokens"])
    )
    tok = AutoTokenizer.from_pretrained(cfg["teacher_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    teacher = FrozenTeacher(cfg["teacher_name"], int(cfg["teacher_layer"]), device=args.device)
    ks = list(cfg["eval"]["rank_ks"])

    ladder_rows = []
    for name, texts in rungs:
        print(f"\n=== rung: {name} ===")
        token_stats = mean_token_count(tok, texts, max_len)
        emb = encode_pool(teacher, texts, max_len, args.batch_size)
        ranks = ranks_of_targets(emb, index, target_ids)
        recall = recall_at_k(ranks, ks)
        row = {
            "rung": name,
            "tokens": token_stats,
            "recall": {str(k): v for k, v in recall.items()},
            "mean_rank": float(ranks.mean()),
            "median_rank": float(np.median(ranks)),
        }
        ladder_rows.append(row)
        print(json.dumps(row, indent=2))

    report: dict = {
        "split": args.split,
        "corpus": args.corpus,
        "n_queries": int(len(eval_df)),
        "n_corpus": int(corpus_emb.shape[0]),
        "teacher": cfg["teacher_name"],
        "layer": int(cfg["teacher_layer"]),
        "max_query_tokens": max_len,
        "body_ks": body_ks,
        "ladder": ladder_rows,
        "method": "teacher_mean_pool_query",
    }

    if args.jepa_eval:
        jepa = json.loads(Path(args.jepa_eval).read_text())
        student = jepa.get("student", {})
        # JEPA query tokens ≈ mean sig tokens on same split
        sig_tok = mean_token_count(tok, eval_df["sig_text"].astype(str).tolist(), max_len)
        report["jepa"] = {
            "ckpt": jepa.get("ckpt"),
            "recall": student,
            "tokens": sig_tok,
            "note": "JEPA query path tokenizes sig only (teacher encode + student).",
        }
        target = float(student.get("10", 0.0))
        match = tokens_to_match(ladder_rows, target, metric_key="10")
        report["tokens_to_match_jepa_rank10"] = match
        if match:
            report["jepa_token_ratio_vs_match"] = (
                sig_tok["mean"] / max(match["tokens_mean"], 1e-8)
            )
            print(
                f"\nTeacher-pool needs rung={match['rung']} "
                f"(~{match['tokens_mean']:.1f} tok) to match JEPA Rank@10={target:.4f}"
            )
            print(
                f"JEPA uses ~{sig_tok['mean']:.1f} tok "
                f"→ ratio vs that rung: {report['jepa_token_ratio_vs_match']:.3f}"
            )
        else:
            print("\nNo teacher-pool rung reached JEPA Rank@10 "
                  "(full_sig+body should — check encoding/index mismatch).")

    out = runs_dir / f"token_ladder_{args.split}_{args.corpus}.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
