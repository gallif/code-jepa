#!/usr/bin/env python3
"""TokenRatio + simple Pareto report vs body-encoding baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from code_jepa.config import load_config


def mean_tokens(tokenizer, texts: list[str], max_length: int) -> dict:
    lengths = []
    for t in texts:
        ids = tokenizer(t, truncation=True, max_length=max_length, add_special_tokens=True)[
            "input_ids"
        ]
        lengths.append(len(ids))
    arr = np.asarray(lengths)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "max": int(arr.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--eval-json", default="", help="Optional eval_retrieval output to merge")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    frames = [pd.read_parquet(data_dir / f"functions_{s}.parquet") for s in ("train", "val", "test")]
    df = pd.concat(frames, ignore_index=True)

    tok = AutoTokenizer.from_pretrained(cfg["teacher_name"], trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    sig_stats = mean_tokens(tok, df["sig_text"].tolist(), int(cfg["max_sig_tokens"]))
    body_stats = mean_tokens(tok, df["body_text"].tolist(), int(cfg["max_body_tokens"]))
    full_stats = mean_tokens(
        tok,
        (df["sig_text"] + "\n" + df["body_text"]).tolist(),
        int(cfg["max_sig_tokens"]) + int(cfg["max_body_tokens"]),
    )

    token_ratio = sig_stats["mean"] / max(full_stats["mean"], 1e-8)
    report = {
        "n_functions": int(len(df)),
        "tokens_sig": sig_stats,
        "tokens_body": body_stats,
        "tokens_full": full_stats,
        "token_ratio_sig_over_full": token_ratio,
        "budget": float(cfg["eval"]["token_ratio_budget"]),
        "under_budget": token_ratio <= float(cfg["eval"]["token_ratio_budget"]),
        "note": (
            "JEPA path consumes ~tokens_sig at query time; "
            "body-encoding baseline consumes ~tokens_full (or tokens_body) per indexed doc "
            "and typically tokens_sig at query — report ratio is query-side compression."
        ),
    }

    if args.eval_json:
        ev = json.loads(Path(args.eval_json).read_text())
        report["retrieval"] = ev
        report["poc_claim_pass"] = bool(
            ev.get("pass_rank10")
            and ev.get("pass_rank1")
            and ev.get("beats_sig_pool")
            and report["under_budget"]
        )

    out = Path(cfg["runs_dir"]) / "token_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
