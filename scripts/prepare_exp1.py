#!/usr/bin/env python3
"""CPU-side prep for Exp 1 (JEPA mask): validate data, prefetch encoder, write report.

Does not require a free GPU. Run before train_jepa_mask.py.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from transformers import AutoModel, AutoTokenizer

from code_jepa.config import load_config


REQUIRED_COLS = ("full_text", "repo", "func_name")


def _split_stats(df: pd.DataFrame) -> dict:
    text = df["full_text"].astype(str)
    lens = text.str.len()
    return {
        "n": int(len(df)),
        "empty_full_text": int((text.str.strip() == "").sum()),
        "chars_mean": float(lens.mean()),
        "chars_p50": float(lens.median()),
        "chars_p95": float(lens.quantile(0.95)),
        "chars_max": int(lens.max()),
        "n_repos": int(df["repo"].nunique()) if "repo" in df.columns else None,
    }


def validate_data(data_dir: Path) -> dict:
    report: dict = {"splits": {}, "ok": True, "errors": []}
    for split in ("train", "val", "test"):
        path = data_dir / f"functions_{split}.parquet"
        if not path.exists():
            report["ok"] = False
            report["errors"].append(f"missing {path}")
            continue
        df = pd.read_parquet(path)
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            report["ok"] = False
            report["errors"].append(f"{split}: missing columns {missing}")
            continue
        stats = _split_stats(df)
        report["splits"][split] = stats
        if stats["n"] == 0 or stats["empty_full_text"] > 0:
            report["ok"] = False
            report["errors"].append(f"{split}: empty rows or zero size ({stats})")
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        report["data_meta"] = json.loads(meta_path.read_text())
    return report


def prefetch_encoder(model_name: str, models_dir: Path) -> dict:
    """Download tokenizer + weights into HF cache (CPU)."""
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Load on CPU to populate cache without touching CUDA
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    n_params = sum(p.numel() for p in model.parameters())
    sample = tok("def foo(x):\n    return x\n", return_tensors="pt")
    with __import__("torch").no_grad():
        out = model(**sample, output_hidden_states=True, use_cache=False)
    n_layers = len(out.hidden_states) - 1  # exclude embed
    hidden = int(model.config.hidden_size)
    del model
    return {
        "model_name": model_name,
        "n_params": int(n_params),
        "hidden_size": hidden,
        "n_layers": int(n_layers),
        "vocab_size": int(tok.vocab_size),
        "pad_token_id": int(tok.pad_token_id),
        "seconds": round(time.time() - t0, 2),
        "hf_home": str(models_dir / "hf"),
    }


def token_length_sample(data_dir: Path, model_name: str, max_tokens: int, n: int = 256) -> dict:
    """Estimate truncation rate on a train subsample (CPU tokenize only)."""
    import numpy as np

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    df = pd.read_parquet(data_dir / "functions_train.parquet")
    texts = df["full_text"].astype(str).tolist()
    rng = np.random.default_rng(42)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    lengths = []
    truncated = 0
    for i in idx:
        ids = tok(texts[i], truncation=False, add_special_tokens=True)["input_ids"]
        lengths.append(len(ids))
        if len(ids) > max_tokens:
            truncated += 1
    arr = np.asarray(lengths)
    return {
        "sample_n": int(len(arr)),
        "tok_mean": float(arr.mean()),
        "tok_p50": float(np.median(arr)),
        "tok_p95": float(np.quantile(arr, 0.95)),
        "tok_max": int(arr.max()),
        "frac_over_max_tokens": float(truncated / max(len(arr), 1)),
        "max_tokens": max_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/jepa_exp1.yaml")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--token-sample", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    models_dir = Path(cfg["models_dir"])
    runs_dir = Path(cfg["runs_dir"])
    out_dir = runs_dir / "prep_exp1"
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": args.config,
        "encoder": cfg.get("encoder", {}),
    }

    print("Validating function parquets…")
    report["data"] = validate_data(data_dir)
    for split, st in report["data"].get("splits", {}).items():
        print(f"  {split}: n={st['n']} chars_p50={st['chars_p50']:.0f} empty={st['empty_full_text']}")
    if report["data"]["errors"]:
        print("DATA ERRORS:", report["data"]["errors"])

    model_name = cfg["encoder"]["model_name"]
    max_tokens = int(cfg["encoder"]["max_tokens"])

    if not args.skip_download:
        print(f"Prefetching encoder on CPU: {model_name}")
        report["encoder_prefetch"] = prefetch_encoder(model_name, models_dir)
        print(
            "  params={n_params:,} hidden={hidden_size} layers={n_layers} ({seconds}s)".format(
                **report["encoder_prefetch"]
            )
        )
    else:
        report["encoder_prefetch"] = {"skipped": True}

    print(f"Token-length sample (n={args.token_sample})…")
    report["token_lengths"] = token_length_sample(
        data_dir, model_name, max_tokens, n=args.token_sample
    )
    tl = report["token_lengths"]
    print(
        f"  tok_p50={tl['tok_p50']:.0f} tok_p95={tl['tok_p95']:.0f} "
        f"frac>{max_tokens}={tl['frac_over_max_tokens']:.3f}"
    )

    report["ready_for_train"] = bool(report["data"]["ok"]) and not report["data"]["errors"]
    out_path = out_dir / "prep_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {out_path}")
    print("ready_for_train:", report["ready_for_train"])
    if report["ready_for_train"]:
        print(
            "When a GPU is free:\n"
            "  export CUDA_VISIBLE_DEVICES=<free>\n"
            "  python scripts/train_jepa_mask.py --config configs/jepa_exp1.yaml --max-steps 20"
        )


if __name__ == "__main__":
    main()
