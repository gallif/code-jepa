#!/usr/bin/env python3
"""Download CodeSearchNet Python, AST-split into sig/body, repo-level splits."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from code_jepa.ast_split import extract_functions
from code_jepa.config import load_config


def _clean_text(s: str) -> str:
    """Drop unpaired surrogates so parquet/pyarrow can encode UTF-8."""
    if not isinstance(s, str):
        return ""
    # errors="ignore" strips lone surrogates that break pyarrow
    return s.encode("utf-8", errors="ignore").decode("utf-8")


def _clean_record(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        out[k] = _clean_text(v) if isinstance(v, str) else v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-raw", type=int, default=200_000, help="Cap raw dataset rows scanned")
    args = ap.parse_args()
    cfg = load_config(args.config)

    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    target_n = int(cfg["dataset"]["target_functions"])
    min_stmts = int(cfg["min_body_stmts"])
    seed = int(cfg["dataset"]["seed"])
    rng = random.Random(seed)

    print("Loading CodeSearchNet python (streaming)...")
    ds = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["language"],
        split="train",
        streaming=True,
    )

    seen_hash: set[str] = set()
    by_repo: dict[str, list[dict]] = defaultdict(list)
    scanned = 0

    for row in tqdm(ds, total=args.max_raw, desc="scan"):
        scanned += 1
        if scanned > args.max_raw:
            break
        code = row.get("func_code_string") or row.get("whole_func_string") or ""
        repo = row.get("repository_name") or row.get("repo") or "unknown"
        path = row.get("func_path_in_repository") or row.get("path") or ""
        if not code.strip():
            continue
        for rec in extract_functions(code, repo=repo, path=path, min_body_stmts=min_stmts):
            if rec.ast_hash in seen_hash:
                continue
            seen_hash.add(rec.ast_hash)
            by_repo[repo].append(_clean_record(rec.to_dict()))

    # Flatten with repo-level split
    repos = sorted(by_repo.keys())
    rng.shuffle(repos)
    # Aim ~30k functions; take repos until filled then split repos 80/10/10
    selected: list[dict] = []
    used_repos: list[str] = []
    for repo in repos:
        if len(selected) >= target_n:
            break
        selected.extend(by_repo[repo])
        used_repos.append(repo)
    selected = selected[:target_n] if len(selected) > target_n else selected
    # Build repo -> rows for selected only
    sel_by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in selected:
        sel_by_repo[r["repo"]].append(r)
    used_repos = sorted(sel_by_repo.keys())
    rng.shuffle(used_repos)
    n = len(used_repos)
    n_train = max(1, int(0.8 * n))
    n_val = max(1, int(0.1 * n))
    train_repos = set(used_repos[:n_train])
    val_repos = set(used_repos[n_train : n_train + n_val])
    test_repos = set(used_repos[n_train + n_val :])

    splits = {"train": [], "val": [], "test": []}
    for repo, rows in sel_by_repo.items():
        if repo in train_repos:
            splits["train"].extend(rows)
        elif repo in val_repos:
            splits["val"].extend(rows)
        else:
            splits["test"].extend(rows)

    for split, rows in splits.items():
        out = data_dir / f"functions_{split}.parquet"
        pd.DataFrame(rows).to_parquet(out, index=False)
        print(f"{split}: {len(rows)} functions -> {out}")

    meta = {
        "scanned_rows": scanned,
        "unique_functions": len(seen_hash),
        "n_repos": len(used_repos),
        "counts": {k: len(v) for k, v in splits.items()},
        "target_n": target_n,
        "min_body_stmts": min_stmts,
        "seed": seed,
    }
    (data_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
