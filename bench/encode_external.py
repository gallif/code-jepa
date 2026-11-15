#!/usr/bin/env python
"""Whole-snippet embeddings from the dedicated baselines, for the text->code tasks of benchmark 4.

  python bench/encode_external.py --pool data/bench/taco_clean.jsonl --models unixcoder gte-modernbert \
      [--queries data/bench/adv_norm.jsonl --query-field query] [--prompt]
Cache: runs/bench/cache/<pool>/ext_<model>.pt = {"rec_ids", "prog": [N_rec, D], "queries"?: [Nq, D], "prompt"?: [N_rec, D]}
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from external import load_external  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--models", nargs="+", default=["unixcoder", "gte-modernbert"])
ap.add_argument("--queries")
ap.add_argument("--query-field", default="query")
ap.add_argument("--prompt", action="store_true")
a = ap.parse_args()
recs = [json.loads(l) for l in open(a.pool)]
out = ROOT / "runs/bench/cache" / Path(a.pool).stem
out.mkdir(parents=True, exist_ok=True)
for name in a.models:
    m = load_external(name)
    res = {"rec_ids": [r["id"] for r in recs], "prog": torch.from_numpy(m.code([r["L4_code"] for r in recs]))}
    if a.queries:
        res["queries"] = torch.from_numpy(m.query([json.loads(l)[a.query_field] for l in open(a.queries)]))
    if a.prompt:
        res["prompt"] = torch.from_numpy(m.query([r.get("prompt", "") for r in recs]))
    torch.save(res, out / f"ext_{name}.pt")
    print(name, "->", out / f"ext_{name}.pt", flush=True)
    del m
    torch.cuda.empty_cache()
