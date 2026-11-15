#!/usr/bin/env python
"""Encode a pool of hierarchy records with one or more models and cache all node vectors.

  python bench/encode_pool.py --pool data/bench/cc_eval.jsonl --models free tree E1 E0 [--no-text]

Cache: runs/bench/cache/<pool>/<model>.pt  =
  {"rec_ids": [...], "nodes": [(rec_idx, node_id, level, (s, e))...],
   "code": {readout: float16[N_nodes, D]},          # node code vectors
   "text": {readout: float16[N_nodes, D]},          # the node's own description, encoded at its level
   "prompt": {readout: float16[N_rec, D]}}          # problem statement, encoded at the intent level
Nodes are aligned across models (same order; nodes a model dropped are filled with NaN).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import encoders as E  # noqa: E402
from code_jepa.data.tree import flatten_record  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--models", nargs="+", default=["free", "tree", "E1", "E0"])
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--queries", default=None, help="jsonl with a text field to encode as queries")
    ap.add_argument("--query-field", default="query")
    ap.add_argument("--query-levels", nargs="+", default=["unit", "intent"])
    a = ap.parse_args()
    pool = Path(a.pool)
    recs = [json.loads(l) for l in open(pool)]
    out_dir = ROOT / "runs/bench/cache" / pool.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    # canonical node list
    nodes, texts = [], []
    for ri, r in enumerate(recs):
        for n in flatten_record(r, step_depth=2):
            nodes.append((ri, n.node_id, n.level, tuple(n.span)))
            texts.append(n.text)
    key = {(ri, nid): k for k, (ri, nid, _, _) in enumerate(nodes)}
    for name in a.models:
        t0 = time.time()
        m = E.load(name)
        code: dict[str, np.ndarray] = {}
        for s in range(0, len(recs), a.chunk):
            enc = m.encode_code(recs[s : s + a.chunk])
            for j, ns in enumerate(enc):
                if not ns:
                    continue
                for n in ns:
                    k = key.get((s + j, n["node_id"]))
                    if k is None:
                        continue
                    for ro, v in n["vecs"].items():
                        if ro not in code:
                            code[ro] = np.full((len(nodes), v.shape[0]), np.nan, dtype=np.float16)
                        code[ro][k] = v
            print(f"[{name}] code {min(s + a.chunk, len(recs))}/{len(recs)} {time.time() - t0:.0f}s", flush=True)
        res = {"rec_ids": [r["id"] for r in recs], "nodes": nodes,
               "code": {k: torch.from_numpy(v) for k, v in code.items()}}
        if not a.no_text:
            text: dict[str, np.ndarray] = {}
            for lv in E.LEVEL_NAMES:
                ks = [k for k, n in enumerate(nodes) if n[2] == lv]
                if not ks:
                    continue
                tv = m.encode_text([texts[k] for k in ks], lv)
                for ro, v in tv.items():
                    if ro not in text:
                        text[ro] = np.full((len(nodes), v.shape[1]), np.nan, dtype=np.float16)
                    text[ro][ks] = v
            pv = m.encode_text([r.get("prompt", "") for r in recs], "intent")
            res["text"] = {k: torch.from_numpy(v) for k, v in text.items()}
            res["prompt"] = {k: torch.from_numpy(v.astype(np.float16)) for k, v in pv.items()}
        if a.queries:
            qs = [json.loads(l)[a.query_field] for l in open(a.queries)]
            res["queries"] = {lv: {k: torch.from_numpy(v.astype(np.float16)) for k, v in m.encode_text(qs, lv).items()}
                              for lv in a.query_levels}
        torch.save(res, out_dir / f"{name}.pt")
        print(f"[{name}] done {time.time() - t0:.0f}s -> {out_dir / (name + '.pt')}", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
