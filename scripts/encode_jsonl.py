#!/usr/bin/env python
"""Encode a hierarchy JSONL with a release folder; writes one .pt with per-node features.

  python scripts/encode_jsonl.py --model release/code-jepa-0.5b-free-s82 --data hierarchy.jsonl --split val --out val_feats.pt
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
from code_jepa.encode import CodeJEPAEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--split", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--max-samples", type=int, default=None)
ap.add_argument("--batch-size", type=int, default=4)
ap.add_argument("--with-text", action="store_true")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()
enc = CodeJEPAEncoder.from_pretrained(a.model, device=a.device)
recs = []
for line in open(a.data):
    if not line.strip():
        continue
    r = json.loads(line)
    if a.split and r.get("split") not in (None, a.split):
        continue
    recs.append(r)
    if a.max_samples and len(recs) >= a.max_samples:
        break
feats = []
for i in range(0, len(recs), 64):
    feats.extend(enc.encode(recs[i:i + 64], batch_size=a.batch_size, with_text=a.with_text))
    print(f"{min(i + 64, len(recs))}/{len(recs)}", flush=True)
torch.save({"model": a.model, "data": a.data, "split": a.split, "records": feats}, a.out)
print(f"wrote {a.out}: {len(feats)} records, {sum(1 for f in feats if f['dropped'])} dropped")
