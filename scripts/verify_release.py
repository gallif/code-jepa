#!/usr/bin/env python
"""Check that a release folder reproduces its run's final validation metrics (own-head centred cosine)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from code_jepa.encode import CodeJEPAEncoder  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
meta = json.loads((Path(a.model) / "metadata.json").read_text())
enc = CodeJEPAEncoder.from_pretrained(a.model, device="cuda", max_row_tokens=2048, max_nodes=96)  # = training packing
cfg = enc.cfg
recs = [json.loads(l) for l in open(cfg.data.jsonl) if l.strip()]
recs = [r for r in recs if r.get("split") == cfg.data.eval_split]
feats = [f for f in enc.encode(recs, batch_size=8, with_text=True) if not f["dropped"]]
print(f"{meta['variant']}: encoded {len(feats)} val programs")
want = meta.get("final_eval", {})
ok = True
for lv in ("step", "unit", "approach", "intent"):
    c = [n["head"] for f in feats for n in f["nodes"] if n["level"] == lv]
    t = [n["text_head"] for f in feats for n in f["nodes"] if n["level"] == lv]
    if len(c) < 2:
        continue
    c, t = torch.stack(c), torch.stack(t)
    cc = F.cosine_similarity(t - t.mean(0), c - c.mean(0), dim=-1).mean().item()
    ref = want.get(f"eval_cos_centered_{lv}")
    diff = abs(cc - ref) if ref is not None else float("nan")
    ok &= ref is None or diff < 0.02
    print(f"  {lv:8s} n={len(c):4d} cc {cc:+.3f}  training eval {ref:+.3f}  |diff| {diff:.3f}")
print("REPRODUCES" if ok else "MISMATCH")
