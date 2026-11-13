#!/usr/bin/env python
"""Export a trained run as a self-contained release folder (see code_jepa/encode.py)."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from code_jepa.encode import export_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--variant", required=True, help="e.g. tree or free")
ap.add_argument("--metrics", default=None, help="metrics.jsonl of the run; the final eval row is stored")
a = ap.parse_args()
meta = {"variant": a.variant}
if a.metrics:
    ev = [json.loads(l) for l in open(a.metrics) if '"eval"' in l]
    meta["final_eval"] = {k: v for k, v in ev[-1].items() if not k.startswith("eval_std")}
print(export_model(a.config, a.checkpoint, a.out, meta))
