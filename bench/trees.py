#!/usr/bin/env python
"""Build code-only hierarchy records (L2_units / steps with spans, empty descriptions) for arbitrary Python code.

Runs in initzan's hjepa_gen env (it has tree-sitter):
  /mnt/storage/initzan_cache/envs/hjepa_gen/bin/python bench/trees.py IN.jsonl OUT.jsonl [--max-lines 400]

IN rows: {"id": str, "code": str, ...extra}. OUT rows: hierarchy schema + the extra fields; rows whose code
cannot be parsed are written with "skeleton_error".
"""
import argparse
import json
import sys

sys.path.insert(0, "/home/initzan/LLM-JEPA-CODE")
from hjepa_data.skeleton import build_skeleton  # noqa: E402

CFG = {"max_code_lines": 200, "max_blocks": 48, "merge_simple_max_lines": 4, "small_compound_max_lines": 2,
       "split_compound_min_lines": 14, "max_depth": 2}


def to_record(rid, code, sk):
    sk = sk.to_dict()
    by_id = {u["id"]: u for u in sk["units"]}

    def step_node(b):
        kids = [c for c in sk["blocks"] if c["parent"] == b["id"]]
        return {"id": b["id"], "span": b["span"], "kind": b["kind"], "text": "", "children": [step_node(c) for c in kids]}

    def unit_node(u):
        top = [b for b in sk["blocks"] if b["unit"] == u["id"] and b["parent"] is None]
        return {"id": u["id"], "kind": u["kind"], "name": u["name"], "signature": u["signature"], "span": u["span"],
                "responsibility": "", "contract": "",
                "steps": [step_node(b) for b in top],
                "methods": [unit_node(m) for m in sk["units"] if m["parent"] == u["id"]]}

    _ = by_id
    return {"id": rid, "L0_intent": "", "L1_approach": "", "L4_code": code,
            "L2_units": [unit_node(u) for u in sk["units"] if u["parent"] is None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--max-lines", type=int, default=400)
    ap.add_argument("--max-blocks", type=int, default=160)
    a = ap.parse_args()
    cfg = dict(CFG, max_code_lines=a.max_lines, max_blocks=a.max_blocks)
    n = bad = 0
    with open(a.inp) as fi, open(a.out, "w") as fo:
        for line in fi:
            row = json.loads(line)
            code = row.pop("code")
            if not code.endswith("\n"):
                code += "\n"
            try:
                sk, err = build_skeleton(code, cfg)
            except Exception as e:  # noqa: BLE001
                sk, err = None, f"exception: {type(e).__name__}"
            if sk is None:
                rec = {"id": row["id"], "L4_code": code, "L2_units": [], "skeleton_error": err}
                bad += 1
            else:
                rec = to_record(row["id"], code, sk)
            for k, v in row.items():
                rec.setdefault(k, v)
            fo.write(json.dumps(rec) + "\n")
            n += 1
    print(f"trees: {n} rows, {bad} failed", file=sys.stderr)


if __name__ == "__main__":
    main()
