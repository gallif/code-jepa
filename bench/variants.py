#!/usr/bin/env python
"""Benchmark 3 data: behaviour-preserving variants and behaviour-changing mutants, checked by running the tests.

Runs in initzan's hjepa_gen env (python 3.12 + executor):
  /mnt/storage/initzan_cache/envs/hjepa_gen/bin/python bench/variants.py --out data/bench/variants_raw.jsonl

Per base program:
  rename            consistent renaming of every user-defined identifier to neutral names (layout kept, same lines)
  reformat          ast.unparse round-trip (comments / formatting / quoting normalised; lines move)
  rename_reformat   both
  mutant_k          one-token change (comparison / arithmetic operator, and/or, small int literal +-1), same lines
A base is kept only if the original passes, and each variant is labelled with its test outcome.
"""
import argparse
import ast
import builtins
import io
import json
import keyword
import random
import sys
import tokenize
import zlib
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/home/initzan/LLM-JEPA-CODE")
from hjepa_data.executor import run_tests  # noqa: E402

DATA = "/mnt/storage/users/glifshitz/code-jepa/data/bench/"
OP_SWAP = {"<": "<=", "<=": "<", ">": ">=", ">=": ">", "==": "!=", "!=": "==", "+": "-", "-": "+",
           "*": "+", "//": "*", "%": "//", "+=": "-=", "-=": "+="}
NAME_SWAP = {"and": "or", "or": "and", "True": "False", "False": "True"}
PROTECT = set(dir(builtins)) | set(keyword.kwlist) | {"self", "cls", "Solution", "__name__", "__main__"}


def _tokens(code):
    return list(tokenize.generate_tokens(io.StringIO(code).readline))


def _user_names(tree, entry_point):
    names, imported = set(), set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                imported.add((al.asname or al.name).split(".")[0])
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
    names -= PROTECT | imported
    names = {x for x in names if not (x.startswith("__") and x.endswith("__"))}
    if entry_point:
        names.discard(entry_point)
    return names


def _replace(code, edits):
    """edits: list of (row, col0, col1, new) on 1-based rows; same-line edits applied right to left."""
    lines = code.split("\n")
    for row, c0, c1, new in sorted(edits, key=lambda e: (e[0], -e[1])):
        ln = lines[row - 1]
        lines[row - 1] = ln[:c0] + new + ln[c1:]
    return "\n".join(lines)


def rename(code, entry_point=None):
    tree = ast.parse(code)
    targets = sorted(_user_names(tree, entry_point))
    if not targets:
        return None
    existing = {t.string for t in _tokens(code) if t.type == tokenize.NAME}
    mapping, i = {}, 0
    for name in targets:
        while f"v{i}" in existing:
            i += 1
        mapping[name] = f"v{i}"
        i += 1
    toks = _tokens(code)
    edits = []
    for k, t in enumerate(toks):
        if t.type != tokenize.NAME or t.string not in mapping:
            continue
        prev = toks[k - 1].string if k else ""
        if prev == ".":
            continue  # attribute access
        if t.start[0] != t.end[0]:
            continue
        edits.append((t.start[0], t.start[1], t.end[1], mapping[t.string]))
    out = _replace(code, edits)
    ast.parse(out)
    return out if out != code else None


def reformat(code):
    out = ast.unparse(ast.parse(code)) + "\n"
    return out if out.strip() != code.strip() else None


def mutants(code, rng, k=3):
    toks = _tokens(code)
    cands = []
    for t in toks:
        if t.start[0] != t.end[0]:
            continue
        if t.type == tokenize.OP and t.string in OP_SWAP:
            cands.append((t, OP_SWAP[t.string]))
        elif t.type == tokenize.NAME and t.string in NAME_SWAP:
            cands.append((t, NAME_SWAP[t.string]))
        elif t.type == tokenize.NUMBER and t.string.isdigit() and len(t.string) <= 2:
            cands.append((t, str(int(t.string) + rng.choice([1, -1]) if int(t.string) > 0 else 1)))
    rng.shuffle(cands)
    out = []
    for t, new in cands:
        m = _replace(code, [(t.start[0], t.start[1], t.end[1], new)])
        try:
            ast.parse(m)
        except SyntaxError:
            continue
        out.append({"code": m, "mutated_line": t.start[0], "mutation": f"{t.string}->{new}"})
        if len(out) >= k:
            break
    return out


def job(rec):
    rng = random.Random(zlib.crc32(rec["id"].encode()))
    code, tests, ep = rec["L4_code"], rec["tests"], rec.get("entry_point")
    run = lambda c: run_tests(c, tests, timeout_s=6, mem_mb=2048, max_tests=10)["passed"]
    try:
        if not run(code):
            return []
        rows = [{"kind": "orig", "code": code, "passed": True}]
        r = rename(code, ep)
        f = reformat(code)
        rf = reformat(r) if r else None
        for kind, c in (("rename", r), ("reformat", f), ("rename_reformat", rf)):
            if c:
                rows.append({"kind": kind, "code": c, "passed": run(c)})
        for j, m in enumerate(mutants(code, rng)):
            rows.append({"kind": f"mutant_{j}", **m, "passed": run(m["code"])})
    except Exception as e:  # noqa: BLE001
        return [{"kind": "error", "error": repr(e)[:200], "code": code, "passed": False, "base_id": rec["id"]}]
    for row in rows:
        row["base_id"] = rec["id"]
        row["id"] = f"{rec['id']}::{row['kind']}"
        row["problem_id"] = rec["problem_id"]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DATA + "variants_raw.jsonl")
    ap.add_argument("--n-taco", type=int, default=500)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    cc = [json.loads(l) for l in open(DATA + "cc_eval.jsonl")]
    ta = [json.loads(l) for l in open(DATA + "taco_clean.jsonl")]
    rng = random.Random(0)
    seen, ta_pick = set(), []
    rng.shuffle(ta)
    for r in ta:  # one solution per problem
        if r["problem_id"] in seen or r["tests"]["kind"] not in ("stdio", "call"):
            continue
        seen.add(r["problem_id"])
        ta_pick.append(r)
        if len(ta_pick) >= a.n_taco:
            break
    base = [r for r in cc if r["tests"]["kind"] in ("stdio", "call")] + ta_pick
    n_rows = 0
    with ProcessPoolExecutor(a.workers) as ex, open(a.out, "w") as fo:
        for rows in ex.map(job, base, chunksize=2):
            for row in rows:
                fo.write(json.dumps(row) + "\n")
                n_rows += 1
    print(f"bases {len(base)} -> rows {n_rows}", file=sys.stderr)


if __name__ == "__main__":
    main()
