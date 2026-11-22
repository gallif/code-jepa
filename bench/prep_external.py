#!/usr/bin/env python
"""External benchmark data -> code-only rows for bench/trees.py.

advtest:  CodeXGLUE NL-code-search-Adv test (identifiers normalised: Func / arg_i) and the same functions in their
          original form (CodeSearchNet python test, matched by url). Docstrings are stripped from the code; the
          docstring is the query. A seeded subsample of --n-adv functions is the candidate pool for both versions.
swe:      SWE-bench Lite. For each instance, every function / method in the files the gold patch edits (at
          base_commit) is a candidate; gold = functions whose span contains an edited line. Query = issue text.
"""
import argparse
import ast
import json
import random
import re
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
EXT = ROOT / "data/bench_ext"
OUT = ROOT / "data/bench"


def strip_docstring(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    fn = tree.body[0] if tree.body else None
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    lines = code.split("\n")
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            and isinstance(fn.body[0].value.value, str):
        d = fn.body[0]
        if len(fn.body) == 1:
            return None
        del lines[d.lineno - 1 : d.end_lineno]
    out = "\n".join(lines)
    try:
        ast.parse(out)
    except SyntaxError:
        return None
    return out


def advtest(n):
    from datasets import load_dataset

    adv = [json.loads(l) for l in open(EXT / "advtest.jsonl")]
    from huggingface_hub import hf_hub_download; csn = load_dataset("parquet", data_files=hf_hub_download("code-search-net/code_search_net", "python/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"), split="train")
    by_url = {u: c for u, c in zip(csn["func_code_url"], csn["whole_func_string"])}
    rows = []
    for r in adv:
        orig = by_url.get(r["url"])
        if orig is None:
            continue
        a, b = strip_docstring(r["code"]), strip_docstring(textwrap.dedent(orig))
        if not a or not b:
            continue
        rows.append({"id": f"adv{r['id']}", "query": " ".join(r["docstring_tokens"]) if isinstance(r["docstring_tokens"], list)
                     else r["docstring_summary"], "norm": a, "orig": b})
    print("advtest matched", len(rows), "of", len(adv), file=sys.stderr)
    random.Random(0).shuffle(rows)
    rows = rows[:n]
    for ver in ("norm", "orig"):
        with open(OUT / f"adv_{ver}_in.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"id": r["id"], "code": r[ver], "query": r["query"]}) + "\n")


def _fetch(url, tries=4):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            time.sleep(2 * (t + 1))
    return None


def _edited_old_lines(patch):
    """{path: set(old line numbers touched)} from a unified diff."""
    out, path, old = {}, None, 0
    for ln in patch.split("\n"):
        if ln.startswith("--- "):
            p = ln[4:].strip()
            path = p[2:] if p.startswith("a/") else None
            if path:
                out.setdefault(path, set())
        elif ln.startswith("@@") and path:
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+", ln)
            old = int(m.group(1))
        elif path and ln.startswith("-") and not ln.startswith("---"):
            out[path].add(old)
            old += 1
        elif path and ln.startswith("+") and not ln.startswith("+++"):
            out[path].add(max(old - 1, 1))  # insertion point: the line before
        elif path and (ln.startswith(" ") or ln == ""):
            old += 1
    return out


def _functions(src):
    tree = ast.parse(src)
    res = []

    def visit(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = min([d.lineno for d in ch.decorator_list] + [ch.lineno])
                res.append((prefix + ch.name, start, ch.end_lineno))
                visit(ch, prefix + ch.name + ".")
            elif isinstance(ch, ast.ClassDef):
                visit(ch, prefix + ch.name + ".")
    visit(tree, "")
    return res


def swe():
    rows_in, meta = [], []
    lite = [json.loads(l) for l in open(EXT / "swe_lite.jsonl")]
    cache = EXT / "swe_files"
    cache.mkdir(exist_ok=True)
    for inst in lite:
        edits = _edited_old_lines(inst["patch"])
        cands, gold = [], []
        for path, lines_touched in edits.items():
            if not path.endswith(".py"):
                continue
            fp = cache / (inst["instance_id"] + "__" + path.replace("/", "__"))
            if not fp.exists():
                src = _fetch(f"https://raw.githubusercontent.com/{inst['repo']}/{inst['base_commit']}/{path}")
                if src is None:
                    continue
                fp.write_text(src)
            src = fp.read_text()
            try:
                fns = _functions(src)
            except SyntaxError:
                continue
            lines = src.split("\n")
            for qn, s, e in fns:
                fid = f"{inst['instance_id']}::{path}::{qn}"
                code = textwrap.dedent("\n".join(lines[s - 1 : e]))
                cands.append(fid)
                # innermost function containing an edited line counts as gold
                if any(s <= x <= e for x in lines_touched):
                    gold.append((fid, s, e))
                rows_in.append({"id": fid, "code": code, "path": path, "qualname": qn})
        # keep only the innermost gold functions
        inner = [g for g in gold if not any(o != g and g[1] <= o[1] and o[2] <= g[2] for o in gold)]
        if cands and inner:
            meta.append({"instance_id": inst["instance_id"], "repo": inst["repo"], "query": inst["problem_statement"],
                         "candidates": cands, "gold": [g[0] for g in inner]})
    with open(OUT / "swe_funcs_in.jsonl", "w") as f:
        for r in rows_in:
            f.write(json.dumps(r) + "\n")
    with open(OUT / "swe_meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m) + "\n")
    print(f"swe: {len(meta)} instances with gold functions, {len(rows_in)} candidate functions", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["advtest", "swe"])
    ap.add_argument("--n-adv", type=int, default=6000)
    a = ap.parse_args()
    advtest(a.n_adv) if a.what == "advtest" else swe()
