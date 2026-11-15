#!/usr/bin/env python
"""Score the generations from codegen.py (greedy pass@1).

Runs in initzan's hjepa_gen env (its executor runs programs with CPU / memory / time limits in a temp dir):
  /mnt/storage/initzan_cache/envs/hjepa_gen/bin/python bench/score_gen.py --models E0 E1 tree free
HumanEval(+) / MBPP(+) are scored separately with EvalPlus (bench/score_evalplus.sh); this script reads its results.
"""
import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/initzan/LLM-JEPA-CODE")
from hjepa_data.executor import run_tests  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
GEN = ROOT / "runs/bench/gen"
G = ROOT / "data/bench_ext/gen"


def _limits():
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (20, 21))


def _run(cmd, cwd, timeout=20):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, preexec_fn=_limits,
                           env={"PATH": "/usr/bin:/bin", "HOME": cwd})
        return p.returncode
    except subprocess.TimeoutExpired:
        return -9


def multipl_e(job):
    lang, program = job
    with tempfile.TemporaryDirectory(prefix="mple_") as d:
        if lang == "cpp":
            Path(d, "a.cpp").write_text(program)
            if _run(["g++", "-std=c++17", "-O0", "-o", "a.out", "a.cpp"], d, 60) != 0:
                return False
            return _run(["./a.out"], d) == 0
        Path(d, "a.js").write_text(program)
        return _run(["node", "a.js"], d) == 0


def crux(job):
    kind, code, inp, out, completion = job
    m = re.search(r"assert\s+(.*)", completion)
    if not m:
        return False
    line = m.group(1).strip().split("\n")[0]
    if " == " not in line:
        return False
    lhs, rhs = line.rsplit(" == ", 1) if kind == "o" else line.split(" == ", 1)
    if kind == "o":
        check = f"assert f({inp}) == {rhs}"
    else:
        lhs = lhs.strip()
        if not (lhs.startswith("f(") and lhs.endswith(")")):
            return False
        check = f"assert {lhs} == {out}"
    r = run_tests(code, {"kind": "assert", "asserts": [check]}, timeout_s=5, mem_mb=2048)
    return r["passed"]


def cc(job):
    program, tests = job
    return run_tests(program, tests, timeout_s=6, mem_mb=2048, max_tests=20)["passed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["E0", "E1", "tree", "free"])
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    rd = lambda p: [json.loads(l) for l in open(p)]
    crux_data = {r["id"]: r for r in rd(G / "cruxeval.jsonl")}
    mple = {lang: {r["name"]: r for r in rd(G / f"multiple_{lang}.jsonl")} for lang in ("cpp", "js")}
    cc_tests = {}
    for r in rd(ROOT / "data/bench/cc_eval.jsonl"):
        cc_tests.setdefault(r["problem_id"], r["tests"])
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for model in a.models:
            res[model] = {}
            for task in ("crux_o", "crux_i", "cpp", "js", "cc_stmt", "cc_guided"):
                p = GEN / model / f"{task}.jsonl"
                if not p.exists():
                    continue
                rows = rd(p)
                if task.startswith("crux"):
                    jobs = [(task[-1], crux_data[r["task_id"]]["code"], crux_data[r["task_id"]]["input"],
                             crux_data[r["task_id"]]["output"], r["completion"]) for r in rows]
                    ok = list(ex.map(crux, jobs, chunksize=8))
                elif task in ("cpp", "js"):
                    jobs = [(task, r["prompt"] + r["completion"] + "\n" + mple[task][r["task_id"]]["tests"]) for r in rows]
                    ok = list(ex.map(multipl_e, jobs, chunksize=2))
                else:
                    progs = [(r["prompt"] + r["completion"]) if task == "cc_stmt" else r["completion"] for r in rows]
                    ok = list(ex.map(cc, [(pg, cc_tests[r["task_id"]]) for pg, r in zip(progs, rows)], chunksize=2))
                res[model][task] = {"pass@1": sum(ok) / len(ok), "n": len(ok)}
                print(model, task, res[model][task], flush=True)
            ep = GEN / model / "evalplus.json"
            if ep.exists():
                res[model].update(json.loads(ep.read_text()))
    out = ROOT / "runs/bench/results/codegen.json"
    out.write_text(json.dumps(res, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
