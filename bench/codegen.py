#!/usr/bin/env python
"""Greedy code generation (pass@1) for the backbones of E0 / E1 / tree / free on standard benchmarks.

  python bench/codegen.py --models E0 E1 tree free --tasks humaneval mbpp crux_o crux_i cpp js cc_stmt cc_guided
Writes runs/bench/gen/<model>/<task>.jsonl with {"task_id", "prompt", "completion"}; scoring is in score_gen.py.

Prompts follow the usual base-model conventions:
  humaneval / mbpp   EvalPlus prompts, plain completion (scored with EvalPlus after its sanitizer)
  crux_o / crux_i    CRUXEval direct-prediction prompts with the two official in-context examples
  cpp / js           MultiPL-E HumanEval prompts with the dataset's stop tokens
  cc_stmt            CodeContests val/test (clean): problem statement in a docstring -> stdin/stdout program
  cc_guided          the training LM format: gold intent + approach text -> code (not a standard metric; it shows
                     what the LM objective learned, and the approach text describes the solution)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import encoders as E  # noqa: E402
from code_jepa.data.tree import flatten_record  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
G = ROOT / "data/bench_ext/gen"
OUT = ROOT / "runs/bench/gen"

CRUX_O = '''You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(n):
    return n
assert f(17) == ??
[/PYTHON]
[ANSWER]
assert f(17) == 17
[/ANSWER]

[PYTHON]
def f(s):
    return s + "a"
assert f("x9j") == ??
[/PYTHON]
[ANSWER]
assert f("x9j") == "x9ja"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[ANSWER]
'''

CRUX_I = '''You will be given a function f and an output in the form f(??) == output. Find any input such that executing f on the input leads to the given output. There may be multiple answers, but you should only output one. In [ANSWER] and [/ANSWER] tags, complete the assertion with one such input that will produce the output when executing the function.

[PYTHON]
def f(my_list):
    count = 0
    for i in my_list:
        if len(i) % 2 == 0:
            count += 1
    return count
assert f(??) == 3
[/PYTHON]
[ANSWER]
assert f(["mq", "px", "zy"]) == 3
[/ANSWER]

[PYTHON]
def f(s1, s2):
    return s1 + s2
assert f(??) == "banana"
[/PYTHON]
[ANSWER]
assert f("ba", "nana") == "banana"
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[ANSWER]
'''

PY_STOPS = ["\nif __name__", "\ndef main(", "\nprint(", "\n```", "\nassert "]


def tasks(name):
    rd = lambda p: [json.loads(l) for l in open(p)]
    if name in ("humaneval", "mbpp"):
        return [(r["task_id"], r["prompt"], PY_STOPS, 768) for r in rd(G / f"{name}_prompts.jsonl")]
    if name == "crux_o":
        return [(r["id"], CRUX_O.replace("{code}", r["code"]).replace("{input}", r["input"]), ["[/ANSWER]"], 128)
                for r in rd(G / "cruxeval.jsonl")]
    if name == "crux_i":
        return [(r["id"], CRUX_I.replace("{code}", r["code"]).replace("{output}", r["output"]), ["[/ANSWER]"], 128)
                for r in rd(G / "cruxeval.jsonl")]
    if name in ("cpp", "js"):
        return [(r["name"], r["prompt"], list(r["stop_tokens"]), 768) for r in rd(G / f"multiple_{name}.jsonl")]
    cc = rd(ROOT / "data/bench/cc_eval.jsonl")
    seen, out = set(), []
    for r in cc:  # one generation per problem
        if r["problem_id"] in seen:
            continue
        seen.add(r["problem_id"])
        if name == "cc_stmt":
            p = f'"""\n{r["prompt"].strip()}\n"""\n# Python 3 solution: read from standard input, write to standard output.\n'
            out.append((r["problem_id"], p, ['\n"""', "\n```", "\n\n\n\n"], 1024))
        elif name == "cc_guided":
            by = {n.level: n.text for n in flatten_record(r) if n.level in ("intent", "approach")}
            p = "\n\n".join(x for x in (by.get("intent"), by.get("approach")) if x)  # training format: no separator
            out.append((r["problem_id"], p, ['\n"""', "\n```", "\n\n\n\n"], 1024))
    return out


def load(name, dev):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if name == "E0":
        path = sorted((Path(E.MODELS["E0"]) / "snapshots").glob("*"))[-1]
    else:
        path = Path(E.MODELS[name]) / "backbone"
    tok = AutoTokenizer.from_pretrained(path, padding_side="left")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
    return m, tok


@torch.no_grad()
def generate(m, tok, items, dev, bs=32):
    out = {}
    order = sorted(range(len(items)), key=lambda i: -len(items[i][1]))
    for s in range(0, len(order), bs):
        b = [items[i] for i in order[s : s + bs]]
        enc = tok([x[1] for x in b], return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
        stops = sorted({st for x in b for st in x[2]})
        g = m.generate(**enc, max_new_tokens=max(x[3] for x in b), do_sample=False, stop_strings=stops,
                       tokenizer=tok, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        for x, row in zip(b, g[:, enc["input_ids"].shape[1] :]):
            text = tok.decode(row, skip_special_tokens=True)
            cut = min([text.find(st) for st in x[2] if st in text] or [len(text)])
            out[x[0]] = text[:cut]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["E0", "E1", "tree", "free"])
    ap.add_argument("--tasks", nargs="+", default=["humaneval", "mbpp", "crux_o", "crux_i", "cpp", "js", "cc_stmt", "cc_guided"])
    a = ap.parse_args()
    dev = torch.device("cuda")
    for name in a.models:
        m, tok = load(name, dev)
        for t in a.tasks:
            items = tasks(t)
            outp = OUT / name / f"{t}.jsonl"
            if outp.exists() and sum(1 for _ in open(outp)) == len(items):
                print(name, t, "cached", flush=True)
                continue
            outp.parent.mkdir(parents=True, exist_ok=True)
            comp = generate(m, tok, items, dev, bs=16 if t.startswith("cc") else 32)
            with open(outp, "w") as f:
                for tid, prompt, _, _ in items:
                    f.write(json.dumps({"task_id": tid, "prompt": prompt, "completion": comp[tid]}) + "\n")
            print(name, t, len(items), "done", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
