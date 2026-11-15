#!/usr/bin/env python
"""Language-model drift vs the original Qwen2.5-Coder-0.5B (E0): perplexity and token-level KL.

  python bench/lm_eval.py --models free tree E1 --out runs/bench/results/lm.json

For every corpus, each document is scored by E0 and the model in the same batch (teacher-forced, same E0 tokens).
Per predicted token:
  NLL                         -> perplexity = exp(mean NLL)
  KL(E0 || M), KL(M || E0)    over the shared original vocabulary (ids < 151665), both renormalised
  top-1 agreement             argmax of E0 == argmax of M
  new-token mass              probability M puts on the added node / predictor tokens (ids >= 151665)
Corpora (all held out from training):
  cc_eval_code       CodeContests val/test reference solutions (edited variant, train-overlap removed)
  cc_eval_trainfmt   the training LM format: intent + approach prompt -> code (loss on code tokens only)
  taco_code          TACO solutions not in the training set (1000)
  csn_python         CodeSearchNet python test functions (1000, original identifiers)
  wikitext           WikiText-2 test paragraphs (English prose; forgetting check)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import encoders as E  # noqa: E402
from code_jepa.data.tree import flatten_record  # noqa: E402

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
DATA = ROOT / "data/bench"
V0 = 151665  # original vocabulary (regular + original special tokens)


def corpora(n_taco=1000, n_csn=1000):
    rd = lambda p: [json.loads(l) for l in open(p)]
    cc = rd(DATA / "cc_eval.jsonl")
    out = {"cc_eval_code": [("", r["L4_code"]) for r in cc]}
    tf = []
    for r in cc:
        nodes = flatten_record(r)
        by = {n.level: n.text for n in nodes if n.level in ("intent", "approach")}
        tf.append(("\n\n".join(p for p in (by.get("intent"), by.get("approach")) if p), r["L4_code"]))
    out["cc_eval_trainfmt"] = tf
    out["taco_code"] = [("", r["L4_code"]) for r in rd(DATA / "taco_clean.jsonl")[:n_taco]]
    out["csn_python"] = [("", json.loads(l)["code"]) for l in list(open(DATA / "adv_orig_in.jsonl"))[:n_csn]]
    wt = ROOT / "data/bench_ext/wikitext2_test.jsonl"
    if wt.exists():
        out["wikitext"] = [("", json.loads(l)["text"]) for l in open(wt)]
    return out


def load_lm(name, dev):
    from transformers import AutoModelForCausalLM

    path = E.MODELS[name]
    if name == "E0":
        path = sorted((Path(path) / "snapshots").glob("*"))[-1]
    else:
        path = Path(path) / "backbone"
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()


@torch.no_grad()
def score(model, base, tok, docs, dev, max_len=2048, bs_tokens=2048):
    seqs = []
    for prompt, text in docs:
        p = tok(prompt + "\n\n", add_special_tokens=False)["input_ids"] if prompt else []
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        seqs.append((ids, len(p)))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    acc = {"nll_m": 0.0, "nll_0": 0.0, "kl_0m": 0.0, "kl_m0": 0.0, "agree": 0.0, "newmass": 0.0, "n": 0}
    s = 0
    while s < len(order):
        e = s + 1
        while e < len(order) and len(seqs[order[e]][0]) * (e - s + 1) <= bs_tokens:
            e += 1
        b = order[s:e]
        s = e
        w = max(len(seqs[i][0]) for i in b)
        ids = torch.full((len(b), w), tok.pad_token_id or 0, dtype=torch.long)
        am = torch.zeros((len(b), w), dtype=torch.long)
        lossmask = torch.zeros((len(b), w), dtype=torch.bool)
        for r, i in enumerate(b):
            x, pl = seqs[i]
            ids[r, : len(x)] = torch.tensor(x)
            am[r, : len(x)] = 1
            lossmask[r, max(pl, 1) : len(x)] = True  # predict tokens after the prompt (and never the first)
        ids, am, lossmask = ids.to(dev), am.to(dev), lossmask.to(dev)
        lm = model(input_ids=ids, attention_mask=am).logits[:, :-1].float()
        l0 = base(input_ids=ids, attention_mask=am).logits[:, :-1, :V0].float()
        tgt, m = ids[:, 1:], lossmask[:, 1:]
        full_m = F.log_softmax(lm, -1)
        newmass = full_m[..., V0:].exp().sum(-1)
        lpm = F.log_softmax(lm[..., :V0], -1)
        lp0 = F.log_softmax(l0, -1)
        g = lambda lp: -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        nll_m = -full_m.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # true model NLL (full vocab)
        nll_0 = g(lp0)
        p0, pm = lp0.exp(), lpm.exp()
        kl_0m = (p0 * (lp0 - lpm)).sum(-1)
        kl_m0 = (pm * (lpm - lp0)).sum(-1)
        agree = (lp0.argmax(-1) == lpm.argmax(-1)).float()
        for k, v in (("nll_m", nll_m), ("nll_0", nll_0), ("kl_0m", kl_0m), ("kl_m0", kl_m0), ("agree", agree),
                     ("newmass", newmass)):
            acc[k] += float(v[m].sum())
        acc["n"] += int(m.sum())
        del lm, l0, full_m, lpm, lp0, p0, pm
    n = max(acc["n"], 1)
    return {"ppl_model": math.exp(acc["nll_m"] / n), "ppl_E0": math.exp(acc["nll_0"] / n),
            "KL(E0||M)": acc["kl_0m"] / n, "KL(M||E0)": acc["kl_m0"] / n, "top1_agree": acc["agree"] / n,
            "new_token_mass": acc["newmass"] / n, "tokens": acc["n"], "docs": len(docs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["free", "tree", "E1"])
    ap.add_argument("--out", default=str(ROOT / "runs/bench/results/lm.json"))
    a = ap.parse_args()
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    e0p = sorted((Path(E.MODELS["E0"]) / "snapshots").glob("*"))[-1]
    tok = AutoTokenizer.from_pretrained(e0p)
    base = load_lm("E0", dev)
    C = corpora()
    res = {}
    for name in a.models:
        model = load_lm(name, dev)
        res[name] = {}
        for cname, docs in C.items():
            res[name][cname] = score(model, base, tok, docs, dev)
            print(name, cname, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[name][cname].items()},
                  flush=True)
        del model
        torch.cuda.empty_cache()
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
