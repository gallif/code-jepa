#!/usr/bin/env python
"""Eval pools from the hierarchy data, all unseen in training (models trained on CodeContests train only).
  cc_eval.jsonl    CodeContests val + test
  taco_clean.jsonl TACO minus problems whose statement matches a CodeContests-train statement; capped sample
  kodcode_test.jsonl KodCode test (function / class style)
"""
import json, random, re, collections
D = "/mnt/storage/initzan/hjepa_code/hierarchy/full/"
OUT = "/mnt/storage/users/glifshitz/code-jepa/data/bench/"
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()[:300]
rd = lambda f: [json.loads(l) for l in open(D + f)]
cc = rd("code_contests.jsonl")
cc_train = {norm(r["prompt"]) for r in cc if r["split"] == "train"}
cc_train_code = {re.sub(r"\s+", "", r["L4_code"]) for r in cc if r["split"] == "train"}
def dump(name, rows):
    with open(OUT + name, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    print(name, len(rows), "programs", len({r["problem_id"] for r in rows}), "problems")
# val/test share no problem_id with train, but some repeat a train statement or solution: drop those
dump("cc_eval.jsonl", [r for r in cc if r["split"] != "train" and norm(r["prompt"]) not in cc_train
                       and re.sub(r"\s+", "", r["L4_code"]) not in cc_train_code])
ta = [r for r in rd("taco.jsonl") if norm(r["prompt"]) not in cc_train and re.sub(r"\s+", "", r["L4_code"]) not in cc_train_code]
by = collections.defaultdict(list)
for r in ta: by[r["problem_id"]].append(r)
rng = random.Random(0)
multi = [p for p, v in by.items() if len(v) >= 2]; single = [p for p, v in by.items() if len(v) == 1]
rng.shuffle(multi); rng.shuffle(single)
keep, n = [], 0
for p in multi + single:
    if n >= 8000: break
    keep += by[p]; n += len(by[p])
dump("taco_clean.jsonl", keep)
dump("kodcode_test.jsonl", [r for r in rd("kodcode.jsonl") if r["split"] == "test"])
