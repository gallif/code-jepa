#!/usr/bin/env python
"""Benchmarks 1-4 ("where Code-JEPA should stand out over E0") from cached vectors.

  python bench/evaluate.py b1|b2|b3|b4|all  -> runs/bench/results/<b>.json

Conventions
  * Zero-shot similarity = cosine after subtracting the pool mean of that level (per model, per readout).
  * JEPA readouts: node_raw / node_head (the node token) and pool_last / pool_mid (mean over the node's code tokens).
    E0 / E1 are also given both pooled readouts, so the baseline gets its best readout.
  * Text->code zero-shot uses node_head on both sides for JEPA (the trained alignment space); E0 has no aligned space,
    so its zero-shot text->code is pooled-text vs pooled-code (reported, but the probe and external rows are the fair
    comparisons there).
"""
from __future__ import annotations

import ast
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/mnt/storage/users/glifshitz/code-jepa")
CACHE = ROOT / "runs/bench/cache"
DATA = ROOT / "data/bench"
RES = ROOT / "runs/bench/results"
MODELS = ["free", "tree", "E1", "E0"]
EXTERNAL = ["unixcoder", "gte-modernbert"]
rng = np.random.default_rng(0)


# ----------------------------------------------------------------------------- helpers
def load(pool, model):
    p = CACHE / pool / f"{model}.pt"
    return torch.load(p, weights_only=False) if p.exists() else None


def readouts(c):
    return list(c["code"].keys())


def f32(x):
    return x.float().numpy() if torch.is_tensor(x) else np.asarray(x, dtype=np.float32)


def normed(x, center=None):
    x = x.astype(np.float32)
    if center is not None:
        x = x - center
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


def level_idx(c, level):
    return np.array([k for k, n in enumerate(c["nodes"]) if n[2] == level])


def rank_of(sims, gold_mask):
    """sims [Q, C], gold_mask [Q, C] bool -> 1-based rank of the best-ranked gold per query."""
    best_gold = np.where(gold_mask, sims, -np.inf).max(1, keepdims=True)
    return (sims > best_gold).sum(1) + 1


def summarize_ranks(ranks, ks=(1, 5, 10)):
    ranks = np.asarray(ranks)
    out = {f"R@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float((1.0 / ranks).mean())
    out["n"] = int(len(ranks))
    return out


def recs(pool):
    return [json.loads(l) for l in open(DATA / f"{pool}.jsonl")]


def ridge_fit(X, Y, lam):
    Xm, Ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - Xm, Y - Ym
    W = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1], dtype=np.float32), Xc.T @ Yc)
    return lambda Z: (Z - Xm) @ W + Ym


def pick_lambda(X, Y, lams=(1e-1, 1, 10, 100, 1000)):
    n = len(X)
    perm = rng.permutation(n)
    tr, va = perm[: int(0.8 * n)], perm[int(0.8 * n) :]
    best, best_l = -1, lams[0]
    for lam in lams:
        f = ridge_fit(X[tr], Y[tr], lam)
        P, T = normed(f(X[va])), normed(Y[va])
        s = float((P * T).sum(1).mean())
        if s > best:
            best, best_l = s, lam
    return best_l


# ----------------------------------------------------------------------------- B1: algorithm-level matching
def _taco_tags(r):
    t = r["meta"].get("tags")
    if isinstance(t, str):
        try:
            t = ast.literal_eval(t)
        except Exception:  # noqa: BLE001
            t = []
    return sorted(set(t or []))


def _complexity_class(r):
    c = (r.get("L1_approach") or {}).get("complexity") or ""
    m = re.search(r"O\(([^()]*(?:\([^()]*\))?[^()]*)\)", c)
    if not m:
        return None
    e = m.group(1).replace(" ", "").lower()
    e = re.sub(r"\b[a-z]\b", "n", e)  # single-letter size variable -> n
    table = {"1": "O(1)", "logn": "O(log n)", "n": "O(n)", "nlogn": "O(n log n)", "n*logn": "O(n log n)",
             "n^2": "O(n^2)", "n**2": "O(n^2)", "n²": "O(n^2)", "n*n": "O(n^2)", "n^3": "O(n^3+)", "n**3": "O(n^3+)",
             "2^n": "O(n^3+)", "n!": "O(n^3+)"}
    return table.get(e)


def b1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    pool = "taco_clean"
    R = recs(pool)
    pid = np.array([r["problem_id"] for r in R])
    tags = [_taco_tags(r) for r in R]
    cplx = [_complexity_class(r) for r in R]
    tag_counts = collections.Counter(t for ts in tags for t in ts)
    top_tags = [t for t, n in tag_counts.most_common() if n >= 300][:12]
    out = {"pool": pool, "n_programs": len(R), "n_problems": int(len(set(pid))), "top_tags": top_tags,
           "complexity_counts": collections.Counter(c for c in cplx if c), "models": {}}
    for model in MODELS:
        c = load(pool, model)
        if c is None:
            continue
        res = {}
        for level in ("approach", "intent"):
            li = level_idx(c, level)
            ri = np.array([c["nodes"][k][0] for k in li])
            for ro in readouts(c):
                X = f32(c["code"][ro][li])
                ok = ~np.isnan(X).any(1)
                X, rr = X[ok], ri[ok]
                Z = normed(X, X.mean(0))
                S = Z @ Z.T
                np.fill_diagonal(S, -np.inf)
                p = pid[rr]
                # (a) same-problem retrieval (the other solution(s) of the same problem)
                gold = p[:, None] == p[None, :]
                np.fill_diagonal(gold, False)
                has = gold.any(1)
                sp = summarize_ranks(rank_of(S[has], gold[has]))
                # (b) algorithm neighbours: tag Jaccard of top-10 neighbours from *other* problems
                S2 = np.where(gold, -np.inf, S)
                tsets = [set(tags[i]) for i in rr]
                tagged = np.array([bool(t) for t in tsets])
                S2[:, ~tagged] = -np.inf
                top = np.argsort(-S2, axis=1)[:, :10]
                jac = lambda a, b: len(a & b) / len(a | b) if a | b else 0.0
                q = np.where(tagged)[0]
                nn_j = float(np.mean([np.mean([jac(tsets[i], tsets[j]) for j in top[i]]) for i in q]))
                rnd = rng.choice(q, size=(len(q), 10))
                rnd_j = float(np.mean([np.mean([jac(tsets[i], tsets[j]) for j in rnd[n]]) for n, i in enumerate(q)]))
                # (c) linear probes, grouped by problem
                do_probe = level == "approach" and ro != "node_head"
                gkf = GroupKFold(n_splits=3)
                Y = np.array([[t in tags[i] for t in top_tags] for i in rr], dtype=int)
                aucs = []
                yc = np.array([cplx[i] for i in rr], dtype=object)
                mc = np.array([v is not None for v in yc])
                cls_pred, cls_true = [], []
                for tr, te in (gkf.split(X, groups=p) if do_probe else []):
                    sc = StandardScaler().fit(X[tr])
                    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
                    for j in range(len(top_tags)):
                        if Y[tr, j].min() == Y[tr, j].max() or Y[te, j].min() == Y[te, j].max():
                            continue
                        clf = LogisticRegression(C=0.05, max_iter=2000).fit(Xtr, Y[tr, j])
                        aucs.append(roc_auc_score(Y[te, j], clf.decision_function(Xte)))
                    trm, tem = tr[mc[tr]], te[mc[te]]
                    clf = LogisticRegression(C=0.05, max_iter=3000).fit(sc.transform(X[trm]), yc[trm].astype(str))
                    cls_pred += list(clf.predict(sc.transform(X[tem])))
                    cls_true += list(yc[tem].astype(str))
                res[f"{level}/{ro}"] = {
                    "same_problem": sp,
                    "tag_jaccard@10": nn_j, "tag_jaccard_random": rnd_j, "tag_lift": nn_j / max(rnd_j, 1e-9),
                    **({"tag_probe_macro_auc": float(np.mean(aucs)),
                        "complexity_probe_acc": float(np.mean(np.array(cls_pred) == np.array(cls_true))),
                        "complexity_probe_macro_f1": float(f1_score(cls_true, cls_pred, average="macro"))} if do_probe else {}),
                }
                print(model, level, ro, {k: (round(v, 3) if isinstance(v, float) else v["MRR"] if isinstance(v, dict) else v)
                                         for k, v in res[f"{level}/{ro}"].items()}, flush=True)
        out["models"][model] = res
    return out


# ----------------------------------------------------------------------------- B2: localization
def _within_program(c, level, code_ro, text_ro, fit=None, min_cands=3):
    """For each node of `level`, rank its own code node among the nodes of the same program from its description."""
    li = level_idx(c, level)
    C = f32(c["code"][code_ro][li])
    T = f32(c["text"][text_ro][li])
    ok = ~(np.isnan(C).any(1) | np.isnan(T).any(1))
    li, C, T = li[ok], C[ok], T[ok]
    if fit is not None:
        T = fit(T)
    Cn, Tn = normed(C, C.mean(0)), normed(T, T.mean(0) if fit is None else C.mean(0))
    ri = np.array([c["nodes"][k][0] for k in li])
    ranks, chance = [], []
    for r in np.unique(ri):
        sel = np.where(ri == r)[0]
        if len(sel) < min_cands:
            continue
        S = Tn[sel] @ Cn[sel].T
        ranks += list(rank_of(S, np.eye(len(sel), dtype=bool)))
        chance += [1.0 / len(sel)] * len(sel)
    res = summarize_ranks(ranks, ks=(1, 3))
    res["chance_R@1"] = float(np.mean(chance)) if chance else None
    return res


def _swe_eval(meta, cand_vecs, query_vecs):
    """cand_vecs: {fid: vec}; query_vecs [N_meta, D]; both already normalised."""
    ranks, chance = [], []
    for i, m in enumerate(meta):
        cands = [f for f in m["candidates"] if f in cand_vecs]
        gold = set(m["gold"]) & set(cands)
        if not gold or len(cands) < 2:
            continue
        S = query_vecs[i : i + 1] @ np.stack([cand_vecs[f] for f in cands]).T
        ranks.append(int(rank_of(S, np.array([[f in gold for f in cands]]))[0]))
        chance.append(len(gold) / len(cands))
    res = summarize_ranks(ranks, ks=(1, 5, 10))
    res["chance_R@1"] = float(np.mean(chance))
    return res


_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _bm25_tokens(s):
    toks = []
    for w in _IDENT.findall(s):
        toks.append(w.lower())
        parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", w)
        toks += [p.lower() for p in parts if len(parts) > 1]
        toks += [p for p in w.lower().split("_") if p and "_" in w]
    return toks


def _bm25_scores(query, docs, k1=1.2, b=0.75):
    dt = [collections.Counter(_bm25_tokens(d)) for d in docs]
    N = len(docs)
    avg = np.mean([sum(d.values()) for d in dt]) or 1
    df = collections.Counter(t for d in dt for t in d)
    q = set(_bm25_tokens(query))
    out = []
    for d in dt:
        L = sum(d.values())
        s = 0.0
        for t in q:
            if t in d:
                idf = np.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * d[t] * (k1 + 1) / (d[t] + k1 * (1 - b + b * L / avg))
        out.append(s)
    return np.array(out)


def b2():
    out = {"step_localization": {}, "unit_localization": {}, "swe": {}}
    # (a) in-program step / unit localization: description -> its own block among the program's blocks
    for model in MODELS:
        out["step_localization"][model] = {}
        out["unit_localization"][model] = {}
        ctr = load("taco_clean", model)  # probe training pool (different problems from the eval pools)
        for pool, level, key in (("cc_eval", "step", "step_localization"), ("taco_clean", "step", "step_localization"),
                                 ("kodcode_test", "unit", "unit_localization"), ("taco_clean", "unit", "unit_localization")):
            c = load(pool, model)
            if c is None:
                continue
            res = {}
            if model != "E0":
                res["zero_shot/node_head"] = _within_program(c, level, "node_head", "node_head")
            for ro in ("node_raw", "pool_mid", "pool_last"):
                if ro not in c["code"]:
                    continue
                # ridge text->code on TACO programs of *other* problems (first half of problems -> fit, eval on rest)
                li = level_idx(ctr, level)
                pids = np.array([ctr["rec_ids"][ctr["nodes"][k][0]].split("/s")[0] for k in li])
                upid = np.unique(pids)
                fit_p = set(rng.permutation(upid)[: len(upid) // 2])
                trm = np.array([p in fit_p for p in pids])
                X, Y = f32(ctr["text"][ro][li[trm]]), f32(ctr["code"][ro][li[trm]])
                ok = ~(np.isnan(X).any(1) | np.isnan(Y).any(1))
                X, Y = X[ok][:20000], Y[ok][:20000]
                lam = pick_lambda(X, Y)
                fit = ridge_fit(X, Y, lam)
                if pool == "taco_clean":  # evaluate only on the held-out problems
                    cc = dict(c)
                    keep = np.array([c["rec_ids"][n[0]].split("/s")[0] not in fit_p for n in c["nodes"]])
                    cc["code"] = {k: torch.where(torch.from_numpy(keep)[:, None], v, torch.tensor(float("nan"), dtype=v.dtype))
                                  for k, v in c["code"].items()}
                    cc["text"] = c["text"]
                    res[f"probe/{ro}"] = _within_program(cc, level, ro, ro, fit=fit)
                else:
                    res[f"probe/{ro}"] = _within_program(c, level, ro, ro, fit=fit)
                res[f"probe/{ro}"]["lambda"] = lam
                if model == "E0":
                    res[f"zero_shot/{ro}"] = _within_program(c, level, ro, ro)
            out[key][model][pool] = res
            print(key, model, pool, {k: round(v["R@1"], 3) for k, v in res.items()}, flush=True)
    # (b) SWE-bench Lite: issue -> edited function among the functions of the edited file(s)
    meta = [json.loads(l) for l in open(DATA / "swe_meta.jsonl")]
    funcs = recs("swe_funcs")
    fid_of = [r["id"] for r in funcs]
    for model in MODELS:
        c = load("swe_funcs", model)
        if c is None:
            continue
        res = {}
        # code vector of each function = its first top-level unit node
        first_unit = {}
        for k, n in enumerate(c["nodes"]):
            if n[2] == "unit" and n[0] not in first_unit:
                first_unit[n[0]] = k
        for code_ro, q_level, q_ro in (("node_head", "unit", "node_head"), ("node_head", "intent", "node_head"),
                                       ("pool_mid", "unit", "pool_mid"), ("pool_last", "unit", "pool_last")):
            if code_ro not in c["code"] or (model == "E0" and code_ro.startswith("node")):
                continue
            ks = [first_unit[r] for r in range(len(funcs)) if r in first_unit]
            V = f32(c["code"][code_ro][ks])
            ok = ~np.isnan(V).any(1)
            Vn = normed(V, V[ok].mean(0))
            cand = {fid_of[c["nodes"][k][0]]: Vn[j] for j, k in enumerate(ks) if ok[j]}
            Q = f32(c["queries"][q_level][q_ro])
            Qn = normed(Q, Q.mean(0))
            res[f"{code_ro}/q_{q_level}"] = _swe_eval(meta, cand, Qn)
        out["swe"][model] = res
        print("swe", model, {k: round(v["R@1"], 3) for k, v in res.items()}, flush=True)
    for name in EXTERNAL:
        p = CACHE / "swe_funcs" / f"ext_{name}.pt"
        if not p.exists():
            continue
        e = torch.load(p, weights_only=False)
        V = f32(e["prog"])
        cand = {fid_of[i]: v for i, v in enumerate(V)}
        out["swe"][name] = {"zero_shot": _swe_eval(meta, cand, f32(e["queries"]))}
        print("swe", name, out["swe"][name], flush=True)
    # BM25 over function source (identifiers split)
    code_of = {r["id"]: r["L4_code"] for r in funcs}
    ranks, chance = [], []
    for m in meta:
        cands = [f for f in m["candidates"] if f in code_of]
        gold = set(m["gold"]) & set(cands)
        if not gold or len(cands) < 2:
            continue
        s = _bm25_scores(m["query"], [code_of[f] for f in cands])
        s = s + 1e-9 * rng.random(len(s))  # random tie-break
        ranks.append(int(rank_of(s[None], np.array([[f in gold for f in cands]]))[0]))
        chance.append(len(gold) / len(cands))
    out["swe"]["BM25"] = {"zero_shot": {**summarize_ranks(ranks), "chance_R@1": float(np.mean(chance))}}
    print("swe BM25", out["swe"]["BM25"], flush=True)
    return out


# ----------------------------------------------------------------------------- B3: robustness to surface changes
def b3():
    V = recs("variants")
    clean = {r["id"] for r in recs("cc_eval")} | {r["id"] for r in recs("taco_clean")}
    by_base = collections.defaultdict(dict)
    for i, r in enumerate(V):
        if r["base_id"] not in clean:
            continue
        by_base[r["base_id"]][r["kind"]] = i
    pres_kinds = ("rename", "reformat", "rename_reformat")
    out = {"n_bases": len(by_base), "triplet": {}, "change_localization": {}, "advtest": {}}
    for model in MODELS:
        c = load("variants", model)
        if c is None:
            continue
        node_at = {(n[0], n[1]): k for k, n in enumerate(c["nodes"])}
        trip, loc = {}, {}
        for ro in readouts(c):
            X = f32(c["code"][ro])
            for level in ("approach", "intent", "unit"):
                li = level_idx(c, level)
                mu = np.nanmean(X[li], 0)
                nid = "approach" if level == "approach" else "intent" if level == "intent" else "u0"
                acc = {k: [] for k in pres_kinds}
                for base, kinds in by_base.items():
                    o = node_at.get((kinds.get("orig"), nid))
                    muts = [node_at.get((kinds[k], nid)) for k in kinds if k.startswith("mutant") and not V[kinds[k]]["passed"]]
                    muts = [m for m in muts if m is not None]
                    if o is None or not muts:
                        continue
                    for pk in pres_kinds:
                        if pk not in kinds or not V[kinds[pk]]["passed"]:
                            continue
                        p = node_at.get((kinds[pk], nid))
                        if p is None:
                            continue
                        vo, vp = normed(X[[o]], mu)[0], normed(X[[p]], mu)[0]
                        for m in muts:
                            vm = normed(X[[m]], mu)[0]
                            acc[pk].append(float(vo @ vp > vo @ vm))
                trip[f"{level}/{ro}"] = {k: {"acc": float(np.mean(v)), "n": len(v)} for k, v in acc.items() if v}
            # change localization: which leaf step moved most between orig and a killed mutant (same layout)
            hits, chance = [], []
            for base, kinds in by_base.items():
                ro_i = kinds.get("orig")
                for k in kinds:
                    if not k.startswith("mutant") or V[kinds[k]]["passed"]:
                        continue
                    mi = kinds[k]
                    line = V[mi]["mutated_line"]
                    so = [(n[1], n[3], kk) for kk, n in enumerate(c["nodes"]) if n[0] == ro_i and n[2] == "step"]
                    leaf = [s for s in so if not any(t is not s and s[1][0] <= t[1][0] and t[1][1] <= s[1][1] for t in so)]
                    if len(leaf) < 3:
                        continue
                    pairs = [(s, node_at.get((mi, s[0]))) for s in leaf]
                    if any(p is None for _, p in pairs):
                        continue
                    if [c["nodes"][p][3] for _, p in pairs] != [s[1] for s, _ in pairs]:
                        continue  # layouts differ
                    true = [s[1][0] <= line <= s[1][1] for s, _ in pairs]
                    if sum(true) != 1:
                        continue
                    d = [1 - float(normed(X[[s[2]]])[0] @ normed(X[[p]])[0]) for s, p in pairs]
                    hits.append(float(true[int(np.argmax(d))]))
                    chance.append(1 / len(pairs))
            if hits:
                loc[ro] = {"acc": float(np.mean(hits)), "chance": float(np.mean(chance)), "n": len(hits)}
        out["triplet"][model] = trip
        out["change_localization"][model] = loc
        print("b3", model, {k: {kk: round(vv["acc"], 3) for kk, vv in v.items()} for k, v in trip.items()}, loc, flush=True)
    out["advtest"] = advtest_eval()
    return out


def advtest_eval():
    """Docstring -> function (6000-function pool), original vs identifier-normalised code, plus code->code identity."""
    res = {}
    n = len(recs("adv_norm"))
    for model in MODELS + EXTERNAL:
        r = {}
        if model in EXTERNAL:
            co, cn = (CACHE / "adv_orig" / f"ext_{model}.pt"), (CACHE / "adv_norm" / f"ext_{model}.pt")
            if not co.exists() or not cn.exists():
                continue
            eo, en = torch.load(co, weights_only=False), torch.load(cn, weights_only=False)
            combos = {"zero_shot": (f32(eo["prog"]), f32(en["prog"]), f32(eo["queries"]), f32(en["queries"]))}
        else:
            co, cn = load("adv_orig", model), load("adv_norm", model)
            if co is None or cn is None:
                continue
            combos = {}
            for code_ro, ql, qro in (("node_head", "unit", "node_head"), ("pool_mid", "unit", "pool_mid"),
                                     ("pool_last", "unit", "pool_last"), ("node_raw", "unit", "node_raw")):
                if code_ro not in co["code"]:
                    continue

                def unit_vecs(c):
                    first = {}
                    for k, nd in enumerate(c["nodes"]):
                        if nd[2] == "unit" and nd[0] not in first:
                            first[nd[0]] = k
                    return f32(c["code"][code_ro][[first[i] for i in range(n)]])

                combos[code_ro] = (unit_vecs(co), unit_vecs(cn), f32(co["queries"][ql][qro]), f32(cn["queries"][ql][qro]))
        for key, (Co, Cn, Qo, Qn) in combos.items():
            eye = np.eye(n, dtype=bool)
            ro_ = {}
            for ver, C, Q in (("orig", Co, Qo), ("norm", Cn, Qn)):
                S = normed(Q, Q.mean(0)) @ normed(C, C.mean(0)).T
                ro_[ver] = summarize_ranks(rank_of(S, eye))
            mu = Co.mean(0)
            S = normed(Co, mu) @ normed(Cn, mu).T
            ro_["orig->norm identity"] = summarize_ranks(rank_of(S, eye))
            ro_["MRR_drop"] = ro_["orig"]["MRR"] - ro_["norm"]["MRR"]
            r[key] = ro_
        res[model] = r
        print("adv", model, {k: (round(v["orig"]["MRR"], 3), round(v["norm"]["MRR"], 3), round(v["orig->norm identity"]["R@1"], 3))
                             for k, v in r.items()}, flush=True)
    return res


# ----------------------------------------------------------------------------- B4: one model text<->code vs dedicated
def b4():
    out = {"problem_to_solution": {}}
    for pool in ("taco_clean", "cc_eval"):
        R = recs(pool)
        pid = np.array([r["problem_id"] for r in R])
        # one query per problem (its statement), candidates = all programs; gold = solutions of that problem
        first = {}
        for i, p in enumerate(pid):
            first.setdefault(p, i)
        qi = np.array(list(first.values()))
        gold = pid[qi][:, None] == pid[None, :]
        res = {}
        for model in MODELS:
            c = load(pool, model)
            if c is None:
                continue
            r = {}
            for level in ("intent",):  # the statement is encoded at the intent level
                li = level_idx(c, level)
                for ro in ("node_head", "pool_mid", "pool_last"):
                    if ro not in c["code"] or (model == "E0" and ro.startswith("node")):
                        continue
                    C = f32(c["code"][ro][li])
                    Q = f32(c["prompt"][ro])[qi]
                    S = normed(Q, Q.mean(0)) @ normed(C, np.nanmean(C, 0)).T
                    S = np.nan_to_num(S, nan=-1)
                    r[f"{level}/{ro}"] = summarize_ranks(rank_of(S, gold))
            res[model] = r
        for name in EXTERNAL:
            p = CACHE / pool / f"ext_{name}.pt"
            if not p.exists():
                continue
            e = torch.load(p, weights_only=False)
            S = f32(e["prompt"])[qi] @ f32(e["prog"]).T
            res[name] = {"zero_shot": summarize_ranks(rank_of(S, gold))}
        out["problem_to_solution"][pool] = res
        print("b4", pool, {m: {k: round(v["MRR"], 3) for k, v in r.items()} for m, r in res.items()}, flush=True)
    return out


if __name__ == "__main__":
    RES.mkdir(parents=True, exist_ok=True)
    which = sys.argv[1:] or ["all"]
    todo = ["b1", "b2", "b3", "b4"] if "all" in which else which
    for b in todo:
        res = globals()[b]()
        (RES / f"{b}.json").write_text(json.dumps(res, indent=1, default=lambda o: dict(o) if isinstance(o, collections.Counter) else str(o)))
        print("wrote", RES / f"{b}.json", flush=True)
