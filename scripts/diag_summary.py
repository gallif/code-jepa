#!/usr/bin/env python3
"""Phase 2 pass rule (pre-registered 2026-11-10, plans/gradient_flow_fix_2026-11-10.md, Phase 2)."""
import json, sys
from pathlib import Path
import os
R = Path(os.environ.get("CODE_JEPA_ROOT", "/mnt/storage/users/glifshitz/code-jepa")) / "runs" / "diag"
LEVELS = ["step", "unit", "approach", "intent"]
E1 = [json.loads(l) for l in (R.parent / "stopgrad" / "e1" / "metrics.jsonl").open() if '"eval"' in l]


def lm_max(n_samples: int) -> float:
    """E1 eval_lm at the same number of training samples (E1 batch 4), + 0.02."""
    step = n_samples / 4
    return min(E1, key=lambda r: abs(r["step"] - step))["eval_lm"] + 0.02

STRICT = "--strict" in sys.argv
NAMES = [a for a in sys.argv[1:] if a.startswith("e4_diag_")] or ["e4_diag_a1", "e4_diag_b1", "e4_diag_a2", "e4_diag_b2"]
for name in NAMES:
    f = R / name / "metrics.jsonl"
    if not f.exists():
        print(f"{name}: not started"); continue
    rows = [json.loads(l) for l in f.open()]
    ev = [r for r in rows if r["split"] == "eval"]
    tr = [r for r in rows if r["split"] == "train"]
    first, last = ev[0], ev[-1]
    print(f"\n== {name}: eval step {last['step']}  (train step {tr[-1]['step'] if tr else 0})")
    checks = []
    cfg = json.loads((R / name / "config.json").read_text())
    LM_MAX = lm_max(last["step"] * cfg["train"]["batch_size"])
    lm = last["eval_lm"]; checks.append(lm <= LM_MAX)
    print(f"  LM {lm:.4f} (max {LM_MAX:.4f}, E1 at equal samples + 0.02) {'ok' if lm <= LM_MAX else 'FAIL'}")
    print(f"  {'level':9s} {'erank_code':>17s} {'erank_text':>17s} {'cos_centered':>16s} {'cc_multi':>7s} {'gn sig/view (code)':>19s}")
    recent = [r for r in tr if r["step"] > last["step"] - 500 and "gn_view_code_step" in r]
    for lv in LEVELS:
        ec0, ec1 = first[f"eval_erank_code_{lv}"], last[f"eval_erank_code_{lv}"]
        et0, et1 = first[f"eval_erank_text_{lv}"], last[f"eval_erank_text_{lv}"]
        cc0, cc1 = first[f"eval_cos_centered_{lv}"], last[f"eval_cos_centered_{lv}"]
        ccm = last.get(f"eval_cos_centered_{lv}_multi", float("nan"))
        gv = sum(r[f"gn_view_code_{lv}"] for r in recent) / max(len(recent), 1)
        gs = sum(r[f"gn_sigreg_code_{lv}"] for r in recent) / max(len(recent), 1)
        ratio = gs / gv if gv > 0 else float("inf")
        ok = [ec1 >= 0.5 * ec0, et1 >= 0.5 * et0, cc1 > cc0, 0.1 <= ratio <= 10]
        checks += ok
        flag = "" if all(ok) else "  <- " + ",".join(n for n, o in zip(["code rank", "text rank", "cos", "grad ratio"], ok) if not o)
        print(f"  {lv:9s} {ec0:6.1f} -> {ec1:6.1f}   {et0:6.1f} -> {et1:6.1f}   {cc0:6.3f} -> {cc1:6.3f} {ccm:7.3f} {ratio:12.2f}{flag}")
    if STRICT:  # minimum effects, fixed before the 2026-11-10 Phase 2b launch
        cc = lambda r, lv: r[f"eval_cos_centered_{lv}"]
        strict = {
            "step cc >= +0.10": cc(last, "step") >= 0.10,
            "approach cc >= step0 + 0.05": cc(last, "approach") >= cc(first, "approach") + 0.05,
            "intent cc >= step0 + 0.05": cc(last, "intent") >= cc(first, "intent") + 0.05,
            "level_sep >= step0 + 0.10": last["eval_level_sep_acc"] >= first["eval_level_sep_acc"] + 0.10,
        }
        checks += list(strict.values())
        print("  strict: " + "; ".join(f"{k} {'ok' if v else 'FAIL'}" for k, v in strict.items()))
    ls = last.get("eval_level_sep_acc"); lc = last.get("eval_level_sep_chance")
    print(f"  level_sep_acc {first.get('eval_level_sep_acc', 0):.3f} -> {ls:.3f} (chance {lc:.3f});"
          f" multi {last.get('eval_level_sep_acc_multi_unit', 0):.3f}")
    done = last["epoch"] == cfg["train"]["epochs"]  # the final eval row carries epoch = train.epochs
    print(f"  VERDICT: {'PASS' if all(checks) else 'FAIL'}{'' if done else ' (provisional, run not finished)'}")
