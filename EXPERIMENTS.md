# Experiment log — Code-JEPA POC

Repo: https://github.com/gallif/code-jepa

> **Active direction:** Program JEPA — see [`PLAN.md`](PLAN.md) (**Where we stand**).  
> Below: status snapshot + **historical** sig→body POC log.

## Program JEPA — status (2026-08-13)

| Exp | Meaning | Status |
|-----|---------|--------|
| **0** Baseline | Frozen non-JEPA `E` + `P_act` on edit transitions | **Not started** — need git/PR `(P_t, a_t, P_{t+1})` data |
| **1** Mask JEPA | Train `E` with EMA mask prediction on `full_text` | **Prep done** (`prepare_exp1.py`); train blocked on free GPU |
| **2** Dynamics | JEPA `E` + `P_act` on same transitions as Exp 0 | Not started |
| **3** Planning | Latent edit search | Deferred |

### Exp 1 prep artifacts

- Script: `scripts/prepare_exp1.py`
- Report: `$CODE_JEPA_ROOT/runs/prep_exp1/prep_report.json` (`ready_for_train: true`)
- Data OK (30k `full_text`); encoder `Qwen2.5-Coder-0.5B` cached under `$HF_HOME`
- Next GPU step: `train_jepa_mask.py --max-steps 20`

### Exp 0 next (CPU-friendly)

- Define transition schema + `prepare_exp0_commits.py` (or similar)
- Scaffold `P_act`; no GPU required until train

---

## Success bars (historical sig→body)

| Gate | Bar |
|------|-----|
| Test Rank@10 | ≥ 5% |
| Test Rank@1 | ≥ 1% |
| Baselines | Beat mean-pool sig (+ CodeBERT-sig when wired) |
| Tokens | ≤ 20% of full-body encoding baseline at Rank@10 bar |

## Hardware

- Host: `132.66.150.42` (tzag), user `glifshitz`
- Train/eval: single GPU is enough — bind `CUDA_VISIBLE_DEVICES=2` (L40S; ranks 0–1 often busy; leave 3 free)
- Code: `~/repos/code-jepa` (also https://github.com/gallif/code-jepa)
- Storage / env: `/mnt/storage/users/glifshitz/code-jepa`
- Activate: `conda activate /mnt/storage/users/glifshitz/code-jepa/env`
- Edit elsewhere: `/home/gallif/projects/code-jepa` on X12SPA, then sync or `git pull` on tzag

## Data

| Item | Value |
|------|-------|
| Source | `code-search-net/code_search_net` Python (streaming) |
| N functions | 30 000 (train 22 757 / val 4 292 / test 2 951) |
| Split | repo-level ~80/10/10 |
| Filter | top-level defs, `min_body_stmts=3`, AST-hash dedupe |
| Paths | `$CODE_JEPA_ROOT/data/functions_{train,val,test}.parquet` |

## Exp 0 — teacher cache

| Item | Value |
|------|-------|
| Teacher | `Qwen/Qwen2.5-Coder-1.5B` |
| Layer | 14 / 28 |
| Cache | `$CODE_JEPA_ROOT/cache/teacher_Qwen_Qwen2.5-Coder-1.5B_L14_{body,sig}.npy` |
| Body off-diag cos (mean) | 0.912 |
| Body feature var (mean) | 3.09 |
| Gate | **PASS** (`exp0_pass: true`) |

Pilot `n=5000` also passed before the full 30k cache.

## Runs

| ID | Date | Teacher | Layer | N | bs | Best val InfoNCE | Rank@1 | Rank@10 | TokenRatio | Notes |
|----|------|---------|-------|---|----|------------------|--------|---------|------------|-------|
| `student_L14_bs512_20260811_153617` | 2026-08-11 | Qwen2.5-Coder-1.5B | 14 | 30k | 512 | 5.189 @ e6 | **3.22%** | **12.06%** | 0.301 | test/corpus=all; beats sig-pool (R@10 5.69%); Rank bars **PASS**; token budget **FAIL** (need ≤0.20) |

Eval artifacts: `$CODE_JEPA_ROOT/runs/student_L14_bs512_20260811_153617/{best.pt,eval_test_all.json,token_report.json,history.json}`

Train trajectory: train loss 6.07→4.28 (e0→e19); val InfoNCE best at e6 then mild rise (final val 5.28). Checkpoint used for Rank@k = `best.pt` (e6).

Aborted / superseded:

- `student_L14_20260811_153120` — bs=64, killed mid epoch 0 (VRAM underuse)
- `student_L14_bs256_20260811_153427` — bs=256 (~15 GB), killed to bump bs

## Exp 2 — tokens

Naive sig/full ratio on all 30k (old gate):

| Item | Mean tokens |
|------|-------------|
| sig | 71.5 |
| body | 154.8 |
| full (sig+body) | 237.7 |
| **TokenRatio** sig/full | **0.301** (budget 0.20) → FAIL |

### Teacher-pool query ladder (test, corpus=all)

Script: `scripts/eval_token_ladder.py`  
Artifact: `$CODE_JEPA_ROOT/runs/token_ladder_test_all.json`

Same FAISS body index; query = frozen teacher **mean-pool** of increasingly rich text. JEPA row uses sig-only tokens + student Rank@k from Exp 1.

| Query rung | Mean tok | Rank@1 | Rank@5 | Rank@10 |
|------------|----------|--------|--------|---------|
| name | 2.6 | 0.00% | 0.03% | 0.07% |
| sig_no_doc | 10.8 | 0.17% | 0.34% | 0.41% |
| sig (teacher-pool) | 53.7 | 3.12% | 4.88% | 5.83% |
| **JEPA (sig → predictor)** | **53.7** | **3.22%** | **8.74%** | **12.06%** |
| sig+body_l1 | 62.4 | 7.42% | 11.22% | 12.57% |
| sig+body_l2 | 70.8 | 13.66% | 18.81% | 20.60% |
| sig+body_l4 | 87.1 | 19.59% | 26.23% | 28.26% |
| sig+body_l8 | 114.0 | 28.53% | 36.23% | 38.66% |
| sig+body_l16 | 149.2 | 37.99% | 46.76% | 50.08% |
| sig+body_l32 | 183.9 | 48.15% | 57.74% | 60.76% |
| full_sig+body | 211.3 | 50.69% | 59.61% | 62.32% |

**Iso-quality vs JEPA Rank@10 (12.06%):** first teacher-pool rung to match is **`sig+body_l1`** (~62.4 tok).  
JEPA / that rung ≈ **0.86** → only ~14% fewer query tokens than leaking **one** body line into the teacher query.

Interpretation:
- Same budget: JEPA ≈ **2×** Rank@10 vs teacher sig-pool (12.1% vs 5.8%).
- Fewer tokens at matched quality: **weak** on this ladder — teacher catches JEPA after +1 body line.
- `full_sig+body` is not 100% because the index is **body-only** encodings (query distribution ≠ key).

## Checklist

- [x] Exp 0: teacher cache + variance / off-diag cosine sanity
- [x] Data: ~30k AST-split functions, repo split
- [x] Exp 1: student InfoNCE meets Rank bars (R@1 3.2%, R@10 12.1%; beats mean-pool sig)
- [x] Exp 2: token ladder vs teacher-pool (iso-quality savings modest; naive ≤20% full-body ratio FAIL)
- [ ] Exp 3: ablations (optional)
