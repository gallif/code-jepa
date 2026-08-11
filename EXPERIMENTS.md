# Experiment log — Code-JEPA POC

Repo: https://github.com/gallif/code-jepa

## Success bars

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

| ID | Date | Teacher | Layer | N | bs | Loss (best val) | Rank@1 | Rank@10 | TokenRatio | Notes |
|----|------|---------|-------|---|----|-----------------|--------|---------|------------|-------|
| `student_L14_bs512_20260811_153617` | 2026-08-11 | Qwen2.5-Coder-1.5B | 14 | 30k | 512 | in progress (e6 val≈5.19; e0→e7 train 6.07→4.76) | — | — | — | tmux `code_jepa_train`, GPU 2; InfoNCE + learnable logit scale |

Aborted / superseded (no full ckpt kept for POC):

- `student_L14_20260811_153120` — bs=64, killed mid epoch 0 (VRAM underuse)
- `student_L14_bs256_20260811_153427` — bs=256 (~15 GB), killed to bump bs

## Checklist

- [x] Exp 0: teacher cache + variance / off-diag cosine sanity
- [x] Data: ~30k AST-split functions, repo split
- [ ] Exp 1: student InfoNCE meets Rank bars (`student_L14_bs512_*` training)
- [ ] Exp 2: token Pareto claim
- [ ] Exp 3: ablations (optional)
