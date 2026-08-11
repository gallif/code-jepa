# Experiment log — Code-JEPA POC

## Success bars

| Gate | Bar |
|------|-----|
| Test Rank@10 | ≥ 5% |
| Test Rank@1 | ≥ 1% |
| Baselines | Beat mean-pool sig + CodeBERT-sig |
| Tokens | ≤ 20% of full-body encoding baseline at Rank@10 bar |

## Hardware

- Host: `132.66.150.42` (tzag), user `glifshitz`
- GPUs: `CUDA_VISIBLE_DEVICES=2,3` (L40S; ranks 0–1 often busy)
- Code: `~/repos/code-jepa`
- Storage / env: `/mnt/storage/users/glifshitz/code-jepa`
- Activate: `conda activate /mnt/storage/users/glifshitz/code-jepa/env`
- Local edit copy: `/home/gallif/projects/code-jepa` on X12SPA

## Runs

| ID | Date | Teacher | Layer | N | Loss | Rank@1 | Rank@10 | TokenRatio | Notes |
|----|------|---------|-------|---|------|--------|---------|------------|-------|
| | | | | | | | | | |

## Checklist

- [ ] Exp 0: teacher cache + KNN / variance sanity
- [ ] Data: ~30k AST-split functions, repo split
- [ ] Exp 1: student InfoNCE meets Rank bars
- [ ] Exp 2: token Pareto claim
- [ ] Exp 3: ablations (optional)
