# Program JEPA

Learn a **semantic latent program state** where software evolution is predictable,
so a coding agent can plan over edits in latent space (V-JEPA 2 analogy).

**Plan:** [`PLAN.md`](PLAN.md) · **Log:** [`EXPERIMENTS.md`](EXPERIMENTS.md) · **Repo:** https://github.com/gallif/code-jepa

## Status (2026-08-13)

| Exp | Status |
|-----|--------|
| **0** Frozen-`E` edit-dynamics baseline | Not started (need commit transitions) |
| **1** JEPA mask representation learning | CPU prep done; waiting on free GPU |
| **2 / 3** JEPA dynamics + planning | After Exp 0/1 |

Details: [`PLAN.md` → Where we stand](PLAN.md#where-we-stand-2026-08-13).

## Goal

Learn `z = E(P)` where edits are predictable (`P(z_t, a_t) → z_{t+1}`), then plan in latent space.

- **Exp 1:** JEPA-train `E` via mask prediction on `full_text`.
- **Exp 0 / 2:** edit dynamics with frozen vs JEPA `E` on the same git transitions.

Historical sig→body POC: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Layout

```
src/code_jepa/     # library code (encoder, predictor, mask, EMA, legacy student)
scripts/           # CLI entrypoints
configs/           # experiment configs (see jepa_exp1.yaml)
PLAN.md            # research plan + Exp checklist
EXPERIMENTS.md     # historical sig→body run log
```

Large artifacts live on the GPU machine under:

```
/mnt/storage/users/glifshitz/code-jepa/{data,cache,models,runs}
```

## Method (Exp 1)

| Role | Model | Input → output |
|------|--------|----------------|
| Encoder `E` | Qwen2.5-Coder-0.5B (trainable warm-start) | `full_text` tokens → per-token features |
| Target `Ē` | EMA of `E` | full sequence → latent targets |
| Predictor `P_mask` | 8× Transformer, d=512 | visible features + mask tokens → masked features |
| Loss | SmoothL1 on masked positions | JEPA feature prediction |
| Mask | Multiblock spans (~40% tokens) | context must not see masked ids |

## Remote GPU machine

- Host: `132.66.150.42` (`tzag`), user `glifshitz`
- Single-GPU jobs: bind one free L40S (prefer rank **2**; leave 3 free unless needed):

```bash
export CUDA_VISIBLE_DEVICES=2
```

Code checkout: `~/repos/code-jepa` (`git pull` from GitHub, or sync from X12SPA).

## Setup (remote)

Env lives on shared storage (home disk is nearly full):

```bash
ssh 132.66.150.42
conda activate /mnt/storage/users/glifshitz/code-jepa/env
cd ~/repos/code-jepa
git pull
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/mnt/storage/users/glifshitz/code-jepa/models/hf
export CODE_JEPA_ROOT=/mnt/storage/users/glifshitz/code-jepa
```

Optional sync without git (from X12SPA edit copy):

```bash
bash scripts/sync_to_remote.sh
```

## Experiment ladder

1. `scripts/prepare_data.py` — CodeSearchNet Python (`full_text` + legacy sig/body)
2. `scripts/prepare_exp1.py` — CPU prep: validate data, prefetch encoder, token-length report
3. `scripts/train_jepa_mask.py` — **Exp 1** JEPA mask pretrain (`configs/jepa_exp1.yaml`)
4. *(next)* edit-conditioned dynamics on git windows — Exp 2
5. *(later)* latent edit ranking / planning — Exp 3

Historical sig→body ladder (cache teacher → SigPredictor → Rank@k → token ladder): see [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Success (Exp 1)

- Masked latent prediction on val beats trivial controls (details in `PLAN.md`)
- Encoder features are trained under the prediction objective (EMA targets), not frozen LM-only
