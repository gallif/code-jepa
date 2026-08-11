# Code-JEPA POC

Token-efficient code embeddings via a [SWE-JEPA](https://github.com/arcyleung/SWE-JEPA)-style pipeline:
a frozen code LLM encodes the signature; a small **predictor** (student) maps those
token hiddens to the teacher's body latent; FAISS retrieves the matching body.

**Repo:** https://github.com/gallif/code-jepa

## Goal

Show that a useful coding decision (body retrieval) can be made from
**signature tokens only**, with far fewer input tokens than encoding the full
function/body (≤ 20% TokenRatio at the Rank@k bars).

At query time the teacher still runs on the **signature**; the savings are from
not encoding the body.

## Layout

```
src/code_jepa/     # library code
scripts/           # CLI entrypoints
configs/           # experiment configs
EXPERIMENTS.md     # run log + checklist
```

Large artifacts live on the GPU machine under:

```
/mnt/storage/users/glifshitz/code-jepa/{data,cache,models,runs}
```

## Method (short)

| Role | Model | Input → output |
|------|--------|----------------|
| Teacher (frozen) | Qwen2.5-Coder-1.5B @ layer 14 | body text → pooled latent (cached); sig text → token hiddens |
| Student / predictor | `SigPredictor` (2× Transformer, d=512) | sig token hiddens → predicted body latent |
| Loss | In-batch InfoNCE | match correct body vs other bodies in batch |
| Train val | same InfoNCE (selects `best.pt`) | proxy only |
| POC metric | FAISS Rank@k on body corpus | Rank@10 ≥ 5%, Rank@1 ≥ 1% |

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

1. `scripts/prepare_data.py` — CodeSearchNet Python, AST sig/body split  
2. `scripts/cache_teacher.py` — teacher body/sig embeddings + Exp 0 sanity  
3. `scripts/train_student.py` — SigPredictor + InfoNCE (`batch_size: 512` on L40S)  
4. `scripts/eval_retrieval.py` — Rank@k vs teacher mean-pool-sig baseline  
5. `scripts/report_tokens.py` — TokenRatio vs full-body encoding  

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for status, paths, and run logs.

## Success (POC)

- Test Rank@10 ≥ 5%, Rank@1 ≥ 1%
- Beat mean-pool teacher-sig baseline (CodeBERT-sig TBD)
- At that quality, JEPA path uses ≤ 20% input tokens vs full-body encoding baseline
