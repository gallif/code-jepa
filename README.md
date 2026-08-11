# Code-JEPA POC

Token-efficient code embeddings via a SWE-JEPA-style pipeline:
signature-only student predicts a frozen teacher's function-body embedding,
then retrieves the correct body with FAISS.

## Goal

Show that a useful coding decision (body retrieval) can be made from
**signature tokens only**, with far fewer tokens than body-in-context baselines.

## Layout

```
src/code_jepa/     # library code
scripts/           # CLI entrypoints
configs/           # experiment configs
EXPERIMENTS.md     # run log
```

Large artifacts live on the GPU machine under:

```
/mnt/storage/users/glifshitz/code-jepa/{data,cache,models,runs}
```

## Remote GPU machine

- Host: `132.66.150.42` (`tzag`), user `glifshitz`
- Use **GPU ranks 2 and 3** (L40S):

```bash
export CUDA_VISIBLE_DEVICES=2,3
```

Code checkout: `~/repos/code-jepa`

## Setup (remote)

```bash
ssh 132.66.150.42
cd ~/repos/code-jepa
conda env create -f environment.yml
conda activate code_jepa
pip install -e .
```

## Experiment ladder

0. `scripts/cache_teacher.py` — teacher body embeddings + sanity checks  
1. `scripts/prepare_data.py` — CodeSearchNet Python, AST sig/body split  
2. `scripts/train_student.py` — SigPredictor + InfoNCE  
3. `scripts/eval_retrieval.py` — Rank@k  
4. `scripts/report_tokens.py` — TokenRatio + Pareto vs baselines  

See `EXPERIMENTS.md` for success bars and run logs.

## Success (POC)

- Test Rank@10 ≥ 5%, Rank@1 ≥ 1%
- Beat mean-pool / CodeBERT-sig baselines
- At that quality, JEPA path uses ≤ 20% input tokens vs full-body encoding baseline
