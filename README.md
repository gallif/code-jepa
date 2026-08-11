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

Env lives on shared storage (home disk is nearly full):

```bash
ssh 132.66.150.42
conda activate /mnt/storage/users/glifshitz/code-jepa/env
cd ~/repos/code-jepa
export CUDA_VISIBLE_DEVICES=2,3
export HF_HOME=/mnt/storage/users/glifshitz/code-jepa/models/hf
export CODE_JEPA_ROOT=/mnt/storage/users/glifshitz/code-jepa
```

Sync code from the login machine (`X12SPA`) after local edits:

```bash
# from /home/gallif/projects/code-jepa
tar czf /tmp/code-jepa.tgz --exclude .git --exclude '__pycache__' .
ssh 132.66.150.42 'cat > /tmp/code-jepa.tgz' < /tmp/code-jepa.tgz
ssh 132.66.150.42 'cd ~/repos/code-jepa && tar xzf /tmp/code-jepa.tgz && rm /tmp/code-jepa.tgz'
```

Or: `bash scripts/sync_to_remote.sh` (uses the same pattern).

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
