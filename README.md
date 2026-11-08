# Code-JEPA (phase 2)

Hierarchical latent representations of source code, supervised by LLM-generated
descriptions at multiple abstraction levels. See
[plans/code_jepa_research_plan.md](plans/code_jepa_research_plan.md).

Phase 1 (Program JEPA) is preserved at git tag `phase1-final`.

## Model

A pretrained Qwen2.5-Coder backbone encodes **code only** with one special token per
tree node (step, unit, approach, intent). A tree attention mask lets each node see
only its span or its children. Each node's description is encoded separately; a
level-specific predictor token produces one vector \(p_v\) that is pulled toward
the matching code vector \(c_v\), with stop-gradient on the view target. The next-token
loss on `intent + approach → code` anchors the backbone. SIGReg with a 256-embedding
bank regularises the un-detached code-node embeddings per level (`sigreg.side: code`).

## Setup

```bash
conda activate code_jepa
pip install -e ".[dev]"
```

## Data

JSONL records in the collaborator's schema (`L0_intent`, `L1_approach`, `L2_units`
with `steps` / `methods`, `L4_code`). A two-unit fixture lives at
`tests/fixtures/sample.jsonl`.

```bash
python scripts/validate_data.py --data /path/to/hierarchy.jsonl --max-samples 200
```

## Train

```bash
python scripts/train.py --config configs/default.yaml --data tests/fixtures/sample.jsonl --max-steps 2 --cpu --tiny
python scripts/train.py --config configs/tzag_e4_sgcode.yaml
```

Default backbone is `Qwen/Qwen2.5-Coder-0.5B` (cached under `models/hf`).

Live comparison is **E1 vs E2 vs E4** at 4 epochs, with stop-grad and SIGReg on code
embeddings, in both view directions (`configs/tzag_e{2,4}_{sgcode,c2t}.yaml`, with E1
from `configs/tzag_e1_stopgrad.yaml`). Without stop-grad, program-level rank collapsed
by step 400. The 2026-11-05/06 stop-grad runs had SIGReg on the detached target (no
gradient) and are discarded; see `AGENTS.md`.

## tzag (132.66.150.42)

Code, env, and checkpoints live on storage, not the root disk. Hierarchy JSONL and
Qwen weights are read in place (not copied).

```bash
# from this laptop
bash scripts/sync_tzag.sh

ssh 132.66.150.42
source /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_env.sh
# smoke
bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_train.sh --max-samples 64 --max-steps 20
# 4-epoch E2 / E4 x {sgcode, c2t}, queued on idle L40S (>=40GB free, no other users)
bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_sweep.sh --spec configs/sgfix_jobs.json
```

Layout on the server:

- `/mnt/storage/users/glifshitz/code-jepa/repo` — this tree
- `/mnt/storage/users/glifshitz/code-jepa/env` — existing Python 3.11 + PyTorch
- `/mnt/storage/users/glifshitz/code-jepa/models/hf` — Qwen 0.5B / 1.5B (already there)
- `/mnt/storage/initzan/hjepa_code/hierarchy/full/*.jsonl` — data (not copied)
- `/mnt/storage/users/glifshitz/code-jepa/runs/sgfix/` — live 4-epoch E2/E4 runs
- `/mnt/storage/users/glifshitz/code-jepa/runs/stopgrad/` — E1 baseline (E2/E4 there are discarded)
- `/mnt/storage/users/glifshitz/code-jepa/runs/e0_e4/` — collapsed 1-epoch wave
- `/mnt/storage/users/glifshitz/code-jepa/runs/sweep/` — hyperparameter sweep

## Test

```bash
pytest -q
```
