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
the matching code vector \(c_v\). The next-token loss on `intent + approach → code`
anchors the backbone; SIGReg regularises node embeddings per level.

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
python scripts/train.py --config configs/default.yaml --data /path/to/hierarchy.jsonl
```

Default backbone is `Qwen/Qwen2.5-Coder-0.5B` (cached under `models/hf`).

## tzag (132.66.150.42)

Code, env, and checkpoints live on storage, not the root disk. Hierarchy JSONL and
Qwen weights are read in place (not copied).

```bash
# from this laptop
bash scripts/sync_tzag.sh

# on tzag; GPU 3 had free memory at last check
ssh 132.66.150.42
source /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_env.sh
bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_train.sh --max-samples 64 --max-steps 20
```

Layout on the server:

- `/mnt/storage/users/glifshitz/code-jepa/repo` — this tree
- `/mnt/storage/users/glifshitz/code-jepa/env` — existing Python 3.11 + PyTorch
- `/mnt/storage/users/glifshitz/code-jepa/models/hf` — Qwen 0.5B / 1.5B (already there)
- `/mnt/storage/initzan/hjepa_code/hierarchy/full/*.jsonl` — data (not copied)
- `/mnt/storage/users/glifshitz/code-jepa/runs/hcode_jepa` — checkpoints

## Test

```bash
pytest -q
```
