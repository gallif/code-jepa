# Agent notes — Code-JEPA (phase 2)

This file is for incoming agents. Read it before changing training, data packing, or remote runs. Canonical research writeup is `plans/code_jepa_research_plan.md`; the paper draft is `paper/draft.tex`. Do not treat this file as a substitute for those.

## What we are building

A **span-grounded hierarchical LLM-JEPA** on a collaborator's Python hierarchy JSONL (111.5k programs: CodeContests, TACO, KodCode). Not a reproduction of vision H-JEPA or of NCM generation.

A shared Qwen2.5-Coder encoder sees **code only**. One special token per tree node (step, unit, approach, intent). A 4D tree attention mask lets each node see its span or its children. Each node's description is encoded with a level-specific predictor token. View loss is \(1-\cos(p_v,c_v)\), averaged per level. LM loss is `intent + approach → code`. SIGReg (LeJEPA / H-JEPA) regularises projected node embeddings. Stop-grad on the view target plus SIGReg (with a FIFO bank) on the **un-detached code-node embeddings** is the current recipe (see below). No EMA.

The hierarchy is **given by tree-sitter spans in the data**, not inferred. Training does not invent node counts.

Phase 1 (single-vector Program JEPA) is frozen at git tag `phase1-final`.

## Hard constraints

- Python 3 only. KodCode is CC-BY-NC: combined data is **non-commercial research only**.
- **Do not copy** the collaborator dataset into this repo or onto the tzag root disk. Read JSONL in place.
- Do not dump HuggingFace / pip / torch caches into `$HOME` on tzag. Use `scripts/tzag_env.sh`.
- Checkpoints, env, and code on tzag live under `/mnt/storage/users/glifshitz/code-jepa`.
- Do not commit `.pt`, `models/`, `runs/`, or JSONL data. `.gitignore` uses **`/data/`** (repo-root only) so `src/code_jepa/data/` stays tracked. Never change that back to unanchored `data/`.
- Do not start a long training run or git commit unless the user asks.
- Local laptop has 2× RTX 3090 + a GTX 1650. Prefer **tzag** (`132.66.150.42`, 4× L40S 48 GB) for real runs.
- Git remote is HTTPS but this environment has no HTTPS credentials. `gh` is logged in as `gallif` with **SSH**. Push with `git push git@github.com:gallif/code-jepa.git HEAD:main`, then fetch that URL into `refs/remotes/origin/main`. Do not `git config` the remote unless asked.

## Terminology (easy to get wrong)

- **Program** in the data stats = one annotated JSONL record (one solution). It is **not** a hierarchy level.
- **Intent** and **approach** are two texts over the **same whole-file span**. They are not two scales. They need two tokens, not two heads on one vector.
- **Unit** = function / class / method / `main`. **Step** = tree-sitter block, nested at most twice.
- **Node** = one (span, text, level) row in the flattened tree. Median 7–8 nodes per program, max ~81.
- Most programs are a **single unit** (63–74%). Unit and program usually share a span. Report single-unit vs multi-unit separately or hierarchy claims are confounded by paraphrase matching.

## Repo map

| Path | Role |
|---|---|
| `src/code_jepa/model.py` | `HierarchicalCodeJEPA`: shared CausalLM, projection heads, view + LM + SIGReg |
| `src/code_jepa/data/tree.py` | Flatten JSONL → nodes |
| `src/code_jepa/data/packer.py` | Line→token map, 4D tree mask, LM row |
| `src/code_jepa/data/dataset.py` | JSONL dataset; packs **all kept records at init** (CPU RAM, not GPU) |
| `src/code_jepa/losses.py` | Cosine view distance, SIGReg, `EmbeddingBank` |
| `src/code_jepa/metrics.py` | Per-level cosine / std / effective rank, within-program level confusion |
| `src/code_jepa/tokenize_utils.py` | Resolves `Qwen/...` names to a local hub snapshot under `cache_dir` |
| `src/code_jepa/config.py` | YAML-backed dataclasses |
| `scripts/train.py` | Train + val eval loop, `metrics.jsonl`, `--override key=value` |
| `scripts/validate_data.py` | Pack stats / drop reasons |
| `scripts/sync_tzag.sh` | rsync **code only** to tzag storage |
| `scripts/tzag_env.sh` | Offline HF + all caches on storage |
| `scripts/tzag_train.sh` | Single-GPU smoke (defaults `CUDA_VISIBLE_DEVICES=3`) |
| `scripts/tzag_run_e0_e4.sh` | **Collapsed** 1-epoch E0–E4 wave; keep for comparison |
| `scripts/tzag_run_stopgrad.sh` | 4-epoch E1/E2/E4 with **inert SIGReg** (2026-11-06); superseded |
| `scripts/tzag_sweep.sh` | Queue a jobs spec (`--spec`, default `configs/sweep_jobs.json`) on idle L40S; skips GPUs running other users' processes |
| `configs/sgfix_jobs.json` | **Live** queue: E2/E4 × {text→code, code→text}, stop-grad + SIGReg on code |
| `configs/default.yaml` | Local defaults (still `stop_grad_target: false`; do not use for tzag) |
| `configs/remote_tzag.yaml` | Earlier smoke config |
| `configs/tzag_e{0,1,2,4}.yaml` | 1-epoch collapse wave |
| `configs/tzag_e{1,2,4}_stopgrad.yaml` | 4-epoch configs that ran with SIGReg on the detached target (`sigreg.side: target`). E1 is still the LM-only baseline |
| `configs/tzag_e{2,4}_{sgcode,c2t}.yaml` | **Live**: stop-grad, `sigreg.side: code`; `sgcode` = text→code, `c2t` = code→text |
| `tests/fixtures/sample.jsonl` | Two-unit fixture |
| `paper/` | Proposal draft + bib |

Python package lives under `src/`. `PYTHONPATH=src` (pytest already sets this). Local conda env name: `code_jepa`. On tzag use `/mnt/storage/users/glifshitz/code-jepa/env` (PyTorch 2.6+cu124, transformers 5.15). Do not create a new remote env.

## Data (not in git)

On tzag, in place:

- `/mnt/storage/initzan/hjepa_code/hierarchy/full/code_contests.jsonl` (11,677 records; splits `train` 11448 / `val` 101 / `test` 128)
- TACO and KodCode JSONL sit next to it. Do not mix sources until a **cluster re-split** exists: 5,034 of 6,365 CodeContests problems also appear in TACO.

Record fields used by the packer: `L0_intent`, `L1_approach`, `L2_units` (with `steps` / `methods` / `span`), `L4_code`, optional `split` / `id`. Spans are 1-based line ranges.

Val is tiny (101). That is the official split; do not silently carve a new one without saying so.

## Model / packing details that bite

- Attention is **`eager`** because the view row uses a 4D additive mask (`attention_mask={"full_attention": ...}`). Do not switch to flash/SDPA without a mask path that still encodes the tree.
- If `hierarchy.node_levels` drops units/steps (E2), approach/intent tokens must see **all code tokens**. Packer already does this when a program-level node has no packed children.
- `view.levels` filters which levels enter view/SIGReg. `view.weight: 0` skips the view forward in training (E1) but eval still requests embeddings.
- `jepa_dropout` (E2/E4: 0.5) skips the view term on a random fraction of **training** steps. `view=0` in logs is expected, not a bug.
- Two graphs per step (view row up to `max_row_tokens`, then LM row) are held until backward.
- Special tokens are mean-init. Node tokens get **no LM gradient**; that is why SIGReg is on for E2/E4.
- `sigreg.side` picks what SIGReg regularises: `code` (default; node-token projections), `text` (predictor tokens), `both`, or `target` (legacy). `code`/`text` are always un-detached, whatever `view.direction` is. `target` + `stop_grad_target` gives SIGReg **no gradient**; the model warns. The key is `side`, not `on`: YAML 1.1 parses `on:` as `true`.
- With `stop_grad_target` and text→code, node-token embeddings get gradient only from SIGReg. The shared backbone and the per-level heads (shared by code and text) still get view gradient through the text side, so targets drift even when detached.
- SIGReg runs whenever `sigreg.weight > 0`, even with `view.weight: 0` (before 2026-11-08 it was silently skipped).
- Per-batch SIGReg on intent/approach is ~4 vectors (`batch_size=4`). Set `sigreg.bank_size: 256` so the regulariser sees a FIFO of detached embeddings. Scale of the bank term is `n_current / n_pooled`.
- `view.stop_grad_target: true` detaches the view target. Without it, program-level rank collapsed by step 400. Under `direction: code_to_text` the target is the **text** side.
- `tokenize_utils.resolve_pretrained` maps `Qwen/Qwen2.5-Coder-0.5B` to `cache_dir/hub/models--Qwen--.../snapshots/<hash>` for `local_files_only`.

## Experiments

| Id | Question | Config | Implemented? |
|---|---|---|---|
| E0 | Zero-shot floor | `configs/tzag_e0.yaml` `--eval-only` | yes |
| E1 | LM fine-tune only | `configs/tzag_e1_stopgrad.yaml` (`view.weight: 0`) | yes |
| E2 | Flat LLM-JEPA (approach+intent, those tokens see all code) | `configs/tzag_e2_stopgrad.yaml` | yes |
| E3 | Chain loss, whole levels as views | — | **no** (different packing) |
| E4 | Hierarchical LLM-JEPA (all levels, tree mask, SIGReg) | `configs/tzag_e4_stopgrad.yaml` | yes |
| E5 | E4 + masked-node prediction | — | no |
| E6 | E4 + cross-solution | — | no |
| B1 | Frozen text targets, InfoNCE, no LM | — | no |

**First scientific comparison is E1 vs E2 vs E4**, plus E0 as floor. Without E1, view-loss drops are uninterpretable. Without E2, you cannot tell hierarchy from extra text. E5/E6/pooling/free-nodes/1.5B come after that triangle is real.

Paper recipe is **4 epochs**, 3 seeds, equal-step and equal-FLOP vs E1.

**Current recipe (2026-11-08):** `view.stop_grad_target: true`, `sigreg.side: code`, `sigreg.bank_size: 256`, `sigreg.weight: 0.3`, `sigreg.num_proj: 1024`, run in both directions. Configs: `configs/tzag_e{2,4}_{sgcode,c2t}.yaml`; E1 baseline stays `runs/stopgrad/e1`. Launch: `bash scripts/tzag_sweep.sh --spec configs/sgfix_jobs.json`. Runs: `/mnt/storage/users/glifshitz/code-jepa/runs/sgfix/`. Untested until those finish.

**Discarded results.** Do not use these as evidence for the method:
- `runs/e0_e4/` (2026-11-04): 1 epoch, no stop-grad; program-level rank collapsed by step 400.
- `runs/sweep/` (2026-11-05): the `bank256_stopgrad` "winner" had SIGReg on the detached target, so its gain came from stop-grad alone. Bank and μ were no-ops.
- `runs/stopgrad/e2`, `e4` (2026-11-06): same inert SIGReg. Eval SIGReg rose (E4 8 → 18). E4 approach/intent effective rank fell from 34 to 8. Raw cosine saturated at 0.99 because of a shared offset direction. Local copy: `runs/stopgrad_tzag/`. Not worth re-evaluating.

Go/no-go: E2 beats E1 on approach/intent cosine **without** rank collapse; E4 beats E2 on **step/unit** cosine especially on multi-unit programs; LM stays near E1.

`eval_confusion_acc` hit 1.0 at step 0 because own-node pairs already win the argmax. Do not use it until fixed (exclude the matched pair, or score level-vs-level without the diagonal).

Live monitors:
- **`eval_cos_centered_*`**, not raw `eval_cos_*`. Raw cosine is dominated by a shared mean direction: at step 0 approach is 0.43 raw vs 0.006 centred. E1 never trains its heads, so raw-cosine wins over E1 are meaningless.
- **`eval_erank_code_*` / `eval_erank_text_*`**. Plain `eval_erank_*` follows the view target, which switches side with `view.direction`.
- **`*_multi` / `*_single`** splits of `n`, `cos` and `cos_centered`. These use the unit count from the full tree (`batch["n_units"]`), so E2 is split correctly too. Val is 31 multi-unit / 67 single-unit programs (98 after drops).
- Eval sets differ: E2 scores 101 val programs, E1/E4 score 98 (the 2048-token limit drops 3).

## tzag

Host: `132.66.150.42`. User storage: `/mnt/storage/users/glifshitz/code-jepa/{repo,env,models/hf,runs,cache,tmp}`.

```bash
# laptop
bash scripts/sync_tzag.sh

# tzag
source /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_env.sh
# smoke
bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_train.sh --max-samples 64 --max-steps 20
# E0–E4 original 1-epoch wave (collapsed; keep for comparison)
bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_run_e0_e4.sh
# E2/E4 x {text->code, code->text}, stop-grad + SIGReg on code (queues on idle GPUs)
nohup bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_sweep.sh --spec configs/sgfix_jobs.json \
  > /mnt/storage/users/glifshitz/code-jepa/runs/sgfix/driver.log 2>&1 < /dev/null &
```

Qwen 0.5B and 1.5B are already in `models/hf`. `HF_HUB_OFFLINE=1`. Do not download.

Run locations:

- `/mnt/storage/users/glifshitz/code-jepa/runs/e0_e4/` — 1-epoch collapse wave (2026-11-04)
- `/mnt/storage/users/glifshitz/code-jepa/runs/sweep/` — hyperparameter sweep (2026-11-05; SIGReg inert, see above)
- `/mnt/storage/users/glifshitz/code-jepa/runs/stopgrad/` — 4-epoch E1/E2/E4 (2026-11-06). Only `e1` is still a valid baseline
- `/mnt/storage/users/glifshitz/code-jepa/runs/sgfix/` — **live** E2/E4 × {sgcode, c2t} (launched 2026-11-08, queued; `driver.log`, `driver.pid`)

Other users often hold GPUs 1–3. The driver only takes a GPU with ≥ 40 GB free and no foreign compute process.

**GPU memory (0.5B, batch 4, eager, grad checkpoint, bf16):** roughly 8–16 GB typical, observed ~20–28 GB on L40S during E1/E4. 1.5B later: ~16–24 GB. GTX 1650 will OOM. Dataset packing is **CPU RAM** (dense `T×T` masks; CodeContests train is on the order of 10 GB per process).

## Local commands

```bash
conda activate code_jepa
pip install -e ".[dev]"
pytest -q
python scripts/train.py --config configs/default.yaml --data tests/fixtures/sample.jsonl --max-steps 2 --cpu --tiny
```

## Not implemented yet (do not pretend they are)

- E3 chain, E5 masked nodes, E6 cross-solution, pooling heads, free node tokens, HWM / residual / stagewise controls
- Cluster re-split across CC/TACO
- pass@1 harness (collaborator `executor.py`)
- Retrieval/probes/CKA beyond `metrics.py`
- DDP / multi-GPU data parallel (multi-GPU = **one experiment per GPU**)
- 1.5B and mixed-source training

## Working agreements

- Prefer editing existing packer/model/train rather than a second trainer.
- Keep remote disk use on `/mnt/storage/users/glifshitz/code-jepa` and the initzan JSONL.
- After substantive Python changes: `pytest -q`, then `bash scripts/sync_tzag.sh` before a tzag run.
- Check `nvidia-smi` before launching; never kill other users' GPU processes.
- New experiment configs should be explicit YAML (what is on, what is off), not flags scattered in a shell one-liner.
