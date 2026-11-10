# Agent notes — Code-JEPA (phase 2)

This file is for incoming agents. Read it before changing training, data packing, or remote runs. Canonical research writeup is `plans/code_jepa_research_plan.md`; the paper draft is `paper/draft.tex`. Do not treat this file as a substitute for those.

## What we are building

A **span-grounded hierarchical LLM-JEPA** on a collaborator's Python hierarchy JSONL (111.5k programs: CodeContests, TACO, KodCode). Not a reproduction of vision H-JEPA or of NCM generation.

A shared Qwen2.5-Coder encoder sees **code only**. One special token per tree node (step, unit, approach, intent). A 4D tree attention mask lets each node see its span or its children. Each node's description is encoded with a level-specific predictor token. View loss is \(1-\cos(p_v,c_v)\), averaged per level. LM loss is `intent + approach → code`. SIGReg (LeJEPA / H-JEPA) regularises projected node embeddings. SIGReg with a FIFO bank runs on the code and text projections. **The code-node embeddings are the trainee and always get view-loss gradient.** Stop-grad is an experimental variable (`view.stop_grad: none | text`), under test in the 2026-11-10 diagnostic sweep (see below and `plans/gradient_flow_fix_2026-11-10.md`). No EMA.

The hierarchy is **given by tree-sitter spans in the data**, not inferred. Training does not invent node counts.

Phase 1 (single-vector Program JEPA) is frozen at git tag `phase1-final`.

## Hard constraints

- Python 3 only. KodCode is CC-BY-NC: combined data is **non-commercial research only**.
- **Do not copy** the collaborator dataset into this repo or onto the tzag root disk. Read JSONL in place.
- Do not dump HuggingFace / pip / torch caches into `$HOME` on tzag. Use `scripts/tzag_env.sh`.
- Checkpoints, env, and code on tzag live under `/mnt/storage/users/glifshitz/code-jepa`.
- Do not commit `.pt`, `models/`, `runs/`, or JSONL data. `.gitignore` uses **`/data/`** (repo-root only) so `src/code_jepa/data/` stays tracked. Never change that back to unanchored `data/`.
- **Never detach the code side.** `resolve_stop_grad` raises on `view.stop_grad: code` and on legacy `stop_grad_target: true` with `direction: text_to_code`; `tests/test_model.py::test_view_loss_always_trains_code_side` must keep passing.
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
| `src/code_jepa/metrics.py` | Per-level cosine / centred cosine / std / effective rank, `level_sep_acc` (own description vs same-program other levels, centred) |
| `src/code_jepa/tokenize_utils.py` | Resolves `Qwen/...` names to a local hub snapshot under `cache_dir` |
| `src/code_jepa/config.py` | YAML-backed dataclasses |
| `scripts/train.py` | Train + val eval loop, `metrics.jsonl`, `--override key=value` |
| `scripts/validate_data.py` | Pack stats / drop reasons |
| `scripts/sync_tzag.sh` | rsync **code only** to tzag storage |
| `scripts/tzag_env.sh` | Offline HF + all caches on storage |
| `scripts/tzag_train.sh` | Single-GPU smoke (defaults `CUDA_VISIBLE_DEVICES=3`) |
| `scripts/tzag_run_e0_e4.sh` | **Collapsed** 1-epoch E0–E4 wave; keep for comparison |
| `scripts/tzag_run_stopgrad.sh` | 4-epoch E1/E2/E4 with **inert SIGReg** (2026-11-06); superseded |
| `scripts/tzag_sweep.sh` | Queue a jobs spec (`--spec`, default `configs/sweep_jobs.json`) on idle L40S; `--gpus 0,1` restricts it to those GPUs; skips GPUs running other users' processes |
| `scripts/diag_summary.py` | Applies the pre-registered Phase 2 pass rule to `runs/diag/*` |
| `configs/diag_jobs.json` | **Live** (2026-11-10): Phase 2 diagnostics `tzag_e4_diag_{a1,b1,a2,b2}.yaml`, 2000 steps, `stop_grad` none/text × μ 0.3/1.0, SIGReg on both sides |
| `configs/sgfix_jobs.json` | 2026-11-08 queue (E2/E4 × {sgcode, c2t}); finished, superseded |
| `configs/default.yaml` | Local defaults (`stop_grad: none`; local paths, do not use for tzag) |
| `configs/remote_tzag.yaml` | Earlier smoke config |
| `configs/tzag_e{0,1,2,4}.yaml` | 1-epoch collapse wave |
| `configs/tzag_e{1,2,4}_stopgrad.yaml` | 4-epoch configs that ran with SIGReg on the detached target. E1 is still the LM-only baseline; **E2/E4 are RETIRED and now raise** (they detach code) |
| `configs/tzag_e{2,4}_{sgcode,c2t}.yaml` | 2026-11-08. `sgcode` detaches code: **RETIRED, raises**. `c2t` maps to `stop_grad: text` (code predicts detached text) |
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
- `sigreg.side` picks what SIGReg regularises: `code` (default; node-token projections), `text` (predictor tokens), `both`, or `target` (legacy). `code`/`text` are always un-detached, whatever `view.direction` and `view.stop_grad` are. `target` with a detached text target gives SIGReg **no gradient**; the model warns. The key is `side`, not `on`: YAML 1.1 parses `on:` as `true`.
- `view.stop_grad` (`none` default, or `text`) detaches only the text side in the view loss, independent of `view.direction`. Before 2026-11-10, `stop_grad_target` detached whatever the target was; with text→code that was the code side, so node tokens got **no view gradient** (the `stopgrad` / `sgcode` runs). Legacy `stop_grad_target: true` still loads: with `code_to_text` it maps to `text`, with `text_to_code` it raises.
- Per-level heads are shared by code and text by default, so even a detached text target drifts through the heads and backbone. `projection_head.shared: false` gives the text side its own heads.
- SIGReg runs whenever `sigreg.weight > 0`, even with `view.weight: 0` (before 2026-11-08 it was silently skipped).
- Per-batch SIGReg on intent/approach is ~4 vectors (`batch_size=4`). Set `sigreg.bank_size: 256` so the regulariser sees a FIFO of detached embeddings. The bank term is **not** rescaled: SIGReg's statistic is already `N * err`, so the gradient on current samples matches batch-only SIGReg. Until 2026-11-10 it was multiplied by `n_current / n_pooled`, which cut the gradient ~65× for intent/approach (effective μ ≈ 0.005). Every bank run before 2026-11-10 had this.
- `compute_losses(grad_diag=True)` (every `log_every` step in training) logs `gn_{view,sigreg}_{code,text}_{level}`: mean per-node gradient norm of each weighted term w.r.t. the projected embeddings. Use it to check the contract live (`gn_view_code_* > 0`; `gn_view_text_* = 0` under `stop_grad: text`) and the SIGReg/view balance.
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

**Current status (2026-11-10): no recipe chosen yet. Phase 2 finished: no usable recipe (B1 passes the rule but barely learns; A runs collapse at approach/intent/unit). See the Phase 2 results in `plans/gradient_flow_fix_2026-11-10.md`. Phase 3 is on hold. Phase 2b finished: no strict pass. C2 (batch 8 x accum 1, bank 32) is the only run with no collapse at any level (fails intent only); centred loss (C1) did not stop collapse; InfoNCE (C3) learns steps only. Phase 2c finished: D3 (no stop-grad, batch 4x2, **bank 32**) and D1 (batch 8, bank 32, 1 epoch) pass the strict rule; bank 256 fails (D2). Decision so far: **no stop-grad; small SIGReg bank (32)**. Phase 3 recipe (D3 vs D1 settings) not yet chosen. See the plan.** Phase 2 diagnostic sweep (`configs/diag_jobs.json`, runs in `/mnt/storage/users/glifshitz/code-jepa/runs/diag/`): E4, 2000 steps, seed 82, SIGReg on both sides with the bank fix, bank 256, num_proj 1024.

| Id | `view.stop_grad` | μ |
|---|---|---|
| A1 / A2 | none (the draft's recipe) | 0.3 / 1.0 |
| B1 / B2 | text | 0.3 / 1.0 |

Pass rule, fixed before launch (`scripts/diag_summary.py`). At step 2000:
- code and text erank at every level ≥ 50% of step 0
- centred cosine up at every level
- eval LM ≤ 0.8465 (E1 at step 2000 + 0.02)
- SIGReg/view gradient ratio on code within [0.1, 10] at every level

If an A run passes, drop the stop-grad. Use B only if no A passes. If none pass, diagnose before any 4-epoch run. E1 baseline stays `runs/stopgrad/e1`. Full plan: `plans/gradient_flow_fix_2026-11-10.md`.

**Discarded results.** Do not use these as evidence for the method:
- `runs/e0_e4/` (2026-11-04): 1 epoch, no stop-grad; program-level rank collapsed by step 400.
- `runs/sweep/` (2026-11-05): the `bank256_stopgrad` "winner" had SIGReg on the detached target, so its gain came from stop-grad alone. Bank and μ were no-ops.
- `runs/sgfix/` (2026-11-08): `sgcode` detached the code side (no view gradient into node tokens); `c2t` collapsed (code erank ≈ 1–2). Both had the diluted bank SIGReg.
- `runs/stopgrad/e2`, `e4` (2026-11-06): same inert SIGReg. Eval SIGReg rose (E4 8 → 18). E4 approach/intent effective rank fell from 34 to 8. Raw cosine saturated at 0.99 because of a shared offset direction. Local copy: `runs/stopgrad_tzag/`. Not worth re-evaluating.

Go/no-go: E2 beats E1 on approach/intent cosine **without** rank collapse; E4 beats E2 on **step/unit** cosine especially on multi-unit programs; LM stays near E1.

`eval_confusion_acc` (1.0 at step 0, because own-node pairs won the argmax) is **replaced by `eval_level_sep_acc`**. Query = centred code embedding of node v. Candidates = v's own description plus the same program's descriptions at other levels. A hit means its own description scores strictly highest. Read it against `eval_level_sep_chance`. Step 0 on val: 0.34 vs chance 0.16. Splits: `_multi_unit` / `_single_unit`.

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
# Phase 2 diagnostics on GPUs 0,1 (queues A1,B1 then A2,B2)
mkdir -p /mnt/storage/users/glifshitz/code-jepa/runs/diag
nohup bash /mnt/storage/users/glifshitz/code-jepa/repo/scripts/tzag_sweep.sh --spec configs/diag_jobs.json --gpus 0,1 \
  > /mnt/storage/users/glifshitz/code-jepa/runs/diag/driver.log 2>&1 < /dev/null &
# verdicts
python3 /mnt/storage/users/glifshitz/code-jepa/repo/scripts/diag_summary.py
```

Qwen 0.5B and 1.5B are already in `models/hf`. `HF_HUB_OFFLINE=1`. Do not download.

Run locations:

- `/mnt/storage/users/glifshitz/code-jepa/runs/e0_e4/` — 1-epoch collapse wave (2026-11-04)
- `/mnt/storage/users/glifshitz/code-jepa/runs/sweep/` — hyperparameter sweep (2026-11-05; SIGReg inert, see above)
- `/mnt/storage/users/glifshitz/code-jepa/runs/stopgrad/` — 4-epoch E1/E2/E4 (2026-11-06). Only `e1` is still a valid baseline
- `/mnt/storage/users/glifshitz/code-jepa/runs/sgfix/` — E2/E4 × {sgcode, c2t} (2026-11-08, finished). Invalid: sg on code or collapse, diluted SIGReg
- `/mnt/storage/users/glifshitz/code-jepa/runs/diag/` — **live** Phase 2 diagnostics (launched 2026-11-10 on GPUs 0,1; `driver.log`, `driver.pid`)

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
