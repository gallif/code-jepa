# Program JEPA — Plan

Repo: https://github.com/gallif/code-jepa

## Where we stand (2026-08-13)

| Exp | Question (short) | Status | Blocker |
|-----|------------------|--------|---------|
| **0** Baseline | Are edits predictable in a **frozen** (non-JEPA) latent? | **Not started** | Need `(P_t, a_t, P_{t+1})` commit/PR windows; no `P_act` code yet |
| **1** Representation | Can JEPA mask-training learn a better `E`? | **CPU prep done**; code + data + weights ready | **No free GPU** for first train smoke |
| **2** Dynamics | Same with **JEPA-trained** `E` frozen + `P_act` | Not started | Needs Exp 1 checkpoint + same transition data as Exp 0 |
| **3** Planning | Search edits in latent toward a goal | Deferred | After Exp 2 |

**Next actions (in order):**
1. When a GPU frees: smoke `train_jepa_mask.py --max-steps 20`, then full Exp 1.
2. In parallel (CPU): build Exp 0 transition dataset + `P_act` scaffold (shared with Exp 2).
3. Run Exp 0 (frozen `E`) and Exp 2 (JEPA `E`) on the **same** transitions; compare.

**Do not confuse with historical “Exp 0”** in [`EXPERIMENTS.md`](EXPERIMENTS.md) (old teacher-cache sanity for sig→body).

## Hypothesis

A coding agent need not reason primarily over source-code tokens. It can operate in a **semantic latent program space**, using LMs only at the boundaries (human I/O, tools/OS, final codegen).

**Central question:** Can we learn a compact semantic program state in which software evolution is predictable, so an agent can plan over edits in latent space rather than generating code token-by-token?

JEPA makes the older density claim precise: not “reconstruct source from a small vector,” but

> Can a small latent preserve what is needed to predict behavior, dependencies, and **consequences of edits**?

## V-JEPA 2 analogy

| V-JEPA 2 | Program JEPA |
|----------|----------------|
| Video observation | Repository / program state `P` |
| Visual encoder `E` | Program encoder `E` |
| Latent state | Semantic program state `z = E(P)` |
| Robot action | Code edit `a` |
| Action-conditioned predictor | `P(z_t, a_t) → z_{t+1}` |
| Goal image + MPC | Desired program / task + search over edits |

### Why sig→body was only partially aligned

The first POC (`signature → body embedding`) was JEPA-*like* feature prediction but a weak V-JEPA 2 analogue:

- A signature does **not** determine a body → high-entropy `P(body | sig)`.
- There is no **action** (the robot’s control input).
- Success as “≤20% tokens vs full body” measured an artificial AST cut, not edit dynamics.

Keep that run as a **historical / negative-control** note, not the main line.

## What `z` should capture (and discard)

Useful for software reasoning: types, control/data flow, deps, side effects, behavior, architecture.

Disposable: formatting, many syntactic variants.

The JEPA predictive objective is what pushes `E` toward that tradeoff.

## Architecture

Follow V-JEPA: **encoder carries semantics; predictor is small and task-specific.**

| Module | Role | POC default |
|--------|------|-------------|
| `E` (online) | Token → per-token features `[T, D]` (no CLS) | Warm-start **Qwen2.5-Coder-0.5B** (or 1.5B), JEPA-finetune |
| `Ē` (target) | EMA of `E` | Same arch; provides prediction targets |
| `P_mask` | Predict masked unit features | ~8-layer Transformer, `d≈512`, ~15–25M |
| `P_act` (later) | Edit-conditioned next-state predictor | New head; **freeze `E`** |

**Not** the main path: pooled retrieval heads (`SigPredictor`) or frozen off-the-shelf embeddings *as the claim*. Those are **baselines**.

### Local features (like V-JEPA tubelets)

V-JEPA 2 emits **one vector per tubelet**, not one global video vector. Analogues:

1. Exp 1: **token** (or statement-span) features.
2. Dynamics: optionally pool to **function/file** tokens so `z` is a short sequence over the touched subgraph.

## Experiment plan

### Exp 0 — Baseline (control, not the claim)

**Question:** Are edits predictable in a *non*-JEPA latent (frozen Qwen / CodeSage)?

Frozen `E_pretrained` + train `P_act` only.  
If this works: some code latents are dynamics-friendly.  
If it fails: does **not** falsify Program JEPA.

**Status:** not started. Blocked on transition data (git/PR file- or function-scoped windows) and `P_act` implementation. CodeSearchNet `full_text` alone is insufficient.

### Exp 1 — Representation learning (runnable path; GPU-blocked)

**Question:** Can we learn `E` with a JEPA-style predictive objective so program state is predictable under masking — i.e. targets come from an EMA encoder trained for prediction, not from an unrelated LM objective alone?

| Item | Choice |
|------|--------|
| Data | Existing CodeSearchNet Python functions (`full_text`); later git trajectories |
| View | Multiblock / span mask over tokens |
| Objective | `P_mask(E(visible), mask_pos) → Ē(masked)` ; SmoothL1 (optional InfoNCE aux on pooled `z`) |
| Pass signal | Held-out mask prediction beats (a) frozen-Qwen targets + student, (b) predict-mean / shuffle controls; features improve a simple semantic probe if available |

**This is the first question we answer.** Embeddings used for the claim must be shaped by the prediction goal.

### Exp 2 — Program dynamics

**Question:** With JEPA-trained `E` frozen, can `P_act(z_t, a_t) → z_{t+1}` beat ablations that ignore the edit or ignore state?

| Item | Choice |
|------|--------|
| Transitions | Git commits / PR hunks (file- or function-scoped windows) |
| Action `a` | Diff/hunk encoding (semantic edit labels later) |
| Pass signal | Next-state latent error ≪ no-action / no-state; error tracks edit magnitude |

### Exp 3 — Planning (later)

Encode current `P` and goal; rank/search edits in `z`; decode selected edits to patches. Only after Exp 2.

## Action spaces (for Exp 2+)

Prefer **real developer edits** (git/PRs/agent trajectories) for natural `(P_t, a_t, P_{t+1})`.

| Space | Note |
|-------|------|
| Token-level edits | Too close to ordinary AR codegen |
| AST ops | Controllable but invented DSL |
| Semantic edits | ExtractFunction, Rename, … — good later coarsening |
| Git commits | Best first dynamics dataset |

## Success framing

| Old POC metric | New |
|----------------|-----|
| Rank@k(body \| sig), TokenRatio ≤ 20% | Masked latent prediction quality (Exp 1) |
| | Edit-conditional next-state prediction vs ablations (Exp 2) |
| | Later: goal-conditioned edit ranking |

Density claim evidence: a **small** `z` that still supports Exp 2 (and probes) implies source is unnecessarily verbose for *reasoning*.

## Implementation status

- [x] Plan written (`PLAN.md`)
- [ ] **Exp 0:** transition dataset + frozen-`E` + `P_act` train/eval
- [x] Exp 1 scaffold: `ProgramEncoder`, `EMATarget`, `JEPAPredictor`, multiblock mask, `configs/jepa_exp1.yaml`
- [x] Exp 1 train entrypoint: `scripts/train_jepa_mask.py` (on `full_text`)
- [x] Exp 1 CPU prep: `scripts/prepare_exp1.py` → `$CODE_JEPA_ROOT/runs/prep_exp1/prep_report.json`
- [ ] Exp 1: first GPU run + refine pass bars / probes *(blocked: no free GPU)*
- [ ] Exp 2: `(P_t, a_t) → z_{t+1}` with JEPA `E` (reuse Exp 0 data)
- [ ] Exp 3: latent edit search (deferred)

### Exp 1 prep snapshot (2026-08-13, CPU)

| Item | Status |
|------|--------|
| Data | train/val/test 22 757 / 4 292 / 2 951; `full_text` present, 0 empty |
| Encoder cache | `Qwen/Qwen2.5-Coder-0.5B` (~494M, d=896, 24L) in `$HF_HOME` |
| Tokenize @512 | sample tok p50≈225, p95≈964; **~15% truncated** at `max_tokens=512` |
| Ready | `ready_for_train: true` — waiting on a free L40S |

Optional later: raise `encoder.max_tokens` (e.g. 1024) if truncation hurts mask-prediction quality.

### Exp 1 quickstart (remote)

```bash
conda activate /mnt/storage/users/glifshitz/code-jepa/env
cd ~/repos/code-jepa && git pull
export HF_HOME=/mnt/storage/users/glifshitz/code-jepa/models/hf
export CODE_JEPA_ROOT=/mnt/storage/users/glifshitz/code-jepa
# CPU prep (safe anytime)
python scripts/prepare_exp1.py --config configs/jepa_exp1.yaml
# when a GPU is free
export CUDA_VISIBLE_DEVICES=<free>   # prefer 2 if idle; leave 3 free unless needed
python scripts/train_jepa_mask.py --config configs/jepa_exp1.yaml --max-steps 20
python scripts/train_jepa_mask.py --config configs/jepa_exp1.yaml
```

## Historical POC (sig→body)

Archived framing: frozen Qwen teacher, `SigPredictor`, InfoNCE, FAISS body retrieval.  
Result (2026-08-11): Rank bars passed; token ladder showed weak iso-quality savings vs leaking one body line — consistent with “no action + artificial split.” See `EXPERIMENTS.md`.
