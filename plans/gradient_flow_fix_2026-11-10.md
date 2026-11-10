# Gradient-flow fix (2026-11-10)

This file records why every JEPA run before 2026-11-10 is discarded, what was changed, and how the recipe is
being chosen. The research plan is [code_jepa_research_plan.md](code_jepa_research_plan.md).

## Decision

- **The code-side node embeddings \(c_v\) are the trainee.** They always get view-loss gradient. This is enforced:
  - `resolve_stop_grad` raises on any setting that detaches code.
  - `tests/test_model.py::test_view_loss_always_trains_code_side` checks it.
- **Stop-grad is an experimental variable.** The allowed values are `view.stop_grad: none | text`. It stays only
  if the diagnostic below shows it is needed; otherwise it is removed.

## What went wrong

| Runs | View gradient into \(c_v\) | SIGReg gradient into \(c_v\) | Outcome |
|---|---|---|---|
| `e0_e4` (11-04, no sg) | yes | per-batch, ~4 vectors for intent/approach, μ=0.1 | program rank collapsed by step 400 |
| `sweep`, `stopgrad` (11-05/06) | **no** (sg on code) | **none** (on the detached target) | discarded |
| `sgfix/*_sgcode` (11-08) | **no** (sg on code) | yes, diluted ~65× | rank held; \(c_v\) trained only by SIGReg and backbone drift |
| `sgfix/*_c2t` (11-08) | yes | yes, diluted ~65× | collapsed: code erank ≈ 1–2, raw cos 0.9999, std ≈ 177 |

1. **The misplaced stop-grad.** `stop_grad_target` detached whatever the view target was. With
   `direction: text_to_code`, the target is code, so the node tokens got no view gradient. The old test suite even
   asserted this behaviour.
2. **The SIGReg bank diluted its own gradient.** `EmbeddingBank` returned `sigreg(pooled) * n_current / n_pooled`.
   SIGReg's statistic is already `N * err`, so the gradient on current samples was already about that of
   batch-only SIGReg, and the factor cut it ~65× for intent/approach (4 of 260) and ~10–15× for steps.
   - Effective μ for program-level nodes was ≈ 0.005.
   - In E2 c2t, final `eval_sigreg` was 5.09 (against 0.30 for sgcode). SIGReg measured the collapse but could
     not push back.
3. **The draft's actual recipe had never been tested:** no stop-grad, with a working SIGReg.

Secondary issues:
- The projection heads are shared by code and text, so a detached target still drifts.
- In `c2t`, `p` has no predictor token.
- `confusion_acc` read 1.0 at step 0.

## Phase 1 — code changes (done 2026-11-10; `pytest -q`: 22 passed)

- **`config.py`**
  - `view.stop_grad: none | text`.
  - `resolve_stop_grad()`: legacy `stop_grad_target: true` maps to `text` under `code_to_text` and raises under
    `text_to_code`.
  - `projection_head.shared`.
- **`losses.py`**: the bank term is no longer rescaled.
- **`model.py`**
  - `_orient` only ever detaches text.
  - Optional `text_heads`.
  - `compute_losses(grad_diag=True)` reports `gn_{view,sigreg}_{code,text}_{level}`, the mean per-node gradient
    norm of each weighted term with respect to the projected embeddings.
- **`metrics.py`**: `level_sep_acc` (+ `_chance`, `_multi_unit`, `_single_unit`) replaces `confusion_acc`.
  - Query = centred code embedding of node v. Candidates = v's own description plus the same program's other-level
    descriptions.
  - A hit means its own description is strictly highest.
- **`train.py`**: logs `grad_diag` every `log_every` steps.
- **`tzag_sweep_driver.py`**: `--gpus` allowlist.
- **Tests**
  - The code side gets view gradient in every allowed config.
  - stop-grad on code is rejected.
  - The bank doesn't dilute the gradient.
  - Separate heads work.
  - The grad-diag contract holds.
  - level_sep ignores the shared level offset.
- **Configs**
  - New `tzag_e4_diag_{a1,a2,b1,b2}.yaml` and `diag_jobs.json`.
  - `tzag_e{2,4}_{sgcode,stopgrad}.yaml` are marked RETIRED; they raise.

Smoke test (B1 config, real 0.5B, 30 steps):
- `gn_view_text_* = 0` and `gn_view_code_* > 0`.
- SIGReg and view gradients on code are both ~1–2e-3.

Still open: per-term gradient norms on the backbone and head *parameters*.

## Phase 2 — diagnostic sweep (running since 2026-11-10 13:50, tzag GPUs 0,1, `runs/diag/`)

E4, 0.5B, seed 82, 2000 steps. SIGReg on code and text, bank 256, num_proj 1024. Otherwise as `tzag_e4_c2t`.

| Id | `view.stop_grad` | μ |
|---|---|---|
| A1 / A2 | none (the draft's recipe) | 0.3 / 1.0 |
| B1 / B2 | text | 0.3 / 1.0 |

**Pass rule (fixed before launch; `scripts/diag_summary.py`).** At step 2000:
- code and text erank at **every** level ≥ 50% of step 0
- centred cosine up at every level
- eval LM ≤ 0.8465 (E1 `runs/stopgrad/e1` at step 2000 is 0.8265; plus 0.02)
- SIGReg/view gradient ratio on code, averaged over the last 500 steps, within [0.1, 10] at every level

**Choice:**
- If an A run passes, drop the stop-grad and use the best A setting.
- Use B only if no A passes and a B run does.
- If nothing passes, diagnose before Phase 3 (μ, `projection_head.shared: false`, learning rate on the new
  parameters).

## Phase 2 results (2026-11-10, finished about 14:45)

All four runs finished cleanly. LM stayed under the threshold in every run (0.832–0.844, against a maximum of 0.8465).

| Run | Rule verdict | Centred cos at step 2000 (s / u / a / i) | Code erank (u / a / i) | level_sep (chance 0.165) | What happened |
|---|---|---|---|---|---|
| A1 none, μ 0.3 | FAIL | +0.32 / −0.44 / −0.04 / +0.02 | 2.1 / 28.7 / 32.6 | 0.34 → **0.72** | Strong learning at step level. Approach/intent collapsed to erank 1.1 at steps 1200–1600 (raw cos 0.99), then partly recovered. Unit collapsed and stayed collapsed. |
| B1 text, μ 0.3 | **PASS** (by the letter) | +0.06 / +0.01 / +0.01 / −0.00 | 84 / 32 / 34 | 0.34 → 0.39 | Rank kept everywhere, but **almost no learning**. The view loss *rose* from 0.52 to 0.74. |
| A2 none, μ 1.0 | FAIL | +0.08 / +0.01 / +0.04 / −0.01 | 87 / **1.1 / 1.1** | 0.46 | Approach/intent collapsed by step 800 and stayed collapsed. |
| B2 text, μ 1.0 | FAIL | +0.03 / +0.02 / +0.02 / +0.10 | 88 / 16 / **3.5** | 0.33 | Intent collapsed; little learning. |

Reading:
- **The pass rule was too lax.** "Centred cosine rises" had no minimum effect size, so B1 passes while learning almost nothing. Under the rule as written, B1 would be selected. That isn't a usable recipe, so **Phase 3 is on hold**, which is the rule's own "if nothing usable passes, diagnose" branch. The next round needs a minimum effect: centred cos at step level ≥ +0.10 and level_sep ≥ step 0 + 0.10.
- **Stop-grad on text mostly blocks learning.** The code side gets view gradient, but it is chasing a text embedding that barely trains, and the view loss gets worse.
- **No stop-grad learns, and then collapses at the levels with the fewest vectors per batch.** Approach and intent have 4 per micro-batch; unit has about 4–8 and mostly shares a span with the program level.
  - The collapse signature is a **shared offset direction**: raw cos goes to 0.99 while centred cos stays near 0. Cosine loss can be cut by aligning both sides' means, and with no stop-grad both sides move together.
  - **A higher μ made program-level collapse worse** (A2 compared with A1). So SIGReg as implemented is not what holds rank at those levels. One likely reason: the bank pools 4 fresh vectors with about 252 stale, detached ones, so the gradient on the fresh points is weak or misdirected while the representation drifts quickly.
- **What's left to test**, each a short 2000-step A-type run:
  1. Remove the offset shortcut: subtract the per-level mean (batch or bank) from `p` and `c` inside the view loss.
  2. Give SIGReg more fresh vectors: larger `batch_size`, plus a smaller or zero bank, to test the stale-bank hypothesis.
  3. InfoNCE with the bank as negatives, which penalises the shared offset directly (research plan §6.6).

## Phase 2b — diagnosis round (launched 2026-11-10 at about 15:40)

Each run is A1 (no stop-grad, μ 0.3, SIGReg on both sides, seed 82) with **one** change:

| Id | Change | Hypothesis tested | GPU |
|---|---|---|---|
| C1 | `view.center: ema`: subtract a detached per-level running mean (momentum 0.99) from p and c before the cosine | the collapse is the shared-offset shortcut in the raw cosine | 0 (shared with another job) |
| C2 | `batch_size 8 × accum 1`, `bank_size 32`, 1000 steps (same samples and optimizer steps as A1) | SIGReg fails because the bank is mostly stale vectors | 1 |
| C3 | `distance: infonce`, τ 0.07, in-batch same-level negatives + a bank of 256 detached targets per level | contrastive negatives remove the shortcut directly | 1, after C2 |

**Stricter pass rule (fixed before launch; `scripts/diag_summary.py --strict`).** The Phase 2 checks, plus:
- centred cos at step level ≥ +0.10
- approach and intent centred cos ≥ step 0 + 0.05
- `level_sep_acc` ≥ step 0 + 0.10

A1 under the strict rule fails approach and intent. Caveats:
- With InfoNCE, `view` / `gn_view` values are on a different scale than with cosine (cross-entropy over ~260 keys). The gradient-ratio check is less meaningful for C3.
- Same-problem pairs are not masked as negatives in the bank.

New code (`pytest -q`: 24 passed):
- `view.center`, `view.center_momentum`, `view.temperature`, `view.infonce_bank`
- `losses.TargetBank`, `losses.infonce_loss`
- sweep driver `--min-free-mib`, `--allow-foreign`

### Phase 2b results (all finished by about 16:10)

No run passes the strict rule. **C2 is the only run with no collapse at any level.** Its one failure is intent, which is still rising.

| Run | Strict | LM | Centred cos (s / u / a / i) | Code erank (u / a / i), step 0 ≈ 73 / 34 / 32 | level_sep | Fails |
|---|---|---|---|---|---|---|
| A1 (ref) | FAIL | 0.834 | +0.32 / −0.44 / −0.04 / +0.02 | 2.1 / 28.7 / 32.6 | 0.72 | unit collapse; approach, intent |
| C1 centred cos | FAIL | 0.834 | +0.31 / +0.01 / +0.06 / −0.04 | **8.0 / 2.9 / 1.5** | 0.70 | code rank u/a/i; intent |
| **C2 batch 8, bank 32** | FAIL (intent only) | **0.804** | +0.17 / **+0.11 / +0.07** / +0.02 | **110 / 67 / 71** | 0.62 | intent cc +0.026 < +0.05 |
| C3 InfoNCE | FAIL | 0.841 | **+0.66** / +0.02 / +0.02 / −0.00 | 15 / 23 / 26 | **0.74** | unit rank; approach, intent |

Reading:
- **The shared-offset hypothesis is rejected.** C1 removes the offset from the loss, and the code side collapses at unit/approach/intent anyway. Raw cos at approach/intent still goes to 1.00. Once collapsed, the centred loss sends almost no gradient: the SIGReg/view ratio at approach is 33.
- **The stale-bank / too-few-fresh-vectors hypothesis is supported.** C2 has the same samples and optimizer steps as A1, with 8 fresh vectors per program level and bank 32. Rank *grows* at every level, centred cos rises steadily at all four levels, and LM is the best so far.
  - Caveat: C2 changed two things at once, batch size and bank size, so it doesn't yet separate them. It also changes optimizer noise.
- **InfoNCE learns steps very well** (centred cos 0.66, level_sep 0.74) but not program levels, and unit rank drops to 15. With in-batch negatives, the program levels have only 4 fresh candidates.
- **Intent fails in every run.** Intent carries problem-statement information that isn't in the code, which sets a floor (research plan §2.3). C2's intent was still rising at step 1000.

Not launched; awaiting a decision. Candidate next steps:
1. C2 for longer (about 1 epoch).
2. Separate batch from bank: batch 8 + bank 256, and batch 4 × 2 + bank 32 / 0.
3. C2 + InfoNCE.

## Phase 2c — separate batch size from bank size, and run C2 longer (launched 2026-11-10 at about 16:35, GPUs 0/1)

| Id | Setting | Question | GPU |
|---|---|---|---|
| D1 | C2 for one full epoch (about 1430 steps, batch 8, bank 32) | does intent pass the strict threshold with more training; does C2 stay collapse-free | 0 |
| D2 | C2 with bank 256 (batch 8, 1000 steps) | is a bigger batch enough even with a mostly-stale bank | 1 |
| D3 | A1 with bank 32 (batch 4 × 2, 2000 steps) | is a smaller bank enough at the original batch size | 1, after D2 |

The strict rule is unchanged, with one exception. The LM threshold is now E1's `eval_lm` **at the same number of training samples**, plus 0.02 (computed in `scripts/diag_summary.py`), because D1 sees more data. Earlier runs are unaffected: they all match E1 at step 2000.

### Phase 2c results (all finished by about 17:10)

**Two runs pass the strict rule: D3 and D1.** Both are no-stop-grad runs with **bank 32**.

| Run | Strict | LM (max) | Centred cos (s / u / a / i) | Code erank (u / a / i), step 0 ≈ 73 / 34 / 32 | level_sep (multi) |
|---|---|---|---|---|---|
| C2 (ref): batch 8, bank 32, 1000 steps | FAIL (intent) | 0.804 | +0.17 / +0.11 / +0.07 / +0.02 | 110 / 67 / 71 | 0.62 |
| **D1**: C2 for 1 epoch (1402 steps) | **PASS** | **0.795** (0.837) | +0.20 / +0.16 / +0.09 / +0.05 | **106 / 69 / 71** | 0.67 (0.66) |
| D2: batch 8, **bank 256** | FAIL | 0.803 | +0.12 / +0.13 / +0.00 / +0.01 | 88 / 24 / 30 (approach *text* rank 14) | 0.56 |
| **D3**: batch 4 × 2, **bank 32**, 2000 steps | **PASS** | 0.832 (0.847) | **+0.38 / +0.50 / +0.32 / +0.10** | 63 / **20** / 29 | **0.87 (0.89)** |

Reading:
- **Bank size explains C2, not batch size.**
  - With bank 256, batch 8 still fails at approach/intent (D2).
  - With bank 32, batch 4 × 2 gives the strongest alignment of any run so far at every level (D3).
  - So the large bank of stale, detached vectors was what stopped SIGReg from holding the program levels. (Before 2026-11-10 it was additionally diluted ~65×.)
- **The two passing runs trade off differently.**
  - D3 (batch 4 × 2) aligns much more strongly. Unit centred cos is +0.50, and level_sep is 0.87, with 0.89 on multi-unit programs. But approach code erank is falling: 52 → 37 → 47 → 25 → 20 across evals. It passes the ≥ 50% rule at 58%, but the trend matters for 4 epochs.
  - D1 (batch 8) is slower to align but rank-stable: ranks *grow* at every level. It also has the best LM so far.
- **The stop-grad question is answered**, per the rule fixed before Phase 2: a no-stop-grad recipe passes, so **drop the stop-grad**. Text-side stop-grad (B1/B2) blocked learning.
- Caveats: one seed, val of 98 programs, and at most 2000 steps against the paper's 4 epochs (about 11k steps at batch 4).

Not launched; awaiting a decision on the Phase 3 recipe. Candidates:
- D3 settings (batch 4 × 2, bank 32), with a watch on approach rank
- D1 settings (batch 8, bank 32)
- a short check first: D3 for 1 epoch, to see whether approach rank keeps falling

### D4: D3 settings for one epoch (finished about 18:50)

| Eval step | Centred cos (s / u / a / i) | Code erank (u / a / i) | level_sep |
|---|---|---|---|
| 0 | +0.04 / −0.01 / +0.01 / −0.01 | 73 / 34 / 32 | 0.34 |
| 1600 | +0.33 / +0.40 / +0.23 / +0.06 | 75 / 25 / 41 | 0.84 |
| 2000 | +0.38 / +0.50 / +0.32 / +0.10 | 63 / 20 / 29 | 0.87 |
| 2400 | +0.43 / +0.58 / +0.37 / +0.15 | 54 / 23 / 28 | 0.89 |
| 2804 (final) | **+0.46 / +0.63 / +0.43 / +0.22** | 50 / 18 / 23 | **0.91 (multi 0.94)** |

Strict verdict: **FAIL**, on one check only. Approach *text* erank fell to 20.1, which is 50% of step 0's 40.2, just under the ≥ 50% line. Approach *code* erank has plateaued at about 18–22 since step 1600 (54% of step 0). All the other checks pass, and LM is 0.823 (max 0.837).

**D1 against D4 is a clean pair.** Both cover one epoch, with an effective batch of 8 and about 1,400 optimizer steps. The only difference is the micro-batch that the view loss and SIGReg see:
- **D1**: one micro-batch of 8; 8 fresh vectors + bank 32 per SIGReg call.
- **D4**: two micro-batches of 4; 4 fresh vectors + bank 32.

| Run | LM | Centred cos (s / u / a / i) | Code erank (u / a / i) | Text erank (u / a / i) | level_sep |
|---|---|---|---|---|---|
| D1 (micro-batch 8) | **0.795** | +0.20 / +0.16 / +0.09 / +0.05 | **106 / 69 / 71** | 96 / 57 / 61 | 0.67 |
| D4 (micro-batch 4) | 0.823 | **+0.46 / +0.63 / +0.43 / +0.22** | 50 / 18 / 23 | 45 / 20 / 28 | **0.91** |

So the micro-batch size sets how strongly SIGReg acts relative to the view loss:
- D1 is regularisation-dominated: rank goes *up*, alignment is slow, and LM is the best.
- D4 is alignment-dominated: alignment is about 2–4× higher, and program-level rank is about 55–70% of step 0.

Over 4 epochs (about 4× longer), D4's slow rank decline is a risk, and D1's alignment is only climbing slowly.

Recommendation, following the rule stated before D4 ("4 × 2 if rank holds, else batch 8"):
- **Use batch 8 for Phase 3.** D4 fails the rank check narrowly.
- The obvious tuning point is μ at batch 8. D1's ranks sit well *above* step 0, so a smaller μ (for example 0.1) should buy alignment without collapse.
- A short check: batch 8, bank 32, μ 0.1, one epoch, about 35 minutes. Run it before committing compute to the 4-epoch, 3-seed comparison.

### D5: D1 with μ 0.1 (finished about 20:01)

Strict verdict: **FAIL**. **Approach and intent collapsed by step 600** (code and text erank 1.6, raw cos 0.97). Unit rank fell to 50%.

| Eval step | Centred cos (s / u / a / i) | Code erank (u / a / i) | level_sep |
|---|---|---|---|
| 0 | +0.04 / −0.01 / +0.01 / −0.01 | 73 / 34 / 32 | 0.34 |
| 400 | +0.29 / +0.35 / +0.31 / +0.11 | 72 / 10 / 22 | 0.83 |
| 600 | +0.41 / +0.51 / +0.27 / +0.28 | 60 / **3.4 / 4.1** | 0.88 |
| 1402 | +0.57 / +0.72 / +0.34 / +0.20 | 37 / **1.6 / 1.6** | 0.91 |

The centred cos at approach/intent is not meaningful after collapse: everything lies along about one direction. LM is 0.792.

**One-epoch summary at effective batch 8.** All rows have the same samples and about the same optimizer steps.

| Run | micro-batch | μ | LM | Centred cos (s / u / a / i) | Code erank (u / a / i) | Strict |
|---|---|---|---|---|---|---|
| D1 | 8 | 0.3 | 0.795 | +0.20 / +0.16 / +0.09 / +0.05 | **106 / 69 / 71** | **PASS** |
| D4 | 4 (× 2) | 0.3 | 0.823 | +0.46 / +0.63 / +0.43 / +0.22 | 50 / 18 / 23 | FAIL (approach text rank 50%) |
| D5 | 8 | 0.1 | 0.792 | +0.57 / +0.72 / (+0.34 / +0.20 collapsed) | 37 / **1.6 / 1.6** | FAIL (collapse) |

Reading:
- **The usable window for μ is narrow, and the program levels set it.** At μ 0.1 they collapse within 600 steps. At μ 0.3 they hold, but alignment is slow.
- Steps and units tolerate a lower μ: D5 reaches the best step/unit alignment of any run, with unit rank at 50%.
- That points to **per-level μ** (research plan §6.6 already writes μ_ℓ): a lower μ for step/unit and μ ≥ 0.3 for approach/intent.
- That needs a small code change: `sigreg.weight` per level. **Not implemented yet.**

**Recommendation for Phase 3:**
- **D1's recipe: no stop-grad, batch 8 × 1, bank 32, μ 0.3.** It is the only one-epoch run that passes the strict rule. It is rank-safe, with ranks well above step 0, and it has the best LM.
- Its alignment is slow but still rising. Phase 3 is 4 epochs.
- Optional, before Phase 3: per-level μ (step/unit 0.1, approach/intent 0.3), one epoch, about 35 minutes. Adopt it only if it passes the strict rule.

**Driver fix.** `tzag_sweep_driver.py` crashed after D5 had finished: an empty slot (`None`) was read when another user's process appeared on a GPU. D5 itself completed (`last.pt`, final eval). The guard is fixed; it is uncommitted.

### D6: D1 with per-level μ (step/unit 0.1, approach/intent 0.3), one epoch (finished about 21:00)

| Run | μ (s, u / a, i) | LM | Centred cos (s / u / a / i) | Code erank (u / a / i), step 0 ≈ 73 / 34 / 32 | level_sep (multi) | Strict |
|---|---|---|---|---|---|---|
| D1 | 0.3 / 0.3 | 0.795 | +0.20 / +0.16 / +0.09 / +0.05 | 106 / 69 / 71 | 0.67 (0.66) | **PASS** |
| D5 | 0.1 / 0.1 | 0.792 | +0.57 / +0.72 / collapsed | 37 / 1.6 / 1.6 | 0.91 | FAIL (collapse) |
| **D6** | 0.1 / 0.3 | 0.793 | **+0.50 / +0.63 / +0.12 / +0.03** | 45 / **60 / 66** | **0.86 (0.91)** | FAIL (intent +0.042 < +0.05) |

Reading:
- **Per-level μ does what it was meant to.**
  - Step and unit align almost as well as in D5: 2.5–4× D1.
  - Approach and intent keep their rank as in D1, above step 0.
  - LM is unchanged.
- D6 fails the strict rule **only on intent**: +0.042 against +0.05. D1 passed with +0.058. The difference is within the noise of one seed on 98 programs.
- Program-level alignment is still slow, as in D1. Approach and intent rank sit well above step 0, so there is room for a program-level μ between 0.1 (collapses) and 0.3, for example 0.2.

Recommendation for Phase 3:
- **D6's recipe (no stop-grad, batch 8 × 1, bank 32, μ step/unit 0.1, approach/intent 0.3).** It is better than D1 at step, unit, approach and level_sep, with the same rank safety and LM.
- It narrowly misses the strict intent threshold. Adopting it means overriding the rule within noise, which is the user's call.
- The conservative alternative is D1, the only strict pass.

## Phase 3 — main comparison

**Decision (2026-11-11): Phase 3 uses the D6 recipe.** No stop-grad, batch 8 × 1, SIGReg bank 32, `level_weights` step/unit 0.1, approach/intent 0.3. This knowingly accepts D6's 0.008 miss on the strict intent threshold, which is within noise.

Configs: `configs/tzag_p3_{e1,e2,e4}_s{82,23,37}.yaml` and `configs/p3_jobs.json`, ordered seed by seed (E4, E2, E1).
- E1 has view weight 0, SIGReg 0 and no `level_weights`, with the same packing and batch.
- E2 uses `node_levels` / `view.levels` [approach, intent].
- Runs go to `runs/p3/`, on GPU 0 only for now: `tzag_sweep.sh --spec configs/p3_jobs.json --gpus 0`.
- About 2 hours per run, so roughly 18 hours sequentially.


**On hold** until a recipe passes a stricter rule (see Phase 2 results).

- E2 and E4 with the chosen recipe, 4 epochs, seeds 82 / 23 / 37. E1 re-run with the same seeds; E0 eval-only.
- Go/no-go:
  - E2 > E1 on approach/intent centred cosine without rank loss.
  - E4 > E2 on step/unit (multi-unit).
  - LM ≈ E1.
- E1 has untrained heads, so compare against probes or frozen-head retrieval on E1's hidden states.

## Phase 4 — evaluation

- Per-level retrieval R@1/5/10, within a program and across val.
- Shuffled-description and all-at-program-level controls.
- Val is 101 programs: also report `test` (128), or a larger, declared dev split.

## Phase 5 — docs and paper

- `AGENTS.md`, `README.md` and the research plan: updated 2026-11-10.
- `paper/draft.tex` §Losses / §Distance: state the final sg choice and the bank formulation once Phase 2 is decided.
  If A wins, the draft's "no stop-gradient by default" stands.
