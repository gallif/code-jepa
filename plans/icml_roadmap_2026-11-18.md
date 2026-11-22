# Road to ICML: hypotheses, experiments and benchmarking

Written 2026-11-18. Read with `plans/next_steps_2026-11-15.md`, which holds the framing and the detailed
setup of runs R1–R5. This file adds the hypotheses behind them, the benchmarking upgrades a top-tier review
will expect, the ablations, a schedule and the paper outline. `AGENTS.md` constraints still apply.

**Deadline.** ICML 2027 submissions are expected in late January 2027. Confirm the exact date on the ICML
site; the schedule in §6 assumes about ten weeks from today.

## 1. The paper in one paragraph

A code LM trained with a span-grounded, multi-level JEPA objective gives, in one forward pass, a vector for
every block, function and whole program that is aligned with a description of that span. We show that these
vectors (a) locate the code a description refers to better than base-model readouts and dedicated embedders,
(b) can replace the code they stand for in the LM's context at one token each, better than
reconstruction-trained vectors at the same budget, and (c) cost the language model nothing beyond ordinary
fine-tuning. We do not claim a better code generator or a general-purpose retriever.

## 2. Hypotheses

Each has a status from `docs/BENCHMARKS_2026-11-15.md` and `docs/RELEASE_2026-11-13.md`, the experiment that
tests it, and what would refute it.

| # | Hypothesis | Status today | Tested by | Refuted if |
|---|---|---|---|---|
| H1 | **Span grounding.** Aligning each node with its own description gives vectors that locate described code better than any pooled readout of the base model, the LM-only fine-tune, or a dedicated embedder. | Partly supported: .86 vs ≤ .70 against E0/E1 only. The shuffled-description and program-grounded controls support the mechanism. | R1, R2 | A per-span embedder baseline matches us, or the margin vanishes on independently written queries. |
| H2 | **Objective.** The JEPA objective (cosine + SIGReg, no negatives) is at least as good as InfoNCE at equal budget on localisation and compression, and drifts the LM less. | Untested. Phase 2b's InfoNCE run learned steps only, but was not tuned. | R5 | A tuned InfoNCE model is better on both and drifts no more. Then the paper is about span grounding, not JEPA. |
| H3 | **Meaning over surface.** With rename augmentation on the code view, vectors become stable under renaming while still moving under a behaviour-changing edit. | Currently false: a renamed correct copy beats a one-token mutant 3% of the time. | R3b | Augmentation does not move the rename and anonymised-AdvTest numbers, or it destroys pointing. |
| H4 | **Compression.** A JEPA node vector is a better one-token stand-in for its code than a reconstruction-trained vector or an LM-only vector at the same token budget. | Untested. | R4 | It does not close a clear share of the none → full-text gap, or reconstruction vectors match it. |
| H5 | **Zoom.** Mixing levels (program → function → block → tokens, expanding only where needed) gives a better quality-per-token curve than any single level. | Untested. This is where the multi-level design has to earn its place, since `tree` composition did not. | E6 below | A single level, or plain truncation, is on the same curve. |
| H6 | **No LM cost.** The alignment objective adds no drift beyond the LM-only fine-tune. | Supported on one seed (KL lower than E1 on 4 of 5 corpora for `free`, 5 of 5 for `tree`). | R3a, and R3c with a matched E1 | The gap reverses across seeds or on mixed data. |
| H7 | **Transfer and scale.** With the full ~100k programs (and later the 1.5B backbone), the gains hold on function-level and real-repository code. | Currently false for real repos: SWE-bench Lite .15 for every model. | R3c, E7 | Pointing stays in-domain only. Then the paper must scope itself to single-file programs. |
| H8 | **Composition.** Building the program vector from function vectors helps. | Refuted: `free` beats `tree` everywhere and `tree` fails on TACO (.07). | Reported as a negative result | — |

H1 and H4 are the paper. H2 decides whether it is a JEPA paper. H3 and H7 decide how wide the claims can be.

## 3. Experiments

R1–R5 are specified in `plans/next_steps_2026-11-15.md` §3. Summary and additions:

| Id | Experiment | Tests | Trains a model |
|---|---|---|---|
| R1 | Strong baselines on pointing and same-problem retrieval | H1 | no |
| R2 | Independent queries: a different LLM, human docstrings, larger candidate pools | H1 | no |
| R3a | Seeds 23 and 37 through the full benchmark | H6, all error bars | no |
| R3b | Rename augmentation on the code view | H3 | yes |
| R3c | Train on CodeContests + TACO + KodCode, with held-out evaluation clusters and a matched E1 | H7, H6 | yes |
| R4 | Compression probe: the LM reads node vectors in place of code | H4 | yes, small |
| R5 | InfoNCE at equal budget, LM loss kept | H2 | yes |
| **E6** | **Zoom curve.** With the R4 reader, sweep the context budget and compare: program vector only; unit vectors; step vectors; mixed levels with the target's neighbours expanded; truncation; text descriptions. | H5 | no (reuses R4) |
| **E7** | **Repository-level completion** (CrossCodeEval, RepoBench, Python). Cross-file functions are retrieved by their unit vector and given to the LM as vectors. Baselines: no cross-file context, BM25 + text, embedder + text, at matched token budgets. Run only if R4 passes. | H4, H7 | fine-tune of the R4 reader |
| **E8** | **Backbone scale.** Repeat the final recipe on Qwen2.5-Coder-1.5B, one seed. Run last. | H7 | yes |

### Ablations (on `free`, final recipe, one seed unless the effect is small)

| Ablation | Question | State |
|---|---|---|
| Shuffled descriptions | Do gains come from correct pairing? | Done (`data.shuffle_descriptions`); rerun on the final recipe |
| Program-level grounding of the same texts | Does span grounding create the step and unit vectors? | Done (`hierarchy.ground: program`); rerun |
| Levels: program only (E2) / + units / + steps | What does each level add to pointing and compression? | E2 exists; others new |
| `free` vs `tree` program tokens | Composition (H8) | Done |
| SIGReg on / off / VICReg | Is SIGReg needed, or any anti-collapse term? | Partly, from Phase 2 diagnostics |
| Readout: `raw` vs `head`, node token vs span pooling | Is the node token needed, or would pooling do? | Partly (probes) |
| Data size: 11k / 33k / 100k | Scaling curve | New, part of R3c |
| Annotator: a second LLM writes the training descriptions for a subset | Sensitivity to the teacher | New, optional |

## 4. Better benchmarking

What the 2026-11-15 report lacks for a top-tier submission.

### 4.1 Statistics

- Three seeds for every trained model. Report mean and spread.
- Bootstrap 95% intervals over programs (grouped by problem) for every retrieval number.
- Report single-function and multi-function programs separately. 63–74% of programs have one function, where
  unit and program cover the same lines.

### 4.2 Baselines

- **Embedders:** gte-modernbert and UniXcoder (have), plus a size-matched LLM embedder (for example
  Qwen3-Embedding-0.6B) and at least one code-specific embedder (CodeSage, CodeRankEmbed, jina-code). Models
  that need `trust_remote_code` should be run at a pinned, reviewed revision inside the sandbox, not skipped.
- **Per-span protocols for every embedder:** encode each span alone, and encode once then pool over the span.
- **Lexical:** BM25 everywhere.
- **Objective:** InfoNCE (R5) and the LM-only fine-tune (E1), each on the same data as the model it is
  compared with.
- **Compression:** truncation, text descriptions, a reconstruction-trained compressor in the ICAE style, and
  a hard prompt compressor (LLMLingua-2).

### 4.3 Tasks

| Purpose | Add | Note |
|---|---|---|
| Pointing, independent of our annotator | R2 sets; candidate pools of 1, 10 and 100 programs | Main table |
| Pointing on real code | SWE-bench Lite and Verified, function-level localisation | Limitation today; rerun after R3c |
| Same-problem retrieval on a standard set | POJ-104 or a Python CodeNet subset (clone detection) | Checks that TACO's shared input parsing is not the signal |
| Text → code retrieval on standard sets | CodeSearchNet Python, CoSQA | We expect to lose to embedders. Report it; it bounds the claim. |
| Compression | In-domain function completion (R4), then CrossCodeEval and RepoBench (E7) | Headline figure: quality against context tokens |
| Robustness | Rename and reformat against one-token mutant (have); anonymised AdvTest (have) | Rerun after R3b |
| No harm to the LM | Perplexity and KL (have); HumanEval+, MBPP+ (have); LiveCodeBench run inside the sandbox | Supporting table only |

### 4.4 Hygiene

- Once TACO and KodCode enter training, the current "unseen" pools are no longer unseen. Hold evaluation
  problems out by cluster before any R3c run, and record the split file in the repo.
- Add a near-duplicate check (token-level similarity) between every training program and every evaluation
  program, and report how many were removed.
- The base model may have seen public benchmarks in pre-training. We cannot control this; say so, and rely on
  comparisons between models that share the backbone.
- Fix each pass rule before the run and write it in the plan. Commit results JSON and the exact command for
  every table.
- Report cost: context tokens, encoder passes and wall-clock per query, next to quality.

## 5. Gates

| Gate | When | Condition | If it fails |
|---|---|---|---|
| G1 | end of week 2 | H1 holds under R1 and R2 | Drop the localisation claim; the paper depends on H4 alone |
| G2 | end of week 4 | H4 holds in-domain (R4) | Localisation-only paper; skip E6 and E7 |
| G3 | end of week 6 | H2: JEPA is not worse than InfoNCE | Reframe around span grounding with the objective as a design choice |
| G4 | end of week 7 | E7 shows a gain at matched budget | Keep compression as an in-domain result and state the limit |

If G1 and G2 both fail, stop and rethink before spending more GPU time.

## 6. Schedule (ten weeks, 4× L40S on tzag)

| Week | Work |
|---|---|
| 1 | R1, R3a (seeds), bootstrap intervals, extra embedder baselines. Literature check (see §8). |
| 2 | R2 query sets. Cluster re-split and held-out pools for R3c. Implement rename augmentation. **G1.** |
| 3 | Train R3b and R3c arms with matched E1. Build the R4 reader and the reconstruction baseline. |
| 4 | R4 results. Rerun the benchmark on R3 models. Pick the final recipe. **G2.** |
| 5 | R5 InfoNCE, tuned. Three seeds of the final recipe. E6 zoom curve. |
| 6 | Ablations. Standard-benchmark rows (CodeSearchNet, clone detection, SWE-bench). **G3.** |
| 7 | E7 repository-level completion. **G4.** |
| 8 | E8 on 1.5B if time allows. Fill every table with three seeds. Start writing. |
| 9 | Full draft, figures, limitations. Internal review. |
| 10 | Revisions, reproducibility package, submission. |

Training runs need the owner's go-ahead (`AGENTS.md`).

## 7. Paper outline

1. **Introduction.** One pass, a vector per span, two uses: find the code, and stand in for the code.
2. **Related work.** JEPA and LLM-JEPA; H-JEPA; code embedders and late-interaction retrieval; context
   compression (gist tokens, ICAE, xRAG).
3. **Data.** The description tree, with spans from tree-sitter and text from an LLM.
4. **Method.** Node tokens, attention mask, per-level view loss, SIGReg, LM anchor. `free` is the main model.
5. **Experiments.**
   - Pointing (H1), with independent queries and embedder baselines.
   - Compression and the zoom curve (H4, H5), then repository-level completion.
   - JEPA against InfoNCE and LM-only (H2, H6).
   - Robustness (H3) and transfer (H7).
   - Ablations.
6. **Limitations.** Whatever of H3 and H7 remains false; composition (H8) as a negative result; Python only;
   LLM-written supervision; non-commercial data licence (KodCode).

Changes needed in `paper/draft.tex`: the planned contributions (line 85) still promise a hierarchical
composition and generation pass@1. Replace them with the three claims in §1 above, and rename the method
"span-grounded multi-level" throughout.

Main figures: (1) method diagram; (2) pointing bars with baselines; (3) quality against context tokens;
(4) JEPA / InfoNCE / LM-only on pointing, compression and KL.

## 8. Risks

| Risk | Mitigation |
|---|---|
| A per-span embedder matches us on pointing | R1 runs first; G1 |
| Prior work already covers JEPA vectors as compressed context, or one-pass multi-granularity code vectors | Literature check in week 1, before R4 is built |
| InfoNCE wins | G3 reframing; the span-grounding contribution stands on the existing controls |
| Rename augmentation hurts pointing | Sweep the augmentation rate; report the trade-off |
| Compression works only in-domain | State the scope; E7 becomes a limitation |
| Too few multi-function programs for R4 and E6 | Use KodCode and TACO multi-function subsets; report counts |
| GPU time | R1, R2, R3a and E6 need no training; E8 is optional |
