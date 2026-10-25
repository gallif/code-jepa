# Phase 2 kickoff discussion — 2026-10-25

Context: phase 2 starts from [code_jepa_research_plan.md](code_jepa_research_plan.md).
Phase 1 (Program JEPA) code is retired to the git-ignored `legacy/` folder and
preserved at git tag `phase1-final`.
The collaborator's data example is in [`../data_sample.txt`](../data_sample.txt).

Status: open discussion. Nothing implemented yet. The plan document has not been
updated with these conclusions.

---

## 1. Review of the research plan (before seeing data)

The plan is sound. The main risk is the one it names itself: nothing in the
current design forces the levels to be distinct.

1. **Span-level vs whole-function descriptions is the deciding question.** If every
   level describes the whole function, all higher levels are pulled toward the
   same global meaning, and multiple latents per level have no reason to
   specialise. Span-grounded descriptions fix this. (Resolved by the data sample;
   see section 2.)
2. **Matching latents to descriptions is unspecified.** With K latents and M
   descriptions per level, we need one of:
   - Hungarian or optimal-transport matching.
   - Best-matching-latent (max-sim) scoring.
   - Span-restricted pooling, where each latent is tied to one span.
3. **Make the levels compete in the loss.** Generic text encoders embed
   paraphrases at different levels close together. Use the same function's
   other-level descriptions as hard negatives. Track a level-by-level retrieval
   confusion matrix, or CKA between levels, as a level-distinctness metric.
4. **Frozen vs trainable text encoder.** A frozen encoder gives fixed latent
   targets and makes the main objective JEPA-like (text model as target
   encoder, code model + head as predictor). A trainable one is CLIP-style
   contrastive. This decides how "JEPA" the core method is; currently JEPA is
   only the last optional experiment.
5. **The code encoder and Experiment 1 are underspecified.** Start from a
   pretrained code encoder (e.g. UniXcoder, CodeT5+ embedding) or train from
   scratch? What objective does the flat baseline use? Leaning towards
   fine-tuning a pretrained encoder.
6. **Experiment 3 vs 4 has a confound.** Multi-level supervision also means more
   text and gradient per sample. Add two controls:
   - Experiment 3 with the same number of descriptions, all at function level.
   - The same descriptions duplicated at every level.
7. **Evaluation:** in-distribution retrieval against the same LLM's descriptions
   rewards learning its writing style. Add an external benchmark (CodeSearchNet,
   CoSQA, AdvTest). Per-level evaluation needs span-level test labels.
8. **Data scope and cost:** the plan doesn't name a language, corpus, size or LLM
   budget. Suggested start: Python, ~50k–200k samples, plus a quality-check
   pass on a sample.

## 2. Analysis of the collaborator's data sample

The sample is a competitive-programming solution (stdio input, multiple test
cases per input, 20 tests), most likely from CodeContests or TACO.

### Level mapping

The collaborator's numbering is reversed relative to the plan (his L0 =
coarsest; the plan's Level 1 = tokens).

| Data level | Covers | Count in sample | Plan level |
|---|---|---|---|
| L3 blocks | line ranges inside a unit | 8 + 2 | Level 2 (fine) |
| L2 units | functions and `<main>`, with line ranges | 2 | Level 3 (intermediate) |
| L1 approach | whole program: how it works (+ data structures, complexity) | 1 | Level 4 (global) |
| L0 intent | whole program: what it does | 1 | Level 4 (global) |
| tokens | — | N | Level 1 |

### Implications

1. **Use variable-count, span-restricted pooling instead of fixed latent counts.**
   The plan's illustrative 16 → 4 → 1 latents don't fit; block and unit counts
   vary per program. Proposed main design:
   - Block vectors: a learned attention query that can only see the tokens in
     that block's lines.
   - Unit vectors: pool over their blocks.
   - Global vector: pool over the units.

   The pooling is still learned, so this is consistent with "learned higher
   levels". It also removes the matching problem. **Ablation:** free latent
   queries without span grounding — does a hierarchy emerge on its own?
2. **Blocks are a flat partition, not a nested tree.** Within a unit, blocks are
   contiguous, non-overlapping line ranges covering the body. Nested statements
   are merged (e.g. lines 13–15). Every token maps to exactly one block. The
   only uncovered line is the signature (line 1); assign it to the unit.
3. **L0 and L1 are two views of the same span, not two scales.** Model them as
   two heads on the global vector. Keep them as separate targets and use each
   as a hard negative for the other; this is the cleanest "what vs how"
   distinctness test.
4. **L0 contains information that is not in the code.** "Jumps", "problems" and
   "tags" come from the problem statement; the code only has `A`, `X`, `Y`.
   Expect an error floor at L0. Don't read low L0 scores as hierarchy failure.
5. **Competitive-programming consequences:**
   - **Split by problem, not by solution**, to avoid leakage.
   - If several solutions share a problem, they share L0. Mask same-problem
     pairs as false negatives in contrastive batches. Those pairs also give free
     positives for solution-to-solution retrieval evaluation.
6. **Extra signal the plan doesn't use yet:**
   - The structured L1 fields (data structures, complexity) can be probe labels
     for plan section 7.3.
   - The stdio tests enable functional-behaviour checks, and possibly a later
     JEPA target based on execution rather than text.
7. **Quality notes:**
   - L3 descriptions are mostly close paraphrases of the code, some trivial
     ("Determine the length of the input array"). Fine-level alignment may be
     easy and uninformative.
   - Block u0.b4 is slightly inaccurate: "more than half the time plus one" vs
     `max(X) > (N+1)//2`.

## 3. Open decisions

| Decision | Current leaning |
|---|---|
| Span-level vs whole-function descriptions | **Resolved:** span-level (provided by the data) |
| Pooling design | Span-restricted learned pooling; free latent queries as ablation |
| L0 vs L1 handling | Two heads on the global level, hard negatives for each other |
| Frozen vs trainable text encoder | Open |
| Pretrained vs from-scratch code encoder | Leaning towards pretrained |
| Corpus and scale | Open; pending collaborator |
| Split | By problem |

## 4. Questions for the collaborator

- What is the source dataset? How many problems, and how many solutions per
  problem?
- Is L0 written from the problem statement?
- Is it Python only?
- How are classes, methods and nested functions handled at L2?
- Do blocks always partition the unit body (contiguous, non-overlapping, full
  coverage)?
- What is the delivery format? JSONL with line spans per unit and block would be
  ideal.

## 5. Next steps

1. Get answers to section 4.
2. Update `code_jepa_research_plan.md` with:
   - The level mapping.
   - Span-restricted pooling as the main design, with free latent queries as
     the ablation.
   - Splitting by problem and masking same-problem negatives.
   - A data schema section.
   - The extra controls for Experiment 3 vs 4.
3. Decide on the text encoder and code encoder.
4. Write a loader and validator for the data format (span coverage, line →
   token mapping).
