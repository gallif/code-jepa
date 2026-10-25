# Code-JEPA: Research Plan (phase 2)

Updated 2026-10-25 to reflect the collaborator's hierarchical descriptor dataset
(`/mnt/storage/initzan/hjepa_code/` on tzag, documented in its `DATA_PRINCIPLES.md`).
Discussion that led here: [discussion_2026-10-25.md](discussion_2026-10-25.md).

## 1. Goal

Test whether a code encoder learns **hierarchical latent representations** of source code, from tokens up to
program intent, when supervised by LLM-written descriptions that are grounded in code spans at several levels of
abstraction.

The approach draws on H-JEPA-style hierarchical representation learning and on latent-space prediction (JEPA). It is a
research direction combining these ideas, not a reproduction of either.

## 2. Data

### 2.1 Source and status

| Source | Kind | Tests | Verified solutions | Problems |
|---|---|---|---|---|
| KodCode-V1 (100k subsample) | function-level tasks | pytest | 74,697 | one solution each |
| TACO train | competitive programming | stdio / call-based | 25,200 | ≤ 2 solutions each |
| CodeContests train | competitive programming | stdio | 11,688 | ≤ 2 solutions each |

- Python 3 only. Every solution passes its tests before it is annotated.
- Decontaminated against HumanEval+, MBPP+, TACO test, and CodeContests valid/test.
- Annotator: Qwen3.8-27B, one guided-JSON call per sample, written bottom-up (steps → units → approach → intent).
- **Status (2026-10-25):** CodeContests is annotated (~11.3k records, < 0.1% rejected); TACO and KodCode annotation
  is still running. Pilots of ~750 records cover all three sources.
- KodCode is CC-BY-NC, so the combined dataset is for non-commercial research only.

### 2.2 The hierarchy and how it maps onto the model

The tree's shape (units, steps, line spans) comes from tree-sitter, not the LLM; the LLM only writes the text. Every
description is therefore tied to a real code span.

| Model level | Data field | Text | Span | Per program (CodeContests, median / p90 / max) |
|---|---|---|---|---|
| Tokens | `L4_code` | none (unsupervised) | each token | ~450 characters median |
| Step | `L3` steps, `children` | one line of pseudocode with key conditions/formulas | AST block | 5 / 15 / 48 |
| Unit | `L2_units`, `methods` | responsibility + contract | function / class / method / `main` | 1 / 5 / 32 |
| Program | `L1_approach` | strategy, data structures, complexity | whole program | 1 |
| Program | `L0_intent` | the task; must hold for any correct solution | whole program | 1 |

In this document, levels are named by these terms, not by number. The dataset uses L0 for the coarsest level, while
the previous plan used Level 1 for the finest; names avoid that conflict.

### 2.3 Structural facts we measured

These were checked on the full CodeContests annotation plus the pilot_v2 sets.

1. **Each program is a tree, not a fixed number of slots.** Step and unit counts vary widely, so the model cannot use a
   fixed number of latents per level.
2. **Units nest.** `main` covers module-level code that surrounds the functions, and classes contain methods. Each
   line belongs to its *innermost* unit.
3. **Steps partition a unit's own code.** All lines not covered by a unit's top-level steps are unit headers, blanks,
   comments, docstrings, imports, or lines of a nested unit. We found no uncovered executable code. Headers and
   docstrings are assigned to their unit.
4. **Steps are at most two deep.** Compound statements of 14 lines or more get child steps; 29% of CodeContests programs
   (10% KodCode, 23% TACO) contain nested steps. A parent step summarises its children.
5. **Many units have a single top-level step** (≈ 39% across sources). Here the step and unit share a span and differ
   only in wording (pseudocode vs contract). This is the dataset's main known redundancy.
6. **Description length** (median words): L0 28, L1 strategy 41, unit 36, step 18.
7. **Multiple solutions share a problem.** In CodeContests, 4,953 of 6,338 problems have two solutions. Their L0 texts
   differ in wording (identical in only 5%), so they form natural paraphrase pairs.
8. **Program length.** Median 23 lines; p90 70; cap 200. Up to 1,024 tokens should cover about 98% of programs
   (an estimate from character counts; verify with the chosen tokenizer).

### 2.4 Quality signals available

- `quality.flags`: identifier leaks in L0/L1 (~5% of records), over-long text. These are soft; we keep the records but
  can filter or down-weight them.
- **Reconstruction test** (pilot_v2, 200 samples): the teacher rewrites code from levels L0..Lk only, and the pass
  rate rises as levels are added: 30.5% → 41.5% → 61.0% → 65.0%, and never drops in 90% of samples. By source, L0 alone
  passes in 12% of CodeContests vs 48% of KodCode. Per-record results exist and can be used as sample weights.
- **L0 contains information not in the code** (from the problem statement). Expect an error floor for the
  intent level; low intent scores are not evidence of a failed hierarchy.

### 2.5 Issues to fix before training

1. **Cross-source duplicate problems leak across splits.** 5,034 of 6,365 CodeContests problems also appear in TACO
   under different IDs (both include Codeforces). The collaborator's split hashes `problem_id`, so 215 duplicate groups
   are split across train and val/test. **Fix:** cluster problems by normalised prompt text across all sources, assign
   a cluster ID, and re-split by cluster. Merging also gives up to 4 solutions per problem.
2. **Effective scale is lower than the solution count.** After merging, there are roughly 15k unique competitive
   problems plus ~75k KodCode tasks.
3. **The 1% test split is small** (~1k solutions). Consider 2–3% for the cluster-level re-split so that per-level
   evaluation has enough steps and units.

## 3. Core Principles

- **Code-only input.** The code encoder sees only code tokens. Descriptions are training targets, never inputs.
- **Tokens are unsupervised.** The token level receives no text loss in the initial design.
- **The hierarchy follows the data tree.** Higher-level vectors are learned, but each is tied to a node (step, unit,
  program) and its span. Which node types are supervised is configurable, and so is the depth.
- **Modular objectives.** Text alignment and JEPA-style latent prediction are separate losses, evaluated separately.
- **Level distinctness must be shown, not assumed.** Every experiment reports a level-distinctness metric
  (section 7.2).

## 4. Model Architecture

### 4.1 Main design: shared encoder with span-restricted pooling

```text
Source code
  -> tokenizer -> shared code Transformer -> token states H (N x d)
  -> step vectors: per leaf step, learned attention pooling over the tokens in its lines
  -> parent-step vectors: attention pooling over their child step vectors
  -> unit vectors: attention pooling over their top-level steps (+ header/docstring tokens, + nested-unit vectors)
  -> program vector: attention pooling over the top-level units
  -> heads: step, unit, approach (L1), intent (L0)
```

- **Span-restricted pooling.** Each node's pooling attends only to its own children, so the hierarchy is enforced by
  structure. Each level has its own learned query; pooling weights are shared across nodes of the same type.
- **The line-to-token map** comes from tokenizer offsets. Each token belongs to exactly one innermost unit, and to at
  most one leaf step.
- **Approach and intent are two heads on the one program vector.** They are two views of the same span, not two
  scales.
- **Code encoder:** start from a pretrained code encoder (candidates: UniXcoder, CodeT5+ 110M embedding, or a small
  decoder LM such as Qwen2.5-Coder-0.5B with mean/attention pooling). Fine-tune it rather than train from scratch.

### 4.2 Ablation: free latent queries

Replace span-restricted pooling with a fixed set of free latent queries per level (e.g. 16 → 4 → 1) that can attend
to all tokens. Match latents to descriptions with Hungarian matching or max-similarity. This tests whether a
hierarchy *emerges* without span grounding.

### 4.3 Alternatives kept for later

- Separate encoder stages per level (a more explicit hierarchy, but more parameters).
- Parallel multi-scale encoders (risk of redundant rather than hierarchical representations).

## 5. Training Objectives

### 5.1 Multi-level text alignment (main objective)

Let \(z_v\) be the vector of node \(v\) at level \(\ell\), \(P_\ell\) a level head, and \(y_v\) the text embedding of
its description. The loss sums over all supervised nodes in a batch:

\[
\mathcal{L}_{\text{text}} = \sum_{\ell} \lambda_\ell \, \frac{1}{|V_\ell|} \sum_{v \in V_\ell}
\operatorname{InfoNCE}\big(P_\ell(z_v),\, y_v;\ \mathcal{N}_v\big)
\]

Negatives \(\mathcal{N}_v\):

- other nodes of the same level in the batch;
- **hard negatives from the same program at other levels** (a unit's step texts, and the program's approach text for
  the intent head, and vice versa). This forces levels apart;
- **same-problem pairs are masked** (other solutions of the same problem cluster), because they share the task. They
  can instead serve as extra positives for the intent head.

Because nodes per program vary, \(\lambda_\ell\) is normalised per node so large programs don't dominate. When a unit
has a single top-level step (section 2.3 item 5), we keep both targets, but log the step/unit similarity for those
cases separately.

### 5.2 Text encoder: frozen vs trainable (open decision)

- **Frozen** (e.g. a sentence-embedding model): fixed targets. The objective is JEPA-like: the text model is the
  target encoder, and the code encoder + head is the predictor. A regression loss (cosine/L2) is possible alongside
  InfoNCE.
- **Trainable**: CLIP-style. Stronger retrieval, but the targets drift, and the text encoder can merge levels.

Start frozen, because that is closer to the JEPA framing and simpler to analyse; run trainable as an experiment.

### 5.3 JEPA-style latent prediction (later experiment)

Two options that use the data tree:

- **Masked-span prediction.** An EMA target code encoder encodes the full program; the online encoder sees the
  program with a step or unit span masked, and a predictor predicts the target node vector of the masked span.
- **Cross-level prediction**, matching the collaborator's framing (predict level k+1 from level k). A predictor maps a
  node's children vectors to the parent's target vector (step → unit → program), with stop-gradient/EMA targets.

\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{text}} + \beta \, \mathcal{L}_{\text{JEPA}}
\]

## 6. Experiments

All experiments use the same pretrained code encoder, the same cluster-level split, and the same budget in steps.

| # | Model | Supervision | Purpose |
|---|---|---|---|
| 1 | Flat encoder, no text | none (or the encoder's own pretraining) | Reference point |
| 2 | Flat, mean-pooled | intent + approach only | Code-text supervision without a hierarchy |
| 3 | Hierarchical (span pooling) | intent + approach only | Does the architecture alone help? |
| 3a | Hierarchical | same *number* of descriptions as exp 4, all program-level | Controls for more text/gradient per sample |
| 3b | Hierarchical | program descriptions duplicated at every level | Controls for extra loss terms |
| 4 | Hierarchical | step + unit + approach + intent | **Main hypothesis** |
| 5 | As 4 | descriptions shuffled across programs (within level) | Do gains depend on meaningful pairs? |
| 6 | As 4 | + JEPA objective (section 5.3) | Incremental value of latent prediction |

Ablations on the main model (exp 4):

- Span-restricted pooling vs free latent queries (section 4.2).
- Drop one level at a time (no steps, no units, no approach).
- Hard negatives on/off; same-problem masking on/off.
- Frozen vs trainable text encoder.
- Filtering by quality flags and reconstruction results vs training on all records.
- Per source: competitive programming only vs KodCode only vs mixed.

The token level of the hierarchical model is not equivalent to a flat encoder, so exp 1 stays a separate setup.

## 7. Evaluation

### 7.1 In-distribution retrieval, per level

On the cluster-level test split: text → code-node and code-node → text retrieval for steps, units, approach and
intent (Recall@1/5/10, MRR). For steps and units, retrieve within the same program and across the test set.

### 7.2 Level distinctness

- **Level-confusion matrix:** for each node, does its vector retrieve its own level's description over the same
  program's descriptions at other levels?
- CKA between level representations.
- Approach vs intent: the program vector's two heads should separate "how" from "what".

### 7.3 Probes (lightweight, frozen representations)

Labels come for free from the data:

- Algorithm tags and difficulty (`meta.tags`, `meta.difficulty`, `cf_rating`) from the program vector.
- Complexity class and key data structures (parsed from `L1_approach`) from the program vector.
- Step kind (`for` / `if` / `while` / `simple`) from step vectors, as a sanity check.

### 7.4 Beyond the teacher's style

In-distribution retrieval partly rewards learning the teacher's writing style. Add:

- **Solution-to-solution retrieval:** two solutions of the same problem should be nearest neighbours (no text
  involved).
- An external code-search benchmark (CodeSearchNet Python, CoSQA, or AdvTest).

A useful hierarchy is not claimed only from a better downstream score; the distinctness metrics in 7.2 must also
show that the levels carry different information.

## 8. Implementation

1. **Loader + validator** for the JSONL format:
   - flatten the tree into nodes `{level, id, parent, span, text}`;
   - build the line → token map from tokenizer offsets and assign each token to its innermost unit and leaf step;
   - check that steps partition each unit's code lines;
   - drop programs over the token cap and log the drop rate.
2. **Re-split:** cluster problems across sources by normalised prompt, then split by cluster (section 2.5).
3. **Text-embedding cache:** embed all descriptions once with the frozen text encoder.
4. **Model:** encoder + tree pooling with batched segment attention (pad nodes per level, mask by parent index).
5. **Training/eval scripts** with the level set configurable, e.g.:

```yaml
hierarchy:
  supervised_levels: [step, unit, approach, intent]
  pooling: span_restricted     # or free_queries
  step_depth: 2                # 1 = flatten child steps into parents
objectives:
  text_alignment: {loss: infonce, hard_negatives: true, mask_same_problem: true}
  jepa_prediction: false
text_encoder: {name: <sentence-embedding model>, frozen: true}
code_encoder: {name: <pretrained code encoder>, max_tokens: 1024}
```

## 9. Next Steps

1. Write the loader/validator and run it over all annotated records.
2. Build the cross-source problem clusters and the new split; report the leakage removed.
3. Choose the code encoder and the frozen text encoder; cache the text embeddings.
4. Train exps 2 and 4 on CodeContests alone (already annotated) to get a first signal.
5. Add TACO and KodCode when annotation finishes; run the full experiment table and ablations.
6. Run the JEPA experiments (exp 6).

## 10. Open Questions for the Collaborator

- Can the pipeline emit a cross-source problem cluster ID (or should we compute it)?
- Should single-step units be "promoted" (step replaced by its children) in the data, or should we handle it in the
  loader?
- Can the reconstruction test be run on more samples (e.g. the whole test split) so we can use it as a per-level
  quality label?
- Would a second teacher on a subset be feasible, to measure teacher-style bias?

## 11. Main Research Hypothesis

**Span-grounded, multi-level textual supervision leads a code encoder to learn representations whose levels are
distinct and individually useful, improving code understanding and retrieval over flat code-text alignment.**

This is a hypothesis to test. The central challenge is showing that the levels become meaningfully distinct, not
redundant copies of the same program meaning.
