# Code-JEPA: Research Plan (phase 2)

Updated 2026-10-26: re-based on H-JEPA as implemented in [kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA)
([arXiv:2610.06805](https://arxiv.org/abs/2610.06805)) and the LLM-JEPA recipe as implemented in
[rbalestr-lab/llm-jepa](https://github.com/rbalestr-lab/llm-jepa), with a pretrained code LLM as the backbone and an
architecture sized to the collaborator's data (`/mnt/storage/initzan/hjepa_code/` on tzag, documented in its
`DATA_PRINCIPLES.md`). Earlier discussion: [discussion_2026-10-25.md](discussion_2026-10-25.md).

**Empirical update 2026-11-05.** On CodeContests 0.5B, training *without* stop-grad collapsed program-level
rank by step 400 (intent effective rank 32 → 1.5) even with SIGReg μ=0.1. A 256-embedding SIGReg bank plus μ=0.3
was not enough by itself. The setting that kept rank at every level and still moved step cosine (0.55 → 0.96 at
step 1200) is **stop-grad on \(c_v\), SIGReg bank 256, μ=0.3, num_proj 1024**. Live configs:
`configs/tzag_e{1,2,4}_stopgrad.yaml` (4 epochs). Do not treat `runs/e0_e4/` as the method.

**Correction 2026-11-08.** In the 2026-11-05 sweep and the 2026-11-06 4-epoch runs, SIGReg was applied to the
*detached* target, so it had no gradient. The "bank 256, μ=0.3" part of that recipe was never tested, and the
results are discarded. They also showed that raw cosine is misleading: at step 0, approach cosine is 0.43 raw but
0.006 after mean-centring. By the end, E4's approach/intent effective rank had fallen from 34 to 8 while raw
cosine read 0.99. The current recipe is stop-grad on the view target with SIGReg on the **un-detached code-node
embeddings** (`sigreg.side: code`), run text → code and code → text (`configs/tzag_e{2,4}_{sgcode,c2t}.yaml`).
Go/no-go is now judged on mean-centred cosine and code-side effective rank, split by single- vs multi-unit
programs.

## 1. Goal

Test whether a pretrained code LLM, trained with a **hierarchical LLM-JEPA objective**, learns latent representations
of programs that are organised by abstraction level (step → unit → approach / intent), and whether this improves
code understanding and description-to-code generation over regular fine-tuning.

The collaborator built the dataset for exactly this: "H-JEPA-style (LeCun, 2022) training of code LLMs, following the
LLM-JEPA recipe. Every level is a separate *view* of the same program at a different level of abstraction."

## 2. Data

### 2.1 Source and status

| Source | Kind | Tests | Annotated programs | Solutions per problem |
|---|---|---|---|---|
| CodeContests train | competitive programming | stdio | 11,677 | ≤ 2 |
| TACO train | competitive programming | stdio / call-based | 25,181 | ≤ 2 |
| KodCode-V1 (100k subsample) | function-level tasks | pytest | 74,667 | 1 |

- **All three sources are annotated (2026-10-26)**, 111.5k programs in total; rejects are kept in separate files.
- Python 3 only. Every solution passes its tests before annotation. Decontaminated against HumanEval+, MBPP+,
  TACO test, and CodeContests valid/test.
- Annotator: Qwen3.8-27B, one guided-JSON call per sample, written bottom-up (steps → units → approach → intent).
- Each record also carries the problem `prompt`, the `tests`, `meta` (tags, difficulty, rating) and `quality.flags`.
- KodCode is CC-BY-NC, so the combined dataset is for non-commercial research only.

### 2.2 The hierarchy

The tree's shape (units, steps, line spans) comes from tree-sitter; the LLM only writes the text. Every description
is tied to a real code span.

| Level name | Data field | Text | Span |
|---|---|---|---|
| Code (tokens) | `L4_code` | none | each token |
| Step | `L2_units[].steps[]` (+ `children`) | one line of pseudocode with key conditions/formulas | AST block |
| Unit | `L2_units[]` (+ `methods`) | `responsibility` + `contract` | function / class / method / `main` |
| Approach | `L1_approach` | strategy, data structures, complexity | whole program |
| Intent | `L0_intent` | the task; must hold for any correct solution | whole program |

Levels are named, not numbered, because the dataset's L0 is the coarsest level.

### 2.3 Measured facts that constrain the architecture

Measured on the full annotation (CodeContests / TACO / KodCode):

| Fact | Value | Consequence |
|---|---|---|
| Nodes per program (steps + units + 2 program nodes), median / p90 / p99 / max | 8 / 21 / 46 / 81 (CC); 7 / 13 / 25 / 67 (KodCode) | Variable-size node sets; no fixed latent slots |
| Programs with a single unit | 63% / 65% / 74% | The unit and program levels usually cover the same span |
| Programs with one unit and one step | 12% / 21% / 16% | All three levels share one span; only the text differs |
| Units with a single top-level step | 39% / 40% / 37% | Step and unit share a span |
| Steps nested two deep | 29% / 23% / 10% of programs | Parent steps exist; depth is at most 2 |
| All descriptions per program, words, median / p99 | 235 / 911 (CC); 172 / 550 (KodCode) | ≈ 330 / 1,300 tokens to encode all text views |
| Code length, characters, median / p99 | 454 / 3,741 (CC); 624 / 2,275 (KodCode) | ≈ 150 / 1,100 tokens (verify with the tokenizer) |
| Problems with two solutions (CC) | 4,953 of 6,338 | Natural second view of the same intent |
| Median description length, words | intent 28, approach 41, unit 36, step 18 | Short text views |

Other facts carried over from the first data review:

- Steps partition each unit's own code. Uncovered lines are headers, blanks, comments, docstrings, imports, or lines of
  a nested unit, and are assigned to their unit.
- Units nest: `main` surrounds module-level code, classes contain methods. Each line belongs to its innermost unit.
- Intent contains information that is not in the code (it comes from the problem statement), so expect an error floor
  at the intent level.

### 2.4 Quality signals

- `quality.flags`: identifier leaks in intent/approach (~5% of records), over-long text. Soft; filter or down-weight.
- **Reconstruction test** (pilot_v2, 200 samples): the teacher rewrites code from intent..level k, and the pass rate
  rises with depth: 30.5% → 41.5% → 61.0% → 65.0%. Intent alone passes 12% (CodeContests) vs 48% (KodCode).

### 2.5 Fixes before training

1. **Cross-source duplicates leak across splits.** 5,034 of 6,365 CodeContests problems also appear in TACO under
   different IDs. Cluster problems by normalised prompt across sources and split by cluster. Merging also gives up to
   4 solutions per problem.
2. **Effective scale:** roughly 15k unique competitive problems plus ~75k KodCode tasks.
3. **Test split:** use 2–3% of clusters, so that per-level evaluation and pass@1 have enough samples.

## 3. What We Take From H-JEPA and LLM-JEPA

### 3.1 H-JEPA

H-JEPA was proposed in LeCun (2022, "A Path Towards Autonomous Machine Intelligence") and implemented end-to-end in
[Zhang, Terver, Rabbat, LeCun, Balestriero, 2026 (arXiv:2610.06805)](https://arxiv.org/abs/2610.06805), with code at
[kevinghst/H-JEPA](https://github.com/kevinghst/H-JEPA) (MIT, built on LeWorldModel). That work is a temporal,
action-conditioned world model for visual planning trained from scratch, so its planning code does not transfer.
Its training recipe and its findings do.

The recipe, as implemented (`h_jepa/models/hjepa.py`, `h_jepa/loss.py`):

- **Each level has its own encoder, predictor and latent space** (embedding size 192 per level). The level ℓ encoder
  reads the level ℓ−1 *embeddings*, not raw input: a pointwise MLP, or `SequenceEncoder`, a 1-layer Transformer with
  a CLS token and a padding mask that pools a variable-length window of lower-level embeddings into one vector.
- **Per-level loss:** \(\mathcal{L}^{(\ell)} = \mathcal{L}^{(\ell)}_{\text{pred}} + \lambda_\ell\,
  \mathrm{SIGReg}(Z^{(\ell)})\), with MSE prediction. The total is the sum over levels.
- **End-to-end, with no EMA, no stop-gradient and no pretrained weights.** Gradients from upper-level losses flow
  into lower-level encoders.
- **SIGReg** (Sketched Isotropic Gaussian Regularizer, about 35 lines and DDP-aware) pushes each level's embeddings
  towards an isotropic Gaussian along 1,024 random projections. Weight 0.18 per level in simulation, 0.08 on DROID.
- **Per-level learning rates:** 5e-5 for level 1, 1e-5 for the upper levels.

Findings that bear on this plan:

1. **SIGReg, not detachment, prevents collapse** (appendix K). Without SIGReg, effective rank falls to 1–14 of 384
   whether or not the target is detached; with it, detachment makes no measurable difference, so the headline models
   do not detach.
2. **Slow-feature collapse is an optimum of the prediction loss** (appendix L). A JEPA can reach zero prediction
   error by encoding only the easy-to-predict content, and no regulariser on the marginal distribution (including
   SIGReg) can rule this out. They needed grounded supervision (an inverse-dynamics head) to keep the relevant
   information.
3. **An upper level helps only if the data has structure at that scale** (appendix G). Stagewise training (freezing
   the lower level) lets each level train on the data that suits it.
4. **Baseline "HWM":** upper levels with identity encoders and no upper-level SIGReg. It is the control for "does a
   learned upper encoder matter".
5. **Future work they list:** variable-duration segments instead of fixed strides, and aligning upper-level latents
   with language. Our AST spans and descriptions provide both.

How this plan applies H-JEPA:

| H-JEPA element | How this plan applies it |
|---|---|
| Predict in representation space | All JEPA losses compare embeddings; the LM loss is a separate anchor |
| Each level encodes the embeddings of the level below | Step nodes read code tokens, unit nodes read step nodes, program nodes read unit nodes (section 5.2) |
| A predictor and a loss per level, summed, trained end-to-end | Level-specific predictor tokens and \(\lambda_\ell\) per level (sections 5.3, 6.2) |
| SIGReg per level, no EMA or stop-gradient | SIGReg on each level's node embeddings is on by default (section 6.6) |
| Grounded supervision against slow-feature collapse | The description views play the role of their inverse-dynamics head; code-only masked prediction (6.4) is never run without them |
| An upper level needs structure at its scale | Unit-level results are reported on multi-unit programs; stagewise training is an ablation |
| HWM control | Parent = mean of its children, with no learned upper encoder (section 5.6) |

Related, but a different method: HiLP (Shi, Pearce, Tomar, Sen, Langford,
[arXiv:2608.05806](https://arxiv.org/abs/2608.05806), no code released). It adds a second latent level to
next-latent prediction during LM *pretraining*, from scratch: the upper latent summarises a fixed window of k = 4
tokens and predicts k tokens ahead, using stop-gradient targets and no text supervision. The goal is better
next-token prediction (HumanEval pass@1 8.8 → 11.3 at 1B / 100B tokens); the hierarchy is discarded at inference and
its levels are not evaluated as representations. We differ in using AST spans instead of fixed windows,
description views per level, and a pretrained backbone, and in making level-wise representations the object of
study. HiLP lists a dynamic lookahead horizon as future work, which our spans provide.

Also related, also a different method: NCM, Next-Concept Modeling (Kumbong et al.,
[arXiv:2610.14162](https://arxiv.org/abs/2610.14162), code at [HazyResearch/ncm](https://github.com/HazyResearch/ncm)).
It is a fast coarse-to-fine text generator: a multiscale residual VQ-VAE turns text into K levels of discrete codes
with fixed lengths, and a transformer trained from scratch generates the levels in parallel. Its levels have no
attached meaning, and it uses no JEPA loss and no code evaluation. We take three ideas from it:
- **Nested dropout:** a random prompt depth for the LM loss (section 6.1), compared against the fixed prompt.
- **Intervention probes:** fix, swap or perturb program-level embeddings and inspect the generated code. This needs
  generation conditioned on node embeddings (the HiLP-style combined head).
- **Residual coding:** an ablation where a child's embedding is its parent's plus a learned residual.

### 3.2 LLM-JEPA (Huang, LeCun, Balestriero, 2025), as implemented in `finetune.py`

Verified in the reference code:

- **Two views of one sample:** `Text` (user message) and `Code` (assistant message), e.g. NL → regex, NL → SQL,
  GSM8K question → answer.
- **One shared encoder:** the LLM itself. An embedding is the last-layer hidden state at the last token of a view.
  There is **no EMA and no stop-gradient**; gradients flow through both views.
- **Predictor = the LLM itself:** `k` special `<|predictor_i|>` tokens are appended to `Text` (`k ∈ {0, 1, 2}` in the
  sweeps). The embedding of the last predictor token is the prediction. The predictor shares all weights.
- **Loss:** `γ · L_LM + λ · (1 − cos(Pred(Enc(Text)), Enc(Code)))`, with `L_LM` the next-token loss on `Code`
  given `Text`. Sweeps use `λ ∈ {0.5, 2.0}`, `γ = 1`, learning rate 1e-5, 4 epochs, 5 seeds, 1–2B models
  (Llama-3.2-1B, gemma-2-2b, OLMo-2-1B, …).
- **Ablation flags:** L2, MSE, InfoNCE (temperature 0.07, in-batch negatives), predictor at the front, reversed
  direction (`Code` predicts `Text`), LoRA, pretraining from scratch.
- **Compute:** the original runs three forward passes (full sequence, `Text`, `Code`). `--additive_mask` packs `Text`
  and `Code` into one row with a block-diagonal 4D mask. **Random JEPA-loss dropout** (`--jepa_ratio`) applies the
  JEPA loss on only a fraction of steps; dropping 75% costs 1.25× regular fine-tuning with little loss in accuracy.
  Comparisons with regular fine-tuning are also made at equal FLOPs (`--same_flop`).
- **The LM loss is what prevents collapse.** It keeps the hidden states informative, so no EMA is needed.

The same repository also contains **Semantic Tube Prediction** (`stp.py`), which embeds a span as the difference of
hidden states at its boundaries (`h[end] − h[start]`) and asks a span to be predictable from the rest of the
sequence. We use the boundary-difference embedding only as a pooling ablation: differences telescope (a parent's
difference is the sum of its children's), so a hierarchy built from them would be trivially additive.

### 3.3 Where this plan departs from the references, and why

1. **More than two views, tied to a tree.** LLM-JEPA has one (Text, Code) pair per sample. We have one pair per tree
   node (median 7–8 per program), so code-side embeddings are per node, not one last token.
2. **New node tokens are not covered by the LM loss.** In LLM-JEPA the target is the last token of a real message, which
   the LM loss also shapes. Our node tokens appear only in JEPA losses, so they could collapse even while the LM loss
   is healthy. Following H-JEPA, each level's node embeddings get SIGReg by default, and per-level rank is monitored
   (section 6.6).
3. **Code-only input for representations.** Node embeddings are computed from code alone; descriptions are separate
   views, never context for the code view.

## 4. Core Principles

- **One pretrained code LLM** is the encoder for every view (code and all descriptions) and also the predictor.
- **Code-only code view.** The code view never attends to descriptions.
- **The hierarchy follows the data tree.** Each node is a token that sees only its own span or its children.
- **Every level has a JEPA loss**, and the LM loss anchors the backbone.
- **Level distinctness must be shown, not assumed.** Every run reports the level-confusion matrix and per-level
  embedding variance and rank (section 8.2).

## 5. Model Architecture

### 5.1 Backbone

| Role | Candidates | Why |
|---|---|---|
| Main backbone | **Qwen2.5-Coder-1.5B** (base) | Code-pretrained decoder in the size range LLM-JEPA validated (1–2B); full fine-tuning fits one L40S (46 GB) with bf16 + gradient checkpointing |
| Development backbone | Qwen2.5-Coder-0.5B (already cached in `models/hf`) | Fast iteration and smoke tests |
| Upper bound (later) | Qwen2.5-Coder-7B with LoRA | LLM-JEPA supports LoRA; only if the 1.5B results justify it |
| Baseline code encoder | UniXcoder (125M) or CodeSage-small | Bidirectional encoder for the previous plan's design (experiment B1) |
| Baseline text target | A frozen sentence-embedding model (e.g. bge-base-en-v1.5) | Frozen targets for B1 |

**Scale check:** ~111k programs and ~1M supervised nodes (≈ 15k + 75k unique tasks after clustering). LLM-JEPA
fine-tunes 1–2B models on datasets of a few thousand samples (GSM8K: 7.5k), so full fine-tuning at 0.5–1.5B is within
the validated range; 7B would need LoRA and is not needed to test the hypothesis.

Use the **base** model with plain formatting (LLM-JEPA's `--plain` path), not a chat template. Node and predictor
positions are then explicit and do not depend on per-model `last_token` offsets.

### 5.2 Main design: node tokens with a tree attention mask

The pretrained LLM itself performs the hierarchical encoding. After the code tokens we append one special token per
tree node, and an attention mask restricts what each node token sees, at every layer:

```text
row (code view):  [code tokens ........][step nodes][parent-step nodes][unit nodes][<approach>][<intent>]

code token     -> earlier code tokens (normal causal attention)
leaf step      -> the code tokens in its lines
parent step    -> its own header tokens + its child step nodes
unit           -> its header / docstring / import tokens + its top-level step nodes + nested unit nodes
<approach>     -> all top-level unit nodes
<intent>       -> all top-level unit nodes
```

- **Embedding of node v:** \(c_v\) = a per-level projection head (2-layer MLP to 256 dimensions, like H-JEPA's
  per-level projector) applied to the last-layer hidden state of v's node token. The predictions \(p_v\) go through
  the same level's head. Using the raw hidden states, as LLM-JEPA does, is an ablation.
- **Special tokens:** `<step>`, `<unit>`, `<approach>`, `<intent>` (initialised to the mean input embedding).
- **Position ids:** a node token takes the position of the last token of its span.
- **Approach and intent are two tokens** with the same mask. Two separate code-side targets are needed: if both
  descriptions predicted one program vector, the loss would pull "what" and "how" together.
- **Code tokens are causal**, so a step's tokens also carry context from earlier code. This is acceptable (meaning
  depends on context); a node still reads only its own span's tokens.
- This is H-JEPA's "each level encodes the level below" implemented inside the pretrained network, with all layers
  shared. It reuses LLM-JEPA's 4D additive-mask mechanism.

### 5.3 Text views and predictors

Each description is its own segment in the same packed row, attending only to itself:

```text
[description of node v] <pred_ℓ,1> ... <pred_ℓ,k>      ->  p_v = last-layer state of <pred_ℓ,k>
```

- **Level-specific predictor tokens** (`ℓ ∈ {step, unit, approach, intent}`, `k ∈ {0, 1, 2}`): one predictor per
  level, as in H-JEPA, with all weights shared, as in LLM-JEPA. `k = 0` means the last description token is used directly.
- Unit text = `responsibility` + `contract`; approach text = `strategy` + data structures + complexity.

### 5.4 Packing (one sample = two rows)

| Row | Contents | Loss |
|---|---|---|
| LM row | `intent + approach` prompt → code | next-token loss on code tokens only |
| View row | code + node tokens, then every description segment with its predictor tokens, block-diagonal mask | JEPA losses |

Estimated lengths: the view row is ≈ 550 tokens at the median and ≈ 2,600 at p99 (CodeContests). Cap at 3,072 tokens
and drop the remainder (< 1%), logging the drop rate. Use length-bucketed batches. A dense 4D mask with SDPA works at
this length (FlashAttention does not accept arbitrary masks); FlexAttention is an option if memory becomes tight.
Random JEPA-loss dropout (section 3.2) skips the view row on a fraction of steps.

### 5.5 How the architecture matches the data

| Data property | Design response |
|---|---|
| Variable number of steps and units (up to 81 nodes) | One token per node, no fixed slots; cap at 96 nodes |
| Steps nested at most two deep | Parent-step tokens read child-step tokens; depth-1 flattening is a config option |
| Units nest (`main`, classes → methods) | Unit tokens read nested unit tokens; each line belongs to its innermost unit |
| 63–74% single-unit programs | Results are reported separately for single-unit and multi-unit programs; masked-unit prediction runs only on programs with ≥ 2 units |
| ~39% single-step units | Masked-step prediction skips them (the context would be empty); log their step/unit similarity separately |
| Approach and intent describe the same span | Two program tokens, used as hard negatives for each other |
| Two to four solutions per problem | Cross-solution intent prediction (section 6.5); same-problem pairs are masked as negatives |
| Intent contains information not in the code | Intent-level scores have a floor; not read as hierarchy failure |
| Tests for every program | Generation pass@1 as an outcome metric (section 8.1) |

### 5.6 Architecture ablations

- **Pooling heads instead of node tokens:** per-level encoders on top of the last-layer token states, as in H-JEPA:
  a step reads its span's tokens, a unit reads its step vectors, and so on. Reuse H-JEPA's `SequenceEncoder` (CLS
  token, padding mask, variable-length input) for each level. Cheaper and well-behaved at initialisation; also the
  first smoke test.
- **HWM-style control:** a parent's vector is the mean of its children's, with no learned upper encoder and no
  upper-level SIGReg.
- **Free node tokens:** node tokens that see all code tokens. Tests whether span grounding is needed.
- **Boundary-difference spans** (`h[end] − h[start]`, from STP) as the step embedding.
- **Bidirectional encoder + frozen text encoder** (experiment B1).

## 6. Losses

Notation: \(c_v\) = code-side embedding of node v (section 5.2); \(p_v\) = prediction from v's description (section
5.3); \(V_\ell\) = nodes of level ℓ in the batch; \(d(\cdot,\cdot)\) = the JEPA distance (section 6.6).

### 6.1 LM anchor (LLM-JEPA)

\[
\mathcal{L}_{\text{LM}} = -\sum_t \log p_\theta(\text{code}_t \mid \text{intent}, \text{approach}, \text{code}_{<t})
\]

The prompt matches the realistic generation setting (task + plan → code) and the collaborator's reconstruction test.
Ablation: a randomly chosen prompt depth (intent; + approach; + units; + steps) per sample.

### 6.2 Same-node view loss (LLM-JEPA at every level), main objective

\[
\mathcal{L}_{\text{view}} = \sum_{\ell \in \{\text{step, unit, approach, intent}\}} \lambda_\ell \,
\frac{1}{|V_\ell|} \sum_{v \in V_\ell} d\big(p_v,\ c_v\big)
\]

Direction is description → code, as in LLM-JEPA (reverse as an ablation). The written default was no stop-gradient
(as in LLM-JEPA / H-JEPA); **on this data, stop-gradient on \(c_v\) is required** (see the 2026-11-05 note at the
top). SIGReg must then act on the un-detached code side, or it has no gradient (2026-11-08 correction). Code →
description is run alongside it, not only as an ablation. \(\lambda_\ell\) is normalised per node so that large
programs do not dominate.

### 6.3 Level-to-level chain (the collaborator's framing)

"Predict level k+1 from level k, all the way down to code", with whole levels as views:

\[
\mathcal{L}_{\text{chain}} = d\big(P(\text{intent}), E(\text{approach})\big)
+ d\big(P(\text{approach}), E(\text{units concatenated})\big)
+ d\big(P(\text{units}), E(\text{steps concatenated})\big)
+ d\big(P(\text{steps}), c_{\text{program}}\big)
\]

This is a direct multi-view extension of LLM-JEPA with no tree mask. It is run as its own experiment (E3) rather than
added to the main loss, because the views it uses are whole levels and are not tied to spans.

### 6.4 Masked-node prediction (H-JEPA / I-JEPA style, code-only)

For each program, sample one step (and one unit if the program has ≥ 2 units). In an extra copy of the code segment,
the sampled node's tokens are hidden from every other token. A level-specific predictor token placed at the node's
position reads the remaining code tokens and sibling nodes, and predicts the clean \(c_v\):

\[
\mathcal{L}_{\text{mask}} = \sum_{\ell \in \{\text{step, unit}\}} d\big(\hat{c}_v^{\text{masked}},\ \operatorname{sg}(c_v)\big)
\]

Following H-JEPA, the default has no stop-gradient and no EMA, and relies on SIGReg; stop-gradient and EMA are
ablations. This is the only objective that uses no text, so it shows how much of the hierarchy comes from structure
alone. H-JEPA's slow-feature result (section 3.1) predicts that on its own it can settle on easy-to-predict content
(input parsing, boilerplate), so it is only ever added on top of the view loss, and probes check what it encodes.

### 6.5 Cross-solution intent prediction

For two solutions A and B of one problem cluster: \(d(P_{\text{intent}}(c_{\text{intent}}^A),\ c_{\text{intent}}^B)\).
Applied to intent only; approaches may differ between solutions. This is a pure code-to-code view pair supplied by the data.

### 6.6 Distance and anti-collapse (experimental axis)

| Option | Definition | Source |
|---|---|---|
| **Cosine + SIGReg (default)** | \(1 - \cos(p, c)\), plus \(\mu_\ell\,\mathrm{SIGReg}(\{c_v\}_{v \in V_\ell})\) per level | LLM-JEPA distance + H-JEPA regulariser |
| Cosine only | \(1 - \cos(p, c)\) | LLM-JEPA default |
| MSE + SIGReg | squared error on the embeddings, plus SIGReg | H-JEPA as published |
| InfoNCE with level hard negatives | temperature 0.07; negatives = same-level nodes in the batch + the same program's nodes at other levels; same-problem pairs masked | LLM-JEPA `--infonce`, extended (previous plan's main loss) |

SIGReg is on by default because node tokens are not anchored by the LM loss (section 3.3), and H-JEPA found that only
SIGReg kept the rank up (section 3.1). It is copied from H-JEPA's `loss.py` and applied to each level's node
embeddings in the batch; \(\mu_\ell\) starts at 0.1 and is swept over {0.03, 0.1, 0.3}. Per-level effective rank is
still logged every N steps. Applying SIGReg to the backbone's hidden states would distort the pretrained geometry,
so it is applied only after a small per-level projection head.

### 6.7 Total loss and starting hyperparameters

\[
\mathcal{L} = \gamma\,\mathcal{L}_{\text{LM}} + \lambda\,\mathcal{L}_{\text{view}} + \beta\,\mathcal{L}_{\text{mask}}
+ \eta\,\mathcal{L}_{\text{xsol}} + \sum_\ell \mu_\ell\,\mathrm{SIGReg}\big(\{c_v\}_{v \in V_\ell}\big)
\]

Starting points taken from LLM-JEPA: \(\gamma = 1\), \(\lambda \in \{0.5, 2.0\}\), \(k \in \{0, 1, 2\}\) predictor
tokens, learning rate 1e-5 for the backbone, 4 epochs, JEPA-loss dropout 0.5 at first (0.75 if compute-bound).
\(\beta, \eta\) start at \(0.5\lambda\). Following H-JEPA's per-level learning rates, the new parameters (node and
predictor token embeddings, projection heads) use a higher rate, 5e-5.

## 7. Experiments

Same backbone, same cluster-level split, 3 seeds for the main comparisons. Comparisons with E1 are made both at equal
steps and at equal FLOPs (LLM-JEPA's protocol).

| # | Name | Losses | Question |
|---|---|---|---|
| E0 | Zero-shot backbone | none | Reference for every metric |
| E1 | Regular fine-tuning | LM | LLM-JEPA's baseline |
| E2 | Flat LLM-JEPA | LM + view loss on approach and intent only | Direct port of the reference recipe |
| E3 | Chain LLM-JEPA | LM + chain loss (6.3) | The collaborator's framing |
| E4 | **Hierarchical LLM-JEPA** | LM + view loss at all levels with the tree mask | **Main hypothesis** |
| E5 | E4 + masked-node prediction | + 6.4 | Value of code-only H-JEPA prediction |
| E6 | E4 + cross-solution | + 6.5 | Value of the multi-solution views |
| B1 | Encoder + frozen text targets | InfoNCE, no LM | The previous plan's design, as a baseline |

Controls on E4:

- **Shuffled descriptions** within each level: do gains depend on correct pairs?
- **Same number of descriptions, all at program level:** controls for extra text and gradient per sample.
- **Free node tokens** (no span restriction): does the hierarchy need grounding?
- **Drop one level** at a time.
- **HWM-style upper levels** (parent = mean of children, no upper SIGReg): does a learned upper encoder matter?
- **Stagewise training** (H-JEPA appendix G): train the step level on all programs, then freeze it and train the
  unit and program levels on multi-unit programs only.

Loss and recipe ablations on E4: the distance options in 6.6; SIGReg weight; projection head vs raw states;
stop-gradient on/off; \(k\); \(\lambda\); reverse
direction; prompt depth for the LM loss; quality-flag filtering; per source (competitive only, KodCode only, mixed).

## 8. Evaluation

### 8.1 Generation (LLM-JEPA's headline metric)

pass@1 on the test split with the `intent + approach` prompt, and with `intent` alone, run against the record's tests.
Report by source, since intent alone is far less informative for competitive problems.

### 8.2 Representation per level

- **Retrieval:** description → node and node → description for steps, units, approach and intent (Recall@1/5/10,
  MRR), within a program and across the test set.
- **Level-confusion matrix:** does a node's embedding retrieve its own level's description over the same program's
  descriptions at other levels? Reported separately for single-unit and multi-unit programs.
- **Collapse monitors:** per-level standard deviation and effective rank of \(c_v\) and \(p_v\).
- **Mapping structure:** following LLM-JEPA's analysis of the Enc(Text) → Enc(Code) relation, fit a linear map per
  level and report its fit and singular-value spectrum.
- CKA between levels.

### 8.3 Probes and transfer

- Program node: algorithm tags, difficulty, `cf_rating`, complexity class and data structures (parsed from the approach).
- Step node: step kind (`for` / `if` / `while` / `simple`) as a sanity check.
- Solution-to-solution retrieval (two solutions of one problem should be nearest neighbours).
- An external code-search benchmark (CodeSearchNet Python, CoSQA, or AdvTest), to check for more than teacher-style
  matching.

## 9. Implementation

1. **Loader + validator:** flatten the tree into nodes `{level, id, parent, span, text}`; map lines → token ranges
   from tokenizer offsets; assign header lines to their unit; check that steps partition unit code; measure the real
   token lengths and replace the estimates in sections 2.3 and 5.4.
2. **Re-split** by cross-source problem cluster (section 2.5).
3. **Packer:** build the LM row and the view row, the tree mask, node and predictor positions, and the masked copies
   for 6.4.
4. **Trainer:** a Hugging Face `Trainer` subclass modelled on LLM-JEPA's `RepresentationTrainer.compute_loss`, with SDPA and a
   4D additive mask.
5. **Evaluation:** a pass@1 harness reusing the collaborator's executor (`hjepa_data/executor.py`), plus retrieval,
   confusion and probe scripts.

```yaml
backbone: {name: Qwen/Qwen2.5-Coder-1.5B, lora: false, dtype: bf16, grad_checkpointing: true}
hierarchy:
  node_levels: [step, unit, approach, intent]
  node_encoding: tree_mask        # or pooling_mean | pooling_attn | free_nodes | boundary_diff
  step_depth: 2
  max_nodes: 96
  max_row_tokens: 3072
predictor: {tokens_per_level: 1, level_specific: true}
objectives:
  lm: {weight: 1.0, prompt: [intent, approach]}
  view: {weight: 0.5, distance: cosine, stop_grad_target: false, direction: text_to_code}
  chain: false
  masked_nodes: {enabled: false, levels: [step, unit], target: no_detach}   # or stop_grad | ema
  cross_solution: false
  sigreg: {weight: 0.1, num_proj: 1024, knots: 17}   # per level, on projected node embeddings
  jepa_dropout: 0.5
projection_head: {dim: 256, layers: 2}                # per level; none = raw hidden states
train: {lr_backbone: 1.0e-5, lr_new_params: 5.0e-5, epochs: 4, seeds: [82, 23, 37]}
```

## 10. Next Steps

1. Loader/validator over all three sources; measure real token lengths with the Qwen tokenizer.
2. Cross-source clustering and re-split; report the leakage removed.
3. Smoke test on CodeContests with the 0.5B backbone: E1 vs E2 with pooling heads, to verify the LLM-JEPA port.
4. Add the tree mask (E4) on the 0.5B model; check collapse monitors and the level-confusion matrix.
5. Move to 1.5B on all sources; run E1–E4 and B1, then E5/E6 and the ablations.

## 11. Open Questions for the Collaborator

- Does the collaborator plan training code of their own (the `LLM-JEPA-CODE` repo currently only generates data), and
  should we share one trainer?
- Can the pipeline emit a cross-source problem cluster ID?
- Should single-step units be "promoted" (step replaced by its children) in the data, or handled in the loader?
- Can the reconstruction test run on the whole test split, to give a per-level quality label?

## 12. Main Research Hypothesis

**Training a pretrained code LLM with a JEPA loss at every level of a span-grounded description tree, anchored by the
LM loss, produces representations whose levels are distinct and individually useful, and improves description-to-code
generation and code understanding over regular fine-tuning and flat LLM-JEPA.**

The central challenge remains showing that levels are distinct rather than redundant copies of program meaning,
especially since most programs have only one unit.
