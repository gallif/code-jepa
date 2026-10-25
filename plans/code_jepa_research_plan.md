# Code-JEPA: Proposed Research Plan

## 1. Goal

Explore whether a code encoder can learn **hierarchical latent representations** of source code, from token-level details to higher-level semantics, using LLM-generated textual descriptions at multiple abstraction levels.

The approach is inspired by hierarchical representation learning, including H-JEPA-style ideas, and latent-space prediction through JEPA. The proposed architecture is a research direction combining these ideas, not an exact reproduction of either method.

## 2. Core Principles

- **Code-only input:** The code encoder receives ordinary source-code tokens. LLM-generated descriptions are training targets, not inputs to the code encoder.
- **Level 1 is raw code:** The lowest level represents the original code tokens and receives no textual-abstraction supervision in the initial design.
- **Configurable hierarchy:** The number of levels, `L`, must be configurable.
- **Learned higher levels:** Higher-level representations are learned latent representations rather than manually supplied AST features.
- **Modular objectives:** Text alignment and JEPA-style latent prediction are distinct objectives and should be evaluated separately.
- **Flexible data format:** The data pipeline must support different numbers of levels and descriptions without requiring architectural redesign.

## 3. Proposed Model Architecture

### 3.1 Recommended starting point

Start with **one shared code Transformer and learned hierarchical pooling**.

The architecture consists of:

1. A tokenizer that converts source code into ordinary code tokens.
2. A shared Transformer encoder that processes the token sequence.
3. Level 1, which contains the token-level hidden representations.
4. Learned pooling or latent-query modules that progressively compress the representations into higher-level latent representations.
5. Level-specific prediction or alignment heads that connect higher-level representations to corresponding textual descriptions.

Conceptually:

```text
Source code
    |
Tokenizer
    |
Shared code Transformer
    |
Level 1: token-level representations
    |
Learned pooling / latent-query module
    |
Level 2: fine-grained latent representations
    |
Learned pooling / latent-query module
    |
Level 3: intermediate representations
    |
        ...
    |
Level L: coarse/global representation
```

### 3.2 Hierarchical representations

Let the input contain `N` code tokens. The shared Transformer produces:

\[
H^{(1)} \in \mathbb{R}^{N \times d}
\]

where `d` is the hidden dimension.

Level 1 contains the token-level code representations. Higher levels are constructed by learned compression modules:

\[
H^{(l+1)} = \operatorname{Pool}_l(H^{(l)})
\]

Each pooling module learns which information to retain when converting a lower-level representation into a more compact representation.

For example, with four levels:

| Level | Representation | Text supervision |
|---|---|---|
| Level 1 | Original code-token representations | None initially |
| Level 2 | A set of fine-grained latent vectors | Fine-grained descriptions |
| Level 3 | A smaller set of intermediate latent vectors | Intermediate descriptions |
| Level 4 | A global latent vector | Function-level description |

The number of latent vectors at each level is configurable. For example, a model might use `N` token representations at Level 1, 16 latent vectors at Level 2, 4 at Level 3, and 1 at Level 4. These numbers are illustrative, not fixed requirements.

The number of levels `L` must also remain configurable.

### 3.3 Alternative architectures

Three architectural options are worth considering:

**A. Shared Transformer with hierarchical pooling — recommended initially**

- Encodes the code once.
- Builds higher levels through learned pooling.
- Relatively simple and computationally efficient.
- Supports a configurable number of levels.

**B. Separate encoder stages per level**

- Each stage transforms the representations from the previous level.
- Provides a more explicit hierarchy.
- Introduces additional architectural complexity.

**C. Parallel multi-scale encoders**

- Separate branches represent the code at different scales.
- Makes scale-specific representations easy to compare.
- May produce redundant representations rather than a genuinely hierarchical structure.

The initial prototype should use option A unless implementation constraints or preliminary experiments motivate a different design.

## 4. Training Data and Textual Supervision

For each code sample, collect:

- The original source code.
- One or more LLM-generated descriptions at different abstraction levels.
- Optionally, associations between descriptions and specific code spans.

The exact number and definitions of the abstraction levels should be decided with the dataset collaborator. The implementation must support a variable number of levels rather than hard-coding a particular hierarchy.

### 4.1 Illustrative example

Source code:

```python
def positive_sum(xs):
    total = 0
    for x in xs:
        if x > 0:
            total += x
    return total
```

Possible descriptions:

**Fine-grained level**
- "Check whether the current element is positive."
- "Add a positive element to the accumulator."

**Intermediate level**
- "Iterate through the collection and accumulate only positive elements."

**Coarse or global level**
- "Calculate the sum of all positive elements in a collection."

These examples illustrate possible abstraction levels. They are not mandatory labels for the final dataset.

### 4.2 Important data-design decision

Descriptions can refer to either:

1. **The entire function at different abstraction levels**, or
2. **Specific code spans**, such as individual operations or logical blocks.

These choices have different architectural implications.

If every description refers to the entire function, each level receives a different semantic view of the same function. However, intermediate latent representations are not automatically associated with distinct code regions.

If descriptions refer to specific spans, the dataset needs reliable associations between each description and its corresponding code span. The model can then learn representations that are more directly tied to local operations and larger logical units.

This decision should be resolved with the dataset collaborator before finalizing the architecture.

### 4.3 Data quality

LLM-generated descriptions should be checked for:

- Semantic correctness.
- Relevance to the source code.
- Appropriate abstraction level.
- Consistency between descriptions and the underlying code.
- Duplicate or contradictory descriptions.

Training, validation, and test splits should minimize code leakage. Where practical, separate repositories or projects across splits.

## 5. Training Objectives

### 5.1 Multi-level text alignment

Let:

- \(Z^{(l)}\) denote the code representation at level `l`.
- \(Y^{(l)}\) denote the embedding of the corresponding textual description.
- \(P_l\) denote a level-specific projection or prediction head.
- \(D\) denote a suitable embedding-alignment loss.
- \(\lambda_l\) denote the weight assigned to level `l`.

The multi-level objective is:

\[
\mathcal{L}_{\text{text}}
=
\sum_{l=2}^{L}
\lambda_l
D\left(P_l(Z^{(l)}),Y^{(l)}\right)
\]

The text encoder converts the descriptions into target embeddings. The code encoder does not receive these descriptions as input.

Possible alignment losses include contrastive learning objectives and regression-style latent losses. The choice should be explicit and evaluated experimentally.

In the initial design, Level 1 receives no textual-abstraction loss. Higher levels receive supervision from their corresponding descriptions.

### 5.2 Optional JEPA-style latent prediction

A separate JEPA-style objective can be introduced to train the model to predict target code representations in latent space.

A JEPA objective involves a predictor and a target representation, typically produced by a target encoder. It is not simply another name for text alignment.

The combined objective could be:

\[
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{text}}
+
\beta \mathcal{L}_{\text{JEPA}}
\]

where `β` controls the contribution of the latent prediction objective.

The first prototype should focus on multi-level text alignment. JEPA-style prediction can then be added as a separate experiment to determine whether it provides additional benefits.

Additional hierarchy regularization may be investigated later, but should not be assumed necessary initially.

## 6. Baselines and Experiments

The following experimental conditions are proposed.

### Experiment 1: Flat code encoder baseline

Train a conventional code encoder without a learned hierarchy.

Purpose: Establish a reference point for downstream performance.

### Experiment 2: Flat code-text alignment

Use a single code representation aligned with function-level textual descriptions.

Purpose: Measure the benefit of code-text supervision without a hierarchy.

### Experiment 3: Hierarchical encoder with global text supervision

Use the hierarchical architecture, but supervise it only with function-level descriptions.

Purpose: Determine whether the hierarchy alone improves performance.

### Experiment 4: Hierarchical encoder with multi-level text supervision

Use the hierarchical architecture with descriptions at multiple abstraction levels.

Purpose: Test the main hypothesis that multi-level supervision produces more useful hierarchical representations.

### Experiment 5: Mismatched-description control

Shuffle or mismatch descriptions between code samples.

Purpose: Check whether improvements depend on meaningful code-description relationships.

### Experiment 6: Hierarchical encoder with JEPA-style prediction

Add the latent prediction objective to the multi-level text-supervised model.

Purpose: Measure the incremental contribution of JEPA-style training.

**Important:** Level 1 within the hierarchical model is not automatically equivalent to a conventional flat encoder baseline. Its representations may depend on different parameters, training losses, and contextual interactions. The flat baseline must therefore remain a separate experimental setup.

## 7. Evaluation

### 7.1 Code-text retrieval

Evaluate whether code representations retrieve relevant textual descriptions and vice versa.

Possible metrics:

- Recall@1, Recall@5, and Recall@10.
- Mean reciprocal rank (MRR).

### 7.2 Abstraction-level evaluation

Where suitable test data are available, evaluate queries and retrieval at different abstraction levels.

For example:

- Local operation descriptions.
- Logical block descriptions.
- Function-level descriptions.

This can help establish whether different levels support different kinds of semantic queries.

### 7.3 Representation probing

Train lightweight probes on representations from each level to test which information they encode.

Possible probe tasks include:

- Identifying local operations.
- Predicting function behavior.
- Matching code with descriptions at different abstraction levels.
- Recovering relevant semantic properties.

### 7.4 Ablation studies

Compare models while varying:

- Number of levels.
- Number of latent vectors per level.
- Text-supervision levels.
- Pooling strategy.
- Text-alignment loss.
- Presence or absence of the JEPA objective.

A useful hierarchy should not be claimed solely because the full model achieves a better downstream score. Evidence should also indicate that the levels contain useful and appropriately differentiated information.

## 8. Configuration and Implementation Requirements

The model, data pipeline, loss computation, and evaluation code should all support a configurable number of levels.

Illustrative configuration:

```yaml
hierarchy:
  num_levels: 4

  level_1:
    type: code_tokens
    text_supervision: false

  higher_levels:
    type: learned_latent_pooling
    text_supervision: true

objectives:
  text_alignment: true
  jepa_prediction: false
```

This is an illustrative configuration; exact field names can change during implementation.

The implementation should distinguish:

- **Number of hierarchy levels:** `L`.
- **Number of latent vectors per level:** A separate configurable parameter.
- **Number of descriptions per sample and level:** A property of the dataset.

These quantities are not necessarily equal. For example, a level may contain multiple latent vectors and may be supervised by multiple descriptions.

The following invariant should be maintained:

- Level 1 represents the raw code-token sequence.
- Level 1 receives no textual-abstraction supervision in the initial design.
- Higher levels are learned latent representations.
- The number and interpretation of higher levels are configurable.

## 9. Immediate Next Steps

1. Decide whether LLM-generated descriptions will refer to entire functions or specific code spans.
2. Coordinate with the dataset collaborator to determine the available abstraction levels and data format.
3. Implement a shared code Transformer with learned hierarchical pooling.
4. Implement multi-level text alignment.
5. Establish the flat code encoder and flat code-text alignment baselines.
6. Evaluate retrieval performance and probe the information represented at each level.
7. Run ablations to determine whether multi-level supervision provides benefits beyond function-level supervision.
8. Add JEPA-style latent prediction as a separate experiment.

## 10. Main Research Hypothesis

**Multi-level textual supervision may encourage a code encoder to learn latent representations that preserve useful information at different abstraction levels, potentially improving code understanding and retrieval compared with flat code-text alignment.**

This is a hypothesis to test, not an assumed outcome.

The central research challenge is to determine whether the learned levels become meaningfully distinct and useful, rather than simply producing redundant representations of the same code.
