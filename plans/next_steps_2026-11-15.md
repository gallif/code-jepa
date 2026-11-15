# Next steps after the 2026-11-15 benchmarks: framing and runs

For the agent picking this up. Read `AGENTS.md` first (its hard constraints still apply), then
`docs/BENCHMARKS_2026-11-15.md`. The benchmark code is in `bench/` on the `bench-code` branch.

This file says what paper we are now aiming for and which runs decide whether it holds. It replaces the
"better code model" goal in `plans/code_jepa_research_plan.md` §1. It does not change the training recipe.

## 1. Framing

**We are not building a better coder. We are building better vectors for code.**

The benchmark report shows:

- The fine-tunes do not generate better code than the base model (MBPP 55 → 35–38), and that loss comes from
  the CodeContests data, not from JEPA.
- As a plain text → code retriever we lose to a general embedder (gte-modernbert .67 vs our .32 on TACO).
- What does work is what the objective trains: a description finds its own block or function inside a program
  (R@1 .86 vs ≤ .70), and a program finds the other solution to the same problem (.54 vs .28).
- JEPA adds no language-model drift beyond the LM-only fine-tune (E1).

So the paper's claim is:

> One forward pass of a code LM gives a description-aligned vector for every block, function and whole
> program. The same vector serves as a **search key** (find the code a description refers to) and as a
> **one-token stand-in** for that code in the LM's context.

"Retrieve with the vector, then read the vector." The tree gives a zoom: program → functions → blocks → raw
tokens, expanding only where needed.

Why this is a JEPA paper: compressors trained by reconstruction (ICAE, gist tokens) spend capacity on surface
tokens. JEPA vectors are trained to hold the description, which is what a reader model needs. The central
experiment is **JEPA vectors vs reconstruction-trained vectors at an equal token budget**.

Wording rules that follow from the results:

- Say **"span-grounded, multi-level"**, not "hierarchical composition". `free` beats `tree` and the
  unit → program link is unused, so we cannot claim levels are built from the level below.
- **`free` is the main model.** `tree` is an ablation.
- Do not claim robustness to renaming, transfer to real repositories, or general-purpose retrieval. Today
  these are limitations (report §3, §2 SWE-bench row, §4).

## 2. What could sink the two results we rely on

Check these before building on them.

| Result | Open problem |
|---|---|
| Pointing (.86) | No strong baseline. Only E0 and E1 were compared. An embedder that encodes each block could match us. |
| Pointing (.86) | Queries are written by the same LLM teacher, in the training style. Only 5–8 candidates per program. |
| Same-problem retrieval (.54) | No embedder or lexical baseline. Two solutions to one contest problem share input-parsing code. |
| All | One seed. No contrastive (InfoNCE) baseline at equal budget. |
| Compression | Untested. The LM has never read node vectors and gives the node tokens no probability. |

## 3. Runs, in order

Rules for all of them:

- Fix the pass rule before looking at results, as in Phase 2 (`scripts/diag_summary.py` style). Write it into
  this file under the run.
- Report every result, including the ones that fail. Add them to a dated report in `docs/`.
- R1 and R2 are evaluation only and can start. R3–R5 train models: get the owner's go-ahead first
  (`AGENTS.md`: no long runs without being asked).

### R1. Baseline check (evaluation only, do first)

Question: do the pointing and same-problem results survive strong baselines?

Add to report §2 (all four in-program rows) and §1 (same-problem rows):

- **gte-modernbert, per-span:** encode each block or function's text on its own.
- **gte-modernbert, pooled spans:** encode the whole program once, mean-pool token states over each span's lines.
- **UniXcoder**, both ways.
- **BM25** over code tokens, as the lexical floor. For same-problem retrieval this measures the shared
  input-parsing shortcut.
- One strong code embedder that loads without `trust_remote_code`, if one exists.

Also re-run same-problem retrieval on the identifier-anonymised copies (`bench/variants.py`).

Pass: `free` beats the best baseline on at least 3 of the 4 in-program rows by more than the seed spread
(needs the seeds from R3a), and on same-problem retrieval. If it fails on pointing, stop and report: the
framing in §1 needs rethinking.

### R2. Independent queries (evaluation only)

Question: is the pointing result a match to the teacher's writing style?

- Re-write the evaluation descriptions with a **different LLM** from the training annotator, with a different
  prompt. Evaluation sets only.
- **Human docstrings:** function docstring → its function, among the functions of the same file. Remove the
  docstring from the code before encoding, or it leaks.
- Report at larger candidate sets too: all blocks of 10 and of 100 pooled programs, not just one program.

Pass: `free` keeps a clear margin over the best R1 baseline on the independent queries.

### R3. One retrain with the cheap fixes (training)

a. **Seeds.** Export seeds 23 and 37 (`scripts/export_release.py`) and run the full benchmark on them. No
   training needed; do this alongside R1.

b. **Rename augmentation.** On the code view only, replace identifiers with fresh names on a random fraction
   of training steps. Descriptions stay unchanged. Renaming keeps line spans, so the packer is unaffected.
   Reformatting moves lines; add it only if spans are remapped correctly. Not implemented yet.

c. **More data.** Train on CodeContests + TACO + KodCode (about 100k programs). Two things first:
   - The cluster re-split (`AGENTS.md`: 5,034 of 6,365 CodeContests problems also appear in TACO).
   - **The benchmark's "unseen" TACO and KodCode pools become training data unless held out.** Hold the
     evaluation pools out by problem cluster, and rebuild them with `bench/prep_pools.py` if they change.

Run b and c as separate arms on `free` first (same recipe as `runs/p4/e4_allcode_s82`), then combined.
Train a matching E1 for each data setting.

Pass for b: the rename-vs-mutant rate (report §3, currently 3%) and anonymised AdvTest (.10) improve clearly,
with pointing R@1 within .02 of the current model.
Pass for c: MBPP+ and CodeSearchNet perplexity recover toward E0, and SWE-bench Lite moves above .15.

### R4. Compression feasibility probe (training, small)

Question: can the LM use a node vector in place of the code it stands for?

Setup:

- Freeze the released `free` encoder. Take each node's `raw` vector (896-d, the backbone's hidden size).
- Reader: the same backbone with a LoRA and a linear adapter that maps a `raw` vector to one input embedding.
- Use multi-function programs only (most programs have one function; see `AGENTS.md` terminology).

Tasks, cheapest first:

1. **Sanity:** generate a node's description from its vector alone.
2. **Completion with compressed context:** predict one function's body, given the other functions as
   context. Metric: perplexity on the target body, then test pass rate.

Context conditions, compared at the same context-token budget:

| Condition | Context for each other function |
|---|---|
| Full text (upper bound) | its code |
| None (lower bound) | nothing |
| Truncation | the first tokens that fit the budget |
| Text description | its LLM-written unit description |
| **JEPA vector** | 1 unit vector (also: its step vectors) |
| Reconstruction vector | 1 vector from the same architecture trained to reconstruct the code (ICAE-style) |
| E1 vector | 1 vector from the LM-only fine-tune, same readout |

Pass: the JEPA vector closes a clear share of the gap between "none" and "full text", and beats the
reconstruction and E1 vectors at the same budget. Report the curve of quality against context tokens.

If it passes in-domain, move to repository-level completion (CrossCodeEval, RepoBench) with cross-file
functions given as vectors. That is the headline experiment.

### R5. InfoNCE baseline (training)

Question: why JEPA and not contrastive?

Same data, spans, backbone, LM loss and step budget as `free`. Replace the cosine view loss and SIGReg with
InfoNCE over in-batch nodes of the same level. This is not the unimplemented B1 in `AGENTS.md` (frozen text
targets, no LM); keep the LM loss so the comparison isolates the objective. Phase 2b's C3 learned steps only,
so tune temperature and batch size before concluding. Run the full benchmark plus the R4 probe on it.

### Literature check (no compute)

Before writing: search for prior work on JEPA-trained vectors used as context compression, and on
span-level or multi-granularity code embeddings from one pass. Check against ICAE, gist tokens, xRAG,
AutoCompressor, late chunking and late-interaction retrievers. Not done yet.

## 4. Decision

| R1 + R2 (pointing holds) | R4 (compression works) | Paper |
|---|---|---|
| yes | yes | Full claim in §1: retrieve with the vector, then read it. |
| yes | no | Localisation paper: one-pass span-grounded vectors. Solid but more incremental. |
| no | yes | Compression paper; pointing becomes a diagnostic. |
| no | no | Stop and rethink before more training. |

Scaling to the 1.5B backbone comes after this table is filled in, not before.
