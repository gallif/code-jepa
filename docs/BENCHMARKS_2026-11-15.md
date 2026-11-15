# Code-JEPA 0.5B — benchmark report (2026-11-15)

Where do the released Code-JEPA encoders beat the model they started from? And what did training cost the language model?

**Models.** All four share the Qwen2.5-Coder-0.5B architecture. The JEPA models are the phase-2 releases on tzag (`release/code-jepa-0.5b-{free,tree}-s82`).

| Name | What it is |
|---|---|
| **free** | Code-JEPA. Program-level tokens read all code tokens. |
| **tree** | Code-JEPA. Program-level tokens read only the function tokens. |
| **E1** | Same training data, next-token loss only, no alignment objective. |
| **E0** | Qwen2.5-Coder-0.5B as released. |

**Fairness.**
- E0 and E1 get two pooled readouts: the mean over each node's own code lines, from the middle and from the last layer. The table always shows **E0's best readout**. Our models use a fixed readout.
- Scores are R@1 unless marked MRR. One seed (82) per model.

**Data hygiene.**
- Training used our edited, annotated variant of CodeContests: 11,448 train programs with tree-sitter trees and LLM-written descriptions.
- Every evaluation set is unseen:
  - CodeContests val/test, minus the 26 programs that repeat a training statement or solution (203 programs left).
  - TACO, with problems or solutions matching CodeContests train removed (8,000 programs, 4,000 problems).
  - KodCode test.
  - External sets: SWE-bench Lite, AdvTest, HumanEval+, MBPP+, CRUXEval and MultiPL-E.

## TL;DR

| Claim | Verdict |
|---|---|
| Matches solutions by what they compute, not how they're written | ✅ **Yes** for same-problem retrieval (.54 vs .28). ➖ Tie on algorithm tags. |
| Points to the block or function a description refers to | ✅ **Yes** inside a program (.86 vs ≤ .70). ❌ **No** on real repos (SWE-bench ≈ E0, far below BM25). |
| Ignores renaming and reformatting | ❌ **No.** Every model, ours included, is dominated by identifiers. |
| One model for text ↔ code search | ✅ Far better than E0. ❌ A general embedding model (gte-modernbert) wins every task. |
| JEPA objective preserves the LM | ✅ Drifts **less** from E0 than the LM-only E1 does. ⚠️ All fine-tunes lose general LM quality (MBPP −18). |

---

## 1. Matching code by algorithm

TACO, 8,000 programs. All rows are zero-shot cosine on centered vectors, except the two probe rows (3-fold, grouped by problem).

| | free | tree | E1 | E0 (best) |
|---|---|---|---|---|
| The other solution of the same problem ranks first (approach vector) | **.54** | .07 | .02 | .28 |
| Same, using the intent vector | **.63** | .07 | .02 | .28 |
| Tag overlap of nearest neighbours from *other* problems, ÷ random | **2.92×** | 1.99× | 1.80× | 2.64× |
| Algorithm-tag probe (12 tags), macro AUC | .82 | .69 | .67 | .81 |
| Complexity-class probe, macro-F1 † | **.58** | .27 | .22 | .53 |

† The complexity labels come from the same LLM teacher that wrote the training descriptions.

Tree's program-level tokens work in-domain but barely transfer to TACO.

## 2. Pointing to the location in the code

The query is a block's or function's description. Candidates are the blocks or functions of the same program (or of the edited files, for SWE-bench).

| | free | tree | E1 | E0 (best) | chance |
|---|---|---|---|---|---|
| Step description → its block: CodeContests | **.86** | .81 | .60 | .58 | .12 |
| Step description → its block: TACO | **.88** | .84 | .71 | .70 | .15 |
| Function description → its function: KodCode | **.86** | .77 | .43 | .49 | .20 |
| Function description → its function: TACO | **.86** | .78 | .61 | .62 | .19 |
| SWE-bench Lite: issue → edited function (272 issues) | .15 | .15 | .15 | .15 | .06 |

- **Ours:** zero-shot in the shared 256-d space, or a ridge probe on the raw vectors. **E0 / E1:** a ridge probe trained on other TACO problems.
- The in-program descriptions are LLM-written in the training style, which favours our models.
- SWE-bench references: BM25 .32, UniXcoder .21, gte-modernbert .39.

## 3. Robustness to renaming and reformatting

703 programs. Each variant was checked by running the tests: renamed and reformatted copies still pass, and one-token mutants are killed.

| | free | E0 (best) |
|---|---|---|
| Correct renamed copy is closer to the original than a broken one-token mutant | 3% | 0% |
| Correct reformatted copy is closer than a broken mutant | 28% | 14% |
| The step that moved most is the mutated one (chance 20%) | 91% | 88% |

**AdvTest** (6,000 functions; docstring → function). MRR before → after identifiers are replaced with `Func` / `arg_i`:

| free | tree | E0 (best) | UniXcoder | gte-modernbert |
|---|---|---|---|---|
| .40 → .10 | .37 → .05 | .13 → .07 | .51 → .35 | .79 → .50 |

## 4. One model for text and code, vs dedicated embedders (MRR)

| | free | tree | E0 (best) | UniXcoder | gte-modernbert |
|---|---|---|---|---|---|
| Problem statement → its solutions, TACO | .32 | .00 | .10 | .12 | **.67** |
| Problem statement → its solutions, CodeContests eval | .66 | .11 | .29 | .16 | **.90** |
| AdvTest, original identifiers | .40 | .37 | .13 | .51 | **.79** |

jina-code and CodeRankEmbed were not tested: they need `trust_remote_code`.

## 5. Language-model drift vs the original model

Teacher-forced on the same tokens. KL is in nats per token, over the shared vocabulary.

| Corpus | E0 ppl | free ppl | tree ppl | E1 ppl | KL(E0‖M): free / tree / E1 |
|---|---|---|---|---|---|
| CodeContests eval code | 2.47 | 2.40 | 2.39 | 2.40 | .147 / **.134** / .162 |
| Training format (intent + approach → code) | 2.33 | 2.12 | 2.11 | 2.12 | .291 / **.272** / .296 |
| TACO code | 2.40 | 2.47 | 2.45 | 2.46 | .115 / **.106** / .131 |
| CodeSearchNet Python | 5.61 | 7.42 | 7.37 | 7.55 | .270 / **.258** / .307 |
| WikiText-2 | 12.8 | 17.1 | 16.8 | 17.1 | .284 / **.265** / .283 |

- Top-1 next-token agreement with E0 is 0.75–0.92.
- The new node tokens get essentially no probability (below 2×10⁻⁸).
- **The JEPA objective adds no drift beyond the LM fine-tune.** The roughly 30% perplexity increase on general Python and English comes from fine-tuning on CodeContests, since E1 has it too.

## 6. Code generation (greedy pass@1, %)

| | HumanEval | HE+ | MBPP | MBPP+ | CRUX-O | CRUX-I | C++ | JS | CodeContests | CodeContests, guided ‡ |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | 26.8 | **22.6** | **55.0** | **48.4** | **25.0** | **26.5** | **24.2** | **27.3** | 0.9 | 0.0 |
| E1 | 23.8 | 18.3 | 38.1 | 33.1 | 20.1 | 21.5 | 21.7 | 13.7 | **3.6** | 9.0 |
| tree | **27.4** | 22.0 | 34.9 | 29.1 | 19.8 | 23.1 | 20.5 | 16.8 | 2.7 | 8.1 |
| free | 25.6 | 20.7 | 37.3 | 31.0 | 18.6 | 22.0 | 21.7 | 14.9 | 0.9 | **9.9** |

- HumanEval/MBPP are scored with EvalPlus. C++ and JS are MultiPL-E HumanEval. CodeContests uses 111 problems with stdin/stdout tests.
- ‡ The "guided" column uses the training prompt (gold intent + approach), so it is not a standard metric.
- E0's HumanEval score (26.8) is close to Qwen's published 28.0.
- **Why MBPP drops:** fine-tuned models often ignore the function name required by the test. 60–87 of 378 MBPP outputs use the wrong name, vs 17 for E0. This comes from training on stdin/stdout CodeContests (E1 has it too), and it looks fixable.

## Caveats

- One seed per model. Seeds 23 and 37 can be exported and added.
- The rename-vs-mutant test pits renaming every identifier against changing one token, which is deliberately hard. Reformatting is the gentler version.
- AdvTest uses 6,000 of the 9,826 test functions that could be matched to their original source.
- LiveCodeBench was not run: its private tests are pickled, and loading them would execute downloaded code.

## Reproduce (on tzag)

Benchmark code: [`bench/`](../bench/) in this repo (see its README). On tzag the working copy is `/mnt/storage/users/glifshitz/code-jepa/bench/`.

Results JSON: `runs/bench/results/{b1,b2,b3,b4,lm,codegen}.json`.

```bash
cd /mnt/storage/users/glifshitz/code-jepa && source repo/scripts/tzag_env.sh && export CUDA_VISIBLE_DEVICES=2
PYG=/mnt/storage/initzan_cache/envs/hjepa_gen/bin/python      # tree-sitter + test executor
python bench/prep_pools.py                                    # unseen eval pools
$PYG bench/variants.py                                        # test-checked variants / mutants
python bench/encode_pool.py --pool data/bench/taco_clean.jsonl   # + cc_eval, kodcode_test, variants, adv_*, swe_funcs
python bench/encode_external.py --pool ...                    # UniXcoder / gte-modernbert
python bench/evaluate.py all                                  # benchmarks 1-4
python bench/lm_eval.py --models free tree E1                 # perplexity + KL vs E0
python bench/codegen.py && bash bench/score_evalplus.sh E0 E1 tree free && $PYG bench/score_gen.py
```
