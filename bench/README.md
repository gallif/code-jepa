# bench/ — Code-JEPA benchmark suite

These scripts produced [`docs/BENCHMARKS_2026-11-15.md`](../docs/BENCHMARKS_2026-11-15.md). They compare the released
`free` / `tree` encoders with E0 (Qwen2.5-Coder-0.5B as released) and E1 (the LM-only fine-tune).

The paths are tzag-specific. The scripts expect to run from `/mnt/storage/users/glifshitz/code-jepa`, with the code-jepa env
(`repo/scripts/tzag_env.sh`) and `CUDA_VISIBLE_DEVICES=2`.
- Scripts marked *hjepa_gen* need initzan's env (`/mnt/storage/initzan_cache/envs/hjepa_gen`), which has tree-sitter and
  the sandboxed test executor.
- EvalPlus runs from its own venv (`envs/evalplus`, `evalplus==0.3.1`).

Data goes to `data/bench/`, vector caches to `runs/bench/cache/`, and results to `runs/bench/results/*.json`.

| Script | Role |
|---|---|
| `prep_pools.py` | Unseen eval pools: CodeContests val/test minus training-statement/solution overlap, TACO minus CodeContests-train overlap, KodCode test |
| `trees.py` *(hjepa_gen)* | Code-only hierarchy records (tree-sitter units / steps with spans, empty descriptions) for new code |
| `variants.py` *(hjepa_gen)* | Renamed / reformatted variants and one-token mutants, each labelled by running the tests |
| `prep_external.py` | AdvTest (normalised + original identifiers, docstrings stripped) and SWE-bench Lite candidate functions |
| `encoders.py` | One interface for all models: node-token readouts (`node_raw`, `node_head`) and span-pooled readouts (`pool_last`, `pool_mid`); text queries via the level's predictor token |
| `encode_pool.py` | Caches node vectors, description vectors and query vectors for one pool and several models |
| `external.py`, `encode_external.py` | Dedicated embedders (UniXcoder, gte-modernbert); only models that load without `trust_remote_code` |
| `evaluate.py` | Benchmarks 1–4: algorithm matching, localization, surface robustness, text→code search |
| `lm_eval.py` | Perplexity and token-level KL vs E0 over the shared vocabulary |
| `codegen.py` | Greedy generation: HumanEval+, MBPP+, CRUXEval-I/O, MultiPL-E C++/JS, CodeContests |
| `score_evalplus.sh`, `score_gen.py` *(hjepa_gen)* | pass@1 scoring |
| `run_queue.sh`, `run_chain2.sh` | The GPU job chains used for the 2026-11-15 run |

Full rerun: see the "Reproduce" section of the report.
