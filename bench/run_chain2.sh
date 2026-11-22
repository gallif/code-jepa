#!/usr/bin/env bash
set -uo pipefail
cd /mnt/storage/users/glifshitz/code-jepa
source repo/scripts/tzag_env.sh
export CUDA_VISIBLE_DEVICES=2
while pgrep -f "bench/run_queue.sh" >/dev/null; do sleep 30; done
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
python bench/encode_external.py --pool data/bench/swe_funcs.jsonl --queries data/bench/swe_meta.jsonl
python bench/encode_external.py --pool data/bench/adv_norm.jsonl --queries data/bench/adv_norm.jsonl
python bench/encode_external.py --pool data/bench/adv_orig.jsonl --queries data/bench/adv_orig.jsonl
python bench/encode_external.py --pool data/bench/taco_clean.jsonl --prompt
python bench/encode_external.py --pool data/bench/cc_eval.jsonl --prompt
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python bench/lm_eval.py --models free tree E1
python bench/codegen.py --models E0 E1 tree free
echo CHAIN2_DONE
