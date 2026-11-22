#!/usr/bin/env bash
# Encode the new-code pools (code-only trees) after the hierarchy pools finish. GPU 2 only.
set -uo pipefail
cd /mnt/storage/users/glifshitz/code-jepa
source repo/scripts/tzag_env.sh
export CUDA_VISIBLE_DEVICES=2
while pgrep -f "bench/encode_pool.py --pool data/bench/(taco_clean|kodcode_test)" >/dev/null; do sleep 30; done
python bench/encode_pool.py --pool data/bench/variants.jsonl --no-text
python bench/encode_pool.py --pool data/bench/adv_norm.jsonl --no-text --queries data/bench/adv_norm.jsonl --query-levels unit intent
python bench/encode_pool.py --pool data/bench/adv_orig.jsonl --no-text --queries data/bench/adv_orig.jsonl --query-levels unit intent
python bench/encode_pool.py --pool data/bench/swe_funcs.jsonl --no-text --queries data/bench/swe_meta.jsonl --query-levels unit intent --chunk 256
echo QUEUE_DONE
