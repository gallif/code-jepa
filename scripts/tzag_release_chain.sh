#!/usr/bin/env bash
set -eu
R=/mnt/storage/users/glifshitz/code-jepa
cd $R/repo
source scripts/tzag_env.sh
export PYTHONPATH=$R/repo/src CUDA_VISIBLE_DEVICES=0
mkdir -p $R/release
echo "[rel] export tree $(date)"
$CODE_JEPA_ENV/bin/python scripts/export_release.py --variant tree --config configs/tzag_p3_e4_s82.yaml \
  --checkpoint $R/runs/p3/e4_s82/last.pt --metrics $R/runs/p3/e4_s82/metrics.jsonl --out $R/release/code-jepa-0.5b-tree-s82
echo "[rel] export free $(date)"
$CODE_JEPA_ENV/bin/python scripts/export_release.py --variant free --config configs/tzag_p4_e4_allcode_s82.yaml \
  --checkpoint $R/runs/p4/e4_allcode_s82/last.pt --metrics $R/runs/p4/e4_allcode_s82/metrics.jsonl --out $R/release/code-jepa-0.5b-free-s82
for v in tree free; do
  echo "[rel] verify $v $(date)"
  $CODE_JEPA_ENV/bin/python scripts/verify_release.py --model $R/release/code-jepa-0.5b-$v-s82
done
echo "[rel] done $(date)"
