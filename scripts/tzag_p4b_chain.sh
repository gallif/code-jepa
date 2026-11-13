#!/usr/bin/env bash
# Phase 4b on GPU 0: free-program-token E4 for seeds 23 and 37, then probes under the all_code packing.
set -u
R=/mnt/storage/users/glifshitz/code-jepa
cd $R/repo
source scripts/tzag_env.sh
export PYTHONPATH=$R/repo/src CUDA_VISIBLE_DEVICES=0
echo "[p4b] training $(date)"
bash scripts/tzag_sweep.sh --spec configs/p4b_jobs.json --gpus 0
M="--model E0=pretrained"
for s in 82 23 37; do
  for e in e1 e2 e4; do M="$M --model ${e^^}_s$s=$R/runs/p3/${e}_s$s/last.pt"; done
  M="$M --model E4FREE_s$s=$R/runs/p4/e4_allcode_s$s/last.pt"
done
echo "[p4b] probes all_code packing $(date)"
$CODE_JEPA_ENV/bin/python scripts/probe_eval.py --config configs/tzag_p4_e4_allcode_s82.yaml --out $R/runs/p4/probes_allcode.json $M
echo "[p4b] all done $(date)"
