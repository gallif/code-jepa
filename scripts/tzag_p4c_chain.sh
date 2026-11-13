#!/usr/bin/env bash
# Phase 4c on GPU 0: waits for chain_b, trains the hybrid E4 (seed 82), then probes it under three readouts:
# hybrid (units + code), all_code (unit tokens blanked from the program tokens) and tree (raw code blanked).
set -u
R=/mnt/storage/users/glifshitz/code-jepa
cd $R/repo
while kill -0 "$(cat $R/runs/p4/chain_b.pid)" 2>/dev/null; do sleep 60; done
source scripts/tzag_env.sh
export PYTHONPATH=$R/repo/src CUDA_VISIBLE_DEVICES=0
echo "[p4c] training hybrid $(date)"
bash scripts/tzag_sweep.sh --spec configs/p4c_jobs.json --gpus 0
H="--model HYB_s82=$R/runs/p4/e4_hybrid_s82/last.pt"
for cfg in tzag_p4_e4_hybrid_s82 tzag_p4_e4_allcode_s82 tzag_p3_e4_s82; do
  echo "[p4c] probes $cfg $(date)"
  $CODE_JEPA_ENV/bin/python scripts/probe_eval.py --config configs/$cfg.yaml --out $R/runs/p4/probes_hybrid_readout_$cfg.json $H
done
echo "[p4c] all done $(date)"
