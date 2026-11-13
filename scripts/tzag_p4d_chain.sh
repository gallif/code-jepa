#!/usr/bin/env bash
# Phase 4d on GPU 0: shuffled-description and program-level-grounding controls on free-token E4 (seed 82),
# then probes under the all_code packing (comparable with runs/p4/probes_allcode.json).
set -u
R=/mnt/storage/users/glifshitz/code-jepa
cd $R/repo
source scripts/tzag_env.sh
export PYTHONPATH=$R/repo/src CUDA_VISIBLE_DEVICES=0
echo "[p4d] training $(date)"
bash scripts/tzag_sweep.sh --spec configs/p4d_jobs.json --gpus 0
echo "[p4d] probes $(date)"
$CODE_JEPA_ENV/bin/python scripts/probe_eval.py --config configs/tzag_p4_e4_allcode_s82.yaml --out $R/runs/p4/probes_controls.json \
  --model E4FREE_s82=$R/runs/p4/e4_allcode_s82/last.pt \
  --model SHUF_s82=$R/runs/p4/e4free_shuffle_s82/last.pt \
  --model PROG_s82=$R/runs/p4/e4free_progground_s82/last.pt
echo "[p4d] all done $(date)"
