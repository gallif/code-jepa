#!/usr/bin/env bash
# Phase 4 on GPU 0, strictly sequential: probes (tree packing, then E2 packing), then the two E4 controls.
set -u
R=/mnt/storage/users/glifshitz/code-jepa
cd $R/repo
source scripts/tzag_env.sh
export PYTHONPATH=$R/repo/src CUDA_VISIBLE_DEVICES=0
P=$R/runs/p3
M_ALL="--model E0=pretrained"
for s in 82 23 37; do for e in e1 e2 e4; do M_ALL="$M_ALL --model ${e^^}_s$s=$P/${e}_s$s/last.pt"; done; done
echo "[p4] probes tree $(date)"
$CODE_JEPA_ENV/bin/python scripts/probe_eval.py --config configs/tzag_p3_e4_s82.yaml --out $R/runs/p4/probes_tree.json $M_ALL
echo "[p4] probes e2-packing $(date)"
$CODE_JEPA_ENV/bin/python scripts/probe_eval.py --config configs/tzag_p3_e2_s82.yaml --out $R/runs/p4/probes_e2pack.json $M_ALL
echo "[p4] controls $(date)"
bash scripts/tzag_sweep.sh --spec configs/p4_jobs.json --gpus 0
echo "[p4] all done $(date)"
