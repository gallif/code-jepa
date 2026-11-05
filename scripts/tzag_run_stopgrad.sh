#!/usr/bin/env bash
# E1 vs E2 vs E4 at 4 epochs with the anti-collapse recipe (stop-grad, SIGReg bank, μ=0.3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/tzag_env.sh"

RUNS="${CODE_JEPA_ROOT}/runs/stopgrad"
mkdir -p "$RUNS"/{e1,e2,e4}
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

mapfile -t FREE < <(
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2+0 >= 40000) print $1}'
)
echo "free GPUs (>=40GB): ${FREE[*]:-none}"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv
if (( ${#FREE[@]} < 3 )); then
  echo "need at least 3 idle L40S with 40GB free" >&2
  exit 1
fi

G0="${FREE[0]}"
G1="${FREE[1]}"
G2="${FREE[2]}"
PY="${CODE_JEPA_ENV}/bin/python"

launch() {
  local gpu="$1" name="$2" cfg="$3"
  echo "[$(date -Is)] GPU ${gpu} ${name} ${cfg}" | tee -a "${RUNS}/launch.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/train.py" --config "$cfg" \
    >"${RUNS}/${name}.stdout.log" 2>&1 &
  echo $! >"${RUNS}/${name}.pid"
}

launch "$G0" e1 "$ROOT/configs/tzag_e1_stopgrad.yaml"
launch "$G1" e2 "$ROOT/configs/tzag_e2_stopgrad.yaml"
launch "$G2" e4 "$ROOT/configs/tzag_e4_stopgrad.yaml"

echo "launched E1 on GPU ${G0}, E2 on GPU ${G1}, E4 on GPU ${G2}"
echo "logs: ${RUNS}/{e1,e2,e4}.stdout.log"
echo "ckpts: ${RUNS}/{e1,e2,e4}/"
