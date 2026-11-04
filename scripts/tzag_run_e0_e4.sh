#!/usr/bin/env bash
# Launch E0–E4 on free tzag GPUs. E3 (chain) is not implemented; E0 is eval-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/tzag_env.sh"

RUNS="${CODE_JEPA_ROOT}/runs/e0_e4"
mkdir -p "$RUNS"/{e0,e1,e2,e4}
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

launch_log() {
  local gpu="$1" name="$2"
  shift 2
  echo "[$(date -Is)] GPU ${gpu} ${name} $*" | tee -a "${RUNS}/launch.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$ROOT/scripts/train.py" "$@"
}

# GPU 0: zero-shot eval, then LM-only train.
(
  launch_log "$G0" e0 --config "$ROOT/configs/tzag_e0.yaml" --eval-only
  launch_log "$G0" e1 --config "$ROOT/configs/tzag_e1.yaml"
) >"${RUNS}/e0_e1.stdout.log" 2>&1 &
echo $! >"${RUNS}/e0_e1.pid"

# GPU 1: flat LLM-JEPA.
(
  launch_log "$G1" e2 --config "$ROOT/configs/tzag_e2.yaml"
) >"${RUNS}/e2.stdout.log" 2>&1 &
echo $! >"${RUNS}/e2.pid"

# GPU 2: hierarchical LLM-JEPA.
(
  launch_log "$G2" e4 --config "$ROOT/configs/tzag_e4.yaml"
) >"${RUNS}/e4.stdout.log" 2>&1 &
echo $! >"${RUNS}/e4.pid"

echo "launched E0→E1 on GPU ${G0}, E2 on GPU ${G1}, E4 on GPU ${G2}"
echo "logs: ${RUNS}/*.stdout.log"
echo "tail: tail -f ${RUNS}/e4.stdout.log"
