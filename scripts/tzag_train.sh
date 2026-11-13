#!/usr/bin/env bash
# Run on tzag. Weights/data are read in place; checkpoints go to storage.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/tzag_env.sh"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
exec python "$ROOT/scripts/train.py" \
  --config "$ROOT/configs/remote_tzag.yaml" \
  "$@"
