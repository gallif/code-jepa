#!/usr/bin/env bash
# Queue the E4 hyperparameter sweep on idle L40S GPUs (>=40GB free).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$ROOT/scripts/tzag_env.sh"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${CODE_JEPA_ENV}/bin/python" "$ROOT/scripts/tzag_sweep_driver.py" "$@"
