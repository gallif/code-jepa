# Source on tzag. Keeps HuggingFace / pip / torch caches off the root disk.
export CODE_JEPA_ROOT=/mnt/storage/users/glifshitz/code-jepa
export CODE_JEPA_REPO="$CODE_JEPA_ROOT/repo"
export CODE_JEPA_ENV="$CODE_JEPA_ROOT/env"

export HF_HOME="$CODE_JEPA_ROOT/models/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$CODE_JEPA_ROOT/cache/hf_datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export PIP_CACHE_DIR="$CODE_JEPA_ROOT/cache/pip"
export TORCH_HOME="$CODE_JEPA_ROOT/cache/torch"
export XDG_CACHE_HOME="$CODE_JEPA_ROOT/cache/xdg"
export TMPDIR="$CODE_JEPA_ROOT/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$XDG_CACHE_HOME" "$TMPDIR" \
  "$HF_DATASETS_CACHE" "$CODE_JEPA_ROOT/runs"

export PATH="$CODE_JEPA_ENV/bin:$PATH"
export PYTHONPATH="$CODE_JEPA_REPO/src${PYTHONPATH:+:$PYTHONPATH}"
