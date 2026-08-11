#!/usr/bin/env bash
# Sync local checkout to GPU machine and remind CUDA device binding.
set -euo pipefail
HOST="${CODE_JEPA_HOST:-132.66.150.42}"
REMOTE_REPO="${CODE_JEPA_REMOTE_REPO:-~/repos/code-jepa}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv' \
  --exclude 'data' \
  --exclude 'cache' \
  --exclude 'runs' \
  --exclude 'models' \
  "$ROOT/" "${HOST}:${REMOTE_REPO}/"

echo "Synced to ${HOST}:${REMOTE_REPO}"
echo "On remote, use: export CUDA_VISIBLE_DEVICES=2,3"
echo "Storage: /mnt/storage/users/glifshitz/code-jepa"
