#!/usr/bin/env bash
# Sync local checkout to GPU machine (tzag).
set -euo pipefail
HOST="${CODE_JEPA_HOST:-132.66.150.42}"
REMOTE_REPO="${CODE_JEPA_REMOTE_REPO:-repos/code-jepa}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TGZ="/tmp/code-jepa-sync.tgz"

tar czf "$TGZ" -C "$ROOT" \
  --exclude .git \
  --exclude '__pycache__' \
  --exclude .venv \
  --exclude data \
  --exclude cache \
  --exclude runs \
  --exclude models \
  .

ssh "$HOST" "cat > /tmp/code-jepa.tgz" < "$TGZ"
ssh "$HOST" "mkdir -p ~/${REMOTE_REPO} && cd ~/${REMOTE_REPO} && tar xzf /tmp/code-jepa.tgz && rm -f /tmp/code-jepa.tgz"
rm -f "$TGZ"

echo "Synced to ${HOST}:~/${REMOTE_REPO}"
echo "On remote: conda activate /mnt/storage/users/glifshitz/code-jepa/env"
echo "           export CUDA_VISIBLE_DEVICES=2,3"
echo "           export CODE_JEPA_ROOT=/mnt/storage/users/glifshitz/code-jepa"
echo "           export HF_HOME=\$CODE_JEPA_ROOT/models/hf"
