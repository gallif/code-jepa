#!/usr/bin/env bash
# Copy only the source tree to tzag storage. Does not copy data, models, or env.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${TZAG_HOST:-132.66.150.42}:/mnt/storage/users/glifshitz/code-jepa/repo/"
rsync -az --delete \
  --exclude '.git/' \
  --exclude '__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude '*.pt' \
  --exclude '*.pyc' \
  --exclude '*.egg-info/' \
  "$ROOT/src/" "$DEST/src/" 
rsync -az --delete --exclude '__pycache__/' --exclude '*.pyc' \
  "$ROOT/scripts/" "$DEST/scripts/"
rsync -az --delete \
  "$ROOT/configs/" "$DEST/configs/"
rsync -az --delete --exclude '__pycache__/' \
  "$ROOT/tests/" "$DEST/tests/"
rsync -az "$ROOT/pyproject.toml" "$ROOT/README.md" "$ROOT/environment.yml" "$DEST/"
echo "synced -> $DEST"
