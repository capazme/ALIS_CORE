#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "Python 3.10+ is required."
  exit 1
fi

if command -v git >/dev/null 2>&1 && git lfs version >/dev/null 2>&1; then
  echo "Pulling Git LFS assets..."
  git lfs pull
fi

echo "Installing merlt-models (editable)..."
python -m pip install -e .

python - <<'PY'
from merlt_models import list_models, list_configs

models = list_models()
configs = list_configs()

print("Available models:", models if models else "none found")
print("Available configs:", configs if configs else "none found")
PY
