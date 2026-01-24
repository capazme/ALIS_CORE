#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! command -v python >/dev/null 2>&1; then
  echo "Python 3.10+ is required."
  exit 1
fi

if ! python -c "import bs4" >/dev/null 2>&1; then
  echo "Installing visualex dependencies..."
  python -m pip install -e .
fi

echo "Starting VisuaLex API on http://localhost:5000"
exec python -m visualex.app
