#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required to start the platform stack."
  echo "Install Docker Desktop and retry."
  exit 1
fi

echo "Starting frontend, backend, python-api, and postgres..."
exec docker compose up --build frontend backend python-api postgres
