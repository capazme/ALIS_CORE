#!/usr/bin/env bash
# Cleanup helper for ALIS dev environment.
# Kills processes bound to known service ports and can optionally clear caches.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

DEFAULT_PORTS=(5173 3001 5000 8000 5432 6379 6333 5174 3000)
CACHE=false
DRY_RUN=false
CUSTOM_PORTS=()

print_usage() {
  echo "Usage: ./cleanup.sh [options]"
  echo ""
  echo "Options:"
  echo "  --ports \"p1,p2\"   Comma-separated list of ports to clean"
  echo "  --cache          Also clear tool caches (docker/npm/pip)"
  echo "  --dry-run        Show what would be killed"
  echo "  -h, --help       Show this help"
  echo ""
  echo "Default ports: ${DEFAULT_PORTS[*]}"
}

parse_args() {
  while [ $# -gt 0 ]; do
    case "$1" in
      --ports)
        shift
        IFS=',' read -r -a CUSTOM_PORTS <<< "${1:-}"
        ;;
      --cache)
        CACHE=true
        ;;
      --dry-run)
        DRY_RUN=true
        ;;
      -h|--help)
        print_usage
        exit 0
        ;;
      *)
        echo "Unknown option: $1"
        print_usage
        exit 1
        ;;
    esac
    shift
  done
}

kill_port() {
  local port="$1"
  local pids=""

  pids="$(lsof -ti TCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [ -z "$pids" ]; then
    echo "Port $port: free"
    return
  fi

  for pid in $pids; do
    if [ "$DRY_RUN" = "true" ]; then
      echo "Port $port: would kill PID $pid"
      continue
    fi

    echo "Port $port: stopping PID $pid"
    kill -TERM "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "Port $port: force killing PID $pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  done
}

clear_caches() {
  if [ "$DRY_RUN" = "true" ]; then
    echo "Cache: would clear docker/npm/pip caches"
    return
  fi

  if command -v docker >/dev/null 2>&1; then
    echo "Cache: docker builder prune"
    docker builder prune -f >/dev/null 2>&1 || true
  fi

  if command -v npm >/dev/null 2>&1; then
    echo "Cache: npm cache clean"
    npm cache clean --force >/dev/null 2>&1 || true
  fi

  if command -v pip >/dev/null 2>&1; then
    echo "Cache: pip cache purge"
    pip cache purge >/dev/null 2>&1 || true
  fi
}

parse_args "$@"

if [ ${#CUSTOM_PORTS[@]} -gt 0 ]; then
  PORTS=("${CUSTOM_PORTS[@]}")
else
  PORTS=("${DEFAULT_PORTS[@]}")
fi

echo "Cleanup ports: ${PORTS[*]}"
for port in "${PORTS[@]}"; do
  kill_port "$port"
done

if [ "$CACHE" = "true" ]; then
  clear_caches
fi

echo "Cleanup complete."
