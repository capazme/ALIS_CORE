#!/usr/bin/env bash
#
# ALIS Development Environment Launcher
# =====================================
# Entry point unificato per l'ecosistema ALIS
#
# Componenti:
#   visualex-platform  - Frontend React + Backend Express + Python API
#   visualex-merlt     - Plugin MERL-T + RLCF Web
#   visualex-api       - Libreria Python scraping (standalone)
#   merlt              - Framework ML (Expert + RLCF + Knowledge Graph)
#
# Porte:
#   5173  - Frontend (Vite dev server)
#   3001  - Backend Express
#   5000  - Python API (visualex)
#   8000  - MERL-T API (FastAPI)
#   5432  - PostgreSQL
#   6379  - FalkorDB (Redis protocol)
#   6333  - Qdrant
#

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

ROOT="$(cd "$(dirname "$0")" && pwd)"
DOCKER_CONFIG_DIR="$ROOT/.docker"
DOCKER_CONFIG_FILE="$DOCKER_CONFIG_DIR/config.json"
COMPOSE_CMD=()
cd "$ROOT"

# ============================================================================
# Helper Functions
# ============================================================================

print_banner() {
  echo -e "${CYAN}"
  echo "╔═══════════════════════════════════════════════════════════════╗"
  echo "║                                                               ║"
  echo "║     █████╗ ██╗     ██╗███████╗                               ║"
  echo "║    ██╔══██╗██║     ██║██╔════╝                               ║"
  echo "║    ███████║██║     ██║███████╗                               ║"
  echo "║    ██╔══██║██║     ██║╚════██║                               ║"
  echo "║    ██║  ██║███████╗██║███████║                               ║"
  echo "║    ╚═╝  ╚═╝╚══════╝╚═╝╚══════╝                               ║"
  echo "║                                                               ║"
  echo "║    Artificial Legal Intelligence System                       ║"
  echo "║    Development Environment                                    ║"
  echo "║                                                               ║"
  echo "╚═══════════════════════════════════════════════════════════════╝"
  echo -e "${NC}"
}

print_usage() {
  echo -e "${BOLD}Usage:${NC} ./start_dev.sh <command> [options]"
  echo ""
  echo -e "${BOLD}Commands:${NC}"
  echo ""
  echo -e "  ${GREEN}vanilla${NC}      Start platform only (no MERL-T)"
  echo "               - Frontend (React) on :5173"
  echo "               - Backend (Express) on :3001"
  echo "               - Python API (visualex) on :5000"
  echo "               - PostgreSQL on :5432"
  echo ""
  echo -e "  ${GREEN}platform${NC}     Alias for 'vanilla'"
  echo ""
  echo -e "  ${GREEN}frontend${NC}     Start platform frontend"
  echo -e "  ${GREEN}backend${NC}      Start platform backend + postgres"
  echo -e "  ${GREEN}python-api${NC}   Start platform Python API container"
  echo ""
  echo -e "  ${GREEN}merlt${NC}        Start platform + MERL-T plugin"
  echo "               - Everything from 'vanilla' plus:"
  echo "               - MERL-T API + databases + RLCF web"
  echo ""
  echo -e "  ${GREEN}full${NC}         Start complete ALIS ecosystem"
  echo "               - Everything from 'merlt' plus:"
  echo "               - MERL-T ML API (FastAPI) on :8000"
  echo "               - FalkorDB (Knowledge Graph) on :6379"
  echo "               - Qdrant (Vector DB) on :6333"
  echo ""
  echo -e "  ${GREEN}merlt-stack${NC}  Start MERL-T stack (api + db + rlcf)"
  echo -e "  ${GREEN}merlt-ui${NC}     Start MERL-T standalone UI"
  echo -e "  ${GREEN}rlcf-web${NC}     Start RLCF web dashboard"
  echo ""
  echo -e "  ${GREEN}api${NC}          Start only visualex-api (Python scraping)"
  echo "               - Python API on :5000"
  echo ""
  echo -e "  ${GREEN}ml${NC}           Start only MERL-T ML backend"
  echo "               - FastAPI on :8000"
  echo "               - Databases (FalkorDB, Qdrant)"
  echo ""
  echo -e "  ${GREEN}db${NC}           Start only databases"
  echo "               - PostgreSQL, FalkorDB, Qdrant, Redis"
  echo ""
  echo -e "  ${GREEN}test${NC} <svc>   Run component tests"
  echo "               - frontend | backend | python-api"
  echo "               - merlt-frontend | merlt-backend | merlt-ml"
  echo ""
  echo -e "  ${GREEN}seed${NC}         Seed platform database with admin user"
  echo "               - uses ADMIN_PASSWORD from visualex-platform/.env"
  echo ""
  echo -e "  ${GREEN}status${NC}       Show status of all services"
  echo ""
  echo -e "  ${GREEN}stop${NC}         Stop all running services"
  echo ""
  echo -e "  ${GREEN}logs${NC} [svc]   View logs (optionally for specific service)"
  echo ""
  echo -e "  ${GREEN}help${NC}         Show this help message"
  echo ""
  echo -e "${BOLD}Examples:${NC}"
  echo "  ./start_dev.sh vanilla     # Basic platform"
  echo "  ./start_dev.sh full        # Everything"
  echo "  ./start_dev.sh stop        # Stop all"
  echo "  ./start_dev.sh logs merlt  # View MERL-T logs"
  echo ""
  echo -e "${BOLD}Documentation:${NC}"
  echo "  README.md           - Project overview"
  echo "  ARCHITETTURA.md     - Architecture (non-technical)"
  echo "  GUIDA_NAVIGAZIONE.md - Codebase navigation"
  echo ""
}

check_docker() {
  ensure_docker_config
  if ! command -v docker >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is required but not installed.${NC}"
    echo "Install Docker Desktop from https://docker.com"
    exit 1
  fi

  if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker daemon is not running.${NC}"
    echo "Start Docker Desktop and retry."
    exit 1
  fi

  init_compose_cmd
}

ensure_docker_config() {
  export DOCKER_CONFIG="$DOCKER_CONFIG_DIR"
  if [ ! -d "$DOCKER_CONFIG_DIR" ]; then
    mkdir -p "$DOCKER_CONFIG_DIR"
  fi
  # Local Docker config without credential helpers to avoid login errors.
  cat > "$DOCKER_CONFIG_FILE" <<'EOF'
{
  "auths": {}
}
EOF
}

init_compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    COMPOSE_CMD=(docker compose)
    echo -e "${CYAN}Using Docker Compose: ${COMPOSE_CMD[*]}${NC}"
    return
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    COMPOSE_CMD=(docker-compose)
    echo -e "${CYAN}Using Docker Compose: ${COMPOSE_CMD[*]}${NC}"
    return
  fi

  echo -e "${RED}Error: Docker Compose is required but not installed.${NC}"
  echo "Install Docker Desktop or docker-compose and retry."
  exit 1
}

compose() {
  "${COMPOSE_CMD[@]}" "$@"
}

check_python() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo -e "${RED}Error: Python 3.10+ is required but not found.${NC}"
    exit 1
  fi
}

ensure_env_file() {
  local dir="$1"
  if [ -d "$dir" ] && [ ! -f "$dir/.env" ] && [ -f "$dir/.env.example" ]; then
    cp "$dir/.env.example" "$dir/.env"
    echo -e "${YELLOW}Created $dir/.env from .env.example${NC}"
  fi
}

print_ports() {
  echo ""
  echo -e "${BOLD}Services will be available at:${NC}"
  echo -e "  ${CYAN}Frontend${NC}        http://localhost:5173"
  echo -e "  ${CYAN}Backend API${NC}     http://localhost:3001"
  echo -e "  ${CYAN}Python API${NC}      http://localhost:5000"
  if [ "${MERLT_ENABLED:-false}" = "true" ]; then
    echo -e "  ${CYAN}MERL-T API${NC}      http://localhost:8000"
    echo -e "  ${CYAN}Qdrant Dashboard${NC} http://localhost:6333/dashboard"
  fi
  echo ""
}

# ============================================================================
# Commands
# ============================================================================

cmd_vanilla() {
  echo -e "${GREEN}Starting VisuaLex Platform (vanilla mode)...${NC}"
  check_docker

  ensure_env_file "visualex-platform"

  cd "$ROOT/visualex-platform"
  echo "Starting frontend, backend, python-api, and postgres..."
  print_ports
  compose up --build frontend backend python-api postgres
}

cmd_platform() {
  cmd_vanilla
}

cmd_frontend() {
  echo -e "${GREEN}Starting platform frontend...${NC}"
  check_docker
  ensure_env_file "visualex-platform"
  cd "$ROOT/visualex-platform"
  compose up --build frontend
}

cmd_backend() {
  echo -e "${GREEN}Starting platform backend + postgres...${NC}"
  check_docker
  ensure_env_file "visualex-platform"
  cd "$ROOT/visualex-platform"
  compose up --build backend postgres
}

cmd_python_api_container() {
  echo -e "${GREEN}Starting platform Python API container...${NC}"
  check_docker
  ensure_env_file "visualex-platform"
  cd "$ROOT/visualex-platform"
  compose up --build python-api
}

cmd_merlt() {
  echo -e "${GREEN}Starting VisuaLex Platform + MERL-T integration...${NC}"
  check_docker

  ensure_env_file "visualex-platform"
  ensure_env_file "visualex-merlt"

  # Start MERL-T integration stack in background (no standalone UI)
  if [ -f "$ROOT/visualex-merlt/docker-compose.yml" ]; then
    echo "Starting MERL-T integration stack..."
    cd "$ROOT/visualex-merlt"
    compose up -d merlt-api qdrant postgres redis rlcf-web
  fi

  # Start platform with MERL-T enabled
  cd "$ROOT/visualex-platform"
  echo "Starting platform with MERL-T enabled..."
  export VITE_MERLT_ENABLED=true
  export MERLT_ENABLED=true
  export MERLT_API_URL=http://host.docker.internal:8000
  print_ports
  compose up --build frontend backend python-api postgres
}

cmd_merlt_stack() {
  echo -e "${GREEN}Starting MERL-T integration stack...${NC}"
  check_docker
  ensure_env_file "visualex-merlt"
  cd "$ROOT/visualex-merlt"
  compose up --build merlt-api qdrant postgres redis rlcf-web
}

cmd_merlt_ui() {
  echo -e "${GREEN}Starting MERL-T standalone UI...${NC}"
  check_docker
  ensure_env_file "visualex-merlt"
  cd "$ROOT/visualex-merlt"
  compose up --build merlt-frontend
}

cmd_rlcf_web() {
  echo -e "${GREEN}Starting RLCF web dashboard...${NC}"
  check_docker
  ensure_env_file "visualex-merlt"
  cd "$ROOT/visualex-merlt"
  compose up --build rlcf-web
}

cmd_full() {
  echo -e "${GREEN}Starting complete ALIS ecosystem...${NC}"
  check_docker

  ensure_env_file "visualex-platform"
  ensure_env_file "visualex-merlt"
  ensure_env_file "merlt"

  # Start MERL-T ML databases first
  if [ -f "$ROOT/merlt/docker-compose.dev.yml" ]; then
    echo "Starting MERL-T databases (FalkorDB, Qdrant)..."
    cd "$ROOT/merlt"
    compose -f docker-compose.dev.yml up -d
  fi

  # Start MERL-T integration stack
  if [ -f "$ROOT/visualex-merlt/docker-compose.yml" ]; then
    echo "Starting MERL-T integration stack..."
    cd "$ROOT/visualex-merlt"
    compose up -d
  fi

  # Start MERL-T ML API in background
  echo "Starting MERL-T ML API on :8000..."
  cd "$ROOT/merlt"
  if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import uvicorn" >/dev/null 2>&1; then
      python3 -m uvicorn merlt.app:app --reload --port 8000 &
      MERLT_PID=$!
      echo "MERL-T API started (PID: $MERLT_PID)"
    else
      echo -e "${YELLOW}Warning: uvicorn not installed, skipping MERL-T API${NC}"
      echo "Install with: cd merlt && pip install -e '.[dev]'"
    fi
  fi

  # Start platform with everything enabled
  cd "$ROOT/visualex-platform"
  echo "Starting platform with full MERL-T integration..."
  export VITE_MERLT_ENABLED=true
  export MERLT_ENABLED=true
  export MERLT_API_URL=http://localhost:8000
  MERLT_ENABLED=true print_ports
  compose up --build frontend backend python-api postgres
}

cmd_api() {
  echo -e "${GREEN}Starting visualex-api only...${NC}"
  check_python

  cd "$ROOT/visualex-api"

  if ! python3 -c "import visualex" >/dev/null 2>&1; then
    echo "Installing visualex dependencies..."
    python3 -m pip install -e .
  fi

  echo -e "${CYAN}VisuaLex API starting on http://localhost:5000${NC}"
  exec python3 -m visualex.app
}

cmd_ml() {
  echo -e "${GREEN}Starting MERL-T ML backend only...${NC}"
  check_docker
  check_python

  ensure_env_file "merlt"

  # Start databases
  if [ -f "$ROOT/merlt/docker-compose.dev.yml" ]; then
    echo "Starting MERL-T databases..."
    cd "$ROOT/merlt"
    compose -f docker-compose.dev.yml up -d
  fi

  # Start API
  cd "$ROOT/merlt"
  if ! python3 -c "import uvicorn" >/dev/null 2>&1; then
    echo -e "${RED}Missing uvicorn. Install with:${NC}"
    echo "  cd merlt && pip install -e '.[dev]'"
    exit 1
  fi

  echo -e "${CYAN}MERL-T API starting on http://localhost:8000${NC}"
  echo -e "${CYAN}Qdrant Dashboard: http://localhost:6333/dashboard${NC}"
  exec python3 -m uvicorn merlt.app:app --reload --port 8000
}

cmd_db() {
  echo -e "${GREEN}Starting databases only...${NC}"
  check_docker

  # Platform databases
  if [ -f "$ROOT/visualex-platform/docker-compose.yml" ]; then
    echo "Starting PostgreSQL..."
    cd "$ROOT/visualex-platform"
    compose up -d postgres
  fi

  # MERL-T databases
  if [ -f "$ROOT/merlt/docker-compose.dev.yml" ]; then
    echo "Starting FalkorDB and Qdrant..."
    cd "$ROOT/merlt"
    compose -f docker-compose.dev.yml up -d
  fi

  echo ""
  echo -e "${GREEN}Databases started:${NC}"
  echo "  PostgreSQL: localhost:5432"
  echo "  FalkorDB:   localhost:6379"
  echo "  Qdrant:     localhost:6333"
}

has_npm_script() {
  local script_name="$1"
  if ! command -v node >/dev/null 2>&1; then
    echo -e "${RED}Node.js is required to inspect package.json scripts.${NC}"
    return 1
  fi
  node -e "const pkg=require('./package.json'); process.exit(pkg.scripts && pkg.scripts['$script_name'] ? 0 : 1)" 2>/dev/null
}

run_npm_script() {
  local dir="$1"
  local script_name="$2"
  if [ ! -f "$dir/package.json" ]; then
    echo -e "${RED}No package.json in $dir${NC}"
    return 1
  fi
  cd "$dir"
  if has_npm_script "$script_name"; then
    npm run "$script_name"
  else
    echo -e "${YELLOW}No npm script '$script_name' in $dir${NC}"
  fi
}

cmd_test() {
  local target="${1:-}"
  case "$target" in
    frontend)
      run_npm_script "$ROOT/visualex-platform/frontend" "test"
      ;;
    backend)
      run_npm_script "$ROOT/visualex-platform/backend" "test"
      ;;
    python-api)
      check_python
      cd "$ROOT/visualex-api"
      python3 -m pytest
      ;;
    merlt-frontend)
      run_npm_script "$ROOT/visualex-merlt/frontend" "test"
      ;;
    merlt-backend)
      run_npm_script "$ROOT/visualex-merlt/backend" "test"
      ;;
    merlt-ml)
      check_python
      cd "$ROOT/merlt"
      python3 -m pytest
      ;;
    *)
      echo -e "${RED}Unknown test target: ${target}${NC}"
      echo "Available: frontend, backend, python-api, merlt-frontend, merlt-backend, merlt-ml"
      exit 1
      ;;
  esac
}

cmd_seed() {
  echo -e "${GREEN}Seeding platform database...${NC}"
  check_docker
  ensure_env_file "visualex-platform"
  cd "$ROOT/visualex-platform"
  compose exec backend sh -c "npx tsx src/utils/seed.ts"
}

cmd_status() {
  echo -e "${BOLD}ALIS Services Status${NC}"
  echo "===================="
  echo ""

  # Check Docker containers
  echo -e "${CYAN}Docker Containers:${NC}"
  docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -E "(visualex|merlt|postgres|falkordb|qdrant|redis)" || echo "  No ALIS containers running"
  echo ""

  # Check ports
  echo -e "${CYAN}Port Status:${NC}"
  for port in 5173 3001 5000 8000 5432 6379 6333; do
    if lsof -i :$port >/dev/null 2>&1; then
      service=""
      case $port in
        5173) service="Frontend (Vite)" ;;
        3001) service="Backend (Express)" ;;
        5000) service="Python API (visualex)" ;;
        8000) service="MERL-T API (FastAPI)" ;;
        5432) service="PostgreSQL" ;;
        6379) service="FalkorDB/Redis" ;;
        6333) service="Qdrant" ;;
      esac
      echo -e "  :$port ${GREEN}LISTENING${NC} - $service"
    else
      echo -e "  :$port ${RED}FREE${NC}"
    fi
  done
}

cmd_stop() {
  echo -e "${YELLOW}Stopping all ALIS services...${NC}"

  # Stop visualex-platform
  if [ -f "$ROOT/visualex-platform/docker-compose.yml" ]; then
    cd "$ROOT/visualex-platform"
    compose down 2>/dev/null || true
  fi

  # Stop visualex-merlt
  if [ -f "$ROOT/visualex-merlt/docker-compose.yml" ]; then
    cd "$ROOT/visualex-merlt"
    compose down 2>/dev/null || true
  fi

  # Stop merlt databases
  if [ -f "$ROOT/merlt/docker-compose.dev.yml" ]; then
    cd "$ROOT/merlt"
    compose -f docker-compose.dev.yml down 2>/dev/null || true
  fi

  # Kill any remaining Python processes
  pkill -f "uvicorn merlt" 2>/dev/null || true
  pkill -f "visualex.app" 2>/dev/null || true

  echo -e "${GREEN}All services stopped.${NC}"
}

cmd_logs() {
  local service="${1:-}"

  if [ -z "$service" ]; then
    echo "Showing logs from all services..."
    echo "(Press Ctrl+C to exit)"
    echo ""

    # Try to show logs from platform
    cd "$ROOT/visualex-platform" 2>/dev/null && compose logs -f --tail=50 2>/dev/null || true
  else
    case "$service" in
      frontend|backend|postgres|python-api)
        cd "$ROOT/visualex-platform"
        compose logs -f --tail=100 "$service"
        ;;
      merlt|rlcf)
        cd "$ROOT/visualex-merlt"
        compose logs -f --tail=100
        ;;
      ml|api)
        cd "$ROOT/merlt"
        compose -f docker-compose.dev.yml logs -f --tail=100 2>/dev/null || echo "No MERL-T database logs"
        ;;
      *)
        echo -e "${RED}Unknown service: $service${NC}"
        echo "Available: frontend, backend, postgres, python-api, merlt, ml"
        ;;
    esac
  fi
}

# ============================================================================
# Main
# ============================================================================

COMMAND="${1:-help}"
shift || true

case "$COMMAND" in
  vanilla)
    print_banner
    cmd_vanilla
    ;;
  platform)
    print_banner
    cmd_platform
    ;;
  frontend)
    print_banner
    cmd_frontend
    ;;
  backend)
    print_banner
    cmd_backend
    ;;
  python-api)
    print_banner
    cmd_python_api_container
    ;;
  merlt)
    print_banner
    cmd_merlt
    ;;
  merlt-stack)
    print_banner
    cmd_merlt_stack
    ;;
  merlt-ui)
    print_banner
    cmd_merlt_ui
    ;;
  rlcf-web)
    print_banner
    cmd_rlcf_web
    ;;
  full)
    print_banner
    cmd_full
    ;;
  api)
    cmd_api
    ;;
  ml)
    cmd_ml
    ;;
  db)
    cmd_db
    ;;
  seed)
    cmd_seed
    ;;
  test)
    cmd_test "$@"
    ;;
  status)
    cmd_status
    ;;
  stop)
    cmd_stop
    ;;
  logs)
    cmd_logs "$@"
    ;;
  help|-h|--help)
    print_banner
    print_usage
    ;;
  *)
    echo -e "${RED}Unknown command: $COMMAND${NC}"
    echo ""
    print_usage
    exit 1
    ;;
esac
