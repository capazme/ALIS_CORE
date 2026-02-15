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
  echo -e "${BOLD}Usage:${NC} ./start_dev.sh [command] [options]"
  echo ""
  echo -e "  Senza argomenti: mostra menu interattivo"
  echo ""
  echo -e "${BOLD}Dev Locale (veloce, senza Docker rebuild per frontend/backend):${NC}"
  echo ""
  echo -e "  ${GREEN}dev${NC} [target]   Avvio dev locale"
  echo -e "                 target: ${CYAN}platform${NC} (p) | ${CYAN}merlt${NC} (m) | ${CYAN}all${NC} (a)"
  echo -e "                 DB via Docker, frontend+backend nativi"
  echo ""
  echo -e "${BOLD}Docker Compose (tutto containerizzato):${NC}"
  echo ""
  echo -e "  ${GREEN}vanilla${NC}        Platform only (frontend+backend+DB+python-api)"
  echo -e "  ${GREEN}merlt${NC}          Platform + MERL-T plugin"
  echo -e "  ${GREEN}full${NC}           Ecosistema completo (Platform+MERL-T+ML)"
  echo ""
  echo -e "${BOLD}Singoli Servizi:${NC}"
  echo ""
  echo -e "  ${GREEN}frontend${NC}       Solo frontend platform (Docker)"
  echo -e "  ${GREEN}backend${NC}        Solo backend + postgres"
  echo -e "  ${GREEN}python-api${NC}     Solo Python API container"
  echo -e "  ${GREEN}db${NC}             Solo database (PostgreSQL+FalkorDB+Qdrant+Redis)"
  echo -e "  ${GREEN}ml${NC}             Solo MERL-T ML backend (FastAPI :8000)"
  echo -e "  ${GREEN}api${NC}            Solo visualex-api (Python :5000)"
  echo -e "  ${GREEN}merlt-stack${NC}    Stack MERL-T (api+db+rlcf)"
  echo -e "  ${GREEN}merlt-ui${NC}       MERL-T standalone UI"
  echo -e "  ${GREEN}rlcf-web${NC}       RLCF web dashboard"
  echo ""
  echo -e "${BOLD}Utility:${NC}"
  echo ""
  echo -e "  ${GREEN}install${NC}        Installa dipendenze (npm+pip) di tutti i moduli"
  echo -e "  ${GREEN}test${NC} <target>  Esegui test (frontend|backend|python-api|merlt-*)"
  echo -e "  ${GREEN}seed${NC}           Popola DB con utente admin"
  echo -e "  ${GREEN}restart${NC} [cmd]  Stop + riavvio (opzionalmente con comando specifico)"
  echo -e "  ${GREEN}status${NC}         Mostra stato di tutti i servizi"
  echo -e "  ${GREEN}stop${NC}           Ferma tutti i servizi"
  echo -e "  ${GREEN}logs${NC} [svc]     Visualizza log (frontend|backend|merlt|ml)"
  echo -e "  ${GREEN}help${NC}           Questo messaggio"
  echo ""
  echo -e "${BOLD}Esempi:${NC}"
  echo "  ./start_dev.sh                 # Menu interattivo"
  echo "  ./start_dev.sh dev platform    # Dev veloce platform"
  echo "  ./start_dev.sh dev merlt       # Dev veloce con MERL-T"
  echo "  ./start_dev.sh vanilla         # Platform via Docker"
  echo "  ./start_dev.sh full            # Tutto"
  echo "  ./start_dev.sh status          # Stato servizi"
  echo "  ./start_dev.sh stop            # Ferma tutto"
  echo ""
  echo -e "${BOLD}Porte:${NC}"
  echo "  :5173  Frontend (Vite)      :3001  Backend (Express)"
  echo "  :5000  Python API           :8000  MERL-T API (FastAPI)"
  echo "  :5432  PostgreSQL           :6379  FalkorDB"
  echo "  :6333  Qdrant               :6379  Redis"
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

  # Start MERL-T integration UI only (DBs already running from merlt dev compose,
  # API started via uvicorn below). --no-deps avoids starting duplicate DB containers.
  if [ -f "$ROOT/visualex-merlt/docker-compose.yml" ]; then
    echo "Starting MERL-T UI containers..."
    cd "$ROOT/visualex-merlt"
    compose up -d --no-deps merlt-frontend rlcf-web
  fi

  # Start MERL-T ML API in background (prefer venv if available)
  echo "Starting MERL-T ML API on :8000..."
  cd "$ROOT/merlt"
  MERLT_PYTHON=""
  if [ -f "$ROOT/merlt/.venv/bin/python" ]; then
    MERLT_PYTHON="$ROOT/merlt/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    MERLT_PYTHON="python3"
  fi
  if [ -n "$MERLT_PYTHON" ]; then
    if "$MERLT_PYTHON" -c "import uvicorn" >/dev/null 2>&1; then
      "$MERLT_PYTHON" -m uvicorn merlt.app:app --reload --port 8000 &
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
  compose up -d frontend backend python-api postgres
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
# Quick Dev Commands (native, no Docker rebuild for frontend/backend)
# ============================================================================

ensure_node_modules() {
  local dir="$1"
  local label="${2:-$(basename "$dir")}"
  if [ ! -d "$dir/node_modules" ]; then
    echo -e "${YELLOW}[$label] node_modules mancante, installo dipendenze...${NC}"
    (cd "$dir" && npm install)
  fi
}

cmd_dev() {
  local target="${1:-platform}"

  case "$target" in
    platform|p)
      cmd_dev_platform
      ;;
    merlt|m)
      cmd_dev_merlt
      ;;
    all|a)
      cmd_dev_all
      ;;
    *)
      echo -e "${RED}Target sconosciuto: $target${NC}"
      echo "Disponibili: platform (p), merlt (m), all (a)"
      exit 1
      ;;
  esac
}

cmd_dev_platform() {
  echo -e "${GREEN}Dev locale platform (no Docker build per frontend/backend)${NC}"
  check_docker

  ensure_env_file "visualex-platform"

  # Start only databases and python-api via Docker
  echo -e "${CYAN}Avvio servizi Docker (postgres, python-api)...${NC}"
  cd "$ROOT/visualex-platform"
  compose up -d postgres python-api

  echo -e "${CYAN}Attendo PostgreSQL...${NC}"
  local retries=0
  while ! compose exec -T postgres pg_isready -q 2>/dev/null; do
    retries=$((retries + 1))
    if [ "$retries" -gt 30 ]; then
      echo -e "${RED}PostgreSQL non disponibile dopo 30s${NC}"
      break
    fi
    sleep 1
  done
  echo -e "${GREEN}PostgreSQL pronto.${NC}"

  ensure_node_modules "$ROOT/visualex-platform/backend" "backend"
  ensure_node_modules "$ROOT/visualex-platform/frontend" "frontend"

  # Start backend in background
  echo -e "${CYAN}Avvio backend Express (:3001)...${NC}"
  (cd "$ROOT/visualex-platform/backend" && npm run dev) &
  local BACKEND_PID=$!

  # Give backend a moment to start
  sleep 2

  echo ""
  echo -e "${BOLD}Servizi attivi:${NC}"
  echo -e "  ${GREEN}PostgreSQL${NC}    :5432  (Docker)"
  echo -e "  ${GREEN}Python API${NC}    :5000  (Docker)"
  echo -e "  ${GREEN}Backend${NC}       :3001  (PID $BACKEND_PID)"
  echo -e "  ${GREEN}Frontend${NC}      :5173  (avvio...)"
  echo ""

  # Trap to cleanup on exit
  trap "echo -e '${YELLOW}Arresto servizi locali...${NC}'; kill $BACKEND_PID 2>/dev/null; exit 0" INT TERM

  # Start frontend in foreground
  cd "$ROOT/visualex-platform/frontend"
  npm run dev

  # Cleanup when frontend exits
  kill $BACKEND_PID 2>/dev/null
}

cmd_dev_merlt() {
  echo -e "${GREEN}Dev locale MERL-T (platform + plugin MERL-T)${NC}"
  check_docker

  ensure_env_file "visualex-platform"
  ensure_env_file "visualex-merlt"

  # Start platform databases
  echo -e "${CYAN}Avvio servizi Docker (postgres, python-api)...${NC}"
  cd "$ROOT/visualex-platform"
  compose up -d postgres python-api

  # Start MERL-T databases if available
  if [ -f "$ROOT/merlt/docker-compose.dev.yml" ]; then
    echo -e "${CYAN}Avvio database MERL-T (FalkorDB, Qdrant, Redis)...${NC}"
    cd "$ROOT/merlt"
    compose -f docker-compose.dev.yml up -d
  fi

  # Start MERL-T integration stack if available
  if [ -f "$ROOT/visualex-merlt/docker-compose.yml" ]; then
    echo -e "${CYAN}Avvio MERL-T stack (merlt-api, rlcf-web)...${NC}"
    cd "$ROOT/visualex-merlt"
    compose up -d merlt-api rlcf-web 2>/dev/null || true
  fi

  echo -e "${CYAN}Attendo PostgreSQL...${NC}"
  cd "$ROOT/visualex-platform"
  local retries=0
  while ! compose exec -T postgres pg_isready -q 2>/dev/null; do
    retries=$((retries + 1))
    if [ "$retries" -gt 30 ]; then
      echo -e "${RED}PostgreSQL non disponibile dopo 30s${NC}"
      break
    fi
    sleep 1
  done

  ensure_node_modules "$ROOT/visualex-platform/backend" "backend"
  ensure_node_modules "$ROOT/visualex-platform/frontend" "frontend"
  ensure_node_modules "$ROOT/visualex-merlt/frontend" "merlt-frontend"

  local PIDS=()

  # Start platform backend
  echo -e "${CYAN}Avvio backend Express (:3001)...${NC}"
  (cd "$ROOT/visualex-platform/backend" && npm run dev) &
  PIDS+=($!)

  # Start MERL-T frontend dev
  if [ -d "$ROOT/visualex-merlt/frontend" ]; then
    echo -e "${CYAN}Avvio MERL-T frontend dev...${NC}"
    (cd "$ROOT/visualex-merlt/frontend" && npm run dev) &
    PIDS+=($!)
  fi

  sleep 2

  echo ""
  echo -e "${BOLD}Servizi attivi:${NC}"
  echo -e "  ${GREEN}PostgreSQL${NC}       :5432  (Docker)"
  echo -e "  ${GREEN}Python API${NC}       :5000  (Docker)"
  echo -e "  ${GREEN}FalkorDB${NC}         :6379  (Docker)"
  echo -e "  ${GREEN}Qdrant${NC}           :6333  (Docker)"
  echo -e "  ${GREEN}Backend${NC}          :3001  (locale)"
  echo -e "  ${GREEN}Frontend${NC}         :5173  (avvio...)"
  echo -e "  ${GREEN}MERL-T Frontend${NC}  plugin (locale)"
  echo ""

  trap "echo -e '${YELLOW}Arresto servizi locali...${NC}'; for p in \${PIDS[*]}; do kill \$p 2>/dev/null; done; exit 0" INT TERM

  # Start platform frontend in foreground
  export VITE_MERLT_ENABLED=true
  cd "$ROOT/visualex-platform/frontend"
  npm run dev

  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
}

cmd_dev_all() {
  cmd_dev_merlt
}

cmd_install() {
  echo -e "${GREEN}Installazione dipendenze di tutti i moduli...${NC}"

  local dirs=(
    "visualex-platform/frontend"
    "visualex-platform/backend"
    "visualex-merlt/frontend"
    "visualex-merlt/rlcf-web"
  )

  for dir in "${dirs[@]}"; do
    local full_path="$ROOT/$dir"
    if [ -f "$full_path/package.json" ]; then
      echo -e "${CYAN}[$dir]${NC} npm install..."
      (cd "$full_path" && npm install) || echo -e "${YELLOW}Warning: npm install fallito per $dir${NC}"
    fi
  done

  # Python deps
  if [ -f "$ROOT/visualex-api/setup.py" ] || [ -f "$ROOT/visualex-api/pyproject.toml" ]; then
    if command -v python3 >/dev/null 2>&1; then
      echo -e "${CYAN}[visualex-api]${NC} pip install..."
      (cd "$ROOT/visualex-api" && python3 -m pip install -e . -q) || true
    fi
  fi

  if [ -f "$ROOT/merlt/pyproject.toml" ]; then
    if command -v python3 >/dev/null 2>&1; then
      echo -e "${CYAN}[merlt]${NC} pip install..."
      (cd "$ROOT/merlt" && python3 -m pip install -e ".[dev]" -q) || true
    fi
  fi

  echo -e "${GREEN}Installazione completata.${NC}"
}

cmd_restart() {
  local target="${1:-}"
  echo -e "${YELLOW}Riavvio servizi...${NC}"
  cmd_stop
  echo ""
  sleep 2
  if [ -n "$target" ]; then
    "$0" "$target"
  else
    cmd_menu
  fi
}

# ============================================================================
# Interactive Menu
# ============================================================================

cmd_menu() {
  print_banner
  echo -e "${BOLD}Cosa vuoi avviare?${NC}"
  echo ""
  echo -e "  ${BOLD}Dev locale${NC} ${CYAN}(veloce, no Docker rebuild per frontend/backend):${NC}"
  echo -e "    ${GREEN}1${NC})  Platform frontend+backend       ${CYAN}dev platform${NC}"
  echo -e "    ${GREEN}2${NC})  Platform + MERL-T                ${CYAN}dev merlt${NC}"
  echo ""
  echo -e "  ${BOLD}Docker compose${NC} ${CYAN}(tutto containerizzato):${NC}"
  echo -e "    ${GREEN}3${NC})  Platform completa (vanilla)      ${CYAN}vanilla${NC}"
  echo -e "    ${GREEN}4${NC})  Platform + MERL-T                ${CYAN}merlt${NC}"
  echo -e "    ${GREEN}5${NC})  Full Stack (tutto)               ${CYAN}full${NC}"
  echo ""
  echo -e "  ${BOLD}Singoli servizi:${NC}"
  echo -e "    ${GREEN}6${NC})  Solo Database                    ${CYAN}db${NC}"
  echo -e "    ${GREEN}7${NC})  Solo Backend + DB                ${CYAN}backend${NC}"
  echo -e "    ${GREEN}8${NC})  Solo MERL-T ML backend           ${CYAN}ml${NC}"
  echo -e "    ${GREEN}9${NC})  Solo Python API (scraping)       ${CYAN}api${NC}"
  echo ""
  echo -e "  ${BOLD}Utility:${NC}"
  echo -e "    ${GREEN}s${NC})  Status servizi                   ${CYAN}status${NC}"
  echo -e "    ${GREEN}x${NC})  Stop tutto                       ${CYAN}stop${NC}"
  echo -e "    ${GREEN}i${NC})  Installa tutte le dipendenze     ${CYAN}install${NC}"
  echo -e "    ${GREEN}t${NC})  Test                             ${CYAN}test <target>${NC}"
  echo -e "    ${GREEN}h${NC})  Help completo                    ${CYAN}help${NC}"
  echo ""
  echo -ne "  ${BOLD}Scegli [1-9, s/x/i/t/h]: ${NC}"
  read -r choice
  echo ""

  case "$choice" in
    1)    cmd_dev_platform ;;
    2)    cmd_dev_merlt ;;
    3)    cmd_vanilla ;;
    4)    cmd_merlt ;;
    5)    cmd_full ;;
    6)    cmd_db ;;
    7)    cmd_backend ;;
    8)    cmd_ml ;;
    9)    cmd_api ;;
    s|S)  cmd_status ;;
    x|X)  cmd_stop ;;
    i|I)  cmd_install ;;
    t|T)
      echo -ne "  Target (frontend/backend/python-api/merlt-frontend): "
      read -r test_target
      cmd_test "$test_target"
      ;;
    h|H)  print_usage ;;
    "")   echo -e "${YELLOW}Nessuna scelta. Usa './start_dev.sh help' per i comandi.${NC}" ;;
    *)    echo -e "${RED}Scelta non valida: $choice${NC}" ;;
  esac
}

# ============================================================================
# Main
# ============================================================================

COMMAND="${1:-menu}"
shift || true

case "$COMMAND" in
  menu)
    cmd_menu
    ;;
  dev)
    print_banner
    cmd_dev "$@"
    ;;
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
  install)
    print_banner
    cmd_install
    ;;
  restart)
    cmd_restart "$@"
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
    echo -e "${RED}Comando sconosciuto: $COMMAND${NC}"
    echo ""
    print_usage
    exit 1
    ;;
esac
