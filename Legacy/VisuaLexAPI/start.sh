#!/bin/bash

# =============================================================================
# VisuaLex + MERL-T Development Startup Script
# =============================================================================
# Avvia l'ambiente di sviluppo completo:
# - VisuaLex API (Python/Quart - port 5000)
# - VisuaLex Backend (Node/Express - port 3001)
# - VisuaLex Frontend (Vite/React - port 5173)
# - [Opzionale] MERL-T Knowledge Graph API (FastAPI - port 8000)
# - [Opzionale] MERL-T Databases (Docker: PostgreSQL, FalkorDB, Qdrant, Redis)
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# Project paths
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
MERLT_ROOT="/Users/gpuzio/Desktop/CODE/ALIS_CORE/Legacy/MERL-T_alpha"

# PID storage
PIDS=()

# Banner
echo -e "${CYAN}"
cat << "EOF"
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║  VisuaLex + MERL-T Development Environment                   ║
║                                                               ║
║  🔍 VisuaLex: Ricerca normativa intelligente                  ║
║  🧠 MERL-T: Knowledge Graph & Multi-Expert Q&A                ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"

# =============================================================================
# Utility Functions
# =============================================================================

print_step() {
    echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

print_success() { echo -e "  ${GREEN}✓${NC} $1"; }
print_warning() { echo -e "  ${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "  ${RED}✗${NC} $1"; }
print_info() { echo -e "  ${CYAN}ℹ${NC} $1"; }

check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo -e "${RED}Port $1 in use${NC}"
        echo -e "  Run: ${YELLOW}kill \$(lsof -t -i:$1)${NC}"
        return 1
    fi
    return 0
}

kill_port() {
    local port=$1
    local pid=$(lsof -t -i:$port 2>/dev/null)
    if [ -n "$pid" ]; then
        echo -e "  ${YELLOW}Terminando processo su porta $port (PID: $pid)${NC}"
        kill -9 $pid 2>/dev/null || true
        sleep 1
    fi
}

cleanup() {
    echo -e "\n${YELLOW}Shutting down all services...${NC}"
    for pid in "${PIDS[@]}"; do
        kill $pid 2>/dev/null || true
    done
    echo -e "${GREEN}All services stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=${3:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    return 1
}

# =============================================================================
# 1. Selezione Modalità
# =============================================================================

print_step "1. Selezione Modalità"

echo "  Quale ambiente vuoi avviare?"
echo ""
echo "    1) 🔍 Solo VisuaLex (senza Knowledge Graph)"
echo "    2) 🚀 VisuaLex + MERL-T (stack completo)"
echo "    3) 🧪 VisuaLex + MERL-T con database Docker"
echo ""
read -p "  Scegli [1-3] (default: 1): " mode
mode=${mode:-1}

START_MERLT=false
START_DOCKER=false

case $mode in
    2)
        START_MERLT=true
        print_info "Modalità: VisuaLex + MERL-T (API only)"
        ;;
    3)
        START_MERLT=true
        START_DOCKER=true
        print_info "Modalità: VisuaLex + MERL-T con Docker databases"
        ;;
    *)
        print_info "Modalità: Solo VisuaLex"
        ;;
esac

# =============================================================================
# 2. Verifica Porte
# =============================================================================

print_step "2. Verifica Porte Disponibili"

# Definisci le porte necessarie
REQUIRED_PORTS=(5000 3001 5173)
if [ "$START_MERLT" = true ]; then
    REQUIRED_PORTS+=(8000)
fi

# Verifica quali porte sono occupate (qualsiasi stato, non solo LISTEN)
OCCUPIED_PORTS=()
for port in "${REQUIRED_PORTS[@]}"; do
    if lsof -i :$port -t >/dev/null 2>&1; then
        OCCUPIED_PORTS+=($port)
    fi
done

# Se ci sono porte occupate, chiedi se terminare i processi
if [ ${#OCCUPIED_PORTS[@]} -gt 0 ]; then
    print_warning "Porte occupate: ${OCCUPIED_PORTS[*]}"
    echo ""
    read -p "  Vuoi terminare i processi esistenti? [y/N]: " kill_existing

    if [[ "$kill_existing" =~ ^[Yy]$ ]]; then
        for port in "${OCCUPIED_PORTS[@]}"; do
            kill_port $port
        done
        print_success "Processi terminati"
    else
        print_error "Porte occupate. Liberale manualmente e riprova."
        exit 1
    fi
else
    print_success "Tutte le porte disponibili"
fi

# =============================================================================
# 3. [Opzionale] Avvio Docker Databases
# =============================================================================

if [ "$START_DOCKER" = true ]; then
    print_step "3. Avvio Database Docker (MERL-T)"

    if ! command -v docker &> /dev/null; then
        print_error "Docker non trovato. Installalo o scegli modalità 2."
        exit 1
    fi

    cd "$MERLT_ROOT"

    if [ -f "docker-compose.dev.yml" ]; then
        print_info "Avvio containers..."
        docker-compose -f docker-compose.dev.yml up -d

        print_info "Attesa inizializzazione database (15s)..."
        sleep 15

        # Verifica containers
        if docker ps | grep -q "merl-t-postgres-dev"; then
            print_success "PostgreSQL (port 5433)"
        else
            print_warning "PostgreSQL non avviato"
        fi

        if docker ps | grep -q "merl-t-falkordb-dev"; then
            print_success "FalkorDB (port 6380)"
        else
            print_warning "FalkorDB non avviato"
        fi

        if docker ps | grep -q "merl-t-qdrant-dev"; then
            print_success "Qdrant (port 6333)"
        else
            print_warning "Qdrant non avviato"
        fi

        if docker ps | grep -q "merl-t-redis-dev"; then
            print_success "Redis (port 6379)"
        else
            print_warning "Redis non avviato"
        fi
    else
        print_error "docker-compose.dev.yml non trovato in $MERLT_ROOT"
        exit 1
    fi

    cd "$PROJECT_ROOT"
fi

# =============================================================================
# 4. [Opzionale] Avvio MERL-T API
# =============================================================================

if [ "$START_MERLT" = true ]; then
    print_step "4. Avvio MERL-T API (port 8000)"

    cd "$MERLT_ROOT"

    if [ -d ".venv" ]; then
        source .venv/bin/activate
        print_success "Virtual environment MERL-T attivato"
    else
        print_error ".venv non trovato in $MERLT_ROOT"
        print_info "Esegui: cd $MERLT_ROOT && python3 -m venv .venv && pip install -e ."
        exit 1
    fi

    # Verifica che uvicorn sia installato
    if ! "$MERLT_ROOT/.venv/bin/python" -c "import uvicorn" 2>/dev/null; then
        print_error "uvicorn non trovato nel virtual environment MERL-T"
        print_info "Installazione dipendenze MERL-T..."
        cd "$MERLT_ROOT"
        source .venv/bin/activate
        pip install -e . || {
            print_error "Errore durante l'installazione delle dipendenze"
            print_info "Esegui manualmente: cd $MERLT_ROOT && source .venv/bin/activate && pip install -e ."
            exit 1
        }
        print_success "Dipendenze installate"
        cd "$PROJECT_ROOT"
    fi

    # Verifica e installa tutte le dipendenze se necessario
    print_info "Verifica dipendenze MERL-T..."
    
    # Lista dipendenze critiche da verificare
    CRITICAL_DEPS=("falkordb" "fastapi" "pydantic" "tiktoken" "qdrant_client" "sentence_transformers")
    MISSING_DEPS=()
    
    for dep in "${CRITICAL_DEPS[@]}"; do
        # Gestisci nomi diversi per import vs package
        import_name="$dep"
        case "$dep" in
            "qdrant_client") import_name="qdrant_client" ;;
            "sentence_transformers") import_name="sentence_transformers" ;;
        esac
        
        if ! "$MERLT_ROOT/.venv/bin/python" -c "import $import_name" 2>/dev/null; then
            MISSING_DEPS+=("$dep")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        print_warning "Dipendenze mancanti: ${MISSING_DEPS[*]}"
        print_info "Installazione dipendenze da pyproject.toml..."
        cd "$MERLT_ROOT"
        source .venv/bin/activate
        
        # Installa da pyproject.toml (include tutte le dipendenze)
        pip install -e . || {
            print_error "Errore durante l'installazione delle dipendenze"
            print_info "Esegui manualmente: cd $MERLT_ROOT && source .venv/bin/activate && pip install -e ."
            print_info "Oppure installa manualmente: pip install ${MISSING_DEPS[*]}"
            exit 1
        }
        
        # Se tiktoken manca ancora, installalo separatamente (potrebbe non essere nel pyproject.toml)
        if "$MERLT_ROOT/.venv/bin/python" -c "import tiktoken" 2>/dev/null; then
            :
        else
            print_info "Installazione tiktoken..."
            pip install tiktoken || print_warning "tiktoken non installato (opzionale per token counting)"
        fi
        
        print_success "Dipendenze installate"
        cd "$PROJECT_ROOT"
    else
        print_success "Tutte le dipendenze critiche installate"
    fi

    # Crea directory logs se non esiste
    mkdir -p logs

    # Avvia MERL-T API con PYTHONPATH corretto
    # Aggiungi VisuaLexAPI al PYTHONPATH per permettere import di visualex
    # Usa il percorso completo al Python del venv per evitare problemi con nohup
    print_info "Avvio uvicorn..."
    export PYTHONPATH="$MERLT_ROOT:$PROJECT_ROOT:$PYTHONPATH"
    # Passa PYTHONPATH esplicitamente al comando nohup
    nohup env PYTHONPATH="$MERLT_ROOT:$PROJECT_ROOT:$PYTHONPATH" "$MERLT_ROOT/.venv/bin/python" -m uvicorn merlt.api.visualex_bridge:app --reload --host 0.0.0.0 --port 8000 > logs/merlt-api.log 2>&1 &
    MERLT_PID=$!
    PIDS+=($MERLT_PID)
    echo $MERLT_PID > logs/merlt-api.pid

    print_info "Attesa avvio MERL-T API..."
    if wait_for_service "http://localhost:8000/health" "MERL-T API" 30; then
        print_success "MERL-T API avviato (PID: $MERLT_PID)"
    else
        print_warning "MERL-T API non risponde ancora"
        print_info "Log file: $MERLT_ROOT/logs/merlt-api.log"
        echo ""
        echo -e "${YELLOW}Ultimi 20 righe del log:${NC}"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        tail -20 "$MERLT_ROOT/logs/merlt-api.log" 2>/dev/null || echo "Log file non trovato"
        echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo ""
        print_info "Per vedere i log in tempo reale: tail -f $MERLT_ROOT/logs/merlt-api.log"
    fi

    cd "$PROJECT_ROOT"
fi

# =============================================================================
# 5. Avvio VisuaLex API (Python/Quart)
# =============================================================================

print_step "5. Avvio VisuaLex API (port 5000)"

cd "$PROJECT_ROOT"

# Crea directory logs se non esiste
mkdir -p logs

if [ -d ".venv" ]; then
    source .venv/bin/activate
    print_success "Virtual environment VisuaLex attivato"
else
    print_warning ".venv non trovato, usando Python di sistema"
fi

python app.py > logs/visualex-api.log 2>&1 &
API_PID=$!
PIDS+=($API_PID)
echo $API_PID > logs/visualex-api.pid

print_info "Attesa avvio VisuaLex API..."
if wait_for_service "http://localhost:5000/health" "VisuaLex API" 15; then
    print_success "VisuaLex API avviato (PID: $API_PID)"
else
    print_warning "VisuaLex API non risponde ancora"
fi

# =============================================================================
# 6. Avvio Platform Backend (Node/Express)
# =============================================================================

print_step "6. Avvio Platform Backend (port 3001)"

cd "$PROJECT_ROOT/backend"

# Usa .env.merlt se MERL-T è attivo
if [ "$START_MERLT" = true ] && [ -f ".env.merlt" ]; then
    cp .env.merlt .env
    print_info "Configurazione: .env.merlt (MERL-T enabled)"
fi

npm run dev > "$PROJECT_ROOT/logs/backend.log" 2>&1 &
BACKEND_PID=$!
PIDS+=($BACKEND_PID)
echo $BACKEND_PID > "$PROJECT_ROOT/logs/backend.pid"

print_info "Attesa avvio Platform Backend..."
if wait_for_service "http://localhost:3001/api/health" "Platform Backend" 15; then
    print_success "Platform Backend avviato (PID: $BACKEND_PID)"
else
    print_warning "Platform Backend non risponde ancora"
fi

cd "$PROJECT_ROOT"

# =============================================================================
# 7. Avvio Frontend (Vite/React)
# =============================================================================

print_step "7. Avvio Frontend (port 5173)"

cd "$PROJECT_ROOT/frontend"

npm run dev > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
PIDS+=($FRONTEND_PID)
echo $FRONTEND_PID > "$PROJECT_ROOT/logs/frontend.pid"

print_info "Attesa avvio Frontend..."
sleep 5  # Vite è veloce
print_success "Frontend avviato (PID: $FRONTEND_PID)"

cd "$PROJECT_ROOT"

# =============================================================================
# 8. Summary
# =============================================================================

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  ✓ Ambiente di sviluppo pronto!                               ║${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╠═══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  🌐 Frontend:        ${CYAN}http://localhost:5173${GREEN}                   ║${NC}"
echo -e "${GREEN}║  📡 VisuaLex API:    ${CYAN}http://localhost:5000${GREEN}                   ║${NC}"
echo -e "${GREEN}║  🔌 Backend:         ${CYAN}http://localhost:3001${GREEN}                   ║${NC}"

if [ "$START_MERLT" = true ]; then
echo -e "${GREEN}║  🧠 MERL-T API:      ${CYAN}http://localhost:8000/docs${GREEN}              ║${NC}"
fi

if [ "$START_DOCKER" = true ]; then
echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}║  📊 PostgreSQL:      localhost:5433                           ║${NC}"
echo -e "${GREEN}║  🔷 FalkorDB:        localhost:6380 (UI: 3000)                ║${NC}"
echo -e "${GREEN}║  🔍 Qdrant:          localhost:6333                           ║${NC}"
echo -e "${GREEN}║  ⚡ Redis:           localhost:6379                           ║${NC}"
fi

echo -e "${GREEN}║                                                               ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$START_MERLT" = true ]; then
    echo -e "${CYAN}🧠 Knowledge Graph Integration:${NC}"
    echo "   Il pulsante Brain (🧠) nella toolbar degli articoli"
    echo "   apre l'Inspector Panel per validare entità estratte."
    echo ""
fi

echo -e "${YELLOW}Premi Ctrl+C per terminare tutti i servizi${NC}"
echo ""

# Salva info runtime
cat > /tmp/visualex-dev.info << EOF
VisuaLex Development Environment
Started: $(date)
Mode: $mode
Frontend PID: $FRONTEND_PID
Backend PID: $BACKEND_PID
API PID: $API_PID
MERL-T PID: ${MERLT_PID:-N/A}
EOF

# =============================================================================
# 9. Live Logs
# =============================================================================

print_step "9. Live Logs (Ctrl+C per uscire)"

echo -e "${CYAN}Mostrando logs in tempo reale...${NC}"
echo ""

# Build log files list
LOG_FILES=("$PROJECT_ROOT/logs/visualex-api.log" "$PROJECT_ROOT/logs/backend.log" "$PROJECT_ROOT/logs/frontend.log")
if [ "$START_MERLT" = true ]; then
    LOG_FILES+=("$MERLT_ROOT/logs/merlt-api.log")
    echo -e "${CYAN}💡 Per vedere solo i log di MERL-T: tail -f $MERLT_ROOT/logs/merlt-api.log${NC}"
    echo ""
fi

# Wait a moment for log files to be created
sleep 2

# Use tail -f to show all logs with prefix
exec tail -f "${LOG_FILES[@]}" 2>/dev/null
