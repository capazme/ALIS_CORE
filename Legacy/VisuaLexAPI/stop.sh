#!/bin/bash

# =============================================================================
# VisuaLex + MERL-T Stop Script
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}Stopping VisuaLex + MERL-T services...${NC}\n"

# Kill processes by port
kill_port() {
    local port=$1
    local name=$2
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        lsof -ti:$port | xargs kill -9 2>/dev/null
        echo -e "  ${GREEN}✓${NC} $name (port $port) stopped"
    else
        echo -e "  ${YELLOW}○${NC} $name (port $port) not running"
    fi
}

kill_port 5173 "Frontend (Vite)"
kill_port 3001 "Platform Backend"
kill_port 5000 "VisuaLex API"
kill_port 8000 "MERL-T API"

# Optionally stop Docker containers
echo ""
read -p "Stop MERL-T Docker containers? [y/N]: " stop_docker

if [[ "$stop_docker" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    MERLT_ROOT="/Users/gpuzio/Desktop/CODE/MERL-T_alpha"
    if [ -f "$MERLT_ROOT/docker-compose.dev.yml" ]; then
        cd "$MERLT_ROOT"
        docker-compose -f docker-compose.dev.yml down
        echo -e "  ${GREEN}✓${NC} Docker containers stopped"
    fi
fi

echo -e "\n${GREEN}All services stopped.${NC}\n"
