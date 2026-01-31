#!/usr/bin/env bash
#
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                         ALIS_CORE Development CLI                         ║
# ║                                                                           ║
# ║  Central entry point for managing the ALIS monorepo                       ║
# ║  Usage: ./dev.sh <command> [options]                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================
ROOT="$(cd "$(dirname "$0")" && pwd)"
PLATFORM_DIR="$ROOT/visualex-platform"
API_DIR="$ROOT/visualex-api"
MERLT_DIR="$ROOT/visualex-merlt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ============================================================================
# Helper Functions
# ============================================================================
print_header() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════════════════╗"
    echo "║                         ALIS_CORE Development CLI                         ║"
    echo "╚═══════════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker Desktop."
        exit 1
    fi
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker Desktop."
        exit 1
    fi
}

check_node() {
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 20+."
        exit 1
    fi
}

check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed."
        exit 1
    fi
}

# ============================================================================
# Commands
# ============================================================================

cmd_help() {
    print_header
    echo -e "${BOLD}Usage:${NC} ./dev.sh <command> [options]"
    echo ""
    echo -e "${BOLD}Development Commands:${NC}"
    echo -e "  ${GREEN}start${NC}              Start development environment (default: vanilla)"
    echo -e "  ${GREEN}start --merlt${NC}      Start with MERL-T plugin enabled"
    echo -e "  ${GREEN}start --minimal${NC}    Start only database (for running services manually)"
    echo -e "  ${GREEN}stop${NC}               Stop all running services"
    echo -e "  ${GREEN}restart${NC}            Restart all services"
    echo -e "  ${GREEN}logs${NC}               Show logs from all services"
    echo -e "  ${GREEN}logs <service>${NC}     Show logs for specific service (frontend|backend|api|postgres)"
    echo ""
    echo -e "${BOLD}Testing Commands:${NC}"
    echo -e "  ${GREEN}test${NC}               Run all tests (unit + integration)"
    echo -e "  ${GREEN}test:unit${NC}          Run unit tests only"
    echo -e "  ${GREEN}test:integration${NC}   Run integration tests only"
    echo -e "  ${GREEN}test:e2e${NC}           Run E2E tests with Playwright"
    echo -e "  ${GREEN}test:e2e --ui${NC}      Run E2E tests with Playwright UI"
    echo -e "  ${GREEN}test:coverage${NC}      Run tests with coverage report"
    echo ""
    echo -e "${BOLD}Database Commands:${NC}"
    echo -e "  ${GREEN}db:studio${NC}          Open Prisma Studio (database GUI)"
    echo -e "  ${GREEN}db:migrate${NC}         Run database migrations"
    echo -e "  ${GREEN}db:reset${NC}           Reset database (WARNING: deletes all data)"
    echo -e "  ${GREEN}db:seed${NC}            Seed database with test data"
    echo ""
    echo -e "${BOLD}Build Commands:${NC}"
    echo -e "  ${GREEN}build${NC}              Build all packages"
    echo -e "  ${GREEN}build:frontend${NC}     Build frontend only"
    echo -e "  ${GREEN}build:backend${NC}      Build backend only"
    echo -e "  ${GREEN}lint${NC}               Run linters on all packages"
    echo ""
    echo -e "${BOLD}Utility Commands:${NC}"
    echo -e "  ${GREEN}install${NC}            Install dependencies for all packages"
    echo -e "  ${GREEN}clean${NC}              Clean build artifacts and node_modules"
    echo -e "  ${GREEN}status${NC}             Show status of all services"
    echo -e "  ${GREEN}shell <service>${NC}    Open shell in running container"
    echo ""
    echo -e "${BOLD}Examples:${NC}"
    echo "  ./dev.sh start                    # Start vanilla VisuaLex"
    echo "  ./dev.sh start --merlt            # Start with MERL-T"
    echo "  ./dev.sh test:e2e --ui            # Run E2E tests with UI"
    echo "  ./dev.sh logs backend             # View backend logs"
    echo ""
}

cmd_start() {
    local mode="vanilla"
    local minimal=false

    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            --merlt)
                mode="merlt"
                shift
                ;;
            --minimal)
                minimal=true
                shift
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    check_docker
    print_header

    cd "$PLATFORM_DIR"

    # Create .env if it doesn't exist
    if [ ! -f ".env" ] && [ -f ".env.example" ]; then
        cp .env.example .env
        print_info "Created .env from .env.example"
    fi

    if [ "$minimal" = true ]; then
        print_info "Starting minimal environment (PostgreSQL only)..."
        docker compose up -d postgres
        print_success "PostgreSQL is running on localhost:5432"
        echo ""
        echo -e "${YELLOW}To start services manually:${NC}"
        echo "  cd visualex-platform/backend && npm run dev"
        echo "  cd visualex-platform/frontend && npm run dev"
        echo "  cd visualex-api && python -m uvicorn app:app --reload"
        return
    fi

    if [ "$mode" = "merlt" ]; then
        print_info "Starting VisuaLex with MERL-T plugin..."
        export VITE_MERLT_ENABLED=true
        export MERLT_ENABLED=true
        docker compose up --build
    else
        print_info "Starting VisuaLex (vanilla)..."
        export VITE_MERLT_ENABLED=false
        export MERLT_ENABLED=false
        docker compose up --build frontend backend python-api postgres
    fi
}

cmd_stop() {
    print_header
    print_info "Stopping all services..."
    cd "$PLATFORM_DIR"
    docker compose down
    print_success "All services stopped"
}

cmd_restart() {
    cmd_stop
    cmd_start "$@"
}

cmd_logs() {
    cd "$PLATFORM_DIR"

    if [ $# -eq 0 ]; then
        docker compose logs -f
    else
        case $1 in
            frontend|backend|postgres|python-api)
                docker compose logs -f "$1"
                ;;
            api)
                docker compose logs -f python-api
                ;;
            *)
                print_error "Unknown service: $1"
                echo "Available: frontend, backend, api, postgres"
                exit 1
                ;;
        esac
    fi
}

cmd_test() {
    print_header
    print_info "Running all tests..."

    echo ""
    echo -e "${BOLD}=== Frontend Unit Tests ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm test -- --run || true

    echo ""
    echo -e "${BOLD}=== Backend Unit Tests ===${NC}"
    cd "$PLATFORM_DIR/backend"
    npm test -- tests/unit || true

    echo ""
    print_success "Test run complete!"
}

cmd_test_unit() {
    print_header
    print_info "Running unit tests..."

    echo ""
    echo -e "${BOLD}=== Frontend Unit Tests ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm test -- --run

    echo ""
    echo -e "${BOLD}=== Backend Unit Tests ===${NC}"
    cd "$PLATFORM_DIR/backend"
    npm test -- tests/unit
}

cmd_test_integration() {
    print_header
    print_info "Running integration tests..."
    print_warning "Make sure PostgreSQL is running (./dev.sh start --minimal)"

    cd "$PLATFORM_DIR/backend"
    npm test -- tests/integration tests/auth.test.ts tests/profile.test.ts tests/consent.test.ts tests/authority.test.ts tests/privacy.test.ts
}

cmd_test_e2e() {
    print_header
    local ui_mode=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --ui)
                ui_mode=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done

    cd "$PLATFORM_DIR/frontend"

    if [ "$ui_mode" = true ]; then
        print_info "Starting Playwright UI..."
        npx playwright test --ui
    else
        print_info "Running E2E tests..."
        print_warning "Make sure the app is running (./dev.sh start)"
        npx playwright test
    fi
}

cmd_test_coverage() {
    print_header
    print_info "Running tests with coverage..."

    echo ""
    echo -e "${BOLD}=== Frontend Coverage ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm run test:coverage -- --run

    echo ""
    echo -e "${BOLD}=== Backend Coverage ===${NC}"
    cd "$PLATFORM_DIR/backend"
    npm run test:coverage
}

cmd_db_studio() {
    print_info "Opening Prisma Studio..."
    cd "$PLATFORM_DIR/backend"
    npx prisma studio
}

cmd_db_migrate() {
    print_info "Running database migrations..."
    cd "$PLATFORM_DIR/backend"
    npx prisma migrate dev
    print_success "Migrations complete!"
}

cmd_db_reset() {
    print_warning "This will delete all data in the database!"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PLATFORM_DIR/backend"
        npx prisma migrate reset --force
        print_success "Database reset complete!"
    else
        print_info "Cancelled"
    fi
}

cmd_db_seed() {
    print_info "Seeding database..."
    cd "$PLATFORM_DIR/backend"
    npm run db:seed
    print_success "Database seeded!"
}

cmd_build() {
    print_header
    print_info "Building all packages..."

    echo ""
    echo -e "${BOLD}=== Building Frontend ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm run build

    echo ""
    echo -e "${BOLD}=== Building Backend ===${NC}"
    cd "$PLATFORM_DIR/backend"
    npm run build

    print_success "Build complete!"
}

cmd_build_frontend() {
    print_info "Building frontend..."
    cd "$PLATFORM_DIR/frontend"
    npm run build
    print_success "Frontend build complete!"
}

cmd_build_backend() {
    print_info "Building backend..."
    cd "$PLATFORM_DIR/backend"
    npm run build
    print_success "Backend build complete!"
}

cmd_lint() {
    print_header
    print_info "Running linters..."

    echo ""
    echo -e "${BOLD}=== Frontend Lint ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm run lint || true

    print_success "Lint complete!"
}

cmd_install() {
    print_header
    print_info "Installing dependencies..."

    echo ""
    echo -e "${BOLD}=== Frontend Dependencies ===${NC}"
    cd "$PLATFORM_DIR/frontend"
    npm install

    echo ""
    echo -e "${BOLD}=== Backend Dependencies ===${NC}"
    cd "$PLATFORM_DIR/backend"
    npm install
    npx prisma generate

    echo ""
    echo -e "${BOLD}=== Python API Dependencies ===${NC}"
    cd "$API_DIR"
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    fi

    print_success "All dependencies installed!"
}

cmd_clean() {
    print_header
    print_warning "This will remove node_modules and build artifacts."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Cleaning..."

        rm -rf "$PLATFORM_DIR/frontend/node_modules" 2>/dev/null || true
        rm -rf "$PLATFORM_DIR/frontend/dist" 2>/dev/null || true
        rm -rf "$PLATFORM_DIR/backend/node_modules" 2>/dev/null || true
        rm -rf "$PLATFORM_DIR/backend/dist" 2>/dev/null || true

        print_success "Clean complete!"
    else
        print_info "Cancelled"
    fi
}

cmd_status() {
    print_header
    print_info "Service Status:"
    echo ""

    cd "$PLATFORM_DIR"
    docker compose ps 2>/dev/null || echo "No services running"

    echo ""
    print_info "Port Usage:"
    echo "  Frontend:  http://localhost:5173"
    echo "  Backend:   http://localhost:3001"
    echo "  Python API: http://localhost:5000"
    echo "  PostgreSQL: localhost:5432"
    echo "  Prisma Studio: http://localhost:5555 (when running)"
}

cmd_shell() {
    if [ $# -eq 0 ]; then
        print_error "Please specify a service: frontend, backend, api, postgres"
        exit 1
    fi

    cd "$PLATFORM_DIR"

    case $1 in
        frontend)
            docker compose exec frontend sh
            ;;
        backend)
            docker compose exec backend sh
            ;;
        api|python-api)
            docker compose exec python-api bash
            ;;
        postgres)
            docker compose exec postgres psql -U visualex -d visualex
            ;;
        *)
            print_error "Unknown service: $1"
            exit 1
            ;;
    esac
}

# ============================================================================
# Main Entry Point
# ============================================================================
main() {
    if [ $# -eq 0 ]; then
        cmd_help
        exit 0
    fi

    local command=$1
    shift

    case $command in
        help|--help|-h)
            cmd_help
            ;;
        start)
            cmd_start "$@"
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart "$@"
            ;;
        logs)
            cmd_logs "$@"
            ;;
        test)
            cmd_test
            ;;
        test:unit)
            cmd_test_unit
            ;;
        test:integration)
            cmd_test_integration
            ;;
        test:e2e)
            cmd_test_e2e "$@"
            ;;
        test:coverage)
            cmd_test_coverage
            ;;
        db:studio)
            cmd_db_studio
            ;;
        db:migrate)
            cmd_db_migrate
            ;;
        db:reset)
            cmd_db_reset
            ;;
        db:seed)
            cmd_db_seed
            ;;
        build)
            cmd_build
            ;;
        build:frontend)
            cmd_build_frontend
            ;;
        build:backend)
            cmd_build_backend
            ;;
        lint)
            cmd_lint
            ;;
        install)
            cmd_install
            ;;
        clean)
            cmd_clean
            ;;
        status)
            cmd_status
            ;;
        shell)
            cmd_shell "$@"
            ;;
        *)
            print_error "Unknown command: $command"
            echo "Run './dev.sh help' for usage"
            exit 1
            ;;
    esac
}

main "$@"
