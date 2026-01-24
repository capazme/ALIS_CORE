#!/bin/bash

# MERLT Plugin Configuration Checker
# Verifica che tutti i file di configurazione siano corretti PRIMA del build

set -e

echo "🔧 MERLT Plugin Configuration Checker"
echo "======================================"
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

echo "1. Checking required files..."
echo "----------------------------"

# Check configuration files
FILES=(
    "vite.config.ts"
    "tsconfig.json"
    "tsconfig.plugin.json"
    "package.json"
    ".npmignore"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✓${NC} $file exists"
    else
        echo -e "${RED}✗${NC} $file missing"
        ERRORS=$((ERRORS + 1))
    fi
done

echo

echo "2. Checking plugin entry point..."
echo "----------------------------------"

if [ ! -f "src/plugin/index.ts" ]; then
    echo -e "${RED}✗${NC} src/plugin/index.ts missing"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}✓${NC} src/plugin/index.ts exists"

    # Check if it exports default
    if grep -q "export default" src/plugin/index.ts; then
        echo -e "${GREEN}✓${NC} Has default export"
    else
        echo -e "${RED}✗${NC} No default export found"
        ERRORS=$((ERRORS + 1))
    fi

    # Check if it imports from @visualex/platform
    if grep -q "@visualex/platform" src/plugin/index.ts; then
        echo -e "${GREEN}✓${NC} Imports platform types"
    else
        echo -e "${YELLOW}⚠${NC} Warning: No platform imports found"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo

echo "3. Checking package.json..."
echo "---------------------------"

if [ ! -f "package.json" ]; then
    echo -e "${RED}✗${NC} package.json missing"
    ERRORS=$((ERRORS + 1))
else
    # Check name
    NAME=$(node -p "require('./package.json').name" 2>/dev/null || echo "")
    if [ -n "$NAME" ]; then
        echo -e "${GREEN}✓${NC} package name: $NAME"
    else
        echo -e "${RED}✗${NC} No package name"
        ERRORS=$((ERRORS + 1))
    fi

    # Check main field
    MAIN=$(node -p "require('./package.json').main" 2>/dev/null || echo "")
    if [ "$MAIN" = "./dist/merlt-plugin.js" ]; then
        echo -e "${GREEN}✓${NC} main field correct: $MAIN"
    else
        echo -e "${RED}✗${NC} main field incorrect: $MAIN (expected: ./dist/merlt-plugin.js)"
        ERRORS=$((ERRORS + 1))
    fi

    # Check types field
    TYPES=$(node -p "require('./package.json').types" 2>/dev/null || echo "")
    if [ "$TYPES" = "./dist/types/plugin/index.d.ts" ]; then
        echo -e "${GREEN}✓${NC} types field correct: $TYPES"
    else
        echo -e "${YELLOW}⚠${NC} Warning: types field: $TYPES (expected: ./dist/types/plugin/index.d.ts)"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check scripts
    HAS_BUILD=$(node -p "require('./package.json').scripts['build:plugin']" 2>/dev/null || echo "")
    if [ -n "$HAS_BUILD" ] && [ "$HAS_BUILD" != "undefined" ]; then
        echo -e "${GREEN}✓${NC} build:plugin script exists"
    else
        echo -e "${RED}✗${NC} build:plugin script missing"
        ERRORS=$((ERRORS + 1))
    fi

    # Check peer dependencies
    HAS_REACT=$(node -p "require('./package.json').peerDependencies?.react" 2>/dev/null || echo "")
    if [ -n "$HAS_REACT" ] && [ "$HAS_REACT" != "undefined" ]; then
        echo -e "${GREEN}✓${NC} react peer dependency: $HAS_REACT"
    else
        echo -e "${RED}✗${NC} react peer dependency missing"
        ERRORS=$((ERRORS + 1))
    fi
fi

echo

echo "4. Checking vite.config.ts..."
echo "-----------------------------"

if [ ! -f "vite.config.ts" ]; then
    echo -e "${RED}✗${NC} vite.config.ts missing"
    ERRORS=$((ERRORS + 1))
else
    # Check if it has plugin mode
    if grep -q "mode === 'plugin'" vite.config.ts; then
        echo -e "${GREEN}✓${NC} Has plugin mode configuration"
    else
        echo -e "${RED}✗${NC} No plugin mode found"
        ERRORS=$((ERRORS + 1))
    fi

    # Check if it externalizes react
    if grep -q "external.*react" vite.config.ts; then
        echo -e "${GREEN}✓${NC} Externalizes react"
    else
        echo -e "${RED}✗${NC} React not externalized"
        ERRORS=$((ERRORS + 1))
    fi

    # Check lib entry
    if grep -q "entry.*src/plugin/index.ts" vite.config.ts; then
        echo -e "${GREEN}✓${NC} Entry point configured"
    else
        echo -e "${YELLOW}⚠${NC} Warning: Entry point not found in config"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check format
    if grep -q "formats.*es" vite.config.ts; then
        echo -e "${GREEN}✓${NC} ESM format configured"
    else
        echo -e "${YELLOW}⚠${NC} Warning: ESM format not explicitly set"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo

echo "5. Checking tsconfig.plugin.json..."
echo "-----------------------------------"

if [ ! -f "tsconfig.plugin.json" ]; then
    echo -e "${RED}✗${NC} tsconfig.plugin.json missing"
    ERRORS=$((ERRORS + 1))
else
    # Check extends
    if grep -q '"extends".*tsconfig.json' tsconfig.plugin.json; then
        echo -e "${GREEN}✓${NC} Extends base tsconfig"
    else
        echo -e "${YELLOW}⚠${NC} Warning: Doesn't extend base tsconfig"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check emitDeclarationOnly
    if grep -q '"emitDeclarationOnly".*true' tsconfig.plugin.json; then
        echo -e "${GREEN}✓${NC} emitDeclarationOnly enabled"
    else
        echo -e "${RED}✗${NC} emitDeclarationOnly not enabled"
        ERRORS=$((ERRORS + 1))
    fi

    # Check outDir
    if grep -q '"outDir".*dist/types' tsconfig.plugin.json; then
        echo -e "${GREEN}✓${NC} outDir points to dist/types"
    else
        echo -e "${YELLOW}⚠${NC} Warning: outDir not set to dist/types"
        WARNINGS=$((WARNINGS + 1))
    fi
fi

echo

echo "6. Checking dependencies..."
echo "---------------------------"

if [ -f "package.json" ]; then
    # Check if node_modules exists
    if [ -d "node_modules" ]; then
        echo -e "${GREEN}✓${NC} node_modules exists"
    else
        echo -e "${YELLOW}⚠${NC} Warning: node_modules not found (run: npm install)"
        WARNINGS=$((WARNINGS + 1))
    fi

    # Check critical dependencies
    DEPS=("vite" "@vitejs/plugin-react" "typescript" "react" "react-dom")
    for dep in "${DEPS[@]}"; do
        if [ -d "node_modules/$dep" ]; then
            echo -e "${GREEN}✓${NC} $dep installed"
        else
            echo -e "${RED}✗${NC} $dep not installed"
            ERRORS=$((ERRORS + 1))
        fi
    done
fi

echo

echo "7. Checking documentation..."
echo "----------------------------"

DOCS=("BUILD.md" "PLUGIN.md" "USAGE.md" "README-BUILD-SYSTEM.md" "BUILD-SYSTEM-SUMMARY.md")
DOC_FOUND=0

for doc in "${DOCS[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "${GREEN}✓${NC} $doc exists"
        DOC_FOUND=$((DOC_FOUND + 1))
    fi
done

if [ $DOC_FOUND -eq 0 ]; then
    echo -e "${YELLOW}⚠${NC} Warning: No documentation files found"
    WARNINGS=$((WARNINGS + 1))
elif [ $DOC_FOUND -lt 3 ]; then
    echo -e "${YELLOW}⚠${NC} Warning: Some documentation missing ($DOC_FOUND/5)"
    WARNINGS=$((WARNINGS + 1))
fi

echo

echo "======================================"
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✅ Configuration looks good!${NC}"
    echo
    echo "Next steps:"
    echo "  1. npm install (if not done)"
    echo "  2. npm run build:plugin"
    echo "  3. ./verify-build.sh"
    echo
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}⚠ $WARNINGS warning(s) found (non-critical)${NC}"
    fi
    exit 0
else
    echo -e "${RED}❌ Configuration has errors!${NC}"
    echo
    echo -e "${RED}$ERRORS error(s) found${NC}"
    if [ $WARNINGS -gt 0 ]; then
        echo -e "${YELLOW}$WARNINGS warning(s) found${NC}"
    fi
    echo
    echo "Fix the errors above, then run this script again."
    exit 1
fi
