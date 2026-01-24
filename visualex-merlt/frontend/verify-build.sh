#!/bin/bash

# MERLT Plugin Build Verification Script
# Verifica che il build del plugin sia corretto

set -e

echo "🔍 MERLT Plugin Build Verification"
echo "===================================="
echo

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if dist exists
if [ ! -d "dist" ]; then
    echo -e "${RED}❌ dist/ directory not found${NC}"
    echo "Run: npm run build:plugin"
    exit 1
fi

echo "1. Checking bundle files..."
echo "----------------------------"

# Check main bundle
if [ ! -f "dist/merlt-plugin.js" ]; then
    echo -e "${RED}❌ dist/merlt-plugin.js not found${NC}"
    exit 1
else
    SIZE=$(stat -f%z dist/merlt-plugin.js 2>/dev/null || stat -c%s dist/merlt-plugin.js)
    SIZE_MB=$(echo "scale=2; $SIZE / 1048576" | bc)

    echo -e "${GREEN}✓${NC} dist/merlt-plugin.js found (${SIZE_MB}MB)"

    if [ $SIZE -gt 2097152 ]; then
        echo -e "${YELLOW}⚠ Warning: Bundle > 2MB, consider optimization${NC}"
    fi
fi

# Check sourcemap
if [ ! -f "dist/merlt-plugin.js.map" ]; then
    echo -e "${YELLOW}⚠ Warning: No sourcemap found${NC}"
else
    echo -e "${GREEN}✓${NC} dist/merlt-plugin.js.map found"
fi

echo

echo "2. Checking type declarations..."
echo "--------------------------------"

# Check types directory
if [ ! -d "dist/types" ]; then
    echo -e "${RED}❌ dist/types/ directory not found${NC}"
    echo "Run: npm run build:plugin:types"
    exit 1
else
    echo -e "${GREEN}✓${NC} dist/types/ directory found"
fi

# Check main type file
if [ ! -f "dist/types/plugin/index.d.ts" ]; then
    echo -e "${RED}❌ dist/types/plugin/index.d.ts not found${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} dist/types/plugin/index.d.ts found"
fi

echo

echo "3. Checking exports..."
echo "----------------------"

# Check if module exports default
EXPORT_CHECK=$(node -e "
import('./dist/merlt-plugin.js')
  .then(m => {
    if (!m.default) {
      console.error('No default export');
      process.exit(1);
    }
    if (!m.default.manifest) {
      console.error('No manifest in default export');
      process.exit(1);
    }
    if (!m.default.initialize) {
      console.error('No initialize function');
      process.exit(1);
    }
    console.log('Plugin ID:', m.default.manifest.id);
    console.log('Plugin Name:', m.default.manifest.name);
    console.log('Version:', m.default.manifest.version);
    console.log('Slots:', m.default.manifest.contributedSlots?.join(', ') || 'none');
  })
  .catch(err => {
    console.error('Import failed:', err.message);
    process.exit(1);
  });
" 2>&1)

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Plugin exports valid"
    echo "$EXPORT_CHECK" | sed 's/^/  /'
else
    echo -e "${RED}❌ Plugin export check failed${NC}"
    echo "$EXPORT_CHECK"
    exit 1
fi

echo

echo "4. Checking for common issues..."
echo "--------------------------------"

# Check if React is bundled (should be external)
if grep -q "function createElement" dist/merlt-plugin.js; then
    echo -e "${RED}❌ React appears to be bundled (should be external)${NC}"
    exit 1
else
    echo -e "${GREEN}✓${NC} React is external"
fi

# Check for source maps reference
if grep -q "sourceMappingURL" dist/merlt-plugin.js; then
    echo -e "${GREEN}✓${NC} Source map reference found"
else
    echo -e "${YELLOW}⚠ Warning: No source map reference${NC}"
fi

# Check if CSS is present (should be inline)
if grep -q "style" dist/merlt-plugin.js; then
    echo -e "${GREEN}✓${NC} CSS appears to be inlined"
else
    echo -e "${YELLOW}⚠ Warning: No CSS found in bundle${NC}"
fi

echo

echo "5. Package.json validation..."
echo "----------------------------"

# Check package.json exports
MAIN=$(node -p "require('./package.json').main" 2>/dev/null)
MODULE=$(node -p "require('./package.json').module" 2>/dev/null)
TYPES=$(node -p "require('./package.json').types" 2>/dev/null)

if [ "$MAIN" = "./dist/merlt-plugin.js" ]; then
    echo -e "${GREEN}✓${NC} package.json main field correct"
else
    echo -e "${RED}❌ package.json main field incorrect: $MAIN${NC}"
fi

if [ "$MODULE" = "./dist/merlt-plugin.js" ]; then
    echo -e "${GREEN}✓${NC} package.json module field correct"
else
    echo -e "${RED}❌ package.json module field incorrect: $MODULE${NC}"
fi

if [ "$TYPES" = "./dist/types/plugin/index.d.ts" ]; then
    echo -e "${GREEN}✓${NC} package.json types field correct"
else
    echo -e "${RED}❌ package.json types field incorrect: $TYPES${NC}"
fi

echo

echo "====================================="
echo -e "${GREEN}✅ Build verification passed!${NC}"
echo
echo "Next steps:"
echo "  1. Test in visualex-platform: npm install file:../visualex-merlt/frontend"
echo "  2. Dynamic import: import('@visualex/merlt-plugin')"
echo "  3. Initialize plugin with context"
echo
echo "For more info, see BUILD.md and USAGE.md"
