# MERLT Plugin Build System - Complete Guide

Guida completa al sistema di build per il plugin MERLT standalone.

## Quick Start

```bash
# Install dependencies
npm install

# Build plugin
npm run build:plugin

# Verify build
chmod +x verify-build.sh
./verify-build.sh

# Test in platform
cd ../visualex-platform
npm install file:../visualex-merlt/frontend
```

## File Structure

```
frontend/
├── src/
│   ├── plugin/
│   │   └── index.ts              # Plugin entry point ⭐
│   ├── components/
│   │   ├── MerltSidebarPanel.tsx
│   │   └── MerltToolbar.tsx
│   ├── services/
│   │   ├── merltInit.ts
│   │   └── tracking.ts
│   ├── hooks/
│   ├── store/
│   └── types/
│
├── dist/                          # Build output (gitignored)
│   ├── merlt-plugin.js           # ESM bundle
│   ├── merlt-plugin.js.map       # Source maps
│   └── types/
│       └── plugin/
│           └── index.d.ts        # Type declarations
│
├── vite.config.ts                # Vite config (dual mode)
├── tsconfig.json                 # Base TS config
├── tsconfig.plugin.json          # Plugin TS config
├── package.json                  # Package metadata
├── .npmignore                    # npm publish ignore
│
├── BUILD.md                      # Build system details
├── PLUGIN.md                     # Plugin architecture
├── USAGE.md                      # Integration guide
├── platform-types-example.d.ts   # Required platform types
└── verify-build.sh               # Build verification script
```

## Build System Components

### 1. Vite Configuration (vite.config.ts)

**Dual-mode config:**

```typescript
defineConfig(({ mode }) => {
  if (mode === 'plugin') {
    // Library mode for plugin build
    return {
      build: {
        lib: { entry: 'src/plugin/index.ts', formats: ['es'] },
        rollupOptions: {
          external: ['react', 'react-dom'],
        },
      },
    };
  }
  // Regular dev mode
  return { server: { port: 5174 } };
});
```

**Key settings:**
- `formats: ['es']` - ESM only
- `external: ['react', 'react-dom']` - Peer dependencies
- `cssCodeSplit: false` - Inline CSS
- `minify: 'esbuild'` - Production optimization

### 2. TypeScript Configuration

**Base (tsconfig.json):**
- `noEmit: true` - Don't emit for dev
- `strict: true` - Strict mode
- Path aliases: `@/* -> src/*`

**Plugin (tsconfig.plugin.json):**
- `extends: "./tsconfig.json"`
- `noEmit: false` - Do emit
- `emitDeclarationOnly: true` - Only .d.ts files
- `outDir: "./dist/types"`

### 3. Package Configuration (package.json)

**Entry points:**
```json
{
  "main": "./dist/merlt-plugin.js",
  "module": "./dist/merlt-plugin.js",
  "types": "./dist/types/plugin/index.d.ts",
  "exports": {
    ".": {
      "types": "./dist/types/plugin/index.d.ts",
      "import": "./dist/merlt-plugin.js"
    }
  }
}
```

**Scripts:**
```json
{
  "build:plugin": "npm run build:plugin:types && npm run build:plugin:bundle",
  "build:plugin:types": "tsc --project tsconfig.plugin.json",
  "build:plugin:bundle": "vite build --mode plugin"
}
```

**Dependencies:**
- `dependencies` - Bundled in output
- `peerDependencies` - Provided by platform
- `devDependencies` - Build tools only

## Build Process

### Step 1: Type Generation

```bash
npm run build:plugin:types
```

**What it does:**
1. Runs `tsc --project tsconfig.plugin.json`
2. Reads from `src/plugin/index.ts` and dependencies
3. Generates `.d.ts` files in `dist/types/`
4. Creates declaration maps for IDE navigation

**Output:**
```
dist/types/
├── plugin/index.d.ts
├── components/MerltSidebarPanel.d.ts
├── services/merltInit.d.ts
└── ...
```

### Step 2: Bundle Generation

```bash
npm run build:plugin:bundle
```

**What it does:**
1. Runs `vite build --mode plugin`
2. Activates plugin mode in vite.config.ts
3. Creates library build from `src/plugin/index.ts`
4. Externalizes react/react-dom
5. Inlines CSS and other assets
6. Minifies with esbuild
7. Generates source maps

**Output:**
```
dist/merlt-plugin.js
dist/merlt-plugin.js.map
```

### Step 3: Verification

```bash
./verify-build.sh
```

**What it checks:**
1. ✅ Files exist (bundle, types, sourcemap)
2. ✅ Bundle size (<2MB)
3. ✅ Exports valid (default export with manifest)
4. ✅ React is external (not bundled)
5. ✅ CSS is inlined
6. ✅ package.json fields correct

## Plugin Architecture

### Entry Point (src/plugin/index.ts)

```typescript
import type { Plugin } from '@visualex/platform/lib/plugins';

const merltPlugin: Plugin = {
  manifest: {
    id: 'merlt',
    name: 'MERLT Research',
    version: '1.0.0',
    requiredFeatures: ['merlt'],
    subscribedEvents: ['article:viewed', ...],
    contributedSlots: ['article-sidebar', 'article-toolbar'],
  },

  async initialize(context) {
    // Setup services, connections, etc.
    await initializeMerltServices(context);

    // Return cleanup function
    return () => shutdownMerltServices();
  },

  getSlotComponents() {
    return [
      { slot: 'article-sidebar', component: MerltSidebarPanel, priority: 100 },
      { slot: 'article-toolbar', component: MerltToolbar, priority: 50 },
    ];
  },

  getEventHandlers() {
    return {
      'article:viewed': (data) => trackArticleView(data),
      // ...
    };
  },
};

export default merltPlugin;
```

### Platform Contract

**Required types from @visualex/platform:**
- `Plugin` - Main plugin interface
- `PluginManifest` - Metadata
- `PluginContext` - Init context
- `PluginEventHandler` - Event handler type
- `SlotComponent` - UI slot definition

See `platform-types-example.d.ts` for full type definitions.

## Integration with visualex-platform

### 1. Install Package

```bash
# Local development
npm install file:../visualex-merlt/frontend

# Production
npm install @visualex/merlt-plugin
```

### 2. Dynamic Import

```typescript
// src/plugins/merlt-loader.ts
export const loadMerltPlugin = async () => {
  const module = await import('@visualex/merlt-plugin');
  return module.default;
};
```

### 3. Initialize Plugin

```typescript
// src/plugins/manager.ts
const plugin = await loadMerltPlugin();
const context = {
  apiBaseUrl: import.meta.env.VITE_API_URL,
  getAuthToken: () => getToken(),
  user: { id: userId, features: ['merlt'] },
};
const cleanup = await plugin.initialize(context);
```

### 4. Render Slot Components

```typescript
// src/components/ArticlePage.tsx
const sidebarComponents = plugin.getSlotComponents()
  .filter(s => s.slot === 'article-sidebar')
  .sort((a, b) => (b.priority || 0) - (a.priority || 0))
  .map(s => s.component);

return (
  <aside>
    {sidebarComponents.map((Component, i) => (
      <Component key={i} urn={urn} />
    ))}
  </aside>
);
```

### 5. Emit Events

```typescript
// src/hooks/useArticle.ts
const handlers = plugin.getEventHandlers();
handlers['article:viewed']?.({ urn, articleId, userId });
```

## Dependency Management

### Bundled Dependencies

Included in `dist/merlt-plugin.js`:
- `@tanstack/react-query` - Data fetching
- `zustand` - State management
- `recharts` - Charts
- `framer-motion` - Animations
- `lucide-react` - Icons
- `reagraph` - Graph visualization

**Why bundled:**
- Plugin-specific versions
- No version conflicts
- Self-contained
- Predictable behavior

### External Dependencies

NOT included, provided by platform:
- `react` >=18.0.0
- `react-dom` >=18.0.0

**Why external:**
- Avoid multiple React instances
- Smaller bundle
- Shared context
- Better performance

## Development Workflow

### Local Development

```bash
# Terminal 1: Dev server
npm run dev

# Terminal 2: Watch types
npm run build:plugin:types -- --watch

# Browser
open http://localhost:5174
```

### Building for Production

```bash
# Full build
npm run build:plugin

# Verify
./verify-build.sh

# Test import
node -e "import('./dist/merlt-plugin.js').then(m => console.log(m.default))"
```

### Testing in Platform

```bash
# Build plugin
cd visualex-merlt/frontend
npm run build:plugin

# Install in platform
cd ../../visualex-platform
npm install file:../visualex-merlt/frontend

# Restart platform dev server
npm run dev
```

### Publishing to npm

```bash
# Build
npm run build:plugin

# Test package contents
npm pack --dry-run

# Publish
npm publish --access public
```

## Troubleshooting

### Issue: "React is not defined"

**Cause:** React bundled instead of external

**Fix:** Check vite.config.ts:
```typescript
external: ['react', 'react-dom', /^react\//, /^react-dom\//, 'react/jsx-runtime']
```

### Issue: "Cannot find module '@visualex/platform'"

**Cause:** Platform types not available during build

**Fix:** Create stub types temporarily:
```typescript
// types/platform.d.ts
declare module '@visualex/platform/lib/plugins' {
  export interface Plugin { ... }
}
```

### Issue: Bundle too large (>2MB)

**Cause:** Heavy dependencies bundled

**Solutions:**
1. Analyze bundle:
   ```bash
   npx vite-bundle-visualizer
   ```

2. Externalize heavy deps if platform has them:
   ```typescript
   external: ['react', 'react-dom', 'recharts']
   ```

3. Lazy load heavy components:
   ```typescript
   const HeavyComponent = lazy(() => import('./HeavyComponent'));
   ```

### Issue: CSS not applied

**Cause:** CSS not inlined or imported

**Fix:**
1. Import CSS in plugin entry:
   ```typescript
   import '../styles/plugin.css';
   ```

2. Verify vite config:
   ```typescript
   cssCodeSplit: false
   ```

### Issue: Type errors in platform

**Cause:** Type definitions incompatible

**Fix:**
1. Check type exports:
   ```typescript
   cat dist/types/plugin/index.d.ts
   ```

2. Verify platform imports correct path:
   ```typescript
   import type { Plugin } from '@visualex/merlt-plugin';
   ```

## Performance Optimization

### Bundle Size Targets

- Raw: <2MB
- Gzip: <500KB
- First load: <1s
- Time to interactive: <2s

### Optimization Techniques

1. **Tree shaking** (automatic with Vite)
2. **Minification** (esbuild)
3. **Code splitting** (optional with preserveModules)
4. **Lazy loading** (React.lazy)
5. **CSS inlining** (single file output)

### Bundle Analysis

```bash
# Size breakdown
npx vite-bundle-visualizer dist/merlt-plugin.js

# Duplicate detection
npx duplicate-package-checker
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Build Plugin
  run: |
    cd frontend
    npm ci
    npm run build:plugin
    ./verify-build.sh

- name: Upload artifacts
  uses: actions/upload-artifact@v4
  with:
    name: plugin-dist
    path: frontend/dist/
```

## References

- **BUILD.md** - Detailed build system documentation
- **PLUGIN.md** - Plugin architecture and design
- **USAGE.md** - Integration guide for platform
- **platform-types-example.d.ts** - Required platform types

## Support

For issues or questions:
1. Check verify-build.sh output
2. Review BUILD.md troubleshooting section
3. Check vite.config.ts configuration
4. Verify package.json exports
5. Test with node import script
