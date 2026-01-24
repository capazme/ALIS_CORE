# MERLT Plugin Build System

Sistema di build configurato per creare un plugin standalone caricabile dinamicamente da visualex-platform.

## File di Configurazione

### 1. vite.config.ts

Configurazione Vite con due modalita':

- **Dev mode** (`npm run dev`): Dev server Vite standard sulla porta 5174
- **Plugin mode** (`npm run build:plugin`): Library mode che produce ESM bundle

**Caratteristiche plugin mode:**

```typescript
{
  entry: 'src/plugin/index.ts',
  format: 'es',                    // ESM only
  external: ['react', 'react-dom'], // Peer deps non bundlate
  cssCodeSplit: false,             // CSS inline nel bundle
  inlineDynamicImports: true,      // Single file output
  minify: 'esbuild',               // Minificazione produzione
  sourcemap: true,                 // Debug maps
}
```

### 2. tsconfig.plugin.json

TypeScript config dedicato per generare type declarations:

```json
{
  "extends": "./tsconfig.json",
  "compilerOptions": {
    "noEmit": false,
    "emitDeclarationOnly": true,
    "outDir": "./dist/types"
  },
  "include": ["src/plugin/", "src/components/", "src/services/", ...],
  "exclude": ["src/main.tsx", "src/App.tsx"]
}
```

### 3. package.json

Entry points configurati per ESM:

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
  },
  "peerDependencies": {
    "react": ">=18.0.0",
    "react-dom": ">=18.0.0"
  }
}
```

## Comandi di Build

### Build Completo Plugin

```bash
npm run build:plugin
```

Esegue in sequenza:
1. `build:plugin:types` - Type declarations con tsc
2. `build:plugin:bundle` - Bundle ESM con Vite

### Build Solo Types

```bash
npm run build:plugin:types
```

Utile per:
- Verificare errori TypeScript
- Testare type exports
- Debug type resolution

### Build Solo Bundle

```bash
npm run build:plugin:bundle
```

Utile per:
- Test build senza types
- Iterazione rapida
- Debug Vite config

### Dev Mode

```bash
npm run dev
```

Avvia Vite dev server per sviluppo locale del plugin.

## Output

Dopo `npm run build:plugin` avrai:

```
dist/
├── merlt-plugin.js           # Bundle ESM principale (~500KB gzip)
├── merlt-plugin.js.map       # Source map
└── types/
    ├── plugin/
    │   └── index.d.ts        # Type entry point
    ├── components/
    │   ├── MerltSidebarPanel.d.ts
    │   └── MerltToolbar.d.ts
    ├── services/
    │   ├── merltInit.d.ts
    │   └── tracking.d.ts
    └── types/
        └── merlt.d.ts
```

## Verifica Build

### 1. Check File Size

```bash
ls -lh dist/merlt-plugin.js
```

Target: <500KB (gzip), <2MB (raw)

### 2. Check Exports

```bash
node -e "import('./dist/merlt-plugin.js').then(m => console.log(m.default))"
```

Deve stampare l'oggetto Plugin con manifest, initialize, getSlotComponents, etc.

### 3. Check Types

```bash
npx tsc --noEmit --project tsconfig.plugin.json
```

Deve completare senza errori.

### 4. Analyze Bundle

```bash
npx vite-bundle-visualizer dist/merlt-plugin.js
```

Mostra dimensione di ogni dipendenza nel bundle.

### 5. Test Import

```bash
# Crea test file
cat > test-import.mjs << 'EOF'
import plugin from './dist/merlt-plugin.js';
console.log('Plugin ID:', plugin.manifest.id);
console.log('Plugin Name:', plugin.manifest.name);
console.log('Slots:', plugin.manifest.contributedSlots);
EOF

# Run
node test-import.mjs
```

## Dependencies

### Bundled (incluse)

Tutte incluse nel bundle finale:

- `@tanstack/react-query` - State management
- `zustand` - Global store
- `recharts` - Charts
- `framer-motion` - Animazioni
- `lucide-react` - Icons
- `reagraph` - Graph visualization

### External (peer dependencies)

NON incluse, fornite da visualex-platform:

- `react` >=18.0.0
- `react-dom` >=18.0.0

### Platform Types

Il plugin importa tipi da `@visualex/platform/lib/plugins`.

Questi devono essere disponibili nel platform:
- `Plugin`
- `PluginManifest`
- `PluginContext`
- `PluginEventHandler`
- `SlotComponent`

Vedi `platform-types-example.d.ts` per esempio completo.

## Optimization

### Code Splitting (futuro)

Attualmente single bundle. Per abilitare code splitting:

```typescript
// vite.config.ts
rollupOptions: {
  output: {
    preserveModules: true,
    preserveModulesRoot: 'src',
  }
}
```

Questo crea:
```
dist/
├── plugin/index.js           # Entry
├── components/
│   ├── MerltSidebarPanel.js
│   └── MerltToolbar.js
└── services/
    ├── merltInit.js
    └── tracking.js
```

Ogni modulo e' lazy-loaded on demand.

### CSS Optimization

CSS e' attualmente inline nel JS bundle.

Per CSS separato:

```typescript
// vite.config.ts
build: {
  cssCodeSplit: true,
}
```

Questo crea:
```
dist/
├── merlt-plugin.js
└── merlt-plugin.css
```

Platform deve importare CSS manualmente:

```typescript
// Platform
import '@visualex/merlt-plugin/dist/merlt-plugin.css';
import plugin from '@visualex/merlt-plugin';
```

### Tree Shaking

Vite automaticamente tree-shake codice non usato.

Per verificare:

```bash
npx vite build --mode plugin --minify false
# Cerca nel bundle cosa e' incluso
```

## Troubleshooting

### "React is not defined"

React non e' esternalizzato correttamente.

Fix:
```typescript
external: ['react', 'react-dom', /^react\//, /^react-dom\//, 'react/jsx-runtime']
```

### "Cannot find module '@visualex/platform'"

Platform types non disponibili durante build.

Fix temporaneo: crea stub types:
```typescript
// types/platform.d.ts
declare module '@visualex/platform/lib/plugins' {
  export interface Plugin { ... }
  // ...
}
```

### Bundle troppo grande

1. Verifica dependencies in bundle:
   ```bash
   npx vite-bundle-visualizer
   ```

2. Externalizza dipendenze grosse se platform le ha:
   ```typescript
   external: ['react', 'react-dom', 'recharts', ...]
   ```

3. Lazy load componenti pesanti:
   ```typescript
   const HeavyComponent = lazy(() => import('./HeavyComponent'));
   ```

### Type errors durante build

1. Check tsconfig include paths
2. Verifica @visualex/platform types esistono
3. Usa `skipLibCheck: true` come workaround temporaneo

## CI/CD Integration

### GitHub Actions

```yaml
name: Build Plugin

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-node@v4
        with:
          node-version: 20

      - name: Install dependencies
        working-directory: frontend
        run: npm ci

      - name: Type check
        working-directory: frontend
        run: npm run typecheck

      - name: Build plugin
        working-directory: frontend
        run: npm run build:plugin

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: plugin-dist
          path: frontend/dist/

      - name: Check bundle size
        working-directory: frontend
        run: |
          size=$(stat -f%z dist/merlt-plugin.js)
          echo "Bundle size: $size bytes"
          if [ $size -gt 2097152 ]; then
            echo "Error: Bundle too large (>2MB)"
            exit 1
          fi
```

### npm publish

```bash
# From frontend/
npm run build:plugin
npm publish --access public
```

## Performance Metrics

Target metrics dopo build:

- **Bundle size (raw)**: <2MB
- **Bundle size (gzip)**: <500KB
- **Build time**: <30s
- **Type check time**: <10s
- **First load time**: <1s
- **TTI (Time to Interactive)**: <2s

Verifica con:

```bash
time npm run build:plugin
```
