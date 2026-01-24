# MERLT Plugin Build

Questo documento spiega come buildare il plugin MERLT standalone per visualex-platform.

## Architettura Plugin

Il plugin MERLT è caricato dinamicamente da visualex-platform tramite:
```typescript
const plugin = await import('@visualex/merlt-plugin');
```

### Struttura Output

```
dist/
├── merlt-plugin.js           # Bundle ESM principale
├── merlt-plugin.js.map       # Source map
└── types/
    └── plugin/
        └── index.d.ts        # Type definitions
```

## Build Plugin

```bash
npm run build:plugin
```

Questo comando esegue:
1. **build:plugin:types** - Genera type declarations con tsc
2. **build:plugin:bundle** - Crea bundle ESM con Vite

## Configurazione

### vite.config.ts

- **Mode**: `plugin` attiva library mode
- **Entry**: `src/plugin/index.ts`
- **Format**: ESM only
- **External**: react, react-dom (peer deps)
- **Output**: `dist/merlt-plugin.js`

### tsconfig.plugin.json

- **Extends**: tsconfig.json base
- **emitDeclarationOnly**: true
- **Include**: Solo file necessari al plugin
- **Exclude**: App.tsx, main.tsx (dev only)

### package.json

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

## Uso nel Platform

### Dynamic Import

```typescript
// visualex-platform/src/plugins/loader.ts
const loadMerltPlugin = async () => {
  try {
    const module = await import('@visualex/merlt-plugin');
    return module.default; // Plugin instance
  } catch (err) {
    console.error('Failed to load MERLT plugin', err);
    return null;
  }
};
```

### Installazione

Da visualex-platform:

```bash
npm install ../visualex-merlt/frontend
# o
npm install file:../visualex-merlt/frontend
```

Oppure dopo publish:
```bash
npm install @visualex/merlt-plugin
```

## CSS Handling

Il CSS è gestito in due modi:

1. **Durante build**: Vite inietta CSS nel bundle
2. **Runtime**: Il plugin può esportare style injection hook

### Approccio 1: CSS incluso nel bundle (attuale)
```typescript
// src/plugin/index.ts
import '../styles/merlt-plugin.css'; // Vite lo include automaticamente
```

### Approccio 2: CSS separato (opzionale)
```typescript
// vite.config.ts
build: {
  cssCodeSplit: false, // CSS in file separato
}
```

## Debugging

### Dev Mode
```bash
npm run dev  # Vite dev server normale
```

### Test Plugin Build
```bash
npm run build:plugin
node -e "import('./dist/merlt-plugin.js').then(m => console.log(m.default))"
```

### Type Check
```bash
npm run typecheck
```

## Dependencies

### Peer Dependencies (external)
- react >=18.0.0
- react-dom >=18.0.0

Questi sono forniti da visualex-platform e NON inclusi nel bundle.

### Bundled Dependencies
Tutte le altre dipendenze sono incluse nel bundle:
- @tanstack/react-query
- zustand
- recharts
- framer-motion
- lucide-react
- reagraph

## Code Splitting

Attualmente il plugin è un singolo bundle (`preserveModules: false`).

Per abilitare code splitting:
```typescript
// vite.config.ts
rollupOptions: {
  output: {
    preserveModules: true, // Mantieni struttura moduli
    preserveModulesRoot: 'src',
  }
}
```

Questo crea multiple chunks che vengono lazy-loaded runtime.

## Troubleshooting

### "Cannot find module '@visualex/platform'"
Il plugin importa tipi da `@visualex/platform/lib/plugins`. Assicurati che:
1. Il tipo sia disponibile nel platform
2. Il package.json del plugin abbia la dipendenza corretta

### "React is not defined"
React non è esternalizzato correttamente. Verifica:
```typescript
external: ['react', 'react-dom', /^react\//, /^react-dom\//]
```

### CSS non applicato
Il CSS potrebbe non essere iniettato. Verifica:
1. Import CSS nel plugin entry point
2. Vite config include il plugin React

## CI/CD

Per automatizzare il build:

```yaml
# .github/workflows/build-plugin.yml
- name: Build plugin
  run: |
    cd frontend
    npm ci
    npm run build:plugin
```
