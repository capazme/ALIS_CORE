# CLAUDE.md - VisuaLex Platform

> **Istruzioni per agenti AI che lavorano su questo repository**

---

## Contesto Progetto

Questo è il repository della **piattaforma web** del sistema ALIS (Artificial Legal Intelligence System). È il frontend/backend principale che gli utenti usano per cercare e analizzare testi normativi italiani.

**Parte di**: Monorepo ALIS_CORE
**Tipo**: Full-stack web application
**Licenza**: Proprietaria

---

## Stack Tecnologico

### Frontend
- **React 19** con TypeScript 5.x
- **Vite 7** per build/dev
- **Tailwind CSS v4** per styling
- **Zustand** per state management
- **React Router 7** per routing

### Backend
- **Express 5** su Node.js 20+
- **Prisma ORM** per database
- **PostgreSQL 15+** database
- **JWT** per autenticazione

---

## Comandi Utili

```bash
# Development
cd frontend && npm run dev          # Start frontend (port 5173)
cd backend && npm run dev           # Start backend (port 3001)
./start_dev.sh                      # Start tutto via Docker

# Build
cd frontend && npm run build        # Build frontend
cd backend && npm run build         # Build backend

# Testing
cd frontend && npm test             # Test frontend
cd backend && npm test              # Test backend

# Database
cd backend && npx prisma studio     # GUI database
cd backend && npx prisma migrate dev # Run migrations
```

---

## File Critici - Leggere Prima di Modificare

| File | Importanza | Note |
|------|------------|------|
| `frontend/src/components/features/search/ArticleTabContent.tsx` | **ALTA** | ~1200 LOC, cuore della UI, contiene PluginSlots |
| `frontend/src/lib/plugins/PluginRegistry.ts` | **ALTA** | Sistema plugin, non modificare signature |
| `frontend/src/lib/plugins/EventBus.ts` | **MEDIA** | Eventi cross-component |
| `frontend/src/store/useAppStore.ts` | **MEDIA** | State globale |
| `backend/prisma/schema.prisma` | **ALTA** | Schema DB, richiede migration |

---

## Architettura Plugin

La piattaforma usa un **sistema a slot** per permettere a plugin esterni (es. MERL-T) di estendere l'UI.

### Slots Definiti
```typescript
// In PluginRegistry.ts
type SlotName =
  | 'article-toolbar'        // Barra strumenti articolo
  | 'article-sidebar'        // Pannello laterale
  | 'article-content-overlay' // Overlay sul testo
  | 'profile-tabs'           // Tab nel profilo
  | 'admin-dashboard'        // Pannello admin
  | 'bulletin-board'         // Bacheca
  | 'dossier-actions'        // Azioni dossier
  | 'graph-view';            // Vista grafo
```

### Eventi Emessi
```typescript
// Eventi che i plugin possono ascoltare
'article:viewed'        // { urn, data }
'article:text-selected' // { text, position }
'citation:detected'     // { citations[] }
'dossier:updated'       // { dossierId }
'search:performed'      // { query, results }
```

### Quando Aggiungi uno Slot
1. Aggiungi il tipo in `frontend/src/lib/plugins/types.ts`
2. Registra in `PluginRegistry.ts`
3. Usa `<PluginSlot name="nuovo-slot" />` nel componente
4. Aggiorna questo CLAUDE.md e il README.md

---

## Convenzioni Codice

### TypeScript
- **Strict mode** sempre attivo
- Evita `any`, usa tipi espliciti
- Interfacce > Types per oggetti complessi

### React
- **Functional components** only
- Hooks personalizzati in `hooks/`
- Componenti feature in `components/features/`
- Componenti UI riutilizzabili in `components/ui/`

### Styling
- **Tailwind CSS v4** - non usare CSS custom
- Classi ordinate: layout → spacing → typography → colors → states
- Responsive: mobile-first (`sm:`, `md:`, `lg:`)

### Naming
- Componenti: PascalCase (`ArticleTabContent.tsx`)
- Hooks: `use` prefix (`useAppStore.ts`)
- Utilities: camelCase (`formatDate.ts`)
- Tipi/Interfacce: PascalCase con `I` o `T` prefix se necessario

---

## Pattern da Seguire

### Componenti Feature
```tsx
// Esempio: components/features/search/SearchResult.tsx
import { FC } from 'react';
import { PluginSlot } from '@/lib/plugins/PluginSlot';
import { eventBus } from '@/lib/plugins/EventBus';

interface SearchResultProps {
  article: Article;
}

export const SearchResult: FC<SearchResultProps> = ({ article }) => {
  const handleClick = () => {
    eventBus.emit('article:viewed', { urn: article.urn, data: article });
  };

  return (
    <div onClick={handleClick}>
      {/* Content */}
      <PluginSlot name="article-sidebar" context={{ article }} />
    </div>
  );
};
```

### Store Zustand
```typescript
// Esempio: store/useSearchStore.ts
import { create } from 'zustand';

interface SearchState {
  query: string;
  results: Article[];
  setQuery: (query: string) => void;
  setResults: (results: Article[]) => void;
}

export const useSearchStore = create<SearchState>((set) => ({
  query: '',
  results: [],
  setQuery: (query) => set({ query }),
  setResults: (results) => set({ results }),
}));
```

---

## Anti-Pattern - Cosa NON Fare

❌ **Non** importare direttamente da `@visualex/merlt-frontend`
   - Usa sempre PluginSlot per estensioni MERL-T

❌ **Non** modificare `ArticleTabContent.tsx` senza capire i PluginSlots
   - È il componente più critico, richiede attenzione

❌ **Non** aggiungere state globale senza discussione
   - Preferisci state locale o context

❌ **Non** usare CSS inline o moduli CSS
   - Solo Tailwind CSS

❌ **Non** creare nuove tabelle DB senza migration
   - Sempre `npx prisma migrate dev`

---

## Testing

### Frontend
- **Vitest** per unit test
- **React Testing Library** per component test
- File test: `*.test.tsx` accanto al componente

### Backend
- **Jest** per unit test
- File test in `__tests__/`

### E2E
- **Playwright** (se configurato)
- File in `e2e/`

---

## Debugging

### Frontend
- React DevTools
- Zustand DevTools middleware
- Console: `eventBus.on('*', console.log)` per debug eventi

### Backend
- Prisma Studio: `npx prisma studio`
- Logs: check `console.log` in routes

---

## Dipendenze Esterne

### visualex-api
- Repository separato, pubblica su PyPI
- Fornisce scraping Normattiva/Brocardi/EUR-Lex
- Comunicazione via HTTP (port 5000)

### visualex-merlt (opzionale)
- Plugin per analisi MERL-T
- Caricato dinamicamente via PluginRegistry
- Non è una dipendenza diretta - usa eventi e slot

---

## Workflow di Sviluppo

1. **Branch** da main: `feature/nome-feature`
2. **Sviluppa** con `npm run dev`
3. **Testa** con `npm test`
4. **Build** verificare con `npm run build`
5. **PR** con descrizione dettagliata

---

## Agenti Consigliati per Task

| Task | Agente |
|------|--------|
| Nuovi componenti UI | `frontend-builder` |
| Refactoring componenti | `frontend-architect` + `frontend-builder` |
| Bug fix frontend | `debugger` |
| Nuovi endpoint API | `builder` |
| Review UX | `ux-reviewer` |
| Documentazione | `scribe` |

---

## Riferimenti

- [README Principale](../README.md)
- [Architettura](../ARCHITETTURA.md)
- [Glossario](../GLOSSARIO.md)
- [Guida Navigazione](../GUIDA_NAVIGAZIONE.md)
- [Paper MERL-T](../papers/markdown/DA%20GP%20-%20MERLT.md)

---

*Ultimo aggiornamento: Gennaio 2026*
