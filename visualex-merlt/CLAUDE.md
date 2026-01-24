# CLAUDE.md - VisuaLex-MERL-T Integration

> **Istruzioni per agenti AI che lavorano su questo repository**

---

## Contesto Progetto

Questo repository implementa il **layer di integrazione** tra VisuaLex Platform e il framework MERL-T. È il "ponte" che porta le funzionalità di AI legal analysis nella piattaforma web.

**Parte di**: Monorepo ALIS_CORE
**Tipo**: Plugin + API Bridge + RLCF Web Interface
**Licenza**: Proprietaria

---

## Concetti Fondamentali

### MERL-T (Multi-Expert Legal Retrieval Transformer)
Sistema di 4 "esperti virtuali" che analizzano questioni giuridiche secondo i canoni ermeneutici dell'Art. 12 Preleggi:
- **LiteralExpert**: Interpretazione letterale
- **SystemicExpert**: Interpretazione sistematica
- **PrinciplesExpert**: Ratio legis, principi costituzionali
- **PrecedentExpert**: Giurisprudenza applicativa

### RLCF (Reinforcement Learning from Community Feedback)
Framework di apprendimento che migliora il sistema tramite feedback di esperti giuridici. 4 pilastri:
1. Dynamic Authority Scoring
2. Uncertainty Preservation
3. Constitutional Governance
4. Devil's Advocate System

### Plugin Architecture
Sistema a **8 slot** dove i componenti MERL-T si inseriscono nell'UI di VisuaLex.

---

## Stack Tecnologico

### Frontend Plugin (`frontend/`)
- React 19 + TypeScript 5.x
- Vite (build as library)
- Tailwind CSS v4

### RLCF Web (`rlcf-web/`)
- React 19 + TypeScript
- Vite
- Charts: Recharts

### Backend Bridge (`backend/`)
- Express + TypeScript
- Python subprocess bridge

---

## Comandi Utili

```bash
# Frontend Plugin
cd frontend && npm run dev         # Dev mode
cd frontend && npm run build:plugin # Build plugin bundle

# RLCF Web
cd rlcf-web && npm run dev         # Dev mode
cd rlcf-web && npm run build       # Build

# Full Stack
./start_dev.sh                     # Docker compose tutto
docker-compose up -d               # Services in background
```

---

## File Critici - Leggere Prima di Modificare

| File | Importanza | Note |
|------|------------|------|
| `frontend/src/plugin/index.ts` | **CRITICO** | Registra 8 slot e 25 eventi - NON modificare signature |
| `frontend/src/services/merltService.ts` | **ALTA** | 40+ metodi API - sincronizzato con merlt backend |
| `frontend/src/components/MerltSidebarPanel.tsx` | **ALTA** | UI principale risultati Expert |
| `rlcf-web/src/components/FeedbackForm.tsx` | **MEDIA** | Form raccolta feedback |

---

## Plugin System - Dettaglio Tecnico

### Registrazione Plugin

```typescript
// frontend/src/plugin/index.ts
export const merltPlugin: PluginManifest = {
  id: 'merlt',
  name: 'MERL-T Legal Analysis',
  version: '1.0.0',
  slots: {
    'article-toolbar': MerltToolbar,
    'article-sidebar': MerltSidebarPanel,
    'article-content-overlay': MerltContentOverlay,
    'profile-tabs': ProfileMerltTab,
    'admin-dashboard': AdminMerltPanel,
    'bulletin-board': BulletinBoardSlot,
    'dossier-actions': DossierActionsSlot,
    'graph-view': GraphViewSlot,
  },
  events: {
    listen: [
      'article:viewed',
      'article:text-selected',
      'citation:detected',
      'dossier:updated',
      'search:performed',
    ],
    emit: [
      'merlt:analysis-started',
      'merlt:analysis-complete',
      'merlt:expert-response',
      'merlt:citation-correction',
      'merlt:graph-node-selected',
    ],
  },
};
```

### Quando Aggiungi uno Slot
1. Crea il componente in `frontend/src/components/`
2. Registralo in `plugin/index.ts`
3. Aggiungi eventi necessari in `listen` e `emit`
4. Aggiorna questo CLAUDE.md e README.md
5. Comunica a visualex-platform di aggiungere il `<PluginSlot>`

### Quando Aggiungi un Evento
1. Definisci il tipo payload in `types/events.ts`
2. Aggiungi in `events.listen` o `events.emit` in `plugin/index.ts`
3. Documenta in README.md

---

## Services API

### merltService.ts - Metodi Principali

```typescript
// Analisi
analyzeArticle(urn: string): Promise<AnalysisResult>
analyzeQuery(query: string): Promise<ExpertResponses>

// Knowledge Graph
getGraphNeighbors(nodeId: string): Promise<GraphNode[]>
searchGraph(query: string): Promise<SearchResult>

// RLCF
submitFeedback(feedback: Feedback): Promise<void>
getUserAuthority(userId: string): Promise<AuthorityScore>

// Export
exportTrainingData(dossierId: string): Promise<TrainingDataset>
```

### Sincronizzazione con Backend
Questi metodi chiamano endpoint in `merlt` (framework Python). Se modifichi l'API:
1. Aggiorna `merltService.ts`
2. Aggiorna tipi in `types/api.ts`
3. Verifica che `merlt/api/` abbia l'endpoint corrispondente

---

## RLCF Web - Struttura

```
rlcf-web/src/
├── components/
│   ├── FeedbackForm.tsx       # Form raccolta feedback
│   ├── AuthorityDisplay.tsx   # Mostra punteggio autorità
│   ├── DissentPanel.tsx       # Gestione disaccordi
│   └── TrainingMonitor.tsx    # Monitoraggio addestramento
│
├── pages/
│   ├── FeedbackPage.tsx       # Pagina principale feedback
│   ├── DashboardPage.tsx      # Dashboard metriche
│   └── AdminPage.tsx          # Amministrazione RLCF
│
└── services/
    └── rlcfApi.ts             # API calls
```

---

## Convenzioni Codice

### TypeScript
- **Strict mode** attivo
- Tipi espliciti per API responses
- Evita `any`

### React
- Functional components only
- Hooks per logica condivisa
- `useMerlt*` prefix per hooks MERL-T specifici

### Styling
- Tailwind CSS v4
- Coerenza con visualex-platform

### Naming
- Componenti: PascalCase (`MerltSidebarPanel.tsx`)
- Services: camelCase (`merltService.ts`)
- Eventi: kebab-case con namespace (`merlt:analysis-complete`)

---

## Pattern da Seguire

### Slot Component
```tsx
// components/MerltToolbar.tsx
import { FC } from 'react';
import { eventBus } from '@visualex/plugin-api';
import { useMerltAnalysis } from '../hooks/useMerltAnalysis';

interface MerltToolbarProps {
  context: { article: Article };
}

export const MerltToolbar: FC<MerltToolbarProps> = ({ context }) => {
  const { analyze, isLoading } = useMerltAnalysis();

  const handleAnalyze = async () => {
    eventBus.emit('merlt:analysis-started', { urn: context.article.urn });
    const result = await analyze(context.article.urn);
    eventBus.emit('merlt:analysis-complete', { urn: context.article.urn, result });
  };

  return (
    <button onClick={handleAnalyze} disabled={isLoading}>
      {isLoading ? 'Analizzando...' : 'Analizza con MERL-T'}
    </button>
  );
};
```

### Event Handler
```tsx
// hooks/useMerltEvents.ts
import { useEffect } from 'react';
import { eventBus } from '@visualex/plugin-api';

export function useMerltEvents() {
  useEffect(() => {
    const handleArticleViewed = (data: { urn: string }) => {
      // Pre-fetch analysis
    };

    eventBus.on('article:viewed', handleArticleViewed);
    return () => eventBus.off('article:viewed', handleArticleViewed);
  }, []);
}
```

---

## Anti-Pattern - Cosa NON Fare

❌ **Non** modificare signature di plugin/index.ts senza coordinare con visualex-platform
   - I tipi devono matchare

❌ **Non** chiamare direttamente API merlt senza passare da merltService
   - Centralizza gestione errori e auth

❌ **Non** emettere eventi non documentati
   - Tutti gli eventi devono essere in plugin manifest

❌ **Non** modificare RLCF authority algorithm senza approvazione
   - È critico per la qualità del sistema

❌ **Non** hardcodare URL backend
   - Usa env vars

---

## Testing

### Frontend
- Vitest per unit test
- React Testing Library per component test
- Mock di eventBus per test isolati

### RLCF Web
- Vitest + React Testing Library

### Integration
- Test E2E con visualex-platform
- Verifica eventi cross-plugin

---

## Debug

### Plugin Loading
```javascript
// In browser console
window.__PLUGIN_REGISTRY__.getPlugins()
window.__EVENT_BUS__.on('*', console.log)
```

### RLCF
- Dashboard in `/rlcf-web/dashboard`
- Logs in Docker: `docker-compose logs merlt-api`

---

## Dipendenze Esterne

### visualex-platform (host)
- Fornisce PluginSlot e EventBus
- Questo plugin viene caricato dinamicamente

### merlt (Python framework)
- Fornisce API per Expert e RLCF
- Comunicazione via HTTP

### merlt-models (private)
- Pesi dei modelli addestrati
- Caricati da merlt framework

---

## Workflow di Sviluppo

1. **Branch** da main: `feature/nome-feature`
2. **Sviluppa** con `npm run dev` (frontend o rlcf-web)
3. **Testa** integrazione con visualex-platform
4. **Build** plugin con `npm run build:plugin`
5. **Test** plugin caricamento in platform
6. **PR** con descrizione dettagliata

---

## Agenti Consigliati per Task

| Task | Agente |
|------|--------|
| Nuovi slot/componenti | `frontend-builder` |
| Modifica eventi | `frontend-architect` prima |
| Bug fix UI | `debugger` |
| RLCF UI | `frontend-builder` + `ux-reviewer` |
| Integrazione API | `builder` |
| Documentazione | `scribe` |

---

## Riferimenti

- [README Principale](../README.md)
- [Architettura](../ARCHITETTURA.md)
- [Glossario](../GLOSSARIO.md)
- [Guida Navigazione](../GUIDA_NAVIGAZIONE.md)
- [Paper MERL-T](../papers/markdown/DA%20GP%20-%20MERLT.md)
- [Paper RLCF](../papers/markdown/DA%20GP%20-%20RLCF.md)

---

*Ultimo aggiornamento: Gennaio 2026*
