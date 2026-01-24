# MERLT Plugin Architecture

Questo documento descrive l'architettura del sistema plugin che permette a MERLT di essere una feature **opzionale** in visualex-platform.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    visualex-platform (vanilla)                   │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Core Components                                          │   │
│  │  - ArticleView emette eventi via EventBus                │   │
│  │  - PluginSlot per extension points                       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Plugin System (src/lib/plugins/)                         │   │
│  │  - PluginRegistry: carica plugin per utenti abilitati    │   │
│  │  - EventBus: pub/sub per comunicazione                   │   │
│  │  - PluginSlot: render componenti plugin                  │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ (dynamic import)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    visualex-merlt (plugin)                       │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  MERLT Plugin Entry (frontend/src/plugin/index.ts)        │   │
│  │  - Si registra agli eventi                               │   │
│  │  - Fornisce componenti ai PluginSlot                     │   │
│  │  - MerltSidebarPanel, ValidationQueue, etc.              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Flow Abilitazione Utente

```
1. Admin abilita feature "merlt" per utente
   POST /api/admin/users/:userId/features/merlt
   { "enabled": true }

2. Utente fa opt-in (consenso)
   POST /api/users/me/features/merlt/consent
   { "consent": true }

3. Al login, frontend riceve lista features
   GET /api/users/me/features
   { "features": ["merlt"] }

4. PluginProvider carica MERLT plugin
   - Solo se user.features include "merlt"
   - Plugin si inizializza e registra eventi

5. PluginSlot renderizza componenti MERLT
   - Solo nelle pagine dove sono definiti slot
```

## Componenti del Plugin System

### 1. EventBus

Sistema pub/sub per la comunicazione tra visualex core e plugin.

**Eventi supportati:**

| Evento | Payload | Descrizione |
|--------|---------|-------------|
| `article:viewed` | `{ urn, articleId, userId }` | Utente apre un articolo |
| `article:scrolled` | `{ urn, visibleSections }` | Scroll nell'articolo |
| `article:highlighted` | `{ urn, text, startOffset, endOffset }` | Testo selezionato |
| `article:text-selected` | `{ urn, text, startOffset, endOffset }` | Testo selezionato (simile a highlighted) |
| `search:performed` | `{ query, filters, resultCount }` | Ricerca effettuata |
| `search:result-clicked` | `{ urn, position, query }` | Click su risultato |
| `bookmark:created` | `{ urn, userId }` | Bookmark creato |
| `bookmark:deleted` | `{ urn, userId }` | Bookmark eliminato |
| `user:logged-in` | `{ userId, features }` | Utente effettuato login |
| `user:logged-out` | `{ userId }` | Utente logout |
| `enrichment:requested` | `{ urn, userId }` | Richiesta enrichment articolo |
| `enrichment:started` | `{ urn, articleKey }` | Enrichment avviato |
| `enrichment:completed` | `{ urn, entitiesCount, relationsCount }` | Enrichment completato |
| `entity:validated` | `{ entityId, vote, userId }` | Entità validata da utente |
| `relation:validated` | `{ relationId, vote, userId }` | Relazione validata da utente |
| `citation:detected` | `{ urn, text, parsed }` | Citazione rilevata nel testo |
| `graph:node-clicked` | `{ nodeId, nodeType }` | Click su nodo nel grafo |
| `graph:edge-clicked` | `{ edgeId, edgeType }` | Click su arco nel grafo |
| `issue:viewed` | `{ issueId }` | Issue/problema visualizzato |
| `issue:voted` | `{ issueId, vote, userId }` | Vote su issue |
| `issue:reported` | `{ nodeId, issueType }` | Issue segnalato |
| `dossier:training-exported` | `{ dossierId, format }` | Dossier esportato per training |

**Uso in visualex-platform:**

```tsx
// ArticleView.tsx
import { EventBus } from '@/lib/plugins';

function ArticleView({ urn, articleId }) {
  useEffect(() => {
    EventBus.emit('article:viewed', { urn, articleId });
  }, [urn, articleId]);
}
```

### 2. PluginSlot

Punti di estensione dove i plugin possono iniettare UI.

**Slot disponibili:**

| Slot | Posizione | Props |
|------|-----------|-------|
| `article-sidebar` | Sidebar destra articolo | `{ urn, articleId }` |
| `article-toolbar` | Toolbar sopra articolo | `{ urn, articleId }` |
| `article-footer` | Footer articolo | `{ urn, articleId }` |
| `article-content-overlay` | Floating overlay su contenuto | `{ urn, articleId, contentRef }` |
| `search-filters` | Filtri ricerca aggiuntivi | `{ currentFilters }` |
| `user-menu` | Menu utente | `{ userId }` |
| `settings-panel` | Pannello impostazioni | `{ userId }` |
| `profile-tabs` | Tab aggiuntivi profilo utente | `{ userId }` |
| `admin-dashboard` | Pannelli dashboard admin | `{ userId }` |
| `bulletin-board` | Knowledge graph e issues | `{ userId }` |
| `graph-explorer` | Esploratore grafo knowledge | `{ urn?, depth? }` |
| `graph-view` | Visualizzazione grafo in workspace | `{ rootUrn, depth?, userId? }` |
| `dossier-actions` | Azioni dossier (export/import) | `{ dossierId, userId, dossier }` |
| `global-overlay` | Overlay full-screen | `{}` |

**Uso in visualex-platform:**

```tsx
// ArticlePage.tsx
import { PluginSlot } from '@/lib/plugins';

function ArticlePage({ urn, articleId }) {
  return (
    <div className="flex">
      <main>
        <ArticleContent />
      </main>
      <aside>
        {/* MERLT sidebar appare qui se plugin caricato */}
        <PluginSlot
          name="article-sidebar"
          props={{ urn, articleId }}
          fallback={null}
        />
      </aside>
    </div>
  );
}
```

### 3. PluginProvider

Gestisce il lifecycle dei plugin basandosi sullo stato utente.

```tsx
// App.tsx
import { PluginProvider } from '@/lib/plugins';

const plugins = [
  {
    id: 'merlt',
    enabled: true,
    loader: () => import('@visualex/merlt-plugin'),
  },
];

function App() {
  const { user, getAuthToken } = useAuth();

  return (
    <PluginProvider
      plugins={plugins}
      user={user ? { id: user.id, features: user.features } : null}
      apiBaseUrl={import.meta.env.VITE_API_URL}
      getAuthToken={getAuthToken}
    >
      <Routes />
    </PluginProvider>
  );
}
```

## Implementazione MERLT Plugin

### Entry Point

```typescript
// visualex-merlt/frontend/src/plugin/index.ts
import type { Plugin, PluginManifest } from '@visualex/platform/lib/plugins';

const manifest: PluginManifest = {
  id: 'merlt',
  name: 'MERLT Research',
  version: '1.0.0',
  description: 'Legal knowledge extraction and validation for research',

  // Feature flags richiesti per abilitare il plugin
  requiredFeatures: ['merlt'],

  // Eventi ascoltati dal plugin
  subscribedEvents: [
    'article:viewed',
    'article:highlighted',
    'article:text-selected',
    'citation:detected',
    'search:performed',
  ],

  // Slot UI dove il plugin inietta componenti
  contributedSlots: [
    'article-sidebar',
    'article-toolbar',
    'article-content-overlay',
    'bulletin-board',
    'dossier-actions',
    'graph-view',
    'profile-tabs',
    'admin-dashboard',
  ],
};

const merltPlugin: Plugin = {
  manifest,

  async initialize(context) {
    // Setup MERLT backend connection
    await initializeMerltServices({
      apiBaseUrl: context.apiBaseUrl,
      getAuthToken: context.getAuthToken,
      userId: context.user?.id,
    });

    // Return cleanup function (called on unmount)
    return () => shutdownMerltServices();
  },

  getSlotComponents() {
    return [
      { slot: 'article-sidebar', component: MerltSidebarPanel, priority: 100 },
      { slot: 'article-toolbar', component: MerltToolbar, priority: 50 },
      { slot: 'article-content-overlay', component: MerltContentOverlay, priority: 100 },
      { slot: 'bulletin-board', component: BulletinBoardSlot, priority: 100 },
      { slot: 'dossier-actions', component: DossierActionsSlot, priority: 100 },
      { slot: 'graph-view', component: GraphViewSlot, priority: 100 },
      { slot: 'profile-tabs', component: ProfilePage, priority: 100 },
      { slot: 'admin-dashboard', component: AcademicDashboard, priority: 100 },
    ];
  },

  getEventHandlers() {
    return {
      'article:viewed': (data) => {
        // Track per raccolta dati ricerca
        trackArticleView(data.urn, data.articleId, data.userId);
      },

      'article:highlighted': (data) => {
        // Track selezioni testo per proposte entità
        trackHighlight(data.urn, data.text, data.startOffset, data.endOffset);
      },

      'search:performed': (data) => {
        // Track pattern ricerca
        trackSearch(data.query, data.filters, data.resultCount);
      },
    };
  },
};

export default merltPlugin;
```

### Componenti MERLT

I componenti MERLT ricevono le props definite dallo slot:

```tsx
// MerltSidebarPanel.tsx
import type { SlotProps } from '@visualex/platform/lib/plugins';

type Props = SlotProps['article-sidebar'];

export function MerltSidebarPanel({ urn, articleId }: Props) {
  const { entities, isLoading } = useMerltArticleAnalysis(urn);

  return (
    <div className="merlt-sidebar">
      <EntityList entities={entities} />
      <ValidationQueue articleUrn={urn} />
      <ContributionPanel articleUrn={urn} />
    </div>
  );
}
```

## Feature Flags

### Struttura

```
merlt                    # Feature principale
├── merlt_contribution   # Sotto-feature: proporre entità
└── merlt_validation     # Sotto-feature: validare proposte
```

### API Admin

```bash
# Lista feature disponibili
GET /api/admin/features

# Vedi features utente
GET /api/admin/users/:userId/features

# Abilita/disabilita feature
PUT /api/admin/users/:userId/features/merlt
{ "enabled": true }
```

### API Utente

```bash
# Vedi proprie features (solo IDs)
GET /api/users/me/features

# Dai consenso per feature
POST /api/users/me/features/merlt/consent
{ "consent": true }
```

### Database Schema

```prisma
model UserFeature {
  id           String    @id @default(uuid())
  userId       String
  featureId    String    // "merlt", "merlt_contribution", etc.
  enabled      Boolean   @default(false)
  consentGiven Boolean   @default(false)
  consentDate  DateTime?
  enabledBy    String?   // Admin che ha abilitato
  enabledAt    DateTime?

  @@unique([userId, featureId])
}
```

## Data Flow

### Tracking (MERLT raccoglie dati)

```
1. Utente naviga in visualex-platform
2. Core emette eventi (article:viewed, search:performed, etc.)
3. MERLT plugin riceve eventi via subscription
4. Tracking service aggrega e invia a MERLT backend
5. Dati usati per ricerca (anonymizzati)
```

### Validation (Utente contribuisce)

```
1. MERLT backend propone entità/relazioni
2. MerltSidebarPanel mostra validazioni pendenti
3. Utente approva/rifiuta
4. Decisione inviata a MERLT backend
5. Authority utente aggiornata
```

## Vantaggi Architettura

1. **Separazione Clean**: visualex-platform non ha import diretti da MERLT
2. **Opt-in**: Solo utenti abilitati vedono features MERLT
3. **Admin Control**: Feature flags gestibili per utente
4. **Privacy**: Consenso esplicito richiesto
5. **Manutenzione Indipendente**: Codici separati, deploy separati
6. **Testing Isolato**: Plugin testabile standalone
7. **Scalabilità**: Altri plugin possono usare stesso sistema

## Migration Path

Per migrare da integrazione diretta a plugin:

1. **Rimuovere import MERLT** da componenti core
2. **Aggiungere PluginSlot** dove c'erano componenti MERLT
3. **Emettere eventi** dove MERLT leggeva stato direttamente
4. **Wrappare componenti MERLT** nell'interfaccia Plugin

Esempio migrazione:

```tsx
// PRIMA (accoppiato)
import { MerltInspectorPanel } from '@/features/merlt';

function ArticleView() {
  return (
    <>
      <ArticleContent />
      <MerltInspectorPanel urn={urn} />
    </>
  );
}

// DOPO (disaccoppiato)
import { PluginSlot, EventBus } from '@/lib/plugins';

function ArticleView() {
  useEffect(() => {
    EventBus.emit('article:viewed', { urn });
  }, [urn]);

  return (
    <>
      <ArticleContent />
      <PluginSlot name="article-sidebar" props={{ urn }} />
    </>
  );
}
```

## MERLT Plugin Implementation Details

### Componenti Implementati

Il plugin MERLT contribuisce i seguenti componenti:

#### 1. **MerltSidebarPanel** (`article-sidebar`)
Panel sulla destra dell'articolo che mostra:
- Entity extraction results
- Validation queue per entities proposte
- User contribution panel
- Analytics sidebar

#### 2. **MerltToolbar** (`article-toolbar`)
Toolbar sopra l'articolo con:
- Pulsante trigger enrichment
- Indicatori stato processing
- Quick actions

#### 3. **MerltContentOverlay** (`article-content-overlay`)
Overlay floating sul contenuto per:
- Citation highlighting e correction
- Entity tagging inline
- Annotation suggestions

#### 4. **BulletinBoardSlot** (`bulletin-board`)
Knowledge graph visualization e issue management

#### 5. **DossierActionsSlot** (`dossier-actions`)
Azioni per export dossier in formato training

#### 6. **GraphViewSlot** (`graph-view`)
Visualizzazione grafo nel workspace

#### 7. **ProfilePage** (`profile-tabs`)
Tab aggiuntivo profilo utente con:
- User authority scores
- Contribution history
- Validation records

#### 8. **AcademicDashboard** (`admin-dashboard`)
Dashboard admin con:
- Pipeline monitoring
- Entity/relation statistics
- User activity analytics

### Plugin Initialization

Quando l'utente ha la feature 'merlt' abilitata:

```
1. PluginProvider rileva feature 'merlt'
2. Carica dinamicamente MERLT plugin (import('@visualex/merlt-plugin'))
3. Chiama merltPlugin.initialize(context)
4. Inizializza servizi backend (API connection, auth tokens)
5. Registra event handlers via EventBus
6. Plugin pronto per renderizzare componenti negli slot
```

### Event Flow

```
visualex-platform core          MERLT Plugin
        |                             |
        | emette 'article:viewed'    |
        |----------------------------->
        |                          traccia lettura
        |                          propone entità
        |
        | emette 'article:highlighted'
        |----------------------------->
        |                          analizza selezione
        |                          potenziale entity
        |
        | emette 'citation:detected'
        |----------------------------->
        |                          processa citazione
        |                          aggiorna knowledge graph
```

### Feature Flags Integration

Il plugin legge i feature flags dal contesto:

```typescript
// In initialize(context)
const canValidate = context.user?.features.includes('merlt_validation');
const canContribute = context.user?.features.includes('merlt_contribution');

// Mostra solo UI abilitata
return {
  showValidationQueue: canValidate,
  showContributionPanel: canContribute,
};
```

## Quick Reference: Plugin Slots

| Slot | Component | Purpose |
|------|-----------|---------|
| `article-sidebar` | MerltSidebarPanel | Main analysis panel |
| `article-toolbar` | MerltToolbar | Quick actions toolbar |
| `article-content-overlay` | MerltContentOverlay | Inline annotations |
| `bulletin-board` | BulletinBoardSlot | Knowledge graph viewer |
| `dossier-actions` | DossierActionsSlot | Export actions |
| `graph-view` | GraphViewSlot | Graph visualization |
| `profile-tabs` | ProfilePage | User profile tab |
| `admin-dashboard` | AcademicDashboard | Admin monitoring |

## Quick Reference: Subscribed Events

| Event | Handler | Action |
|-------|---------|--------|
| `article:viewed` | trackArticleView | Track letture per ricerca |
| `article:highlighted` | trackHighlight | Track selezioni testo |
| `article:text-selected` | (passthrough) | Monitorare selezioni |
| `citation:detected` | (processare) | Arricchire knowledge graph |
| `search:performed` | trackSearch | Track query patterns |

## Deployment Checklist

- [ ] MERLT plugin buildata: `npm run build:plugin` in visualex-merlt/frontend
- [ ] Plugin distribuito in versioning (PyPI o npm registry)
- [ ] visualex-platform aggiornata con import dinamico
- [ ] Feature flag 'merlt' creato nel backend
- [ ] Almeno un utente test abilitato con 'merlt' feature
- [ ] EventBus emette eventi correttamente
- [ ] Plugin slot renderizzano componenti
- [ ] Event handlers ricevono dati

## Troubleshooting

### Plugin non caricato
- Verificare user.features include 'merlt'
- Controllare console per import errors
- Verificare percorso import del plugin

### Eventi non ricevuti
- Controllare manifest.subscribedEvents
- Verificare core emette evento con nome corretto
- Debuggare con EventBus.getHistory()

### Componenti non visualizzati
- Verificare slot name corrisponde PluginSlotName type
- Controllare priority (higher = rendered first)
- Verificare props structure corrisponde SlotProps type

### Auth/API errors
- Controllare getAuthToken() ritorna token valido
- Verificare apiBaseUrl raggiunge backend
- Controllare CORS headers
