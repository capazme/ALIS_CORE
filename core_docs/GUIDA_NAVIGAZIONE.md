# Guida alla Navigazione del Codebase ALIS

> **Come orientarsi nel codice sorgente del progetto**

Questa guida è pensata per chi deve revisionare o comprendere il codice sorgente di ALIS senza necessariamente essere uno sviluppatore. Ogni sezione indica cosa cercare e dove trovarlo.

---

## Mappa Generale

```
ALIS_CORE/
│
├── papers/                    # Pubblicazioni accademiche
│   └── markdown/              # Versioni leggibili dei paper
│
├── visualex-api/              # Libreria Python (open source)
│   └── visualex/              # Codice sorgente
│
├── visualex-platform/         # Piattaforma web
│   ├── frontend/              # Interfaccia utente (React)
│   └── backend/               # Server (Express/Node.js)
│
├── visualex-merlt/            # Integrazione MERL-T
│   ├── frontend/              # Componenti plugin
│   ├── backend/               # Bridge API
│   └── rlcf-web/              # Interfaccia RLCF
│
├── merlt/                     # Framework ML (open source)
│   └── merlt/                 # Codice sorgente
│
├── merlt-models/              # Modelli addestrati
│
├── Legacy/                    # Codice storico (archivio)
│
├── README.md                  # Questo documento introduttivo
├── GLOSSARIO.md               # Terminologia tecnico-giuridica
├── ARCHITETTURA.md            # Come funziona il sistema
└── GUIDA_NAVIGAZIONE.md       # Questa guida
```

---

## 1. Per Capire la Metodologia

### Papers Accademici
**Dove**: `papers/markdown/`

| File | Contenuto |
|------|-----------|
| `DA GP - ALIS.md` | ALIS come piattaforma multi-stakeholder |
| `DA GP - MERLT.md` | Architettura a 5 livelli e 4 Expert |
| `DA GP - RLCF.md` | Framework di apprendimento dalla comunità |
| `DA GP - The Knowledge Commoditization Paradox.md` | Fondamenti teorici sull'entropia semantica |

### Documentazione Originale
**Dove**: `Legacy/MERL-T_alpha/docs/archive/`

Contiene la documentazione metodologica della versione alpha, con schemi e ragionamenti che hanno portato all'architettura attuale.

---

## 2. Per Esplorare la Piattaforma Web

### Frontend (Interfaccia Utente)
**Dove**: `visualex-platform/frontend/src/`

```
src/
├── main.tsx                           # Punto di ingresso dell'app
├── App.tsx                            # Componente principale
│
├── pages/                             # Pagine dell'applicazione
│   ├── SearchPage.tsx                 # Ricerca giuridica
│   ├── LoginPage.tsx                  # Autenticazione
│   ├── ProfilePageWrapper.tsx         # Profilo utente
│   ├── AdminPage.tsx                  # Amministrazione
│   └── SettingsPage.tsx               # Impostazioni
│
├── components/                        # Componenti riutilizzabili
│   ├── features/                      # Funzionalità specifiche
│   │   ├── search/                    # Ricerca e risultati
│   │   │   ├── ArticleTabContent.tsx  # Visualizzazione articoli
│   │   │   └── SelectionPopup.tsx     # Menu contestuale
│   │   ├── dossier/                   # Gestione dossier
│   │   ├── bulletin/                  # Bacheca
│   │   └── workspace/                 # Area di lavoro
│   └── ui/                            # Componenti grafici base
│
├── lib/                               # Librerie interne
│   └── plugins/                       # Sistema a plugin
│       ├── PluginRegistry.ts          # Registro dei plugin
│       ├── PluginSlot.tsx             # Punti di estensione UI
│       ├── EventBus.ts                # Comunicazione eventi
│       └── types.ts                   # Definizioni tipi
│
├── services/                          # Comunicazione con backend
│   └── api.ts                         # Client HTTP
│
└── store/                             # Gestione stato applicazione
    └── useAppStore.ts                 # Store Zustand
```

**File chiave da esaminare**:
- `ArticleTabContent.tsx` (~1200 righe): Cuore della visualizzazione articoli, contiene i PluginSlot per MERL-T
- `PluginRegistry.ts`: Come i plugin vengono caricati dinamicamente
- `EventBus.ts`: Come i componenti comunicano tra loro

### Backend (Server)
**Dove**: `visualex-platform/backend/src/`

```
src/
├── index.ts                           # Avvio server
├── routes/                            # Endpoint API
│   ├── auth.ts                        # Autenticazione
│   ├── articles.ts                    # Articoli normativi
│   └── dossier.ts                     # Gestione dossier
└── services/                          # Logica business
```

---

## 3. Per Capire l'Integrazione MERL-T

### Plugin Frontend
**Dove**: `visualex-merlt/frontend/src/`

```
src/
├── plugin/                            # Sistema plugin
│   └── index.ts                       # Registrazione 8 slot
│
├── components/                        # Componenti MERL-T
│   ├── MerltSidebarPanel.tsx          # Pannello laterale Expert
│   ├── MerltToolbar.tsx               # Barra strumenti
│   ├── MerltContentOverlay.tsx        # Overlay citazioni
│   ├── BulletinBoardSlot.tsx          # Knowledge Graph
│   ├── DossierActionsSlot.tsx         # Export training data
│   └── GraphViewSlot.tsx              # Visualizzazione grafo
│
├── services/                          # Comunicazione con MERL-T
│   └── merltService.ts                # 40+ metodi API
│
└── hooks/                             # React hooks
    └── useMerltAnalysis.ts            # Hook per analisi
```

**File chiave da esaminare**:
- `plugin/index.ts`: Definisce tutti gli 8 slot e i 25 eventi
- `MerltSidebarPanel.tsx`: Come vengono mostrati i risultati dei 4 Expert
- `merltService.ts`: Tutte le chiamate al framework ML

### RLCF Web Interface
**Dove**: `visualex-merlt/rlcf-web/src/`

Interfaccia dedicata per:
- Raccolta feedback dagli esperti
- Risoluzione disaccordi
- Monitoraggio addestramento
- Visualizzazione punteggi autorità

---

## 4. Per Capire lo Scraping delle Fonti

### Libreria visualex-api
**Dove**: `visualex-api/visualex/`

```
visualex/
├── __init__.py                        # Export principali
│
├── scrapers/                          # Scaricatori per fonte
│   ├── normattiva.py                  # Normattiva.it
│   ├── brocardi.py                    # Brocardi.it
│   └── eurlex.py                      # EUR-Lex
│
├── models/                            # Strutture dati
│   ├── article.py                     # Modello articolo
│   ├── reference.py                   # Riferimento normativo
│   └── tree.py                        # Struttura gerarchica
│
├── utils/                             # Utilities
│   ├── text_op.py                     # Operazioni su testo
│   ├── treextractor.py                # Estrazione struttura
│   ├── urngenerator.py                # Generazione URN NormeInRete
│   ├── cache_manager.py               # Gestione cache
│   └── http_client.py                 # Client HTTP robusto
│
└── api/                               # Server API
    └── app.py                         # Endpoint Quart
```

**File chiave da esaminare**:
- `scrapers/normattiva.py`: Come vengono scaricati i testi ufficiali
- `models/article.py`: Struttura dati dell'articolo normativo
- `utils/treextractor.py`: Come viene estratta la struttura gerarchica

---

## 5. Per Capire il Framework ML

### Framework merlt
**Dove**: `merlt/merlt/`

```
merlt/
├── experts/                           # I 4 Expert
│   ├── literal.py                     # LiteralExpert
│   ├── systemic.py                    # SystemicExpert
│   ├── principles.py                  # PrinciplesExpert
│   └── precedent.py                   # PrecedentExpert
│
├── rlcf/                              # Sistema RLCF
│   ├── authority.py                   # Calcolo autorità
│   ├── feedback.py                    # Gestione feedback
│   └── training.py                    # Addestramento
│
├── retrieval/                         # Ricerca ibrida
│   ├── vector.py                      # Ricerca vettoriale
│   ├── graph.py                       # Ricerca nel grafo
│   └── hybrid.py                      # Combinazione
│
├── synthesis/                         # Sintesi risposte
│   └── synthesizer.py                 # Combina Expert
│
└── knowledge_graph/                   # Knowledge Graph
    ├── builder.py                     # Costruzione grafo
    └── queries.py                     # Query Cypher
```

**File chiave da esaminare**:
- `experts/literal.py` (e gli altri Expert): Implementazione canoni ermeneutici
- `rlcf/authority.py`: Algoritmo di calcolo autorità
- `retrieval/hybrid.py`: Come viene combinata ricerca vettoriale e grafo

---

## 6. Configurazioni e Ambiente

### File di Configurazione

| File | Scopo | Dove |
|------|-------|------|
| `.env.example` | Variabili d'ambiente | Root di ogni repo |
| `docker-compose.yml` | Orchestrazione servizi | Root di ogni repo |
| `vite.config.ts` | Build frontend | `*/frontend/` |
| `tsconfig.json` | Configurazione TypeScript | `*/frontend/`, `*/backend/` |
| `pyproject.toml` | Configurazione Python | `visualex-api/`, `merlt/` |

### Database

| Database | Scopo | Porta | File schema |
|----------|-------|-------|-------------|
| PostgreSQL | Utenti, sessioni | 5432 | `backend/prisma/schema.prisma` |
| FalkorDB | Knowledge Graph | 6379 | `merlt/knowledge_graph/schema.cypher` |
| Qdrant | Vettori semantici | 6333 | Configurazione in `merlt/config/` |
| Redis | Cache | 6379 | Nessuno (key-value) |

---

## 7. Percorsi di Lettura Consigliati

### Per capire "come funziona una ricerca"
1. `visualex-platform/frontend/src/pages/SearchPage.tsx` - L'utente fa una ricerca
2. `visualex-platform/backend/src/routes/articles.ts` - Il server riceve la richiesta
3. `visualex-api/visualex/scrapers/normattiva.py` - Il testo viene scaricato
4. `visualex-merlt/frontend/src/components/MerltSidebarPanel.tsx` - Gli Expert analizzano

### Per capire "come funzionano gli Expert"
1. `papers/markdown/DA GP - MERLT.md` - Fondamento teorico
2. `merlt/merlt/experts/` - Implementazione
3. `merlt/merlt/synthesis/synthesizer.py` - Come vengono combinati

### Per capire "come funziona RLCF"
1. `papers/markdown/DA GP - RLCF.md` - I 4 pilastri
2. `merlt/merlt/rlcf/authority.py` - Calcolo autorità
3. `visualex-merlt/rlcf-web/src/` - Interfaccia raccolta feedback

### Per capire "come funziona il plugin system"
1. `visualex-platform/frontend/src/lib/plugins/types.ts` - Definizioni
2. `visualex-platform/frontend/src/lib/plugins/PluginRegistry.ts` - Registro
3. `visualex-merlt/frontend/src/plugin/index.ts` - Implementazione

---

## 8. Convenzioni del Codice

### Naming Conventions
- **Componenti React**: PascalCase (es. `ArticleTabContent.tsx`)
- **Funzioni/variabili**: camelCase (es. `fetchArticle`)
- **Classi Python**: PascalCase (es. `LiteralExpert`)
- **Funzioni Python**: snake_case (es. `get_article`)
- **File TypeScript**: camelCase o PascalCase per componenti
- **File Python**: snake_case

### Commenti
- I commenti in italiano sono usati dove il contesto giuridico è rilevante
- I commenti tecnici sono in inglese
- La documentazione principale è in italiano

### Tipi di File
| Estensione | Linguaggio | Scopo |
|------------|------------|-------|
| `.tsx` | TypeScript + JSX | Componenti React |
| `.ts` | TypeScript | Logica, tipi, servizi |
| `.py` | Python | Backend ML, scraping |
| `.cypher` | Cypher | Query Knowledge Graph |
| `.prisma` | Prisma | Schema database |

---

## 9. Come Trovare Specifiche Funzionalità

### Usando la ricerca nel codice

**Cercare un componente UI**:
```bash
# Nella cartella frontend
find . -name "*.tsx" | xargs grep "NomeComponente"
```

**Cercare un endpoint API**:
```bash
# Nella cartella backend
grep -r "router\." src/routes/
```

**Cercare una funzione Expert**:
```bash
# Nella cartella merlt
grep -r "def analyze" merlt/experts/
```

### Parole chiave utili

| Se cerchi... | Cerca nel codice... |
|--------------|---------------------|
| Ricerca articoli | `searchArticle`, `fetchArticle` |
| Analisi Expert | `LiteralExpert`, `analyzeQuery` |
| Feedback RLCF | `submitFeedback`, `authority` |
| Knowledge Graph | `FalkorDB`, `cypher`, `graph_query` |
| Plugin | `PluginSlot`, `registerPlugin` |
| Eventi | `EventBus`, `emit`, `on` |

---

## 10. Riferimenti Accademici nel Codice

Il codice implementa i concetti descritti nei paper. Ecco le corrispondenze:

| Concetto nel Paper | Implementazione |
|-------------------|-----------------|
| "Canoni ermeneutici Art. 12" (MERL-T) | `merlt/experts/*.py` |
| "Architettura a 5 livelli" (MERL-T) | Struttura delle cartelle `merlt/` |
| "Dynamic Authority Scoring" (RLCF) | `merlt/rlcf/authority.py` |
| "Uncertainty Preservation" (RLCF) | Confidence scores negli Expert |
| "Devil's Advocate System" (RLCF) | `merlt/rlcf/dissent.py` |
| "Entropia semantica" (Knowledge Commoditization) | Chunking in `visualex-api/utils/` |
| "Piattaforma multi-stakeholder" (ALIS) | Plugin system in `visualex-platform` |

---

*Ultimo aggiornamento: Gennaio 2026*
