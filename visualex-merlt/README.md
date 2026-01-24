# VisuaLex-MERL-T Integration

> **Layer di integrazione tra VisuaLex Platform e il framework MERL-T**

---

## Panoramica

Questo repository contiene i componenti che collegano la piattaforma VisuaLex con il framework di machine learning MERL-T. Implementa un sistema a **plugin** che permette di estendere la piattaforma base con funzionalità di analisi giuridica avanzata.

### Funzionalità Principali
- **Sistema Multi-Expert**: 4 esperti che analizzano secondo i canoni ermeneutici
- **Knowledge Graph**: Visualizzazione delle relazioni tra norme
- **RLCF Feedback**: Interfaccia per raccolta feedback esperti
- **Plugin Architecture**: 8 slot di estensione, 25+ eventi

---

## Architettura

```
+-------------------+     +-------------------+     +-------------------+
|                   |     |                   |     |                   |
|  VisuaLex         |     |  Integration      |     |     MERL-T        |
|  Platform         |<--->|     Layer         |<--->|   Framework       |
|                   |     |   (this repo)     |     |                   |
| - Frontend        |     | - React Components|     | - Multi-Expert    |
| - Backend         |     | - API Bridge      |     | - RLCF            |
| - Auth            |     | - Services        |     | - Knowledge Graph |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     +-------------------+
        |                         |                         |
        v                         v                         v
+-------------------+     +-------------------+     +-------------------+
|   PostgreSQL      |     |     Redis         |     |  FalkorDB+Qdrant  |
|   (Users/Auth)    |     |    (Cache)        |     | (Knowledge/Vector)|
+-------------------+     +-------------------+     +-------------------+
```

---

## Componenti

### 1. Frontend Integration (`frontend/`)

Plugin React che estende VisuaLex con funzionalità MERL-T.

**Struttura**:
```
frontend/src/
├── plugin/
│   └── index.ts              # Registrazione plugin (8 slot)
│
├── components/
│   ├── MerltSidebarPanel.tsx # Pannello risultati Expert
│   ├── MerltToolbar.tsx      # Barra strumenti analisi
│   ├── MerltContentOverlay.tsx # Overlay citazioni
│   ├── BulletinBoardSlot.tsx # Knowledge Graph explorer
│   ├── DossierActionsSlot.tsx # Export training data
│   └── GraphViewSlot.tsx     # Visualizzazione grafo
│
├── services/
│   └── merltService.ts       # 40+ metodi API MERL-T
│
└── hooks/
    └── useMerltAnalysis.ts   # Hook analisi
```

### 2. Backend Integration (`backend/`)

Bridge API tra VisuaLex e MERL-T.

- Routes per comunicazione VisuaLex → MERL-T
- Python bridge per chiamate dirette
- Pass-through autenticazione

### 3. RLCF Web (`rlcf-web/`)

Interfaccia standalone per il sistema RLCF:
- Raccolta feedback da esperti giuridici
- Risoluzione disaccordi tra Expert
- Monitoraggio addestramento
- Visualizzazione punteggi autorità

### 4. Apps (`apps/`)

Utility applications:
- `expert_debugger.py` - Debug risposte Expert
- `rlcf_dashboard.py` - Dashboard metriche RLCF

---

## Plugin System

### 8 Slot Registrati

| Slot | Componente | Scopo |
|------|------------|-------|
| `article-toolbar` | `MerltToolbar` | Pulsante "Analizza con MERL-T" |
| `article-sidebar` | `MerltSidebarPanel` | Risultati 4 Expert |
| `article-content-overlay` | `MerltContentOverlay` | Correzioni citazioni |
| `profile-tabs` | Tab profilo | Impostazioni MERL-T utente |
| `admin-dashboard` | Dashboard admin | Gestione pipeline |
| `bulletin-board` | `BulletinBoardSlot` | Knowledge Graph explorer |
| `dossier-actions` | `DossierActionsSlot` | Export training data |
| `graph-view` | `GraphViewSlot` | Visualizzazione grafo completa |

### 25+ Eventi Gestiti

**Ascolto** (dal platform):
- `article:viewed` - Articolo caricato
- `article:text-selected` - Testo selezionato
- `citation:detected` - Citazioni trovate
- `dossier:updated` - Dossier modificato
- `search:performed` - Ricerca effettuata

**Emissione** (verso platform):
- `merlt:analysis-started` - Analisi iniziata
- `merlt:analysis-complete` - Analisi completata
- `merlt:expert-response` - Risposta singolo Expert
- `merlt:citation-correction` - Correzione citazione
- `merlt:graph-node-selected` - Nodo grafo selezionato

---

## I 4 Expert

Implementazione computazionale dei canoni ermeneutici dell'Art. 12 Preleggi.

| Expert | Canone | Fonti | Output |
|--------|--------|-------|--------|
| **LiteralExpert** | Interpretazione letterale | Testo normativo | Significato testuale, definizioni |
| **SystemicExpert** | Interpretazione sistematica | Knowledge Graph | Contesto normativo, collegamenti |
| **PrinciplesExpert** | Ratio legis | Costituzione, lavori preparatori | Principi, intenzione legislatore |
| **PrecedentExpert** | Giurisprudenza | Massime, sentenze | Precedenti, orientamenti |

### Flusso Analisi

```
Query utente
    │
    ├──► LiteralExpert ──────┐
    ├──► SystemicExpert ─────┼──► Synthesizer ──► Risposta
    ├──► PrinciplesExpert ───┤
    └──► PrecedentExpert ────┘
```

---

## RLCF (Reinforcement Learning from Community Feedback)

### I 4 Pilastri

1. **Dynamic Authority Scoring**: Peso feedback basato su competenza valutatore
2. **Uncertainty Preservation**: Mantiene incertezza dove appropriato
3. **Constitutional Governance**: Principi che guidano il sistema
4. **Devil's Advocate System**: Sfida deliberata per evitare conformismo

### Tipi di Feedback

| Tipo | Livello | Valuta |
|------|---------|--------|
| Retrieval | Fonti | Qualità fonti recuperate |
| Reasoning | Expert | Correttezza ragionamento |
| Synthesis | Output | Qualità risposta finale |

### Autorità

```
authority = f(background, consistency, consensus, expertise_domain)
```

- **Background**: Titoli, esperienza dichiarata
- **Consistency**: Coerenza feedback nel tempo
- **Consensus**: Allineamento con altri esperti
- **Expertise Domain**: Peso per dominio specifico (civile, penale, admin)

---

## Development

### Prerequisites
- Docker & Docker Compose
- Node.js 20+
- Python 3.10+
- Accesso a `visualex-platform` e `merlt` repositories

### Quick Start

```bash
# Clone repository
git clone <repo-url>
cd visualex-merlt

# Copy environment variables
cp .env.example .env
# Edit .env with credentials

# Start all services
docker-compose up -d

# Development mode
cd frontend && npm install && npm run dev
cd rlcf-web && npm install && npm run dev
```

### Build Plugin

```bash
cd frontend
npm run build:plugin    # Produce dist/merlt-plugin.js
```

### Local Start Script

```bash
./start_dev.sh
```

### Service URLs

| Service | URL |
|---------|-----|
| VisuaLex Platform | http://localhost:3000 |
| MERL-T API | http://localhost:8000 |
| RLCF Web | http://localhost:3001 |
| Neo4j Browser | http://localhost:7474 |
| Qdrant Dashboard | http://localhost:6333/dashboard |

---

## Feature Flags

```bash
ENABLE_RLCF=true              # Raccolta feedback
ENABLE_EXPERT_FEEDBACK=true   # UI disaccordi
ENABLE_KNOWLEDGE_GRAPH=true   # Visualizzazione grafo
ENABLE_CITATION_CORRECTION=true # Correzioni citazioni
```

---

## File Chiave

| File | Scopo | Complessità |
|------|-------|-------------|
| `frontend/src/plugin/index.ts` | Registrazione 8 slot, 25 eventi | **ALTA** |
| `frontend/src/components/MerltSidebarPanel.tsx` | UI risultati Expert | Alta |
| `frontend/src/services/merltService.ts` | 40+ metodi API | Alta |
| `rlcf-web/src/components/FeedbackForm.tsx` | Form raccolta feedback | Media |

---

## Dependencies

| Package | Repository | Purpose |
|---------|------------|---------|
| `visualex` | visualex-api | Scraping (PyPI) |
| `merlt` | merlt | ML Pipeline (PyPI) |
| `merlt-models` | merlt-models | Trained Models (Private) |
| `visualex-platform` | visualex-platform | Platform (Private) |

---

## Riferimenti Accademici

Questo repository implementa i concetti descritti in:

- **Allega, D., & Puzio, G. (2025b)**. *MERL-T: A multi-expert architecture for trustworthy artificial legal intelligence*. CIDE 2025.
- **Allega, D., & Puzio, G. (2025c)**. *Reinforcement learning from community feedback (RLCF)*. CIDE 2025.

---

## License

Proprietary - All rights reserved (c) 2026

---

## Collegamenti

- [README Principale ALIS](../README.md)
- [Architettura Sistema](../ARCHITETTURA.md)
- [Glossario](../GLOSSARIO.md)
- [Guida Navigazione](../GUIDA_NAVIGAZIONE.md)
- [Plugin Architecture Docs](docs/PLUGIN_ARCHITECTURE.md)

---

*Ultimo aggiornamento: Gennaio 2026*
