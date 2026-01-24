# CLAUDE.md

> **Versione**: 5.0 | **Ultimo aggiornamento**: 29 Dicembre 2025 - **Multi-Agent Workflow Configured**

---

## MISSIONE PRINCIPALE

**Stai costruendo `merlt`**: la libreria Python di riferimento per l'informatica giuridica italiana.

Ogni riga di codice che scrivi sarà usata da giuristi-programmatori per costruire il codice civile digitale del futuro. Scrivi come se stessi creando `pandas` o `requests` - API chiare, documentazione eccellente, zero duplicazioni.

---

## Prima di Ogni Sessione

**Leggi in ordine:**
1. `docs/claude-context/LIBRARY_VISION.md` - **Principi guida della libreria**
2. `docs/claude-context/CURRENT_STATE.md` - Stato attuale
3. `docs/claude-context/PROGRESS_LOG.md` - Contesto recente

---

## API Target della Libreria

```python
# Questo è ciò che l'utente finale deve poter fare:
from merlt import LegalKnowledgeGraph

kg = LegalKnowledgeGraph()
await kg.connect()

# Una riga per ingestion
article = await kg.ingest("codice penale", "52")

# Una riga per ricerca
results = await kg.search("legittima difesa")

# Tutto il resto (grafo, vettori, bridge, multivigenza) è automatico
```

Se un'operazione comune richiede più di 3 righe, **ripensa l'API**.

---

## Principi di Sviluppo

### 1. ZERO DUPLICAZIONI

```python
# PRIMA di scrivere qualsiasi funzione:
# 1. Cerca se esiste già in merlt/
# 2. Se esiste, riutilizzala
# 3. Se non esiste, creala nel posto giusto (non negli scripts)

# MAI così:
def my_custom_scraper():  # ❌ Duplica NormattivaScraper
    ...

# SEMPRE così:
from merlt.sources import NormattivaScraper  # ✅ Riusa
```

### 2. COMPOSABILITÀ

```python
# Ogni componente DEVE funzionare da solo:
from merlt.sources import NormattivaScraper
scraper = NormattivaScraper()
text = await scraper.fetch("codice civile", "1453")  # ✅ Funziona isolato

# Ma anche insieme:
from merlt import LegalKnowledgeGraph  # ✅ Orchestrazione automatica
```

### 3. MAI LOGICA NEGLI SCRIPTS

```python
# scripts/ sono SOLO entry points:

# scripts/ingest_cp.py - CORRETTO
from merlt import LegalKnowledgeGraph

async def main():
    kg = LegalKnowledgeGraph()
    await kg.ingest_batch("codice penale", libro="I")

# scripts/ingest_cp.py - SBAGLIATO
async def main():
    # 200 righe di logica custom ❌
    for article in articles:
        text = await scraper.fetch(...)
        parsed = parse_article(text)
        # ... altro codice che dovrebbe essere in merlt/
```

### 4. DOCUMENTAZIONE ITALIANA

```python
async def cerca(query: str, top_k: int = 5) -> List[Risultato]:
    """
    Cerca nel knowledge graph giuridico.

    Args:
        query: Domanda in linguaggio naturale
               (es. "Cos'è la legittima difesa?")
        top_k: Numero massimo di risultati

    Returns:
        Lista di Risultato con articoli e contesto

    Example:
        >>> risultati = await kg.cerca("responsabilità del debitore")
        >>> print(risultati[0].articolo)
        "Art. 1218 c.c."
    """
```

---

## Struttura Package

```
merlt/                           # Package principale
├── __init__.py                  # API pubblica: LegalKnowledgeGraph, MerltConfig
├── config/                      # ⚙️ Configurazione
│   └── environments.py          # TEST_ENV, PROD_ENV
│
├── core/                        # 🎯 Orchestrazione (entry point)
│   └── legal_knowledge_graph.py # LegalKnowledgeGraph, MerltConfig
│
├── sources/                     # 📥 Fonti dati
│   ├── base.py                  # BaseScraper (interfaccia)
│   ├── normattiva.py            # NormattivaScraper
│   ├── brocardi.py              # BrocardiScraper
│   └── utils/                   # Utilities (norma, urn, tree, text, http)
│
├── storage/                     # 🗄️ Persistence
│   ├── graph/                   # FalkorDB client
│   ├── vectors/                 # EmbeddingService
│   ├── bridge/                  # Bridge Table (chunk ↔ nodo)
│   └── retriever/               # GraphAwareRetriever
│
├── pipeline/                    # ⚙️ Processing
│   ├── ingestion.py             # IngestionPipelineV2
│   ├── parsing.py               # CommaParser
│   ├── chunking.py              # StructuralChunker
│   └── multivigenza.py          # MultivigenzaPipeline
│
├── rlcf/                        # 🧠 RLCF Framework
│   ├── authority.py             # AuthorityModule
│   └── aggregation.py           # AggregationEngine
│
├── models/                      # 📦 Data models
└── utils/                       # 🔧 Utilities
```

---

## Pattern di Codice

### Async First

```python
# Tutte le operazioni I/O sono async
async def ingest(self, tipo_atto: str, articolo: str) -> IngestionResult:
    ...
```

### Type Hints Sempre

```python
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class IngestionResult:
    article_urn: str
    nodes_created: List[str]
    errors: List[str]
```

### Error Handling Graceful

```python
# Mai fallire completamente, degradare gracefully
result = await kg.ingest("codice penale", "52")
if result.errors:
    logger.warning(f"Completato con warning: {result.errors}")
# L'operazione continua per il resto
```

---

## Contesto Utente

| Aspetto | Valore |
|---------|--------|
| **Chi** | Studente di giurisprudenza (non programmatore) |
| **Cosa** | Tesi su "sociologia computazionale del diritto" |
| **Obiettivo** | Creare libreria di riferimento per informatica giuridica IT |
| **Lingua** | Italiano per documentazione, inglese per codice |

---

## Checklist Pre-Commit

- [ ] Nessuna duplicazione di codice
- [ ] Logica nel package, non negli scripts
- [ ] Type hints completi
- [ ] Docstring in italiano
- [ ] Test per funzioni pubbliche
- [ ] CURRENT_STATE.md aggiornato

---

## WORKFLOW MULTI-AGENTE PER MERL-T

### Agenti Consigliati per Task Type

Questo progetto e' una libreria **Legal Tech** complessa con Knowledge Graph, RLCF, Multi-Expert System, e pipeline ETL.
Usa gli agenti specializzati per massimizzare efficienza e qualita'.

| Task Type | Agente | Quando Usarlo | Esempio |
|-----------|--------|---------------|---------|
| **Nuove Feature Libreria** | `builder` | Implementare nuove API pubbliche, estendere componenti core | "Aggiungi metodo `kg.export_training_data()`" |
| **Bug Fix Complessi** | `debugger` | Problemi async, memory leak, race condition, integration bugs | "Debug: FalkorDB connection pool esaurito dopo 100 query" |
| **Design API/Architettura** | `api-designer` | Prima di implementare nuovi moduli, design interfacce pubbliche | "Progetta API per TraversalPolicy integration" |
| **Schema Database** | `database-architect` | Design schema FalkorDB (nodi/relazioni), PostgreSQL, Qdrant collections | "Ottimizza schema grafo per relazioni multivigenza" |
| **Scraping Robusto** | `scraper-builder` | Migliorare NormattivaScraper/BrocardiScraper con retry, checkpoint | "Aggiungi rate limiting e proxy rotation a NormattivaScraper" |
| **Pipeline ETL** | `data-pipeline` | Batch ingestion, enrichment pipeline, multivigenza processing | "Ottimizza batch_ingest con parallel processing e checkpoint" |
| **LLM Integration** | `llm-orchestrator` | Multi-provider LLM (OpenRouter), prompt optimization, cost tracking | "Implementa fallback provider per LiteralExpert" |
| **Knowledge Graph** | `graph-engineer` | Cypher query optimization, embeddings, RAG, GraphAwareRetriever | "Ottimizza shortest_path con cache per retrieval" |
| **Policy Gradient/RLCF** | `prompt-engineer` | Ottimizzazione prompt expert, few-shot learning, chain-of-thought | "Migliora prompt PrinciplesExpert per reasoning quality" |
| **Testing** | `validator` | Test suite completa, integration test con database reali | "Scrivi test per PolicyGradientTrainer con mock feedback" |
| **Documentazione** | `scribe` | README, docstring, API docs, architecture docs | "Documenta RLCF loop end-to-end con diagrammi" |
| **Code Review** | Usa skill `code-review` | Review PR, quality check, security audit | "Review implement GatingPolicy integration" |

### Workflow Suggeriti per Task Comuni

#### 1. Nuova Feature Completa

```bash
# Step 1: Design architettura
> Usa api-designer per progettare API per export_training_data()

# Step 2: Implementazione
> Usa builder per implementare merlt/core/export.py seguendo il design

# Step 3: Testing
> Usa validator per scrivere test completi in tests/core/test_export.py

# Step 4: Documentazione
> Usa scribe per documentare la nuova API in docs/ e aggiornare README
```

#### 2. Bug Fix Critico

```bash
# Step 1: Analisi e debug
> Usa debugger per analizzare il problema di memory leak in EmbeddingService

# Step 2: Implementa fix
> Usa builder per applicare la correzione con proper cleanup

# Step 3: Regression test
> Usa validator per aggiungere test che prevengano la regression
```

#### 3. Ottimizzazione Pipeline

```bash
# Step 1: Profiling
> Usa data-pipeline per analizzare bottleneck in batch_ingestion

# Step 2: Ottimizzazione
> Usa builder per implementare parallel processing con asyncio.TaskGroup

# Step 3: Benchmark
> Usa validator per eseguire benchmark e validare miglioramenti
```

#### 4. Schema Evolution

```bash
# Step 1: Design schema
> Usa database-architect per progettare nuove relazioni :DERIVED_FROM per amendments

# Step 2: Migration
> Usa builder per implementare migration script in merlt/storage/graph/

# Step 3: Test con dati reali
> Usa validator per testare migration su database di sviluppo
```

#### 5. Expert System Enhancement

```bash
# Step 1: Prompt engineering
> Usa prompt-engineer per ottimizzare SystemicExpert reasoning quality

# Step 2: Policy tuning
> Usa llm-orchestrator per bilanciare cost/quality tra provider

# Step 3: RLCF validation
> Usa validator per simulare feedback loop con EXP-022
```

### Convenzioni Specifiche MERL-T

#### Database Naming
```python
# SEMPRE usa _dev e _prod suffixes
FalkorDB: "merl_t_dev" / "merl_t_prod"
Qdrant: "merl_t_dev_chunks" / "merl_t_prod_chunks"
PostgreSQL: "rlcf_dev" / "rlcf_prod"
Redis: key prefix "dev:" / "prod:"
```

#### Logging
```python
# SEMPRE salva in logs/ non in /tmp
import structlog
logger = structlog.get_logger()
# Logs salvati automaticamente in logs/merlt.log
```

#### Testing
```python
# SEMPRE testa con database reali, NO mock
# Test suite: 311+ test passing
pytest tests/rlcf/ tests/experts/ tests/storage/ -v

# Test specifici
pytest tests/test_gating_policy_integration.py -v
pytest tests/test_policy_gradient.py -v
```

#### Parsing Robustness
```python
# Aggiungi SEMPRE nuovi edge case a CommaParser quando li trovi
# File: merlt/pipeline/parsing.py
# Test: tests/pipeline/test_parsing.py
```

#### Config Externalization
```yaml
# SEMPRE esternalizza prompt e parametri in YAML
# merlt/experts/config/experts.yaml
# merlt/rlcf/model_config.yaml
# docs/experiments/EXP-XXX/config.yaml
```

---

## Comandi Utili

```bash
# === Ambiente ===
cd /Users/gpuzio/Desktop/CODE/MERL-T_alpha
source .venv/bin/activate

# === Database (Development) ===
docker-compose -f docker-compose.dev.yml up -d
# Services: FalkorDB (6380), Qdrant (6333), PostgreSQL (5433), Redis (6379)

docker-compose -f docker-compose.dev.yml down  # Stop
docker-compose -f docker-compose.dev.yml logs -f  # Logs

# === Test ===
pytest tests/ -v  # All tests (311+ passing)
pytest tests/rlcf/ -v  # RLCF tests only
pytest tests/experts/ -v  # Expert system tests
pytest tests/storage/ -v  # Storage tests

pytest tests/test_gating_policy_integration.py -v  # GatingPolicy integration
pytest tests/test_policy_gradient.py -v  # Policy gradient training

# Test con coverage
pytest tests/ --cov=merlt --cov-report=html

# === Ingestion Scripts ===
# Ingest Libro IV Codice Civile
python scripts/batch_ingest_libro_iv.py

# Ingest Codice Penale Libro I
python scripts/ingest_libro_primo_cp.py

# Enrichment Brocardi batch
python scripts/batch_enrich_brocardi.py

# === Esperimenti ===
# RLCF Simulation
python scripts/run_rlcf_simulation.py

# Policy Gradient Simulation (EXP-022)
python scripts/exp022_policy_simulation.py
python scripts/exp022_policy_simulation.py --config docs/experiments/EXP-022_policy_gradient_simulation/config.yaml

# Expert Comparison (EXP-018)
python scripts/exp018_expert_comparison.py

# E2E Pipeline (EXP-019)
python scripts/exp019_e2e_pipeline.py

# === Streamlit Expert Debugger ===
streamlit run apps/expert_debugger.py
# Access: http://localhost:8501

# === Database Management ===
# Reset storage (ATTENZIONE: cancella tutti i dati!)
python scripts/reset_storage.py

# Migrate database schema
python scripts/migrate_database.py

# Initialize environments
python scripts/init_environments.py

# === Importa libreria ===
python -c "from merlt import LegalKnowledgeGraph; print('OK')"

# === Logs ===
tail -f logs/merlt.log  # Application logs
tail -f logs/rlcf.log  # RLCF logs
tail -f logs/expert_debugger.log  # Streamlit logs

# === Utilities ===
# Kill ports if stuck
./scripts/kill_ports.sh

# Start dev environment (all services + watch)
./scripts/start_dev.sh

# Overnight ingestion (batch processing)
./scripts/run_ingestion_overnight.sh
```

---

## Stack Tecnologico

### Backend & Core
- **Python**: 3.11+ (async-first, type hints completi)
- **Framework**: Nessun web framework (e' una libreria, non un server)
- **Async**: asyncio, aiohttp, asyncpg

### Storage Layer
- **Graph DB**: FalkorDB (Redis-compatible, Cypher query language)
- **Vector DB**: Qdrant (embeddings con sentence-transformers E5-large)
- **RDBMS**: PostgreSQL (Bridge Table, RLCF metadata)
- **Cache**: Redis (caching layer)

### AI/ML
- **LLM Provider**: OpenRouter (multi-provider: Claude, GPT-4, Gemini)
- **Embeddings**: sentence-transformers (multilingual-e5-large-instruct)
- **Policy Learning**: PyTorch (GatingPolicy, TraversalPolicy networks)
- **RLCF**: Custom implementation (authority scoring, feedback aggregation)

### Data Processing
- **Scraping**: aiohttp, BeautifulSoup4 (Normattiva, Brocardi)
- **Parsing**: Custom CommaParser per testi giuridici italiani
- **Chunking**: Semantic + Structural (multi-strategy)

### DevOps
- **Containerization**: Docker Compose (multi-service stack)
- **Logging**: structlog (structured logging JSON)
- **Testing**: pytest, pytest-asyncio (311+ test)
- **Linting**: ruff, mypy

### UI (Optional)
- **Streamlit**: Expert Debugger app (apps/expert_debugger.py)

---

## Entry Points Principali

### API Pubblica Libreria

| File | Classe/Funzione | Scopo |
|------|-----------------|-------|
| `merlt/__init__.py` | Export API pubblica | Import principale: `from merlt import LegalKnowledgeGraph` |
| `merlt/core/legal_knowledge_graph.py` | `LegalKnowledgeGraph` | Entry point principale - orchestrazione completa |
| `merlt/core/legal_knowledge_graph.py` | `MerltConfig` | Configurazione centralizzata |

### Expert System

| File | Classe | Scopo |
|------|--------|-------|
| `merlt/experts/literal.py` | `LiteralExpert` | Interpretazione letterale (Art. 12 Preleggi) |
| `merlt/experts/systemic.py` | `SystemicExpert` | Interpretazione sistematica |
| `merlt/experts/principles.py` | `PrinciplesExpert` | Intenzione del legislatore |
| `merlt/experts/precedent.py` | `PrecedentExpert` | Giurisprudenza applicativa |
| `merlt/experts/orchestrator.py` | `MultiExpertOrchestrator` | Coordinamento multi-expert |
| `merlt/experts/router.py` | `ExpertRouter` | Query classification |
| `merlt/experts/gating.py` | `GatingNetwork` | Response aggregation |

### RLCF Framework

| File | Classe | Scopo |
|------|--------|-------|
| `merlt/rlcf/policy_gradient.py` | `GatingPolicy` | Neural routing policy |
| `merlt/rlcf/policy_gradient.py` | `TraversalPolicy` | Graph traversal policy |
| `merlt/rlcf/policy_gradient.py` | `PolicyGradientTrainer` | REINFORCE training |
| `merlt/rlcf/authority.py` | `AuthorityModule` | Expert authority scoring |
| `merlt/rlcf/aggregation.py` | `AggregationEngine` | Feedback aggregation |
| `merlt/rlcf/simulator/experiment.py` | `RLCFExperiment` | RLCF simulation framework |

### Storage & Retrieval

| File | Classe | Scopo |
|------|--------|-------|
| `merlt/storage/graph/client.py` | `FalkorDBClient` | Graph database client |
| `merlt/storage/vectors/embeddings.py` | `EmbeddingService` | Embeddings + Qdrant |
| `merlt/storage/bridge/bridge_table.py` | `BridgeTable` | Chunk-to-node mapping |
| `merlt/storage/retriever/hybrid.py` | `GraphAwareRetriever` | Hybrid semantic+graph retrieval |

### Pipeline

| File | Classe | Scopo |
|------|--------|-------|
| `merlt/pipeline/ingestion.py` | `IngestionPipelineV2` | Ingestion completa |
| `merlt/pipeline/parsing.py` | `CommaParser` | Parser testi giuridici |
| `merlt/pipeline/chunking.py` | `StructuralChunker` | Chunking semantico |
| `merlt/pipeline/multivigenza.py` | `MultivigenzaPipeline` | Amendments tracking |

### Scrapers

| File | Classe | Scopo |
|------|--------|-------|
| `merlt/sources/normattiva.py` | `NormattivaScraper` | Testi ufficiali Normattiva |
| `merlt/sources/brocardi.py` | `BrocardiScraper` | Enrichment dottrina |
| `merlt/sources/eurlex.py` | `EurlexScraper` | Normativa EU (futuro) |

### Scripts Entry Points

| Script | Scopo |
|--------|-------|
| `scripts/batch_ingest_libro_iv.py` | Ingest batch Codice Civile Libro IV |
| `scripts/ingest_libro_primo_cp.py` | Ingest Codice Penale Libro I |
| `scripts/exp022_policy_simulation.py` | EXP-022 Policy Gradient Simulation |
| `scripts/run_rlcf_simulation.py` | RLCF loop simulation |
| `apps/expert_debugger.py` | Streamlit Expert Debugger UI |

---

## File Chiave

| File | Scopo |
|------|-------|
| `docs/claude-context/LIBRARY_VISION.md` | Principi guida libreria |
| `docs/claude-context/CURRENT_STATE.md` | Stato attuale - LEGGI SEMPRE ALL'INIZIO |
| `docs/claude-context/LIBRARY_ARCHITECTURE.md` | Architettura componenti |
| `merlt/core/legal_knowledge_graph.py` | Entry point principale |
| `merlt/experts/__init__.py` | Export Expert System |
| `merlt/rlcf/__init__.py` | Export RLCF Framework |
| `INTEGRATION_SUMMARY.md` | Summary integrazioni recenti |
| `docker-compose.dev.yml` | Stack development (FalkorDB, Qdrant, PostgreSQL, Redis) |

---

## Cosa NON Fare

1. **MAI duplicare codice** - Cerca prima, riusa sempre
2. **MAI logica negli scripts** - Solo entry points
3. **MAI hardcodare valori** - Usa config
4. **MAI ignorare errori** - Gestisci gracefully
5. **MAI dimenticare type hints** - Sempre

---

## Quick Reference: Frasi Magiche per Agenti

### Per Nuove Feature
```
> Usa api-designer per progettare API per [feature]
> Usa builder per implementare [componente] seguendo LIBRARY_VISION.md
> Usa validator per scrivere test completi per [modulo]
> Usa scribe per documentare [feature] in docs/
```

### Per Bug Fixing
```
> Usa debugger per analizzare [problema specifico]
> Usa builder per applicare fix con proper error handling
> Usa validator per aggiungere regression test
```

### Per Ottimizzazioni
```
> Usa data-pipeline per ottimizzare [pipeline] con checkpoint e retry
> Usa graph-engineer per ottimizzare query Cypher in [componente]
> Usa llm-orchestrator per bilanciare cost/quality in [expert]
```

### Per Schema/Database
```
> Usa database-architect per progettare schema [nuove relazioni]
> Usa builder per implementare migration in merlt/storage/graph/
> Usa validator per testare migration con database reali
```

### Multi-Step Workflow
```
> Usa orchestrator per coordinare [task complesso multi-step]
```

---

## Database Stato Corrente (Aggiornato: 28 Dicembre 2025)

| Storage | Nome | Contenuto |
|---------|------|-----------|
| **FalkorDB** | `merl_t_dev` | 27,740 nodi, 43,935 relazioni (Codice Civile Libro IV) |
| **Qdrant** | `merl_t_dev_chunks` | 5,926 vectors (E5-large embeddings) |
| **PostgreSQL** | `rlcf_dev` | Bridge Table (27,114 mappings), RLCF metadata |
| **Redis** | `dev:*` keys | Cache layer |

**IMPORTANTE**: Usa SEMPRE i nomi `_dev` in sviluppo. I database `_prod`, `_test`, `_legal` sono vuoti o inesistenti.

---

*Questa è la libreria dell'informatica giuridica italiana. Ogni riga conta.*

### Note Critiche da Ricordare

- **Test senza mock**: ricordiamoci di aggiungere test in CI/CD per tutto con database reali
- **Logging centralizzato**: salva sempre i log nella cartella `logs/` non in `/tmp`
- **Parsing robusto**: dobbiamo irrobustire il parsing per gestire tutti i possibili casi presenti su normattiva. Quindi man mano che troviamo nuovi casi dobbiamo aggiungerli per un'estrazione robusta
- **Regression prevention**: quando viene aggiunta una feature o un fix, aggiungilo ai test per evitare regression
- **Model externalization**: esternalizziamo sempre i modelli nell'env in modo da poter usare sempre il migliore per il caso d'uso specifico
- **CONVENZIONE DATABASE**: ogni database deve avere versione `_dev` e `_prod`. FalkorDB usa `merl_t_dev` per sviluppo e `merl_t_prod` per produzione. Stessa logica per Redis e altri storage
- **Config externalization**: esternalizziamo sempre i prompt e i parametri di ogni elemento della pipeline in file YAML di config
- **Source of truth**: Gli Expert devono usare SOLO fonti recuperate dal database, mai "inventare" articoli
- **API design**: Se un'operazione comune richiede più di 3 righe, ripensa l'API
- **Zero duplicazioni**: Prima cerca se esiste già in `merlt/`, poi riutilizza, infine crea nel posto giusto
- **Documentazione italiana**: Docstring in italiano, codice in inglese, esempi con casi d'uso giuridici reali