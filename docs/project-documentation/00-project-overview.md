# ALIS_CORE - Project Overview

> **Generated:** 2026-01-23 | **Scan Level:** Exhaustive | **Focus:** ML/AI

---

## Executive Summary

**ALIS** (Artificial Legal Intelligence System) is a computational sociology of law platform that implements the hermeneutic canons of Article 12 of the Italian Civil Code Preliminaries (Preleggi) as executable algorithms.

The system uses a **Multi-Expert architecture** where 4 specialized AI agents analyze legal questions according to different interpretive approaches, with results synthesized through a **gating network** and continuously improved via **RLCF** (Reinforcement Learning from Community Feedback).

---

## Repository Type

**Monorepo** with 5 active components + Legacy archive

| Component | Type | Stack | License | Status |
|-----------|------|-------|---------|--------|
| `merlt` | Python ML Library | PyTorch, FastAPI, FalkorDB, Qdrant | Apache 2.0 | Active |
| `merlt-models` | Model Weights | PyYAML, SafeTensors | Proprietary | Active |
| `visualex-api` | Python Scraping Library | Quart, BeautifulSoup, Playwright | MIT | Active |
| `visualex-platform` | Web Application | React 19, Express 5, Prisma, PostgreSQL | Proprietary | Active |
| `visualex-merlt` | Integration Plugin | React 19, TypeScript | Proprietary | Active |
| `Legacy/` | Archive | - | - | Archived |

---

## Core Concepts

### 1. Multi-Expert System (MERL-T)

The system implements 4 "virtual experts" based on Italian legal interpretation canons:

| Expert | Canon | Implementation |
|--------|-------|----------------|
| **LiteralExpert** | "significato proprio delle parole" (Art. 12, I) | Textual analysis, definitions |
| **SystemicExpert** | "connessione di esse" (Art. 12, I) | Normative context, graph traversal |
| **PrinciplesExpert** | "intenzione del legislatore" (Art. 12, II) | Constitutional principles, ratio legis |
| **PrecedentExpert** | Jurisprudential practice | Case law, precedents |

**Architecture Flow:**
```
Query → ExpertRouter → [Expert1, Expert2, Expert3, Expert4] → GatingNetwork → Response
                              ↓                                   ↓
                          Tools (search, graph)              Synthesizer
```

### 2. RLCF (Reinforcement Learning from Community Feedback)

Novel framework for improving AI through expert community feedback:

| Pillar | Description |
|--------|-------------|
| **Dynamic Authority Scoring** | Weights feedback based on user competence |
| **Uncertainty Preservation** | Maintains uncertainty where appropriate |
| **Constitutional Governance** | Immutable guiding principles |
| **Devil's Advocate System** | Deliberate challenge to avoid conformism |

**Components:**
- `policy_gradient.py`: REINFORCE algorithm for gating/traversal policies
- `ppo_trainer.py`: PPO for complex policies (legacy)
- `react_ppo_trainer.py`: PPO for multi-step expert reasoning
- `authority.py`: Dynamic authority calculation
- `devils_advocate.py`: Critical thinking assignment
- `bias_detection.py`: 6-dimensional bias detection

### 3. Knowledge Graph

Legal knowledge represented as a graph:

- **Nodes**: Articles, clauses, definitions, concepts
- **Edges**: Semantic relationships (RIFERIMENTO, MODIFICA, DEROGA, CITATO_DA, etc.)
- **Storage**: FalkorDB (Redis-compatible graph database)
- **Vector Search**: Qdrant for semantic retrieval

---

## Data Flow

```
                        ┌─────────────────────────┐
                        │       USER/CLIENT       │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │      VISUALEX-PLATFORM        │
                    │   (React 19 + Express 5)      │
                    └───────────────┬───────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
              ▼                     ▼                     ▼
    ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
    │  VISUALEX-API   │   │ VISUALEX-MERLT  │   │   PostgreSQL    │
    │  (Scraping)     │   │  (Plugin)       │   │   (User Data)   │
    └────────┬────────┘   └────────┬────────┘   └─────────────────┘
             │                     │
             │                     ▼
             │            ┌─────────────────┐
             │            │     MERLT       │
             │            │  (ML Framework) │
             │            └────────┬────────┘
             │                     │
             │    ┌────────────────┼────────────────┐
             │    │                │                │
             │    ▼                ▼                ▼
             │  ┌─────────┐  ┌─────────────┐  ┌─────────┐
             │  │ Qdrant  │  │  FalkorDB   │  │  Redis  │
             │  │(Vectors)│  │   (Graph)   │  │ (Cache) │
             │  └─────────┘  └─────────────┘  └─────────┘
             │
             └──────────────▶ External Sources
                             - Normattiva.it
                             - Brocardi.it
                             - EUR-Lex
```

---

## Technology Stack Summary

### Backend (Python)
- **Python 3.10+**
- **PyTorch 2.0+** - ML training
- **Transformers (HuggingFace)** - Embeddings, LLM
- **FastAPI** - MERL-T API server
- **Quart** - VisuaLex-API async server
- **SQLAlchemy 2.0** - ORM
- **Pydantic 2.0** - Data validation
- **structlog** - Logging

### Frontend (TypeScript)
- **React 19**
- **TypeScript 5.x**
- **Vite 7** - Build tool
- **Tailwind CSS v4** - Styling
- **Zustand 5** - State management
- **React Router 7** - Routing

### Backend (Node.js)
- **Express 5**
- **Prisma ORM**
- **JWT** - Authentication
- **Helmet** - Security

### Databases
- **PostgreSQL 15+** - User data, platform data
- **FalkorDB** - Knowledge graph (Cypher queries)
- **Qdrant** - Vector search
- **Redis** - Caching

### DevOps
- **Docker Compose** - Local development
- **pytest** - Python testing
- **Vitest** - Frontend testing
- **Black + Ruff** - Python formatting/linting
- **ESLint** - TypeScript linting

---

## Academic Foundation

This project implements research documented in:

1. **Allega, D. (2025)** - "The Artificial Legal Intelligence Society as an open, multi-sided platform for law-as-computation"

2. **Allega, D., & Puzio, G. (2025b)** - "MERL-T: A multi-expert architecture for trustworthy artificial legal intelligence"

3. **Allega, D., & Puzio, G. (2025c)** - "Reinforcement learning from community feedback (RLCF): A novel framework for artificial intelligence in social science domains"

4. **Allega, D., & Puzio, G. (2025a)** - "The knowledge commoditization paradox: Theoretical and practical challenges of AI-driven value extraction in information-intensive organizations"

---

## Quick Start

### Development Setup

```bash
# Clone and setup
cd ALIS_CORE

# Start all services
./start_dev.sh   # Docker Compose

# Or individually:

# MERL-T Framework (port 8000)
cd merlt && pip install -e ".[dev]" && ./start_dev.sh

# VisuaLex API (port 5000)
cd visualex-api && pip install -e ".[dev]" && ./start_dev.sh

# VisuaLex Platform Frontend (port 5173)
cd visualex-platform/frontend && npm install && npm run dev

# VisuaLex Platform Backend (port 3001)
cd visualex-platform/backend && npm install && npm run dev
```

---

## Next Steps

1. **[Architecture Documentation](./01-architecture.md)** - Deep dive into system architecture
2. **[MERL-T Expert System](./02-merlt-experts.md)** - Multi-Expert implementation details
3. **[RLCF Framework](./03-rlcf.md)** - Reinforcement Learning from Community Feedback
4. **[API Reference](./04-api-reference.md)** - API endpoints and usage
5. **[Development Guide](./05-development.md)** - Contributing and development workflow
