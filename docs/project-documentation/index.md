# ALIS_CORE Documentation

> **Artificial Legal Intelligence System - Technical Documentation**
>
> Generated: 2026-01-23 | Scan Level: Exhaustive | Focus: ML/AI

---

## Quick Navigation

| Document | Description |
|----------|-------------|
| [**00-project-overview.md**](./00-project-overview.md) | Executive summary, repository structure, quick start |
| [**01-architecture.md**](./01-architecture.md) | System architecture, data flow, deployment |
| [**02-merlt-experts.md**](./02-merlt-experts.md) | Multi-Expert system, 4 canons, orchestration |
| [**03-rlcf.md**](./03-rlcf.md) | Reinforcement Learning from Community Feedback |

---

## Project Summary

**ALIS** (Artificial Legal Intelligence System) is a computational sociology of law platform that implements Italian legal interpretation canons as AI algorithms.

### Key Components

| Component | Purpose | Stack |
|-----------|---------|-------|
| **merlt** | ML framework (Multi-Expert + RLCF) | Python, PyTorch, FastAPI |
| **visualex-api** | Legal source scraping | Python, Quart, Playwright |
| **visualex-platform** | Web application | React 19, Express 5 |
| **visualex-merlt** | Integration plugin | React 19, TypeScript |
| **merlt-models** | Trained model weights | SafeTensors |

### The Four Experts

| Expert | Canon | Focus |
|--------|-------|-------|
| LiteralExpert | Art. 12, I | Textual meaning |
| SystemicExpert | Art. 12, I + 14 | Normative context |
| PrinciplesExpert | Art. 12, II | Legislative intent |
| PrecedentExpert | Practice | Case law |

### RLCF Pillars

1. **Dynamic Authority Scoring** - Weight feedback by competence
2. **Uncertainty Preservation** - Maintain calibrated uncertainty
3. **Constitutional Governance** - Immutable principles
4. **Devil's Advocate** - Challenge conformism

---

## Development Quick Start

```bash
# Start all services
cd ALIS_CORE
./start_dev.sh

# Or individually:
cd merlt && pip install -e ".[dev]" && ./start_dev.sh          # Port 8000
cd visualex-api && pip install -e ".[dev]" && ./start_dev.sh    # Port 5000
cd visualex-platform/frontend && npm install && npm run dev     # Port 5173
cd visualex-platform/backend && npm install && npm run dev      # Port 3001
```

---

## Existing Documentation

### Per-Component CLAUDE.md Files
- `/merlt/CLAUDE.md` - ML framework instructions
- `/visualex-api/CLAUDE.md` - Scraping library instructions
- `/visualex-platform/CLAUDE.md` - Platform instructions
- `/visualex-merlt/CLAUDE.md` - Integration instructions

### Core Documentation
- `/core_docs/GLOSSARIO.md` - Legal/technical terminology
- `/core_docs/ARCHITETTURA.md` - Non-technical architecture
- `/core_docs/GUIDA_NAVIGAZIONE.md` - Code navigation

### Academic Papers
- `/papers/markdown/DA GP - MERLT.md` - MERL-T paper
- `/papers/markdown/DA GP - RLCF.md` - RLCF paper
- `/papers/markdown/DA GP - ALIS.md` - ALIS platform paper

---

## Recommended Agent Usage

| Task | Agent |
|------|-------|
| Explore codebase | `architect` |
| Implement features | `builder` |
| Fix bugs | `debugger` |
| ML components | `graph-engineer`, `builder` |
| API design | `api-designer` |
| Frontend | `frontend-builder` |
| Documentation | `scribe` |
| Testing | `validator` |

---

## Contact

- **Project Lead**: Gpuzio
- **Repository**: ALIS_CORE (private)
