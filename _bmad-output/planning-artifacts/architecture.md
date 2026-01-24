---
stepsCompleted: ['step-01-init', 'step-02-context']
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/research/technical-vector-space-legal-interpretation-research-2026-01-23.md
  - docs/project-documentation/index.md
  - docs/project-documentation/00-project-overview.md
  - docs/project-documentation/01-architecture.md
  - docs/project-documentation/02-merlt-experts.md
  - docs/project-documentation/03-rlcf.md
workflowType: 'architecture'
project_name: 'ALIS_CORE'
user_name: 'Gpuzio'
date: '2026-01-24'
lastStep: 'step-02-context'
---

# Architecture Decision Document - ALIS_CORE

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Project Context Analysis

### Requirements Overview

**Functional Requirements (50 FR in 9 categories):**

| Category | # FR | Architectural Implications |
|----------|------|---------------------------|
| Query & Legal Analysis | 6 | Expert pipeline, NLP, temporal queries |
| Traceability & Source | 5 | Audit trail, citation linking, export |
| Knowledge Graph & Norm | 7 | Graph DB, scraping, versioning |
| RLCF Feedback | 8 | Authority scoring, policy learning |
| User & Authority | 5 | Identity, scoring calculation |
| Consent & Privacy | 5 | GDPR Art. 6/17/20, anonymization |
| System Admin | 6 | Monitoring, circuit breakers, regression |
| API & Integration | 4 | REST API, auth, documentation |
| Academic Research | 4 | Reproducibility, export, analytics |

**Non-Functional Requirements:**

| Category | Key Target | Impact |
|----------|------------|--------|
| Performance | <3min first, <500ms cached, 20 concurrent | Progressive loading, caching strategy |
| Security | AES-256, TLS 1.3, JWT, PII anonymization | Encryption at rest/transit, consent mgmt |
| Reliability | 99% uptime, 7y audit retention, circuit breakers | Graceful degradation, backup strategy |
| Scalability | 50 MVP users, 10k norms, 1k feedback/month | Fork-friendly, horizontal ready |
| Integration | LLM abstraction, API versioning, Normattiva | Multi-provider, scraping resilience |
| Compliance | GDPR Art. 6/17/20/89 | Consent flow, data portability, research exemption |

### Scale & Complexity

| Indicator | Value | Notes |
|-----------|-------|-------|
| Complexity | HIGH | 4-Expert pipeline + RLCF + Knowledge Graph |
| Technical Domain | Full-stack ML Platform | Python ML + Node API + React Frontend |
| Real-time Features | Medium | Progressive enrichment, not real-time collab |
| Multi-tenancy | Single + Fork | No SaaS multi-tenant, but fork-friendly |
| Regulatory | GDPR + Legal domain | High compliance burden |
| Data Complexity | High | Graph + Vector + Relational + Temporal |
| Integration | Medium-High | LLM providers, Normattiva, EUR-Lex |

**Complexity Level:** HIGH (thesis project + production aspirations)

### Existing Architecture (Brownfield)

```
┌─────────────────────────────────────────────────────────────┐
│                    CURRENT STACK                             │
├─────────────────────────────────────────────────────────────┤
│  Frontend: React 19 + Vite 7 + Tailwind v4 + TypeScript     │
│  Backend:  Express 5 (Platform) + FastAPI (MERL-T) + Quart  │
│  ML:       PyTorch 2.0+ + Transformers + Custom Experts     │
│  Data:     PostgreSQL + FalkorDB + Qdrant + Redis           │
│  Infra:    Docker Compose (dev)                             │
└─────────────────────────────────────────────────────────────┘
```

**Existing Components:**
- `merlt/` - ML Framework (Expert pipeline, RLCF)
- `visualex-api/` - Scraping library (Normattiva, Brocardi)
- `visualex-platform/` - Web app (frontend + backend)
- `visualex-merlt/` - Integration plugin (8 slots, 25 events)
- `merlt-models/` - Model weights

### Technical Constraints & Dependencies

1. **Dual API Stack** - FastAPI (MERL-T) + Quart (VisuaLex) + Express (Platform)
2. **GPU Separation** - LLM/embedding inference needs dedicated resources
3. **Thesis Deadline** - May 2026 (demo quality required)
4. **Team Size** - ~20 association members (modular monolith appropriate)
5. **Fork-Friendly** - Must support other organizations forking

### Cross-Cutting Concerns

| Concern | Components Impacted | Decisions Required |
|---------|--------------------|--------------------|
| Authentication | Platform, MERL-T API | Unified JWT vs separate |
| Traceability | Experts, RLCF, API | Trace ID propagation |
| Caching | Experts, Graph, Vectors | Redis strategy, TTL policy |
| Temporal Versioning | Graph, Experts, API | as_of_date implementation |
| GDPR Consent | All with user data | Consent middleware |
| Error Handling | Expert pipeline | Circuit breaker patterns |
| Logging | All services | Structured logging, correlation |
| LLM Abstraction | Experts | Multi-provider interface |

### Research-Validated Decisions

The Technical Research already validated:

- ✅ Modular Monolith - Appropriate for team <50
- ✅ HybridRAG (Graph+Vector) - Best practice 2024-2025
- ✅ Code-driven orchestration - Compliance with Art. 12
- ✅ ReAct for Expert reasoning - Multi-step reasoning standard
- ✅ RLCF as RLHF variant - Authority weighting + Constitutional Governance

**Proposed ADRs from research:**
- ADR-001: Modular Monolith with GPU Separation
- ADR-002: ReAct for Expert Internals
- ADR-003: Code-Driven Expert Orchestration
- ADR-004: Neuro-Symbolic with External Structure
- ADR-005: RLCF as Authority-Weighted RLHF

