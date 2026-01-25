---
stepsCompleted: ['step-01-init', 'step-02-context', 'step-03-starter', 'step-04-decisions', 'step-05-system', 'step-06-data', 'step-07-security', 'step-08-deployment', 'step-09-summary']
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/research/technical-vector-space-legal-interpretation-research-2026-01-23.md
  - docs/project-documentation/index.md
  - docs/project-documentation/00-project-overview.md
  - docs/project-documentation/01-architecture.md
  - docs/project-documentation/02-merlt-experts.md
  - docs/project-documentation/03-rlcf.md
  - Legacy/MERL-T_alpha/docs/MERL_T_ARCHITECTURE_MAP.md
  - merlt/merlt/pipeline/multivigenza.py
workflowType: 'architecture'
project_name: 'ALIS_CORE'
user_name: 'Gpuzio'
date: '2026-01-24'
lastStep: 'step-09-summary'
status: 'complete'
adrs:
  - ADR-001: Circuit Breaker Strategy
  - ADR-002: GDPR Consent Management
  - ADR-003: API Versioning Strategy
  - ADR-004: Audit Trail (7-Year Retention)
  - ADR-005: Warm-Start Caching Strategy
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

---

## Stack Evaluation (Brownfield)

### Layered Service Architecture

The API separation is **intentional by design** to support different deployment scenarios and risk profiles:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LAYERED SERVICE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  LAYER 3: MERL-T (Optional AI Layer)                                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  FastAPI - Port 8000                                         │   │
│  │  • Expert pipeline (AI-powered)                              │   │
│  │  • RLCF feedback loop                                        │   │
│  │  • Higher risk profile                                       │   │
│  │  • Requires consent, audit trail                             │   │
│  │  Target: Association members, researchers                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓ consumes                             │
│  LAYER 2: VisuaLex-API (Mechanical Layer)                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Quart - Port 5000                                           │   │
│  │  • Norm scraping (Normattiva, Brocardi, EUR-Lex)            │   │
│  │  • URN generation                                            │   │
│  │  • No AI, deterministic                                      │   │
│  │  • Low risk, can be distributed to PA                        │   │
│  │  Target: PA, legal publishers, anyone                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓ consumes                             │
│  LAYER 1: Platform Backend (User Layer)                            │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Express 5 - Port 3001                                       │   │
│  │  • User auth, profiles                                       │   │
│  │  • Dossier management                                        │   │
│  │  • Preferences                                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Deployment Scenarios

| Scenario | VisuaLex | MERL-T | Platform | Target |
|----------|----------|--------|----------|--------|
| PA-only | ✅ | ❌ | Optional | Pubbliche Amministrazioni |
| Research | ✅ | ✅ | ✅ | Association + Researchers |
| Full Platform | ✅ | ✅ | ✅ | End users |
| API-only | ✅ | ✅ | ❌ | Developers, integrators |

### Risk Profile per Layer

| Layer | AI Risk | GDPR Burden | Distribution |
|-------|---------|-------------|--------------|
| VisuaLex-API | None | Minimal | Open (PA, publishers) |
| MERL-T | High | Full compliance | Controlled (association) |
| Platform | Low | User data only | Controlled |

### Current Stack Status

| Component | Version | Status |
|-----------|---------|--------|
| React | 19 | ✅ Current |
| Vite | 7 | ⚠️ Beta |
| Tailwind | v4 | ✅ Current |
| FastAPI | Latest | ✅ Current |
| Quart | Latest | ✅ Current |
| Express | 5 | ✅ Current |
| PyTorch | 2.0+ | ✅ Current |
| PostgreSQL | 15+ | ✅ Current |
| FalkorDB | Latest | ✅ Current |
| Qdrant | Latest | ✅ Current |

### Gaps to Address

| Gap | Priority | Layer Affected | Status |
|-----|----------|----------------|--------|
| ~~Temporal versioning~~ | - | - | ✅ Already implemented (MultivigenzaPipeline) |
| ~~Circuit breakers~~ | - | - | ✅ ADR-001 |
| ~~GDPR consent management~~ | - | - | ✅ ADR-002 |
| ~~API versioning~~ | - | - | ✅ ADR-003 |
| ~~Audit trail (7 year)~~ | - | - | ✅ ADR-004 |
| ~~Warm-start caching~~ | - | - | ✅ ADR-005 |

### Stack Decision Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Keep API separation | 3 services distinct | Different risk profiles, deployment flexibility |
| VisuaLex standalone | Quart async | Can be distributed to PA without AI risk |
| MERL-T optional layer | FastAPI | AI layer with full compliance requirements |
| Platform separate runtime | Express 5 | User-facing, Node ecosystem |
| Keep frontend | React 19 + Vite + Tailwind | Already cutting-edge |
| Keep ML stack | PyTorch + Transformers | Industry standard |
| Keep data layer | PG + FalkorDB + Qdrant + Redis | Appropriate for hybrid RAG |

---

## Architectural Decisions Record (ADR)

### ADR-001: Circuit Breaker Strategy

**Status:** Accepted
**Date:** 2026-01-24
**Context:** MultiExpertOrchestrator runs 4 Experts in parallel (asyncio). Each Expert may depend on external services (LLM providers, FalkorDB, Qdrant). Need resilience without over-engineering for thesis timeline.

**Decision:** Lightweight Circuit Breaker + Observability

**Implementation:**
- Per-service circuit breakers: `falkordb`, `qdrant`, `llm`
- States: CLOSED (normal) → OPEN (blocked after 3-5 failures) → HALF_OPEN (test recovery)
- Recovery timeout: 30 seconds
- GatingNetwork already handles missing Expert responses gracefully
- Structured logging for observability (successful/failed counts, circuit states)

**Consequences:**
- ✅ Prevents cascading failures when external services are down
- ✅ Automatic recovery without manual intervention
- ✅ Full visibility via structured logs
- ✅ Minimal code change (decorator pattern)
- ⚠️ Responses may be degraded (2-3 Experts instead of 4)

**Code Location:** `merlt/resilience/circuit_breaker.py` (to create)

**Integration Points:**
- `MultiExpertOrchestrator._run_expert_with_circuit()`
- Metadata propagated to response: `degraded: bool`, `experts_failed: list`

---

### ADR-002: GDPR Consent Management

**Status:** Accepted
**Date:** 2026-01-24
**Context:** MERL-T processes user queries and collects RLCF feedback. GDPR requires explicit consent for AI processing (Art. 6), right to withdraw (Art. 7), and data portability (Art. 20). 7-year audit trail adds complexity.

**Decision:** Consent Gateway in Platform with JWT Claims

**Implementation:**

1. **ConsentService (Platform/Express)**
   - PostgreSQL table: `user_consents` with versioning
   - Consent types: `ai_analysis`, `audit_trail`, `rlcf_feedback`, `research`
   - API: `GET/PUT /user/{id}/consents`, `POST /user/{id}/withdraw`, `GET /user/{id}/export`
   - Generates JWT with consent claims

2. **ConsentMiddleware (MERL-T/FastAPI)**
   - Validates JWT consent claims on every request
   - Blocks requests if required consent missing
   - Logs consent state for audit trail

3. **Consent Withdrawal Flow**
   - Platform updates DB, generates new JWT
   - Webhook to MERL-T: `POST /internal/consent-withdrawn`
   - MERL-T anonymizes: feedback → hash, queries → soft delete

**JWT Claims Structure:**
```json
{
  "consents": {
    "ai_analysis": true,
    "audit_trail": true,
    "rlcf_feedback": false,
    "research": false
  },
  "consent_version": "2026-01"
}
```

**Consequences:**
- ✅ Single source of truth (Platform DB)
- ✅ Stateless verification (JWT claims)
- ✅ Clear audit trail (consent logged with each request)
- ✅ Supports VisuaLex standalone (no consent needed for mechanical API)
- ⚠️ JWT refresh needed after consent change

**Code Locations:**
- `visualex-platform/src/services/consent.ts` (to create)
- `merlt/api/middleware/consent.py` (to create)

---

### ADR-003: API Versioning Strategy

**Status:** Accepted
**Date:** 2026-01-24
**Context:** 3 distinct APIs (Platform, VisuaLex, MERL-T) with different consumers and stability requirements. VisuaLex may be distributed to PA (external consumers).

**Decision:** URL Path Versioning

**Implementation:**
```
Platform:   /api/v1/users, /api/v1/dossiers
VisuaLex:   /api/v1/norme, /api/v1/urn
MERL-T:     /api/v1/analyze, /api/v1/feedback
```

**Conventions:**
- `v1` = MVP release
- Breaking change → increment version, maintain previous for 6 months
- Deprecation headers: `Deprecation: true`, `Sunset: <date>`
- OpenAPI spec maintained per version

**Consequences:**
- ✅ Industry standard, widely understood
- ✅ Easy to test (visible in URL)
- ✅ Clear versioning for external consumers (PA)
- ✅ Native support in FastAPI, Express, Quart
- ⚠️ Route handler duplication when versioning

---

### ADR-004: Audit Trail (7-Year Retention)

**Status:** Accepted
**Date:** 2026-01-24
**Context:** GDPR and legal traceability require 7-year audit retention. Every query must have traceable sources. Estimated volume: ~1.8 GB over 7 years (50 MVP users).

**Decision:** PostgreSQL with Time-Based Partitioning

**Schema:**
```sql
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL,  -- UUID v7 (time-ordered)
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID,
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL
) PARTITION BY RANGE (created_at);

-- Annual partitions
CREATE TABLE audit_trail_2026 PARTITION OF audit_trail
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');
```

**Event Types:**
- `query_submitted` - User query with consent state
- `expert_responded` - Each expert's response (4 per query)
- `synthesis_complete` - Final aggregated response
- `feedback_received` - RLCF feedback
- `consent_changed` - Consent grant/withdrawal

**Indexes:**
- `(trace_id)` - Reconstruct full request flow
- `(user_id, created_at)` - User history queries
- `(event_type, created_at)` - Analytics

**Retention Policy:**
- Partitions older than 7 years → `DROP PARTITION`
- GDPR withdrawal → `UPDATE SET user_id = hash(user_id)` (pseudonymization)

**Consequences:**
- ✅ Uses existing PostgreSQL (no new infrastructure)
- ✅ SQL queries for audit/analytics
- ✅ Partition pruning for performance
- ✅ Easy backup (pg_dump per partition)
- ✅ GDPR-compliant pseudonymization
- ⚠️ Requires annual partition creation (automate via cron/pg_partman)

**Code Location:** `merlt/audit/logger.py`, `merlt/audit/models.py` (to create)

---

### ADR-005: Warm-Start Caching Strategy

**Status:** Accepted
**Date:** 2026-01-24
**Context:** NFR requires <3min first response, <500ms cached, 20 concurrent users. LLM cold start can exceed 3 minutes. Redis already in stack.

**Decision:** Hybrid Redis Cache + Startup Warm-up

**Implementation:**

1. **Startup Warm-up Script** (`scripts/warm_up.py`)
   - Pre-load LLM models into memory
   - Pre-compute embeddings for top 100 query patterns (from audit analytics)
   - Pre-cache graph traversals for core norms (Art. 1-100 CC)
   - Health check all connections

2. **Runtime Redis Cache**
   - Multi-layer: embeddings, expert responses, synthesis, graph queries
   - Deterministic cache keys via SHA256 hash of (query + context)

**Redis Key Structure:**
```
merlt:embedding:{hash}     → numpy bytes      TTL: 24h
merlt:expert:{name}:{hash} → ExpertResponse   TTL: 1h
merlt:synthesis:{hash}     → SynthesisResult  TTL: 24h
merlt:graph:{hash}         → CypherResult     TTL: 1h
```

**Invalidation Strategy:**
- Norm updated → Pattern scan and delete matching keys
- Graph updated → Delete all graph cache keys
- Manual: `redis-cli KEYS "merlt:*" | xargs redis-cli DEL`

**Cache Key Function:**
```python
def cache_key(prefix: str, query: str, context: dict) -> str:
    payload = {"query": query, **context}
    hash_input = json.dumps(payload, sort_keys=True)
    return f"merlt:{prefix}:{hashlib.sha256(hash_input.encode()).hexdigest()[:16]}"
```

**Consequences:**
- ✅ First query <3min (models pre-loaded)
- ✅ Cached queries <500ms (Redis hit)
- ✅ 20 concurrent users supported (Redis scales)
- ✅ Uses existing Redis (no new infra)
- ⚠️ Warm-up adds ~2min to startup time
- ⚠️ Cache invalidation on norm updates needs care

**Code Locations:**
- `merlt/cache/redis_cache.py` (to create)
- `scripts/warm_up.py` (to create)

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              ALIS_CORE SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         PRESENTATION LAYER                           │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              React 19 + Vite 7 + Tailwind v4                 │    │    │
│  │  │  • Query Interface    • Dossier Management    • RLCF UI     │    │    │
│  │  │  • Source Viewer      • Authority Dashboard   • Consent UI  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          API GATEWAY LAYER                           │    │
│  │                                                                      │    │
│  │   Layer 1: Platform         Layer 2: VisuaLex      Layer 3: MERL-T  │    │
│  │   ┌──────────────┐         ┌──────────────┐       ┌──────────────┐  │    │
│  │   │ Express 5    │         │ Quart        │       │ FastAPI      │  │    │
│  │   │ :3001        │         │ :5000        │       │ :8000        │  │    │
│  │   │              │         │              │       │              │  │    │
│  │   │ • Auth/JWT   │ ──────▶ │ • Scraping   │ ◀──── │ • Experts    │  │    │
│  │   │ • Users      │         │ • URN Gen    │       │ • RLCF       │  │    │
│  │   │ • Consent    │         │ • Parsing    │       │ • Synthesis  │  │    │
│  │   │ • Dossiers   │         │              │       │ • Audit      │  │    │
│  │   └──────────────┘         └──────────────┘       └──────────────┘  │    │
│  │        │                         │                      │           │    │
│  │   Risk: Low                 Risk: None              Risk: High      │    │
│  │   Dist: Controlled          Dist: PA/Open           Dist: Controlled│    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        ML/AI PROCESSING LAYER                        │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │              MultiExpertOrchestrator (Parallel)              │    │    │
│  │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │    │    │
│  │  │  │ Literal  │ │ Systemic │ │Principles│ │Precedent │       │    │    │
│  │  │  │ Expert   │ │ Expert   │ │ Expert   │ │ Expert   │       │    │    │
│  │  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘       │    │    │
│  │  │       └────────────┴────────────┴────────────┘              │    │    │
│  │  │                         │                                    │    │    │
│  │  │              ┌──────────▼──────────┐                        │    │    │
│  │  │              │   GatingNetwork     │                        │    │    │
│  │  │              │ (Weighted Aggregate)│                        │    │    │
│  │  │              └──────────┬──────────┘                        │    │    │
│  │  │                         │                                    │    │    │
│  │  │              ┌──────────▼──────────┐                        │    │    │
│  │  │              │ AdaptiveSynthesizer │                        │    │    │
│  │  │              └────────────────────┘                         │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │                    RLCF Feedback Loop                        │    │    │
│  │  │  AuthorityScoring → FeedbackAggregation → PolicyLearning    │    │    │
│  │  │  DevilsAdvocate → ConstitutionalGovernance                  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│                                      ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                          DATA LAYER                                  │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │    │
│  │   │ PostgreSQL   │  │  FalkorDB    │  │   Qdrant     │             │    │
│  │   │              │  │              │  │              │             │    │
│  │   │ • Users      │  │ • Norms      │  │ • Chunks     │             │    │
│  │   │ • Consents   │  │ • Relations  │  │ • Embeddings │             │    │
│  │   │ • Audit Trail│  │ • Concepts   │  │ • Case Law   │             │    │
│  │   │ • Dossiers   │  │ • Temporal   │  │              │             │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │    │
│  │          │                  │                  │                    │    │
│  │          └──────────────────┼──────────────────┘                    │    │
│  │                             │                                       │    │
│  │                    ┌────────▼────────┐                             │    │
│  │                    │     Redis       │                             │    │
│  │                    │  • Cache        │                             │    │
│  │                    │  • Sessions     │                             │    │
│  │                    └─────────────────┘                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      EXTERNAL INTEGRATIONS                           │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐             │    │
│  │   │ Normattiva   │  │  Brocardi    │  │  EUR-Lex     │             │    │
│  │   │ (Scraping)   │  │  (Scraping)  │  │  (API)       │             │    │
│  │   └──────────────┘  └──────────────┘  └──────────────┘             │    │
│  │                                                                      │    │
│  │   ┌──────────────┐  ┌──────────────┐                               │    │
│  │   │ LLM Provider │  │ Embedding    │                               │    │
│  │   │ (Anthropic/  │  │ Provider     │                               │    │
│  │   │  OpenAI/etc) │  │ (local/API)  │                               │    │
│  │   └──────────────┘  └──────────────┘                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow (Query Analysis)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUERY ANALYSIS FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. User submits query via React UI                                         │
│     │                                                                        │
│     ▼                                                                        │
│  2. Platform API validates JWT + consent claims                             │
│     │                                                                        │
│     ▼                                                                        │
│  3. Forward to MERL-T /api/v1/analyze                                       │
│     │                                                                        │
│     ▼                                                                        │
│  4. ConsentMiddleware checks ai_analysis=true                               │
│     │                                                                        │
│     ▼                                                                        │
│  5. AuditLogger creates trace_id, logs QUERY_SUBMITTED                      │
│     │                                                                        │
│     ▼                                                                        │
│  6. Cache check (Redis) ─── HIT ──▶ Return cached response                  │
│     │                                                                        │
│     │ MISS                                                                   │
│     ▼                                                                        │
│  7. ExpertRouter classifies query → selects relevant experts                │
│     │                                                                        │
│     ▼                                                                        │
│  8. MultiExpertOrchestrator dispatches to 4 Experts (parallel)              │
│     │                                                                        │
│     ├──▶ LiteralExpert ──▶ Vector search + LLM                              │
│     ├──▶ SystemicExpert ──▶ Graph query + LLM                               │
│     ├──▶ PrinciplesExpert ──▶ Constitutional principles + LLM               │
│     └──▶ PrecedentExpert ──▶ Case law search + LLM                          │
│     │                                                                        │
│     ▼                                                                        │
│  9. GatingNetwork aggregates (confidence-weighted)                          │
│     │                                                                        │
│     ▼                                                                        │
│  10. AdaptiveSynthesizer generates final response                           │
│      │                                                                       │
│      ▼                                                                       │
│  11. AuditLogger logs SYNTHESIS_COMPLETE                                    │
│      │                                                                       │
│      ▼                                                                       │
│  12. Cache store (Redis)                                                    │
│      │                                                                       │
│      ▼                                                                       │
│  13. Return response with sources + confidence + trace_id                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Architecture

### Database Responsibilities

| Database | Purpose | Data Types | Query Patterns |
|----------|---------|------------|----------------|
| **PostgreSQL** | Relational data | Users, consents, dossiers, audit trail | CRUD, joins, time-range |
| **FalkorDB** | Knowledge Graph | Norms, relations, concepts, temporal versions | Cypher traversals, paths |
| **Qdrant** | Vector Search | Document chunks, embeddings, case law | ANN similarity, filters |
| **Redis** | Cache + Sessions | Query cache, JWT sessions, warm-up data | Key-value, TTL |

### PostgreSQL Schema

```sql
-- Core entities
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    consent_type VARCHAR(50) NOT NULL,  -- ai_analysis, audit_trail, rlcf_feedback, research
    granted BOOLEAN NOT NULL,
    granted_at TIMESTAMPTZ,
    withdrawn_at TIMESTAMPTZ,
    version VARCHAR(20) NOT NULL,  -- consent version for legal tracking
    UNIQUE(user_id, consent_type)
);

CREATE TABLE dossiers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE dossier_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dossier_id UUID REFERENCES dossiers(id) ON DELETE CASCADE,
    item_type VARCHAR(50) NOT NULL,  -- norm, query, note
    content JSONB NOT NULL,
    position INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Audit trail (partitioned - see ADR-004)
CREATE TABLE audit_trail (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    user_id UUID,  -- nullable for anonymized records
    event_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL
) PARTITION BY RANGE (created_at);

-- RLCF data
CREATE TABLE rlcf_feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    trace_id UUID NOT NULL,  -- links to audit_trail
    expert_name VARCHAR(50),
    feedback_type VARCHAR(50) NOT NULL,  -- agree, disagree, partial, correction
    feedback_value JSONB,
    authority_score FLOAT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE user_authority_scores (
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    domain VARCHAR(100) NOT NULL,
    score FLOAT NOT NULL,
    components JSONB,  -- breakdown: background, consistency, consensus, domain_expertise
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (user_id, domain)
);
```

### FalkorDB Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KNOWLEDGE GRAPH SCHEMA                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  NODE TYPES                                                                  │
│  ═══════════                                                                 │
│  (:Norma {                                                                   │
│      urn: STRING,           # urn:nir:stato:legge:1942-03-16;262~art1       │
│      tipo: STRING,          # legge, decreto, regolamento                   │
│      titolo: STRING,                                                         │
│      testo: STRING,                                                          │
│      data_pubblicazione: DATE,                                               │
│      data_vigenza: DATE,                                                     │
│      versione: STRING       # For multivigenza: !vig=2024-01-01             │
│  })                                                                          │
│                                                                              │
│  (:Articolo {                                                                │
│      urn: STRING,                                                            │
│      numero: STRING,        # "1", "1-bis", "2"                             │
│      rubrica: STRING,                                                        │
│      testo: STRING,                                                          │
│      versione: STRING                                                        │
│  })                                                                          │
│                                                                              │
│  (:Comma {                                                                   │
│      urn: STRING,                                                            │
│      numero: INTEGER,                                                        │
│      testo: STRING                                                           │
│  })                                                                          │
│                                                                              │
│  (:Concetto {                                                                │
│      id: STRING,            # "risoluzione_contratto"                       │
│      label: STRING,         # "Risoluzione del contratto"                   │
│      definizione: STRING                                                     │
│  })                                                                          │
│                                                                              │
│  (:Massima {                                                                 │
│      id: STRING,                                                             │
│      corte: STRING,         # "Cassazione", "Corte Costituzionale"          │
│      numero: STRING,                                                         │
│      anno: INTEGER,                                                          │
│      testo: STRING,                                                          │
│      principio: STRING                                                       │
│  })                                                                          │
│                                                                              │
│  EDGE TYPES                                                                  │
│  ══════════                                                                  │
│  (:Norma)-[:CONTIENE]->(:Articolo)                                          │
│  (:Articolo)-[:CONTIENE]->(:Comma)                                          │
│  (:Articolo)-[:RINVIA]->(:Articolo)           # Cross-references            │
│  (:Articolo)-[:DEFINISCE]->(:Concetto)                                      │
│  (:Massima)-[:INTERPRETA]->(:Articolo)                                      │
│  (:Massima)-[:APPLICA]->(:Concetto)                                         │
│                                                                              │
│  # Temporal/Modification edges (MultivigenzaPipeline)                       │
│  (:Articolo)-[:ABROGA {data: DATE}]->(:Articolo)                            │
│  (:Articolo)-[:SOSTITUISCE {data: DATE}]->(:Articolo)                       │
│  (:Articolo)-[:MODIFICA {data: DATE}]->(:Articolo)                          │
│  (:Articolo)-[:INSERISCE {data: DATE}]->(:Articolo)                         │
│  (:Articolo)-[:VERSIONE_PRECEDENTE]->(:Articolo)                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Qdrant Collections

```yaml
collections:
  legal_chunks:
    vectors:
      size: 1536  # OpenAI ada-002 or equivalent
      distance: Cosine
    payload_schema:
      urn: keyword
      chunk_index: integer
      text: text
      article_number: keyword
      norm_type: keyword
      vigenza_date: datetime

  case_law:
    vectors:
      size: 1536
      distance: Cosine
    payload_schema:
      massima_id: keyword
      corte: keyword
      anno: integer
      principio: text
      articoli_riferiti: keyword[]  # URNs
```

### Data Flow: Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  External Sources                                                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                       │
│  │ Normattiva   │  │  Brocardi    │  │  EUR-Lex     │                       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                       │
│         │                 │                 │                                │
│         └─────────────────┼─────────────────┘                                │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │  VisuaLex API   │                                        │
│                  │  (Scraping)     │                                        │
│                  └────────┬────────┘                                        │
│                           │                                                  │
│                           ▼                                                  │
│                  ┌─────────────────┐                                        │
│                  │ MultivigenzaPipe│  ← Handles temporal versions           │
│                  └────────┬────────┘                                        │
│                           │                                                  │
│         ┌─────────────────┼─────────────────┐                               │
│         ▼                 ▼                 ▼                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ FalkorDB    │  │  Chunker    │  │ Concept     │                         │
│  │ Graph Build │  │  + Embed    │  │ Extractor   │                         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                         │
│         │                │                │                                  │
│         ▼                ▼                ▼                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │ FalkorDB   │  │   Qdrant    │  │ FalkorDB   │                          │
│  │ (Norms)    │  │ (Vectors)   │  │ (Concepts) │                          │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Security Architecture

### Security Requirements (from NFR)

| Requirement | Target | Implementation |
|-------------|--------|----------------|
| NFR-SEC-1 | AES-256 encryption at rest | PostgreSQL TDE, Qdrant encryption |
| NFR-SEC-2 | TLS 1.3 in transit | Reverse proxy, internal mTLS |
| NFR-SEC-3 | JWT authentication | Platform issues, all services verify |
| NFR-SEC-4 | PII anonymization | Hash user_id on consent withdrawal |
| NFR-SEC-5 | Dependency scanning | Dependabot, Snyk in CI |

### Authentication & Authorization Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        AUTHENTICATION FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. LOGIN                                                                    │
│     ┌──────────┐                    ┌──────────────┐                        │
│     │  Client  │ ── credentials ──▶ │   Platform   │                        │
│     │          │ ◀── JWT + refresh ─│   /auth/login│                        │
│     └──────────┘                    └──────────────┘                        │
│                                                                              │
│  2. JWT STRUCTURE                                                            │
│     {                                                                        │
│       "sub": "user-uuid",                                                    │
│       "email": "user@example.com",                                          │
│       "role": "user|admin|researcher",                                      │
│       "consents": {                                                          │
│         "ai_analysis": true,                                                 │
│         "audit_trail": true,                                                 │
│         "rlcf_feedback": false,                                             │
│         "research": false                                                    │
│       },                                                                     │
│       "consent_version": "2026-01",                                         │
│       "iat": 1706140800,                                                     │
│       "exp": 1706144400  // 1 hour                                          │
│     }                                                                        │
│                                                                              │
│  3. REQUEST AUTHORIZATION                                                    │
│     ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐        │
│     │  Client  │────▶│ Platform │────▶│ VisuaLex │────▶│  MERL-T  │        │
│     └──────────┘     └──────────┘     └──────────┘     └──────────┘        │
│                           │                                  │              │
│                      Verify JWT                         Verify JWT          │
│                      Check role                         Check consents      │
│                                                                              │
│  4. SERVICE-TO-SERVICE (Internal)                                           │
│     Platform ──[Internal API Key]──▶ MERL-T /internal/*                     │
│     (Webhook calls, consent sync)                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Role-Based Access Control (RBAC)

| Role | Platform | VisuaLex | MERL-T |
|------|----------|----------|--------|
| **anonymous** | Public pages only | Full API access | ❌ Blocked |
| **user** | Dossiers, profile | Full API access | Requires consent |
| **researcher** | + Export, analytics | Full API access | + Research endpoints |
| **admin** | + User management | Full API access | + System config |

### Endpoint Protection Matrix

```yaml
# Platform (Express)
/api/v1/auth/*:          public
/api/v1/users/*:         authenticated + role:admin
/api/v1/dossiers/*:      authenticated + owner
/api/v1/consents/*:      authenticated + owner

# VisuaLex (Quart) - No auth required (public utility)
/api/v1/norme/*:         public
/api/v1/urn/*:           public

# MERL-T (FastAPI)
/api/v1/analyze:         authenticated + consent:ai_analysis
/api/v1/feedback:        authenticated + consent:rlcf_feedback
/api/v1/authority/*:     authenticated
/api/v1/research/*:      authenticated + role:researcher + consent:research
/internal/*:             internal_api_key only
/health:                 public
```

### Data Protection

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PROTECTION LAYERS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  IN TRANSIT                                                                  │
│  ══════════                                                                  │
│  • TLS 1.3 for all external connections                                     │
│  • Internal: Docker network (no encryption) OR mTLS (production)            │
│  • Redis: requirepass + TLS optional                                        │
│                                                                              │
│  AT REST                                                                     │
│  ════════                                                                    │
│  • PostgreSQL: pgcrypto for sensitive columns (not full TDE for MVP)        │
│  • FalkorDB: Filesystem encryption (Docker volume)                          │
│  • Qdrant: Filesystem encryption (Docker volume)                            │
│  • Backups: AES-256 encrypted before upload                                 │
│                                                                              │
│  PII HANDLING                                                                │
│  ════════════                                                                │
│  • Minimal collection: email, hashed password only                          │
│  • Consent withdrawal:                                                       │
│    - audit_trail.user_id → SHA256(user_id + salt)                          │
│    - rlcf_feedback.user_id → SHA256(user_id + salt)                        │
│    - Queries → soft delete or anonymize                                     │
│  • Data export: JSON format via /api/v1/users/{id}/export                   │
│                                                                              │
│  SECRETS MANAGEMENT                                                          │
│  ══════════════════                                                          │
│  • Development: .env files (gitignored)                                     │
│  • Production: Docker secrets or environment variables                      │
│  • Never in code: API keys, DB passwords, JWT secrets                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Security Checklist for Deployment

- [ ] JWT secret rotated (min 256-bit)
- [ ] Database passwords unique per service
- [ ] TLS certificates valid and auto-renewed
- [ ] Rate limiting enabled on public endpoints
- [ ] CORS restricted to known origins
- [ ] SQL injection prevention (parameterized queries)
- [ ] XSS prevention (React default escaping)
- [ ] CSRF tokens for state-changing operations
- [ ] Dependency vulnerabilities scanned (CI/CD)
- [ ] Audit logging enabled and tested

---

## Deployment Architecture

### Environment Strategy

| Environment | Purpose | Infrastructure | Data |
|-------------|---------|----------------|------|
| **local** | Development | Docker Compose | Sample/synthetic |
| **staging** | Integration testing | Docker Compose (VPS) | Anonymized production |
| **production** | Live system | Docker Compose (VPS) | Real data |

### Docker Compose Architecture

```yaml
# docker-compose.yml (simplified)
version: '3.8'

services:
  # ─────────────────────────────────────────────────────
  # PRESENTATION LAYER
  # ─────────────────────────────────────────────────────
  frontend:
    build: ./visualex-platform/frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_URL=http://platform:3001
    depends_on:
      - platform

  # ─────────────────────────────────────────────────────
  # API LAYER
  # ─────────────────────────────────────────────────────
  platform:
    build: ./visualex-platform/backend
    ports:
      - "3001:3001"
    environment:
      - DATABASE_URL=postgresql://...
      - JWT_SECRET=${JWT_SECRET}
      - MERLT_URL=http://merlt:8000
      - VISUALEX_URL=http://visualex:5000
    depends_on:
      - postgres
      - redis

  visualex:
    build: ./visualex-api
    ports:
      - "5000:5000"
    # No database dependencies - stateless scraping

  merlt:
    build: ./merlt
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - FALKORDB_URL=redis://falkordb:6379
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      - LLM_PROVIDER=${LLM_PROVIDER}
      - LLM_API_KEY=${LLM_API_KEY}
    depends_on:
      - postgres
      - falkordb
      - qdrant
      - redis
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]  # Optional: for local LLM

  # ─────────────────────────────────────────────────────
  # DATA LAYER
  # ─────────────────────────────────────────────────────
  postgres:
    image: postgres:16
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=alis
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}

  falkordb:
    image: falkordb/falkordb:latest
    ports:
      - "6379:6379"
    volumes:
      - falkordb_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  falkordb_data:
  qdrant_data:
  redis_data:
```

### Container Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DOCKER COMPOSE DEPLOYMENT                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      REVERSE PROXY (Caddy/Nginx)                     │    │
│  │                           :80, :443                                  │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │    │
│  │  │ /           │  │ /api/v1/    │  │ /merlt/     │                  │    │
│  │  │ → frontend  │  │ → platform  │  │ → merlt     │                  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│      ┌───────────────────────┼───────────────────────┐                      │
│      │                       │                       │                      │
│      ▼                       ▼                       ▼                      │
│  ┌─────────┐           ┌─────────┐            ┌─────────┐                   │
│  │frontend │           │platform │            │  merlt  │                   │
│  │  :3000  │           │  :3001  │            │  :8000  │                   │
│  │ React   │           │ Express │◀──────────▶│ FastAPI │                   │
│  └─────────┘           └────┬────┘            └────┬────┘                   │
│                              │                     │                        │
│                              │    ┌────────────────┤                        │
│                              │    │                │                        │
│                              ▼    ▼                ▼                        │
│  ┌─────────┐           ┌─────────┐          ┌─────────┐  ┌─────────┐       │
│  │visualex │           │postgres │          │falkordb │  │  qdrant │       │
│  │  :5000  │           │  :5432  │          │  :6379  │  │  :6333  │       │
│  │  Quart  │           └─────────┘          └─────────┘  └─────────┘       │
│  └─────────┘                 │                                              │
│       │                      │                                              │
│       │                      ▼                                              │
│       │                ┌─────────┐                                         │
│       └───────────────▶│  redis  │                                         │
│                        │  :6379  │                                         │
│                        └─────────┘                                         │
│                                                                              │
│  NETWORK: alis_network (bridge)                                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Startup Order & Health Checks

```yaml
# Health check configuration
services:
  merlt:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 120s  # Allow warm-up time

  platform:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3001/health"]
      interval: 30s
      timeout: 5s
      retries: 3

  postgres:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### Startup Sequence

```
1. postgres, redis, falkordb, qdrant  (parallel - data layer)
         │
         ▼ health checks pass
2. visualex                            (no dependencies)
         │
         ▼
3. merlt                               (depends on data layer)
         │
         ▼ warm_up.py completes
4. platform                            (depends on merlt for some features)
         │
         ▼
5. frontend                            (depends on platform)
         │
         ▼
6. reverse proxy                       (depends on all services)
```

### Scaling Considerations (Post-MVP)

| Component | Scaling Strategy | Notes |
|-----------|------------------|-------|
| frontend | CDN + replicas | Static assets |
| platform | Horizontal (stateless) | Session in Redis |
| visualex | Horizontal (stateless) | Rate limit per instance |
| merlt | Vertical (GPU) or horizontal (CPU) | LLM is bottleneck |
| postgres | Vertical, read replicas | 7-year audit data |
| falkordb | Vertical | Graph queries need RAM |
| qdrant | Horizontal (sharding) | Built-in clustering |
| redis | Sentinel/Cluster | High availability |

### Backup Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKUP STRATEGY                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  POSTGRESQL (Critical - 7 year retention)                                   │
│  • Daily: pg_dump → encrypted → S3/B2                                       │
│  • Weekly: Full backup with pg_basebackup                                   │
│  • Retention: Daily for 30 days, weekly for 1 year, monthly for 7 years    │
│                                                                              │
│  FALKORDB (Important - can rebuild from sources)                            │
│  • Daily: BGSAVE → copy RDB → S3/B2                                        │
│  • Retention: 7 days                                                        │
│                                                                              │
│  QDRANT (Important - can rebuild from sources)                              │
│  • Daily: Snapshot API → S3/B2                                              │
│  • Retention: 7 days                                                        │
│                                                                              │
│  REDIS (Cache - no backup needed)                                           │
│  • Ephemeral data, warm-up regenerates                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

### Architecture Decision Records

| ADR | Decision | Status |
|-----|----------|--------|
| ADR-001 | Circuit Breaker: Lightweight per-service + observability | ✅ Accepted |
| ADR-002 | GDPR Consent: Platform gateway with JWT claims | ✅ Accepted |
| ADR-003 | API Versioning: URL path (/api/v1/) | ✅ Accepted |
| ADR-004 | Audit Trail: PostgreSQL with time-based partitioning | ✅ Accepted |
| ADR-005 | Caching: Hybrid Redis + startup warm-up | ✅ Accepted |

### Key Architectural Principles

1. **3-Layer Service Separation** - Different risk profiles, deployment flexibility
2. **Parallel Expert Execution** - asyncio for performance, GatingNetwork for aggregation
3. **Consent-First** - JWT claims propagate consent, middleware enforces
4. **Observability** - Structured logging, trace_id propagation, audit trail
5. **Graceful Degradation** - Circuit breakers, partial responses when experts fail

### Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React 19 + Vite 7 + Tailwind v4 | SPA |
| API | Express 5 + Quart + FastAPI | 3-tier services |
| ML | PyTorch + Transformers | Expert pipeline |
| Graph | FalkorDB | Knowledge graph |
| Vector | Qdrant | Semantic search |
| Relational | PostgreSQL 16 | Users, audit, RLCF |
| Cache | Redis 7 | Sessions, warm-up |
| Container | Docker Compose | Deployment |

### Cross-References

| Document | Relationship |
|----------|--------------|
| [PRD](./prd.md) | Requirements implemented by this architecture |
| [Technical Research](./research/technical-vector-space-legal-interpretation-research-2026-01-23.md) | Research validating architectural choices |
| [Project Documentation](../../docs/project-documentation/index.md) | Existing codebase analysis |
| [MERL-T CLAUDE.md](../../merlt/CLAUDE.md) | Framework-specific patterns |

### Implementation Priorities (MVP)

1. **Consent Management** - ADR-002 implementation (GDPR compliance)
2. **Audit Trail** - ADR-004 schema and logging (traceability requirement)
3. **API Versioning** - ADR-003 router setup (breaking change protection)
4. **Caching** - ADR-005 warm-up script (performance NFR)
5. **Circuit Breakers** - ADR-001 resilience layer (stability)

### Open Questions for Future Iterations

1. **LLM Provider Strategy** - Lock-in vs multi-provider abstraction?
2. **Horizontal Scaling** - When to move from vertical to horizontal for MERL-T?
3. **Real-time Features** - WebSocket for RLCF live feedback? (post-MVP)
4. **Mobile** - PWA sufficient or native app needed? (post-MVP)

---

_Architecture Document Complete - Ready for Implementation Planning_

