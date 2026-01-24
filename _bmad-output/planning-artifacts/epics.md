---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics']
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/architecture.md
  - _bmad-output/planning-artifacts/ux-design-specification.md
---

# ALIS_CORE - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for ALIS_CORE, decomposing the requirements from the PRD, UX Design, and Architecture requirements into implementable stories.

---

## Requirements Inventory

### Functional Requirements

**Query & Legal Analysis (FR1-FR6)**

- **FR1:** Legal professional can submit natural language queries about Italian law
- **FR2:** Legal professional can receive structured responses following Art. 12 Preleggi sequence
- **FR3:** Legal professional can view which Expert (Literal, Systemic, Principles, Precedent) contributed each part of a response
- **FR4:** Legal professional can see confidence level for each response
- **FR5:** Legal professional can query about specific norm articles by URN or article number
- **FR6:** Legal professional can query with temporal context ("as of date X")

**Traceability & Source Verification (FR7-FR11)**

- **FR7:** User can view complete reasoning trace for any response
- **FR8:** User can navigate from any statement to its source URN
- **FR9:** User can export reasoning trace in citation-ready format
- **FR10:** User can see which sources each Expert consulted
- **FR11:** User can verify temporal validity of cited norms and precedents

**Knowledge Graph & Norm Browsing (FR12-FR18)**

- **FR12:** User can browse norms with hierarchical navigation
- **FR13:** User can search norms by keyword, article number, or URN
- **FR14:** User can view norm annotations and cross-references
- **FR15:** User can see historical versions of norms with modification dates
- **FR16:** User can query norms as they existed at a specific date
- **FR17:** Admin can trigger manual ingest of new/modified norms
- **FR18:** System can detect when cited norms have been modified

**RLCF Feedback & Learning (FR19-FR26)**

- **FR19:** RLCF participant can rate response quality (numeric scale)
- **FR20:** RLCF participant can provide textual feedback on responses
- **FR21:** RLCF participant can flag specific errors in responses
- **FR22:** User can see their feedback contribution history
- **FR23:** High-authority user can see aggregated impact of their feedback
- **FR24:** Researcher can view policy weight evolution over time
- **FR25:** Researcher can export anonymized feedback analytics
- **FR26:** System can weight feedback by authority score

**User & Authority Management (FR27-FR31)**

- **FR27:** New member can onboard with invitation from existing member
- **FR28:** User can view their authority score
- **FR29:** User can understand how authority score is calculated
- **FR30:** Admin can view and adjust authority parameters
- **FR31:** User can configure privacy/consent preferences

**Consent & Data Privacy (FR32-FR36)**

- **FR32:** User can choose opt-in level (Basic/Learning/Research)
- **FR33:** User can revoke consent and request data deletion
- **FR34:** User can export personal data (GDPR Art. 20)
- **FR35:** System can anonymize queries before RLCF storage
- **FR36:** System can maintain immutable audit trail

**System Administration (FR37-FR42)**

- **FR37:** Admin can monitor knowledge graph coverage and freshness
- **FR38:** Admin can configure scraping schedules for norm sources
- **FR39:** Admin can view system health and Expert pipeline status
- **FR40:** Admin can manage circuit breakers for Expert resilience
- **FR41:** Admin can run regression tests against gold standard queries
- **FR42:** Admin can flag/quarantine problematic feedback

**API & Integration (FR43-FR46)**

- **FR43:** Developer can access MERL-T analysis via REST API
- **FR44:** Developer can receive structured JSON with reasoning trace
- **FR45:** Developer can authenticate with API credentials
- **FR46:** Developer can access API documentation

**Academic & Research Support (FR47-FR50)**

- **FR47:** Researcher can access RLCF dashboard with policy analytics
- **FR48:** Researcher can reproduce historical queries with same model version
- **FR49:** Researcher can export datasets for academic validation
- **FR50:** Researcher can compare RLCF vs baseline performance metrics

**RLCF Feedback-Specific (FR-F1 to FR-F13)**

- **FR-F1:** System can collect NER confirmation/correction feedback inline
- **FR-F2:** System can collect Router decision feedback from high-authority users
- **FR-F3-F6:** System can collect Expert output feedback with 4-level granularity
- **FR-F7:** System can collect Synthesizer feedback including usability assessment
- **FR-F8:** System can collect Bridge quality feedback (source relevance rating)
- **FR-F8a:** System can infer F8 implicitly from F7↔F3-F6 correlation
- **FR-F8b:** System can display "Fonti usate" panel to 🎓 Contributore
- **FR-F8c:** System can update expert_affinity weights based on F8 feedback
- **FR-F8d:** System can train TraversalPolicy via PolicyGradientTrainer
- **FR-F9:** System can weight all feedback by user authority score
- **FR-F10:** System can aggregate feedback per component for training
- **FR-F11:** System can trigger training pipeline when buffer threshold reached
- **FR-F12:** System can display Devil's Advocate for high-consensus responses
- **FR-F13:** System can collect Devil's Advocate evaluation feedback

---

### Non-Functional Requirements

**Performance (NFR-P1 to NFR-P7)**

- **NFR-P1:** Norm base data display <500ms
- **NFR-P2:** Expert enrichment (first visit) <3 min
- **NFR-P3:** Expert enrichment (cached) <500ms
- **NFR-P4:** Knowledge graph query <200ms
- **NFR-P5:** Concurrent users: 20 simultaneous
- **NFR-P6:** Feedback submission <1s
- **NFR-P7:** Cache hit rate >80% after warm start

**Security (NFR-S1 to NFR-S7)**

- **NFR-S1:** Data encryption at rest AES-256
- **NFR-S2:** Data encryption in transit TLS 1.3
- **NFR-S3:** Authentication JWT with rotation
- **NFR-S4:** API authentication: API key + rate limiting
- **NFR-S5:** PII anonymization before RLCF storage
- **NFR-S6:** Audit log integrity: append-only, tamper-evident
- **NFR-S7:** Consent verification on every learning-related action

**Reliability (NFR-R1 to NFR-R6)**

- **NFR-R1:** System availability 99% uptime (excl. maintenance)
- **NFR-R2:** Data backup frequency: daily, 30-day retention
- **NFR-R3:** Expert pipeline degradation: graceful (skip low-confidence Expert)
- **NFR-R4:** LLM provider failover: automatic to backup provider
- **NFR-R5:** Audit trail retention: 7 years, immutable
- **NFR-R6:** Historical query reproducibility: exact response recreation

**Scalability (NFR-SC1 to NFR-SC4)**

- **NFR-SC1:** User capacity MVP: 50 registered, 20 concurrent
- **NFR-SC2:** User capacity Growth: 500 registered, 100 concurrent
- **NFR-SC3:** Knowledge graph size: 10k norms without performance degradation
- **NFR-SC4:** Feedback volume: 1000 entries/month processable

**Integration (NFR-I1 to NFR-I4)**

- **NFR-I1:** API versioning: semantic versioning, backward compatibility
- **NFR-I2:** LLM provider abstraction: switchable without code changes
- **NFR-I3:** Normattiva scraping resilience: retry with exponential backoff
- **NFR-I4:** Export formats: JSON + CSV for datasets

**Maintainability (NFR-M1 to NFR-M5)**

- **NFR-M1:** Deployment reproducibility: Docker Compose, single command
- **NFR-M2:** Configuration externalization: all settings in YAML/env
- **NFR-M3:** Logging: structured JSON, queryable
- **NFR-M4:** Documentation: API docs auto-generated
- **NFR-M5:** Test coverage: >80% for core pipeline

**Compliance (NFR-C1 to NFR-C4)**

- **NFR-C1:** GDPR Art. 6(1)(a): Explicit consent for learning
- **NFR-C2:** GDPR Art. 17: Right to erasure implemented
- **NFR-C3:** GDPR Art. 20: Data portability in 30 days
- **NFR-C4:** GDPR Art. 89: Research exemption documented

---

### Additional Requirements

**From Architecture (Brownfield Context)**

- Existing codebase migration from `Legacy/MERL-T_alpha/` - no starter template needed
- 3-Layer service architecture: Platform (Express) → VisuaLex-API (Quart) → MERL-T (FastAPI)
- 4-Database setup: PostgreSQL + FalkorDB + Qdrant + Redis
- GPU separation for LLM/embedding inference
- Fork-friendly architecture for other organizations
- Docker Compose deployment for thesis demo

**From Architecture (ADRs)**

- ADR-001: Circuit Breaker Strategy - Expert pipeline resilience
- ADR-002: GDPR Consent Management - Middleware-based consent verification
- ADR-003: API Versioning Strategy - /v1/ prefix, backward compatibility
- ADR-004: Audit Trail (7-Year Retention) - Append-only immutable log
- ADR-005: Warm-Start Caching Strategy - Pre-compute top 100 norms

**From UX Design**

- IDE per Giuristi paradigm: Command Palette, Peek Definition, Split View, Problems Panel
- 4-Profile System: ⚡ Consultazione | 📖 Ricerca | 🔍 Analisi | 🎓 Contributore
- Progressive Loading Pattern (T+0ms → T+200ms → T+500ms → T+3min)
- NER Citation Highlighting with linking automatico
- Expert Accordion UI for reasoning trace display
- Devil's Advocate panel (collapsed by default, opt-in for 🔍/🎓)
- Keyboard-first design with shortcuts (Ctrl+Shift+P, F12, etc.)
- Responsive design for desktop-first (lawyers at desk)
- Accessibility: WCAG 2.1 AA compliance target

**From Test Design (Testability)**

- LLM response mocking for deterministic tests
- Testcontainers for isolated DB testing
- Synthetic feedback generator for RLCF validation
- 80+ legacy tests need migration from `Legacy/MERL-T_alpha/tests/`

---

### FR Coverage Map

| FR | Epic | Description |
|----|------|-------------|
| FR1 | Epic 4 | Submit NL queries |
| FR2 | Epic 4 | Art. 12 sequence response |
| FR3 | Epic 4 | View Expert contributions |
| FR4 | Epic 4 | See confidence level |
| FR5 | Epic 3 | Query by URN/article |
| FR6 | Epic 4 | Temporal context queries |
| FR7 | Epic 5 | View reasoning trace |
| FR8 | Epic 5 | Navigate to source URN |
| FR9 | Epic 5 | Export citation-ready |
| FR10 | Epic 4+5 | See Expert sources |
| FR11 | Epic 5 | Verify temporal validity |
| FR12 | Epic 3 | Browse norms hierarchically |
| FR13 | Epic 3 | Search by keyword/URN |
| FR14 | Epic 3 | View annotations/cross-refs |
| FR15 | Epic 3 | Historical versions |
| FR16 | Epic 3 | Query as-of-date |
| FR17 | Epic 2b | Manual ingest trigger |
| FR18 | Epic 3 | Detect norm modifications |
| FR19 | Epic 6 | Rate response quality |
| FR20 | Epic 6 | Provide textual feedback |
| FR21 | Epic 6 | Flag specific errors |
| FR22 | Epic 6 | View feedback history |
| FR23 | Epic 7 | See authority impact |
| FR24 | Epic 8 | View policy evolution |
| FR25 | Epic 8 | Export anonymized analytics |
| FR26 | Epic 7 | Weight feedback by authority |
| FR27 | Epic 1 | Onboard with invitation |
| FR28 | Epic 1 | View authority score |
| FR29 | Epic 1 | Understand authority calc |
| FR30 | Epic 7 | Admin authority params |
| FR31 | Epic 1 | Configure preferences |
| FR32 | Epic 1 | Choose opt-in level |
| FR33 | Epic 1 | Revoke consent/erasure |
| FR34 | Epic 1 | Export personal data |
| FR35 | Epic 6 | Anonymize before storage |
| FR36 | Epic 6 | Maintain audit trail |
| FR37 | Epic 3 | Monitor KG coverage |
| FR38 | Epic 2a | Configure scraping |
| FR39 | Epic 4 | View Expert pipeline status |
| FR40 | Epic 4 | Manage circuit breakers |
| FR41 | Epic 4 | Run gold standard regression |
| FR42 | Epic 6 | Flag problematic feedback |
| FR43 | Epic 10 | REST API access |
| FR44 | Epic 10 | JSON with trace |
| FR45 | Epic 10 | API authentication |
| FR46 | Epic 10 | API documentation |
| FR47 | Epic 8 | RLCF dashboard |
| FR48 | Epic 8 | Reproduce historical queries |
| FR49 | Epic 8 | Export datasets |
| FR50 | Epic 8 | Compare RLCF vs baseline |
| FR-F1 | Epic 6 | NER feedback |
| FR-F2 | Epic 7 | Router feedback |
| FR-F3-F6 | Epic 6 | Expert output feedback |
| FR-F7 | Epic 6 | Synthesizer feedback |
| FR-F8 | Epic 6 | Bridge quality feedback |
| FR-F8a | Epic 7 | Implicit F8 inference |
| FR-F8b | Epic 6 | Fonti usate panel |
| FR-F8c | Epic 7 | Update expert_affinity |
| FR-F8d | Epic 7 | Train TraversalPolicy |
| FR-F9 | Epic 7 | Weight by authority |
| FR-F10 | Epic 7 | Aggregate per component |
| FR-F11 | Epic 7 | Trigger training pipeline |
| FR-F12 | Epic 8 | Devil's Advocate display |
| FR-F13 | Epic 8 | Devil's Advocate feedback |

---

## Epic List

### Epic 1: Foundation & User Identity
**Goal:** Users can register, authenticate, configure profile (4-Profile System), and manage privacy/consent preferences.

**User Outcome:** Il giurista può accedere alla piattaforma con le proprie credenziali, selezionare il proprio profilo (⚡📖🔍🎓), e configurare i livelli di consent GDPR.

**FRs covered:** FR27, FR28, FR29, FR31, FR32, FR33, FR34
**NFRs addressed:** NFR-S3 (JWT), NFR-C1/C2/C3 (GDPR)
**Stories estimate:** ~6

---

### Epic 2a: Scraping & URN Pipeline
**Goal:** System can scrape Normattiva and generate canonical URNs for Italian legal norms.

**User Outcome:** Il sistema può acquisire norme da Normattiva (scope: Libro IV C.C. - Obbligazioni e Contratti, ~800 articoli) e generare URN canonici.

**FRs covered:** FR38 (scraping config)
**NFRs addressed:** NFR-I3 (scraping resilience)
**Stories estimate:** ~3
**Scope:** Libro IV Codice Civile only (Obbligazioni + Contratti)

---

### Epic 2b: Graph Building
**Goal:** System can build and populate the Knowledge Graph with legal norms, relations, and temporal versioning.

**User Outcome:** Il Knowledge Graph FalkorDB contiene nodi (Norma, Articolo, Comma) e relazioni (RIFERIMENTO, MODIFICA, CITATO_DA) per il Libro IV C.C.

**FRs covered:** FR17 (manual ingest trigger)
**NFRs addressed:** NFR-SC3 (10k norms capacity)
**Stories estimate:** ~5
**Includes:** Manual ingest trigger (moved from Epic 9)

---

### Epic 2c: Vector & Bridge Table
**Goal:** System can generate embeddings and maintain chunk-to-graph mappings via Bridge Table.

**User Outcome:** Qdrant contiene embeddings per tutti i chunks del Libro IV, con expert_affinity inizializzata. Bridge Table mappa chunk_id ↔ graph_node_urn.

**FRs covered:** (infrastructure for FR-F8)
**NFRs addressed:** ADR-005 (warm-start caching)
**Stories estimate:** ~4

---

### Epic 3: Norm Browsing & Search
**Goal:** Users can browse, search, and explore legal norms with temporal versioning and KG visualization.

**User Outcome:** Il giurista può navigare il Libro IV C.C., cercare per keyword/URN, vedere versioni storiche, cross-references, e un dashboard semplice della copertura KG.

**FRs covered:** FR5, FR12, FR13, FR14, FR15, FR16, FR18, FR37
**NFRs addressed:** NFR-P1 (<500ms), NFR-P4 (<200ms KG query)
**Stories estimate:** ~8
**Includes:** KG coverage dashboard (moved from Epic 9)

---

### Epic 4: MERL-T Analysis Pipeline
**Goal:** Users can submit queries and receive AI-powered legal analysis following Art. 12 Preleggi sequence.

**User Outcome:** Il giurista può fare query in linguaggio naturale e ricevere analisi strutturata dai 4 Expert (Literal → Systemic → Principles → Precedent) con confidence, circuit breaker per resilienza, e regression test su gold standard queries.

**FRs covered:** FR1, FR2, FR3, FR4, FR6, FR10, FR39, FR40, FR41
**NFRs addressed:** NFR-P2 (<3min first), NFR-P3 (<500ms cached), NFR-R3 (graceful degradation), NFR-I2 (LLM abstraction)
**Stories estimate:** ~12
**Includes:** Circuit breaker config, gold standard regression (moved from Epic 9)

---

### Epic 5: Traceability & Source Verification
**Goal:** Users can trace every statement to its source and export reasoning for legal briefs.

**User Outcome:** Il giurista può vedere la traccia completa di ragionamento, navigare da ogni affermazione alla fonte URN, verificare validità temporale, ed esportare in formato citazione.

**FRs covered:** FR7, FR8, FR9, FR10, FR11
**NFRs addressed:** Success Criteria (100% traceability)
**Stories estimate:** ~5

---

### Epic 6: RLCF Feedback Collection
**Goal:** Users can provide feedback on AI responses across all 8 feedback points (F1-F8).

**User Outcome:** Il giurista può valutare risposte (rating + commenti), segnalare errori, vedere storico contributi. Sistema raccoglie feedback F1-F8 con UI adattiva per profilo. Include synthetic feedback generator per bootstrap.

**FRs covered:** FR19, FR20, FR21, FR22, FR35, FR36, FR42, FR-F1, FR-F3-F7, FR-F8, FR-F8b
**NFRs addressed:** NFR-P6 (<1s), NFR-S5 (PII anonymization), NFR-R5 (audit trail)
**Stories estimate:** ~10
**Includes:** Synthetic feedback generator, audit trail

---

### Epic 7: Authority & Learning Loop
**Goal:** System learns from community feedback with authority weighting (RLCF training loop).

**User Outcome:** Il sistema pesa feedback per authority score, aggrega per training, aggiorna policy weights. High-authority users vedono impatto contributi. TraversalPolicy impara "path virtuosi" da F8.

**FRs covered:** FR23, FR26, FR30, FR-F2, FR-F8a, FR-F8c, FR-F8d, FR-F9, FR-F10, FR-F11
**NFRs addressed:** NFR-R6 (reproducibility), NFR-SC4 (1k feedback/month)
**Stories estimate:** ~6

---

### Epic 8: Research & Academic Support (Minimale)
**Goal:** Researchers can validate RLCF framework with reproducible data and basic analytics.

**User Outcome:** Il ricercatore può visualizzare evoluzione policy weights, esportare dataset anonimizzati (CSV/JSON), riprodurre query storiche, vedere Devil's Advocate per high-consensus.

**FRs covered:** FR24, FR25, FR47, FR48, FR49, FR50, FR-F12, FR-F13
**NFRs addressed:** NFR-M5 (80% coverage), NFR-I4 (export)
**Stories estimate:** ~4
**Scope:** Minimale - export + reproducibility + basic dashboard

---

### Epic 9: Admin & Monitoring (Post-Thesis)
**Goal:** Admins can monitor, maintain, and ensure system health at scale.

**Status:** POST-THESIS - Manual admin operations acceptable for ~20 users during thesis demo.

**FRs covered:** (remaining admin FRs)
**Stories estimate:** TBD

---

### Epic 10: API & External Integration (Post-Thesis)
**Goal:** Developers can integrate ALIS capabilities into their applications via REST API.

**Status:** POST-THESIS - UI demo sufficient for thesis defense.

**FRs covered:** FR43, FR44, FR45, FR46
**Stories estimate:** TBD

---

## Implementation Timeline

| Sprint | Weeks | Epic(s) | Deliverable |
|--------|-------|---------|-------------|
| 1-2 | Jan 27 - Feb 9 | Epic 1 + 2a | Auth + Scraper Libro IV |
| 3-4 | Feb 10 - Feb 23 | Epic 2b + 2c | KG populated + embeddings |
| 5-6 | Feb 24 - Mar 9 | Epic 3 + 4 (start) | Browsing UI + Expert pipeline base |
| 7-8 | Mar 10 - Mar 23 | Epic 4 (complete) | 4 Experts + circuit breaker + gold standard |
| 9-10 | Mar 24 - Apr 6 | Epic 5 + 6 | Traceability + Feedback collection |
| 11-12 | Apr 7 - Apr 20 | Epic 7 + 8 | Learning loop + Research dashboard |
| 13-14 | Apr 21 - May 4 | **Buffer** | Bug fix, polish, documentation |
| 15-16 | May 5 - May 18 | **Thesis prep** | Demo prep, slides, rehearsal |

**Thesis Defense:** Late May 2026

---

## Summary

- **Total MVP Stories:** ~63 stories
- **Sprint Duration:** 12 sprints (+ 2 buffer + 2 thesis prep)
- **Scope:** Libro IV Codice Civile (Obbligazioni + Contratti, ~800 articoli)
- **Post-Thesis:** Epic 9 (Admin) + Epic 10 (API)
