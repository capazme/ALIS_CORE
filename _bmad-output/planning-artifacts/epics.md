---
stepsCompleted: ['step-01-validate-prerequisites', 'step-02-design-epics', 'step-03-create-stories']
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

## Epic 1: Foundation & User Identity

**Goal:** Users can register, authenticate, configure profile (4-Profile System), and manage privacy/consent preferences.

### Story 1.1: User Registration

As a **new member**,
I want to **register with my email and password after receiving an invitation**,
So that **I can access the ALIS platform as an association member**.

**Acceptance Criteria:**

**Given** I have received an invitation link from an existing member
**When** I click the invitation link and enter my email, password, and basic info (name, role)
**Then** my account is created with "pending" status
**And** I receive a verification email
**And** my password is hashed with bcrypt before storage

**Given** I try to register without a valid invitation
**When** I submit the registration form
**Then** I see an error "Registration requires an invitation from an existing member"

**Given** I try to register with an email already in use
**When** I submit the registration form
**Then** I see an error "This email is already registered"

**Technical Notes:**
- Creates: `users` table (id, email, password_hash, name, role, status, created_at)
- Creates: `invitations` table (id, inviter_id, email, token, expires_at, used_at)
- JWT secret configured via environment variable

---

### Story 1.2: User Login

As a **registered user**,
I want to **login with my email and password**,
So that **I can access my personalized ALIS experience**.

**Acceptance Criteria:**

**Given** I am a verified user with valid credentials
**When** I enter my email and password and submit
**Then** I receive a JWT access token (1h expiry) and refresh token (7d expiry)
**And** I am redirected to the main dashboard

**Given** I enter incorrect credentials
**When** I submit the login form
**Then** I see an error "Invalid email or password" (no credential leak)
**And** the failed attempt is logged for security

**Given** my JWT access token has expired
**When** I make an API request
**Then** the system uses my refresh token to issue a new access token
**And** if refresh token is also expired, I am redirected to login

**Technical Notes:**
- JWT with rotation (NFR-S3)
- Rate limiting: 5 failed attempts → 15min lockout

---

### Story 1.3: Profile Setup

As a **logged-in user**,
I want to **select my usage profile and configure basic preferences**,
So that **ALIS adapts its interface and feedback options to my expertise level**.

**Acceptance Criteria:**

**Given** I am logged in for the first time (or access profile settings)
**When** I view the profile selector
**Then** I see 4 profile options with descriptions:
  - ⚡ Consultazione Rapida: "Risposte veloci, minima interazione"
  - 📖 Ricerca Assistita: "Esplorazione guidata con suggerimenti"
  - 🔍 Analisi Esperta: "Accesso completo a Expert trace e feedback"
  - 🎓 Contributore Attivo: "Feedback granulare, impatto sul training"

**Given** I select a profile
**When** I confirm my selection
**Then** my profile is saved to my user record
**And** the UI adapts to show/hide features based on profile
**And** I can change profile anytime from settings

**Given** I am a new user without expertise credentials
**When** I try to select 🎓 Contributore Attivo
**Then** I see a message explaining this profile requires authority score ≥ 0.5
**And** I am offered 🔍 Analisi Esperta as alternative

**Technical Notes:**
- Adds: `profile_type` column to users table
- Adds: `user_preferences` table (user_id, theme, language, notifications)

---

### Story 1.4: Consent Configuration

As a **user**,
I want to **choose my data consent level for RLCF participation**,
So that **I control how my interactions contribute to system learning (GDPR Art. 6)**.

**Acceptance Criteria:**

**Given** I am in settings or first-time setup
**When** I view consent options
**Then** I see 3 levels clearly explained:
  - **Basic**: "No data collected beyond session. System use only."
  - **Learning**: "Anonymized queries + feedback used for RLCF training."
  - **Research**: "Aggregated data available for academic analysis."

**Given** I select a consent level
**When** I confirm my choice
**Then** my consent is recorded with timestamp
**And** a consent audit log entry is created (immutable)
**And** the system respects my choice for all future interactions

**Given** I change my consent level to a lower level
**When** I confirm the change
**Then** my new preference is applied immediately
**And** I am informed that previously collected data (if any) remains until I request erasure

**Technical Notes:**
- Creates: `user_consents` table (id, user_id, consent_level, granted_at, ip_hash)
- Creates: `consent_audit_log` table (append-only, 7-year retention per NFR-R5)
- Middleware checks consent before any RLCF-related storage

---

### Story 1.5: Authority Score Display

As a **user**,
I want to **view my authority score and understand how it's calculated**,
So that **I know my influence on RLCF training and can work to increase it**.

**Acceptance Criteria:**

**Given** I am logged in and view my profile
**When** I access the authority score section
**Then** I see my current authority score (0.0 - 1.0 scale)
**And** I see a breakdown of the score components:
  - Baseline credentials (α=0.3): qualifications, years of experience
  - Track record (β=0.5): historical feedback accuracy
  - Recent performance (γ=0.2): last N feedback quality

**Given** I am a new user with no feedback history
**When** I view my authority score
**Then** I see my baseline score based on credentials
**And** I see a message "Contribute feedback to increase your authority"

**Given** I hover over or click on a score component
**When** the tooltip/modal appears
**Then** I see a plain-language explanation of that component
**And** I understand how to improve it

**Technical Notes:**
- Creates: `user_authority` table (user_id, baseline_score, track_record, recent_performance, computed_score, updated_at)
- Authority formula: A_u(t) = 0.3·B_u + 0.5·T_u(t) + 0.2·P_u(t)
- Initial baseline from registration data (role, experience)

---

### Story 1.6: Data Export & Erasure

As a **user**,
I want to **export my personal data and request account deletion**,
So that **I can exercise my GDPR rights (Art. 20 portability, Art. 17 erasure)**.

**Acceptance Criteria:**

**Given** I am logged in and access privacy settings
**When** I request data export
**Then** the system generates a JSON + human-readable summary containing:
  - My profile information
  - My query history (if consent was Learning/Research)
  - My feedback submissions
  - My authority score history
  - My consent change history
**And** the export is available for download within 24 hours
**And** I receive email notification when ready

**Given** I request account deletion
**When** I confirm with my password
**Then** I see a warning about irreversibility
**And** my account enters "deletion_pending" status (30-day grace period)
**And** I receive confirmation email with cancellation link

**Given** my account has been in "deletion_pending" for 30 days
**When** the grace period expires
**Then** my personal data is permanently deleted
**And** my anonymized feedback contributions remain (no PII link)
**And** audit log retains anonymized record per legal requirements

**Technical Notes:**
- Export job runs async, stores in secure temporary location
- Soft delete with 30-day grace period
- Anonymization preserves feedback value while removing PII

---

## Epic 2a: Scraping & URN Pipeline

**Goal:** Integrate existing scrapers to acquire Italian legal norms from multiple sources.

**Scope:** Libro IV Codice Civile (Obbligazioni + Contratti) + related Brocardi commentary + relevant EUR-Lex

### Story 2a.1: Normattiva Scraper Integration

As a **system administrator**,
I want to **configure the existing Normattiva scraper for Libro IV C.C.**,
So that **the system can acquire the ~800 articles needed for the thesis demo**.

**Acceptance Criteria:**

**Given** the existing Normattiva scraper in visualex-api
**When** I configure it with scope "Codice Civile, Libro IV"
**Then** it scrapes all articles from Titolo I (Obbligazioni) to Titolo III (Contratti)
**And** each article includes: text, rubrica, URN, last_modified date
**And** modifications/abrogations are detected and stored

**Given** the scraper encounters a transient error (network, rate limit)
**When** the error occurs
**Then** it retries with exponential backoff (max 3 retries)
**And** failed articles are logged for manual review

**Given** I need to re-scrape after norm modifications
**When** I trigger an incremental update
**Then** only modified articles are re-fetched (based on last_modified)

**Existing Code:**
- `Legacy/VisuaLexAPI/api/scraping/normattiva.py` - Core scraper logic
- `Legacy/VisuaLexAPI/api/utils/urn_parser.py` - URN parsing utilities
- Migration: Verify compatibility with current `visualex-api/` Quart structure

**Technical Notes:**
- Output format: JSON with URN, text, metadata, vigenza dates
- Storage: Temporary staging area before graph ingestion (Epic 2b)

---

### Story 2a.2: Brocardi Scraper Integration

As a **system administrator**,
I want to **integrate the existing Brocardi scraper for commentary and massime**,
So that **the system enriches norms with doctrinal commentary and case law references**.

**Acceptance Criteria:**

**Given** the existing Brocardi scraper
**When** I run it for articles in Libro IV C.C.
**Then** it retrieves:
  - Doctrinal commentary (spiegazione articolo)
  - Massime (legal maxims from jurisprudence)
  - Related case citations (Cassazione references)
**And** each item is linked to its source article URN

**Given** an article has no Brocardi entry
**When** the scraper processes it
**Then** it gracefully skips with a log entry
**And** the article is still usable without commentary

**Given** Brocardi has multiple versions/updates of commentary
**When** scraping
**Then** the latest version is captured with retrieval timestamp

**Existing Code:**
- `Legacy/VisuaLexAPI/api/scraping/brocardi.py` - BeautifulSoup scraper
- `Legacy/VisuaLexAPI/api/models/brocardi_models.py` - Data models
- Migration: Adapt output format to match Epic 2b input requirements

**Technical Notes:**
- Brocardi provides jurisprudence links → input for PrecedentExpert
- Commentary → input for doctrinal analysis

---

### Story 2a.3: EUR-Lex Scraper Integration

As a **system administrator**,
I want to **integrate the EUR-Lex scraper for EU directives related to contract law**,
So that **SystemicExpert can reference EU law when interpreting Italian implementation**.

**Acceptance Criteria:**

**Given** the existing EUR-Lex scraper
**When** I configure it for directives related to Libro IV topics
**Then** it retrieves key EU directives (e.g., Consumer Rights Directive 2011/83/EU)
**And** each directive includes: CELEX number, title, text, Italian implementation refs

**Given** an Italian norm implements an EU directive
**When** both are in the system
**Then** a "ATTUA" relationship can be created in Epic 2b

**Given** EUR-Lex API rate limits are hit
**When** the error occurs
**Then** the scraper backs off and retries
**And** partial results are saved (don't lose work)

**Existing Code:**
- `Legacy/VisuaLexAPI/api/scraping/eurlex.py` - API/scraping hybrid
- Review: Check if API access is still valid or needs refresh
- Migration: Minimal - mostly config for scope

**Technical Notes:**
- Scope: ~10-20 key directives for contract/consumer law
- Lower priority than Normattiva/Brocardi for thesis MVP

---

### Story 2a.4: URN Canonicalization Pipeline

As a **system**,
I want to **generate and validate canonical URNs for all scraped content**,
So that **every piece of legal content has a unique, standards-compliant identifier**.

**Acceptance Criteria:**

**Given** raw scraped content from any source
**When** it passes through the URN pipeline
**Then** a canonical URN is generated following NIR (Norme In Rete) standard
**And** the URN format is: `urn:nir:stato:legge:YYYY-MM-DD;N:art.X`

**Given** content with an existing URN
**When** processed by the pipeline
**Then** the existing URN is validated against NIR standard
**And** malformed URNs are flagged for manual review

**Given** content from different sources referring to the same norm
**When** URNs are generated
**Then** they resolve to the same canonical form
**And** duplicates are detected and merged

**Existing Code:**
- `Legacy/VisuaLexAPI/api/utils/urn_parser.py` - URN parsing/generation
- `Legacy/VisuaLexAPI/api/utils/citationParser.ts` - Frontend citation parsing
- `Legacy/MERL-T_alpha/merlt/ner/` - NER for citation extraction
- Migration: Consolidate into single authoritative URN service

**Technical Notes:**
- URN is the primary key linking all systems (Graph, Vector, Bridge)
- Must handle edge cases: commi, lettere, allegati

---

## Epic 2b: Graph Building

**Goal:** Build and populate the Knowledge Graph with legal norms, relations, and temporal versioning.

### Story 2b.1: Graph Schema Setup

As a **system**,
I want to **define the FalkorDB schema with all node and edge types**,
So that **the Knowledge Graph has a consistent, queryable structure**.

**Acceptance Criteria:**

**Given** the FalkorDB instance is running
**When** I initialize the graph schema
**Then** the following node types are created:
  - `Norma` (urn, tipo, data_emanazione, titolo)
  - `Articolo` (urn, numero, rubrica, testo_vigente)
  - `Comma` (urn, numero, testo)
  - `Sentenza` (urn, corte, data, massima)
  - `Concetto` (id, nome, definizione)
  - `Definizione` (id, termine, significato, fonte_urn)

**Given** the schema is initialized
**When** I query node types
**Then** all types have appropriate indexes on URN fields
**And** full-text indexes exist on testo fields for search

**Given** edge types are needed
**When** I define relationships
**Then** the following edge types are available:
  - `RIFERIMENTO` (source_urn → target_urn, tipo)
  - `MODIFICA` (norma_modificante → norma_modificata, data_effetto)
  - `CITATO_DA` (norma → sentenza)
  - `DEFINISCE` (articolo → concetto)
  - `PRINCIPIO` (articolo → principio_giuridico)
  - `ATTUA` (norma_italiana → direttiva_eu)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/graph/schema.py` - Existing schema definitions
- `Legacy/MERL-T_alpha/merlt/storage/graph/falkordb_client.py` - Client wrapper
- `docs/project-documentation/01-architecture.md` - Node/edge documentation
- Migration: Validate schema against current FalkorDB version

**Technical Notes:**
- FalkorDB uses Cypher query language
- Indexes critical for NFR-P4 (<200ms query)

---

### Story 2b.2: Norm Node Ingestion

As a **system**,
I want to **ingest scraped norms into the Knowledge Graph as nodes**,
So that **the legal corpus is navigable and queryable**.

**Acceptance Criteria:**

**Given** scraped content from Epic 2a (JSON format)
**When** the ingestion pipeline processes it
**Then** nodes are created for each Norma, Articolo, and Comma
**And** hierarchical relationships are established (Norma → Articolo → Comma)
**And** each node has its canonical URN as primary identifier

**Given** an article already exists in the graph (re-ingestion)
**When** new content is ingested
**Then** the existing node is updated (not duplicated)
**And** a modification timestamp is recorded
**And** previous version is preserved for temporal queries

**Given** a batch of ~800 articles from Libro IV
**When** bulk ingestion runs
**Then** all nodes are created within acceptable time (<5 min for full batch)
**And** progress is logged for monitoring
**And** failures are isolated (one bad article doesn't stop batch)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/graph/node_builder.py` - Node creation logic
- `Legacy/MERL-T_alpha/merlt/pipeline/ingestion.py` - Ingestion pipeline
- Migration: Adapt to new schema, add batch processing improvements

**Technical Notes:**
- Use MERGE (upsert) for idempotent ingestion
- Batch size: ~100 nodes per transaction for performance

---

### Story 2b.3: Relation Extraction & Creation

As a **system**,
I want to **extract and create relationships between legal norms**,
So that **SystemicExpert can traverse connections for interpretation**.

**Acceptance Criteria:**

**Given** article text containing citations (e.g., "ai sensi dell'art. 1453")
**When** the NER pipeline processes it
**Then** citations are extracted with source and target URNs
**And** `RIFERIMENTO` edges are created in the graph

**Given** a norm that modifies another (e.g., "L'art. X è sostituito da...")
**When** modification is detected
**Then** `MODIFICA` edge is created with data_effetto
**And** `MODIFICATO_DA` reverse edge is also created

**Given** Brocardi content with Cassazione references
**When** jurisprudence links are processed
**Then** `CITATO_DA` edges connect norms to sentenze
**And** edges include citation context (massima excerpt)

**Given** relationships are created
**When** I query neighbors of an article
**Then** I can traverse all relationship types
**And** edge metadata (tipo, data, context) is accessible

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/ner/citation_extractor.py` - Citation NER
- `Legacy/MERL-T_alpha/merlt/ner/relation_classifier.py` - Relation type detection
- `Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts` - Frontend parsing (reference)
- Migration: Integrate NER output with graph edge creation

**Technical Notes:**
- Relation types map to Expert affinities (F8 Bridge Table)
- Critical for SystemicExpert traversal quality

---

### Story 2b.4: Temporal Versioning

As a **user**,
I want to **query norms as they existed at a specific date**,
So that **I can analyze contracts based on the law in force at signing time**.

**Acceptance Criteria:**

**Given** an article with multiple historical versions
**When** I query with `as_of_date` parameter
**Then** I receive the version that was in force on that date
**And** the response indicates vigenza_dal and vigenza_al dates

**Given** an article was modified on 2024-03-01
**When** I query as_of_date=2024-02-15
**Then** I get the pre-modification version
**And** I can see that a newer version exists

**Given** an article was abrogated
**When** I query without as_of_date
**Then** I see the article marked as "abrogato"
**And** I can still access the historical text
**And** the abrogating norm is linked via `ABROGA` edge

**Given** I browse an article in the UI
**When** multiple versions exist
**Then** I see a timeline of modifications
**And** I can compare versions side-by-side (diff view)

**Existing Code:**
- `merlt/merlt/pipeline/multivigenza.py` - Temporal versioning logic ⭐
- `Legacy/MERL-T_alpha/merlt/storage/graph/temporal_queries.py` - Cypher temporal patterns
- Migration: Integrate multivigenza.py as authoritative implementation

**Technical Notes:**
- Store versions as node properties array or separate version nodes
- Index on vigenza dates for efficient temporal queries

---

### Story 2b.5: Manual Ingest Trigger

As an **administrator**,
I want to **manually trigger norm ingestion for specific content**,
So that **I can quickly add new or modified norms without waiting for scheduled scraping**.

**Acceptance Criteria:**

**Given** I am an authenticated admin user
**When** I access the ingest trigger UI/endpoint
**Then** I can specify:
  - Single article by URN
  - Range of articles (e.g., "art. 1470-1490 c.c.")
  - Entire title/chapter
  - Custom source URL

**Given** I trigger a manual ingest
**When** the process runs
**Then** I see real-time progress (articles processed, errors)
**And** I receive notification when complete
**And** the ingest is logged in audit trail

**Given** an ingest fails partially
**When** I review the results
**Then** I see which articles succeeded and which failed
**And** I can retry failed articles individually

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/api/admin_routes.py` - Admin endpoints (partial)
- New: Build dedicated ingest trigger endpoint
- Integration: Connect to Epic 2a scrapers + 2b ingestion pipeline

**Technical Notes:**
- Endpoint: POST /api/v1/admin/ingest
- Auth: Admin role required
- Async execution with job tracking

---

## Epic 2c: Vector & Bridge Table

**Goal:** Generate embeddings and maintain intelligent chunk-to-graph mappings with source-aware expert affinities.

### Story 2c.1: Text Chunking Pipeline

As a **system**,
I want to **chunk legal texts intelligently based on source type**,
So that **embeddings capture meaningful semantic units for retrieval**.

**Acceptance Criteria:**

**Given** norm text (articles, commi)
**When** chunking is applied
**Then** chunks respect legal structure boundaries (comma, lettera, periodo)
**And** chunk size is optimized for embedding model (256-512 tokens)
**And** overlap ensures context continuity (50-100 tokens)

**Given** Brocardi commentary (longer prose)
**When** chunking is applied
**Then** chunks respect paragraph boundaries
**And** massime are kept as single chunks (self-contained)
**And** case citations are preserved within chunks

**Given** doctrinal text (manuals, treatises)
**When** chunking is applied
**Then** chunks respect section/subsection boundaries
**And** footnotes/references are linked to parent chunk
**And** source attribution (author, work, page) is preserved

**Given** any chunk
**When** it's created
**Then** it includes metadata:
  - `source_type`: norm | jurisprudence | commentary | doctrine
  - `source_urn`: canonical URN of parent document
  - `source_authority`: 0.0-1.0 based on source prestige
  - `chunk_position`: location within parent document

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/retrieval/chunker.py` - Basic chunking
- `Legacy/MERL-T_alpha/merlt/retrieval/legal_chunker.py` - Legal-aware chunking
- Migration: Extend with source-type-aware strategies

**Technical Notes:**
- Different chunking strategies per source_type
- Preserve legal citation integrity within chunks

---

### Story 2c.2: Embedding Generation

As a **system**,
I want to **generate embeddings for all chunks with configurable models**,
So that **semantic search can find relevant content across all sources**.

**Acceptance Criteria:**

**Given** a batch of chunks from various sources
**When** embedding generation runs
**Then** each chunk receives a vector embedding
**And** the embedding model is configurable (e.g., multilingual-e5, BGE-M3)
**And** Italian legal terminology is well-represented

**Given** the embedding model changes
**When** I switch models via configuration
**Then** I can re-embed all content without code changes
**And** old embeddings are archived (not deleted) for comparison

**Given** a large batch (~10k chunks)
**When** embedding generation runs
**Then** processing is batched for GPU efficiency
**And** progress is logged
**And** failures are isolated per chunk

**Given** embeddings are generated
**When** I query similar chunks
**Then** legal concepts cluster appropriately (e.g., "risoluzione" near "inadempimento")

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/retrieval/embedder.py` - Embedding generation
- `Legacy/MERL-T_alpha/merlt/retrieval/models/` - Model wrappers
- Migration: Ensure GPU batching, add model versioning

**Technical Notes:**
- Model versioning critical for reproducibility (NFR-R6)
- Store model_id with each embedding for audit

---

### Story 2c.3: Qdrant Collection Setup

As a **system**,
I want to **configure Qdrant collections with rich payload schema**,
So that **retrieval can filter by source type and expert affinity**.

**Acceptance Criteria:**

**Given** Qdrant is running
**When** I initialize the legal_chunks collection
**Then** the schema includes:
  - Vector: embedding dimension (e.g., 1024 for BGE-M3)
  - Payload fields indexed for filtering:
    - `chunk_id` (string, primary key)
    - `source_urn` (string, indexed)
    - `source_type` (keyword: norm|jurisprudence|commentary|doctrine)
    - `source_authority` (float, 0.0-1.0)
    - `article_urn` (string, indexed) - parent article
    - `text` (string, stored for display)
    - `expert_affinity` (JSON object with per-expert weights)

**Given** the collection is initialized
**When** I query with filters
**Then** I can filter by source_type (e.g., "only jurisprudence")
**And** I can filter by source_authority threshold
**And** I can boost results by expert_affinity for specific expert

**Given** chunks are inserted
**When** I perform semantic search
**Then** results include full payload for display
**And** search respects HNSW index settings for NFR-P4 (<200ms)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/qdrant/client.py` - Qdrant client
- `Legacy/MERL-T_alpha/merlt/storage/qdrant/collections.py` - Collection setup
- Migration: Extend payload schema with new fields

**Technical Notes:**
- HNSW parameters: ef_construct=128, m=16 for quality/speed balance
- Payload indexes on frequently filtered fields

---

### Story 2c.4: Bridge Table Population

As a **system**,
I want to **populate the Bridge Table with intelligent chunk-to-graph mappings**,
So that **Experts can traverse from vectors to graph with source-aware preferences**.

**Acceptance Criteria:**

**Given** chunks in Qdrant and nodes in FalkorDB
**When** the Bridge Builder processes them
**Then** mappings are created with:
  - `chunk_id` → `graph_node_urn` (primary mapping)
  - `source_type` (inherited from chunk)
  - `source_authority` (inherited from chunk)
  - `mapping_type`: PRIMARY | REFERENCE | CONCEPT | DOCTRINE
  - `expert_affinity` (initial weights based on source_type):

| source_type | Literal | Systemic | Principles | Precedent |
|-------------|---------|----------|------------|-----------|
| norm | 0.9 | 0.8 | 0.5 | 0.3 |
| jurisprudence | 0.3 | 0.5 | 0.6 | 0.9 |
| commentary | 0.5 | 0.6 | 0.7 | 0.6 |
| doctrine | 0.4 | 0.5 | 0.9 | 0.4 |

**Given** a chunk references multiple norms
**When** the Bridge Builder processes it
**Then** multiple mappings are created (one per referenced URN)
**And** mapping_type distinguishes PRIMARY (main topic) from REFERENCE (citation)

**Given** F8 feedback is received (future, from Epic 6)
**When** feedback indicates source quality
**Then** expert_affinity weights are updated for that mapping
**And** the update is logged for training audit

**Given** I need to find chunks for a specific Expert
**When** I query the Bridge Table
**Then** I can filter/sort by that Expert's affinity column
**And** response time is <50ms for single URN lookup

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/bridge/bridge_table.py` - BridgeTable service ⭐
- `Legacy/MERL-T_alpha/merlt/storage/bridge/bridge_builder.py` - Builder helper ⭐
- Migration: Extend with source_type awareness, richer expert_affinity initialization

**Technical Notes:**
- Bridge Table in PostgreSQL (joins with user data, RLCF)
- expert_affinity as JSONB for flexibility
- Indexes on (graph_node_urn, expert_type) for fast Expert retrieval
- This is the foundation for F8 learning in Epic 6/7

---

## Epic 3: Norm Browsing & Search

**Goal:** Users can browse, search, and explore legal norms with temporal versioning and KG visualization.

### Story 3.1: Norm Hierarchical Browser

As a **legal professional**,
I want to **navigate norms through a hierarchical tree structure**,
So that **I can explore the legal corpus systematically**.

**Acceptance Criteria:**

**Given** I am on the browse page
**When** I view the norm tree
**Then** I see a collapsible hierarchy: Codice → Libro → Titolo → Capo → Articolo
**And** each level shows count of children (e.g., "Libro IV (523 articoli)")
**And** modified/abrogated items are visually marked (⚠️ icon)

**Given** I expand a node in the tree
**When** children are loaded
**Then** loading is lazy (on-demand) for performance
**And** response time is <200ms per level (NFR-P4)

**Given** I click on an article in the tree
**When** the article is selected
**Then** the Article Viewer (Story 3.2) displays its content
**And** the tree highlights my current position

**Existing Code:**
- `Legacy/VisuaLexAPI/frontend/src/components/NormTree/` - Tree component
- `visualex-platform/` - React component patterns
- Migration: Adapt to new React 19 + integrate with FalkorDB API

**Technical Notes:**
- Use React virtualization for large lists
- Cache expanded nodes in session

---

### Story 3.2: Article Viewer

As a **legal professional**,
I want to **view an article with its full content and metadata**,
So that **I can read and understand the norm in context**.

**Acceptance Criteria:**

**Given** I select an article
**When** the Article Viewer loads
**Then** I see:
  - Rubrica (title)
  - Full text with comma/lettera structure preserved
  - Vigenza dates (dal/al)
  - Source URN
  - Last modification info

**Given** the article has annotations (from Brocardi)
**When** I view the article
**Then** I see an "Annotations" toggle
**And** clicking it shows doctrinal commentary inline or in sidebar

**Given** I am viewing a long article
**When** I scroll
**Then** the rubrica stays visible (sticky header)
**And** a minimap shows my position in the document

**Given** the article text contains citations
**When** rendered
**Then** citations are highlighted (Story 3.4 integration)

**Existing Code:**
- `visualex-platform/frontend/src/components/ArticleTabContent.tsx` - Core viewer (~1200 LOC)
- `Legacy/VisuaLexAPI/frontend/src/components/ArticleView/` - Legacy viewer
- Migration: Refactor ArticleTabContent, integrate new data sources

**Technical Notes:**
- Progressive loading: base text first, then enrichments
- IDE paradigm: similar to code editor with syntax highlighting

---

### Story 3.3: Search by Keyword/URN

As a **legal professional**,
I want to **search norms by keyword or URN**,
So that **I can quickly find relevant articles**.

**Acceptance Criteria:**

**Given** I am on the search page
**When** I enter a keyword (e.g., "inadempimento")
**Then** I see results from:
  - Norm text (full-text search)
  - Rubricae (article titles)
  - Brocardi commentary
**And** results are ranked by relevance
**And** search completes in <500ms (NFR-P1)

**Given** I enter a URN or article reference (e.g., "art. 1453 c.c.")
**When** I submit the search
**Then** the exact article is shown as top result
**And** I can click to navigate directly

**Given** search results are displayed
**When** I view them
**Then** each result shows:
  - Article URN and rubrica
  - Snippet with keyword highlighted
  - Source type badge (norm/commentary/doctrine)
**And** I can filter by source type

**Given** I search with temporal context
**When** I add "as of date" filter
**Then** only norms in force on that date are shown

**Existing Code:**
- `Legacy/VisuaLexAPI/frontend/src/components/Search/` - Search UI
- `Legacy/MERL-T_alpha/merlt/retrieval/search.py` - Hybrid search (Qdrant + FalkorDB)
- Migration: Unify search across all content types

**Technical Notes:**
- Hybrid search: semantic (Qdrant) + keyword (FalkorDB full-text)
- Autocomplete for URN patterns

---

### Story 3.4: Citation Highlighting & Linking

As a **legal professional**,
I want to **see citations highlighted and clickable in article text**,
So that **I can quickly navigate to referenced norms**.

**Acceptance Criteria:**

**Given** article text contains a citation (e.g., "art. 2043 c.c.")
**When** the text is rendered
**Then** the citation is highlighted with a distinct style
**And** hovering shows a tooltip with article preview (Peek Definition - IDE paradigm)
**And** clicking navigates to that article

**Given** a citation references a norm not in the system
**When** rendered
**Then** the citation is still highlighted but marked as "external"
**And** tooltip shows "Norma non presente nel sistema"

**Given** multiple citation formats exist
**When** NER processes text
**Then** all formats are recognized:
  - "art. 1453 c.c."
  - "articolo 1453 del codice civile"
  - "artt. 1453-1455 c.c." (range)
  - "comma 2 dell'art. 1453"

**Given** I am in 🔍 Analisi or 🎓 Contributore profile
**When** I see a citation
**Then** I can provide F1 feedback (confirm/correct) via right-click menu

**Existing Code:**
- `Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts` - Citation parsing
- `Legacy/VisuaLexAPI/frontend/src/utils/citationMatcher.ts` - Citation matching
- `Legacy/MERL-T_alpha/merlt/ner/citation_extractor.py` - NER backend
- Migration: Consolidate frontend + backend, ensure consistency

**Technical Notes:**
- Real-time NER on frontend for responsiveness
- Backend NER for accuracy validation
- Tooltip = Peek Definition (IDE per Giuristi)

---

### Story 3.5: Cross-References Panel

As a **legal professional**,
I want to **see norms related to the current article**,
So that **I understand the systemic context**.

**Acceptance Criteria:**

**Given** I am viewing an article
**When** I open the Cross-References panel
**Then** I see grouped relationships:
  - "Riferimenti in uscita" (norms this article cites)
  - "Riferimenti in entrata" (norms citing this article)
  - "Modificato da" (modifying legislation)
  - "Giurisprudenza" (case law citing this norm)

**Given** relationships are displayed
**When** I view them
**Then** each shows: URN, rubrica, relationship type
**And** clicking navigates to that norm
**And** I can expand to see relationship context (citation snippet)

**Given** the article has many relationships (>20)
**When** I view the panel
**Then** relationships are paginated or virtualized
**And** I can filter by relationship type

**Given** I want to visualize the graph
**When** I click "View in Graph"
**Then** a graph visualization shows the article and its neighbors
**And** I can expand nodes to explore further

**Existing Code:**
- `visualex-platform/frontend/` - Reagraph for graph visualization
- `Legacy/MERL-T_alpha/merlt/storage/graph/neighbors.py` - Neighbor queries
- Migration: Build React panel component, integrate graph viz

**Technical Notes:**
- Graph query: `MATCH (a:Articolo {urn: $urn})-[r]-(b) RETURN a, r, b LIMIT 50`
- Reagraph for interactive visualization

---

### Story 3.6: Historical Versions Timeline

As a **legal professional**,
I want to **see the modification history of an article and compare versions**,
So that **I can understand how the norm evolved over time**.

**Acceptance Criteria:**

**Given** an article has multiple historical versions
**When** I open the History panel
**Then** I see a timeline showing:
  - Each version with vigenza_dal date
  - Modifying legislation for each change
  - Current version highlighted

**Given** I select two versions
**When** I click "Compare"
**Then** I see a side-by-side diff view
**And** additions are highlighted in green, deletions in red
**And** unchanged sections are collapsed

**Given** I select a historical version
**When** I click "View as of this date"
**Then** the Article Viewer shows that version
**And** a banner indicates "Versione storica - non più in vigore"

**Given** the article was abrogated
**When** I view history
**Then** abrogation is shown as final event
**And** abrogating norm is linked

**Existing Code:**
- `merlt/merlt/pipeline/multivigenza.py` - Temporal versioning logic ⭐
- `Legacy/VisuaLexAPI/frontend/src/components/Timeline/` - Timeline UI (partial)
- Migration: Build diff view component, integrate with multivigenza

**Technical Notes:**
- Diff algorithm: word-level for legal precision
- Timeline inspired by Git history view (IDE paradigm)

---

### Story 3.7: Modification Detection Alert

As a **legal professional**,
I want to **be alerted when a norm I'm viewing or have saved was recently modified**,
So that **I don't rely on outdated information**.

**Acceptance Criteria:**

**Given** I view an article that was modified in the last 30 days
**When** the article loads
**Then** I see a prominent alert banner:
  "⚠️ Questo articolo è stato modificato il [data] da [norma modificante]"
**And** I can click to see what changed

**Given** I have articles saved in a Dossier
**When** any saved article is modified
**Then** I receive a notification (in-app and/or email based on preferences)
**And** the Dossier shows a badge indicating updates

**Given** I query with a temporal context (as_of_date)
**When** a newer version exists
**Then** I see a subtle indicator "Versione più recente disponibile"
**And** I can click to see current version

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/graph/change_detection.py` - Change tracking
- New: Notification system integration
- Migration: Build alert component, integrate with scraping updates

**Technical Notes:**
- Change detection runs after each scraping cycle
- Notification preferences stored in user_preferences

---

### Story 3.8: KG Coverage Dashboard

As an **administrator**,
I want to **view a simple dashboard of Knowledge Graph coverage**,
So that **I can monitor system health and identify gaps**.

**Acceptance Criteria:**

**Given** I am an admin user
**When** I access the KG Dashboard
**Then** I see summary statistics:
  - Total norms in graph (by type: Codice, Legge, D.Lgs., etc.)
  - Total articles, commi
  - Total relationships by type
  - Last scraping timestamp per source

**Given** I view the dashboard
**When** I look at coverage metrics
**Then** I see:
  - Libro IV C.C. coverage percentage
  - Articles with/without Brocardi enrichment
  - Articles with/without jurisprudence links

**Given** there are gaps in coverage
**When** I identify them
**Then** I can trigger manual ingest (Story 2b.5) for missing content

**Given** the dashboard loads
**When** statistics are computed
**Then** response time is acceptable (<2s)
**And** data is cached with 1-hour TTL

**Existing Code:**
- New: Build dashboard component
- `Legacy/MERL-T_alpha/merlt/api/stats_routes.py` - Basic stats endpoints
- Migration: Extend stats API, build React dashboard

**Technical Notes:**
- Simple dashboard for MVP - not full admin panel
- Key metrics for thesis demo health check

---

## Epic 4: MERL-T Analysis Pipeline

**Goal:** Users can submit queries and receive AI-powered legal analysis following Art. 12 Preleggi sequence with circuit breaker resilience.

### Story 4.1: NER Pipeline Integration

As a **system**,
I want to **extract legal entities from user queries**,
So that **the Expert Router can identify relevant norms and concepts**.

**Acceptance Criteria:**

**Given** a user submits a natural language query
**When** the NER pipeline processes it
**Then** the following entities are extracted:
  - Article references (e.g., "art. 1453 c.c." → URN)
  - Legal concepts (e.g., "inadempimento", "risoluzione")
  - Temporal context (e.g., "contratto del 2019")
  - Party references (e.g., "compratore", "venditore")
**And** each entity has a confidence score (0.0-1.0)

**Given** entities are extracted
**When** I view the extraction result
**Then** I can see the original text span and entity type
**And** ambiguous entities are flagged for potential F1 feedback

**Given** extraction fails or times out
**When** the error occurs
**Then** the pipeline continues with partial results
**And** a warning is logged for monitoring

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/ner/entity_extractor.py` - Core NER model
- `Legacy/MERL-T_alpha/merlt/ner/citation_extractor.py` - Citation-specific NER
- `Legacy/MERL-T_alpha/merlt/ner/legal_ner_model.py` - Fine-tuned model wrapper
- Migration: Integrate with Epic 2a URN pipeline for resolution

**Technical Notes:**
- NER model: fine-tuned Italian BERT or spaCy legal
- Latency target: <500ms for typical query
- Entities feed into Expert Router decisions

---

### Story 4.2: Expert Router

As a **system**,
I want to **route queries to the appropriate Expert sequence**,
So that **the Art. 12 Preleggi hierarchy is followed correctly**.

**Acceptance Criteria:**

**Given** NER entities from a query
**When** the Router processes them
**Then** it determines Expert activation order: Literal → Systemic → Principles → Precedent
**And** each Expert receives relevance score (some may be skipped if not relevant)

**Given** a simple definitional query (e.g., "cos'è la risoluzione?")
**When** routed
**Then** LiteralExpert is primary, others have lower weight
**And** query type is classified: DEFINITION | INTERPRETATION | COMPARISON | CASE_ANALYSIS

**Given** a complex interpretive query (e.g., "posso risolvere se il ritardo è di 1 giorno?")
**When** routed
**Then** all 4 Experts are activated with appropriate weights
**And** SystemicExpert and PrinciplesExpert receive higher relevance

**Given** Router decision is made
**When** logged
**Then** the routing rationale is recorded for F2 feedback opportunity
**And** high-authority users (🎓) can see and rate the routing decision

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/router.py` - Routing logic ⭐
- `Legacy/MERL-T_alpha/merlt/experts/query_classifier.py` - Query type classification
- Migration: Enhance with F2 feedback integration hooks

**Technical Notes:**
- Router is trainable via RLCF (F2 feedback)
- Initial weights from heuristics, refined by community
- Sequential execution for Art. 12 compliance

---

### Story 4.3: LiteralExpert Implementation

As a **system**,
I want to **provide literal/textual interpretation of legal norms**,
So that **users receive Art. 12 comma 1 analysis (il significato proprio delle parole)**.

**Acceptance Criteria:**

**Given** a query with identified norm references
**When** LiteralExpert processes it
**Then** it retrieves relevant norm text chunks via Bridge Table
**And** it analyzes literal meaning using LLM with legal prompt
**And** output includes: interpretation, relevant text excerpts, confidence

**Given** the norm text contains defined terms
**When** LiteralExpert analyzes it
**Then** definitions from Concetto/Definizione nodes are included
**And** cross-references to defining articles are provided

**Given** LiteralExpert produces output
**When** formatted for display
**Then** it includes:
  - "Interpretazione Letterale" section header
  - Relevant norm text (with URN links)
  - Plain-language explanation
  - Confidence score (0.0-1.0)
  - Processing time

**Given** the query has no clear norm reference
**When** LiteralExpert runs
**Then** it returns low-confidence result
**And** suggests user clarify with specific article

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/literal.py` - LiteralExpert implementation ⭐
- `Legacy/MERL-T_alpha/merlt/experts/base_expert.py` - Base class with common patterns
- `Legacy/MERL-T_alpha/merlt/prompts/literal_prompts.py` - LLM prompts
- Migration: Integrate with new Bridge Table, add F3 feedback hooks

**Technical Notes:**
- Uses Bridge Table with expert_affinity[Literal] for chunk selection
- Prefers source_type=norm (highest Literal affinity)
- LLM: configurable provider (OpenAI, Anthropic, local)

---

### Story 4.4: SystemicExpert Implementation

As a **system**,
I want to **provide systematic interpretation via Knowledge Graph traversal**,
So that **users understand how norms connect within the legal system (Art. 12 - intenzione del legislatore)**.

**Acceptance Criteria:**

**Given** LiteralExpert has identified relevant norms
**When** SystemicExpert processes
**Then** it traverses the Knowledge Graph to find:
  - Related norms via RIFERIMENTO edges
  - Modifying legislation via MODIFICA edges
  - EU directives via ATTUA edges
  - General principles that govern the specific norm

**Given** traversal is performed
**When** results are analyzed
**Then** LLM synthesizes how related norms affect interpretation
**And** output shows the "contesto normativo" with graph visualization

**Given** SystemicExpert produces output
**When** formatted for display
**Then** it includes:
  - "Interpretazione Sistematica" section header
  - Graph of related norms (mini visualization)
  - Synthesis of systemic context
  - Key cross-references with explanations
  - Confidence score

**Given** the norm is isolated (few relationships)
**When** SystemicExpert runs
**Then** it notes the limited systemic context
**And** returns lower confidence with explanation

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/systemic.py` - SystemicExpert ⭐
- `Legacy/MERL-T_alpha/merlt/storage/graph/traversal.py` - Graph traversal utilities
- `Legacy/MERL-T_alpha/merlt/experts/traversal_policy.py` - TraversalPolicy (F8 target)
- Migration: Integrate TraversalPolicy with F8 learning, add F4 feedback hooks

**Technical Notes:**
- Traversal depth: configurable (default 2 hops)
- Uses expert_affinity[Systemic] for chunk boosting
- TraversalPolicy learns "path virtuosi" from F8 feedback

---

### Story 4.5: PrinciplesExpert Implementation

As a **system**,
I want to **provide teleological interpretation based on legal principles**,
So that **users understand the purpose and values behind the norm (Art. 12 - ratio legis)**.

**Acceptance Criteria:**

**Given** prior Expert outputs
**When** PrinciplesExpert processes
**Then** it identifies:
  - Underlying legal principles (e.g., buona fede, tutela affidamento)
  - Legislative intent from travaux préparatoires (if available)
  - Constitutional values that inform interpretation
  - Doctrinal commentary on purpose

**Given** principles are identified
**When** analyzed by LLM
**Then** output explains how principles guide interpretation
**And** potential tensions between principles are highlighted

**Given** PrinciplesExpert produces output
**When** formatted for display
**Then** it includes:
  - "Interpretazione Teleologica" section header
  - Identified principles with definitions
  - Ratio legis explanation
  - Doctrinal support (from Brocardi/doctrine chunks)
  - Confidence score

**Given** doctrinal sources disagree
**When** PrinciplesExpert analyzes
**Then** it presents multiple views (Devil's Advocate preparation)
**And** notes the interpretive uncertainty

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/principles.py` - PrinciplesExpert ⭐
- `Legacy/MERL-T_alpha/merlt/retrieval/principle_matcher.py` - Principle identification
- Migration: Enhance with doctrine source_type integration (commentary, doctrine chunks)

**Technical Notes:**
- Heavily uses source_type=doctrine chunks (high Principles affinity)
- May access external principle taxonomy if available
- F5 feedback for principle relevance

---

### Story 4.6: PrecedentExpert Implementation

As a **system**,
I want to **provide jurisprudential interpretation from case law**,
So that **users understand how courts have applied the norm**.

**Acceptance Criteria:**

**Given** prior Expert outputs with norm references
**When** PrecedentExpert processes
**Then** it retrieves:
  - Relevant Cassazione decisions via CITATO_DA edges
  - Massime (legal maxims) from Brocardi
  - Recent case law trends
  - Minority/dissenting views

**Given** case law is retrieved
**When** analyzed by LLM
**Then** output synthesizes judicial interpretation
**And** distinguishes between consolidated and evolving positions

**Given** PrecedentExpert produces output
**When** formatted for display
**Then** it includes:
  - "Giurisprudenza" section header
  - Key decisions with citations (court, date, case number)
  - Massime relevant to the query
  - Trend analysis (if sufficient data)
  - Confidence score

**Given** case law conflicts exist
**When** PrecedentExpert analyzes
**Then** it presents the conflict explicitly
**And** notes the most recent authoritative position
**And** flags for Devil's Advocate consideration

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/precedent.py` - PrecedentExpert ⭐
- `Legacy/MERL-T_alpha/merlt/retrieval/case_retriever.py` - Case law retrieval
- Migration: Integrate with Brocardi jurisprudence, add F6 feedback hooks

**Technical Notes:**
- Primarily uses source_type=jurisprudence chunks (highest Precedent affinity)
- Cassazione > Appello > Tribunale for authority weighting
- F6 feedback for precedent relevance

---

### Story 4.7: Gating Network

As a **system**,
I want to **combine Expert outputs with learned weights**,
So that **the final response reflects appropriate balance of interpretive methods**.

**Acceptance Criteria:**

**Given** all activated Experts have produced outputs
**When** the Gating Network processes them
**Then** it assigns combination weights based on:
  - Query type (from Router)
  - Expert confidence scores
  - Learned preferences from RLCF
  - User profile preferences

**Given** weights are assigned
**When** combination is performed
**Then** outputs are merged without losing individual Expert identity
**And** the combined output preserves traceability to each Expert

**Given** one Expert has very low confidence
**When** gating occurs
**Then** that Expert's contribution is minimized
**And** a note indicates limited input from that interpretive method

**Given** Gating produces combined output
**When** passed to Synthesizer
**Then** the structure includes all Expert outputs with their weights
**And** the weighting rationale is logged for F7 feedback

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/gating.py` - Gating Network ⭐
- `Legacy/MERL-T_alpha/merlt/experts/weight_learner.py` - RLCF weight learning
- Migration: Integrate with F7 feedback loop

**Technical Notes:**
- Softmax over Expert weights for normalization
- Weights are user-profile-aware (🔍 Analisi gets full trace, ⚡ gets summary)
- Gating weights are RLCF-trainable (aggregate feedback)

---

### Story 4.8: Synthesizer

As a **system**,
I want to **produce a coherent final response from combined Expert outputs**,
So that **users receive a unified, readable answer**.

**Acceptance Criteria:**

**Given** Gating Network output with weighted Expert contributions
**When** Synthesizer processes
**Then** it generates a unified response that:
  - Answers the user's original question
  - Integrates insights from multiple Experts coherently
  - Maintains Art. 12 sequence visibility
  - Includes confidence assessment

**Given** the response is generated
**When** formatted for display
**Then** structure depends on user profile:
  - ⚡ Consultazione: Summary + key conclusion
  - 📖 Ricerca: Summary + expandable Expert sections
  - 🔍 Analisi: Full trace with all Expert outputs
  - 🎓 Contributore: Full trace + feedback options

**Given** Experts disagree on interpretation
**When** Synthesizer processes
**Then** disagreement is noted explicitly
**And** the synthesis explains the tension
**And** Devil's Advocate is flagged for activation

**Given** Synthesizer produces output
**When** displayed to user
**Then** response includes:
  - Main answer (always visible)
  - Expert Accordion (collapsed by default, expandable)
  - Source links (URNs clickable)
  - Confidence indicator
  - F7 feedback opportunity (for eligible profiles)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/synthesizer.py` - Synthesis logic ⭐
- `Legacy/MERL-T_alpha/merlt/prompts/synthesis_prompts.py` - LLM prompts
- Migration: Add profile-aware formatting, F7 hooks

**Technical Notes:**
- LLM-based synthesis with structured prompting
- Output includes metadata for traceability (Epic 5)
- Progressive disclosure per UX spec

---

### Story 4.9: Circuit Breaker Implementation

As a **system**,
I want to **gracefully handle Expert failures without crashing the pipeline**,
So that **users receive partial results rather than errors (ADR-001)**.

**Acceptance Criteria:**

**Given** an Expert times out or fails
**When** the circuit breaker triggers
**Then** the pipeline continues with remaining Experts
**And** the failed Expert's contribution is marked as "unavailable"
**And** user sees a note: "Analisi [Expert] temporaneamente non disponibile"

**Given** an Expert fails repeatedly (>3 failures in 5 minutes)
**When** circuit breaker state is evaluated
**Then** that Expert is temporarily disabled (circuit "open")
**And** requests skip that Expert until health check passes
**And** alert is logged for admin review

**Given** circuit is open for an Expert
**When** health check interval passes (default 60s)
**Then** circuit enters "half-open" state
**And** next request tests that Expert
**And** success closes circuit, failure keeps it open

**Given** LLM provider fails
**When** detected
**Then** automatic failover to backup provider (NFR-R4)
**And** original provider is retried after cooldown

**Given** circuit breaker events occur
**When** logged
**Then** admin can view circuit breaker status in dashboard (Story 4.11)
**And** historical failure patterns are analyzable

**Existing Code:**
- New: Implement circuit breaker pattern
- Reference: `pybreaker` library or custom implementation
- Integration: Wrap each Expert call with circuit breaker

**Technical Notes:**
- Per-Expert circuit breakers (not global)
- States: CLOSED → OPEN → HALF_OPEN → CLOSED
- Configurable thresholds via environment/config

---

### Story 4.10: LLM Provider Abstraction

As a **system**,
I want to **abstract LLM provider details behind a common interface**,
So that **I can switch providers without code changes (NFR-I2)**.

**Acceptance Criteria:**

**Given** multiple LLM providers are configured
**When** the system initializes
**Then** providers are loaded from configuration:
  - Primary: e.g., OpenAI GPT-4
  - Backup: e.g., Anthropic Claude
  - Local: e.g., Ollama for development

**Given** an LLM call is made
**When** using the abstraction layer
**Then** the call uses a common interface:
  - `generate(prompt, max_tokens, temperature, ...)`
  - `embed(texts)` (for embedding models)
**And** provider-specific details are encapsulated

**Given** the primary provider fails
**When** failover is triggered
**Then** backup provider is used transparently
**And** response format remains consistent
**And** cost/latency differences are logged for monitoring

**Given** provider is changed via configuration
**When** system restarts
**Then** new provider is used without code deployment
**And** model versioning is recorded for reproducibility (NFR-R6)

**Given** different Experts need different models
**When** configured
**Then** each Expert can specify preferred model
**And** fallback chain is per-Expert configurable

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/llm/provider.py` - Base provider interface ⭐
- `Legacy/MERL-T_alpha/merlt/llm/openai_provider.py` - OpenAI implementation
- `Legacy/MERL-T_alpha/merlt/llm/anthropic_provider.py` - Anthropic implementation
- Migration: Ensure interface completeness, add model versioning

**Technical Notes:**
- litellm as potential unified interface
- Track: model_id, provider, version per request for audit
- Cost tracking for budget management

---

### Story 4.11: Expert Pipeline Status UI

As a **legal professional**,
I want to **see the status of each Expert while my query is processed**,
So that **I understand the analysis progress and any issues**.

**Acceptance Criteria:**

**Given** I submit a query
**When** processing begins
**Then** I see a progress indicator showing:
  - Which Experts will be activated
  - Current Expert being processed
  - Completed Experts with status (✓ success, ⚠️ partial, ✗ failed)

**Given** an Expert is processing
**When** I view the status
**Then** I see:
  - Expert name and icon
  - Elapsed time
  - Brief description of what it's doing

**Given** an Expert completes
**When** status updates
**Then** the next Expert begins (or pipeline completes)
**And** completed Expert shows confidence score preview

**Given** circuit breaker triggers for an Expert
**When** I view the status
**Then** I see a warning indicator for that Expert
**And** tooltip explains "Servizio temporaneamente limitato"
**And** remaining Experts continue processing

**Given** all Experts complete
**When** final response is displayed
**Then** the status UI collapses into an Expert summary
**And** I can expand to see detailed per-Expert metrics

**Existing Code:**
- `visualex-platform/frontend/` - React component patterns
- New: Build pipeline status component
- WebSocket or polling for real-time updates

**Technical Notes:**
- Progressive disclosure: simple for ⚡, detailed for 🔍/🎓
- IDE-inspired: like build/compile progress
- Mobile-friendly: collapsible on small screens

---

### Story 4.12: Gold Standard Regression

As a **system administrator**,
I want to **run regression tests against validated query-response pairs**,
So that **I can detect when system changes degrade quality**.

**Acceptance Criteria:**

**Given** a set of gold standard queries with expected responses
**When** I trigger regression test suite
**Then** each query is processed through the full pipeline
**And** responses are compared to expected outputs
**And** similarity scores are computed (semantic + structural)

**Given** regression tests complete
**When** I view results
**Then** I see:
  - Pass/fail status per query
  - Similarity scores with thresholds
  - Queries that degraded significantly (score drop >10%)
  - New queries that improved

**Given** a model or provider change is deployed
**When** regression runs automatically
**Then** results are compared to previous baseline
**And** alerts are raised if aggregate quality drops
**And** detailed diff is available for failed queries

**Given** the gold standard set
**When** I manage it
**Then** I can:
  - Add new validated Q&A pairs
  - Update expected responses after intentional changes
  - Tag queries by topic/Expert focus
  - Export/import for backup

**Given** thesis demo preparation
**When** I run full regression
**Then** I have confidence the system performs as expected
**And** any issues are identified before defense

**Existing Code:**
- `Legacy/MERL-T_alpha/tests/benchmark/` - Benchmark test framework
- `Legacy/MERL-T_alpha/tests/benchmark/gold_standard.json` - Example Q&A pairs
- Migration: Extend with semantic comparison, integrate with CI

**Technical Notes:**
- Gold standard: ~50 validated queries for MVP
- Semantic similarity: cosine distance on embeddings
- Structural check: Expert presence, citation coverage
- Run before each deployment (CI/CD integration)

---

## Epic 5: Traceability & Source Verification

**Goal:** Users can trace every statement to its source and export reasoning for legal briefs.

### Story 5.1: Reasoning Trace Storage

As a **system**,
I want to **store complete reasoning traces for every response**,
So that **users can audit the analysis process and researchers can study system behavior**.

**Acceptance Criteria:**

**Given** an Expert pipeline completes
**When** the response is generated
**Then** a complete trace is stored containing:
  - Query ID (unique identifier)
  - Original query text
  - NER extraction results (entities, confidence)
  - Router decision (Expert weights, query type)
  - Per-Expert trace:
    - Input (query + context)
    - Retrieved chunks (chunk_ids, scores)
    - LLM prompt sent
    - LLM response received
    - Processing time
    - Confidence score
  - Gating weights
  - Synthesizer output
  - Total processing time

**Given** a trace is stored
**When** I query by trace_id
**Then** I can retrieve the complete trace
**And** all chunk_ids resolve to actual content
**And** all URNs link to graph nodes

**Given** the user has Basic consent level
**When** trace is stored
**Then** query text is NOT stored (only trace_id)
**And** trace is marked "anonymous"

**Given** traces accumulate
**When** storage grows
**Then** old traces are archived per retention policy (30 days active, then archive)
**And** archived traces remain accessible for research (if consent permits)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/trace/trace_store.py` - Trace storage service
- `Legacy/MERL-T_alpha/merlt/trace/trace_model.py` - Trace data model
- Migration: Extend with consent-aware storage, archive policy

**Technical Notes:**
- Storage: PostgreSQL JSONB for flexibility
- Index on: trace_id, user_id, created_at, query_type
- Trace is foundation for F3-F7 feedback and Epic 8 research

---

### Story 5.2: Trace Viewer UI

As a **legal professional**,
I want to **view the complete reasoning trace for any response**,
So that **I can understand how the system reached its conclusions**.

**Acceptance Criteria:**

**Given** I receive a response from MERL-T
**When** I am in 🔍 Analisi or 🎓 Contributore profile
**Then** I see an "Expert Accordion" below the main response
**And** each Expert section is collapsible (collapsed by default)

**Given** I expand an Expert section
**When** viewing the trace
**Then** I see:
  - Expert name and confidence score
  - Brief summary of contribution
  - "Sources used" list with URN links
  - "Show full trace" toggle for detailed view

**Given** I click "Show full trace"
**When** detailed view expands
**Then** I see:
  - Retrieved chunks with relevance scores
  - LLM reasoning (if available)
  - Processing time
  - Any warnings or limitations

**Given** I am in ⚡ Consultazione or 📖 Ricerca profile
**When** I view a response
**Then** I see a simplified "View sources" link
**And** clicking shows a flat list of cited sources
**And** full trace is not shown (progressive disclosure)

**Given** I hover over any source URN
**When** tooltip appears
**Then** I see Peek Definition (article preview)
**And** I can click to navigate to full article

**Existing Code:**
- `visualex-platform/frontend/` - React component patterns
- New: Build Expert Accordion component per UX spec
- Integration: Fetch trace from Story 5.1 storage

**Technical Notes:**
- Accordion: Radix UI or custom implementation
- Lazy load detailed trace on expand (reduce initial payload)
- IDE paradigm: similar to debugger call stack view

---

### Story 5.3: Source Navigation

As a **legal professional**,
I want to **navigate from any statement in a response to its source**,
So that **I can verify the system's interpretation against primary sources**.

**Acceptance Criteria:**

**Given** a response contains statements derived from sources
**When** rendered
**Then** each statement has an inline source indicator
**And** source indicators are subtle but visible (small superscript number or icon)

**Given** I click on a source indicator
**When** navigation occurs
**Then** I am taken to the exact source:
  - For norms: Article Viewer at relevant comma
  - For jurisprudence: Case reference with massima
  - For doctrine: Commentary section
**And** the relevant text is highlighted

**Given** a statement has multiple sources
**When** I view the indicator
**Then** I see a tooltip listing all sources
**And** I can choose which to navigate to

**Given** a statement is synthesized (not directly from source)
**When** I view the indicator
**Then** indicator shows "Synthesis" marker
**And** tooltip explains: "Questa affermazione è una sintesi basata su [sources]"

**Given** I am reading in Split View (IDE paradigm)
**When** I click a source
**Then** the source opens in the adjacent panel
**And** I can see response and source side-by-side

**Existing Code:**
- `Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts` - Citation linking
- `visualex-platform/frontend/` - Split view component
- Integration: Connect with Epic 3.4 citation highlighting

**Technical Notes:**
- Source mapping stored in trace (Statement → [chunk_ids] → URNs)
- Split View: Ctrl+\ to toggle (IDE shortcut)
- Preserve scroll position when returning

---

### Story 5.4: Temporal Validity Check

As a **legal professional**,
I want to **verify that cited sources are still valid/in force**,
So that **I don't rely on outdated legal information**.

**Acceptance Criteria:**

**Given** a response cites a norm
**When** the response is displayed
**Then** the system checks if the norm is still in force
**And** if modified/abrogated, a warning indicator appears

**Given** a cited norm was modified after the query date
**When** I view the source
**Then** I see an alert: "⚠️ Modificato il [data] - verificare vigenza"
**And** I can click to see what changed (Story 3.6 integration)

**Given** a cited norm was abrogated
**When** I view the source
**Then** I see a clear warning: "⛔ Abrogato il [data] da [norma]"
**And** the abrogating norm is linked

**Given** the original query had temporal context (as_of_date)
**When** I view the response
**Then** all validity checks are relative to that date
**And** a banner shows "Analisi basata sulla normativa vigente al [data]"

**Given** multiple sources have validity issues
**When** I view the response
**Then** a summary warning appears at the top
**And** lists the affected sources with brief status

**Existing Code:**
- `merlt/merlt/pipeline/multivigenza.py` - Temporal versioning ⭐
- `Legacy/MERL-T_alpha/merlt/storage/graph/change_detection.py` - Change detection
- Integration: Connect with Epic 3.7 modification alerts

**Technical Notes:**
- Check performed at render time (not stored - validity can change)
- Cache validity status for 24 hours
- Critical for legal reliability

---

### Story 5.5: Citation Export

As a **legal professional**,
I want to **export the reasoning trace in citation-ready format**,
So that **I can use ALIS analysis in legal documents and briefs**.

**Acceptance Criteria:**

**Given** I have a response with traced sources
**When** I click "Export Citations"
**Then** I can choose format:
  - Italian legal citation style (default)
  - BibTeX (for academic papers)
  - Plain text (for quick copy)
  - JSON (for integration)

**Given** I export in Italian legal style
**When** the export is generated
**Then** each source is formatted correctly:
  - Norms: "Art. 1453 c.c." or full "Art. 1453 Codice Civile"
  - Case law: "Cass. Civ., Sez. II, 15/03/2023, n. 1234"
  - Doctrine: "Autore, Titolo, Anno, p. XX"
**And** the query summary is included as context

**Given** I export in BibTeX
**When** the export is generated
**Then** each source has a proper @misc or @article entry
**And** URNs are included as notes/URLs

**Given** I export
**When** the file is downloaded
**Then** filename includes query summary and date
**And** format is UTF-8 with Italian characters preserved

**Given** the response includes synthesized statements
**When** exporting
**Then** synthesis is noted: "Elaborazione a cura di ALIS basata su:"
**And** underlying sources are listed

**Existing Code:**
- New: Build citation formatter service
- Reference: Italian legal citation conventions
- Integration: Use trace data from Story 5.1

**Technical Notes:**
- Italian legal citation style per prassi forense
- Include ALIS version/date for reproducibility
- Consider PDF export for final documents (future)

---

## Epic 6: RLCF Feedback Collection

**Goal:** Users can provide feedback on AI responses across all 8 feedback points (F1-F8) with profile-adaptive UI.

### Story 6.1: Feedback Data Model

As a **system**,
I want to **store feedback with a unified schema across all feedback types**,
So that **RLCF training can aggregate and weight feedback consistently**.

**Acceptance Criteria:**

**Given** the feedback system is initialized
**When** I define the data model
**Then** the schema supports all feedback types:
```
Feedback {
  feedback_id: UUID
  trace_id: UUID (link to reasoning trace)
  user_id: UUID (anonymized for storage)
  feedback_type: F1|F2|F3|F4|F5|F6|F7|F8
  component_id: string (NER|Router|Literal|Systemic|Principles|Precedent|Synthesizer|Bridge)
  target_id: string (entity_id, chunk_id, or null)

  # Feedback content
  rating: int (1-5 scale, nullable)
  correction: JSON (for F1 entity corrections)
  comment: text (nullable, anonymized)
  flags: string[] (ERROR|INCOMPLETE|IRRELEVANT|EXCELLENT)

  # Metadata
  authority_score: float (user's score at submission time)
  consent_level: BASIC|LEARNING|RESEARCH
  created_at: timestamp
  session_context: JSON (profile, query_type, etc.)
}
```

**Given** feedback is submitted
**When** stored
**Then** authority_score is captured at submission time (immutable)
**And** consent_level determines what is stored (Story 6.8)

**Given** feedback types exist
**When** I query by type
**Then** I can aggregate feedback per component
**And** indexes support efficient aggregation queries

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/feedback/models.py` - Basic feedback model
- Migration: Extend with full F1-F8 support, authority capture

**Technical Notes:**
- PostgreSQL with JSONB for flexible content
- Partitioning by month for retention management
- Foreign key to traces (if consent permits)

---

### Story 6.2: NER Feedback (F1)

As a **legal professional (🔍/🎓)**,
I want to **confirm or correct entity extractions in my query**,
So that **the NER model learns from my legal expertise**.

**Acceptance Criteria:**

**Given** I submit a query and see NER highlights
**When** I right-click on a highlighted entity
**Then** I see a context menu:
  - "✓ Corretto" (confirm extraction)
  - "✗ Errato" (mark as wrong)
  - "Correggi..." (provide correction)

**Given** I select "Correggi..."
**When** a correction dialog appears
**Then** I can:
  - Change entity type (e.g., "articolo" → "comma")
  - Change entity span (select correct text)
  - Change resolved URN (if wrong article detected)
**And** my correction is stored as F1 feedback

**Given** I confirm an extraction ("✓ Corretto")
**When** feedback is stored
**Then** it counts as positive training signal
**And** the interaction is quick (<1 click after right-click)

**Given** the system detects I corrected many entities
**When** session ends
**Then** a summary shows: "Hai migliorato 5 estrazioni - grazie!"
**And** my authority score benefits from quality contributions

**Given** I am in ⚡ Consultazione profile
**When** I view entities
**Then** I do NOT see the feedback options (progressive disclosure)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/feedback/ner_feedback.py` - Basic F1 collection
- `visualex-platform/frontend/` - Context menu patterns
- Migration: Build inline feedback UI component

**Technical Notes:**
- F1 feedback is high-value for NER model fine-tuning
- Low friction: confirm with single click
- Correction requires more effort but weighted higher

---

### Story 6.3: Expert Output Feedback (F3-F6)

As a **legal professional (🔍/🎓)**,
I want to **rate each Expert's contribution independently**,
So that **the system learns which interpretive methods work best**.

**Acceptance Criteria:**

**Given** I view a response with Expert Accordion
**When** I expand an Expert section (e.g., LiteralExpert)
**Then** I see feedback controls for that Expert:
  - 5-star rating scale
  - Quick flags: [Preciso] [Incompleto] [Irrilevante] [Eccellente]
  - Optional comment field

**Given** I provide feedback for an Expert
**When** I submit
**Then** F3/F4/F5/F6 feedback is stored (based on Expert type)
**And** the Expert section shows "Feedback inviato ✓"
**And** I can still modify until I navigate away

**Given** I rate multiple Experts
**When** ratings differ significantly
**Then** the Gating Network receives signal about relative quality
**And** future queries may adjust Expert weights

**Given** I flag an Expert output as "Irrilevante"
**When** submitting
**Then** I'm prompted for brief reason (optional)
**And** this feedback is weighted for improvement priority

**Given** I am in 📖 Ricerca profile
**When** I view Expert sections
**Then** I see simplified feedback (just rating, no detailed flags)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/feedback/expert_feedback.py` - Basic Expert feedback
- New: Build per-Expert feedback component
- Integration: Connect with Epic 5.2 Trace Viewer

**Technical Notes:**
- F3=Literal, F4=Systemic, F5=Principles, F6=Precedent
- Feedback per Expert stored separately for targeted training
- Rating distribution helps identify weak Experts

---

### Story 6.4: Synthesizer Feedback (F7)

As a **legal professional**,
I want to **rate the overall response quality**,
So that **the system learns to produce better syntheses**.

**Acceptance Criteria:**

**Given** I receive a response
**When** viewing the main answer
**Then** I see feedback controls (all profiles except ⚡):
  - Thumbs up / Thumbs down (quick)
  - "Rate in detail" expansion for more options

**Given** I click thumbs up
**When** feedback is recorded
**Then** F7 positive signal is stored
**And** visual confirmation appears (subtle animation)
**And** no further interaction required

**Given** I click thumbs down or "Rate in detail"
**When** expanded feedback appears
**Then** I see:
  - 5-star rating
  - Checklist: [Risponde alla domanda] [Ben strutturato] [Citazioni corrette] [Utile per il mio caso]
  - Comment field
  - "Cosa manca?" text field

**Given** I submit detailed F7 feedback
**When** stored
**Then** all dimensions are captured
**And** linked to Expert F3-F6 feedback (if provided)
**And** correlation with F8 is computed (Story 6.5)

**Given** I am 🎓 Contributore
**When** I submit F7 feedback
**Then** I also see option: "Suggerisci risposta migliore"
**And** can provide a model answer for training

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/feedback/response_feedback.py` - Response-level feedback
- New: Build progressive feedback widget
- Integration: Link with F3-F6 for correlation

**Technical Notes:**
- F7 is most visible feedback type - must be low friction
- Thumbs up/down → detailed is progressive disclosure
- Model answer suggestions are gold (authority-weighted heavily)

---

### Story 6.5: Bridge Quality Feedback (F8)

As a **system**,
I want to **collect feedback on source relevance for Expert-chunk mappings**,
So that **TraversalPolicy learns "path virtuosi" per Expert**.

**Acceptance Criteria:**

**Given** a response includes sources from Bridge Table
**When** user provides F7 feedback (positive or detailed)
**Then** F8 is inferred implicitly:
  - If F7 positive + F3-F6 positive → sources were relevant
  - If F7 negative + specific Expert negative → that Expert's sources may be poor

**Given** inference is computed
**When** F8 implicit signal is stored
**Then** it includes:
  - chunk_ids used by each Expert
  - correlation with Expert ratings
  - overall response rating
**And** expert_affinity update is queued

**Given** user is 🎓 Contributore
**When** they view Fonti Usate panel (Story 6.6)
**Then** they can provide EXPLICIT F8 feedback
**And** explicit feedback is weighted higher than implicit

**Given** F8 feedback accumulates for a chunk
**When** threshold is reached
**Then** expert_affinity in Bridge Table is updated
**And** TraversalPolicy retraining is triggered (Epic 7)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/bridge/bridge_table.py` - Bridge Table
- `Legacy/MERL-T_alpha/merlt/feedback/bridge_feedback.py` - Basic F8
- Migration: Add implicit inference, correlation computation

**Technical Notes:**
- F8 is unique: mostly implicit, explicit for experts only
- Correlation: F7↔F3-F6 determines F8 signal strength
- Critical for "grafo pubblico, testi privati, bridge impara" separation

---

### Story 6.6: Fonti Usate Panel

As a **🎓 Contributore**,
I want to **see and rate all sources used in a response**,
So that **I can provide explicit F8 feedback on source quality**.

**Acceptance Criteria:**

**Given** I am 🎓 Contributore viewing a response
**When** I look below the Expert Accordion
**Then** I see a "Fonti Usate" panel showing:
  - All chunks/sources used across Experts
  - Source type badge (norm/jurisprudence/commentary/doctrine)
  - Which Expert(s) used each source
  - Relevance score (system's assessment)

**Given** I view a source in the panel
**When** I want to rate it
**Then** I see rating options:
  - [Rilevante] [Parzialmente rilevante] [Irrilevante]
  - [Manca fonte migliore] (triggers suggestion prompt)

**Given** I mark a source as "Irrilevante"
**When** feedback is stored
**Then** F8 negative signal is recorded for that chunk
**And** I'm asked: "Per quale Expert era irrilevante?" (multi-select)

**Given** I click "Manca fonte migliore"
**When** prompted
**Then** I can enter a URN or search for the missing source
**And** this suggestion is stored as gold feedback (highest weight)

**Given** Fonti Usate panel is open
**When** I hover over a source
**Then** I see Peek Definition preview
**And** can click to navigate (Story 5.3)

**Existing Code:**
- New: Build Fonti Usate panel component
- Integration: Fetch source data from trace (Story 5.1)
- Integration: Store F8 explicit feedback (Story 6.5)

**Technical Notes:**
- Only visible to 🎓 Contributore (highest expertise)
- Source suggestion is extremely valuable for training
- Panel is collapsible (default collapsed)

---

### Story 6.7: Feedback History

As a **user**,
I want to **view my feedback contribution history**,
So that **I can see my impact on the system**.

**Acceptance Criteria:**

**Given** I have submitted feedback
**When** I access my profile
**Then** I see a "I miei contributi" section showing:
  - Total feedback count by type (F1, F3-F7, F8)
  - Feedback this month
  - Quality rating (if computed - Epic 7)

**Given** I click on feedback history
**When** detailed view opens
**Then** I see a list of my feedback:
  - Date and query summary (if consent permits)
  - Feedback type and rating given
  - Status: [Pending] [Used in training] [Validated]

**Given** I click on a historical feedback entry
**When** details expand
**Then** I see my original feedback
**And** if 🎓: see if it aligned with community consensus

**Given** I want to export my feedback history
**When** I request export
**Then** I receive JSON with all my feedback data
**And** this fulfills GDPR Art. 20 portability

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/api/user_routes.py` - User endpoints
- New: Build feedback history UI component
- Integration: Query from feedback storage (Story 6.1)

**Technical Notes:**
- Query summary shown only if consent=LEARNING|RESEARCH
- "Used in training" status from Epic 7 training loop
- Pagination for heavy contributors

---

### Story 6.8: PII Anonymization

As a **system**,
I want to **anonymize personal information before RLCF storage**,
So that **feedback can be used for training without privacy risk (NFR-S5)**.

**Acceptance Criteria:**

**Given** feedback is submitted
**When** it contains PII (query text, comments)
**Then** anonymization is applied before storage:
  - Names → [NOME]
  - Specific dates → [DATA]
  - Case-specific details → generalized
  - User email → never stored with feedback

**Given** user has BASIC consent
**When** feedback is submitted
**Then** only the rating and flags are stored
**And** query text is NOT stored
**And** trace link is anonymized (hash, not direct FK)

**Given** user has LEARNING consent
**When** feedback is submitted
**Then** anonymized query is stored
**And** feedback can be linked to trace for analysis

**Given** user has RESEARCH consent
**When** feedback is submitted
**Then** full anonymized data is stored
**And** data can be exported for academic research

**Given** anonymization runs
**When** I audit the stored data
**Then** no PII is recoverable
**And** feedback remains useful for training

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/privacy/anonymizer.py` - Basic anonymization
- Migration: Enhance with legal-specific patterns, consent integration

**Technical Notes:**
- Anonymization before storage (not after)
- Legal NER can help identify entities to anonymize
- Middleware enforces consent checks

---

### Story 6.9: Audit Trail

As a **system**,
I want to **maintain an immutable audit log of all feedback operations**,
So that **I can demonstrate compliance and debug issues (NFR-R5)**.

**Acceptance Criteria:**

**Given** any feedback is submitted, modified, or deleted
**When** the operation occurs
**Then** an audit log entry is created:
  - timestamp (immutable)
  - operation (CREATE|UPDATE|DELETE)
  - actor_id (anonymized user reference)
  - feedback_type
  - content_hash (what was submitted)
  - consent_level at time of operation

**Given** the audit log exists
**When** I query it
**Then** entries are append-only (cannot be modified)
**And** tampering is detectable (hash chain)
**And** retention is 7 years (NFR-R5)

**Given** a user requests erasure (GDPR Art. 17)
**When** feedback is deleted
**Then** audit log retains:
  - anonymized entry: "feedback deleted at user request"
  - original content is NOT retained
  - compliance with both erasure and audit requirements

**Given** an admin needs to investigate
**When** querying audit log
**Then** they can trace feedback lifecycle
**And** without accessing PII

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/audit/audit_log.py` - Basic audit
- Migration: Add hash chain, retention policy, GDPR compliance

**Technical Notes:**
- Append-only table with no UPDATE/DELETE permissions
- Hash chain: each entry includes hash of previous entry
- Separate from consent_audit_log (Story 1.4)

---

### Story 6.10: Synthetic Feedback Generator

As a **system administrator**,
I want to **generate synthetic feedback for initial training**,
So that **RLCF can bootstrap before real users provide feedback**.

**Acceptance Criteria:**

**Given** the system is deployed with minimal feedback
**When** I run the synthetic generator
**Then** it produces realistic feedback for:
  - Gold standard queries (Story 4.12)
  - Expected positive ratings for correct responses
  - Expected negative ratings for known weaknesses

**Given** synthetic feedback is generated
**When** stored
**Then** it is clearly marked as `source=SYNTHETIC`
**And** weighted lower than real user feedback
**And** can be excluded from research exports

**Given** real feedback accumulates
**When** threshold is reached (e.g., 100 real feedback items)
**Then** synthetic feedback weight decreases automatically
**And** eventually excluded from active training

**Given** I need to test feedback pipeline
**When** running in dev environment
**Then** synthetic generator can produce high volume
**And** all consent/anonymization rules are bypassed (dev only)

**Existing Code:**
- `Legacy/MERL-T_alpha/tests/fixtures/feedback_fixtures.py` - Test fixtures
- New: Build synthetic generator service
- Integration: Use gold standard Q&A from Story 4.12

**Technical Notes:**
- Synthetic feedback for cold start problem
- Clearly separated from real data
- Useful for thesis demo if limited real users

---

## Epic 7: Authority & Learning Loop

**Goal:** System learns from community feedback with authority weighting (RLCF training loop).

### Story 7.1: Authority Score Computation

As a **system**,
I want to **dynamically compute and update user authority scores**,
So that **feedback is weighted by expertise and track record**.

**Acceptance Criteria:**

**Given** a user has registered and provided credentials
**When** authority score is computed
**Then** the formula is applied:
  `A_u(t) = α·B_u + β·T_u(t) + γ·P_u(t)`
  Where:
  - α=0.3: Baseline from credentials (fixed after registration)
  - β=0.5: Track record (historical accuracy)
  - γ=0.2: Recent performance (last N feedback)

**Given** baseline score B_u is determined
**When** user registers
**Then** B_u is computed from:
  - Professional role (avvocato=0.8, praticante=0.5, studente=0.3)
  - Years of experience (0.1 per 5 years, max +0.3)
  - Specialization relevance (if contract law → +0.1 for Libro IV queries)

**Given** track record T_u(t) is updated
**When** user feedback is validated against consensus
**Then** T_u increases if feedback aligns with majority
**And** T_u decreases if feedback is outlier (but not penalized for minority position)

**Given** recent performance P_u(t) is computed
**When** evaluating last 20 feedback items
**Then** recency-weighted average is applied
**And** more recent feedback has higher weight

**Given** authority score changes
**When** threshold is crossed (e.g., 0.5 for 🎓 eligibility)
**Then** user is notified
**And** new capabilities are unlocked

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/authority/score_calculator.py` - Basic scoring
- `Legacy/MERL-T_alpha/merlt/authority/track_record.py` - Historical tracking
- Migration: Implement full formula, add dynamic update

**Technical Notes:**
- Score computed on-demand with caching (1-hour TTL)
- Batch update nightly for computational efficiency
- Store score history for research (Epic 8)

---

### Story 7.2: Router Feedback (F2)

As a **high-authority user (🎓 with score ≥0.7)**,
I want to **evaluate the Expert routing decision**,
So that **the Router learns from expert judgment**.

**Acceptance Criteria:**

**Given** I am 🎓 Contributore with authority ≥0.7
**When** I view a response
**Then** I see a "Valuta Routing" option in advanced section
**And** clicking shows the Router's decision

**Given** I view Router decision
**When** evaluating
**Then** I see:
  - Query classification (DEFINITION|INTERPRETATION|etc.)
  - Expert activation weights
  - Rationale (if available)
**And** I can rate: "Routing appropriato" / "Routing migliorabile"

**Given** I mark "Routing migliorabile"
**When** providing details
**Then** I can suggest:
  - Which Expert(s) should have higher/lower weight
  - Alternative query classification
  - Brief explanation
**And** this becomes F2 feedback

**Given** F2 feedback is submitted
**When** stored
**Then** it is weighted heavily (high-authority only)
**And** Router model receives training signal

**Given** I am below authority threshold
**When** viewing response
**Then** Router feedback option is NOT visible
**And** no message explains why (avoid frustration)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/router.py` - Router logic
- New: Build F2 feedback UI (high-authority only)
- Integration: Connect with Epic 4.2 Router

**Technical Notes:**
- F2 is rare but high-value (expert evaluation)
- Threshold configurable (default 0.7)
- Router retraining uses F2 with highest weight

---

### Story 7.3: Feedback Aggregation

As a **system**,
I want to **aggregate feedback per component with authority weighting**,
So that **training signals reflect community expertise distribution**.

**Acceptance Criteria:**

**Given** multiple users provide feedback for same component
**When** aggregation runs
**Then** weighted average is computed:
  `Agg_c = Σ(feedback_i × authority_i) / Σ(authority_i)`
**And** outliers are flagged but not excluded

**Given** feedback is aggregated by component
**When** I query aggregations
**Then** I can see per-component metrics:
  - NER (F1): precision, recall proxy from corrections
  - Router (F2): routing accuracy
  - Experts (F3-F6): average rating per Expert
  - Synthesizer (F7): overall satisfaction
  - Bridge (F8): source relevance scores

**Given** aggregation runs periodically
**When** batch completes
**Then** results are stored in training buffer
**And** previous aggregations are archived for trend analysis

**Given** conflicting feedback exists
**When** analyzing
**Then** variance is computed
**And** high variance indicates interpretive disagreement (not error)
**And** Devil's Advocate is triggered for high-variance topics

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/rlcf/aggregator.py` - Basic aggregation
- Migration: Add authority weighting, variance analysis

**Technical Notes:**
- Aggregation runs hourly (configurable)
- Store raw feedback + aggregated signals
- High variance → pluralism signal (RLCF Pillar IV)

---

### Story 7.4: Training Buffer & Trigger

As a **system**,
I want to **buffer aggregated feedback and trigger training when threshold is reached**,
So that **model updates are batched for efficiency**.

**Acceptance Criteria:**

**Given** aggregated feedback accumulates
**When** buffer size reaches threshold (default: 100 items)
**Then** training pipeline is triggered
**And** buffer is marked "in training"

**Given** training is triggered
**When** pipeline runs
**Then** the following models can be updated:
  - NER model (from F1)
  - Router weights (from F2)
  - Expert prompts/weights (from F3-F6)
  - Synthesizer prompts (from F7)
  - TraversalPolicy (from F8 - separate Story 7.6)

**Given** training completes
**When** models are updated
**Then** new model versions are stored
**And** old versions are retained for rollback
**And** model version is recorded for reproducibility (NFR-R6)

**Given** training fails
**When** error occurs
**Then** buffer is preserved (not lost)
**And** alert is sent to admin
**And** retry is scheduled

**Given** no training has run for 7 days
**When** buffer has any items
**Then** training is triggered regardless of threshold
**And** ensures freshness

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/rlcf/training_buffer.py` - Buffer management
- `Legacy/MERL-T_alpha/merlt/rlcf/trainer.py` - Training orchestration
- Migration: Add model versioning, rollback capability

**Technical Notes:**
- Training is async (background job)
- Model versioning: timestamp + hash
- Rollback: keep last 5 versions

---

### Story 7.5: Expert Affinity Update (F8c)

As a **system**,
I want to **update Bridge Table expert_affinity weights based on F8 feedback**,
So that **chunk-to-Expert mapping improves over time**.

**Acceptance Criteria:**

**Given** F8 feedback (implicit or explicit) accumulates for a chunk
**When** update threshold is reached (10 feedback items)
**Then** expert_affinity weights are recalculated:
  - Positive F8 + positive Expert rating → increase affinity
  - Negative F8 or negative Expert rating → decrease affinity

**Given** affinity update is computed
**When** applied to Bridge Table
**Then** the update is:
  - Incremental (not replacement)
  - Bounded (min 0.1, max 0.95)
  - Logged for audit

**Given** a chunk consistently receives negative F8
**When** affinity drops below 0.2 for all Experts
**Then** chunk is flagged for review
**And** may indicate bad source or wrong mapping

**Given** a chunk is explicitly rated by 🎓 Contributore
**When** affinity is updated
**Then** explicit feedback weight = 3× implicit
**And** update is more significant

**Given** updates are applied
**When** I query Bridge Table
**Then** I can see affinity history
**And** trace learning over time

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/storage/bridge/bridge_table.py` - Bridge Table
- `Legacy/MERL-T_alpha/merlt/rlcf/affinity_updater.py` - Basic updater
- Migration: Add bounded updates, explicit feedback weighting

**Technical Notes:**
- Update formula: `new = old + lr × (target - old)`
- Learning rate configurable (default 0.1)
- Bounds prevent extreme values

---

### Story 7.6: TraversalPolicy Training (F8d)

As a **system**,
I want to **train TraversalPolicy to learn "path virtuosi" per Expert**,
So that **Knowledge Graph traversal prioritizes valuable connections**.

**Acceptance Criteria:**

**Given** F8 feedback indicates source quality patterns
**When** TraversalPolicy training runs
**Then** it learns:
  - Which edge types are valuable per Expert
  - Optimal traversal depth per query type
  - Path patterns that lead to high-rated sources

**Given** training data is prepared
**When** PolicyGradientTrainer runs
**Then** it optimizes traversal weights:
  - Edge type weights (RIFERIMENTO, CITATO_DA, etc.)
  - Node type preferences (Articolo, Sentenza, Concetto)
  - Depth penalty function

**Given** trained policy is deployed
**When** SystemicExpert traverses the graph
**Then** it uses learned weights to prioritize paths
**And** "path virtuosi" (expert-approved paths) are preferred

**Given** policy is updated
**When** deployed
**Then** A/B testing can compare old vs new policy
**And** regression detection triggers rollback if quality drops

**Given** different Experts need different traversal
**When** policy is queried
**Then** Expert-specific weights are applied:

| Expert | RIFERIMENTO | CITATO_DA | MODIFICA | PRINCIPIO |
|--------|-------------|-----------|----------|-----------|
| Literal | 0.3 | 0.1 | 0.2 | 0.1 |
| Systemic | 0.9 | 0.4 | 0.8 | 0.3 |
| Principles | 0.4 | 0.3 | 0.2 | 0.9 |
| Precedent | 0.2 | 0.9 | 0.3 | 0.4 |

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/traversal_policy.py` - TraversalPolicy ⭐
- `Legacy/MERL-T_alpha/merlt/rlcf/policy_gradient.py` - PolicyGradientTrainer
- Migration: Integrate with F8 feedback loop, add A/B testing

**Technical Notes:**
- TraversalPolicy is RLCF's unique contribution
- "Path virtuosi" = paths that consistently lead to good sources
- This is core thesis innovation (F8 validates bridge quality)

---

## Epic 8: Research & Academic Support

**Goal:** Researchers can validate RLCF framework with reproducible data and basic analytics.

### Story 8.1: Policy Evolution Dashboard

As a **researcher**,
I want to **visualize how RLCF policy weights evolve over time**,
So that **I can validate the learning hypothesis and prepare thesis visualizations**.

**Acceptance Criteria:**

**Given** I am a user with RESEARCH consent or admin
**When** I access the RLCF Dashboard
**Then** I see an overview showing:
  - Total feedback collected (by type)
  - Training runs completed
  - Current model versions
  - System health indicators

**Given** I view the Policy Evolution section
**When** data is displayed
**Then** I see time-series charts for:
  - Expert weights in Gating Network (F3-F6 → weight changes)
  - Router accuracy over time (F2 signal)
  - Average response quality (F7 trend)
  - Bridge affinity distribution (F8 learning)

**Given** I select a specific component (e.g., LiteralExpert)
**When** drilling down
**Then** I see:
  - Weight history with change points
  - Training events marked on timeline
  - Correlation with feedback volume

**Given** I need thesis visualizations
**When** I export a chart
**Then** I can download as:
  - PNG (for slides)
  - SVG (for papers)
  - CSV (underlying data)

**Given** the dashboard loads
**When** rendering
**Then** performance is acceptable (<3s for 6-month data)
**And** data is cached with 1-hour refresh

**Existing Code:**
- New: Build RLCF Dashboard component
- `Legacy/MERL-T_alpha/merlt/rlcf/metrics.py` - Metrics collection
- Integration: Query from training history, feedback aggregations

**Technical Notes:**
- Charts: Recharts or Plotly for React
- MVP: 3-4 key charts, not full analytics suite
- Focus on thesis-relevant visualizations

---

### Story 8.2: Dataset Export

As a **researcher**,
I want to **export anonymized datasets for academic analysis**,
So that **I can perform statistical validation outside the system**.

**Acceptance Criteria:**

**Given** I am authorized for research export
**When** I access the Export section
**Then** I can configure export parameters:
  - Date range
  - Feedback types to include
  - Aggregation level (raw, daily, weekly)
  - Format (JSON, CSV)

**Given** I request an export
**When** the job runs
**Then** all data is anonymized:
  - User IDs → random UUIDs (consistent within export)
  - Query text → removed or heavily anonymized
  - Timestamps → preserved (needed for time series)
  - Authority scores → preserved (key for RLCF analysis)

**Given** export completes
**When** I download
**Then** file includes:
  - Data dictionary (field descriptions)
  - Export metadata (date, parameters, system version)
  - Anonymization confirmation

**Given** I export feedback data
**When** reviewing content
**Then** I can analyze:
  - Feedback distribution by type
  - Authority score distribution
  - Rating patterns per component
  - Time-to-feedback metrics

**Given** I need to cite the dataset
**When** using in paper
**Then** export includes suggested citation format
**And** DOI can be generated (future)

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/export/dataset_exporter.py` - Basic export
- Migration: Add anonymization, configurable parameters

**Technical Notes:**
- Export is async (can take minutes for large datasets)
- Size limit: 100k records per export
- GDPR: only RESEARCH consent data exportable

---

### Story 8.3: Query Reproducibility

As a **researcher**,
I want to **reproduce historical queries with the exact model version used**,
So that **I can validate results and ensure scientific reproducibility (NFR-R6)**.

**Acceptance Criteria:**

**Given** a historical query trace exists
**When** I request reproduction
**Then** the system:
  - Retrieves the model versions used (LLM, embeddings, policy)
  - Loads those specific versions
  - Re-runs the query with same parameters
  - Returns comparison: original vs reproduced

**Given** reproduction runs
**When** comparing outputs
**Then** I see:
  - Side-by-side response comparison
  - Diff highlighting (what changed)
  - Confidence scores comparison
  - Note if exact reproduction or approximate

**Given** the exact model version is unavailable
**When** reproduction is requested
**Then** I see a warning: "Versione esatta non disponibile"
**And** closest available version is offered
**And** differences are documented

**Given** I want to test model improvement
**When** I run reproduction with current model
**Then** I can compare: historical model vs current
**And** quantify improvement/regression

**Given** multiple queries need reproduction
**When** batch processing
**Then** I can submit a list of trace_ids
**And** receive batch comparison report

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/reproduce/reproducer.py` - Basic reproduction
- Integration: Use model versioning from Story 7.4
- Integration: Use trace storage from Story 5.1

**Technical Notes:**
- Model versions stored with each trace
- LLM providers may not support exact reproduction (temperature)
- Document limitations in thesis methodology section

---

### Story 8.4: Devil's Advocate

As a **legal professional (🔍/🎓)**,
I want to **see alternative interpretations for high-consensus responses**,
So that **I'm aware of minority positions and the system preserves interpretive pluralism (RLCF Pillar IV)**.

**Acceptance Criteria:**

**Given** a response has high Expert consensus (all Experts agree)
**When** displayed to 🔍 Analisi or 🎓 Contributore
**Then** a "Devil's Advocate" section is available (collapsed by default)
**And** subtle indicator shows: "⚖️ Posizione alternativa disponibile"

**Given** I expand Devil's Advocate
**When** content is displayed
**Then** I see:
  - Alternative interpretation that contradicts the main response
  - Sources supporting the alternative view
  - Explanation of the interpretive tension
  - Note: "Questa è una posizione minoritaria basata su..."

**Given** Devil's Advocate is generated
**When** analyzing the main response
**Then** the system:
  - Identifies the strongest claim
  - Searches for contradicting precedents/doctrine
  - Constructs counter-argument using same Expert pipeline

**Given** I view Devil's Advocate
**When** I want to evaluate
**Then** I see F13 feedback option:
  - "Posizione valida - utile considerarla"
  - "Posizione debole - non rilevante"
  - "Posizione interessante - richiede approfondimento"

**Given** F13 feedback is collected
**When** aggregated
**Then** Devil's Advocate quality improves
**And** pluralism is preserved (not trained away)

**Given** I am ⚡ Consultazione or 📖 Ricerca
**When** viewing response
**Then** Devil's Advocate is NOT shown
**And** keeps experience simple for basic users

**Existing Code:**
- `Legacy/MERL-T_alpha/merlt/experts/devils_advocate.py` - Basic implementation
- Migration: Integrate with F13 feedback, improve generation quality

**Technical Notes:**
- Devil's Advocate is RLCF Pillar IV implementation
- Prevents "echo chamber" in legal interpretation
- Critical for thesis argument on pluralism preservation
- Generation uses same Experts with "find contradicting evidence" prompt

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

### Story Count by Epic

| Epic | Name | Stories |
|------|------|---------|
| 1 | Foundation & User Identity | 6 |
| 2a | Scraping & URN Pipeline | 4 |
| 2b | Graph Building | 5 |
| 2c | Vector & Bridge Table | 4 |
| 3 | Norm Browsing & Search | 8 |
| 4 | MERL-T Analysis Pipeline | 12 |
| 5 | Traceability & Source Verification | 5 |
| 6 | RLCF Feedback Collection | 10 |
| 7 | Authority & Learning Loop | 6 |
| 8 | Research & Academic Support | 4 |
| **Total MVP** | | **64** |

### Key Metrics

- **Total MVP Stories:** 64 stories
- **Sprint Duration:** 12 sprints (+ 2 buffer + 2 thesis prep)
- **Stories per Sprint:** ~5.3 average
- **Scope:** Libro IV Codice Civile (Obbligazioni + Contratti, ~800 articoli)
- **Post-Thesis:** Epic 9 (Admin) + Epic 10 (API)

### Brownfield Assets Leveraged

| Legacy Path | Stories Using |
|-------------|---------------|
| `Legacy/MERL-T_alpha/merlt/experts/` | 4.2-4.8, 7.6 |
| `Legacy/MERL-T_alpha/merlt/ner/` | 2b.3, 4.1 |
| `Legacy/MERL-T_alpha/merlt/storage/bridge/` | 2c.4, 7.5 |
| `Legacy/MERL-T_alpha/merlt/rlcf/` | 7.1, 7.3-7.6 |
| `Legacy/VisuaLexAPI/api/scraping/` | 2a.1-2a.3 |
| `merlt/merlt/pipeline/multivigenza.py` | 2b.4, 3.6, 5.4 |

### RLCF Feedback Points Coverage

| Feedback | Epic | Stories |
|----------|------|---------|
| F1 (NER) | 6 | 6.2 |
| F2 (Router) | 7 | 7.2 |
| F3-F6 (Experts) | 6 | 6.3 |
| F7 (Synthesizer) | 6 | 6.4 |
| F8 (Bridge) | 6, 7 | 6.5, 6.6, 7.5, 7.6 |
| F13 (Devil's Advocate) | 8 | 8.4 |
