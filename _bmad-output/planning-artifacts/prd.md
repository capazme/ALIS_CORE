---
stepsCompleted: ['step-01-init', 'step-02-discovery', 'step-03-success', 'step-04-journeys', 'step-05-domain', 'step-06-innovation', 'step-07-project-type', 'step-08-scoping', 'step-09-functional', 'step-10-nonfunctional', 'step-11-polish', 'step-12-complete']
workflowStatus: 'complete'
completedAt: '2026-01-24'
inputDocuments:
  - _bmad-output/planning-artifacts/research/technical-vector-space-legal-interpretation-research-2026-01-23.md
  - _bmad-output/analysis/brainstorming-session-2026-01-23.md
  - docs/project-documentation/index.md
  - docs/project-documentation/00-project-overview.md
  - docs/project-documentation/01-architecture.md
  - docs/project-documentation/02-merlt-experts.md
  - docs/project-documentation/03-rlcf.md
workflowType: 'prd'
documentCounts:
  briefs: 0
  research: 1
  brainstorming: 1
  projectDocs: 5
lastStep: 'step-11-polish'
projectType: 'brownfield'
classification:
  projectType: 'saas_b2b'
  secondaryType: 'api_backend'
  domain: 'legaltech'
  complexity: 'high'
  projectContext: 'brownfield'
  keyConcerns:
    - 'Legal ethics and deontology'
    - 'Data retention and confidentiality'
    - 'Academic validation for thesis'
    - 'Attorney-client privilege implications'
---

# Product Requirements Document - ALIS_CORE

**Author:** Gpuzio
**Date:** 2026-01-23

---

## Executive Summary

**ALIS** (Artificial Legal Intelligence System) è una piattaforma di interpretazione giuridica computazionale che implementa i canoni ermeneutici dell'Art. 12 Preleggi come pipeline AI sequenziale.

**Core Innovation:**
- **MERL-T** (Multi-Expert Legal Retrieval Transformer) - 4 Expert sequenziali: Literal → Systemic → Principles → Precedent
- **RLCF** (Reinforcement Learning from Community Feedback) - Estensione di RLHF con authority weighting
- **Living Law Observation** - Prima implementazione computazionale del framework di Ehrlich per osservare il diritto vivente

**Value Proposition:** Tracciabilità completa del ragionamento giuridico, utilizzabile in atti legali.

**Target:** Thesis defense Maggio 2026 | ~20 utenti associazione | 1k+ norme nel Knowledge Graph

**Workflow Integrato:** Browse (VisuaLex) → Analyze (MERL-T) → Feedback (RLCF) → Learn

---

## Success Criteria

### User Success

**Core Value Proposition:** Traceability as justification tool

| Criterion | Description | Metric |
|-----------|-------------|--------|
| **Defensible Reasoning** | User can cite Expert → Sources → Reasoning chain to support a legal position | 100% of responses include complete reasoning trace |
| **Source Verification** | Every statement traceable to verifiable URN source | Zero unsourced statements |
| **Expert Attribution** | User knows which hermeneutic canon (Art. 12) produced which part of the response | Expert contribution visible in UI |
| **Confidence Calibration** | User understands when system is uncertain vs confident | Confidence score accurate (±10% vs user assessment) |

**"Aha!" Moment:** *"I can use this reasoning trace in a legal brief."*

### Academic/Thesis Success

**Original Contribution:** Framework for observing living law (diritto vivente)

| Criterion | Description | Validation |
|-----------|-------------|------------|
| **Novelty Claim** | First computational implementation of a framework for observing living law (Ehrlich) | Literature review confirms gap + reviewer acceptance |
| **Methodological Rigor** | RLCF formally defined as RLHF extension with authority weighting | Peer-reviewed paper or approved thesis chapter |
| **Empirical Validation** | Data (synthetic initially acceptable) demonstrating framework operation | Dataset + metrics + reproducibility |
| **Isomorphism Formalization** | Shortest path ↔ "significato proprio" mathematically formalized | Formal definition + examples |

**Thesis Defense Success:** *The committee recognizes an original contribution in methodology, not just implementation.*

### Technical Success

| Criterion | Target | Priority |
|-----------|--------|----------|
| Expert response latency | <3min (first visit), <500ms (cached) | MVP |
| Knowledge graph coverage | >1k norms (MVP), >10k (Growth) | MVP: 1k |
| Traceability completeness | 100% responses with full trace | MVP |
| RLCF feedback collection | Operational, even with synthetic data | MVP |
| Vector DB recall@10 | >0.85 | Growth |
| Multi-provider LLM fallback | Operational | Growth |

### Business/Community Success

| Criterion | Target | Timeline |
|-----------|--------|----------|
| Working demo | Functional system for thesis defense | May 2026 |
| Genuine interest | 20 active users using spontaneously | Jun 2026 |
| Feedback volume | >100 feedback/month (real or guided synthetic) | Jun 2026 |
| Open source traction | Stars/forks on public repo | Post-thesis |

---

## Product Scope

### MVP - Minimum Viable Product (Thesis Defense)

**Must have for thesis defense:**

1. **Sequential Expert Pipeline** - 4 Experts with Art. 12 hierarchy operational
2. **Full Traceability** - Every response traceable to Expert + Sources + Reasoning
3. **RLCF Collection** - Feedback system operational (synthetic data acceptable)
4. **Working Demo** - Functional UI demonstrating the framework
5. **1k+ Norms** - Knowledge graph with minimal but real corpus
6. **Reproducibility** - Docker Compose + README for replication

**Acceptable for MVP:**
- Synthetic data for RLCF training
- Single LLM provider
- Minimal but functional UI
- ~20 users (association)

### Growth Features (Post-Thesis)

- Multi-provider LLM with fallback
- 10k+ norms in knowledge graph
- Authority scoring with temporal decay
- Complete Devil's Advocate pillar
- Public API for integrations
- External community onboarding

### Vision (Future)

- MERL-T framework exportable to other civil law systems
- SALI-IT taxonomy for Italian law
- Paper publication at academic venue
- Adoption by law firms outside the association

---

## User Journeys

### Journey 1: Avv. Marco Benedetti - Legal Professional (Success Path)

> Marco è un avvocato civilista di 38 anni, socio dell'associazione ALIS. Sta preparando una memoria difensiva su un caso di inadempimento contrattuale.
>
> **Opening Scene:** È le 22:00, la memoria va depositata domani. Marco ha trovato una norma che supporta la sua tesi ma il giudice è noto per chiedere "qual è la base dottrinale?" Una voce interna sussurra: *"Posso davvero fidarmi di un'AI per questo?"*
>
> **Rising Action:** Marco inserisce la sua query in ALIS: "Quali sono i presupposti per la risoluzione ex art. 1453 c.c.?". Il sistema attiva LiteralExpert → SystemicExpert → PrecedentExpert in sequenza.
>
> **Climax:** La risposta arriva con una **traccia completa**: "LiteralExpert ha identificato i presupposti testuali, SystemicExpert ha collegato all'art. 1455 (non scarsa importanza), PrecedentExpert ha trovato Cass. 12345/2020 che conferma l'interpretazione."
>
> **Resolution:** Marco copia la traccia di ragionamento nella memoria: *"Come confermato dall'analisi ermeneutica sequenziale dei canoni ex Art. 12 Preleggi, supportata da Cass. 12345/2020..."*. Il giudice apprezza la ricostruzione metodologica.

**Capabilities:** Query interface, Sequential expert pipeline, Traceable reasoning output, Citation export

### Journey 2: Dott.ssa Elena Ferraro - Legal Academic (Thesis Validation)

> Elena è una dottoranda in Metodologia delle Scienze Giuridiche. Sta scrivendo un capitolo sulla "computabilità dell'interpretazione giuridica".
>
> **Opening Scene:** Ha letto i paper su MERL-T e RLCF ma ha bisogno di **dati empirici**. La sua paura: *la commissione dirà "questo è solo un giocattolo, non scienza."*
>
> **Rising Action:** Elena accede al pannello RLCF di ALIS. Vuole vedere: (1) come il feedback della community ha modificato i pesi degli Expert, (2) esempi di "diritto vivente" emergente dai dati.
>
> **Climax:** Trova un caso interessante: su una questione di interpretazione dell'art. 2043 c.c., il feedback della community ha progressivamente aumentato il peso di PrinciplesExpert rispetto a LiteralExpert, evidenziando un'evoluzione interpretativa. I dati sono statisticamente significativi.
>
> **Resolution:** Elena documenta questo come "prima osservazione empirica di diritto vivente computazionale" nella sua tesi. I dati sono esportabili e reproducibili. La sua paura si trasforma in confidenza.

**Capabilities:** RLCF dashboard, Feedback analytics, Policy evolution visualization, Data export, Reproducibility

### Journey 3: Ing. Paolo Ricci - System Admin (Edge Case: Knowledge Graph Gap)

> Paolo è il referente tecnico dell'associazione ALIS, gestisce l'infrastruttura.
>
> **Opening Scene:** Un utente segnala che ALIS non trova riferimenti su una norma del Codice del Consumo recentemente modificata.
>
> **Rising Action:** Paolo accede al pannello admin. Verifica: (1) la norma non è nel KG, (2) l'ultimo scraping da Normattiva è di 3 settimane fa.
>
> **Climax:** Paolo lancia un ingest manuale per le modifiche recenti al Codice del Consumo. Il sistema processa, crea nodi/edge nel KG, genera embeddings.
>
> **Resolution:** Entro 10 minuti la norma è disponibile. Paolo imposta un alert per future modifiche a quella sezione.

**Capabilities:** Admin dashboard, KG status monitoring, Manual ingest trigger, Scraping management, Alerts

### Journey 4: Prof.ssa Lucia Parisi - High-Authority Reviewer (RLCF Feedback)

> Lucia è professoressa ordinario di Diritto Civile e membro senior dell'associazione con authority score 0.92. Sente il **peso della responsabilità**: il suo feedback plasma il sistema per tutti.
>
> **Opening Scene:** Riceve una notifica: ALIS ha prodotto una risposta su "natura giuridica del contratto di leasing" che ha ricevuto feedback contrastanti.
>
> **Rising Action:** Lucia esamina la risposta, le fonti citate, e il reasoning trace. Nota che SystemicExpert ha dato peso eccessivo a una norma abrogata. Si chiede: *"Se approvo questo, quanti avvocati si affideranno a informazioni sbagliate?"*
>
> **Climax:** Lucia fornisce feedback dettagliato: rating 0.6, commento "L'art. X è stato abrogato dal D.Lgs. Y/2024, l'analisi sistemica va aggiornata."
>
> **Resolution:** Il suo feedback, pesato per authority 0.92, influenza significativamente il policy update. La prossima query simile beneficerà della correzione. Lucia vede il suo contributo nel changelog del sistema.

**Capabilities:** Notification system, Feedback UI with rating + comments, Authority-weighted influence, Policy learning feedback loop, Contribution visibility

### Journey 5: Avv. Marco Benedetti - Error Recovery (Edge Case)

> **Opening Scene:** Marco ha usato ALIS la settimana scorsa per una memoria. In udienza, la controparte ha demolito il suo argomento - il precedente citato da ALIS era stato superato dalla Corte di Cassazione 3 mesi fa.
>
> **Rising Action:** Marco è furioso. Apre ALIS, trova la traccia originale, e vede che PrecedentExpert ha citato Cass. 5678/2019 ma ha mancato l'overruling Cass. 9012/2025. Invia un feedback severo: rating 0.2, "Il precedente era stato superato - errore catastrofico."
>
> **Climax:** Il sistema riconosce il feedback, segnala risposte simili per revisione, e Marco vede che il suo feedback ha contribuito a un policy update entro pochi giorni.
>
> **Resolution:** La fiducia di Marco non è completamente ripristinata, ma ora controlla manualmente la giurisprudenza recente E usa ALIS. Apprezza la trasparenza nel vedere che il suo feedback conta. Il sistema non ha mai preteso di essere infallibile.

**Capabilities:** Error feedback flow, Policy update visibility, Trust recovery UX, Feedback acknowledgment

### Journey 6: Dott. Andrea Corsini - New Member Onboarding

> **Opening Scene:** Andrea è un giovane avvocato appena entrato nell'associazione ALIS. Ha sentito parlare del sistema ma è scettico: *"Un altro chatbot legale?"*
>
> **Rising Action:** Andrea prova una query semplice su art. 2043 c.c. Invece di una risposta generica, vede: "LiteralExpert → SystemicExpert → PrinciplesExpert" con fonti verificabili. Clicca su ogni step per capire la metodologia.
>
> **Climax:** Andrea chiede qualcosa di più complesso. La risposta include incertezza esplicita: "Confidence: 0.65 - giurisprudenza non univoca su questo punto." ALIS non finge di sapere tutto.
>
> **Resolution:** Andrea pensa: *"Questo non è un chatbot - è uno strumento metodologico."* Inizia a usarlo come secondo parere strutturato. Il suo scetticismo si trasforma in rispetto.

**Capabilities:** Onboarding UX, Methodology explanation, Uncertainty display, Trust-building through transparency

### Journey 7: Ing. Sara Marchetti - API Integration Developer

> **Opening Scene:** Sara è una sviluppatrice di uno studio legale partner. Vuole integrare ALIS nel loro document management system.
>
> **Rising Action:** Sara legge la documentazione API, ottiene le credenziali, e fa la prima chiamata a `/api/v1/analyze`. La risposta JSON include `reasoning_trace`, `sources`, `confidence` - tutto strutturato.
>
> **Climax:** Sara costruisce un plugin che mostra un pannello ALIS accanto a ogni documento nel DMS. Gli avvocati dello studio possono fare query senza lasciare il loro workflow.
>
> **Resolution:** Lo studio diventa early adopter. Sara propone miglioramenti all'API che vengono implementati. L'integrazione porta nuovi utenti e feedback.

**Capabilities:** API documentation, Developer experience, JSON response structure, Integration patterns, Feedback from integrators

### Journey Requirements Summary

| Journey | User Type | Focus | Key Capabilities |
|---------|-----------|-------|------------------|
| 1. Marco (Success) | Legal Pro | Traceability for justification | Expert pipeline, trace export |
| 2. Elena (Academic) | Researcher | Validation + thesis data | RLCF analytics, data export |
| 3. Paolo (Admin) | Sys Admin | KG maintenance | Admin tools, ingest, alerts |
| 4. Lucia (Authority) | Senior Expert | Weighted feedback influence | Authority system, visibility |
| 5. Marco (Error) | Legal Pro | Trust recovery after failure | Error feedback, transparency |
| 6. Andrea (New) | New Member | Onboarding + trust building | Methodology UX, uncertainty |
| 7. Sara (API) | Developer | Integration | API, docs, JSON structure |

---

## Domain-Specific Requirements

### Compliance & Regulatory

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| **GDPR Art. 6(1)(a)** | Required | Explicit consent for learning contribution |
| **GDPR Art. 89** | Applicable | Research exemption with appropriate safeguards |
| **GDPR Art. 20** | Required | Data portability with clear definition |
| **Privacy Notice** | Required | Clear notice on: collection, anonymization, purposes |
| **Right to Erasure** | Required | Ability to revoke consent and delete data |

### Confidentiality Framework

| Layer | Protection | Implementation |
|-------|------------|----------------|
| **Query Content** | Anonymized if opt-in, not logged if opt-out | Configurable per user |
| **User Identity** | Pseudonymized (hashed user_id) | One-way hash |
| **Feedback** | Linked to trace_id, not original query | Decoupled storage |
| **Authority Score** | Aggregated, not traceable to single queries | Statistical aggregation |
| **Tenant Isolation** | Data separation for future multi-tenancy | Architecture constraint |

### Consent & Opt-In Model

| Level | What's Collected | Purpose | Default |
|-------|------------------|---------|---------|
| **Basic** | No data beyond session | Basic system use | ✅ Default |
| **Learning** | Anonymized queries + feedback | RLCF training | Opt-in |
| **Research** | Aggregated data for analysis | Thesis/paper | Opt-in |

### Technical Constraints (LegalTech)

| Constraint | Requirement | Priority |
|------------|-------------|----------|
| **Traceability** | 100% reasoning traceable to sources | MVP |
| **Source Verification** | Zero statements without URN | MVP |
| **Temporal Accuracy** | Norms with last update date | MVP |
| **Temporal Versioning** | Historical norm versions queryable | MVP |
| **Anonymization Pipeline** | PII stripping before storage | MVP |
| **Circuit Breakers** | Expert pipeline resilience | MVP |
| **Immutable Audit Trail** | 7-year retention, append-only | MVP |

### Temporal Versioning (Knowledge Graph)

| Aspect | Requirement |
|--------|-------------|
| **Storage** | Timestamped node versions in FalkorDB |
| **Query Support** | `as_of_date` parameter in API |
| **Use Case** | Analyzing contracts from specific dates |
| **Modification Tracking** | Link to modifying legislation |

### Feedback Lifecycle Management

| Aspect | Implementation |
|--------|----------------|
| **Expiration** | Flag feedback older than 24 months for review |
| **Re-validation** | Trigger when cited norm is modified |
| **Toxicity Detection** | Identify feedback that became incorrect |
| **Quarantine** | Exclude questionable feedback from training |

### Expert Pipeline Resilience

| Aspect | Implementation |
|--------|----------------|
| **Low Confidence Threshold** | 0.3 |
| **Propagation Warning** | Flag downstream Experts when upstream confidence < threshold |
| **Fallback Mode** | Skip low-confidence Expert, proceed with others |
| **User Notification** | Show which Experts had issues |

### Quality Assurance

| Requirement | Implementation |
|-------------|----------------|
| **Gold Standard Queries** | 100+ human-verified Q&A pairs |
| **Regression Testing** | Run on every model/policy update |
| **Consistency Threshold** | 95% required to deploy |
| **Adversarial Test Suite** | Test error propagation between Experts |

### Immutable Audit Trail

| Aspect | Specification |
|--------|---------------|
| **What** | Query, response, model_version, timestamp, user_id_hash |
| **Retention** | 7 years (legal document retention standard) |
| **Immutability** | Append-only log, no modifications |
| **Reconstruction** | Ability to replay exact response for any historical query |

### Data Portability (GDPR Art. 20)

| Category | Included in Export |
|----------|-------------------|
| **User's Data** | Query history (if opted in), Feedback submitted, Authority score, Consent preferences |
| **Excluded** | Aggregated model weights, Other users' influenced data |
| **Format** | JSON + human-readable summary |

### Ethical Safeguards

| Risk | Mitigation |
|------|------------|
| **Over-reliance** | Disclaimer + prominent uncertainty display |
| **Hallucination** | Source-grounded responses only |
| **Stale precedents** | Temporal validation + feedback expiration |
| **Authority gaming** | Bias detection + Devil's Advocate |
| **Error propagation** | Circuit breakers between Experts |
| **Constitutional violation** | Immutable principles |

### Legal Disclaimer

> *"ALIS is a methodological support tool for legal interpretation. It does not constitute legal advice. The user is responsible for verifying sources and applying them to the specific case. The ALIS association disclaims all liability for the use of the information provided."*

---

## Innovation & Novel Patterns

### Detected Innovation Areas

#### 1. Methodological Innovation: RLCF (Reinforcement Learning from Community Feedback)

**What's novel:** RLHF treats all feedback equally. RLCF introduces **authority weighting** - feedback from a full professor weighs more than from a student. No one has formalized this in an ML framework before.

**Why it matters:** Applies the legal concept of "source authority" to machine learning.

**Validation approach:** Empirical comparison RLHF vs RLCF on same dataset.

#### 2. Theoretical Innovation: Computational Living Law

**What's novel:** Ehrlich's "Lebendes Recht" (1913) remained a sociological concept for 100+ years. ALIS implements it computationally - legal community feedback dynamically modifies system weights.

**Why it matters:** First observable implementation of "living law" - the thesis can document interpretive evolution in real-time.

**Validation approach:** Time-series analysis of policy weights + correlation with external legal events.

#### 3. Formalization Innovation: Vector Space ↔ Hermeneutic Canons

**What's novel:** The isomorphism "shortest path in embedding space = significato proprio delle parole (Art. 12)" has never been formalized. It bridges computational NLP and legal interpretation theory.

**Why it matters:** Provides a mathematical basis for literal interpretation.

**Validation approach:** Experiments with human judges: "does the shortest path correspond to the most literal interpretation?"

#### 4. Architectural Innovation: Constrained Expert Sequencing

**What's novel:** All MoE (Mixture of Experts) systems use probabilistic routing. ALIS imposes a **fixed sequence** (Literal → Systemic → Principles → Precedent) for Art. 12 compliance. It's an MoE with legal constraints.

**Why it matters:** Demonstrates how normative constraints can guide ML architectures.

**Validation approach:** Ablation study: fixed sequence vs free routing.

### Market Context & Competitive Landscape

| Competitor | Approach | ALIS Differentiator |
|------------|----------|---------------------|
| **Harvey AI** | General-purpose LLM for legal | No methodological traceability |
| **CaseText** | Search + summarization | No structured interpretation |
| **Westlaw Edge** | AI-assisted research | No feedback-based learning |
| **Lexis+ AI** | Document analysis | No authority weighting |

**Gap identified:** No Legal AI system implements civil law hermeneutic canons. All are oriented toward common law (case-based reasoning).

### Validation Approach

| Innovation | Validation Method | Success Metric |
|------------|-------------------|----------------|
| RLCF vs RLHF | A/B comparison | RLCF converges faster with less noise |
| Living law observation | Time-series policy weights | Measurable drift correlating with legal events |
| Vector space isomorphism | Human judge study | >80% agreement on "most literal" = shortest path |
| Constrained sequencing | Ablation study | Fixed sequence produces more defensible reasoning |

### Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| RLCF doesn't outperform RLHF | Thesis claims weakened | Document conditions where it helps (domain expertise matters) |
| Isomorphism doesn't hold | Theoretical contribution invalid | Fall back to "useful heuristic" framing |
| Not enough users for statistical significance | Can't validate living law | Use synthetic feedback with realistic distributions |
| Sequencing constraint hurts performance | Architecture choice questioned | Document tradeoff: compliance vs optimization |

### Fallback Strategies

If innovation doesn't validate:

1. **RLCF fallback:** "Authority weighting is a configurable extension to RLHF for domain-expert contexts"
2. **Isomorphism fallback:** "Vector distance provides a computational proxy for textual interpretation distance"
3. **Living law fallback:** "RLCF enables observation of interpretive drift in feedback patterns"

---

## SaaS B2B Specific Requirements

### Project-Type Overview

ALIS is an **atypical SaaS B2B**: a research platform for the legal community, not a commercial product.

| Aspect | Typical SaaS B2B | ALIS |
|--------|-----------------|------|
| Multi-tenancy | Required | Fork-friendly single tenant |
| RBAC | Complex roles | Authority-based only |
| Subscription | Tiered pricing | Associative membership |
| Integrations | Many | Chat-first, minimal |
| Compliance | Heavy | GDPR + ethics (users aware) |

### Integrated Workflow Model

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORDINARY WORKFLOW                             │
├─────────────────────────────────────────────────────────────────┤
│   1. BROWSE (VisuaLex) - User views norms with annotations      │
│      ↓                                                           │
│   2. ANALYZE (MERL-T) - Sequential Expert pipeline              │
│      ↓                                                           │
│   3. FEEDBACK (RLCF) - User rates, system learns                │
│      ↓                                                           │
│   4. REPEAT - Improved responses next time                       │
└─────────────────────────────────────────────────────────────────┘
```

### Technical Architecture Considerations

#### Tenant Model

```yaml
tenant_model: "fork-friendly-single"
current_deployment: "Single ALIS association instance"
future_model: "Other organizations can fork and deploy"
data_isolation: "Complete separation via fork, not multi-tenant"
```

**Two-Layer Architecture:**

| Layer | Audience | Scope |
|-------|----------|-------|
| **Tool Layer** | Everyone (society, PA, lawyers) | Chat interface + MERL-T analysis |
| **Framework Layer** | Developers/researchers | SDK, APIs, RLCF internals |

#### Permission Model

```yaml
permission_model: "authority-based"
roles:
  - tool_user: "Uses ALIS for queries, no RLCF contribution"
  - rlcf_participant: "Contributes to RLCF learning with authority score"
  - admin: "System administration"
```

No traditional RBAC - authority score determines influence, not role hierarchy.

#### Membership Model

```yaml
membership_model: "associative"
monetization: none
access_control: "Association membership (invitation-based)"
```

### Interface Requirements

**Integrated workflow, NOT standalone chat:**

| Component | Source | MVP Status |
|-----------|--------|------------|
| **Norm Viewer** | VisuaLex patterns (Legacy reference) | Integrate |
| **Chat/Query** | New MERL-T interface | Build |
| **Expert Trace** | New visualization | Build |
| **Feedback UI** | RLCF panel | Build |

#### MVP Interface Flow

```yaml
interface_flow:
  1_browse:
    component: "NormViewer"
    features: ["Search norms", "View with annotations", "Dossier management"]

  2_analyze:
    component: "MerltChat"
    features: ["Context-aware query", "Expert visualization", "Traceable response"]

  3_feedback:
    component: "RlcfPanel"
    features: ["Rating", "Optional comment", "Consent check"]
```

### Consolidation Strategy

```yaml
consolidation:
  alis-ml:
    - merlt (core ML framework)
    - merlt-models (proprietary weights)

  alis-api:
    - visualex-api (scraping)
    - merlt API endpoints

  alis-web:
    - visualex-platform (norm viewer)
    - visualex-merlt (MERL-T integration plugin)
```

### Fork-Friendly Architecture

| Aspect | Requirement |
|--------|-------------|
| **Configuration** | All settings in YAML/env, no hardcoded values |
| **Branding** | Configurable name, logo, colors |
| **Data separation** | Framework code vs trained models |
| **Licensing** | Apache 2.0 (framework), Proprietary (models) |

---

## Project Scoping & Phased Development

### MVP Strategy

**Type:** Validation + Experience MVP

The MVP must achieve two goals:
1. **Academic Validation** - Produce empirical data for thesis defense (May 2026)
2. **User Experience** - Create a tool that generates genuine interest from association members

### MVP Feature Set (Thesis Defense)

| Feature | Priority | Rationale |
|---------|----------|-----------|
| **Sequential Expert Pipeline** | P0 | Core thesis claim - Art. 12 compliance |
| **Full Traceability** | P0 | User value proposition |
| **RLCF Feedback Collection** | P0 | Living law observation capability |
| **Basic Norm Viewer** | P0 | Integrated workflow (not just chat) |
| **1k+ Norms in KG** | P0 | Minimum viable corpus |
| **Docker Compose Deployment** | P0 | Reproducibility for academia |
| **Authority-Weighted Feedback** | P1 | RLCF differentiation from RLHF |
| **Expert Trace Visualization** | P1 | Transparency and trust |
| **Feedback Analytics Dashboard** | P1 | Academic data extraction |

### Phased Roadmap

#### Phase 1: MVP (Feb - May 2026)

**Goal:** Working system for thesis defense

| Milestone | Target Date | Deliverables |
|-----------|-------------|--------------|
| **M1: Core Pipeline** | Feb 2026 | 4 Experts operational, sequential execution |
| **M2: Knowledge Graph** | Mar 2026 | 1k+ norms, FalkorDB + Qdrant integrated |
| **M3: RLCF Basic** | Apr 2026 | Feedback collection, authority scoring |
| **M4: Thesis Demo** | May 2026 | Full system demo, initial data collection |

**Acceptable Constraints:**
- Synthetic data for RLCF training
- Single LLM provider (no fallback)
- Minimal but functional UI
- ~20 users (association only)

#### Phase 2: Growth (Jun - Dec 2026)

**Goal:** Production-ready system with real community data

| Feature | Priority |
|---------|----------|
| Multi-provider LLM with fallback | P1 |
| 10k+ norms in knowledge graph | P1 |
| Complete Devil's Advocate pillar | P2 |
| Authority temporal decay | P2 |
| Public API for integrations | P2 |
| External community onboarding | P2 |

#### Phase 3: Expansion (2027+)

**Goal:** Framework adoption beyond ALIS association

| Feature | Priority |
|---------|----------|
| MERL-T framework for other civil law systems | P3 |
| SALI-IT taxonomy for Italian law | P3 |
| Academic publication | P3 |
| Adoption by external law firms | P3 |

### Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Insufficient real feedback** | Can't validate living law | Synthetic feedback with realistic distributions acceptable for thesis |
| **Thesis timeline pressure** | Feature creep | Strict MVP scope, synthetic data fallback |
| **Association engagement** | No users to test | Commitment secured, ~20 members identified |
| **Technical complexity** | Delays | Leverage existing codebase (Legacy/VisuaLexAPI patterns) |
| **LLM costs** | Budget constraints | Single provider MVP, optimize prompts |

### Success Gate Criteria

**Before Phase 2:**
- [ ] Thesis defended successfully
- [ ] 20+ active users
- [ ] 100+ feedback entries collected
- [ ] Empirical data supports methodology claims

**Before Phase 3:**
- [ ] RLCF paper submitted/accepted
- [ ] API used by at least 1 external integration
- [ ] Framework documented for forking

---

## Functional Requirements

### Query & Legal Analysis

- **FR1:** Legal professional can submit natural language queries about Italian law
- **FR2:** Legal professional can receive structured responses following Art. 12 Preleggi sequence
- **FR3:** Legal professional can view which Expert (Literal, Systemic, Principles, Precedent) contributed each part of a response
- **FR4:** Legal professional can see confidence level for each response
- **FR5:** Legal professional can query about specific norm articles by URN or article number
- **FR6:** Legal professional can query with temporal context ("as of date X")

### Traceability & Source Verification

- **FR7:** User can view complete reasoning trace for any response
- **FR8:** User can navigate from any statement to its source URN
- **FR9:** User can export reasoning trace in citation-ready format
- **FR10:** User can see which sources each Expert consulted
- **FR11:** User can verify temporal validity of cited norms and precedents

### Knowledge Graph & Norm Browsing

- **FR12:** User can browse norms with hierarchical navigation
- **FR13:** User can search norms by keyword, article number, or URN
- **FR14:** User can view norm annotations and cross-references
- **FR15:** User can see historical versions of norms with modification dates
- **FR16:** User can query norms as they existed at a specific date
- **FR17:** Admin can trigger manual ingest of new/modified norms
- **FR18:** System can detect when cited norms have been modified

### RLCF Feedback & Learning

- **FR19:** RLCF participant can rate response quality (numeric scale)
- **FR20:** RLCF participant can provide textual feedback on responses
- **FR21:** RLCF participant can flag specific errors in responses
- **FR22:** User can see their feedback contribution history
- **FR23:** High-authority user can see aggregated impact of their feedback
- **FR24:** Researcher can view policy weight evolution over time
- **FR25:** Researcher can export anonymized feedback analytics
- **FR26:** System can weight feedback by authority score

### User & Authority Management

- **FR27:** New member can onboard with invitation from existing member
- **FR28:** User can view their authority score
- **FR29:** User can understand how authority score is calculated
- **FR30:** Admin can view and adjust authority parameters
- **FR31:** User can configure privacy/consent preferences

### Consent & Data Privacy

- **FR32:** User can choose opt-in level (Basic/Learning/Research)
- **FR33:** User can revoke consent and request data deletion
- **FR34:** User can export personal data (GDPR Art. 20)
- **FR35:** System can anonymize queries before RLCF storage
- **FR36:** System can maintain immutable audit trail

### System Administration

- **FR37:** Admin can monitor knowledge graph coverage and freshness
- **FR38:** Admin can configure scraping schedules for norm sources
- **FR39:** Admin can view system health and Expert pipeline status
- **FR40:** Admin can manage circuit breakers for Expert resilience
- **FR41:** Admin can run regression tests against gold standard queries
- **FR42:** Admin can flag/quarantine problematic feedback

### API & Integration

- **FR43:** Developer can access MERL-T analysis via REST API
- **FR44:** Developer can receive structured JSON with reasoning trace
- **FR45:** Developer can authenticate with API credentials
- **FR46:** Developer can access API documentation

### Academic & Research Support

- **FR47:** Researcher can access RLCF dashboard with policy analytics
- **FR48:** Researcher can reproduce historical queries with same model version
- **FR49:** Researcher can export datasets for academic validation
- **FR50:** Researcher can compare RLCF vs baseline performance metrics

---

## Non-Functional Requirements

### Performance

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-P1:** Norm base data display | <500ms | MVP | Immediate feedback, dati già in KG |
| **NFR-P2:** Expert enrichment (first visit) | <3 min | MVP | Accettabile per analisi complessa, utente vede norma base |
| **NFR-P3:** Expert enrichment (cached) | <500ms | MVP | Retrieval da cache per visite successive |
| **NFR-P4:** Knowledge graph query | <200ms | MVP | Navigazione norme fluida |
| **NFR-P5:** Concurrent users | 20 simultaneous | MVP | Association size |
| **NFR-P6:** Feedback submission | <1s | MVP | Low friction RLCF |
| **NFR-P7:** Cache hit rate | >80% after warm start | Growth | Ottimizzazione costi LLM |

### Caching Strategy

| Aspect | Specification |
|--------|---------------|
| **Cache scope** | Per-norm (shared across users) |
| **Invalidation** | On norm update detection |
| **Warm start** | Pre-compute top 100 norms più richieste |
| **TTL** | Until norm modification detected |
| **Storage** | Redis o equivalente per hot data |

### Progressive Loading UX

```
┌─────────────────────────────────────────┐
│  User richiede Art. 1453 c.c.           │
├─────────────────────────────────────────┤
│  T+0ms:   "Caricamento..."              │
│  T+200ms: Norma base visibile           │
│  T+500ms: Cross-references visibili     │
│  T+30s-3min: Expert analysis complete   │
│           (se non cached)               │
│  T+200ms: Expert analysis (se cached)   │
└─────────────────────────────────────────┘
```

### Security

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-S1:** Data encryption at rest | AES-256 | MVP | GDPR + professional confidentiality |
| **NFR-S2:** Data encryption in transit | TLS 1.3 | MVP | Secure communication |
| **NFR-S3:** Authentication | JWT with rotation | MVP | User identity protection |
| **NFR-S4:** API authentication | API key + rate limiting | Growth | Prevent abuse |
| **NFR-S5:** PII anonymization | Before RLCF storage | MVP | GDPR Art. 6 compliance |
| **NFR-S6:** Audit log integrity | Append-only, tamper-evident | MVP | Legal document standards |
| **NFR-S7:** Consent verification | On every learning-related action | MVP | GDPR Art. 6(1)(a) |

### Reliability

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-R1:** System availability | 99% uptime (excl. maintenance) | MVP | Professional tool reliability |
| **NFR-R2:** Data backup frequency | Daily, 30-day retention | MVP | Recovery capability |
| **NFR-R3:** Expert pipeline degradation | Graceful (skip low-confidence Expert) | MVP | Circuit breaker pattern |
| **NFR-R4:** LLM provider failover | Automatic to backup provider | Growth | Multi-provider resilience |
| **NFR-R5:** Audit trail retention | 7 years, immutable | MVP | Legal document standard |
| **NFR-R6:** Historical query reproducibility | Exact response recreation | MVP | Academic validation |

### Scalability

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-SC1:** User capacity MVP | 50 registered, 20 concurrent | MVP | Association + buffer |
| **NFR-SC2:** User capacity Growth | 500 registered, 100 concurrent | Growth | External community |
| **NFR-SC3:** Knowledge graph size | 10k norms without performance degradation | Growth | Expanded corpus |
| **NFR-SC4:** Feedback volume | 1000 entries/month processable | Growth | RLCF training data |

### Integration

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-I1:** API versioning | Semantic versioning, backward compatibility | Growth | External developers |
| **NFR-I2:** LLM provider abstraction | Switchable without code changes | MVP | Cost optimization |
| **NFR-I3:** Normattiva scraping resilience | Retry with exponential backoff | MVP | Source data freshness |
| **NFR-I4:** Export formats | JSON + CSV for datasets | MVP | Academic data portability |

### Maintainability

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-M1:** Deployment reproducibility | Docker Compose, single command | MVP | Academic replication |
| **NFR-M2:** Configuration externalization | All settings in YAML/env | MVP | Fork-friendly |
| **NFR-M3:** Logging | Structured JSON, queryable | MVP | Debugging + analytics |
| **NFR-M4:** Documentation | API docs auto-generated | Growth | Developer experience |
| **NFR-M5:** Test coverage | >80% for core pipeline | Growth | Regression prevention |

### Compliance

> *Per dettagli implementativi GDPR, vedi sezione "Domain-Specific Requirements > Compliance & Regulatory"*

| Requirement | Target | Priority | Rationale |
|-------------|--------|----------|-----------|
| **NFR-C1:** GDPR Art. 6(1)(a) | Explicit consent for learning | MVP | Legal requirement |
| **NFR-C2:** GDPR Art. 17 | Right to erasure implemented | MVP | Legal requirement |
| **NFR-C3:** GDPR Art. 20 | Data portability in 30 days | MVP | Legal requirement |
| **NFR-C4:** GDPR Art. 89 | Research exemption documented | MVP | Thesis data usage |

