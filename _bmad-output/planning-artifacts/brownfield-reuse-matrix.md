# Brownfield Reuse Matrix

**Project:** ALIS_CORE
**Date:** 2026-01-25
**Purpose:** Map existing components to epic stories, identify reuse vs build requirements

---

## Executive Summary

| Category | Count | % of Stories |
|----------|-------|--------------|
| **Reuse** (minimal changes) | 18 | 28% |
| **Adapt** (significant refactoring) | 31 | 48% |
| **Build** (from scratch) | 15 | 24% |
| **Total Stories** | 64 | 100% |

**Key Finding:** ~76% of stories can leverage existing code (reuse + adapt), significantly reducing implementation effort.

---

## Repository Structure

```
ALIS_CORE/
├── merlt/                    # ✅ ACTIVE - ML Framework (FastAPI)
│   └── merlt/
│       ├── api/              # API routers, models
│       ├── experts/          # 4 Experts + Router + Gating + Synthesizer
│       ├── rlcf/             # RLCF framework
│       ├── pipeline/         # Enrichment, chunking
│       ├── ner/              # NER models
│       └── benchmark/        # Gold standard
│
├── visualex-api/             # ✅ ACTIVE - Scraping Library (Quart)
│   └── visualex/
│       ├── scrapers/         # Normattiva, Brocardi, EUR-Lex
│       ├── models/           # Norma models
│       └── utils/            # URN, cache, circuit breaker
│
├── visualex-platform/        # ✅ ACTIVE - Web Platform
│   ├── frontend/src/         # React 19 + Vite
│   └── backend/              # Express 5 (auth, CRUD)
│
├── visualex-merlt/           # ✅ ACTIVE - Integration Plugin
│
└── Legacy/                   # 📦 ARCHIVED - Reference only
    ├── MERL-T_alpha/         # Old ML code, archived experiments
    └── VisuaLexAPI/          # Old API structure
```

---

## Epic-by-Epic Component Mapping

### Epic 1: Foundation & User Identity

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 1.1 User Registration | **Adapt** | `visualex-platform/backend/` (Express auth) | Add invitation system, JWT rotation |
| 1.2 User Login | **Reuse** | `visualex-platform/backend/` (JWT auth exists) | Minor: add refresh token logic |
| 1.3 Profile Setup | **Build** | None - 4-Profile System is new | Build from scratch, integrate with UX spec |
| 1.4 Consent Configuration | **Build** | None - GDPR consent system is new | Build consent middleware, audit log |
| 1.5 Authority Score Display | **Adapt** | `merlt/rlcf/` (authority scoring exists) | Add UI component, expose via API |
| 1.6 Data Export & Erasure | **Build** | None - GDPR features new | Build export job, soft delete logic |

**Epic 1 Summary:** 2 Reuse, 2 Adapt, 2 Build

---

### Epic 2a: Scraping & URN Pipeline

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 2a.1 Normattiva Scraper | **Reuse** | `visualex-api/visualex/scrapers/normattiva.py` | Configure for Libro IV scope |
| 2a.2 Brocardi Scraper | **Reuse** | `visualex-api/visualex/scrapers/brocardi.py` | Verify output format |
| 2a.3 EUR-Lex Scraper | **Reuse** | `visualex-api/visualex/scrapers/eurlex.py` | Configure for relevant directives |
| 2a.4 URN Canonicalization | **Reuse** | `visualex-api/visualex/utils/urngenerator.py` | Consolidate into single service |

**Epic 2a Summary:** 4 Reuse, 0 Adapt, 0 Build

---

### Epic 2b: Graph Building

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 2b.1 Graph Schema Setup | **Adapt** | `merlt/merlt/storage/` (FalkorDB client exists) | Validate schema, add missing node types |
| 2b.2 Norm Node Ingestion | **Adapt** | `merlt/pipeline/enrichment/` | Connect to scraper output |
| 2b.3 Relation Extraction | **Adapt** | `merlt/pipeline/enrichment/extractors/` | Tune for Libro IV relations |
| 2b.4 Temporal Versioning | **Build** | None - versioning logic new | Build temporal node management |
| 2b.5 Manual Ingest Trigger | **Adapt** | `merlt/api/document_router.py` | Add admin endpoint |

**Epic 2b Summary:** 0 Reuse, 4 Adapt, 1 Build

---

### Epic 2c: Vector & Bridge Table

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 2c.1 Text Chunking | **Reuse** | `merlt/pipeline/semantic_chunking/` | Use existing chunkers |
| 2c.2 Embedding Generation | **Reuse** | `merlt/pipeline/` (embeddings exist) | Configure for legal domain |
| 2c.3 Qdrant Collection | **Adapt** | `merlt/` (Qdrant integration exists) | Add expert_affinity field |
| 2c.4 Bridge Table Population | **Build** | None - Bridge Table is new | Build chunk↔graph mapping logic |

**Epic 2c Summary:** 2 Reuse, 1 Adapt, 1 Build

---

### Epic 3: Norm Browsing & Search

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 3.1 Hierarchical Browser | **Adapt** | `visualex-platform/frontend/src/components/features/search/TreeViewPanel.tsx` | Adapt for Libro IV structure |
| 3.2 Article Viewer | **Reuse** | `visualex-platform/frontend/src/components/features/workspace/` | Already exists, polish |
| 3.3 Search by Keyword/URN | **Adapt** | `visualex-platform/frontend/src/components/features/search/GlobalSearch.tsx` | Add URN parsing |
| 3.4 Citation Highlighting | **Reuse** | `visualex-platform/frontend/src/utils/citationParser.ts`, `citationMatcher.ts` | Already exists |
| 3.5 Cross-References Panel | **Adapt** | Graph view components exist | Add cross-ref display |
| 3.6 Historical Versions | **Build** | None - temporal UI new | Build timeline component |
| 3.7 Modification Detection | **Adapt** | Scraper has change detection | Add UI notification |
| 3.8 KG Coverage Dashboard | **Build** | None - admin dashboard new | Build dashboard component |

**Epic 3 Summary:** 2 Reuse, 4 Adapt, 2 Build

---

### Epic 4: MERL-T Analysis Pipeline

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 4.1 NER Pipeline | **Reuse** | `merlt/ner/spacy_model.py`, `training.py` | Already trained, integrate |
| 4.2 Expert Router | **Reuse** | `merlt/experts/router.py`, `neural_gating/hybrid_router.py` | Already implemented |
| 4.3 LiteralExpert | **Reuse** | `merlt/experts/literal.py` | Already implemented |
| 4.4 SystemicExpert | **Reuse** | `merlt/experts/systemic.py` | Already implemented |
| 4.5 PrinciplesExpert | **Reuse** | `merlt/experts/principles.py` | Already implemented |
| 4.6 PrecedentExpert | **Reuse** | `merlt/experts/precedent.py` | Already implemented |
| 4.7 Gating Network | **Reuse** | `merlt/experts/gating.py`, `neural_gating/neural.py` | Already implemented |
| 4.8 Synthesizer | **Reuse** | `merlt/experts/synthesizer.py` | Already implemented |
| 4.9 Circuit Breaker | **Adapt** | `visualex-api/visualex/utils/circuit_breaker.py` | Adapt for Expert pipeline |
| 4.10 LLM Abstraction | **Adapt** | `merlt/rlcf/ai_service.py` | Add provider failover |
| 4.11 Expert Status UI | **Adapt** | Plugin slots exist | Add status panel |
| 4.12 Gold Standard | **Reuse** | `merlt/benchmark/gold_standard.py`, `metrics.py` | Already implemented |

**Epic 4 Summary:** 10 Reuse, 2 Adapt, 0 Build ⭐

---

### Epic 5: Traceability & Source Verification

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 5.1 Reasoning Trace Storage | **Adapt** | `merlt/rlcf/persistence.py` (traces exist) | Ensure full trace storage |
| 5.2 Trace Viewer UI | **Build** | None - Expert Accordion UI new | Build per UX spec |
| 5.3 Source Navigation | **Adapt** | Citation linking exists | Add Peek Definition |
| 5.4 Temporal Validity Check | **Adapt** | Graph has timestamps | Add validity check logic |
| 5.5 Citation Export | **Adapt** | `visualex-platform/frontend/src/components/ui/AdvancedExportModal.tsx` | Add citation format |

**Epic 5 Summary:** 0 Reuse, 4 Adapt, 1 Build

---

### Epic 6: RLCF Feedback Collection

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 6.1 Feedback Data Model | **Adapt** | `merlt/rlcf/persistence.py`, `merlt/api/feedback_api.py` | Extend for F1-F8 |
| 6.2 NER Feedback (F1) | **Adapt** | NER exists | Add inline feedback UI |
| 6.3 Expert Feedback (F3-F6) | **Adapt** | Feedback API exists | Add per-Expert rating |
| 6.4 Synthesizer Feedback (F7) | **Adapt** | Feedback API exists | Add usability assessment |
| 6.5 Bridge Quality (F8) | **Build** | None - F8 is new | Build source relevance rating |
| 6.6 Fonti Usate Panel | **Build** | None - Contributore UI new | Build per UX spec |
| 6.7 Feedback History | **Adapt** | `merlt/api/rlcf_router.py` | Add user history endpoint |
| 6.8 PII Anonymization | **Adapt** | Some logic exists | Ensure GDPR compliance |
| 6.9 Audit Trail | **Build** | None - immutable log new | Build append-only storage |
| 6.10 Synthetic Feedback | **Reuse** | `merlt/rlcf/simulator/feedback_synthesizer.py` | Already exists |

**Epic 6 Summary:** 1 Reuse, 6 Adapt, 3 Build

---

### Epic 7: Authority & Learning Loop

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 7.1 Authority Computation | **Adapt** | `merlt/rlcf/authority_sync.py` | Validate formula, expose API |
| 7.2 Router Feedback (F2) | **Build** | None - Router feedback new | Build high-authority feedback |
| 7.3 Feedback Aggregation | **Adapt** | `merlt/rlcf/aggregation.py` | Add per-component aggregation |
| 7.4 Training Buffer | **Adapt** | Training pipeline exists | Add threshold trigger |
| 7.5 Expert Affinity (F8c) | **Build** | None - Bridge learning new | Build affinity update logic |
| 7.6 TraversalPolicy (F8d) | **Build** | None - PolicyGradient new | Build training loop |

**Epic 7 Summary:** 0 Reuse, 3 Adapt, 3 Build

---

### Epic 8: Research & Academic Support

| Story | Status | Existing Component | Action Required |
|-------|--------|-------------------|-----------------|
| 8.1 Policy Evolution Dashboard | **Adapt** | `merlt/experts/policy_metrics.py` | Add visualization |
| 8.2 Dataset Export | **Adapt** | Export logic partial | Add GDPR-compliant export |
| 8.3 Query Reproducibility | **Adapt** | Checkpointing exists | Add version pinning |
| 8.4 Devil's Advocate | **Adapt** | `merlt/disagreement/` (detector, explainer exist) | Integrate with UI |

**Epic 8 Summary:** 0 Reuse, 4 Adapt, 0 Build

---

## Summary by Status

### Reuse (18 Stories) - Minimal Changes

| Epic | Stories |
|------|---------|
| 2a | 2a.1, 2a.2, 2a.3, 2a.4 (all 4) |
| 2c | 2c.1, 2c.2 |
| 3 | 3.2, 3.4 |
| 4 | 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.12 (9 of 12) |
| 6 | 6.10 |

### Adapt (31 Stories) - Significant Refactoring

Most stories in Epics 1, 2b, 3, 5, 6, 7, 8 require adaptation of existing components.

### Build (15 Stories) - From Scratch

| Epic | Stories | Reason |
|------|---------|--------|
| 1 | 1.3, 1.4, 1.6 | 4-Profile System, GDPR consent, data export are new features |
| 2b | 2b.4 | Temporal versioning logic new |
| 2c | 2c.4 | Bridge Table is new concept |
| 3 | 3.6, 3.8 | Historical timeline UI, KG dashboard new |
| 5 | 5.2 | Expert Accordion UI per UX spec |
| 6 | 6.5, 6.6, 6.9 | F8 feedback, Fonti panel, audit trail new |
| 7 | 7.2, 7.5, 7.6 | F2 feedback, F8c/F8d training loops new |

---

## Key Existing Components Inventory

### merlt/ - Core ML Framework

| Component | Path | Status | Coverage |
|-----------|------|--------|----------|
| **4 Experts** | `merlt/experts/{literal,systemic,principles,precedent}.py` | ✅ Complete | Epic 4 |
| **Router** | `merlt/experts/router.py` | ✅ Complete | Epic 4 |
| **Gating** | `merlt/experts/gating.py` | ✅ Complete | Epic 4 |
| **Synthesizer** | `merlt/experts/synthesizer.py` | ✅ Complete | Epic 4 |
| **NER** | `merlt/ner/` | ✅ Complete | Epic 4 |
| **RLCF Core** | `merlt/rlcf/` | ⚠️ Partial | Epic 6, 7 |
| **Pipeline** | `merlt/pipeline/` | ✅ Complete | Epic 2b, 2c |
| **Benchmark** | `merlt/benchmark/` | ✅ Complete | Epic 4 |
| **Disagreement** | `merlt/disagreement/` | ✅ Complete | Epic 8 |

### visualex-api/ - Scraping Library

| Component | Path | Status | Coverage |
|-----------|------|--------|----------|
| **Normattiva** | `visualex/scrapers/normattiva.py` | ✅ Complete | Epic 2a |
| **Brocardi** | `visualex/scrapers/brocardi.py` | ✅ Complete | Epic 2a |
| **EUR-Lex** | `visualex/scrapers/eurlex.py` | ✅ Complete | Epic 2a |
| **URN Generator** | `visualex/utils/urngenerator.py` | ✅ Complete | Epic 2a |
| **Circuit Breaker** | `visualex/utils/circuit_breaker.py` | ✅ Complete | Epic 4 |

### visualex-platform/ - Web UI

| Component | Path | Status | Coverage |
|-----------|------|--------|----------|
| **Auth** | `frontend/src/components/auth/` | ✅ Complete | Epic 1 |
| **Search** | `frontend/src/components/features/search/` | ✅ Complete | Epic 3 |
| **Workspace** | `frontend/src/components/features/workspace/` | ✅ Complete | Epic 3, 5 |
| **Command Palette** | `frontend/src/components/features/search/CommandPalette.tsx` | ✅ Complete | Epic 3 |
| **Citation Parser** | `frontend/src/utils/citationParser.ts` | ✅ Complete | Epic 3, 5 |
| **Export Modal** | `frontend/src/components/ui/AdvancedExportModal.tsx` | ✅ Complete | Epic 5, 8 |

---

## Risk Assessment

### Low Risk (Well-Tested Existing Code)
- Epic 4: MERL-T Pipeline - **90% reuse**
- Epic 2a: Scrapers - **100% reuse**

### Medium Risk (Requires Careful Integration)
- Epic 2b, 2c: Data layer - existing components need coordination
- Epic 3: UI - existing but needs UX polish
- Epic 6, 7: RLCF - core exists but F8 is new

### High Risk (New Development)
- Epic 1: 4-Profile System, GDPR consent - new features, legal compliance
- Epic 7: TraversalPolicy training - new ML component
- Epic 5: Expert Accordion UI - new UX pattern per spec

---

## Recommendations

1. **Start with Epic 4** - Highest reuse (90%), validates core pipeline
2. **Epic 2a in parallel** - Pure reuse, unblocks Epic 2b/2c
3. **Epic 1 early** - Foundation needed, but high build %
4. **Defer Epic 7.5/7.6** - F8c/F8d are complex ML, can iterate post-MVP

---

*Generated by BMM Brownfield Analysis*
