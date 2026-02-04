# Story 2b-3: Relation Extraction & Creation

## Status
- **Epic**: Epic 2b: Graph Building
- **Status**: done
- **Priority**: High

## Context
Extract and create relationships between legal norms in the Knowledge Graph. This enables SystemicExpert to traverse connections for legal interpretation, supporting citations, modifications, and jurisprudence links.

## Existing Code
- `Legacy/MERL-T_alpha/merlt/ner/citation_extractor.py` - Citation NER
- `Legacy/MERL-T_alpha/merlt/ner/relation_classifier.py` - Relation type detection
- `Legacy/VisuaLexAPI/frontend/src/utils/citationParser.ts` - Frontend parsing (reference)
- `visualex/graph/ingestion.py` - Node ingestion from Story 2b-2
- `visualex/graph/schema.py` - Edge types (CITA, MODIFICA, etc.)

## Acceptance Criteria

### AC1: Citation Extraction from Article Text
**Given** article text containing citations (e.g., "ai sensi dell'art. 1453")
**When** the NER pipeline processes it
**Then** citations are extracted with source and target URNs
**And** `CITA` edges are created in the graph

### AC2: Modification Relation Detection
**Given** a norm that modifies another (e.g., "L'art. X è sostituito da...")
**When** modification is detected
**Then** appropriate modification edges are created (`sostituisce`, `abroga_totalmente`, etc.)
**And** edges include `data_efficacia` property
**And** reverse traversal is supported

### AC3: Jurisprudence Links from Brocardi
**Given** Brocardi content with Cassazione/court references
**When** jurisprudence links are processed
**Then** `INTERPRETA` edges connect AttoGiudiziario to Norma nodes
**And** edges include citation context

### AC4: Relationship Traversal
**Given** relationships are created
**When** I query neighbors of an article
**Then** I can traverse all relationship types
**And** edge metadata (tipo, data, context) is accessible

## Tasks/Subtasks

- [x] **T1**: Explore existing NER/citation extraction code
- [x] **T2**: Create `visualex/graph/relations.py` module
- [x] **T3**: Implement `CitationExtractor` class for parsing references
- [x] **T4**: Implement `RelationCreator` class for edge creation
- [x] **T5**: Integrate with NormIngester for automatic relation extraction
- [x] **T6**: Write tests for relation extraction and creation (42 tests)
- [x] **T7**: Code review

## Technical Details

### Citation Patterns (Italian Legal)
```
- "ai sensi dell'art. X" → CITA
- "di cui all'articolo X, comma Y" → CITA
- "art. X del codice civile" → CITA
- "L. n. 123/2020" → CITA (to law)
- "Cass. n. 12345/2021" → CITA (to jurisprudence)
```

### Modification Patterns
```
- "è sostituito da" → sostituisce
- "è abrogato" → abroga_totalmente
- "è modificato" → sostituisce (with partial flag)
- "è integrato" → integra
- "è sospeso" → sospende
```

### Edge Schema (from schema.py)
```python
EdgeType.CITA = "cita"           # Citation reference
EdgeType.SOSTITUISCE = "sostituisce"  # Text replacement
EdgeType.ABROGA_TOTALMENTE = "abroga_totalmente"
EdgeType.INTEGRA = "integra"
EdgeType.INTERPRETA = "interpreta"  # Jurisprudence interpretation
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/relations.py` | Created | CitationExtractor, RelationCreator, data structures |
| `visualex/graph/__init__.py` | Modified | Export relation extraction classes |
| `visualex/graph/ingestion.py` | Modified | Integrated RelationCreator for auto-extraction (T5) |
| `tests/unit/test_graph_relations.py` | Created | 42 unit tests for relation extraction |

### Change Log
- **2026-02-01**: Created relation extraction module with:
  - `CitationExtractor` class for parsing Italian legal citations
    - Codice references (c.c., c.p., c.p.c., c.p.p.)
    - Contextual citations (ai sensi, di cui, ex art., dall', all')
    - Law citations (L. n. 123/2020, legge 30 dicembre 2020, n. 178)
    - Decree citations (D.Lgs., D.L., D.P.R., D.M., D.P.C.M.)
    - Jurisprudence (Cass., Corte Cost.)
    - Modification patterns (sostituisce, abroga, integra, sospende, deroga)
  - `RelationCreator` class for creating graph edges
  - Deduplication logic prioritizing citations with resolved URNs
  - 38 tests covering all extraction patterns and edge creation

- **2026-02-01**: Code Review Fixes (42 tests now)
  - **H1 Fixed**: Added `data_efficacia` extraction from modification context (AC2)
  - **H2 Fixed**: Jurisprudence citations now use INTERPRETA edge type (AC3)
  - **H3 Fixed**: Integrated RelationCreator with NormIngester.ingest_article() (T5)
  - **M1 Fixed**: Added traversal metadata tests (AC4)
  - **M2 Fixed**: Removed unused imports (canonicalize_urn, VALID_ACT_TYPES)
  - **M4 Fixed**: Added target_urn extraction from modification context
  - **L1 Fixed**: Removed unused APOSTROPHE_RE constant
  - **L2 Fixed**: Strengthened test assertion for Brocardi relations

---

## Senior Developer Review (AI)

**Date:** 2026-02-01
**Reviewer:** Claude (Code Review Workflow)
**Outcome:** Changes Requested → Fixed

### Issues Found: 9 total (3 High, 4 Medium, 2 Low)

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| H1 | HIGH | AC2: data_efficacia never extracted | ✅ Fixed |
| H2 | HIGH | AC3: INTERPRETA edge uses CITA type | ✅ Fixed |
| H3 | HIGH | T5 marked [x] but not integrated | ✅ Fixed |
| M1 | MEDIUM | AC4: No traversal test | ✅ Fixed |
| M2 | MEDIUM | Unused imports | ✅ Fixed |
| M3 | MEDIUM | RelationType duplication | ⏳ Deferred (design decision) |
| M4 | MEDIUM | Modifications never resolve target_urn | ✅ Fixed |
| L1 | LOW | Unused APOSTROPHE_RE constant | ✅ Fixed |
| L2 | LOW | Test assertion too weak | ✅ Fixed |

### Test Results After Fixes
- **42 tests passing** (was 38)
- New tests: data_efficacia extraction, target_urn extraction, traversal metadata, strengthened assertions

