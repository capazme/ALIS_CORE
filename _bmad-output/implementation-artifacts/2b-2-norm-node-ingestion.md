# Story 2b-2: Norm Node Ingestion

## Status
- **Epic**: Epic 2b: Graph Building
- **Status**: done
- **Priority**: High

## Context
Ingest scraped norms from Epic 2a into the Knowledge Graph as nodes. This creates the foundation for legal corpus navigation and querying.

## Existing Code
- `visualex/models/norma.py` - Norma, NormaVisitata data models
- `visualex/scrapers/` - Normattiva, Brocardi, EUR-Lex scrapers
- `visualex/graph/` - FalkorDB client and schema (from Story 2b-1)
- `Legacy/MERL-T_alpha/merlt/pipeline/ingestion.py` - Reference ingestion logic

## Acceptance Criteria

### AC1: Node Creation from Scraped Content
**Given** scraped content from Epic 2a (NormaVisitata format)
**When** the ingestion pipeline processes it
**Then** nodes are created for:
- `Norma` (the legal document/article with URN, titolo, testo_vigente)
- `Comma` (paragraph-level divisions within articles)
- `Lettera` (lettered sub-points a), b), c))
- `Numero` (numbered sub-sub-points)
**And** hierarchical relationships are established via `contiene` edges
**And** each node has its canonical URN as primary identifier

### AC2: Idempotent Re-ingestion (MERGE)
**Given** an article already exists in the graph
**When** new content is ingested for the same URN
**Then** the existing node is updated (not duplicated)
**And** a `data_versione` timestamp is recorded
**And** modifications are tracked via `Versione` nodes for temporal queries

### AC3: Batch Processing Performance
**Given** a batch of ~800 articles (Libro IV Codice Civile scale)
**When** bulk ingestion runs
**Then** all nodes are created within acceptable time (<5 min for full batch)
**And** progress is logged for monitoring
**And** failures are isolated (one bad article doesn't stop batch)

### AC4: Brocardi Enrichment Integration
**Given** Brocardi enrichment data is available
**When** ingestion processes the article
**Then** `Dottrina` nodes are created for ratio/spiegazione
**And** `AttoGiudiziario` nodes are created for massime
**And** appropriate edges connect them to the Norma

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/ingestion.py` module
- [x] **T2**: Implement `NormIngester` class with MERGE logic
- [x] **T3**: Implement `ArticleParser` for comma/lettera/numero extraction
- [x] **T4**: Implement batch processing with progress logging
- [x] **T5**: Add Brocardi enrichment integration
- [x] **T6**: Write tests for ingestion operations (35 tests)
- [x] **T7**: Code review

## Technical Details

### Ingestion Flow
```
NormaVisitata + article_text + brocardi_info
        ↓
    NormIngester.ingest()
        ↓
    ┌───────────────────┐
    │ 1. Parse article  │ → ArticleStructure (commas, lettere, numeri)
    │ 2. Build URNs     │ → Canonical URNs for each element
    │ 3. MERGE nodes    │ → FalkorDB MERGE (upsert)
    │ 4. Create edges   │ → contiene relationships
    │ 5. Add enrichment │ → Dottrina, AttoGiudiziario nodes
    └───────────────────┘
        ↓
    IngestionResult (nodes_created, edges_created)
```

### URN Hierarchy
```
urn:nir:stato:regio.decreto:1942-03-16;262              # Codice Civile
urn:nir:stato:regio.decreto:1942-03-16;262~art1453     # Article 1453
urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1  # Comma 1
urn:nir:stato:regio.decreto:1942-03-16;262~art1453-com1-leta  # Lettera a)
```

### Batch Processing
```python
# Process in batches of 100 for transaction efficiency
BATCH_SIZE = 100

async def ingest_batch(articles: List[VisualexArticle]) -> BatchResult:
    results = []
    for batch in chunked(articles, BATCH_SIZE):
        async with client.transaction():
            for article in batch:
                try:
                    result = await ingest_article(article)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed: {article.urn}: {e}")
                    results.append(IngestionError(article.urn, e))
    return BatchResult(results)
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/ingestion.py` | Created | NormIngester, ArticleParser, data structures |
| `visualex/graph/__init__.py` | Modified | Export ingestion classes |
| `tests/unit/test_graph_ingestion.py` | Created | 31 unit tests for ingestion module |

### Change Log
- **2026-02-01**: Created ingestion module with NormIngester and ArticleParser
  - ArticleParser extracts comma/lettera/numero structure from article text
  - NormIngester handles MERGE operations for idempotent ingestion
  - BatchResult provides batch processing with isolated failures
  - Brocardi enrichment creates Dottrina and AttoGiudiziario nodes
  - 31 tests passing covering parser, ingester, and integration scenarios

- **2026-02-01**: Code Review Fixes (35 tests now)
  - **H1 Fixed**: Added Versione node creation for temporal tracking (AC2)
  - **H3 Fixed**: nodes_updated now properly tracks updated vs created nodes
  - **M1 Fixed**: Changed `callable` to `Callable[[int, int, str], None]` type hint
  - **M2 Fixed**: Moved `import time` to top of file
  - **M3 Fixed**: Added tests for URN without article number edge case
  - **L2 Fixed**: Replaced magic number 50 with `PROGRESS_LOG_INTERVAL` constant
  - Note: H2 (batch transactions) deferred - FalkorDB client doesn't expose transaction API

---

## Senior Developer Review (AI)

**Date:** 2026-02-01
**Reviewer:** Claude (Code Review Workflow)
**Outcome:** Changes Requested → Fixed

### Issues Found: 9 total (3 High, 4 Medium, 2 Low)

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| H1 | HIGH | AC2: Versione nodes not created | ✅ Fixed |
| H2 | HIGH | Batch transaction missing | ⏳ Deferred (API limitation) |
| H3 | HIGH | nodes_updated never incremented | ✅ Fixed |
| M1 | MEDIUM | callable type hint incorrect | ✅ Fixed |
| M2 | MEDIUM | import time inside function | ✅ Fixed |
| M3 | MEDIUM | Missing test for URN without article | ✅ Fixed |
| M4 | MEDIUM | Parser doesn't extract rubrica | ⏳ Deferred (future story) |
| L1 | LOW | Unused data_versione parameter | Accepted |
| L2 | LOW | Magic number for progress logging | ✅ Fixed |

### Test Results After Fixes
- **35 tests passing** (was 31)
- New tests: URN edge cases, nodes_updated tracking, Versione creation

