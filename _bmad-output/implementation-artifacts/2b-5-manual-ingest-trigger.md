# Story 2b-5: Manual Ingest Trigger

## Status
- **Epic**: Epic 2b: Graph Building
- **Status**: done
- **Priority**: High

## Context
Enable administrators to manually trigger norm ingestion for specific content without waiting for scheduled scraping. This is critical for quickly adding new or modified norms to the Knowledge Graph during development and for operational flexibility.

## Existing Code
- `visualex/graph/ingestion.py` - NormIngester for creating graph nodes
- `visualex/graph/client.py` - FalkorDB client
- `visualex/graph/relations.py` - RelationCreator for extracting citations
- `visualex/scrapers/normattiva.py` - NormattivaScraper for fetching articles
- `visualex/scrapers/brocardi.py` - BrocardiScraper for enrichment
- `visualex/app.py` - Quart API with NormaController

## Acceptance Criteria

### AC1: Trigger Ingest by URN
**Given** I call the ingest endpoint with a valid URN
**When** the process runs
**Then** the article is fetched from source and ingested into the graph
**And** I receive a result indicating success/failure with details

### AC2: Trigger Ingest by Article Range
**Given** I call the ingest endpoint with an article range (e.g., "1470-1490")
**When** the process runs
**Then** all articles in the range are fetched and ingested
**And** I receive per-article results

### AC3: Progress and Error Reporting
**Given** I trigger an ingest operation
**When** the process runs
**Then** I can see which articles succeeded and which failed
**And** failed articles include error details
**And** I can identify partial successes

### AC4: Audit Logging
**Given** an ingest operation completes
**When** I check the logs
**Then** the operation is recorded with timestamp, user, articles processed, and outcome

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/admin.py` module for admin operations
- [x] **T2**: Implement `IngestService` class with single article ingestion
- [x] **T3**: Add article range support
- [x] **T4**: Add audit logging
- [x] **T5**: Create `/admin/ingest` endpoint in app.py
- [x] **T6**: Write tests for ingest service
- [x] **T7**: Code review

## Technical Details

### Endpoint Specification
```
POST /admin/ingest
Content-Type: application/json

Request Body:
{
  "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",  // Single article
  // OR
  "act_type": "codice civile",
  "article_range": "1470-1490",  // Range of articles
  // OR
  "act_type": "codice civile",
  "articles": ["1453", "1454", "1455"]  // List of specific articles
}

Response:
{
  "job_id": "uuid",
  "status": "completed",
  "total": 21,
  "succeeded": 20,
  "failed": 1,
  "results": [
    {"urn": "...", "status": "success", "nodes_created": 5},
    {"urn": "...", "status": "error", "error": "Article not found"}
  ]
}
```

### IngestService Interface
```python
class IngestService:
    async def ingest_by_urn(self, urn: str) -> IngestResult
    async def ingest_range(self, act_type: str, start: int, end: int) -> BatchIngestResult
    async def ingest_articles(self, act_type: str, articles: List[str]) -> BatchIngestResult
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/admin.py` | Created | IngestService, IngestRequest, IngestJobResult classes |
| `visualex/graph/__init__.py` | Modified | Added admin module exports |
| `visualex/app.py` | Modified | Added /admin/ingest endpoint |
| `tests/unit/test_graph_admin.py` | Created | 21 tests for all 4 ACs |

### Change Log
- **2026-02-01**: Created admin.py with IngestService for manual ingestion
- **2026-02-01**: Implemented URN parsing, article range resolution, Brocardi integration
- **2026-02-01**: Added audit logging via logger.info with AUDIT prefix
- **2026-02-01**: Added POST /admin/ingest endpoint in app.py
- **2026-02-01**: Added 19 unit tests - all passing
- **2026-02-02**: Code review fixes:
  - H1: Removed unused BatchResult import
  - H2: Moved `re` import to module level, removed duplicates
  - H3: Added MAX_ARTICLES_PER_REQUEST (100) rate limiting
  - M1: Added TODO for force_refresh implementation
  - M2: Added test for malformed URN handling
  - M3: Made exception handling consistent with IngestionResult
  - L1: Improved validate() docstring
  - L2: Using full UUID for job_id
  - Added 2 new tests (malformed URN, max limit)
  - All 21 tests passing

