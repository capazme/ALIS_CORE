# Story 3-3: Search by Keyword/URN

## Status
- **Epic**: Epic 3: Norm Browsing & Search
- **Status**: done
- **Priority**: High

## Context
Implement hybrid search combining semantic (Qdrant) and keyword (FalkorDB full-text) search for legal norms. Support URN pattern matching and temporal filtering.

## Existing Code
- `visualex/graph/qdrant.py` - QdrantCollectionManager with semantic search
- `visualex/graph/schema.py` - FULLTEXT_INDEXES for FalkorDB
- `visualex/graph/client.py` - FalkorDBClient with query method
- `visualex/graph/bridge.py` - BridgeTableManager for chunk→node mapping

## Acceptance Criteria

### AC1: Keyword Search
**Given** a search query
**When** the user submits a keyword search
**Then** results are returned from:
  - Norm text (testo_vigente)
  - Article titles (rubrica/titolo)
  - Brocardi commentary (Dottrina.descrizione)

### AC2: Semantic Search
**Given** a search query
**When** semantic search is enabled
**Then** Qdrant returns semantically similar chunks
**And** chunks are mapped to graph nodes via Bridge Table

### AC3: URN Pattern Search
**Given** a search like "art. 1453 c.c." or "articolo 2043 codice civile"
**When** the pattern is recognized
**Then** the exact article is returned as top result

### AC4: Hybrid Ranking
**Given** results from both keyword and semantic search
**When** results are merged
**Then** ranking considers:
  - Exact matches (highest priority)
  - Full-text relevance score
  - Semantic similarity score
  - Source authority

### AC5: Temporal Filter
**Given** an "as_of_date" parameter
**When** the search is executed
**Then** only norms in force on that date are returned

### AC6: Performance
**Given** any search query
**When** executed
**Then** results return in <500ms (NFR-P1)

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/search.py` module
- [x] **T2**: Implement `HybridSearchService` class
- [x] **T3**: Implement FalkorDB full-text search method
- [x] **T4**: Implement Qdrant semantic search method
- [x] **T5**: Implement URN pattern recognition
- [x] **T6**: Implement hybrid result merging with ranking
- [x] **T7**: Add temporal filtering
- [x] **T8**: Create `SearchResultItem`, `SearchRequest`, `SearchResponse` dataclasses
- [x] **T9**: Update `visualex/graph/__init__.py` exports
- [x] **T10**: Write tests for all ACs (33 tests passing)
- [x] **T11**: Code review - minor fix (unused imports)

## Technical Details

### Hybrid Search Algorithm
```
1. Parse query:
   - Check for URN pattern → direct lookup
   - Otherwise → hybrid search

2. Execute in parallel:
   a. FalkorDB full-text search
   b. Qdrant semantic search → Bridge lookup

3. Merge results:
   - Deduplicate by URN
   - Score = α * fulltext_score + β * semantic_score + γ * authority
   - Sort by final score descending

4. Apply filters:
   - Source type filter
   - Temporal filter (as_of_date)
```

### URN Patterns to Recognize
- `art. 1453 c.c.` → Codice Civile art. 1453
- `articolo 2043 del codice civile`
- `art. 110 c.p.` → Codice Penale
- `d.lgs. 231/2001` → Decreto Legislativo
- `l. 241/1990` → Legge

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/search.py` | Create | HybridSearchService |
| `visualex/graph/__init__.py` | Modify | Add search exports |
| `tests/unit/test_graph_search.py` | Create | Tests for all ACs |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implementation complete, 33 tests passing
