# Story 2c-3: Qdrant Collection Setup

## Status
- **Epic**: Epic 2c: Vector & Bridge Table
- **Status**: done
- **Priority**: High

## Context
Configure Qdrant collections with rich payload schema for legal chunks. Enable filtering by source type and expert affinity boosting for retrieval.

## Existing Code
- `merlt/merlt/storage/retriever/retriever.py` - GraphAwareRetriever uses Qdrant
- `visualex/graph/chunking.py` - SourceType enum (Story 2c-1)
- `visualex/graph/embeddings.py` - LegalEmbedder (Story 2c-2)

## Acceptance Criteria

### AC1: Collection Schema
**Given** Qdrant is running
**When** I initialize the legal_chunks collection
**Then** the schema includes:
  - Vector: embedding dimension (e.g., 1024 for E5-large)
  - Payload fields indexed for filtering:
    - `chunk_id` (string, primary key)
    - `source_urn` (string, indexed)
    - `source_type` (keyword: norm|jurisprudence|commentary|doctrine)
    - `source_authority` (float, 0.0-1.0)
    - `article_urn` (string, indexed) - parent article
    - `text` (string, stored for display)
    - `expert_affinity` (JSON object with per-expert weights)

### AC2: Filtered Queries
**Given** the collection is initialized
**When** I query with filters
**Then** I can filter by source_type (e.g., "only jurisprudence")
**And** I can filter by source_authority threshold
**And** I can boost results by expert_affinity for specific expert

### AC3: Semantic Search
**Given** chunks are inserted
**When** I perform semantic search
**Then** results include full payload for display
**And** search respects HNSW index settings for NFR-P4 (<200ms)

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/qdrant.py` module
- [x] **T2**: Implement `QdrantConfig` dataclass
- [x] **T3**: Implement `QdrantCollectionManager` class
- [x] **T4**: Add collection creation with HNSW parameters
- [x] **T5**: Add payload index configuration
- [x] **T6**: Add point insertion with ChunkResult/EmbeddingResult integration
- [x] **T7**: Add search with filters and expert affinity boosting
- [x] **T8**: Update `visualex/graph/__init__.py` exports
- [x] **T9**: Write tests for all ACs
- [x] **T10**: Code review

## Technical Details

### HNSW Parameters
- `ef_construct`: 128 (build-time accuracy)
- `m`: 16 (connections per node)
- `ef`: 128 (search-time accuracy)

### Payload Schema
```python
{
    "chunk_id": str,           # UUID, indexed
    "source_urn": str,         # URN, indexed
    "source_type": str,        # keyword index
    "source_authority": float, # range filter
    "article_urn": str,        # indexed
    "text": str,               # stored, not indexed
    "expert_affinity": {       # JSON object
        "literal": float,
        "systemic": float,
        "principles": float,
        "precedent": float
    },
    "model_id": str,           # embedding model version
    "created_at": str          # ISO timestamp
}
```

### Default Expert Affinities by Source Type
| source_type | Literal | Systemic | Principles | Precedent |
|-------------|---------|----------|------------|-----------|
| norm | 0.9 | 0.8 | 0.5 | 0.3 |
| jurisprudence | 0.3 | 0.5 | 0.6 | 0.9 |
| commentary | 0.5 | 0.6 | 0.7 | 0.6 |
| doctrine | 0.4 | 0.5 | 0.9 | 0.4 |

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/qdrant.py` | Created | QdrantCollectionManager with schema setup |
| `visualex/graph/__init__.py` | Modified | Added qdrant module exports |
| `tests/unit/test_graph_qdrant.py` | Created | Tests for all 3 ACs |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implementation complete, 23 tests passing with real Qdrant
- **2026-02-02**: Code review fixes applied:
  - H1: Removed unused `HnswConfigDiff` import in search method
  - M1: Fixed docstring to reflect sync API (not async)
  - M2: Added validation for invalid QDRANT_PORT/QDRANT_VECTOR_SIZE env vars
  - M3: Documented that delete_points doesn't verify actual deletion count
  - M4: Fixed SearchResult mutation in expert boost (now creates new objects)
  - L1: Added random seed for test reproducibility
  - L2: Added test for invalid expert_type
  - L3: Removed unused datetime import in tests
- **2026-02-02**: 24 tests passing, story marked done

