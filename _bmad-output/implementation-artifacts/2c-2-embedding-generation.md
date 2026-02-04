# Story 2c-2: Embedding Generation

## Status
- **Epic**: Epic 2c: Vector & Bridge Table
- **Status**: done
- **Priority**: High

## Context
Enable embedding generation for legal text chunks with configurable models. Support batch processing with progress logging and model versioning for reproducibility (NFR-R6).

## Existing Code
- `merlt/merlt/storage/vectors/embeddings.py` - EmbeddingService singleton with E5-large
- `visualex/graph/chunking.py` - ChunkResult dataclass (Story 2c-1)

## Acceptance Criteria

### AC1: Embedding Generation
**Given** a batch of chunks from various sources
**When** embedding generation runs
**Then** each chunk receives a vector embedding
**And** the embedding model is configurable (e.g., multilingual-e5, BGE-M3)
**And** Italian legal terminology is well-represented

### AC2: Model Configurability
**Given** the embedding model changes
**When** I switch models via configuration
**Then** I can re-embed all content without code changes
**And** model_id is stored with each embedding for audit

### AC3: Batch Processing
**Given** a large batch (~10k chunks)
**When** embedding generation runs
**Then** processing is batched for GPU efficiency
**And** progress is logged
**And** failures are isolated per chunk

### AC4: Embedding Result Metadata
**Given** embeddings are generated
**When** results are returned
**Then** each result includes:
  - `chunk_id`: ID from ChunkResult
  - `embedding`: vector (list of floats)
  - `model_id`: identifier for reproducibility
  - `dimension`: embedding dimension
  - `created_at`: timestamp

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/embeddings.py` module
- [x] **T2**: Implement `EmbeddingConfig` dataclass
- [x] **T3**: Implement `EmbeddingResult` dataclass
- [x] **T4**: Implement `LegalEmbedder` class with configurable models
- [x] **T5**: Add batch embedding with progress logging
- [x] **T6**: Add async wrappers for non-blocking operation
- [x] **T7**: Update `visualex/graph/__init__.py` exports
- [x] **T8**: Write tests for all ACs
- [x] **T9**: Code review

## Technical Details

### Supported Models
| Model | Dimension | Notes |
|-------|-----------|-------|
| intfloat/multilingual-e5-large | 1024 | Default, good for Italian |
| BAAI/bge-m3 | 1024 | Alternative multilingual |
| sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | 384 | Lighter option |

### E5 Prefix Requirements
- Query: `"query: <text>"`
- Document: `"passage: <text>"`

### Environment Variables
- `EMBEDDING_MODEL`: Model name (default: multilingual-e5-large)
- `EMBEDDING_DEVICE`: cpu/cuda (auto-detect)
- `EMBEDDING_BATCH_SIZE`: Batch size (default: 32)

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/embeddings.py` | Created | LegalEmbedder with configurable models |
| `visualex/graph/__init__.py` | Modified | Added embeddings module exports |
| `tests/unit/test_graph_embeddings.py` | Created | Tests for all 4 ACs |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Created embeddings.py with EmbeddingConfig, EmbeddingResult, LegalEmbedder
- **2026-02-02**: Implemented configurable models (E5-large, BGE-M3, MiniLM)
- **2026-02-02**: Added batch processing with progress logging and failure isolation
- **2026-02-02**: Added async wrappers for non-blocking operation
- **2026-02-02**: Updated graph/__init__.py with exports
- **2026-02-02**: Added 34 unit tests - all passing
- **2026-02-02**: Code review fixes:
  - H1: Removed unused `Union` import
  - H2: Moved `_lock` to instance level (was shared across instances)
  - M1: Removed misleading progress logging (was after batch complete)
  - M2: Removed duplicate torch import
  - M3: Removed unused `_actual_device` attribute
  - L1: Added validation for invalid EMBEDDING_BATCH_SIZE env var
  - L2: Added test for batch fallback path
- **2026-02-02**: 36 tests passing after review fixes

