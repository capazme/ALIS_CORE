# Story 2c-1: Text Chunking Pipeline

## Status
- **Epic**: Epic 2c: Vector & Bridge Table
- **Status**: done
- **Priority**: High

## Context
Enable intelligent chunking of legal texts based on source type. Different sources (norms, jurisprudence, commentary, doctrine) require different chunking strategies to capture meaningful semantic units for embedding and retrieval.

## Existing Code
- `merlt/merlt/pipeline/chunking.py` - StructuralChunker for comma-level chunking
- `merlt/merlt/pipeline/parsing.py` - ArticleStructure, Comma dataclasses
- `visualex/graph/ingestion.py` - ArticleParser with structure extraction

## Acceptance Criteria

### AC1: Norm Text Chunking
**Given** norm text (articles, commi)
**When** chunking is applied
**Then** chunks respect legal structure boundaries (comma, lettera, periodo)
**And** chunk size is optimized for embedding model (256-512 tokens)
**And** overlap ensures context continuity (50-100 tokens)

### AC2: Brocardi Commentary Chunking
**Given** Brocardi commentary (longer prose)
**When** chunking is applied
**Then** chunks respect paragraph boundaries
**And** massime are kept as single chunks (self-contained)
**And** case citations are preserved within chunks

### AC3: Source Type Metadata
**Given** any chunk
**When** it's created
**Then** it includes metadata:
  - `source_type`: norm | jurisprudence | commentary | doctrine
  - `source_urn`: canonical URN of parent document
  - `source_authority`: 0.0-1.0 based on source prestige
  - `chunk_position`: location within parent document

### AC4: Batch Processing
**Given** a batch of documents from various sources
**When** chunking is applied
**Then** all documents are processed with appropriate strategy
**And** progress is logged
**And** failures are isolated per document

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/chunking.py` module
- [x] **T2**: Implement `SourceType` enum and `ChunkResult` dataclass
- [x] **T3**: Implement `LegalChunker` with source-type strategies
- [x] **T4**: Add norm chunking strategy (comma-level)
- [x] **T5**: Add commentary chunking strategy (paragraph-level)
- [x] **T6**: Add batch processing with progress logging
- [x] **T7**: Write tests for all chunking strategies
- [x] **T8**: Code review

## Technical Details

### Source Types
```python
class SourceType(str, Enum):
    NORM = "norm"                  # Legislative text
    JURISPRUDENCE = "jurisprudence"  # Court decisions
    COMMENTARY = "commentary"        # Brocardi, commentaries
    DOCTRINE = "doctrine"            # Academic treatises
```

### Source Authority Defaults
| Source Type | Default Authority |
|-------------|-------------------|
| NORM | 1.0 (primary source) |
| JURISPRUDENCE | 0.8 (Cassazione) / 0.6 (other) |
| COMMENTARY | 0.5 |
| DOCTRINE | 0.4 |

### Chunk Metadata Schema
```python
@dataclass
class ChunkResult:
    chunk_id: str
    text: str
    source_type: SourceType
    source_urn: str
    source_authority: float
    chunk_position: int
    token_count: int
    parent_article_urn: Optional[str]
    metadata: Dict[str, Any]
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/chunking.py` | Created | LegalChunker with source-type strategies |
| `visualex/graph/__init__.py` | Modified | Added chunking module exports |
| `tests/unit/test_graph_chunking.py` | Created | 25 tests for all 4 ACs |

### Change Log
- **2026-02-02**: Created chunking.py with SourceType enum, ChunkResult, LegalChunker
- **2026-02-02**: Implemented norm (comma-level), commentary (paragraph), jurisprudence, doctrine strategies
- **2026-02-02**: Added batch processing with progress logging
- **2026-02-02**: Added 25 unit tests - all passing
- **2026-02-02**: Code review fixes:
  - H1: Removed unused CITATION_PATTERN (dead code)
  - H2: Updated _chunk_doctrine docstring (honest about current impl)
  - H3: Removed unused preserve_citations config parameter
  - M1: Added test for batch exception isolation (invalid source_type)
  - M2: Added documentation for TOKENS_PER_WORD magic number
- **2026-02-02**: 26 tests passing after review fixes

