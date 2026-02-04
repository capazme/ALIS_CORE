# Story 2c-4: Bridge Table Population

## Status
- **Epic**: Epic 2c: Vector & Bridge Table
- **Status**: done
- **Priority**: High

## Context
Populate the Bridge Table with intelligent chunk-to-graph mappings. Experts can traverse from vectors to graph with source-aware preferences.

## Existing Code
- `merlt/merlt/storage/bridge/bridge_table.py` - BridgeTable service with basic operations
- `merlt/merlt/storage/bridge/bridge_builder.py` - BridgeBuilder helper
- `visualex/graph/chunking.py` - SourceType enum (Story 2c-1)
- `visualex/graph/qdrant.py` - QdrantCollectionManager (Story 2c-3)

## Acceptance Criteria

### AC1: Mapping Schema Extension
**Given** chunks in Qdrant and nodes in FalkorDB
**When** the Bridge Builder processes them
**Then** mappings are created with:
  - `chunk_id` -> `graph_node_urn` (primary mapping)
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

### AC2: Multiple Mappings
**Given** a chunk references multiple norms
**When** the Bridge Builder processes it
**Then** multiple mappings are created (one per referenced URN)
**And** mapping_type distinguishes PRIMARY (main topic) from REFERENCE (citation)

### AC3: Expert Query Support
**Given** I need to find chunks for a specific Expert
**When** I query the Bridge Table
**Then** I can filter/sort by that Expert's affinity column
**And** response time is <50ms for single URN lookup

### AC4: Future F8 Feedback Hook (Stub)
**Given** F8 feedback is received (future, from Epic 6)
**When** feedback indicates source quality
**Then** expert_affinity weights can be updated for that mapping
**And** the update is logged for training audit

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/bridge.py` module
- [x] **T2**: Implement `BridgeMapping` dataclass with extended schema
- [x] **T3**: Implement `MappingType` enum
- [x] **T4**: Implement `BridgeTableManager` class
- [x] **T5**: Add expert_affinity auto-computation from source_type
- [x] **T6**: Add insert/upsert methods with batch support
- [x] **T7**: Add query methods with expert affinity filtering
- [x] **T8**: Add update_expert_affinity stub for F8 feedback
- [x] **T9**: Update `visualex/graph/__init__.py` exports
- [x] **T10**: Write tests for all ACs (34 tests passing after code review)
- [x] **T11**: Code review - Fixed 9 issues (3 HIGH, 4 MEDIUM, 2 LOW skipped)

## Technical Details

### Schema Extension
```python
{
    "chunk_id": UUID,              # From Qdrant
    "graph_node_urn": str,         # FalkorDB node URN
    "source_type": str,            # norm|jurisprudence|commentary|doctrine
    "source_authority": float,     # 0.0-1.0
    "mapping_type": str,           # PRIMARY|REFERENCE|CONCEPT|DOCTRINE
    "expert_affinity": {           # JSONB
        "literal": float,
        "systemic": float,
        "principles": float,
        "precedent": float
    },
    "created_at": datetime,
    "updated_at": datetime
}
```

### Indexes
- (graph_node_urn, expert_type) for fast Expert retrieval
- (chunk_id) for reverse lookup
- (source_type) for filtered queries

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/bridge.py` | Created | BridgeTableManager with extended schema |
| `visualex/graph/__init__.py` | Modified | Added bridge module exports |
| `tests/unit/test_graph_bridge.py` | Created | Tests for all 4 ACs |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implementation complete, 25 tests passing with real PostgreSQL
- **2026-02-02**: Code review completed - Fixed 7 issues (H1-H3, M1-M4), 34 tests passing

