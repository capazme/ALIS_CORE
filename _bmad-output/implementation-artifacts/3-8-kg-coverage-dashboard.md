# Story 3-8: KG Coverage Dashboard (Backend)

## Status
- **Epic**: Epic 3: Norm Browsing & Search
- **Status**: done
- **Priority**: Medium

## Context
Backend API for Knowledge Graph statistics and coverage metrics. Provides data for admin dashboard showing system health and coverage gaps.

## Existing Code
- `visualex/graph/client.py` - FalkorDBClient for queries
- `visualex/graph/schema.py` - Node and edge type definitions

## Acceptance Criteria

### AC1: Summary Statistics
**Given** an admin accesses the dashboard
**When** stats are loaded
**Then** show:
- Total norms by type (Codice, Legge, D.Lgs., etc.)
- Total articles, commi
- Total relationships by type
- Last scraping timestamp per source

### AC2: Coverage Metrics
**Given** viewing coverage metrics
**When** stats are displayed
**Then** show:
- Libro IV C.C. coverage percentage
- Articles with/without Brocardi enrichment
- Articles with/without jurisprudence links

### AC3: Identify Gaps
**Given** there are coverage gaps
**When** viewing stats
**Then** can identify which areas need ingestion

### AC4: Performance
**Given** dashboard loads
**When** computing statistics
**Then** response time < 2s with 1-hour cache TTL

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/stats.py` module
- [x] **T2**: Create `NodeTypeCount` and `EdgeTypeCount` dataclasses
- [x] **T3**: Create `CoverageMetrics` dataclass
- [x] **T4**: Create `SourceStatus` dataclass
- [x] **T5**: Create `KGStats` aggregate dataclass
- [x] **T6**: Implement `KGStatsService` class
- [x] **T7**: Implement `get_node_counts()` with fallback
- [x] **T8**: Implement `get_edge_counts()` with fallback
- [x] **T9**: Implement `get_coverage_metrics()`
- [x] **T10**: Implement `get_source_status()`
- [x] **T11**: Add caching with 1-hour TTL
- [x] **T12**: Update `visualex/graph/__init__.py` exports
- [x] **T13**: Write tests (26 tests passing)

## Technical Details

### Caching Strategy
- Cache TTL: 3600 seconds (1 hour)
- Cache invalidation on demand
- Fresh fetch with `use_cache=False`

### Italian Labels
Node types and edge types have Italian display labels for frontend.

### API Methods
```python
service = KGStatsService(falkor_client)

# Get complete stats (uses cache)
stats = await service.get_stats()

# Get fresh stats
stats = await service.get_stats(use_cache=False)

# Individual methods
node_counts = await service.get_node_counts()
edge_counts = await service.get_edge_counts()
coverage = await service.get_coverage_metrics()
sources = await service.get_source_status()

# Invalidate cache
service.invalidate_cache()
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/stats.py` | Create | KGStatsService |
| `visualex/graph/__init__.py` | Modify | Add exports |
| `tests/unit/test_graph_stats.py` | Create | 26 tests |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implemented KGStatsService with all AC criteria
- **2026-02-02**: Added caching with 1-hour TTL
- **2026-02-02**: All 26 tests passing - Story completed
