# Story 3-6: Historical Versions Timeline (Backend)

## Status
- **Epic**: Epic 3: Norm Browsing & Search
- **Status**: done
- **Priority**: High

## Context
Backend API for retrieving historical versions of legal norms and comparing them. Extends the existing TemporalQuery module with diff functionality.

## Existing Code
- `visualex/graph/temporal.py` - TemporalQuery with version timeline (from Story 2b-4)
- Tests in `tests/unit/test_graph_temporal.py`

## Acceptance Criteria

### AC1: Version Timeline
**Given** an article has multiple historical versions
**When** querying for version timeline
**Then** return all versions with vigenza_dal dates and modifying legislation

### AC2: Version Comparison (Diff)
**Given** two versions of an article
**When** comparing them
**Then** return word-level diff with additions, deletions, unchanged segments

### AC3: Version at Date
**Given** an article URN and a date
**When** querying with as_of_date
**Then** return the version that was in force at that date

### AC4: Abrogation Info
**Given** an abrogated article
**When** querying history
**Then** show abrogation as final event with abrogating norm linked

## Tasks/Subtasks

- [x] **T1**: Extend `VersionedNorm` with `modifying_norm_urn` and `modifying_norm_title`
- [x] **T2**: Create `DiffSegment` dataclass for diff representation
- [x] **T3**: Create `VersionDiff` dataclass for comparison result
- [x] **T4**: Implement `compare_versions()` method in TemporalQuery
- [x] **T5**: Implement `_compute_word_diff()` helper for word-level diffing
- [x] **T6**: Update `visualex/graph/__init__.py` exports
- [x] **T7**: Write tests for new diff functionality (12 new tests)
- [x] **T8**: All tests passing (32 total)

## Technical Details

### Diff Algorithm
Uses Python's `difflib.SequenceMatcher` for word-level diff:
- Split text into words
- Compare word sequences
- Generate segments with change_type: "unchanged", "added", "removed"

### API Methods
```python
# Get version timeline
timeline = await temporal_query.get_version_timeline(urn)

# Compare two versions
diff = await temporal_query.compare_versions(
    urn="urn:...",
    date_a="2020-01-01",
    date_b="2021-01-01"
)
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/temporal.py` | Modify | Add DiffSegment, VersionDiff, compare_versions |
| `visualex/graph/__init__.py` | Modify | Add new exports |
| `tests/unit/test_graph_temporal.py` | Modify | Add 12 diff tests |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Extended VersionedNorm with modifying_norm fields
- **2026-02-02**: Implemented word-level diff with DiffSegment and VersionDiff
- **2026-02-02**: All 32 tests passing - Story completed
