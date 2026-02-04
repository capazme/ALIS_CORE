# Story 2b-4: Temporal Versioning

## Status
- **Epic**: Epic 2b: Graph Building
- **Status**: done
- **Priority**: High

## Context
Enable querying norms as they existed at a specific date. This is crucial for legal analysis of contracts and disputes where the applicable law is determined by the date of the transaction, not the current date. Italian legal texts undergo frequent modifications (multivigenza), and users need to access historical versions.

## Existing Code
- `visualex/graph/schema.py` - Graph schema with Versione node type
- `visualex/graph/ingestion.py` - Creates Versione nodes with data_inizio/data_fine
- `visualex/graph/client.py` - FalkorDB client
- `visualex/models/norma.py` - Norma, NormaVisitata with data_versione

## Acceptance Criteria

### AC1: Query with as_of_date Parameter
**Given** an article with multiple historical versions
**When** I query with `as_of_date` parameter
**Then** I receive the version that was in force on that date
**And** the response indicates vigenza_dal and vigenza_al dates

### AC2: Pre-modification Version Retrieval
**Given** an article was modified on 2024-03-01
**When** I query as_of_date=2024-02-15
**Then** I get the pre-modification version
**And** I can see that a newer version exists

### AC3: Abrogated Article Handling
**Given** an article was abrogated
**When** I query without as_of_date
**Then** I see the article marked as "abrogato"
**And** I can query historical versions before abrogation

### AC4: Version Timeline Access
**Given** an article has version history
**When** I request the version timeline
**Then** I receive all versions with their validity periods
**And** versions are ordered chronologically

## Tasks/Subtasks

- [x] **T1**: Analyze existing Versione node structure
- [x] **T2**: Create `visualex/graph/temporal.py` module
- [x] **T3**: Implement `TemporalQuery` class with as_of_date support
- [x] **T4**: Add version timeline retrieval method
- [x] **T5**: Add abrogation status detection
- [x] **T6**: Write tests for temporal queries
- [x] **T7**: Code review

## Technical Details

### Versione Node Schema (from ingestion.py)
```python
# Versione nodes track temporal changes
Versione:
  - urn: str  # Parent article URN
  - data_inizio: date  # When this version became effective
  - data_fine: Optional[date]  # When superseded (null if current)
  - testo_vigente: str  # Text content of this version
  - tipo_modifica: str  # sostituisce, integra, abroga, etc.
```

### Query Examples
```cypher
# Get version valid on specific date
MATCH (n:Norma {urn: $urn})-[:ha_versione]->(v:Versione)
WHERE v.data_inizio <= $as_of_date
  AND (v.data_fine IS NULL OR v.data_fine > $as_of_date)
RETURN v

# Get all versions timeline
MATCH (n:Norma {urn: $urn})-[:ha_versione]->(v:Versione)
RETURN v ORDER BY v.data_inizio ASC

# Check if abrogated
MATCH (n:Norma {urn: $urn})
WHERE EXISTS((n)<-[:abroga_totalmente]-())
RETURN n.stato = 'abrogato'
```

### Response Structure
```python
@dataclass
class VersionedNorm:
    urn: str
    testo_vigente: str
    vigenza_dal: date
    vigenza_al: Optional[date]
    is_current: bool
    is_abrogato: bool
    newer_version_exists: bool

@dataclass
class VersionTimeline:
    urn: str
    versions: List[VersionedNorm]
    current_version: Optional[VersionedNorm]
    abrogation_date: Optional[date]
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/temporal.py` | Created | TemporalQuery class with point-in-time queries |
| `visualex/graph/__init__.py` | Modified | Added temporal module exports |
| `tests/unit/test_graph_temporal.py` | Created | 20 tests for all 4 ACs |

### Change Log
- **2026-02-01**: Created temporal.py module with TemporalQuery, VersionedNorm, VersionTimeline, NormStatus
- **2026-02-01**: Implemented get_norm_at_date, get_version_timeline, is_abrogated, get_abrogation_info, get_versions_in_range
- **2026-02-01**: Added 18 unit tests - all passing
- **2026-02-01**: Code review fixes:
  - H1: Removed unused NodeType/EdgeType imports
  - H2: Fixed off-by-one boundary bug in version date comparison (>= instead of >)
  - H3: Removed misleading ORDER BY from Cypher, added clarifying comment
  - M1: Added test for exact end date boundary condition
  - M2: Removed unused `versions` parameter from `_build_current_version`
  - M3: Changed `is_abrogated` to return Optional[bool] (None for non-existent norms)
  - Added 2 new tests (boundary condition, norm not found for abrogation)
  - All 20 tests passing

