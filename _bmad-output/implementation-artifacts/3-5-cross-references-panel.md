# Story 3-5: Cross-References Panel (Backend)

## Status
- **Epic**: Epic 3: Norm Browsing & Search
- **Status**: done
- **Priority**: High

## Context
Backend API for retrieving cross-references (graph neighbors) for a given article. Returns grouped relationships for the frontend panel.

## Existing Code
- `visualex/graph/client.py` - FalkorDBClient with query method
- `visualex/graph/schema.py` - EdgeType definitions (CITA, MODIFICA, INTERPRETA, etc.)
- `visualex/graph/relations.py` - CitationExtractor, RelationCreator

## Acceptance Criteria

### AC1: Outgoing References
**Given** an article URN
**When** querying for outgoing references
**Then** return all norms this article cites (CITA edges)

### AC2: Incoming References
**Given** an article URN
**When** querying for incoming references
**Then** return all norms citing this article (CITA edges reversed)

### AC3: Modifying Legislation
**Given** an article URN
**When** querying for modifications
**Then** return all norms that modified this article (MODIFICA, ABROGA, SOSTITUISCE edges)

### AC4: Jurisprudence
**Given** an article URN
**When** querying for jurisprudence
**Then** return all case law citing/interpreting this article (INTERPRETA edges)

### AC5: Grouped Response
**Given** cross-reference query results
**When** formatting response
**Then** results are grouped by relationship type with counts

### AC6: Pagination
**Given** >20 results in any category
**When** querying
**Then** results support offset/limit pagination

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/neighbors.py` module
- [x] **T2**: Implement `CrossReferenceService` class
- [x] **T3**: Implement outgoing references query (AC1)
- [x] **T4**: Implement incoming references query (AC2)
- [x] **T5**: Implement modifications query (AC3)
- [x] **T6**: Implement jurisprudence query (AC4)
- [x] **T7**: Implement grouped response format (AC5)
- [x] **T8**: Add pagination support (AC6)
- [x] **T9**: Update `visualex/graph/__init__.py` exports
- [x] **T10**: Write tests (20 tests passing)
- [x] **T11**: Code review

## Technical Details

### Cypher Queries
```cypher
# Outgoing references
MATCH (n:Norma {urn: $urn})-[r:CITA]->(m)
RETURN m.urn, m.rubrica, type(r), r

# Incoming references
MATCH (m)-[r:CITA]->(n:Norma {urn: $urn})
RETURN m.urn, m.rubrica, type(r), r

# Modifications
MATCH (m)-[r:MODIFICA|ABROGA|SOSTITUISCE|INTEGRA]->(n:Norma {urn: $urn})
RETURN m.urn, m.rubrica, type(r), r.data_efficacia

# Jurisprudence
MATCH (j:AttoGiudiziario)-[r:INTERPRETA]->(n:Norma {urn: $urn})
RETURN j.node_id, j.massima, j.organo_emittente, j.data
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex/graph/neighbors.py` | Create | CrossReferenceService |
| `visualex/graph/__init__.py` | Modify | Add exports |
| `tests/unit/test_graph_neighbors.py` | Create | Tests |

### Change Log
- **2026-02-02**: Created story file
- **2026-02-02**: Implemented CrossReferenceService with all AC criteria met
- **2026-02-02**: Fixed EdgeType naming to match schema (ABROGA_TOTALMENTE, DEROGA_A, etc.)
- **2026-02-02**: All 20 tests passing - Story completed
