# Story 2b-1: Graph Schema Setup

## Status
- **Epic**: Epic 2b: Graph Building
- **Status**: done
- **Priority**: High

## Context
Define the FalkorDB schema with all node and edge types for the Knowledge Graph. This is the foundation for all graph-based operations in ALIS.

## Existing Code
- `Legacy/MERL-T_alpha/merlt/storage/graph/client.py` - FalkorDBClient (async)
- `Legacy/MERL-T_alpha/merlt/storage/graph/config.py` - FalkorDBConfig

## Acceptance Criteria

### AC1: Node Types (26 Types - Full MERL-T Specification)
**Given** the FalkorDB instance is running
**When** I initialize the graph schema
**Then** the following node type categories are created:
- **Normative Sources (4):** Norma, Versione, DirettivaUE, RegolamentoUE
- **Text Structure (4):** Comma, Lettera, Numero, DefinizioneLegale
- **Case Law & Doctrine (3):** AttoGiudiziario, Caso, Dottrina
- **Subjects & Roles (3):** SoggettoGiuridico, RuoloGiuridico, Organo
- **Legal Concepts (5):** Concetto, Principio, DirittoSoggettivo, InteresseLegittimo, Responsabilita
- **Dynamics (4):** FattoGiuridico, Procedura, Sanzione, Termine
- **Logic & Reasoning (3):** Regola, Proposizione, ModalitaGiuridica

### AC2: Edge Types (65 Types - Full MERL-T Specification)
**Given** edge types are needed
**When** I define relationships
**Then** the following edge type categories are available:
- **Structural (5):** contiene, parte_di, versione_precedente, versione_successiva, ha_versione
- **Modification (9):** sostituisce, inserisce, abroga_totalmente, abroga_parzialmente, sospende, proroga, integra, deroga_a, consolida
- **Semantic (6):** disciplina, applica_a, definisce, prevede_sanzione, stabilisce_termine, prevede
- **Dependency (3):** dipende_da, presuppone, species
- **Citation & Interpretation (3):** cita, interpreta, commenta
- **European (3):** attua, recepisce, conforme_a
- **Institutional (3):** emesso_da, ha_competenza_su, gerarchicamente_superiore
- **Case-based (3):** riguarda, applica_norma_a_caso, precedente_di
- **Classification (2):** fonte, classifica_in
- **LKIF Modalities (28):** impone, conferisce, titolare_di, riveste_ruolo, attribuisce_responsabilita, responsabile_per, esprime_principio, conforma_a_principio, deroga_principio, bilancia_con, produce_effetto, presupposto_di, costitutivo_di, estingue, modifica_efficacia, applica_regola, implica, contradice, giustifica, limita, tutela, viola, compatibile_con, incompatibile_con, specifica, esemplifica, causa_di, condizione_di

### AC3: Indexes
**Given** the schema is initialized
**When** I query node types
**Then** all types have appropriate indexes:
- 25 standard indexes on urn/node_id for all node types
- 10 full-text indexes on text fields for search
- Filtering indexes on stato, data_pubblicazione, organo_emittente, etc.

## Tasks/Subtasks

- [x] **T1**: Create `visualex/graph/` module
- [x] **T2**: Implement `schema.py` with node/edge definitions (full MERL-T spec: 26 nodes, 65 edges)
- [x] **T3**: Implement `client.py` (port from Legacy with improvements)
- [x] **T4**: Implement `config.py` for FalkorDB config
- [x] **T5**: Add index creation in schema initialization (25 standard + 10 full-text)
- [x] **T6**: Write tests for schema operations (119 tests)
- [x] **T7**: Code review (7 issues fixed)
- [x] **T8**: Expand schema to full MERL-T specification
- [x] **T9**: Update tests for expanded schema
- [x] **T10**: Second code review (9 issues fixed)

## Technical Details

### FalkorDB Cypher for Schema
```cypher
// Create indexes (sample)
CREATE INDEX ON :Norma(urn)
CREATE INDEX ON :Norma(node_id)
CREATE INDEX ON :Comma(urn)
CREATE INDEX ON :AttoGiudiziario(urn)

// Full-text index (FalkorDB specific)
CALL db.idx.fulltext.createNodeIndex('Norma', 'testo_vigente')
CALL db.idx.fulltext.createNodeIndex('Comma', 'testo')
CALL db.idx.fulltext.createNodeIndex('AttoGiudiziario', 'massima')
```

### Node Properties (Dict-based Schema)
```python
from visualex.graph.schema import NODE_PROPERTIES, NodeType

# Access properties for any node type
norma_props = NODE_PROPERTIES[NodeType.NORMA]
# ['node_id', 'estremi', 'urn', 'fonte', 'titolo', 'descrizione', ...]

comma_props = NODE_PROPERTIES[NodeType.COMMA]
# ['node_id', 'urn', 'tipo', 'posizione', 'testo', 'testo_originale', ...]

# Schema summary
from visualex.graph import GraphSchema
schema = GraphSchema()
summary = schema.get_schema_summary()
# {'node_types': 26, 'edge_types': 65, 'indexes': 25, 'fulltext_indexes': 10, ...}
```

---

## Dev Agent Record

### File List
| File | Action | Description |
|------|--------|-------------|
| `visualex-api/visualex/graph/__init__.py` | Created | Module exports (updated for MERL-T) |
| `visualex-api/visualex/graph/config.py` | Created | FalkorDB configuration with env vars |
| `visualex-api/visualex/graph/schema.py` | Created | Full MERL-T schema: 26 nodes, 65 edges |
| `visualex-api/visualex/graph/client.py` | Created | Async FalkorDB client wrapper |
| `visualex-api/tests/unit/test_graph_config.py` | Created | 24 config tests |
| `visualex-api/tests/unit/test_graph_schema.py` | Created | 62 schema tests |
| `visualex-api/tests/unit/test_graph_client.py` | Created | 33 client tests |

### Change Log

**2026-02-01 - Initial Implementation**
- Created `visualex/graph/` module structure
- Implemented `FalkorDBConfig` with environment variable support
- Implemented initial `GraphSchema` with simplified types
- Ported `FalkorDBClient` from Legacy with async improvements

**2026-02-01 - First Code Review Fixes**
- [M1] Replaced deprecated `asyncio.get_event_loop()` with `asyncio.get_running_loop()`
- [M2] Added `__all__` to config.py for explicit public API
- [M3] Added `__all__` to schema.py for explicit public API
- [L1] Documented reserved config fields (max_connections, timeout_ms)
- [L2] Added `Direction` enum for type-safe traversal direction
- [L3] Added `confirm=True` safety parameter to `delete_all()`
- [L4] Updated client to import and use Direction enum

**2026-02-01 - MERL-T Schema Expansion**
- Expanded schema to full MERL-T Knowledge Graph specification
- 26 Node Types across 7 categories (was 6)
- 65 Edge Types across 11 categories (was 6)
- NODE_PROPERTIES dict replaces individual property lists
- EDGE_PROPERTIES with COMMON_EDGE_PROPERTIES for provenance
- 25 standard indexes + 10 full-text indexes
- Updated tests: 119 tests total (was 95)

**2026-02-01 - Second Code Review Fixes**
- [H1] Updated story ACs to reflect actual 26 node types, 65 edge types
- [H2] Corrected test count from 95 to 119
- [H3] Updated Technical Details with correct code examples
- [M1] Added TYPE_CHECKING imports for better type hints
- [M2] Added COMMON_EDGE_PROPERTIES to __all__ export
- [M3] Fixed docstrings referencing non-existent NodeType.ARTICOLO
- [L1] Updated module docstring to say 26 Node Types
- [L2] Fixed create_edge to use flexible match keys
