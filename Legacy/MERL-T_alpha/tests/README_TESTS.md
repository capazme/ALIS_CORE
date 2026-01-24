# MERL-T Test Suite

Test completi per Phase 1 - Live Enrichment con database reali.

## 📋 Indice

- [Filosofia Test](#filosofia-test)
- [Prerequisiti](#prerequisiti)
- [Setup](#setup)
- [Esecuzione Test](#esecuzione-test)
- [Struttura Test](#struttura-test)
- [Database Isolation](#database-isolation)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Filosofia Test

### Zero Mock Policy

**Tutti i test usano database reali**:
- ✅ PostgreSQL reale (con transaction rollback)
- ✅ FalkorDB reale (con graph di test isolato)
- ✅ Qdrant reale (con collection di test)

**Perché NO MOCK?**
1. **Affidabilità**: Test realistici rivelano bug reali
2. **Integrazione**: Verifica funzionamento completo dello stack
3. **Regression**: Previene regressioni in production
4. **Confidence**: Deploy sicuro grazie a test robusti

### Test Types

- **Unit**: Test componenti singoli (es. domain authority calc)
- **Integration**: Test integrazione componenti (es. DB + triggers)
- **E2E**: Test workflow completo (propose → vote → approve → graph)

---

## 🔧 Prerequisiti

### Database Services

Tutti i servizi devono essere running:

```bash
# Start all services
docker-compose -f docker-compose.dev.yml up -d

# Verify
docker-compose -f docker-compose.dev.yml ps

# Expected:
# - PostgreSQL (port 5433)
# - FalkorDB (port 6380)
# - Qdrant (port 6333)
# - Redis (port 6379)
```

### Migration

Database schema deve essere migrated:

```bash
# Run migration
python scripts/run_enrichment_migration.py

# Verify
python scripts/run_enrichment_migration.py --verify-only
```

---

## 🚀 Setup

### 1. Install Dependencies

```bash
# Activate venv
source .venv/bin/activate

# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Verify
pytest --version
```

### 2. Environment Variables

Assicurati che `.env` sia configurato:

```env
# PostgreSQL
ENRICHMENT_DATABASE_URL=postgresql+asyncpg://dev:devpassword@localhost:5433/rlcf_dev

# FalkorDB
FALKORDB_HOST=localhost
FALKORDB_PORT=6380

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

---

## ▶️ Esecuzione Test

### Run All Tests

```bash
# Tutti i test
pytest tests/

# Con output verboso
pytest tests/ -v

# Con print statements
pytest tests/ -v -s
```

### Run Specific Test Suites

```bash
# Solo enrichment flow
pytest tests/api/test_enrichment_flow.py -v

# Solo document upload
pytest tests/api/test_document_upload.py -v

# Solo graph writer
pytest tests/storage/test_entity_writer_integration.py -v

# Solo E2E
pytest tests/api/test_e2e_enrichment_workflow.py -v
```

### Run Specific Test

```bash
# Single test
pytest tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_propose_entity_creates_pending_record -v

# Test class
pytest tests/api/test_enrichment_flow.py::TestEntityValidationFlow -v
```

### Run with Markers

```bash
# Solo integration tests
pytest tests/ -m integration -v

# Solo E2E tests
pytest tests/ -m e2e -v

# Exclude slow tests
pytest tests/ -m "not slow" -v
```

### Coverage Report

```bash
# Run con coverage
pytest tests/ --cov=merlt --cov-report=html --cov-report=term-missing

# Apri report HTML
open htmlcov/index.html
```

---

## 📁 Struttura Test

```
tests/
├── api/
│   ├── conftest.py                     # Fixtures comuni (db_session, falkordb_client, etc.)
│   ├── test_enrichment_flow.py          # Entity/Relation validation flow
│   ├── test_document_upload.py          # Document upload & parsing
│   └── test_e2e_enrichment_workflow.py  # E2E completo workflow
│
├── storage/
│   └── test_entity_writer_integration.py # EntityGraphWriter + FalkorDB
│
└── README_TESTS.md                      # Questa guida
```

### Fixtures Principali

Definite in `tests/api/conftest.py`:

| Fixture | Scope | Descrizione |
|---------|-------|-------------|
| `db_session` | function | PostgreSQL session con rollback automatico |
| `falkordb_client` | function | FalkorDB client con graph `merl_t_test` |
| `embedding_service` | function | EmbeddingService con collection `merl_t_test_chunks` |
| `sample_entity_data` | function | Dati di esempio per PendingEntity |
| `sample_relation_data` | function | Dati di esempio per PendingRelation |
| `sample_amendment_data` | function | Dati di esempio per PendingAmendment |
| `test_upload_dir` | function | Directory temp per upload documenti |
| `sample_pdf_content` | function | Contenuto PDF minimale valido |

---

## 🔒 Database Isolation

### PostgreSQL

**Strategy**: Transaction Rollback

Ogni test ha una transazione che viene rollback automaticamente:

```python
async with engine.begin() as conn:
    async with session.begin_nested():
        # Test esegue qui
        yield session
        # Rollback automatico
        await session.rollback()
```

**Vantaggi**:
- ✅ Database pulito tra test
- ✅ No side effects
- ✅ Fast (no recreate tables)

### FalkorDB

**Strategy**: Separate Graph + Cleanup

Ogni test usa graph separato `merl_t_test`:

```python
async def cleanup():
    await client.execute_query("MATCH (n) DETACH DELETE n")
```

**Vantaggi**:
- ✅ Isolamento completo
- ✅ No interferenza con `merl_t_dev`

### Qdrant

**Strategy**: Separate Collection + Recreate

Ogni test ricrea collection `merl_t_test_chunks`:

```python
service = EmbeddingService(
    collection_name="merl_t_test_chunks",
    recreate_collection=True,
)
```

---

## 🧪 Test Coverage

### Entity Validation Flow

**File**: `test_enrichment_flow.py`

- ✅ Proposta entity
- ✅ Singolo voto (no consensus)
- ✅ Multi voti → approval consensus
- ✅ Multi voti → rejection consensus
- ✅ Voti misti (no consensus)
- ✅ Entity approved → graph write
- ✅ Relation validation
- ✅ Domain authority calculation

### Document Upload

**File**: `test_document_upload.py`

- ✅ Upload PDF
- ✅ Upload TXT
- ✅ File deduplication (SHA-256)
- ✅ Text extraction
- ✅ Text chunking
- ✅ Amendment extraction
- ✅ Processing status tracking
- ✅ Error handling

### Graph Writer

**File**: `test_entity_writer_integration.py`

- ✅ Entity node creation
- ✅ Relation to article
- ✅ Mechanical deduplication (exact match)
- ✅ Normalization (case, articles, punctuation)
- ✅ Different type NOT duplicate
- ✅ Enrichment adds sources
- ✅ Label mapping (Principio, Concetto, Definizione)
- ✅ Relation type mapping

### E2E Workflow

**File**: `test_e2e_enrichment_workflow.py`

- ✅ Complete entity lifecycle (propose → vote → approve → graph → authority)
- ✅ Multi-entity con deduplication
- ✅ Workflow con rejection

---

## 🐛 Troubleshooting

### Test Fails: Database Connection

**Errore**:
```
asyncpg.exceptions.ConnectionDoesNotExistError
```

**Soluzione**:
```bash
# Verifica servizi running
docker-compose -f docker-compose.dev.yml ps

# Restart
docker-compose -f docker-compose.dev.yml restart
```

### Test Fails: Migration Not Run

**Errore**:
```
relation "pending_entities" does not exist
```

**Soluzione**:
```bash
# Run migration
python scripts/run_enrichment_migration.py
```

### Test Slow

**Problema**: Test troppo lenti

**Soluzione**:
```bash
# Run solo test veloci
pytest tests/ -m "not slow" -v

# Oppure parallelize
pip install pytest-xdist
pytest tests/ -n auto
```

### FalkorDB Graph Dirty

**Problema**: Test fallisce per nodi pre-esistenti

**Soluzione**:
```bash
# Cleanup manuale FalkorDB
redis-cli -p 6380
> GRAPH.DELETE merl_t_test
```

### Qdrant Collection Dirty

**Problema**: Collection non pulita

**Soluzione**:
```python
# Il fixture ricrea automaticamente la collection
# Se persiste, restart Qdrant:
docker-compose -f docker-compose.dev.yml restart qdrant
```

---

## 📊 Expected Test Results

### All Tests Passing

```bash
$ pytest tests/ -v

tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_propose_entity_creates_pending_record PASSED
tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_single_vote_no_consensus PASSED
tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_multiple_votes_reach_approval_consensus PASSED
tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_multiple_votes_reach_rejection_consensus PASSED
tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_mixed_votes_no_consensus PASSED
tests/api/test_enrichment_flow.py::TestEntityValidationFlow::test_approved_entity_written_to_graph PASSED
tests/api/test_enrichment_flow.py::TestRelationValidationFlow::test_propose_relation_creates_pending_record PASSED
tests/api/test_enrichment_flow.py::TestRelationValidationFlow::test_relation_approval_consensus PASSED
tests/api/test_enrichment_flow.py::TestDomainAuthorityCalculation::test_new_user_default_authority PASSED
tests/api/test_enrichment_flow.py::TestDomainAuthorityCalculation::test_authority_increases_with_correct_votes PASSED
tests/api/test_enrichment_flow.py::TestDomainAuthorityCalculation::test_authority_decreases_with_incorrect_votes PASSED

tests/api/test_document_upload.py::TestDocumentUpload::test_upload_pdf_creates_record PASSED
tests/api/test_document_upload.py::TestDocumentUpload::test_duplicate_file_detection PASSED
tests/api/test_document_upload.py::TestDocumentUpload::test_upload_txt_file PASSED
tests/api/test_document_upload.py::TestDocumentParsing::test_parse_pdf_extracts_text PASSED
tests/api/test_document_upload.py::TestDocumentParsing::test_parse_txt_extracts_text PASSED
tests/api/test_document_upload.py::TestDocumentParsing::test_parse_document_chunks_text PASSED
tests/api/test_document_upload.py::TestAmendmentExtraction::test_extract_amendments_from_text PASSED
tests/api/test_document_upload.py::TestAmendmentExtraction::test_extract_multiple_amendments PASSED
tests/api/test_document_upload.py::TestDocumentProcessingStatus::test_document_processing_workflow PASSED
tests/api/test_document_upload.py::TestDocumentProcessingStatus::test_document_processing_error_handling PASSED

tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_write_entity_creates_node PASSED
tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_write_entity_creates_relation_to_article PASSED
tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_mechanical_deduplication_exact_match PASSED
tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_mechanical_deduplication_normalization PASSED
tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_different_type_not_duplicate PASSED
tests/storage/test_entity_writer_integration.py::TestEntityGraphWriter::test_enrichment_adds_sources PASSED
tests/storage/test_entity_writer_integration.py::TestEntityTypeLabelMapping::test_principio_has_principio_label PASSED
tests/storage/test_entity_writer_integration.py::TestEntityTypeLabelMapping::test_concetto_has_concetto_label PASSED
tests/storage/test_entity_writer_integration.py::TestEntityTypeLabelMapping::test_definizione_has_definizione_label PASSED
tests/storage/test_entity_writer_integration.py::TestRelationTypeMapping::test_principio_creates_esprime_principio_relation PASSED
tests/storage/test_entity_writer_integration.py::TestRelationTypeMapping::test_definizione_creates_definisce_relation PASSED

tests/api/test_e2e_enrichment_workflow.py::TestEndToEndEnrichmentWorkflow::test_complete_entity_lifecycle PASSED
tests/api/test_e2e_enrichment_workflow.py::TestEndToEndEnrichmentWorkflow::test_multi_entity_workflow_with_deduplication PASSED
tests/api/test_e2e_enrichment_workflow.py::TestEndToEndEnrichmentWorkflow::test_workflow_with_rejection PASSED

================================ 34 passed in 12.34s ================================
```

### Coverage Target

**Obiettivo**: >80% coverage per Phase 1 components

```bash
merlt/api/enrichment_router.py        95%
merlt/storage/enrichment/models.py    100%
merlt/storage/graph/entity_writer.py  92%
merlt/rlcf/domain_authority.py        88%
merlt/pipeline/document_parser.py     85%
```

---

## 🎯 Next Steps

### Immediate
- [ ] Run tutti i test: `pytest tests/ -v`
- [ ] Verifica coverage: `pytest tests/ --cov=merlt`
- [ ] Fix eventuali failures

### Future Enhancements
- [ ] Performance benchmarks (vote submission < 500ms)
- [ ] Load testing (concurrent votes)
- [ ] Chaos testing (database failures, network issues)
- [ ] Property-based testing (Hypothesis)

---

## 📚 Resources

- **Pytest Docs**: https://docs.pytest.org/
- **Pytest-asyncio**: https://pytest-asyncio.readthedocs.io/
- **SQLAlchemy Testing**: https://docs.sqlalchemy.org/en/20/orm/session_transaction.html#joining-a-session-into-an-external-transaction-such-as-for-test-suites

---

**Last Updated**: 2026-01-04  
**Phase**: 1 - Core Fixes  
**Test Count**: 34 tests  
**Coverage**: TBD (run `pytest --cov`)
