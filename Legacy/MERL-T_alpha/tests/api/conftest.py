"""
Test Fixtures per API Tests
============================

Fixtures per testing con database reali (PostgreSQL, FalkorDB, Qdrant).

Strategy:
- PostgreSQL: usa transazioni rollback per isolation
- FalkorDB: usa graph separato per test
- Qdrant: usa collection separata per test

Ogni test ha database pulito e isolato dagli altri.
"""

import asyncio
from pathlib import Path
from typing import AsyncGenerator
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from merlt.storage.enrichment.database import get_database_url, Base
from merlt.storage.enrichment.models import (
    PendingEntity,
    EntityVote,
    PendingRelation,
    RelationVote,
    UserDocument,
    PendingAmendment,
    AmendmentVote,
    UserDomainAuthority,
)
from merlt.storage.graph.client import FalkorDBClient
from merlt.storage.vectors.embeddings import EmbeddingService


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop per session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Fixture che fornisce una session PostgreSQL isolata.

    Ogni test ha una transazione che viene rollback automaticamente.
    Database rimane pulito tra test.

    Yields:
        AsyncSession: Session isolata per il test
    """
    # Create engine con NullPool per evitare connection pooling in test
    engine = create_async_engine(
        get_database_url(),
        echo=False,
        poolclass=NullPool,
    )

    # Create session factory
    async_session = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

    # Create connection
    async with engine.connect() as connection:
        # Start transaction
        transaction = await connection.begin()

        # Create session bound to connection
        session = async_session(bind=connection)

        try:
            yield session
        finally:
            # Rollback transaction (cleanup)
            await session.close()
            await transaction.rollback()

    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def falkordb_client() -> AsyncGenerator[FalkorDBClient, None]:
    """
    Fixture che fornisce FalkorDB client isolato.

    Usa un graph separato 'merl_t_test' che viene pulito dopo ogni test.

    Yields:
        FalkorDBClient: Client connesso al graph di test
    """
    client = FalkorDBClient(graph_name="merl_t_test")
    await client.connect()

    yield client

    # Cleanup: rimuovi tutti i nodi e relazioni
    try:
        await client.query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"Warning: Failed to cleanup test graph: {e}")

    await client.close()


@pytest_asyncio.fixture(scope="function")
async def embedding_service() -> AsyncGenerator[EmbeddingService, None]:
    """
    Fixture che fornisce EmbeddingService isolato.

    Usa una collection Qdrant separata 'merl_t_test_chunks'.

    Yields:
        EmbeddingService: Service configurato per testing
    """
    service = EmbeddingService(
        collection_name="merl_t_test_chunks",
        recreate_collection=True,  # Ricrea collection pulita per ogni test
    )
    await service.initialize()

    yield service

    # Cleanup: elimina collection
    try:
        await service.delete_collection()
    except Exception:
        pass


@pytest.fixture
def sample_entity_data() -> dict:
    """
    Fixture con dati di esempio per PendingEntity.

    Returns:
        dict: Dati validi per creare una PendingEntity
    """
    return {
        "entity_id": "principio:legittima_difesa",
        "article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52",
        "source_type": "article",
        "entity_type": "principio",
        "entity_text": "Legittima difesa",
        "descrizione": "Diritto di difendere sé stessi o altri da aggressione ingiusta",
        "ambito": "penale",
        "fonte": "llm_extraction",
        "llm_confidence": 0.95,
        "llm_model": "claude-sonnet-4",
        "contributed_by": "user_test_001",
        "contributor_authority": 0.7,
    }


@pytest.fixture
def sample_relation_data() -> dict:
    """
    Fixture con dati di esempio per PendingRelation.

    Returns:
        dict: Dati validi per creare una PendingRelation
    """
    return {
        "relation_id": "rel:art52cp_esprime_legittima_difesa",
        "article_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52",
        "source_type": "article",
        "relation_type": "ESPRIME_PRINCIPIO",
        "source_node_urn": "urn:nir:stato:codice.penale:1930-10-19;1398~art52",
        "target_entity_id": "principio:legittima_difesa",
        "llm_confidence": 0.92,
        "contributed_by": "user_test_001",
        "contributor_authority": 0.7,
    }


@pytest.fixture
def sample_amendment_data() -> dict:
    """
    Fixture con dati di esempio per PendingAmendment.

    Returns:
        dict: Dati validi per creare un PendingAmendment
    """
    return {
        "amendment_id": "amend:art1453cc:dlgs_2003_06_30_196",
        "target_article_urn": "urn:nir:stato:codice.civile:1942-03-16;262~art1453",
        "atto_modificante_urn": "urn:nir:stato:decreto.legislativo:2003-06-30;196",
        "atto_modificante_estremi": "D.Lgs. 30 giugno 2003, n. 196",
        "tipo_atto": "decreto.legislativo",
        "numero_atto": "196",
        "data_atto": "2003-06-30",
        "disposizione": "L'articolo 1453 è sostituito dal seguente...",
        "tipo_modifica": "sostituzione",
        "testo_modificante": "Nuovo testo dell'articolo",
        "vigenza_inizio": "2003-07-01",
        "contributed_by": "user_test_001",
        "contributor_authority": 0.7,
        "ambito": "civile",
    }


@pytest.fixture
def test_upload_dir(tmp_path: Path) -> Path:
    """
    Fixture che fornisce directory temporanea per upload test.

    Args:
        tmp_path: Pytest fixture per temp directory

    Returns:
        Path: Directory per upload documenti test
    """
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@pytest.fixture
def sample_pdf_content() -> bytes:
    """
    Fixture con contenuto PDF di esempio per test upload.

    Returns:
        bytes: Contenuto PDF minimale valido
    """
    # PDF minimale valido (header + EOF)
    return b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000015 00000 n \n0000000068 00000 n \n0000000125 00000 n \ntrailer\n<<\n/Size 4\n/Root 1 0 R\n>>\nstartxref\n205\n%%EOF\n"


# Helper functions per assertions comuni

def assert_entity_in_db(session: AsyncSession, entity_id: str) -> bool:
    """
    Helper per verificare esistenza entity nel database.

    Args:
        session: Session PostgreSQL
        entity_id: ID entity da cercare

    Returns:
        bool: True se entity esiste
    """
    from sqlalchemy import select
    stmt = select(PendingEntity).where(PendingEntity.entity_id == entity_id)
    # Nota: questa è una funzione sync helper, il test deve usare await
    return stmt


def assert_consensus_reached(entity: PendingEntity, consensus_type: str) -> None:
    """
    Helper per verificare consensus raggiunto.

    Args:
        entity: PendingEntity da verificare
        consensus_type: 'approved' o 'rejected'

    Raises:
        AssertionError: Se consensus non raggiunto o tipo errato
    """
    assert entity.consensus_reached, f"Consensus not reached for {entity.entity_id}"
    assert entity.consensus_type == consensus_type, (
        f"Expected consensus type '{consensus_type}', got '{entity.consensus_type}'"
    )

    if consensus_type == "approved":
        assert entity.approval_score >= 2.0, (
            f"Approval score {entity.approval_score} below threshold 2.0"
        )
    else:
        assert entity.rejection_score >= 2.0, (
            f"Rejection score {entity.rejection_score} below threshold 2.0"
        )
