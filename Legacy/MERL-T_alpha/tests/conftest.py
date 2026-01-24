"""
Shared Test Fixtures
====================

Fixtures condivisi tra tests/api/ e tests/storage/.
"""

import asyncio
from typing import AsyncGenerator
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from merlt.storage.enrichment.database import get_database_url
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
    """PostgreSQL session with transaction rollback."""
    engine = create_async_engine(get_database_url(), echo=False, poolclass=NullPool)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session = async_session(bind=connection)
        
        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()
    
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def falkordb_client() -> AsyncGenerator[FalkorDBClient, None]:
    """FalkorDB client with isolated test graph."""
    client = FalkorDBClient(graph_name="merl_t_test")
    await client.connect()
    yield client
    
    try:
        await client.query("MATCH (n) DETACH DELETE n")
    except Exception as e:
        print(f"Warning: Failed to cleanup test graph: {e}")
    
    await client.close()


@pytest_asyncio.fixture(scope="function")
async def embedding_service() -> AsyncGenerator[EmbeddingService, None]:
    """Qdrant service with isolated collection."""
    service = EmbeddingService(collection_name="merl_t_test_chunks", recreate_collection=True)
    await service.initialize()
    yield service

    try:
        await service.delete_collection()
    except Exception:
        pass


@pytest.fixture
def sample_entity_data() -> dict:
    """Sample data for PendingEntity."""
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
    """Sample data for PendingRelation."""
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
    """Sample data for PendingAmendment."""
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
