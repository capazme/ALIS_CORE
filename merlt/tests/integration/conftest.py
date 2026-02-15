"""
Integration Test Fixtures
===========================

Fixtures for RLCF integration tests requiring real PostgreSQL.

Requirements:
- Docker services running: docker-compose -f docker-compose.dev.yml up -d
- PostgreSQL on port 5433 (rlcf_dev database)
"""

import pytest
import pytest_asyncio
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from merlt.rlcf.database import Base as RLCFBase, get_async_database_url
from merlt.storage.bridge.models import Base as BridgeBase

# Ensure all RLCF models are registered on their Base.metadata
import merlt.experts.models  # noqa: F401 — QATrace, QAFeedback, AggregatedFeedback
import merlt.rlcf.models  # noqa: F401 — User, Feedback, etc.
import merlt.rlcf.persistence  # noqa: F401 — RLCFTrace, PolicyCheckpoint, etc.


@pytest_asyncio.fixture(scope="function")
async def rlcf_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async PostgreSQL session for RLCF integration tests.

    Creates tables before each test and wraps in a transaction
    that rolls back after the test for isolation.
    """
    url = get_async_database_url()
    engine = create_async_engine(url, echo=False, poolclass=NullPool)

    # Ensure RLCF + Bridge tables exist
    async with engine.begin() as conn:
        await conn.run_sync(RLCFBase.metadata.create_all)
        await conn.run_sync(BridgeBase.metadata.create_all)

    # Wrap test in a rolled-back transaction for isolation
    async with engine.connect() as connection:
        transaction = await connection.begin()
        session_factory = async_sessionmaker(bind=connection, class_=AsyncSession, expire_on_commit=False)
        session = session_factory()

        try:
            yield session
        finally:
            await session.close()
            await transaction.rollback()

    await engine.dispose()
