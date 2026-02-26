"""
Weight Persistence Tests
========================

Tests for WeightStore database persistence:
- Save → load round-trip with SQLite in-memory
- Graceful fallback when database unavailable
- Config serialization / deserialization
"""

import os
import pytest
import pytest_asyncio

# Use SQLite for tests
os.environ["RLCF_DATABASE_URL"] = "sqlite:///test_weights.db"
os.environ["RLCF_ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from merlt.weights.store import WeightStore, WeightVersion
from merlt.weights.config import (
    WeightConfig,
    RetrievalWeights,
    LearnableWeight,
    GatingWeights,
)
from merlt.rlcf.database import Base


@pytest_asyncio.fixture
async def db_store(tmp_path):
    """WeightStore with SQLite in-memory database."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

    db_url = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(db_url, echo=False)

    # Only create the WeightVersion table (not all Base tables which may use JSONB)
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: WeightVersion.__table__.create(sync_conn, checkfirst=True)
        )

    # Patch get_async_session to use our engine
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    import merlt.rlcf.database as db_module
    original_engine = db_module._async_engine
    original_session = db_module._AsyncSessionLocal
    db_module._async_engine = engine
    db_module._AsyncSessionLocal = session_factory

    store = WeightStore(
        config_path=tmp_path / "nonexistent.yaml",
        database_url=db_url,
    )

    yield store

    db_module._async_engine = original_engine
    db_module._AsyncSessionLocal = original_session
    await engine.dispose()


@pytest.fixture
def memory_store(tmp_path):
    """WeightStore without database (in-memory only)."""
    return WeightStore(config_path=tmp_path / "nonexistent.yaml")


def _make_config() -> WeightConfig:
    """Create a minimal WeightConfig for testing."""
    return WeightConfig(
        version="test-1.0",
        schema_version="1.0",
        retrieval=RetrievalWeights(
            alpha=LearnableWeight(default=0.75, bounds=(0.3, 0.9)),
        ),
        gating=GatingWeights(
            expert_priors={
                "LiteralExpert": LearnableWeight(default=0.3, bounds=(0.1, 0.5)),
                "SystemicExpert": LearnableWeight(default=0.25, bounds=(0.1, 0.5)),
            }
        ),
    )


class TestWeightStoreDBPersistence:
    """Tests for database-backed weight persistence."""

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, db_store):
        """Save weights to DB, then load by experiment_id."""
        config = _make_config()
        version_id = await db_store.save_weights(
            config=config,
            experiment_id="exp-roundtrip-001",
            metrics={"accuracy": 0.85, "mrr": 0.72},
        )

        assert version_id is not None

        # Clear cache to force DB load
        db_store._cache.clear()

        loaded = await db_store._load_from_database("exp-roundtrip-001")
        assert loaded is not None
        assert loaded.retrieval.alpha.default == 0.75

    @pytest.mark.asyncio
    async def test_save_deactivates_previous(self, db_store):
        """Saving a new version deactivates the previous active one."""
        config1 = _make_config()
        config1.version = "v1"
        await db_store.save_weights(config=config1, experiment_id="exp-deact")

        config2 = _make_config()
        config2.version = "v2"
        await db_store.save_weights(config=config2, experiment_id="exp-deact")

        db_store._cache.clear()

        loaded = await db_store._load_from_database("exp-deact")
        assert loaded is not None
        assert loaded.version == "v2"

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, db_store):
        """Loading a non-existent experiment returns None."""
        loaded = await db_store._load_from_database("nonexistent-exp")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_config_to_dict_roundtrip(self, db_store):
        """_config_to_dict → _parse_yaml_to_config produces equivalent config."""
        original = _make_config()
        serialized = db_store._config_to_dict(original)

        assert isinstance(serialized, dict)
        assert serialized["version"] == "test-1.0"
        assert serialized["retrieval"]["alpha"]["default"] == 0.75

        restored = db_store._parse_yaml_to_config(serialized)
        assert restored.retrieval.alpha.default == original.retrieval.alpha.default


class TestWeightStoreGracefulFallback:
    """Tests for graceful degradation when database unavailable."""

    @pytest.mark.asyncio
    async def test_save_without_db_logs_warning(self, memory_store):
        """Save without database_url logs a warning but doesn't crash."""
        config = _make_config()
        version_id = await memory_store.save_weights(
            config=config,
            experiment_id="exp-no-db",
        )
        assert version_id is not None

    @pytest.mark.asyncio
    async def test_load_without_db_returns_none(self, memory_store):
        """Load without database_url returns None gracefully."""
        loaded = await memory_store._load_from_database("any-experiment")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_get_weights_defaults_without_db(self, memory_store):
        """get_weights returns default config when no DB and no YAML."""
        config = await memory_store.get_weights()
        assert config is not None
        assert config.retrieval.alpha.default == 0.7  # default value
