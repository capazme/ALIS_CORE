"""
Test per Weight Store DB Persistence (STORY-11-1)
==================================================

Test per:
- WeightVersion SQLAlchemy model
- WeightStore.save_weights() → database INSERT
- WeightStore._load_from_database() → database SELECT
- Roundtrip: save → load preserva tutti i campi
- Active version switching
- Graceful degradation senza database
"""

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

from merlt.rlcf.persistence import WeightVersion
from merlt.rlcf.database import Base

# WeightStore lives in merlt-models — add to path
import sys
from pathlib import Path

_merlt_models_root = Path(__file__).resolve().parents[3] / "merlt-models"
if str(_merlt_models_root) not in sys.path:
    sys.path.insert(0, str(_merlt_models_root))

from weights.store import WeightStore
from weights.config import WeightConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest_asyncio.fixture
async def sqlite_engine():
    """Create SQLite in-memory async engine with weight_versions table only."""
    engine = create_async_engine("sqlite+aiosqlite://", echo=False)
    async with engine.begin() as conn:
        # Only create the weight_versions table, not all Base tables
        # (some use JSONB which SQLite doesn't support)
        await conn.run_sync(
            Base.metadata.create_all,
            tables=[WeightVersion.__table__],
        )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def store(sqlite_engine):
    """WeightStore wired to SQLite in-memory."""
    s = WeightStore(database_url="sqlite+aiosqlite://")
    # Override session factory to use the same engine (tables already created)
    factory = async_sessionmaker(
        bind=sqlite_engine, class_=AsyncSession, expire_on_commit=False
    )
    s._session_factory = factory
    return s


@pytest.fixture
def sample_config() -> WeightConfig:
    """A full WeightConfig for testing."""
    return WeightConfig(
        version="2.0",
        schema_version="1.0",
    )


# =============================================================================
# TEST WeightVersion MODEL
# =============================================================================


class TestWeightVersionModel:
    """Test SQLAlchemy model maps to weight_versions table."""

    @pytest.mark.asyncio
    async def test_insert_and_select(self, sqlite_engine):
        """Can INSERT and SELECT a WeightVersion row."""
        factory = async_sessionmaker(
            bind=sqlite_engine, class_=AsyncSession, expire_on_commit=False
        )
        async with factory() as session:
            row = WeightVersion(
                id="wv-001",
                experiment_id="exp-test",
                version_tag="v1",
                config_json={"version": "2.0"},
                metrics_json={"accuracy": 0.85},
                is_active=True,
                created_by="test",
            )
            session.add(row)
            await session.commit()

        async with factory() as session:
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == "wv-001")
            )
            loaded = result.scalar_one()

            assert loaded.experiment_id == "exp-test"
            assert loaded.version_tag == "v1"
            assert loaded.config_json == {"version": "2.0"}
            assert loaded.metrics_json == {"accuracy": 0.85}
            assert loaded.is_active is True
            assert loaded.created_by == "test"


# =============================================================================
# TEST save_weights()
# =============================================================================


class TestSaveWeights:
    """Test WeightStore.save_weights() database persistence."""

    @pytest.mark.asyncio
    async def test_save_returns_version_id(self, store, sample_config):
        """save_weights() returns a UUID string."""
        version_id = await store.save_weights(sample_config, "exp-001")
        assert isinstance(version_id, str)
        assert len(version_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_save_inserts_row(self, store, sample_config):
        """save_weights() creates a row in weight_versions."""
        version_id = await store.save_weights(sample_config, "exp-001")

        async with store._session_factory() as session:
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == version_id)
            )
            row = result.scalar_one()

            assert row.experiment_id == "exp-001"
            assert row.is_active is True
            assert row.config_json is not None
            assert row.config_json["version"] == "2.0"

    @pytest.mark.asyncio
    async def test_save_stores_metrics(self, store, sample_config):
        """save_weights() persists metrics_json."""
        metrics = {"loss": 0.05, "accuracy": 0.92}
        version_id = await store.save_weights(
            sample_config, "exp-001", metrics=metrics
        )

        async with store._session_factory() as session:
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == version_id)
            )
            row = result.scalar_one()
            assert row.metrics_json == metrics

    @pytest.mark.asyncio
    async def test_save_auto_generates_version_tag(self, store, sample_config):
        """save_weights() generates version_tag from timestamp if not provided."""
        version_id = await store.save_weights(sample_config, "exp-001")

        async with store._session_factory() as session:
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == version_id)
            )
            row = result.scalar_one()
            # Format: YYYYMMDD_HHMMSS
            assert row.version_tag is not None
            assert len(row.version_tag) == 15
            assert "_" in row.version_tag

    @pytest.mark.asyncio
    async def test_save_custom_version_tag(self, store, sample_config):
        """save_weights() uses provided version_tag."""
        await store.save_weights(
            sample_config, "exp-001", version_tag="epoch-42"
        )

        async with store._session_factory() as session:
            result = await session.execute(
                select(WeightVersion).where(
                    WeightVersion.experiment_id == "exp-001"
                )
            )
            row = result.scalar_one()
            assert row.version_tag == "epoch-42"

    @pytest.mark.asyncio
    async def test_save_deactivates_previous(self, store, sample_config):
        """New save deactivates previous active versions for same experiment."""
        v1 = await store.save_weights(sample_config, "exp-001", version_tag="v1")
        v2 = await store.save_weights(sample_config, "exp-001", version_tag="v2")

        async with store._session_factory() as session:
            # v1 should be deactivated
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == v1)
            )
            row1 = result.scalar_one()
            assert row1.is_active is False

            # v2 should be active
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == v2)
            )
            row2 = result.scalar_one()
            assert row2.is_active is True

    @pytest.mark.asyncio
    async def test_save_different_experiments_independent(self, store, sample_config):
        """Active versions for different experiments are independent."""
        v1 = await store.save_weights(sample_config, "exp-A")
        v2 = await store.save_weights(sample_config, "exp-B")

        async with store._session_factory() as session:
            # Both should be active (different experiments)
            for vid in (v1, v2):
                result = await session.execute(
                    select(WeightVersion).where(WeightVersion.id == vid)
                )
                assert result.scalar_one().is_active is True

    @pytest.mark.asyncio
    async def test_save_stores_created_by(self, store, sample_config):
        """save_weights() persists created_by field."""
        version_id = await store.save_weights(
            sample_config, "exp-001", created_by="training_scheduler"
        )

        async with store._session_factory() as session:
            result = await session.execute(
                select(WeightVersion).where(WeightVersion.id == version_id)
            )
            assert result.scalar_one().created_by == "training_scheduler"

    @pytest.mark.asyncio
    async def test_save_does_not_mutate_caller_config(self, store):
        """save_weights() must not modify the caller's WeightConfig object."""
        config = WeightConfig(version="2.0")
        original_experiment_id = config.experiment_id  # None
        original_metrics = config.metrics  # None

        await store.save_weights(
            config, "exp-mutation", metrics={"loss": 0.1}
        )

        assert config.experiment_id == original_experiment_id
        assert config.metrics == original_metrics


# =============================================================================
# TEST _load_from_database()
# =============================================================================


class TestLoadFromDatabase:
    """Test WeightStore._load_from_database()."""

    @pytest.mark.asyncio
    async def test_load_returns_config(self, store, sample_config):
        """load returns a WeightConfig after save."""
        await store.save_weights(sample_config, "exp-001")
        loaded = await store._load_from_database("exp-001")

        assert loaded is not None
        assert isinstance(loaded, WeightConfig)
        assert loaded.version == "2.0"

    @pytest.mark.asyncio
    async def test_load_nonexistent_returns_none(self, store):
        """load for non-existent experiment returns None."""
        loaded = await store._load_from_database("nonexistent")
        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_returns_latest_active(self, store, sample_config):
        """load returns the most recently activated version."""
        await store.save_weights(sample_config, "exp-001", version_tag="v1")

        config2 = WeightConfig(version="3.0", schema_version="1.1")
        await store.save_weights(config2, "exp-001", version_tag="v2")

        loaded = await store._load_from_database("exp-001")
        assert loaded is not None
        assert loaded.version == "3.0"
        assert loaded.schema_version == "1.1"

    @pytest.mark.asyncio
    async def test_load_no_database_returns_none(self):
        """load without database_url returns None."""
        store = WeightStore(database_url=None)
        loaded = await store._load_from_database("exp-001")
        assert loaded is None


# =============================================================================
# TEST ROUNDTRIP
# =============================================================================


class TestRoundtrip:
    """Test save -> load preserves all WeightConfig fields."""

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_retrieval(self, store):
        """Roundtrip preserves retrieval weights."""
        config = WeightConfig()
        config.retrieval.alpha.default = 0.85
        config.retrieval.over_retrieve_factor = 5
        config.retrieval.max_graph_hops = 4

        await store.save_weights(config, "exp-rt")
        loaded = await store._load_from_database("exp-rt")

        assert loaded is not None
        assert loaded.retrieval.alpha.default == 0.85
        assert loaded.retrieval.over_retrieve_factor == 5
        assert loaded.retrieval.max_graph_hops == 4

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_rlcf(self, store):
        """Roundtrip preserves RLCF authority weights."""
        config = WeightConfig()
        config.rlcf.baseline_credentials.default = 0.35
        config.rlcf.track_record.default = 0.45

        await store.save_weights(config, "exp-rt")
        loaded = await store._load_from_database("exp-rt")

        assert loaded is not None
        assert loaded.rlcf.baseline_credentials.default == 0.35
        assert loaded.rlcf.track_record.default == 0.45

    @pytest.mark.asyncio
    async def test_roundtrip_preserves_gating(self, store):
        """Roundtrip preserves gating expert priors."""
        config = WeightConfig()
        config.gating.expert_priors["LiteralExpert"].default = 0.4

        await store.save_weights(config, "exp-rt")
        loaded = await store._load_from_database("exp-rt")

        assert loaded is not None
        assert loaded.gating.expert_priors["LiteralExpert"].default == 0.4

    @pytest.mark.asyncio
    async def test_roundtrip_via_get_weights(self, store):
        """Full path: save_weights -> get_weights(experiment_id) returns DB version."""
        config = WeightConfig(version="5.0")
        await store.save_weights(config, "exp-full")

        loaded = await store.get_weights(experiment_id="exp-full")
        assert loaded.version == "5.0"


# =============================================================================
# TEST GRACEFUL DEGRADATION
# =============================================================================


class TestGracefulDegradation:
    """Test behavior without database."""

    @pytest.mark.asyncio
    async def test_save_without_db_returns_version_id(self):
        """save_weights without database_url still returns a version_id."""
        store = WeightStore(database_url=None)
        config = WeightConfig()
        version_id = await store.save_weights(config, "exp-001")
        assert isinstance(version_id, str)
        assert len(version_id) == 36

    @pytest.mark.asyncio
    async def test_save_without_db_invalidates_cache(self):
        """save_weights without database still invalidates cache."""
        store = WeightStore(database_url=None)
        config = WeightConfig()

        # Prime cache
        _ = await store.get_weights(experiment_id="exp-001")
        assert len(store._cache) > 0

        # Save should clear it
        await store.save_weights(config, "exp-001")
        assert not any("exp-001" in k for k in store._cache)

    @pytest.mark.asyncio
    async def test_get_weights_falls_back_to_yaml(self):
        """get_weights without DB falls back to YAML config."""
        store = WeightStore(database_url=None)
        config = await store.get_weights(experiment_id="any-experiment")
        # Should get default YAML config, not crash
        assert isinstance(config, WeightConfig)
        assert config.version == "2.0"
