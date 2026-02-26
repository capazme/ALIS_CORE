"""
Experiment Redis Persistence Tests
====================================

Tests for ExperimentTracker Redis persistence:
- Verify SET/GET calls on create, record, stop, complete
- Verify graceful degradation when Redis unavailable
- Verify cold-start recovery (assign_variant loads from Redis)
"""

import json
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

import pytest
import pytest_asyncio

from merlt.weights.experiment import (
    ExperimentTracker,
    Experiment,
    _EXP_REDIS_TTL,
)
from merlt.weights.store import WeightStore
from merlt.weights.config import WeightConfig, RetrievalWeights, LearnableWeight


def _make_config(alpha: float = 0.7) -> WeightConfig:
    """Create a minimal WeightConfig for testing."""
    return WeightConfig(
        version="test",
        schema_version="1.0",
        retrieval=RetrievalWeights(
            alpha=LearnableWeight(default=alpha),
        ),
    )


@pytest.fixture
def mock_store():
    """Mock WeightStore."""
    store = MagicMock(spec=WeightStore)
    store.save_weights = AsyncMock(return_value="version-123")
    return store


@pytest.fixture
def tracker(mock_store):
    """ExperimentTracker with mocked store."""
    return ExperimentTracker(mock_store)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.get = AsyncMock(return_value=None)
    redis.ping = AsyncMock(return_value=True)
    return redis


class TestExperimentRedisCreate:
    """Tests that create_experiment persists to Redis."""

    @pytest.mark.asyncio
    async def test_create_calls_redis_set(self, tracker, mock_redis):
        """create_experiment should SET experiment data to Redis."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=mock_redis):
            exp = await tracker.create_experiment(
                name="test-exp",
                control_weights=_make_config(0.7),
                treatment_weights=_make_config(0.8),
            )

        # Redis SET should have been called
        mock_redis.set.assert_called_once()
        call_args = mock_redis.set.call_args
        key = call_args[0][0]
        data = json.loads(call_args[0][1])

        assert key == f"experiment:{exp.id}"
        assert data["name"] == "test-exp"
        assert data["status"] == "running"
        assert call_args[1]["ex"] == _EXP_REDIS_TTL


class TestExperimentRedisRecordOutcome:
    """Tests that record_outcome persists to Redis."""

    @pytest.mark.asyncio
    async def test_record_outcome_updates_redis(self, tracker, mock_redis):
        """record_outcome should update experiment in Redis."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=mock_redis):
            exp = await tracker.create_experiment(
                name="outcome-test",
                control_weights=_make_config(),
                treatment_weights=_make_config(),
            )
            mock_redis.set.reset_mock()

            await tracker.record_outcome(exp.id, "control", {"mrr": 0.85})

        # Redis SET should have been called for update
        mock_redis.set.assert_called_once()
        data = json.loads(mock_redis.set.call_args[0][1])
        assert len(data["metrics_by_variant"]["control"]) == 1


class TestExperimentRedisStopComplete:
    """Tests for stop/complete persistence."""

    @pytest.mark.asyncio
    async def test_stop_persists_to_redis(self, tracker, mock_redis):
        """stop_experiment should update Redis with stopped status."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=mock_redis):
            exp = await tracker.create_experiment(
                name="stop-test",
                control_weights=_make_config(),
                treatment_weights=_make_config(),
            )
            mock_redis.set.reset_mock()

            await tracker.stop_experiment(exp.id)

        data = json.loads(mock_redis.set.call_args[0][1])
        assert data["status"] == "stopped"
        assert data["completed_at"] is not None

    @pytest.mark.asyncio
    async def test_complete_persists_to_redis(self, tracker, mock_redis):
        """complete_experiment should update Redis with completed status."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=mock_redis):
            exp = await tracker.create_experiment(
                name="complete-test",
                control_weights=_make_config(),
                treatment_weights=_make_config(),
            )
            mock_redis.set.reset_mock()

            await tracker.complete_experiment(exp.id)

        data = json.loads(mock_redis.set.call_args[0][1])
        assert data["status"] == "completed"


class TestExperimentRedisRecovery:
    """Tests for cold-start recovery from Redis."""

    @pytest.mark.asyncio
    async def test_assign_variant_recovers_from_redis(self, tracker, mock_redis):
        """assign_variant should load experiment from Redis when not in memory."""
        stored_exp = {
            "id": "recovered-exp",
            "name": "recovered",
            "status": "running",
            "allocation": {"control": 0.5, "treatment": 0.5},
            "created_at": datetime.now().isoformat(),
            "metrics_by_variant": {"control": [], "treatment": []},
            "user_assignments": {},
        }
        mock_redis.get = AsyncMock(return_value=json.dumps(stored_exp))

        with patch("merlt.weights.experiment._get_exp_redis", return_value=mock_redis):
            variant = await tracker.assign_variant("recovered-exp", "user-123")

        assert variant in ("control", "treatment")
        # Should now be cached in memory
        assert "recovered-exp" in tracker._experiments


class TestExperimentGracefulDegradation:
    """Tests for graceful degradation when Redis unavailable."""

    @pytest.mark.asyncio
    async def test_create_works_without_redis(self, tracker):
        """create_experiment works even when Redis is unavailable."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=None):
            exp = await tracker.create_experiment(
                name="no-redis-test",
                control_weights=_make_config(),
                treatment_weights=_make_config(),
            )

        assert exp is not None
        assert exp.status == "running"

    @pytest.mark.asyncio
    async def test_record_outcome_works_without_redis(self, tracker):
        """record_outcome works even when Redis is unavailable."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=None):
            exp = await tracker.create_experiment(
                name="no-redis-outcome",
                control_weights=_make_config(),
                treatment_weights=_make_config(),
            )
            await tracker.record_outcome(exp.id, "control", {"mrr": 0.9})

        assert len(exp.metrics_by_variant["control"]) == 1

    @pytest.mark.asyncio
    async def test_assign_raises_when_not_found_and_no_redis(self, tracker):
        """assign_variant raises ValueError when experiment not found and no Redis."""
        with patch("merlt.weights.experiment._get_exp_redis", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                await tracker.assign_variant("nonexistent", "user-123")
