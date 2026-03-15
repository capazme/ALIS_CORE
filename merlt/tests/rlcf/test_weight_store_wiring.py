"""
Test WeightStore wiring into training loop (STORY-11-5)
========================================================

Tests for:
- _extract_weight_config() builds WeightConfig from policy state
- _persist_weight_config() calls WeightStore.save_weights()
- run_training_epoch() includes weight_version_id in result
- Graceful degradation when DB unavailable
- TrainingResult.weight_version_id field

Example:
    pytest tests/rlcf/test_weight_store_wiring.py -v
"""

import pytest
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from merlt.rlcf.training_scheduler import (
    TrainingScheduler,
    TrainingResult,
    SchedulerConfig,
)
from merlt.weights.config import (
    WeightConfig,
    GatingWeights,
    ExpertTraversalWeights,
    LearnableWeight,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """TrainingScheduler with tmp checkpoint dir."""
    config = SchedulerConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        buffer_persistence_path=str(tmp_path / "buffer.json"),
    )
    return TrainingScheduler(config=config)


@pytest.fixture
def trained_policy():
    """A GatingPolicy that has been trained (real PyTorch)."""
    from merlt.rlcf.policy_gradient import GatingPolicy
    policy = GatingPolicy(input_dim=1024, hidden_dim=256)
    return policy


@pytest.fixture
def traversal_weights():
    """Sample traversal weights table."""
    return {
        "literal": {"RIFERIMENTO": 0.35, "CITATO_DA": 0.15, "MODIFICA": 0.20},
        "systemic": {"RIFERIMENTO": 0.25, "CITATO_DA": 0.30, "MODIFICA": 0.25},
        "principles": {"RIFERIMENTO": 0.20, "CITATO_DA": 0.10},
        "precedent": {"RIFERIMENTO": 0.40, "CITATO_DA": 0.35},
    }


# =============================================================================
# TEST _extract_weight_config()
# =============================================================================


class TestExtractWeightConfig:
    """Test weight extraction from trained policy."""

    def test_extracts_gating_priors_from_policy(self, scheduler, trained_policy):
        """GatingPolicy softmax output → expert_priors in WeightConfig."""
        config = scheduler._extract_weight_config(trained_policy)

        assert isinstance(config, WeightConfig)
        assert isinstance(config.gating, GatingWeights)

        # Should have all 4 experts
        assert "LiteralExpert" in config.gating.expert_priors
        assert "SystemicExpert" in config.gating.expert_priors
        assert "PrinciplesExpert" in config.gating.expert_priors
        assert "PrecedentExpert" in config.gating.expert_priors

        # Priors should sum to ~1.0 (softmax output)
        total = sum(
            lw.default for lw in config.gating.expert_priors.values()
        )
        assert abs(total - 1.0) < 0.01, f"Expert priors should sum to ~1.0, got {total}"

    def test_extracts_traversal_weights(self, scheduler, trained_policy, traversal_weights):
        """Traversal weights → expert_traversal in WeightConfig."""
        config = scheduler._extract_weight_config(trained_policy, traversal_weights)

        # Should have all 4 experts mapped to PascalCase
        assert "LiteralExpert" in config.expert_traversal
        assert "SystemicExpert" in config.expert_traversal
        assert "PrinciplesExpert" in config.expert_traversal
        assert "PrecedentExpert" in config.expert_traversal

        # Check specific weight
        literal = config.expert_traversal["LiteralExpert"]
        assert isinstance(literal, ExpertTraversalWeights)
        assert "RIFERIMENTO" in literal.weights
        assert literal.weights["RIFERIMENTO"].default == 0.35

    def test_handles_no_traversal_weights(self, scheduler, trained_policy):
        """Without traversal weights, expert_traversal is empty."""
        config = scheduler._extract_weight_config(trained_policy, traversal_weights=None)
        assert config.expert_traversal == {}

    def test_fallback_on_policy_error(self, scheduler):
        """If policy forward fails, uses uniform 0.25 defaults."""
        broken_policy = MagicMock()
        broken_policy.get_expert_priors.side_effect = RuntimeError("broken")
        # Remove hasattr shortcut — spec=[] ensures no auto-attributes
        broken_policy.configure_mock(**{"get_expert_priors.side_effect": RuntimeError("broken")})

        config = scheduler._extract_weight_config(broken_policy)

        # Should still return valid config with defaults
        assert isinstance(config, WeightConfig)
        for lw in config.gating.expert_priors.values():
            assert lw.default == 0.25

    def test_priors_are_learnable_weights(self, scheduler, trained_policy):
        """Each prior is a LearnableWeight with correct bounds."""
        config = scheduler._extract_weight_config(trained_policy)

        for name, lw in config.gating.expert_priors.items():
            assert isinstance(lw, LearnableWeight)
            assert lw.bounds == (0.1, 0.5)
            assert lw.learnable is True


# =============================================================================
# TEST _persist_weight_config()
# =============================================================================


class TestPersistWeightConfig:
    """Test weight persistence to database."""

    @pytest.mark.asyncio
    async def test_skips_when_no_db_url(self, scheduler, trained_policy, monkeypatch):
        """No RLCF_DATABASE_URL → returns None, no error."""
        monkeypatch.delenv("RLCF_DATABASE_URL", raising=False)

        result = await scheduler._persist_weight_config(
            policy=trained_policy,
            checkpoint_version="v20260312",
            samples_processed=100,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_calls_save_weights_when_db_url_set(
        self, scheduler, trained_policy, monkeypatch
    ):
        """With RLCF_DATABASE_URL, calls WeightStore.save_weights()."""
        monkeypatch.setenv("RLCF_DATABASE_URL", "postgresql://fake:5432/test")

        mock_store = MagicMock()
        mock_store.save_weights = AsyncMock(return_value="version-123")

        with patch("merlt.weights.store.WeightStore", return_value=mock_store) as mock_cls:
            result = await scheduler._persist_weight_config(
                policy=trained_policy,
                checkpoint_version="v20260312",
                samples_processed=100,
            )

        assert result == "version-123"
        mock_cls.assert_called_once_with(database_url="postgresql://fake:5432/test")
        mock_store.save_weights.assert_called_once()

        # Check call args
        call_kwargs = mock_store.save_weights.call_args[1]
        assert isinstance(call_kwargs["config"], WeightConfig)
        assert call_kwargs["experiment_id"] == "rlcf_training"
        assert call_kwargs["metrics"]["checkpoint_version"] == "v20260312"
        assert call_kwargs["metrics"]["samples_processed"] == 100.0

    @pytest.mark.asyncio
    async def test_includes_traversal_weights(
        self, scheduler, trained_policy, traversal_weights, monkeypatch
    ):
        """Traversal weights are included in persisted config."""
        monkeypatch.setenv("RLCF_DATABASE_URL", "postgresql://fake:5432/test")

        saved_config = None

        async def capture_save(config, experiment_id, metrics=None):
            nonlocal saved_config
            saved_config = config
            return "version-456"

        mock_store = MagicMock()
        mock_store.save_weights = AsyncMock(side_effect=capture_save)

        with patch("merlt.weights.store.WeightStore", return_value=mock_store):
            await scheduler._persist_weight_config(
                policy=trained_policy,
                checkpoint_version="v1",
                samples_processed=50,
                traversal_weights=traversal_weights,
            )

        assert saved_config is not None
        assert "LiteralExpert" in saved_config.expert_traversal
        assert "RIFERIMENTO" in saved_config.expert_traversal["LiteralExpert"].weights

    @pytest.mark.asyncio
    async def test_graceful_on_save_failure(
        self, scheduler, trained_policy, monkeypatch
    ):
        """WeightStore.save_weights() failure → returns None, no crash."""
        monkeypatch.setenv("RLCF_DATABASE_URL", "postgresql://fake:5432/test")

        mock_store = MagicMock()
        mock_store.save_weights = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("merlt.weights.store.WeightStore", return_value=mock_store):
            result = await scheduler._persist_weight_config(
                policy=trained_policy,
                checkpoint_version="v1",
                samples_processed=50,
            )

        assert result is None


# =============================================================================
# TEST TrainingResult WEIGHT FIELDS
# =============================================================================


class TestTrainingResultWeightFields:
    """Test weight_version_id field on TrainingResult."""

    def test_default_none(self):
        """weight_version_id defaults to None."""
        result = TrainingResult()
        assert result.weight_version_id is None

    def test_in_to_dict(self):
        """weight_version_id appears in to_dict output."""
        result = TrainingResult(weight_version_id="ver-abc-123")
        d = result.to_dict()
        assert d["weight_version_id"] == "ver-abc-123"

    def test_set_value(self):
        """Can set weight_version_id."""
        result = TrainingResult(weight_version_id="ver-xyz")
        assert result.weight_version_id == "ver-xyz"

    def test_none_in_to_dict(self):
        """None weight_version_id serializes as None."""
        result = TrainingResult()
        d = result.to_dict()
        assert d["weight_version_id"] is None


# =============================================================================
# TEST WIRING IN run_training_epoch()
# =============================================================================


class TestRunTrainingEpochWiring:
    """Test that run_training_epoch calls _persist_weight_config."""

    @pytest.mark.asyncio
    async def test_persist_called_after_training(self, tmp_path, monkeypatch):
        """_persist_weight_config is called during run_training_epoch."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        config = SchedulerConfig(
            buffer_threshold=1,
            batch_size=1,
            epochs_per_run=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            buffer_persistence_path=None,
        )
        scheduler = TrainingScheduler(config=config)

        # Add experience via public API
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        persist_called = False

        async def mock_persist(**kwargs):
            nonlocal persist_called
            persist_called = True
            return None

        scheduler._persist_weight_config = mock_persist

        result = await scheduler.run_training_epoch()

        assert persist_called, "_persist_weight_config must be called during training"
        assert result.success is True

    @pytest.mark.asyncio
    async def test_weight_version_in_result(self, tmp_path, monkeypatch):
        """weight_version_id from _persist appears in TrainingResult."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "checkpoints").mkdir()

        config = SchedulerConfig(
            buffer_threshold=1,
            batch_size=1,
            epochs_per_run=1,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            buffer_persistence_path=None,
        )
        scheduler = TrainingScheduler(config=config)

        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        async def mock_persist(**kwargs):
            return "weight-ver-abc"

        scheduler._persist_weight_config = mock_persist

        result = await scheduler.run_training_epoch()

        assert result.weight_version_id == "weight-ver-abc"
