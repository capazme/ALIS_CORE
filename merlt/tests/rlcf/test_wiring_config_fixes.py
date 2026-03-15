"""
Tests for Sprint F wiring and config fixes (I4, I6).

I4: TraversalTrainingService reuses existing instance (no double-instantiation)
I6: experiment_id is configurable via SchedulerConfig (not hardcoded)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from merlt.rlcf.training_scheduler import SchedulerConfig, TrainingScheduler


# =============================================================================
# I6 — experiment_id configurable in SchedulerConfig
# =============================================================================


class TestSchedulerConfigExperimentId:
    def test_default_is_rlcf_training(self):
        config = SchedulerConfig()
        assert config.experiment_id == "rlcf_training"

    def test_custom_experiment_id(self):
        config = SchedulerConfig(experiment_id="exp_ablation_v2")
        assert config.experiment_id == "exp_ablation_v2"

    def test_experiment_id_in_to_dict(self):
        config = SchedulerConfig(experiment_id="exp_test")
        d = config.to_dict()
        assert "experiment_id" in d
        assert d["experiment_id"] == "exp_test"

    def test_to_dict_default_value(self):
        config = SchedulerConfig()
        assert config.to_dict()["experiment_id"] == "rlcf_training"


class TestPersistWeightConfigUsesConfigExperimentId:
    """_persist_weight_config must use self.config.experiment_id, not a hardcoded string."""

    @pytest.mark.asyncio
    async def test_custom_experiment_id_passed_to_save_weights(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLCF_DATABASE_URL", "postgresql://fake:5432/test")

        config = SchedulerConfig(
            experiment_id="custom_exp_abc",
            buffer_persistence_path=None,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        scheduler = TrainingScheduler(config=config)

        from merlt.rlcf.policy_gradient import GatingPolicy
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)

        mock_store = MagicMock()
        mock_store.save_weights = AsyncMock(return_value="ver-xyz")

        with patch("merlt.weights.store.WeightStore", return_value=mock_store):
            result = await scheduler._persist_weight_config(
                policy=policy,
                checkpoint_version="v1",
                samples_processed=10,
            )

        assert result == "ver-xyz"
        call_kwargs = mock_store.save_weights.call_args[1]
        assert call_kwargs["experiment_id"] == "custom_exp_abc"

    @pytest.mark.asyncio
    async def test_default_experiment_id_is_rlcf_training(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RLCF_DATABASE_URL", "postgresql://fake:5432/test")

        config = SchedulerConfig(
            buffer_persistence_path=None,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        scheduler = TrainingScheduler(config=config)

        from merlt.rlcf.policy_gradient import GatingPolicy
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)

        mock_store = MagicMock()
        mock_store.save_weights = AsyncMock(return_value="ver-default")

        with patch("merlt.weights.store.WeightStore", return_value=mock_store):
            await scheduler._persist_weight_config(
                policy=policy,
                checkpoint_version="v1",
                samples_processed=10,
            )

        call_kwargs = mock_store.save_weights.call_args[1]
        assert call_kwargs["experiment_id"] == "rlcf_training"


# =============================================================================
# I4 — TraversalTrainingService not double-instantiated
# =============================================================================


class TestTraversalTrainingServiceNoDoubleInstantiation:
    """TraversalTrainingService must be instantiated once per training epoch."""

    @pytest.mark.asyncio
    async def test_single_instantiation_when_traversal_trained(self, tmp_path, monkeypatch):
        """When traversal training succeeds, TraversalTrainingService is created once."""
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

        instantiation_count = 0

        class MockTraversalTrainingService:
            MIN_SAMPLES = 0

            def __init__(self):
                nonlocal instantiation_count
                instantiation_count += 1

            async def prepare_training_data(self, session):
                return [MagicMock()] * 25

            async def train_traversal_policy(self, samples):
                result = MagicMock()
                result.epochs_completed = 1
                result.samples_used = 25
                result.to_dict.return_value = {}
                return result

            def get_domain_weights_table(self):
                return {"literal": {"RIFERIMENTO": 0.3}}

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=MagicMock())
        mock_session.__aexit__ = AsyncMock(return_value=False)

        scheduler._persist_weight_config = AsyncMock(return_value=None)

        with patch(
            "merlt.rlcf.traversal_training_service.TraversalTrainingService",
            MockTraversalTrainingService,
        ), patch(
            "merlt.rlcf.database.get_async_session",
            return_value=mock_session,
        ):
            await scheduler.run_training_epoch()

        assert instantiation_count == 1, (
            f"TraversalTrainingService should be instantiated once, got {instantiation_count}"
        )
