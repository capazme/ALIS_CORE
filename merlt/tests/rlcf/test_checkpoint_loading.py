"""
Test per Training Scheduler Checkpoint Loading (STORY-11-3)
===========================================================

Test per:
- _get_or_create_policy(): fresh vs checkpoint loading
- _save_checkpoint(): versioned + trainer latest + inference latest
- run_training_epoch() incremental training
- Graceful degradation on corrupted checkpoint
- SchedulerConfig.checkpoint_dir propagation
- TrainingResult.loaded_from_checkpoint flag
"""

import pytest
from pathlib import Path
from datetime import datetime, UTC

from merlt.rlcf.training_scheduler import (
    TrainingScheduler,
    SchedulerConfig,
    TrainingResult,
)
from merlt.rlcf.policy_gradient import GatingPolicy, PolicyGradientTrainer
from merlt.experts.neural_gating.neural import ExpertGatingMLP, GatingConfig


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Temporary checkpoint directory."""
    d = tmp_path / "checkpoints"
    d.mkdir()
    return d


@pytest.fixture
def scheduler(checkpoint_dir):
    """Scheduler with temp checkpoint dir and no buffer persistence."""
    config = SchedulerConfig(
        buffer_threshold=2,
        epochs_per_run=1,
        batch_size=2,
        max_buffer_size=100,
        checkpoint_dir=str(checkpoint_dir),
        buffer_persistence_path=None,
        auto_save_checkpoint=True,
    )
    return TrainingScheduler(config=config)


@pytest.fixture
def trace_data():
    """Minimal trace data for ExecutionTrace.from_dict()."""
    return {
        "query_id": "q-ckpt-test",
        "query_text": "Test query for checkpoint",
        "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "actions": [
            {
                "action_type": "expert_selection",
                "expert_type": "literal",
                "parameters": {"weight": 0.4},
                "log_prob": -0.9,
                "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "metadata": {"source": "gating_policy"},
            },
            {
                "action_type": "expert_selection",
                "expert_type": "systemic",
                "parameters": {"weight": 0.3},
                "log_prob": -1.2,
                "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "metadata": {"source": "gating_policy"},
            },
            {
                "action_type": "expert_selection",
                "expert_type": "principles",
                "parameters": {"weight": 0.2},
                "log_prob": -1.6,
                "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "metadata": {"source": "gating_policy"},
            },
            {
                "action_type": "expert_selection",
                "expert_type": "precedent",
                "parameters": {"weight": 0.1},
                "log_prob": -2.3,
                "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
                "metadata": {"source": "gating_policy"},
            },
        ],
        "expert_responses": {},
        "reward": 0.0,
    }


@pytest.fixture
def feedback_data():
    """Minimal feedback data for MultilevelFeedback.from_dict()."""
    return {
        "query_id": "q-ckpt-test",
        "user_id": "user-ckpt-test",
        "timestamp": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "levels": {
            "F7": {
                "level": "F7",
                "score": 0.8,
                "dimension": "overall",
                "metadata": {},
            }
        },
    }


def _fill_buffer(scheduler, trace_data, feedback_data, count=3):
    """Add experiences to the scheduler buffer."""
    for i in range(count):
        td = dict(trace_data)
        td["query_id"] = f"q-ckpt-{i}"
        fd = dict(feedback_data)
        fd["query_id"] = f"q-ckpt-{i}"
        scheduler.add_experience(
            trace=td,
            feedback=fd,
            reward=0.7 + i * 0.05,
        )


# =============================================================================
# TEST _get_or_create_policy()
# =============================================================================


class TestGetOrCreatePolicy:
    """Test GatingPolicy checkpoint loading logic."""

    def test_fresh_policy_when_no_checkpoint(self, scheduler):
        """Creates fresh ExpertGatingMLP when no checkpoint exists."""
        policy, trainer, loaded = scheduler._get_or_create_policy()

        assert isinstance(policy, ExpertGatingMLP)
        assert isinstance(trainer, PolicyGradientTrainer)
        assert loaded is False
        assert trainer.num_updates == 0
        assert trainer.baseline == 0.0

    def test_fresh_policy_uses_1024_input_dim(self, scheduler):
        """Fresh policy uses input_dim=1024 matching E5-large embeddings."""
        policy, _, _ = scheduler._get_or_create_policy()
        assert policy.config.input_dim == 1024

    def test_loads_from_existing_checkpoint(self, scheduler, checkpoint_dir):
        """Loads policy from trainer-format checkpoint when it exists."""
        # Create and save a trainer checkpoint with ExpertGatingMLP
        policy = ExpertGatingMLP(GatingConfig(input_dim=1024))
        trainer = PolicyGradientTrainer(policy)
        trainer.baseline = 0.42
        trainer.num_updates = 17

        trainer_path = checkpoint_dir / "gating_trainer_latest.pt"
        trainer.save_checkpoint(str(trainer_path))

        # Load it back
        loaded_policy, loaded_trainer, loaded = scheduler._get_or_create_policy()

        assert loaded is True
        assert loaded_trainer.num_updates == 17
        assert loaded_trainer.baseline == pytest.approx(0.42)

    def test_corrupted_checkpoint_fallback(self, scheduler, checkpoint_dir):
        """Falls back to fresh policy when checkpoint is corrupted."""
        trainer_path = checkpoint_dir / "gating_trainer_latest.pt"
        trainer_path.write_text("corrupted data")

        policy, trainer, loaded = scheduler._get_or_create_policy()

        assert loaded is False
        assert trainer.num_updates == 0
        assert trainer.baseline == 0.0

    def test_checkpoint_dir_from_config(self, tmp_path):
        """checkpoint_dir propagates from SchedulerConfig."""
        custom_dir = tmp_path / "my_checkpoints"
        custom_dir.mkdir()

        config = SchedulerConfig(
            checkpoint_dir=str(custom_dir),
            buffer_persistence_path=None,
        )
        scheduler = TrainingScheduler(config=config)

        # Create checkpoint in custom dir
        policy = ExpertGatingMLP(GatingConfig(input_dim=1024))
        trainer = PolicyGradientTrainer(policy)
        trainer.num_updates = 99
        trainer.save_checkpoint(str(custom_dir / "gating_trainer_latest.pt"))

        _, loaded_trainer, loaded = scheduler._get_or_create_policy()
        assert loaded is True
        assert loaded_trainer.num_updates == 99


# =============================================================================
# TEST _save_checkpoint()
# =============================================================================


class TestSaveCheckpoint:
    """Test checkpoint saving with versioned + trainer latest + inference latest."""

    def test_saves_trainer_latest(self, scheduler, checkpoint_dir):
        """_save_checkpoint creates gating_trainer_latest.pt for training resumption."""
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)

        version = scheduler._save_checkpoint(policy, trainer)

        assert version is not None
        assert version.startswith("v")
        assert (checkpoint_dir / "gating_trainer_latest.pt").exists()

    def test_saves_inference_latest(self, scheduler, checkpoint_dir):
        """_save_checkpoint creates gating_policy_latest.pt for PolicyManager inference."""
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)

        scheduler._save_checkpoint(policy, trainer)

        assert (checkpoint_dir / "gating_policy_latest.pt").exists()

    def test_saves_versioned_checkpoint(self, scheduler, checkpoint_dir):
        """_save_checkpoint creates versioned .pt file."""
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)

        scheduler._save_checkpoint(policy, trainer)

        versioned_files = list(checkpoint_dir.glob("gating_v*.pt"))
        assert len(versioned_files) >= 1

    def test_trainer_latest_is_loadable(self, scheduler, checkpoint_dir):
        """Saved trainer checkpoint can be loaded back with full state."""
        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)
        trainer.baseline = 0.55
        trainer.num_updates = 10

        scheduler._save_checkpoint(policy, trainer)

        # Load back via trainer format
        new_policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        new_trainer = PolicyGradientTrainer(new_policy)
        new_trainer.load_checkpoint(str(checkpoint_dir / "gating_trainer_latest.pt"))

        assert new_trainer.baseline == pytest.approx(0.55)
        assert new_trainer.num_updates == 10

    def test_inference_latest_has_mlp_state_dict(self, scheduler, checkpoint_dir):
        """Inference checkpoint uses mlp_state_dict format for PolicyManager."""
        import torch

        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)

        scheduler._save_checkpoint(policy, trainer)

        ckpt = torch.load(
            str(checkpoint_dir / "gating_policy_latest.pt"),
            map_location="cpu",
        )
        # PolicyManager format uses mlp_state_dict key
        assert "mlp_state_dict" in ckpt
        assert "input_dim" in ckpt
        assert ckpt["input_dim"] == 1024

    def test_two_formats_coexist(self, scheduler, checkpoint_dir):
        """Trainer and inference formats are separate files, not overwriting each other."""
        import torch

        policy = GatingPolicy(input_dim=1024, hidden_dim=256)
        trainer = PolicyGradientTrainer(policy)
        trainer.num_updates = 7

        scheduler._save_checkpoint(policy, trainer)

        # Trainer format has optimizer_state_dict
        trainer_ckpt = torch.load(
            str(checkpoint_dir / "gating_trainer_latest.pt"),
            map_location="cpu",
        )
        assert "optimizer_state_dict" in trainer_ckpt
        assert trainer_ckpt["num_updates"] == 7

        # Inference format has mlp_state_dict (no optimizer)
        inference_ckpt = torch.load(
            str(checkpoint_dir / "gating_policy_latest.pt"),
            map_location="cpu",
        )
        assert "mlp_state_dict" in inference_ckpt
        assert "optimizer_state_dict" not in inference_ckpt


# =============================================================================
# TEST run_training_epoch() INTEGRATION
# =============================================================================


class TestRunTrainingEpochCheckpoint:
    """Test run_training_epoch with checkpoint loading and saving."""

    @pytest.mark.asyncio
    async def test_first_run_fresh_policy(self, scheduler, trace_data, feedback_data):
        """First run uses fresh policy (no checkpoint)."""
        _fill_buffer(scheduler, trace_data, feedback_data)

        result = await scheduler.run_training_epoch()

        assert result.success is True
        assert result.loaded_from_checkpoint is False

    @pytest.mark.asyncio
    async def test_first_run_saves_both_checkpoints(self, scheduler, checkpoint_dir, trace_data, feedback_data):
        """First run saves both trainer and inference checkpoints."""
        _fill_buffer(scheduler, trace_data, feedback_data)

        result = await scheduler.run_training_epoch()

        assert result.success is True
        assert result.checkpoint_version is not None
        assert (checkpoint_dir / "gating_trainer_latest.pt").exists()
        assert (checkpoint_dir / "gating_policy_latest.pt").exists()

    @pytest.mark.asyncio
    async def test_second_run_loads_checkpoint(self, scheduler, checkpoint_dir, trace_data, feedback_data):
        """Second run loads checkpoint from first run."""
        _fill_buffer(scheduler, trace_data, feedback_data)

        # First run — fresh
        result1 = await scheduler.run_training_epoch()
        assert result1.loaded_from_checkpoint is False
        assert (checkpoint_dir / "gating_trainer_latest.pt").exists()

        # Reset scheduler state (simulate process restart)
        from merlt.rlcf.training_scheduler import TrainingStatus
        scheduler._status = TrainingStatus.IDLE
        scheduler._last_training_at = None

        # Refill buffer
        _fill_buffer(scheduler, trace_data, feedback_data)

        # Second run — should load checkpoint
        result2 = await scheduler.run_training_epoch()
        assert result2.loaded_from_checkpoint is True

    @pytest.mark.asyncio
    async def test_incremental_training(self, scheduler, checkpoint_dir, trace_data, feedback_data):
        """Two consecutive runs accumulate trainer state (num_updates, baseline)."""
        import torch

        _fill_buffer(scheduler, trace_data, feedback_data, count=5)

        # Run 1: fresh → trains → saves checkpoint
        result1 = await scheduler.run_training_epoch()
        assert result1.success is True
        assert result1.loaded_from_checkpoint is False

        # Read num_updates from saved checkpoint
        ckpt1 = torch.load(
            str(checkpoint_dir / "gating_trainer_latest.pt"),
            map_location="cpu",
        )
        num_updates_after_run1 = ckpt1["num_updates"]
        assert num_updates_after_run1 > 0

        # Reset state for second run
        from merlt.rlcf.training_scheduler import TrainingStatus
        scheduler._status = TrainingStatus.IDLE
        scheduler._last_training_at = None
        _fill_buffer(scheduler, trace_data, feedback_data, count=5)

        # Run 2: loads checkpoint → continues training
        result2 = await scheduler.run_training_epoch()
        assert result2.success is True
        assert result2.loaded_from_checkpoint is True

        # Read num_updates from updated checkpoint
        ckpt2 = torch.load(
            str(checkpoint_dir / "gating_trainer_latest.pt"),
            map_location="cpu",
        )
        num_updates_after_run2 = ckpt2["num_updates"]

        # Trainer state accumulated: num_updates grew across runs
        assert num_updates_after_run2 > num_updates_after_run1

    @pytest.mark.asyncio
    async def test_loaded_from_checkpoint_in_result_dict(self, scheduler, trace_data, feedback_data):
        """loaded_from_checkpoint appears in to_dict() output."""
        _fill_buffer(scheduler, trace_data, feedback_data)
        result = await scheduler.run_training_epoch()

        d = result.to_dict()
        assert "loaded_from_checkpoint" in d
        assert d["loaded_from_checkpoint"] is False

    @pytest.mark.asyncio
    async def test_checkpoint_dir_in_config_dict(self):
        """checkpoint_dir appears in SchedulerConfig.to_dict()."""
        config = SchedulerConfig(checkpoint_dir="/tmp/test_ckpt")
        d = config.to_dict()
        assert d["checkpoint_dir"] == "/tmp/test_ckpt"

    @pytest.mark.asyncio
    async def test_auto_save_disabled_no_checkpoint(self, checkpoint_dir, trace_data, feedback_data):
        """With auto_save_checkpoint=False, no checkpoint files are created."""
        config = SchedulerConfig(
            buffer_threshold=2,
            epochs_per_run=1,
            batch_size=2,
            max_buffer_size=100,
            checkpoint_dir=str(checkpoint_dir),
            buffer_persistence_path=None,
            auto_save_checkpoint=False,
        )
        scheduler = TrainingScheduler(config=config)
        _fill_buffer(scheduler, trace_data, feedback_data)

        result = await scheduler.run_training_epoch()

        assert result.success is True
        assert result.checkpoint_version is None
        assert not (checkpoint_dir / "gating_trainer_latest.pt").exists()
        assert not (checkpoint_dir / "gating_policy_latest.pt").exists()

    @pytest.mark.asyncio
    async def test_corrupted_checkpoint_still_trains(self, checkpoint_dir, trace_data, feedback_data):
        """Training succeeds even with corrupted checkpoint (fallback to fresh)."""
        # Write corrupted trainer checkpoint
        (checkpoint_dir / "gating_trainer_latest.pt").write_text("bad data")

        config = SchedulerConfig(
            buffer_threshold=2,
            epochs_per_run=1,
            batch_size=2,
            max_buffer_size=100,
            checkpoint_dir=str(checkpoint_dir),
            buffer_persistence_path=None,
        )
        scheduler = TrainingScheduler(config=config)
        _fill_buffer(scheduler, trace_data, feedback_data)

        result = await scheduler.run_training_epoch()

        assert result.success is True
        assert result.loaded_from_checkpoint is False


# =============================================================================
# TEST TrainingResult DATACLASS
# =============================================================================


class TestTrainingResultFields:
    """Test TrainingResult loaded_from_checkpoint field."""

    def test_default_false(self):
        """loaded_from_checkpoint defaults to False."""
        result = TrainingResult()
        assert result.loaded_from_checkpoint is False

    def test_in_to_dict(self):
        """loaded_from_checkpoint is in to_dict output."""
        result = TrainingResult(loaded_from_checkpoint=True)
        d = result.to_dict()
        assert d["loaded_from_checkpoint"] is True

    def test_set_true(self):
        """Can set loaded_from_checkpoint=True."""
        result = TrainingResult(loaded_from_checkpoint=True)
        assert result.loaded_from_checkpoint is True
