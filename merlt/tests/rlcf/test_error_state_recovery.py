"""
Test TrainingScheduler ERROR state recovery (STORY-12-3)
=========================================================

Tests for:
- ERROR state blocks should_train() during cooldown
- ERROR -> IDLE transition after cooldown elapses
- get_status() includes error_cooldown_remaining_seconds
- error_cooldown_seconds is configurable

Example:
    pytest tests/rlcf/test_error_state_recovery.py -v
"""

import time
import pytest
from unittest.mock import patch

from merlt.rlcf.training_scheduler import (
    TrainingScheduler,
    TrainingStatus,
    TrainingTrigger,
    SchedulerConfig,
    SchedulerStatus,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def scheduler(tmp_path):
    """Scheduler with low thresholds for testing."""
    config = SchedulerConfig(
        buffer_threshold=1,
        error_cooldown_seconds=2,  # 2 seconds for fast tests
        checkpoint_dir=str(tmp_path / "checkpoints"),
        buffer_persistence_path=None,
    )
    return TrainingScheduler(config=config)


# =============================================================================
# TEST ERROR STATE BLOCKS should_train()
# =============================================================================


class TestErrorStateBlocking:
    """ERROR state prevents training during cooldown."""

    def test_error_state_blocks_should_train(self, scheduler):
        """should_train() returns False while in ERROR state within cooldown."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic()

        # Add experience so buffer threshold is met
        from unittest.mock import MagicMock
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        assert scheduler.should_train() is False

    def test_idle_state_allows_should_train(self, scheduler):
        """should_train() returns True when IDLE and buffer has data."""
        from unittest.mock import MagicMock
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        assert scheduler._status == TrainingStatus.IDLE
        assert scheduler.should_train() is True

    def test_error_without_timestamp_blocks(self, scheduler):
        """ERROR state with no timestamp still blocks (defensive)."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = None

        assert scheduler.should_train() is False


# =============================================================================
# TEST ERROR -> IDLE RECOVERY
# =============================================================================


class TestErrorRecovery:
    """ERROR transitions to IDLE after cooldown elapses."""

    def test_recovery_after_cooldown(self, scheduler):
        """should_train() resets ERROR->IDLE after cooldown."""
        scheduler._status = TrainingStatus.ERROR
        # Set timestamp far enough in the past
        scheduler._error_timestamp = time.monotonic() - 10  # 10s ago, cooldown is 2s

        from unittest.mock import MagicMock
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        result = scheduler.should_train()

        assert result is True
        assert scheduler._status == TrainingStatus.IDLE
        assert scheduler._error_timestamp is None

    def test_no_recovery_before_cooldown(self, scheduler):
        """should_train() keeps ERROR state if cooldown not elapsed."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic()  # just now

        assert scheduler.should_train() is False
        assert scheduler._status == TrainingStatus.ERROR

    def test_custom_cooldown_seconds(self, tmp_path):
        """error_cooldown_seconds is configurable."""
        config = SchedulerConfig(
            error_cooldown_seconds=600,  # 10 minutes
            checkpoint_dir=str(tmp_path / "checkpoints"),
            buffer_persistence_path=None,
        )
        scheduler = TrainingScheduler(config=config)
        assert scheduler.config.error_cooldown_seconds == 600

        # Set error 5 minutes ago — still within 10 min cooldown
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic() - 300

        assert scheduler.should_train() is False

    def test_cooldown_in_config_to_dict(self):
        """error_cooldown_seconds appears in config.to_dict()."""
        config = SchedulerConfig(error_cooldown_seconds=120)
        d = config.to_dict()
        assert d["error_cooldown_seconds"] == 120


# =============================================================================
# TEST get_status() ERROR INFO
# =============================================================================


class TestGetStatusErrorInfo:
    """get_status() includes error cooldown information."""

    def test_idle_has_no_cooldown(self, scheduler):
        """IDLE state has error_cooldown_remaining_seconds=None."""
        status = scheduler.get_status()
        assert status.error_cooldown_remaining_seconds is None

    def test_error_has_cooldown_remaining(self, scheduler):
        """ERROR state shows remaining cooldown seconds."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic() - 1  # 1 second ago

        status = scheduler.get_status()
        assert status.error_cooldown_remaining_seconds is not None
        # Cooldown is 2s, elapsed 1s => ~1s remaining
        assert 0.0 < status.error_cooldown_remaining_seconds <= 2.0

    def test_error_cooldown_in_to_dict(self, scheduler):
        """error_cooldown_remaining_seconds appears in to_dict."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic()

        d = scheduler.get_status().to_dict()
        assert "error_cooldown_remaining_seconds" in d
        assert d["error_cooldown_remaining_seconds"] is not None

    def test_idle_cooldown_none_in_to_dict(self, scheduler):
        """IDLE state: error_cooldown_remaining_seconds is None in to_dict."""
        d = scheduler.get_status().to_dict()
        assert d["error_cooldown_remaining_seconds"] is None

    def test_cooldown_remaining_never_negative(self, scheduler):
        """Cooldown remaining is clamped to 0.0, never negative."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic() - 100  # way past cooldown

        status = scheduler.get_status()
        assert status.error_cooldown_remaining_seconds == 0.0


# =============================================================================
# TEST run_training_epoch() ERROR GUARD
# =============================================================================


class TestRunTrainingEpochErrorGuard:
    """run_training_epoch() respects ERROR cooldown."""

    @pytest.mark.asyncio
    async def test_blocks_during_cooldown(self, scheduler):
        """Direct call to run_training_epoch() blocked during cooldown."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic()  # just now, within 2s cooldown

        result = await scheduler.run_training_epoch(trigger=TrainingTrigger.MANUAL)

        assert result.success is False
        assert "cooldown" in result.error.lower()
        assert scheduler._status == TrainingStatus.ERROR

    @pytest.mark.asyncio
    async def test_allows_after_cooldown(self, scheduler):
        """Direct call allowed after cooldown, clears error state."""
        scheduler._status = TrainingStatus.ERROR
        scheduler._error_timestamp = time.monotonic() - 10  # 10s ago, cooldown is 2s

        # Add experience so training has data
        from unittest.mock import MagicMock
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test", "experts": {}, "expert_weights": [0.25] * 4}
        feedback = MagicMock()
        feedback.to_dict.return_value = {"overall": 4, "dimensions": {}}
        scheduler.add_experience(trace, feedback, reward=0.8)

        result = await scheduler.run_training_epoch(trigger=TrainingTrigger.MANUAL)

        # Training runs (may succeed or fail depending on policy availability,
        # but the ERROR guard should not block it)
        assert scheduler._error_timestamp is None
