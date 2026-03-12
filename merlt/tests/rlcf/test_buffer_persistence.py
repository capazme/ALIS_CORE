"""
Test per Buffer Persistence (STORY-11-2)
==========================================

Test per:
- PrioritizedReplayBuffer.save() / load()
- TrainingScheduler auto-save/load wiring
- Error handling (corrupted files, missing files)
"""

import json
import os
import pytest
import tempfile
from pathlib import Path

from merlt.rlcf.replay_buffer import (
    Experience,
    ExperienceReplayBuffer,
    PrioritizedReplayBuffer,
)
from merlt.rlcf.training_scheduler import (
    TrainingScheduler,
    SchedulerConfig,
    TrainingTrigger,
)


# =============================================================================
# TEST PrioritizedReplayBuffer SAVE / LOAD
# =============================================================================


class TestPrioritizedReplayBufferPersistence:
    """Test save/load per PrioritizedReplayBuffer."""

    def _populate_buffer(self, buffer, count=10):
        """Helper: popola buffer con esperienze di test."""
        for i in range(count):
            buffer.add(
                trace={"query": f"test_query_{i}", "expert": "literal"},
                feedback={"score": 0.5 + i * 0.05, "reason": "test"},
                reward=0.3 + i * 0.07,
                td_error=0.1 * (i + 1),
                metadata={"source": "test", "index": i},
            )

    def test_save_creates_file(self):
        """save() crea il file JSON."""
        buffer = PrioritizedReplayBuffer(capacity=100)
        self._populate_buffer(buffer, 5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer.save(path)
            assert os.path.exists(path)

            with open(path) as f:
                data = json.load(f)

            assert data["buffer_type"] == "prioritized"
            assert data["capacity"] == 100
            assert len(data["experiences"]) == 5
        finally:
            os.unlink(path)

    def test_save_load_roundtrip(self):
        """save() → load() preserva tutte le esperienze."""
        buffer1 = PrioritizedReplayBuffer(capacity=100, alpha=0.6, epsilon=0.01)
        self._populate_buffer(buffer1, 10)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100)
            buffer2.load(path)

            # Same number of experiences
            assert len(buffer2) == len(buffer1)

            # Config preserved
            assert buffer2.alpha == buffer1.alpha
            assert buffer2.epsilon == buffer1.epsilon
            assert buffer2._total_added == buffer1._total_added
            assert buffer2._total_sampled == buffer1._total_sampled
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_experience_fields(self):
        """Roundtrip preserva tutti i campi di ogni Experience."""
        buffer1 = PrioritizedReplayBuffer(capacity=100)
        buffer1.add(
            trace={"query": "art. 1218 c.c.", "context": {"domain": "civile"}},
            feedback={"score": 0.85, "reason": "accurate"},
            reward=0.9,
            td_error=0.5,
            metadata={"user_id": "u123", "session": "s456"},
        )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100)
            buffer2.load(path)

            # Sample the single experience
            batch = buffer2.sample(1)
            assert len(batch) == 1

            exp = batch[0]
            assert exp.trace_data["query"] == "art. 1218 c.c."
            assert exp.trace_data["context"]["domain"] == "civile"
            assert exp.feedback_data["score"] == 0.85
            assert exp.reward == 0.9
            assert exp.metadata["user_id"] == "u123"
            assert exp.experience_id.startswith("exp_")
            assert exp.timestamp  # non-empty
        finally:
            os.unlink(path)

    def test_roundtrip_rebuilds_sumtree_priorities(self):
        """Load ricostruisce il SumTree con priorità corrette."""
        buffer1 = PrioritizedReplayBuffer(capacity=100, alpha=0.6, epsilon=0.01)

        # Add experiences with different rewards → different priorities
        buffer1.add({"q": "low"}, {}, reward=0.1, td_error=0.1)
        buffer1.add({"q": "high"}, {}, reward=5.0, td_error=5.0)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100, alpha=0.6, epsilon=0.01)
            buffer2.load(path)

            # SumTree total should be > 0
            assert buffer2.tree.total() > 0
            assert buffer2.tree.n_entries == 2

            # Sampling should work (proves tree is valid)
            batch, indices, weights = buffer2.sample_with_priority(2, beta=0.4)
            assert len(batch) == 2
            assert all(w > 0 for w in weights)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_oldest_timestamp(self):
        """Load ricostruisce _oldest_ts correttamente."""
        buffer1 = PrioritizedReplayBuffer(capacity=100)
        self._populate_buffer(buffer1, 5)

        oldest_before = buffer1.oldest_timestamp()
        assert oldest_before is not None

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100)
            buffer2.load(path)

            oldest_after = buffer2.oldest_timestamp()
            assert oldest_after is not None
            assert oldest_after == oldest_before
        finally:
            os.unlink(path)

    def test_save_empty_buffer(self):
        """save() funziona su buffer vuoto."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100)
            buffer2.load(path)
            assert len(buffer2) == 0
        finally:
            os.unlink(path)

    def test_load_nonexistent_file_raises(self):
        """load() con file inesistente solleva FileNotFoundError."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        with pytest.raises(FileNotFoundError):
            buffer.load("/nonexistent/path/buffer.json")

    def test_load_corrupted_file_raises(self):
        """load() con file corrotto solleva eccezione."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            f.write("not valid json {{{")
            path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                buffer.load(path)
        finally:
            os.unlink(path)

    def test_load_wrong_buffer_type_raises(self):
        """load() con buffer_type sbagliato solleva ValueError."""
        buffer = PrioritizedReplayBuffer(capacity=100)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            json.dump({"buffer_type": "standard", "experiences": []}, f)
            path = f.name

        try:
            with pytest.raises(ValueError, match="Expected buffer_type='prioritized'"):
                buffer.load(path)
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_original_priority(self):
        """Roundtrip preserva la priorità originale (da td_error), non ricalcola da reward."""
        buffer1 = PrioritizedReplayBuffer(capacity=100, alpha=0.6, epsilon=0.01)

        # High reward but low td_error → low priority
        buffer1.add({"q": "learned"}, {}, reward=0.95, td_error=0.01)
        # Low reward but high td_error → high priority
        buffer1.add({"q": "surprise"}, {}, reward=0.1, td_error=5.0)

        # Get priorities before save
        exp_learned = buffer1.tree.data[0]
        exp_surprise = buffer1.tree.data[1]
        priority_learned_before = exp_learned.priority
        priority_surprise_before = exp_surprise.priority

        # Surprise should have much higher priority (td_error driven)
        assert priority_surprise_before > priority_learned_before * 5

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = PrioritizedReplayBuffer(capacity=100, alpha=0.6, epsilon=0.01)
            buffer2.load(path)

            # Verify priorities are preserved, not recalculated from reward
            loaded_exps = [
                buffer2.tree.data[i]
                for i in range(buffer2.tree.n_entries)
                if buffer2.tree.data[i] is not None
            ]

            loaded_by_query = {e.trace_data["q"]: e for e in loaded_exps}
            assert loaded_by_query["learned"].priority == pytest.approx(
                priority_learned_before
            )
            assert loaded_by_query["surprise"].priority == pytest.approx(
                priority_surprise_before
            )
        finally:
            os.unlink(path)


# =============================================================================
# TEST ExperienceReplayBuffer SAVE / LOAD (regression)
# =============================================================================


class TestExperienceReplayBufferPersistenceRegression:
    """Regression test: ExperienceReplayBuffer.save/load continua a funzionare."""

    def test_save_load_roundtrip(self):
        """ExperienceReplayBuffer save/load roundtrip preserva esperienze."""
        buffer1 = ExperienceReplayBuffer(capacity=100)
        for i in range(5):
            buffer1.add(
                trace={"q": f"query_{i}"},
                feedback={"s": i * 0.2},
                reward=i * 0.1,
            )

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            buffer2 = ExperienceReplayBuffer(capacity=100)
            buffer2.load(path)

            assert len(buffer2) == 5
            assert buffer2._total_added == 5
        finally:
            os.unlink(path)


# =============================================================================
# TEST TrainingScheduler AUTO-PERSISTENCE WIRING
# =============================================================================


class TestSchedulerBufferPersistence:
    """Test auto-save/load wiring nel TrainingScheduler."""

    def test_config_has_buffer_persistence_path(self):
        """SchedulerConfig ha buffer_persistence_path con default."""
        config = SchedulerConfig()
        assert config.buffer_persistence_path == "data/rlcf/replay_buffer.json"

    def test_config_persistence_path_in_dict(self):
        """buffer_persistence_path appare in to_dict()."""
        config = SchedulerConfig()
        d = config.to_dict()
        assert "buffer_persistence_path" in d

    def test_config_persistence_path_none_disables(self):
        """buffer_persistence_path=None disabilita la persistenza."""
        config = SchedulerConfig(buffer_persistence_path=None)
        scheduler = TrainingScheduler(config)
        # Should not crash, just skip load
        assert len(scheduler.buffer) == 0

    def test_auto_load_on_init_missing_file(self):
        """Init con file mancante → buffer vuoto, nessun errore."""
        config = SchedulerConfig(
            buffer_persistence_path="/nonexistent/path/buffer.json"
        )
        scheduler = TrainingScheduler(config)
        assert len(scheduler.buffer) == 0

    def test_auto_load_on_init_existing_file(self):
        """Init carica buffer da file esistente."""
        # Create a buffer and save it
        buffer = PrioritizedReplayBuffer(capacity=100)
        for i in range(5):
            buffer.add({"q": i}, {}, reward=0.5)

        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, dir=tempfile.gettempdir()
        ) as f:
            path = f.name

        try:
            buffer.save(path)

            # Create scheduler pointing to that file
            config = SchedulerConfig(buffer_persistence_path=path)
            scheduler = TrainingScheduler(config)

            assert len(scheduler.buffer) == 5
        finally:
            os.unlink(path)

    def test_auto_load_on_init_corrupted_file(self):
        """Init con file corrotto → buffer vuoto, warning loggato."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w"
        ) as f:
            f.write("corrupted data!!!")
            path = f.name

        try:
            config = SchedulerConfig(buffer_persistence_path=path)
            scheduler = TrainingScheduler(config)

            # Should start with empty buffer, not crash
            assert len(scheduler.buffer) == 0
        finally:
            os.unlink(path)

    def test_try_save_buffer_creates_dirs(self):
        """_try_save_buffer crea directory parent se non esistono."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = os.path.join(tmpdir, "sub", "dir", "buffer.json")

            config = SchedulerConfig(buffer_persistence_path=nested_path)
            scheduler = TrainingScheduler(config)

            # Add some data
            scheduler.add_experience({"q": "test"}, {}, reward=0.5)

            # Save
            scheduler._try_save_buffer()

            assert os.path.exists(nested_path)

    def test_try_save_buffer_content(self):
        """_try_save_buffer salva il contenuto corretto."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            config = SchedulerConfig(buffer_persistence_path=path)
            scheduler = TrainingScheduler(config)

            for i in range(3):
                scheduler.add_experience({"q": f"q{i}"}, {}, reward=0.5)

            scheduler._try_save_buffer()

            # Verify file content
            with open(path) as f:
                data = json.load(f)

            assert data["buffer_type"] == "prioritized"
            assert len(data["experiences"]) == 3
        finally:
            os.unlink(path)

    @pytest.mark.asyncio
    async def test_run_training_epoch_triggers_save(self):
        """run_training_epoch salva il buffer su disco dopo training reale."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            config = SchedulerConfig(
                buffer_persistence_path=path,
                buffer_threshold=2,
                epochs_per_run=1,
                batch_size=2,
                auto_save_checkpoint=False,
            )
            scheduler = TrainingScheduler(config)

            # Add experiences with valid trace/feedback structures
            for i in range(5):
                trace_data = {
                    "query_id": f"test_q_{i}",
                    "actions": [
                        {
                            "action_type": "expert_selection",
                            "parameters": {"expert_type": "literal", "weight": 0.7},
                            "log_prob": -0.357,
                            "metadata": {"source": "gating_policy"},
                        }
                    ],
                }
                feedback_data = {
                    "query_id": f"test_q_{i}",
                    "overall_rating": 0.5 + i * 0.05,
                }
                scheduler.add_experience(
                    trace_data, feedback_data, reward=0.5 + i * 0.05
                )

            # Remove any pre-existing file content (from auto-load attempt)
            if os.path.exists(path):
                os.unlink(path)

            result = await scheduler.run_training_epoch(trigger=TrainingTrigger.MANUAL)

            # Buffer file must exist after successful training
            assert os.path.exists(path), "Buffer file not saved after training"

            with open(path) as f:
                data = json.load(f)

            assert data["buffer_type"] == "prioritized"
            assert len(data["experiences"]) == 5
            assert result.success is True
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_save_load_cycle_via_scheduler(self):
        """Full cycle: scheduler1 saves → scheduler2 loads."""
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False
        ) as f:
            path = f.name

        try:
            # Scheduler 1: add experiences and save
            config1 = SchedulerConfig(buffer_persistence_path=path)
            s1 = TrainingScheduler(config1)
            for i in range(7):
                s1.add_experience(
                    {"q": f"query_{i}"}, {"fb": i}, reward=0.3 + i * 0.1
                )
            s1._try_save_buffer()

            # Scheduler 2: should auto-load on init
            config2 = SchedulerConfig(buffer_persistence_path=path)
            s2 = TrainingScheduler(config2)

            assert len(s2.buffer) == 7
        finally:
            os.unlink(path)
