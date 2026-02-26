"""
Test per Experience Replay Buffer (Core)
==========================================

Test per Experience, BufferStats, ExperienceReplayBuffer, factory, e integrazione.
"""

import pytest
import tempfile
import os
import threading
import time
from datetime import datetime
from unittest.mock import MagicMock

from merlt.rlcf.replay_buffer import (
    Experience,
    BufferStats,
    ExperienceReplayBuffer,
    PrioritizedReplayBuffer,
    create_replay_buffer
)


# =============================================================================
# TEST EXPERIENCE DATACLASS
# =============================================================================

class TestExperience:
    """Test per Experience dataclass."""

    def test_create_experience(self):
        """Test creazione esperienza."""
        exp = Experience(
            experience_id="exp_1",
            trace_data={"query": "test"},
            feedback_data={"score": 0.8},
            reward=0.75
        )

        assert exp.experience_id == "exp_1"
        assert exp.reward == 0.75
        assert exp.priority == 1.0  # default

    def test_experience_with_metadata(self):
        """Test esperienza con metadata."""
        exp = Experience(
            experience_id="exp_2",
            trace_data={"query": "test"},
            feedback_data={"score": 0.8},
            reward=0.5,
            priority=0.9,
            metadata={"source": "test"}
        )

        assert exp.priority == 0.9
        assert exp.metadata["source"] == "test"

    def test_to_dict(self):
        """Test serializzazione in dict."""
        exp = Experience(
            experience_id="exp_3",
            trace_data={"query": "test"},
            feedback_data={"score": 0.8},
            reward=0.6,
            priority=0.7
        )

        data = exp.to_dict()

        assert data["experience_id"] == "exp_3"
        assert data["reward"] == 0.6
        assert data["priority"] == 0.7
        assert "timestamp" in data

    def test_from_dict(self):
        """Test deserializzazione da dict."""
        data = {
            "experience_id": "exp_4",
            "trace_data": {"query": "test"},
            "feedback_data": {"score": 0.9},
            "reward": 0.85,
            "priority": 0.6,
            "timestamp": "2025-01-01T12:00:00",
            "metadata": {"key": "value"}
        }

        exp = Experience.from_dict(data)

        assert exp.experience_id == "exp_4"
        assert exp.reward == 0.85
        assert exp.metadata["key"] == "value"

    def test_from_dict_defaults(self):
        """Test deserializzazione con defaults."""
        data = {
            "experience_id": "exp_5",
            "trace_data": {},
            "feedback_data": {},
            "reward": 0.5
        }

        exp = Experience.from_dict(data)

        assert exp.priority == 1.0  # default
        assert exp.metadata == {}  # default


# =============================================================================
# TEST BUFFER STATS
# =============================================================================

class TestBufferStats:
    """Test per BufferStats."""

    def test_create_stats(self):
        """Test creazione stats."""
        stats = BufferStats(
            size=100,
            capacity=1000,
            total_added=150,
            total_sampled=200
        )

        assert stats.size == 100
        assert stats.capacity == 1000

    def test_stats_to_dict(self):
        """Test serializzazione stats."""
        stats = BufferStats(
            size=50,
            capacity=100,
            total_added=60,
            total_sampled=30,
            avg_reward=0.756789,
            avg_priority=0.912345
        )

        data = stats.to_dict()

        assert data["size"] == 50
        assert data["fill_ratio"] == 0.5
        assert data["avg_reward"] == 0.7568  # rounded
        assert data["avg_priority"] == 0.9123  # rounded

    def test_stats_fill_ratio_empty(self):
        """Test fill ratio con buffer vuoto."""
        stats = BufferStats(size=0, capacity=0)
        data = stats.to_dict()

        assert data["fill_ratio"] == 0


# =============================================================================
# TEST EXPERIENCE REPLAY BUFFER
# =============================================================================

class TestExperienceReplayBuffer:
    """Test per ExperienceReplayBuffer standard."""

    def test_create_buffer(self):
        """Test creazione buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)

        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert not buffer.is_full()

    def test_add_experience(self):
        """Test aggiunta esperienza."""
        buffer = ExperienceReplayBuffer(capacity=100)

        exp_id = buffer.add(
            trace={"query": "test"},
            feedback={"score": 0.8},
            reward=0.75
        )

        assert exp_id.startswith("exp_")
        assert len(buffer) == 1

    def test_add_with_to_dict(self):
        """Test aggiunta oggetti con to_dict."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Mock oggetti con to_dict
        trace = MagicMock()
        trace.to_dict.return_value = {"query": "test"}

        feedback = MagicMock()
        feedback.to_dict.return_value = {"score": 0.9}

        exp_id = buffer.add(trace, feedback, reward=0.8)

        assert len(buffer) == 1
        trace.to_dict.assert_called_once()
        feedback.to_dict.assert_called_once()

    def test_add_with_metadata(self):
        """Test aggiunta con metadata."""
        buffer = ExperienceReplayBuffer(capacity=100)

        buffer.add(
            trace={"query": "test"},
            feedback={"score": 0.8},
            reward=0.75,
            metadata={"source": "unit_test"}
        )

        exp = buffer.get_all()[0]
        assert exp.metadata["source"] == "unit_test"

    def test_sample_empty_buffer(self):
        """Test sampling da buffer vuoto."""
        buffer = ExperienceReplayBuffer(capacity=100)

        sampled = buffer.sample(batch_size=10)

        assert sampled == []

    def test_sample_uniform(self):
        """Test sampling uniforme."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Aggiungi 50 esperienze
        for i in range(50):
            buffer.add(
                trace={"query": f"test_{i}"},
                feedback={"score": 0.5},
                reward=i / 50
            )

        sampled = buffer.sample(batch_size=10)

        assert len(sampled) == 10
        assert all(isinstance(s, Experience) for s in sampled)

    def test_sample_more_than_available(self):
        """Test sampling più elementi di quanti disponibili."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(5):
            buffer.add({"q": i}, {"s": 0.5}, reward=0.5)

        sampled = buffer.sample(batch_size=20)

        assert len(sampled) == 5  # max disponibili

    def test_sample_recent(self):
        """Test sampling elementi recenti."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(20):
            buffer.add(
                trace={"idx": i},
                feedback={},
                reward=0.5
            )

        recent = buffer.sample_recent(batch_size=5)

        assert len(recent) == 5
        # Gli ultimi devono avere idx più alti
        indices = [r.trace_data["idx"] for r in recent]
        assert indices == [15, 16, 17, 18, 19]

    def test_get_all(self):
        """Test recupero tutte le esperienze."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add({"q": i}, {}, 0.5)

        all_exp = buffer.get_all()

        assert len(all_exp) == 10

    def test_clear(self):
        """Test svuotamento buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add({}, {}, 0.5)

        assert len(buffer) == 10

        buffer.clear()

        assert len(buffer) == 0

    def test_capacity_limit(self):
        """Test limite capacità (FIFO)."""
        buffer = ExperienceReplayBuffer(capacity=10)

        for i in range(20):
            buffer.add(
                trace={"idx": i},
                feedback={},
                reward=0.5
            )

        assert len(buffer) == 10  # Max capacity

        # I primi 10 dovrebbero essere rimossi
        all_exp = buffer.get_all()
        indices = [e.trace_data["idx"] for e in all_exp]
        assert min(indices) == 10  # Solo ultimi 10

    def test_is_full(self):
        """Test controllo buffer pieno."""
        buffer = ExperienceReplayBuffer(capacity=5)

        assert not buffer.is_full()

        for i in range(5):
            buffer.add({}, {}, 0.5)

        assert buffer.is_full()

    def test_get_stats(self):
        """Test statistiche buffer."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add(
                trace={},
                feedback={},
                reward=i / 10,  # 0.0 to 0.9
                priority=1.0
            )

        buffer.sample(5)  # Sample some

        stats = buffer.get_stats()

        assert stats.size == 10
        assert stats.capacity == 100
        assert stats.total_added == 10
        assert stats.total_sampled == 5
        assert 0.4 <= stats.avg_reward <= 0.5  # (0+0.1+...+0.9)/10 = 0.45

    def test_get_stats_empty(self):
        """Test stats buffer vuoto."""
        buffer = ExperienceReplayBuffer(capacity=100)

        stats = buffer.get_stats()

        assert stats.size == 0
        assert stats.avg_reward == 0.0

    def test_save_and_load(self):
        """Test salvataggio e caricamento."""
        buffer = ExperienceReplayBuffer(capacity=100)

        for i in range(10):
            buffer.add(
                trace={"query": f"test_{i}"},
                feedback={"score": i / 10},
                reward=i / 10
            )

        # Salva
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            buffer.save(path)

            # Carica in nuovo buffer
            buffer2 = ExperienceReplayBuffer(capacity=50)
            buffer2.load(path)

            assert len(buffer2) == 10
            assert buffer2.capacity == 100  # Caricato da file

            all_exp = buffer2.get_all()
            assert all_exp[0].trace_data["query"] == "test_0"
        finally:
            os.unlink(path)

    def test_thread_safety(self):
        """Test thread safety."""
        buffer = ExperienceReplayBuffer(capacity=1000)
        errors = []

        def add_experiences():
            try:
                for i in range(100):
                    buffer.add({}, {}, 0.5)
            except Exception as e:
                errors.append(e)

        def sample_experiences():
            try:
                for i in range(50):
                    buffer.sample(10)
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=add_experiences))
            threads.append(threading.Thread(target=sample_experiences))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# TEST FACTORY FUNCTION
# =============================================================================

class TestCreateReplayBuffer:
    """Test per factory function."""

    def test_create_standard_buffer(self):
        """Test creazione buffer standard."""
        buffer = create_replay_buffer(capacity=100, prioritized=False)

        assert isinstance(buffer, ExperienceReplayBuffer)
        assert not isinstance(buffer, PrioritizedReplayBuffer)
        assert buffer.capacity == 100

    def test_create_prioritized_buffer(self):
        """Test creazione buffer prioritizzato."""
        buffer = create_replay_buffer(
            capacity=200,
            prioritized=True,
            alpha=0.7,
            epsilon=0.02
        )

        assert isinstance(buffer, PrioritizedReplayBuffer)
        assert buffer.capacity == 200
        assert buffer.alpha == 0.7
        assert buffer.epsilon == 0.02

    def test_default_values(self):
        """Test valori default."""
        buffer = create_replay_buffer()

        assert isinstance(buffer, ExperienceReplayBuffer)
        assert buffer.capacity == 10000


# =============================================================================
# TEST INTEGRAZIONE
# =============================================================================

class TestReplayBufferIntegration:
    """Test integrazione replay buffer."""

    def test_training_loop_simulation(self):
        """Simula loop di training con replay buffer."""
        buffer = ExperienceReplayBuffer(capacity=1000)

        # Fase 1: Popola buffer con esperienze iniziali
        for episode in range(100):
            trace = {
                "query": f"query_{episode}",
                "experts": ["literal", "systemic"],
                "response": f"response_{episode}"
            }
            feedback = {
                "retrieval": {"precision": 0.8},
                "reasoning": {"logical_coherence": 0.7},
                "synthesis": {"clarity": 0.9}
            }
            reward = 0.5 + (episode % 10) * 0.05

            buffer.add(trace, feedback, reward)

        assert len(buffer) == 100

        # Fase 2: Training con replay
        total_sampled = 0
        for training_step in range(10):
            batch = buffer.sample(batch_size=32)
            total_sampled += len(batch)

            # Simula update (normalmente calcolerebbe loss)
            for exp in batch:
                _ = exp.reward * 1.1  # Dummy operation

        stats = buffer.get_stats()
        assert stats.total_sampled == total_sampled

    def test_buffer_persistence_workflow(self):
        """Test workflow con persistenza."""
        # Crea e popola buffer
        buffer1 = ExperienceReplayBuffer(capacity=100)

        for i in range(50):
            buffer1.add(
                trace={"step": i},
                feedback={"score": i / 50},
                reward=i / 50
            )

        # Salva
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = f.name

        try:
            buffer1.save(path)

            # Simula restart: carica in nuovo buffer
            buffer2 = ExperienceReplayBuffer()
            buffer2.load(path)

            # Continua ad aggiungere
            for i in range(50, 70):
                buffer2.add(
                    trace={"step": i},
                    feedback={"score": i / 70},
                    reward=i / 70
                )

            assert len(buffer2) == 70

            # Sampling dovrebbe funzionare
            batch = buffer2.sample(20)
            assert len(batch) == 20
        finally:
            os.unlink(path)

    def test_mixed_sampling_strategies(self):
        """Test strategie di sampling miste."""
        buffer = ExperienceReplayBuffer(capacity=100)

        # Popola
        for i in range(50):
            buffer.add({"idx": i}, {}, reward=i / 50)

        # Sampling uniforme
        uniform_batch = buffer.sample(10)

        # Sampling recenti
        recent_batch = buffer.sample_recent(10)

        # Verifica che siano diversi
        uniform_indices = {e.trace_data["idx"] for e in uniform_batch}
        recent_indices = {e.trace_data["idx"] for e in recent_batch}

        # I recenti dovrebbero avere indici alti
        assert min(recent_indices) >= 40

        # Uniforme potrebbe avere mix
        assert len(uniform_indices) == 10


# =============================================================================
# TEST EDGE CASES (standard buffer)
# =============================================================================

class TestEdgeCases:
    """Test casi limite."""

    def test_single_element_buffer(self):
        """Test buffer con singolo elemento."""
        buffer = ExperienceReplayBuffer(capacity=1)

        buffer.add({}, {}, 0.5)

        assert len(buffer) == 1

        sampled = buffer.sample(10)
        assert len(sampled) == 1

    def test_zero_capacity(self):
        """Test buffer con capacità zero."""
        buffer = ExperienceReplayBuffer(capacity=0)

        # Non dovrebbe poter aggiungere
        buffer.add({}, {}, 0.5)
        assert len(buffer) == 0

    def test_negative_reward(self):
        """Test con reward negativo."""
        buffer = ExperienceReplayBuffer(capacity=100)

        buffer.add({}, {}, reward=-0.5)

        exp = buffer.get_all()[0]
        assert exp.reward == -0.5
