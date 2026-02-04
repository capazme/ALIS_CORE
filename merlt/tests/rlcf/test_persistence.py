"""
Test per RLCF Persistence Layer.

Verifica:
1. Salvataggio/recupero traces
2. Salvataggio/recupero feedback
3. Training data retrieval
4. Policy checkpointing
5. Training sessions tracking
"""

import pytest
import pytest_asyncio
import asyncio
import os
from datetime import datetime, timedelta
import numpy as np
import torch

# Schema uses PostgreSQL ARRAY type — requires real PostgreSQL
pytestmark = pytest.mark.integration

# Set SQLite for tests
os.environ["RLCF_ASYNC_DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"

from merlt.rlcf.persistence import (
    RLCFPersistence,
    RLCFTrace,
    RLCFFeedback,
    PolicyCheckpoint,
    TrainingSession,
    create_persistence,
)
from merlt.rlcf.execution_trace import ExecutionTrace, Action
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_trace() -> ExecutionTrace:
    """Crea trace di esempio."""
    trace = ExecutionTrace(query_id="test_query_001")

    # Add expert selection action
    trace.add_expert_selection(
        expert_type="literal",
        weight=0.7,
        log_prob=-0.357,
        metadata={
            "query_embedding": [0.1] * 64,  # Simplified for test
            "source": "gating_policy"
        }
    )

    # Add graph traversal
    trace.add_graph_traversal(
        relation_type="RIFERIMENTO",
        weight=0.8,
        log_prob=-0.223,
        source_node="urn:norma:cc:art1337"
    )

    return trace


@pytest.fixture
def sample_feedback() -> MultilevelFeedback:
    """Crea feedback di esempio."""
    return MultilevelFeedback(
        query_id="test_query_001",
        retrieval_feedback=RetrievalFeedback(
            precision=0.8,
            recall=0.7,
            ranking_quality=0.85
        ),
        reasoning_feedback=ReasoningFeedback(
            logical_coherence=0.9,
            legal_soundness=0.85,
            citation_quality=0.8
        ),
        synthesis_feedback=SynthesisFeedback(
            clarity=0.9,
            completeness=0.85,
            usefulness=0.9
        )
    )


@pytest_asyncio.fixture
async def persistence():
    """Crea persistence con SQLite in-memory."""
    p = await create_persistence("sqlite+aiosqlite:///:memory:")
    yield p


# =============================================================================
# TEST TRACE OPERATIONS
# =============================================================================

class TestTraceOperations:
    """Test operazioni su traces."""

    @pytest.mark.asyncio
    async def test_save_trace(self, persistence, sample_trace):
        """Test salvataggio trace."""
        trace_id = await persistence.save_trace(
            trace=sample_trace,
            policy_version="v1.0.0",
            query_text="Cos'è la legittima difesa?",
            expert_type="literal"
        )

        assert trace_id is not None
        assert len(trace_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_get_trace(self, persistence, sample_trace):
        """Test recupero trace."""
        trace_id = await persistence.save_trace(sample_trace)

        # Recupera
        retrieved = await persistence.get_trace(trace_id)

        assert retrieved is not None
        assert retrieved.query_id == sample_trace.query_id
        assert len(retrieved.actions) == len(sample_trace.actions)
        assert abs(retrieved.total_log_prob - sample_trace.total_log_prob) < 0.001

    @pytest.mark.asyncio
    async def test_get_nonexistent_trace(self, persistence):
        """Test recupero trace inesistente."""
        retrieved = await persistence.get_trace("nonexistent-id")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_traces_without_feedback(self, persistence, sample_trace):
        """Test recupero traces senza feedback."""
        # Salva alcuni traces
        await persistence.save_trace(sample_trace, policy_version="v1.0.0")

        trace2 = ExecutionTrace(query_id="test_query_002")
        trace2.add_expert_selection("systemic", 0.6, -0.5)
        await persistence.save_trace(trace2, policy_version="v1.0.0")

        # Recupera traces senza feedback
        traces = await persistence.get_traces_without_feedback(limit=10)

        assert len(traces) == 2
        assert all(isinstance(t[1], ExecutionTrace) for t in traces)


# =============================================================================
# TEST FEEDBACK OPERATIONS
# =============================================================================

class TestFeedbackOperations:
    """Test operazioni su feedback."""

    @pytest.mark.asyncio
    async def test_save_feedback(self, persistence, sample_trace, sample_feedback):
        """Test salvataggio feedback."""
        trace_id = await persistence.save_trace(sample_trace)

        feedback_id = await persistence.save_feedback(
            trace_id=trace_id,
            feedback=sample_feedback,
            user_id="user_001",
            user_authority=0.8,
            source="visualex"
        )

        assert feedback_id is not None
        assert len(feedback_id) == 36

    @pytest.mark.asyncio
    async def test_feedback_updates_trace_flag(self, persistence, sample_trace, sample_feedback):
        """Test che feedback aggiorna has_feedback su trace."""
        trace_id = await persistence.save_trace(sample_trace)

        # Prima del feedback
        traces_without = await persistence.get_traces_without_feedback()
        assert len(traces_without) == 1

        # Aggiungi feedback
        await persistence.save_feedback(trace_id, sample_feedback)

        # Dopo il feedback
        traces_without = await persistence.get_traces_without_feedback()
        assert len(traces_without) == 0

    @pytest.mark.asyncio
    async def test_get_feedback_for_trace(self, persistence, sample_trace, sample_feedback):
        """Test recupero feedback per trace."""
        trace_id = await persistence.save_trace(sample_trace)

        # Aggiungi multipli feedback
        await persistence.save_feedback(trace_id, sample_feedback, user_id="user1")
        await persistence.save_feedback(trace_id, sample_feedback, user_id="user2")

        # Recupera
        feedbacks = await persistence.get_feedback_for_trace(trace_id)

        assert len(feedbacks) == 2
        assert all(isinstance(f, MultilevelFeedback) for f in feedbacks)


# =============================================================================
# TEST TRAINING DATA
# =============================================================================

class TestTrainingData:
    """Test retrieval dati per training."""

    @pytest.mark.asyncio
    async def test_get_training_data(self, persistence, sample_trace, sample_feedback):
        """Test recupero dati per training."""
        # Salva trace + feedback
        trace_id = await persistence.save_trace(
            sample_trace,
            policy_version="v1.0.0"
        )
        await persistence.save_feedback(trace_id, sample_feedback)

        # Recupera training data
        training_data = await persistence.get_training_data(
            policy_version="v1.0.0",
            limit=100
        )

        assert len(training_data) == 1
        trace, feedback = training_data[0]
        assert isinstance(trace, ExecutionTrace)
        assert isinstance(feedback, MultilevelFeedback)

    @pytest.mark.asyncio
    async def test_get_training_data_with_date_filter(self, persistence, sample_trace, sample_feedback):
        """Test filtro per data."""
        trace_id = await persistence.save_trace(sample_trace)
        await persistence.save_feedback(trace_id, sample_feedback)

        # Futuro - non dovrebbe trovare nulla
        training_data = await persistence.get_training_data(
            min_date=datetime.utcnow() + timedelta(days=1)
        )

        assert len(training_data) == 0

    @pytest.mark.asyncio
    async def test_get_training_stats(self, persistence, sample_trace, sample_feedback):
        """Test statistiche training."""
        # Salva alcuni dati
        for i in range(3):
            trace = ExecutionTrace(query_id=f"query_{i}")
            trace.add_expert_selection("literal", 0.5, -0.5)
            trace_id = await persistence.save_trace(trace, policy_version="v1.0.0")

            if i < 2:  # Solo 2 hanno feedback
                await persistence.save_feedback(trace_id, sample_feedback)

        # Get stats
        stats = await persistence.get_training_stats(policy_version="v1.0.0")

        assert stats["total_traces"] == 3
        assert stats["traces_with_feedback"] == 2
        assert stats["total_feedback"] == 2


# =============================================================================
# TEST POLICY CHECKPOINTS
# =============================================================================

class TestPolicyCheckpoints:
    """Test gestione checkpoint policy."""

    @pytest.mark.asyncio
    async def test_save_policy_checkpoint(self, persistence):
        """Test salvataggio checkpoint."""
        checkpoint_id = await persistence.save_policy_checkpoint(
            version="v1.0.0",
            policy_type="gating",
            state_dict_path="/tmp/policy_v1.pt",
            config={"input_dim": 768, "hidden_dim": 256},
            training_metrics={"loss": 0.5, "accuracy": 0.8}
        )

        assert checkpoint_id is not None

    @pytest.mark.asyncio
    async def test_get_active_policy(self, persistence):
        """Test recupero policy attiva."""
        # Crea checkpoint
        await persistence.save_policy_checkpoint(
            version="v1.0.0",
            policy_type="gating"
        )

        # Non è ancora attiva
        active = await persistence.get_active_policy("gating")
        assert active is None

        # Attiva
        success = await persistence.activate_policy("v1.0.0", "gating")
        assert success

        # Ora è attiva
        active = await persistence.get_active_policy("gating")
        assert active is not None
        assert active.version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_activate_deactivates_previous(self, persistence):
        """Test che attivare una policy disattiva la precedente."""
        # Crea e attiva v1
        await persistence.save_policy_checkpoint(version="v1.0.0", policy_type="gating")
        await persistence.activate_policy("v1.0.0", "gating")

        # Crea e attiva v2
        await persistence.save_policy_checkpoint(version="v1.0.1", policy_type="gating")
        await persistence.activate_policy("v1.0.1", "gating")

        # Solo v1.0.1 dovrebbe essere attiva
        active = await persistence.get_active_policy("gating")
        assert active.version == "v1.0.1"


# =============================================================================
# TEST TRAINING SESSIONS
# =============================================================================

class TestTrainingSessions:
    """Test tracking sessioni di training."""

    @pytest.mark.asyncio
    async def test_start_training_session(self, persistence):
        """Test avvio sessione."""
        session_id = await persistence.start_training_session(
            policy_type="gating",
            policy_version_from="v1.0.0",
            config={"learning_rate": 0.001}
        )

        assert session_id is not None
        assert len(session_id) == 36

    @pytest.mark.asyncio
    async def test_complete_training_session(self, persistence):
        """Test completamento sessione."""
        session_id = await persistence.start_training_session(
            policy_type="gating"
        )

        await persistence.complete_training_session(
            session_id=session_id,
            policy_version_to="v1.0.1",
            num_traces=100,
            num_feedback=100,
            metrics={"avg_reward": 0.75, "loss": 0.3}
        )

        # Verifica via stats non disponibile direttamente,
        # ma il test passa se non ci sono errori

    @pytest.mark.asyncio
    async def test_fail_training_session(self, persistence):
        """Test sessione fallita."""
        session_id = await persistence.start_training_session(
            policy_type="gating"
        )

        await persistence.fail_training_session(
            session_id=session_id,
            error_message="Out of memory"
        )

        # Verifica via stats non disponibile direttamente,
        # ma il test passa se non ci sono errori


# =============================================================================
# TEST MODEL CONVERSIONS
# =============================================================================

class TestModelConversions:
    """Test conversioni tra dataclass e SQLAlchemy models."""

    def test_trace_to_execution_trace(self, sample_trace):
        """Test conversione RLCFTrace → ExecutionTrace."""
        db_trace = RLCFTrace.from_execution_trace(
            sample_trace,
            policy_version="v1.0.0",
            query_text="Test query",
            expert_type="literal"
        )

        # Converti indietro
        converted = db_trace.to_execution_trace()

        assert converted.query_id == sample_trace.query_id
        assert len(converted.actions) == len(sample_trace.actions)
        assert abs(converted.total_log_prob - sample_trace.total_log_prob) < 0.001

    def test_feedback_to_multilevel(self, sample_feedback):
        """Test conversione RLCFFeedback → MultilevelFeedback."""
        db_feedback = RLCFFeedback.from_multilevel_feedback(
            trace_id="test-trace-id",
            feedback=sample_feedback,
            user_id="user_001",
            user_authority=0.8
        )

        # Verifica denormalizzazione
        assert db_feedback.retrieval_precision == 0.8
        assert db_feedback.reasoning_coherence == 0.9
        assert db_feedback.synthesis_clarity == 0.9

        # Converti indietro
        converted = db_feedback.to_multilevel_feedback()

        assert converted.query_id == sample_feedback.query_id
        assert abs(converted.overall_score() - sample_feedback.overall_score()) < 0.01


# =============================================================================
# TEST CONTEXT MANAGER
# =============================================================================

class TestContextManager:
    """Test uso come context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with RLCFPersistence("sqlite+aiosqlite:///:memory:") as persistence:
            trace = ExecutionTrace(query_id="ctx_test")
            trace.add_expert_selection("literal", 0.5, -0.5)

            trace_id = await persistence.save_trace(trace)
            assert trace_id is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
