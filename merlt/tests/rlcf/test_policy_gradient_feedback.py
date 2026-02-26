"""
Test per MultilevelFeedback e Policy Gradient Integration.

Verifica:
5. MultilevelFeedback: creazione, serializzazione, partial feedback
6. RetrievalFeedback: F1 score
7. ReasoningFeedback: average score
8. Factory Functions
9. Integration: full training step, batch training, execution trace utilities

Basato su docs/architecture/learning-layer.md S3 Policy Gradient.
"""

import pytest

# Conditional imports per torch
try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE,
    reason="PyTorch non disponibile - test policy gradient richiede torch"
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_query_embedding():
    """Embedding di esempio per test."""
    return torch.randn(1, 768)  # batch_size=1, embedding_dim=768


@pytest.fixture
def sample_feedback():
    """Feedback multilivello di esempio."""
    from merlt.rlcf.multilevel_feedback import (
        MultilevelFeedback,
        RetrievalFeedback,
        ReasoningFeedback,
        SynthesisFeedback
    )

    retrieval = RetrievalFeedback(
        precision=0.85,
        recall=0.75,
        sources_relevant=4,
        sources_total=5,
        ranking_quality=0.8
    )

    reasoning = ReasoningFeedback(
        logical_coherence=0.9,
        legal_soundness=0.85,
        citation_quality=0.8,
        interpretation_accuracy=0.85,
        expert_agreement=0.75,
        reasoning_steps_clear=0.9
    )

    synthesis = SynthesisFeedback(
        clarity=0.9,
        completeness=0.85,
        usefulness=0.9,
        conciseness=0.8,
        language_quality=0.85,
        structure_quality=0.85,
        user_satisfaction=0.9
    )

    return MultilevelFeedback(
        query_id="test_query_1",
        retrieval_feedback=retrieval,
        reasoning_feedback=reasoning,
        synthesis_feedback=synthesis,
        user_id="user_123"
    )


# ============================================================================
# TEST 5: MultilevelFeedback
# ============================================================================

class TestMultilevelFeedback:
    """Test MultilevelFeedback schema."""

    def test_creation(self, sample_feedback):
        """Verifica creazione feedback multilivello."""
        assert sample_feedback.query_id == "test_query_1"
        assert sample_feedback.user_id == "user_123"

        # Tutti i livelli presenti
        assert sample_feedback.retrieval_feedback is not None
        assert sample_feedback.reasoning_feedback is not None
        assert sample_feedback.synthesis_feedback is not None

    def test_overall_score(self, sample_feedback):
        """Verifica calcolo overall score."""
        score = sample_feedback.overall_score()

        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_to_dict(self, sample_feedback):
        """Verifica serializzazione."""
        d = sample_feedback.to_dict()

        assert d["query_id"] == "test_query_1"
        assert "retrieval_feedback" in d
        assert "reasoning_feedback" in d
        assert "synthesis_feedback" in d

    def test_from_dict(self, sample_feedback):
        """Verifica deserializzazione."""
        from merlt.rlcf.multilevel_feedback import MultilevelFeedback

        d = sample_feedback.to_dict()
        restored = MultilevelFeedback.from_dict(d)

        assert restored.query_id == sample_feedback.query_id
        assert restored.overall_score() == pytest.approx(sample_feedback.overall_score(), abs=0.01)

    def test_partial_feedback_retrieval_only(self):
        """Verifica feedback parziale (solo retrieval)."""
        from merlt.rlcf.multilevel_feedback import (
            MultilevelFeedback,
            RetrievalFeedback
        )

        feedback = MultilevelFeedback(
            query_id="partial_1",
            retrieval_feedback=RetrievalFeedback(
                precision=0.8,
                recall=0.7
            )
            # reasoning_feedback e synthesis_feedback = None
        )

        assert feedback.retrieval_feedback is not None
        assert feedback.reasoning_feedback is None
        assert feedback.synthesis_feedback is None

    def test_is_complete(self, sample_feedback):
        """Verifica check completezza."""
        assert sample_feedback.is_complete() is True

    def test_is_complete_partial(self):
        """Verifica check completezza con feedback parziale."""
        from merlt.rlcf.multilevel_feedback import (
            MultilevelFeedback,
            RetrievalFeedback
        )

        feedback = MultilevelFeedback(
            query_id="partial",
            retrieval_feedback=RetrievalFeedback(precision=0.8, recall=0.7)
        )

        assert feedback.is_complete() is False

    def test_summary(self, sample_feedback):
        """Verifica summary del feedback."""
        summary = sample_feedback.summary()

        assert "query_id" in summary
        assert "overall_score" in summary
        assert "is_complete" in summary


# ============================================================================
# TEST 6: RetrievalFeedback
# ============================================================================

class TestRetrievalFeedback:
    """Test RetrievalFeedback dataclass."""

    def test_f1_score(self):
        """Verifica calcolo F1 score."""
        from merlt.rlcf.multilevel_feedback import RetrievalFeedback

        feedback = RetrievalFeedback(
            precision=0.8,
            recall=0.6
        )

        f1 = feedback.f1_score()

        # F1 = 2 * (P * R) / (P + R) = 2 * 0.48 / 1.4 ~ 0.686
        expected = 2 * (0.8 * 0.6) / (0.8 + 0.6)
        assert f1 == pytest.approx(expected, abs=1e-6)

    def test_f1_score_zero(self):
        """Verifica F1 con precision e recall zero."""
        from merlt.rlcf.multilevel_feedback import RetrievalFeedback

        feedback = RetrievalFeedback(precision=0.0, recall=0.0)

        f1 = feedback.f1_score()
        assert f1 == 0.0


# ============================================================================
# TEST 7: ReasoningFeedback
# ============================================================================

class TestReasoningFeedback:
    """Test ReasoningFeedback dataclass."""

    def test_average_score(self):
        """Verifica calcolo average score."""
        from merlt.rlcf.multilevel_feedback import ReasoningFeedback

        feedback = ReasoningFeedback(
            logical_coherence=0.9,
            legal_soundness=0.8,
            citation_quality=0.7,
            interpretation_accuracy=0.85,
            expert_agreement=0.75,
            reasoning_steps_clear=0.8
        )

        avg = feedback.average_score()

        expected = (0.9 + 0.8 + 0.7 + 0.85 + 0.75 + 0.8) / 6
        assert avg == pytest.approx(expected, abs=1e-6)


# ============================================================================
# TEST 8: Factory Functions
# ============================================================================

class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_feedback_from_user_rating(self):
        """Verifica creazione feedback da rating singolo."""
        from merlt.rlcf.multilevel_feedback import create_feedback_from_user_rating

        feedback = create_feedback_from_user_rating(
            query_id="test_query",
            user_rating=0.8,
            user_id="user_123"
        )

        assert feedback.query_id == "test_query"
        assert feedback.overall_rating == 0.8
        assert feedback.user_id == "user_123"
        assert feedback.is_complete() is True

    def test_create_gating_policy(self):
        """Verifica factory per gating policy."""
        from merlt.rlcf.policy_gradient import create_gating_policy

        policy, trainer = create_gating_policy(input_dim=768, hidden_dim=256)

        assert policy.input_dim == 768
        assert policy.hidden_dim == 256
        assert trainer.policy == policy


# ============================================================================
# TEST INTEGRAZIONE
# ============================================================================

class TestPolicyGradientIntegration:
    """Test integrazione completa training loop."""

    def test_full_training_step(self, sample_query_embedding, sample_feedback):
        """Verifica step completo: forward -> feedback -> backward -> update."""
        from merlt.rlcf.policy_gradient import (
            GatingPolicy,
            PolicyGradientTrainer
        )
        from merlt.rlcf.execution_trace import ExecutionTrace, Action

        # 1. Setup
        gating = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        trainer = PolicyGradientTrainer(policy=gating)

        # 2. Forward pass (simula query processing)
        expert_weights, log_probs = gating.forward(sample_query_embedding)

        # 3. Costruisci trace con log_prob dalle policies
        trace = ExecutionTrace(query_id=sample_feedback.query_id)

        # Simula azione gating
        log_prob_gating = log_probs[0, 1].item()  # Scelta expert 1
        trace.add_action(Action(
            action_type="expert_selection",
            parameters={"expert_idx": 1},
            log_prob=log_prob_gating
        ))

        # 4. Update
        metrics = trainer.update_from_feedback(trace, sample_feedback)

        # Verifica che training sia avvenuto
        assert "loss" in metrics
        assert "reward" in metrics

    def test_batch_training(self):
        """Verifica training con batch di feedback."""
        from merlt.rlcf.policy_gradient import (
            GatingPolicy,
            PolicyGradientTrainer
        )
        from merlt.rlcf.multilevel_feedback import create_feedback_from_user_rating
        from merlt.rlcf.execution_trace import ExecutionTrace, Action

        trainer = PolicyGradientTrainer(
            policy=GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        )

        # Crea batch di 5 trace/feedback
        traces = []
        feedbacks = []

        for i in range(5):
            feedback = create_feedback_from_user_rating(
                query_id=f"query_{i}",
                user_rating=0.5 + (i * 0.1)  # Rating da 0.5 a 0.9
            )

            trace = ExecutionTrace(query_id=f"query_{i}")
            trace.add_action(Action(
                action_type="expert_selection",
                parameters={"expert_idx": i % 4},
                log_prob=-0.5
            ))

            traces.append(trace)
            feedbacks.append(feedback)

        # Training su batch
        metrics = trainer.update_from_batch(traces, feedbacks)

        assert "loss" in metrics
        assert "avg_reward" in metrics
        assert metrics["batch_size"] == 5

    def test_execution_trace_utilities(self):
        """Verifica utility functions per execution trace."""
        from merlt.rlcf.execution_trace import (
            ExecutionTrace,
            Action,
            merge_traces,
            compute_returns,
            compute_baseline
        )

        # Crea piu traces
        traces = []
        for i in range(3):
            trace = ExecutionTrace(query_id=f"q{i}")
            trace.add_action(Action(
                action_type="test",
                parameters={},
                log_prob=-0.3 * (i + 1)
            ))
            trace.set_reward(0.7 + (i * 0.1))
            traces.append(trace)

        # Test merge
        merged = merge_traces(traces)
        assert merged.num_actions == 3
        assert merged.reward == pytest.approx(sum(t.reward for t in traces) / 3)

        # Test compute_returns
        returns = compute_returns(traces)
        assert len(returns) == 3

        # Test compute_baseline
        baseline = compute_baseline(traces, method="mean")
        expected = sum(t.reward for t in traces) / 3
        assert baseline == pytest.approx(expected)
