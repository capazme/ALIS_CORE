"""
Test per Policy Gradient Training Loop (RLCF v2).

Verifica:
1. ExecutionTrace: azioni, log-prob, serializzazione
2. GatingPolicy: forward pass, output shape, trainability
3. TraversalPolicy: forward pass, output range sigmoid
4. PolicyGradientTrainer: inizializzazione, compute_reward, update, checkpoint
5. MultilevelFeedback: creazione, serializzazione, partial feedback

Basato su docs/architecture/learning-layer.md §3 Policy Gradient.
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
def sample_execution_trace():
    """Trace di esecuzione di esempio."""
    from merlt.rlcf.execution_trace import ExecutionTrace, Action

    trace = ExecutionTrace(query_id="test_query_1")

    # Simula 3 azioni prese dal sistema
    trace.add_action(Action(
        action_type="expert_selection",
        parameters={"expert": "literal", "weight": 0.7},
        log_prob=-0.5
    ))
    trace.add_action(Action(
        action_type="graph_traversal",
        parameters={"relation": "RIFERIMENTO", "weight": 0.8},
        log_prob=-0.3
    ))
    trace.add_action(Action(
        action_type="tool_use",
        parameters={"tool_name": "search", "query": "legittima difesa"},
        log_prob=-0.2
    ))

    return trace


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
# TEST 1: ExecutionTrace
# ============================================================================

class TestExecutionTrace:
    """Test ExecutionTrace per tracking azioni durante esecuzione."""

    def test_action_creation(self):
        """Verifica creazione Action con log_prob."""
        from merlt.rlcf.execution_trace import Action

        action = Action(
            action_type="expert_selection",
            parameters={"expert": "literal", "weight": 0.7},
            log_prob=-0.5
        )

        assert action.action_type == "expert_selection"
        assert action.log_prob == pytest.approx(-0.5)
        assert action.parameters["expert"] == "literal"

    def test_trace_add_action(self, sample_execution_trace):
        """Verifica aggiunta azioni alla trace."""
        from merlt.rlcf.execution_trace import Action

        initial_count = len(sample_execution_trace.actions)

        sample_execution_trace.add_action(Action(
            action_type="tool_use",
            parameters={"tool_name": "verify"},
            log_prob=-0.1
        ))

        assert len(sample_execution_trace.actions) == initial_count + 1
        assert sample_execution_trace.actions[-1].action_type == "tool_use"

    def test_total_log_prob(self, sample_execution_trace):
        """Verifica calcolo log_prob totale."""
        # Trace ha 3 azioni: -0.5, -0.3, -0.2
        total = sample_execution_trace.total_log_prob
        expected = -0.5 + -0.3 + -0.2  # = -1.0

        assert total == pytest.approx(expected, abs=1e-6)

    def test_total_log_prob_empty_trace(self):
        """Verifica log_prob con trace vuota."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        trace = ExecutionTrace(query_id="empty")
        assert trace.total_log_prob == 0.0

    def test_to_dict(self, sample_execution_trace):
        """Verifica serializzazione trace."""
        d = sample_execution_trace.to_dict()

        assert d["query_id"] == "test_query_1"
        assert "actions" in d
        assert len(d["actions"]) == 3
        assert d["actions"][0]["action_type"] == "expert_selection"
        assert "log_prob" in d["actions"][0]

    def test_from_dict(self, sample_execution_trace):
        """Verifica deserializzazione trace."""
        from merlt.rlcf.execution_trace import ExecutionTrace

        d = sample_execution_trace.to_dict()
        restored = ExecutionTrace.from_dict(d)

        assert restored.query_id == sample_execution_trace.query_id
        assert len(restored.actions) == len(sample_execution_trace.actions)
        assert restored.total_log_prob == pytest.approx(sample_execution_trace.total_log_prob)


# ============================================================================
# TEST 2: GatingPolicy
# ============================================================================

class TestGatingPolicy:
    """Test GatingPolicy per selezione Expert."""

    def test_forward_pass(self, sample_query_embedding):
        """Verifica forward pass produce output corretto."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        policy = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        weights, log_probs = policy.forward(sample_query_embedding)

        # Output deve essere tensor
        assert isinstance(weights, torch.Tensor)
        # Note: weights doesn't require grad in forward, but network does

    def test_output_shape(self, sample_query_embedding):
        """Verifica shape dell'output."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        num_experts = 4
        policy = GatingPolicy(input_dim=768, num_experts=num_experts, device="cpu")
        weights, log_probs = policy.forward(sample_query_embedding)

        # Shape: (batch_size, num_experts)
        assert weights.shape == (1, num_experts)
        assert log_probs.shape == (1, num_experts)

    def test_softmax_sum_to_one(self, sample_query_embedding):
        """Verifica che i pesi sommino a 1 (softmax)."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        policy = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        weights, _ = policy.forward(sample_query_embedding)

        # Softmax → somma deve essere 1
        total = weights.sum(dim=-1).item()
        assert total == pytest.approx(1.0, abs=1e-5)

    def test_weights_positive(self, sample_query_embedding):
        """Verifica che tutti i pesi siano positivi."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        policy = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        weights, _ = policy.forward(sample_query_embedding)

        # Softmax garantisce positività
        assert (weights >= 0).all()

    def test_parameters_trainable(self):
        """Verifica che i parametri siano trainable."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        policy = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        params = list(policy.parameters())

        # Deve avere parametri
        assert len(params) > 0

        # Tutti devono richiedere gradient
        for p in params:
            assert p.requires_grad

    def test_different_inputs_different_outputs(self):
        """Verifica che input diversi producano output diversi."""
        from merlt.rlcf.policy_gradient import GatingPolicy

        policy = GatingPolicy(input_dim=768, num_experts=4, device="cpu")

        emb1 = torch.randn(1, 768)
        emb2 = torch.randn(1, 768)

        weights1, _ = policy.forward(emb1)
        weights2, _ = policy.forward(emb2)

        # Output devono essere diversi (con alta probabilità)
        assert not torch.allclose(weights1, weights2, atol=1e-6)


# ============================================================================
# TEST 3: TraversalPolicy
# ============================================================================

class TestTraversalPolicy:
    """Test TraversalPolicy per pesi relazioni grafo."""

    def test_forward_pass(self):
        """Verifica forward pass per singola relazione."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(input_dim=768, relation_dim=64, device="cpu")

        # Context embedding (es. query)
        context = torch.randn(1, 768)
        # Relation index
        relation_idx = torch.tensor([0])  # First relation

        weights, log_probs = policy.forward(context, relation_idx)

        assert isinstance(weights, torch.Tensor)

    def test_output_shape(self):
        """Verifica shape output."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(input_dim=768, relation_dim=64, device="cpu")

        context = torch.randn(1, 768)
        relation_idx = torch.tensor([0])

        weights, log_probs = policy.forward(context, relation_idx)

        # Un peso per ogni sample nel batch
        assert weights.shape == (1, 1)
        assert log_probs.shape == (1, 1)

    def test_output_range_sigmoid(self):
        """Verifica che output sia in [0, 1] tramite sigmoid."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(input_dim=768, device="cpu")

        context = torch.randn(1, 768)
        relation_idx = torch.tensor([0])

        weights, _ = policy.forward(context, relation_idx)

        # Sigmoid garantisce [0, 1]
        assert (weights >= 0).all()
        assert (weights <= 1).all()

    def test_relation_types_defined(self):
        """Verifica che i relation types siano definiti."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(input_dim=768, device="cpu")

        # Verifica che le relazioni siano registrate
        assert len(policy.relation_types) > 0
        assert "RIFERIMENTO" in policy.relation_types

    def test_get_relation_index(self):
        """Verifica recupero indice relazione."""
        from merlt.rlcf.policy_gradient import TraversalPolicy

        policy = TraversalPolicy(input_dim=768, device="cpu")

        idx = policy.get_relation_index("RIFERIMENTO")
        assert isinstance(idx, int)
        assert 0 <= idx < policy.num_relations


# ============================================================================
# TEST 4: PolicyGradientTrainer
# ============================================================================

class TestPolicyGradientTrainer:
    """Test PolicyGradientTrainer per RLCF update loop."""

    def test_initialization(self):
        """Verifica inizializzazione trainer."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy,
            TrainerConfig
        )

        gating = GatingPolicy(input_dim=768, num_experts=4, device="cpu")

        trainer = PolicyGradientTrainer(
            policy=gating,
            config=TrainerConfig(learning_rate=1e-4)
        )

        assert trainer.policy == gating
        assert trainer.optimizer is not None
        assert trainer.baseline == 0.0  # Iniziale

    def test_compute_reward(self, sample_feedback):
        """Verifica calcolo reward da feedback multilivello."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy
        )

        gating = GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        trainer = PolicyGradientTrainer(policy=gating)

        reward = trainer.compute_reward(sample_feedback)

        # Reward deve essere float in [0, 1]
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0

        # Con feedback ottimo, reward deve essere alto
        assert reward > 0.7

    def test_update_from_feedback(self, sample_execution_trace, sample_feedback):
        """Verifica update parametri da feedback."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy
        )

        gating = GatingPolicy(input_dim=768, num_experts=4, device="cpu")

        trainer = PolicyGradientTrainer(
            policy=gating,
        )

        # Salva parametri iniziali
        initial_params = [p.clone() for p in gating.parameters()]

        # Update
        metrics = trainer.update_from_feedback(sample_execution_trace, sample_feedback)

        # Metrics deve contenere loss
        assert "loss" in metrics
        assert "reward" in metrics
        assert "baseline" in metrics

    def test_baseline_update(self, sample_execution_trace, sample_feedback):
        """Verifica che baseline venga aggiornata."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy
        )

        trainer = PolicyGradientTrainer(
            policy=GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        )

        initial_baseline = trainer.baseline

        # Simula 5 update
        for i in range(5):
            trainer.update_from_feedback(sample_execution_trace, sample_feedback)

        # Baseline dovrebbe essere cambiata (moving average)
        assert trainer.baseline != initial_baseline

    def test_checkpoint_save_load(self, tmp_path):
        """Verifica salvataggio e caricamento checkpoint."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy
        )

        # Crea trainer
        gating = GatingPolicy(input_dim=768, num_experts=4, device="cpu")

        trainer = PolicyGradientTrainer(policy=gating)

        # Modifica baseline per verificare salvataggio
        trainer.baseline = 0.75

        # Salva checkpoint
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path))

        assert checkpoint_path.exists()

        # Carica in nuovo trainer
        new_trainer = PolicyGradientTrainer(
            policy=GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        )

        new_trainer.load_checkpoint(str(checkpoint_path))

        # Baseline deve essere uguale
        assert new_trainer.baseline == pytest.approx(0.75)

    def test_get_stats(self):
        """Verifica statistiche trainer."""
        from merlt.rlcf.policy_gradient import (
            PolicyGradientTrainer,
            GatingPolicy
        )

        trainer = PolicyGradientTrainer(
            policy=GatingPolicy(input_dim=768, num_experts=4, device="cpu")
        )

        stats = trainer.get_stats()

        assert "num_updates" in stats
        assert "baseline" in stats
        assert "learning_rate" in stats


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

        # F1 = 2 * (P * R) / (P + R) = 2 * 0.48 / 1.4 ≈ 0.686
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
        """Verifica step completo: forward → feedback → backward → update."""
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

        # Crea più traces
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
