"""
Test per SingleStepTrainer.

Valida che il trainer REINFORCE ottimizzato per single-step:
1. Calcola correttamente advantage e baseline
2. Esegue backpropagation reale
3. Aggiorna correttamente i parametri della policy
4. Converge su task sintetici
"""

import pytest
import numpy as np
import torch

from merlt.rlcf.policy_gradient import GatingPolicy
from merlt.rlcf.single_step_trainer import (
    SingleStepTrainer,
    SingleStepConfig,
    create_single_step_trainer
)
from merlt.rlcf.execution_trace import ExecutionTrace, Action
from merlt.rlcf.multilevel_feedback import (
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def policy():
    """Crea GatingPolicy per test."""
    return GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4)


@pytest.fixture
def config():
    """Crea config per test."""
    return SingleStepConfig(
        learning_rate=0.01,
        baseline_decay=0.9,
        clip_grad_norm=1.0,
        entropy_coef=0.01
    )


@pytest.fixture
def trainer(policy, config):
    """Crea trainer per test."""
    return SingleStepTrainer(policy, config)


def create_trace_with_embedding(query_embedding: np.ndarray, weights: np.ndarray) -> ExecutionTrace:
    """Crea un trace con query_embedding nei metadata."""
    trace = ExecutionTrace(query_id="test_001")

    expert_types = ["literal", "systemic", "principles", "precedent"]
    log_probs = np.log(weights + 1e-8)

    for i, expert_type in enumerate(expert_types):
        trace.add_expert_selection(
            expert_type=expert_type,
            weight=float(weights[i]),
            log_prob=float(log_probs[i]),
            metadata={
                "source": "gating_policy",
                "query_embedding_dim": len(query_embedding),
                "query_embedding": query_embedding.tolist(),
                "action_index": i
            }
        )

    return trace


def create_feedback(score: float) -> MultilevelFeedback:
    """Crea feedback con punteggio specifico."""
    return MultilevelFeedback(
        query_id="test_query",
        retrieval_feedback=RetrievalFeedback(precision=score, recall=score),
        reasoning_feedback=ReasoningFeedback(logical_coherence=score, legal_soundness=score),
        synthesis_feedback=SynthesisFeedback(clarity=score, usefulness=score)
    )


# =============================================================================
# TEST INITIALIZATION
# =============================================================================

class TestSingleStepTrainerInit:
    """Test inizializzazione trainer."""

    def test_init_default_config(self, policy):
        """Test inizializzazione con config default."""
        trainer = SingleStepTrainer(policy)

        assert trainer.policy is policy
        assert trainer.config is not None
        assert trainer.baseline == 0.5
        assert trainer.num_updates == 0

    def test_init_custom_config(self, policy, config):
        """Test inizializzazione con config custom."""
        trainer = SingleStepTrainer(policy, config)

        assert trainer.config.learning_rate == 0.01
        assert trainer.config.baseline_decay == 0.9

    def test_factory_function(self, policy):
        """Test factory function."""
        trainer = create_single_step_trainer(
            policy,
            learning_rate=0.001,
            entropy_coef=0.02
        )

        assert trainer.config.learning_rate == 0.001
        assert trainer.config.entropy_coef == 0.02


# =============================================================================
# TEST GRADIENT FLOW
# =============================================================================

class TestGradientFlow:
    """Test che i gradienti fluiscono correttamente."""

    def test_gradient_exists_after_update(self, trainer, policy):
        """Verifica che i gradienti esistono dopo update."""
        query_embedding = np.random.randn(64).astype(np.float32)

        # Forward per ottenere weights
        with torch.no_grad():
            input_tensor = torch.tensor(query_embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        trace = create_trace_with_embedding(query_embedding, weights)
        feedback = create_feedback(0.9)

        # Update
        metrics = trainer.update(trace, feedback)

        # Verifica metriche
        assert metrics["grad_norm"] > 0, "Gradient norm should be > 0"
        assert metrics["loss"] != 0, "Loss should be computed"
        assert metrics["num_updates"] == 1

    def test_parameters_change_after_update(self, trainer, policy):
        """Verifica che i parametri cambiano dopo update."""
        # Salva parametri prima
        params_before = {
            name: param.clone().detach()
            for name, param in policy.mlp.named_parameters()
        }

        query_embedding = np.random.randn(64).astype(np.float32)
        with torch.no_grad():
            input_tensor = torch.tensor(query_embedding, device=policy.device).unsqueeze(0)
            weights, _ = policy.forward(input_tensor)
            weights = weights.cpu().numpy().flatten()

        trace = create_trace_with_embedding(query_embedding, weights)
        feedback = create_feedback(0.9)

        trainer.update(trace, feedback)

        # Verifica che almeno un parametro è cambiato
        params_changed = False
        for name, param in policy.mlp.named_parameters():
            delta = (param - params_before[name]).abs().sum().item()
            if delta > 1e-8:
                params_changed = True
                break

        assert params_changed, "At least one parameter should change"


# =============================================================================
# TEST BASELINE UPDATE
# =============================================================================

class TestBaselineUpdate:
    """Test aggiornamento baseline EMA."""

    def test_baseline_increases_with_high_reward(self, trainer):
        """Baseline aumenta con reward alti."""
        initial_baseline = trainer.baseline

        query_embedding = np.random.randn(64).astype(np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Update con high reward
        for _ in range(10):
            trace = create_trace_with_embedding(query_embedding, weights)
            feedback = create_feedback(0.95)
            trainer.update(trace, feedback)

        assert trainer.baseline > initial_baseline

    def test_baseline_decreases_with_low_reward(self, trainer):
        """Baseline diminuisce con reward bassi."""
        # Prima aumenta baseline
        trainer.baseline = 0.8

        query_embedding = np.random.randn(64).astype(np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        # Update con low reward
        for _ in range(10):
            trace = create_trace_with_embedding(query_embedding, weights)
            feedback = create_feedback(0.2)
            trainer.update(trace, feedback)

        assert trainer.baseline < 0.8


# =============================================================================
# TEST ADVANTAGE COMPUTATION
# =============================================================================

class TestAdvantageComputation:
    """Test calcolo advantage."""

    def test_positive_advantage_with_high_reward(self, trainer):
        """Advantage positivo con reward > baseline."""
        trainer.baseline = 0.5

        query_embedding = np.random.randn(64).astype(np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        trace = create_trace_with_embedding(query_embedding, weights)
        feedback = create_feedback(0.9)  # > baseline

        metrics = trainer.update(trace, feedback)

        assert metrics["raw_advantage"] > 0

    def test_negative_advantage_with_low_reward(self, trainer):
        """Advantage negativo con reward < baseline."""
        trainer.baseline = 0.8

        query_embedding = np.random.randn(64).astype(np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        trace = create_trace_with_embedding(query_embedding, weights)
        feedback = create_feedback(0.3)  # < baseline

        metrics = trainer.update(trace, feedback)

        assert metrics["raw_advantage"] < 0


# =============================================================================
# TEST BATCH UPDATE
# =============================================================================

class TestBatchUpdate:
    """Test update da batch di traces."""

    def test_batch_update_multiple_traces(self, trainer):
        """Update con batch di traces."""
        traces = []
        feedbacks = []

        for i in range(5):
            query_embedding = np.random.randn(64).astype(np.float32)
            weights = np.array([0.25, 0.25, 0.25, 0.25])
            traces.append(create_trace_with_embedding(query_embedding, weights))
            feedbacks.append(create_feedback(0.5 + i * 0.1))

        metrics = trainer.update_batch(traces, feedbacks)

        assert metrics["batch_size"] == 5
        assert metrics["valid_traces"] == 5
        assert "avg_reward" in metrics
        assert "grad_norm" in metrics

    def test_batch_update_mismatched_length_raises(self, trainer):
        """Errore se traces e feedbacks hanno lunghezze diverse."""
        traces = [create_trace_with_embedding(np.random.randn(64).astype(np.float32), np.array([0.25]*4))]
        feedbacks = [create_feedback(0.5), create_feedback(0.6)]

        with pytest.raises(ValueError):
            trainer.update_batch(traces, feedbacks)


# =============================================================================
# TEST CONVERGENCE
# =============================================================================

class TestConvergence:
    """Test convergenza su task sintetici."""

    def test_convergence_simple_task(self, policy):
        """Test convergenza su task semplice.

        Verifica che dopo training su un task con pattern chiaro,
        la policy mostri miglioramento rispetto a selezione random.
        """
        config = SingleStepConfig(learning_rate=0.05, baseline_decay=0.95)
        trainer = SingleStepTrainer(policy, config)

        np.random.seed(42)
        torch.manual_seed(42)

        # FASE 1: Training
        n_train = 100
        for _ in range(n_train):
            query_embedding = np.random.randn(64).astype(np.float32)
            target_expert = 0 if query_embedding[0] > 0 else 2

            with torch.no_grad():
                input_tensor = torch.tensor(query_embedding, device=policy.device).unsqueeze(0)
                weights, _ = policy.forward(input_tensor)
                weights = weights.cpu().numpy().flatten()

            chosen_expert = int(np.argmax(weights))
            reward = 1.0 if chosen_expert == target_expert else 0.0

            trace = create_trace_with_embedding(query_embedding, weights)
            feedback = create_feedback(reward)
            trainer.update(trace, feedback)

        # FASE 2: Evaluation (dopo training)
        n_eval = 50
        correct_count = 0
        for _ in range(n_eval):
            query_embedding = np.random.randn(64).astype(np.float32)
            target_expert = 0 if query_embedding[0] > 0 else 2

            with torch.no_grad():
                input_tensor = torch.tensor(query_embedding, device=policy.device).unsqueeze(0)
                weights, _ = policy.forward(input_tensor)
                weights = weights.cpu().numpy().flatten()

            chosen_expert = int(np.argmax(weights))
            if chosen_expert == target_expert:
                correct_count += 1

        # Dopo training, accuracy dovrebbe essere > random (25%)
        # Con task binario, random sarebbe ~50% ma abbiamo 4 esperti
        final_accuracy = correct_count / n_eval
        assert final_accuracy > 0.25, f"Accuracy {final_accuracy} should be > random (0.25)"


# =============================================================================
# TEST CHECKPOINT
# =============================================================================

class TestCheckpoint:
    """Test save/load checkpoint."""

    def test_save_and_load_checkpoint(self, trainer, tmp_path):
        """Test salvataggio e caricamento checkpoint."""
        # Train un po'
        query_embedding = np.random.randn(64).astype(np.float32)
        weights = np.array([0.25, 0.25, 0.25, 0.25])

        for _ in range(5):
            trace = create_trace_with_embedding(query_embedding, weights)
            feedback = create_feedback(0.8)
            trainer.update(trace, feedback)

        # Salva
        checkpoint_path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(checkpoint_path), metadata={"test": True})

        assert checkpoint_path.exists()

        # Valori prima del load
        baseline_before = trainer.baseline
        num_updates_before = trainer.num_updates

        # Nuovo trainer
        new_policy = GatingPolicy(input_dim=64, hidden_dim=32, num_experts=4)
        new_trainer = SingleStepTrainer(new_policy)

        # Load
        metadata = new_trainer.load_checkpoint(str(checkpoint_path))

        assert metadata.get("test") == True
        assert new_trainer.baseline == baseline_before
        assert new_trainer.num_updates == num_updates_before


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_trace_no_crash(self, trainer):
        """Trace senza azioni non causa crash."""
        trace = ExecutionTrace(query_id="empty")
        feedback = create_feedback(0.5)

        metrics = trainer.update(trace, feedback)

        assert metrics["loss"] == 0.0
        assert metrics["num_updates"] == 0

    def test_missing_embedding_no_crash(self, trainer):
        """Trace senza embedding non causa crash."""
        trace = ExecutionTrace(query_id="no_emb")
        trace.add_expert_selection(
            expert_type="literal",
            weight=0.5,
            log_prob=-0.69,
            metadata={"source": "gating_policy"}  # No query_embedding
        )

        feedback = create_feedback(0.5)
        metrics = trainer.update(trace, feedback)

        assert metrics["loss"] == 0.0

    def test_get_stats(self, trainer):
        """Test get_stats."""
        stats = trainer.get_stats()

        assert "num_updates" in stats
        assert "baseline" in stats
        assert "reward_variance" in stats
        assert "config" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
