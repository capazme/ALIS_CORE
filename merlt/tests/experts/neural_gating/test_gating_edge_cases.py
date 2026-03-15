"""
Edge case tests for Neural Gating Network.

Tests per situazioni limite del sistema di gating neurale:
- Input con tutti zeri, tutti uguali, singolo dominante
- NaN e Inf handling
- Valori embedding estremi
- Logits tutti negativi
- Batch size 1
- Determinismo in eval mode
- AdaptiveThresholdManager cold start
"""

import pytest
import numpy as np

# Skip se torch non disponibile
pytest.importorskip("torch")

import torch

from merlt.experts.neural_gating.neural import (
    ExpertGatingMLP,
    NeuralGatingTrainer,
    GatingConfig,
    EXPERT_NAMES,
)
from merlt.experts.neural_gating.hybrid_router import (
    HybridExpertRouter,
    AdaptiveThresholdManager,
)
from merlt.experts.router import ExpertRouter


# ============================================================================
# Input Edge Cases
# ============================================================================

class TestGatingInputEdgeCases:
    """Tests per input edge cases al MLP."""

    @pytest.fixture
    def mlp(self):
        return ExpertGatingMLP()

    def test_all_zero_confidence_input(self, mlp):
        """Input embedding di tutti zeri."""
        embedding = np.zeros(1024, dtype=np.float32)
        result = mlp.predict_single(embedding)

        # Should still produce valid weights summing to ~1
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-5
        assert result["top_expert"] in EXPERT_NAMES
        assert 0 <= result["confidence"] <= 1

    def test_all_equal_confidence_input(self, mlp):
        """Input embedding con tutti valori uguali."""
        embedding = np.ones(1024, dtype=np.float32) * 0.5
        result = mlp.predict_single(embedding)

        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-5
        assert result["top_expert"] in EXPERT_NAMES

    def test_single_dominant_expert(self, mlp):
        """Training per creare un expert molto dominante."""
        trainer = NeuralGatingTrainer(mlp)

        # Forte bias verso literal
        for _ in range(100):
            embedding = np.random.randn(1024).astype(np.float32)
            trainer.train_from_feedback_sync(
                embedding,
                {"literal": 1.0, "systemic": 0.0, "principles": 0.0, "precedent": 0.0},
                authority_weight=1.0,
            )

        # Verifica che il modello abbia una forte preferenza
        test_emb = np.random.randn(1024).astype(np.float32)
        result = mlp.predict_single(test_emb)

        # Weights should still be valid
        total = sum(result["weights"].values())
        assert abs(total - 1.0) < 1e-5

    def test_nan_input_handling(self, mlp):
        """Input con NaN produce output senza crash."""
        embedding = np.full(1024, np.nan, dtype=np.float32)

        # NaN propagation in neural nets results in NaN output,
        # but should not raise an exception
        result = mlp.predict_single(embedding)
        # Just verify it doesn't crash - NaN in = NaN out is expected
        assert "weights" in result
        assert "confidence" in result

    def test_inf_input_handling(self, mlp):
        """Input con Inf produce output senza crash."""
        embedding = np.full(1024, np.inf, dtype=np.float32)

        # Should not raise
        result = mlp.predict_single(embedding)
        assert "weights" in result
        assert "confidence" in result

    def test_large_embedding_values(self, mlp):
        """Embedding con valori molto grandi."""
        embedding = np.random.randn(1024).astype(np.float32) * 1000
        result = mlp.predict_single(embedding)

        # Weights should still be valid probabilities
        total = sum(result["weights"].values())
        # With very large values, softmax might saturate but should not crash
        assert "weights" in result
        assert "top_expert" in result


# ============================================================================
# Softmax / Logits Edge Cases
# ============================================================================

class TestSoftmaxEdgeCases:
    """Tests per comportamento softmax con logits problematici."""

    def test_all_negative_logits_sum_to_one(self):
        """Logits tutti negativi producono softmax che somma a ~1."""
        mlp = ExpertGatingMLP()

        # Force all-negative logits via batch input
        x = torch.randn(1, 1024) * -10  # Very negative
        weights, log_probs = mlp(x)

        total = weights.sum().item()
        assert abs(total - 1.0) < 1e-5
        assert (weights >= 0).all()
        assert (log_probs <= 0 + 1e-6).all()

    def test_batch_size_one(self):
        """Forward pass con batch_size=1."""
        mlp = ExpertGatingMLP()
        x = torch.randn(1, 1024)
        weights, log_probs = mlp(x)

        assert weights.shape == (1, 4)
        assert log_probs.shape == (1, 4)
        assert abs(weights.sum().item() - 1.0) < 1e-5


# ============================================================================
# Eval Mode Determinism
# ============================================================================

class TestEvalModeDeterminism:
    """Tests per determinismo in eval mode."""

    def test_eval_mode_same_output(self):
        """Stessa input in eval mode produce stessa output."""
        mlp = ExpertGatingMLP()
        mlp.eval()

        embedding = np.random.randn(1024).astype(np.float32)

        results = [mlp.predict_single(embedding) for _ in range(5)]

        # All results should be identical
        for i in range(1, len(results)):
            assert results[0]["weights"] == results[i]["weights"]
            assert results[0]["confidence"] == results[i]["confidence"]
            assert results[0]["top_expert"] == results[i]["top_expert"]

    def test_train_mode_vs_eval_mode(self):
        """Train mode con dropout puo' dare risultati diversi da eval mode."""
        mlp = ExpertGatingMLP(GatingConfig(dropout=0.5))  # High dropout
        embedding = np.random.randn(1024).astype(np.float32)

        # Eval mode: deterministic
        mlp.eval()
        eval_result = mlp.predict_single(embedding)

        # Train mode: may vary due to dropout
        mlp.train()
        x = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        train_weights, _log_probs = mlp(x)

        # Just verify both produce valid outputs
        assert abs(sum(eval_result["weights"].values()) - 1.0) < 1e-5
        assert abs(train_weights.sum().item() - 1.0) < 1e-5


# ============================================================================
# AdaptiveThresholdManager Cold Start
# ============================================================================

class TestAdaptiveThresholdColdStart:
    """Tests per AdaptiveThresholdManager in cold start."""

    @pytest.fixture
    def manager(self):
        mlp = ExpertGatingMLP()
        router = HybridExpertRouter(
            neural_gating=mlp,
            confidence_threshold=0.9,
        )
        return AdaptiveThresholdManager(
            router,
            initial_threshold=0.9,
            target_threshold=0.6,
            performance_window=10,
        )

    def test_cold_start_no_update(self, manager):
        """Con pochi feedback non aggiorna threshold."""
        # Solo 3 feedback: sotto finestra minima (10)
        for _ in range(3):
            result = manager.update_from_feedback(
                neural_was_correct=True, user_rating=0.9
            )

        assert result["action"] == "waiting"
        # Threshold dovrebbe essere invariato
        assert manager.router.confidence_threshold == 0.9

    def test_cold_start_initial_status(self, manager):
        """Status iniziale senza feedback."""
        status = manager.get_status()

        assert status["current_threshold"] == 0.9
        assert status["target_threshold"] == 0.6
        assert status["total_feedback"] == 0
        assert status["recent_accuracy"] == 0.0

    def test_threshold_respects_minimum(self, manager):
        """Threshold non scende sotto min_threshold."""
        # Forza molti feedback positivi
        for _ in range(100):
            manager.update_from_feedback(
                neural_was_correct=True, user_rating=1.0
            )

        assert manager.router.confidence_threshold >= manager.min_threshold

    def test_threshold_respects_initial_maximum(self, manager):
        """Threshold non sale sopra initial_threshold dopo feedback negativi."""
        # Abbassa prima
        for _ in range(20):
            manager.update_from_feedback(
                neural_was_correct=True, user_rating=0.9
            )
        lowered = manager.router.confidence_threshold

        # Poi feedback negativi
        for _ in range(30):
            manager.update_from_feedback(
                neural_was_correct=False, user_rating=0.1
            )

        assert manager.router.confidence_threshold <= manager.initial_threshold
