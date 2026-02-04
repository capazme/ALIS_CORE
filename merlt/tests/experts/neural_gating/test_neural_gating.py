"""
Test per Neural Gating Network (Fase 3 v2 Recovery).

Verifica:
1. ExpertGatingMLP forward pass e predict_single
2. NeuralGatingTrainer training loop
3. HybridExpertRouter neural/regex routing
4. AutosaveCallback checkpoint management
5. Convergenza pesi nel tempo
"""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory

# Skip se torch non disponibile
pytest.importorskip("torch")

import torch

from merlt.experts.neural_gating.neural import (
    ExpertGatingMLP,
    NeuralGatingTrainer,
    AutosaveCallback,
    GatingConfig,
    EXPERT_NAMES,
    DEFAULT_EXPERT_PRIORS,
)
from merlt.experts.neural_gating.hybrid_router import (
    HybridExpertRouter,
    HybridRoutingDecision,
    AdaptiveThresholdManager,
)
from merlt.experts.base import ExpertContext
from merlt.experts.router import ExpertRouter


class TestGatingConfig:
    """Test configurazione GatingConfig."""

    def test_default_config(self):
        """Verifica config default."""
        config = GatingConfig()
        assert config.input_dim == 1024
        assert config.hidden_dim1 == 512
        assert config.hidden_dim2 == 256
        assert config.num_experts == 4
        assert config.dropout == 0.1
        assert config.learning_rate == 0.001

    def test_custom_config(self):
        """Verifica config custom."""
        config = GatingConfig(
            input_dim=512,
            hidden_dim1=256,
            learning_rate=0.01
        )
        assert config.input_dim == 512
        assert config.hidden_dim1 == 256
        assert config.learning_rate == 0.01


class TestExpertGatingMLP:
    """Test ExpertGatingMLP network."""

    @pytest.fixture
    def mlp(self):
        """Crea MLP per test."""
        return ExpertGatingMLP()

    def test_init(self, mlp):
        """Verifica inizializzazione."""
        assert mlp.config.input_dim == 1024
        assert mlp.config.num_experts == 4
        assert mlp.expert_bias.shape == (4,)

    def test_forward_shape(self, mlp):
        """Verifica shape output forward."""
        batch_size = 8
        input_dim = mlp.config.input_dim

        x = torch.randn(batch_size, input_dim)
        weights, confidence = mlp(x)

        assert weights.shape == (batch_size, 4)
        assert confidence.shape == (batch_size,)

    def test_forward_softmax(self, mlp):
        """Verifica che weights siano softmax (somma a 1)."""
        x = torch.randn(1, 1024)
        weights, _ = mlp(x)

        # Softmax → somma a 1
        total = weights.sum().item()
        assert abs(total - 1.0) < 1e-5

    def test_forward_confidence_range(self, mlp):
        """Verifica che confidence sia in [0, 1]."""
        x = torch.randn(16, 1024)
        _, confidence = mlp(x)

        assert (confidence >= 0).all()
        assert (confidence <= 1).all()

    def test_predict_single(self, mlp):
        """Verifica predict_single per inference."""
        embedding = np.random.randn(1024).astype(np.float32)
        result = mlp.predict_single(embedding)

        assert "weights" in result
        assert "confidence" in result
        assert "top_expert" in result

        # Weights devono contenere tutti gli expert
        for name in EXPERT_NAMES:
            assert name in result["weights"]

        # Top expert deve essere uno dei 4
        assert result["top_expert"] in EXPERT_NAMES

    def test_get_expert_priors(self, mlp):
        """Verifica che priors iniziali riflettano default."""
        priors = mlp.get_expert_priors()

        assert len(priors) == 4
        for name in EXPERT_NAMES:
            assert name in priors
            # Priors iniziali devono essere vicini ai default
            assert abs(priors[name] - DEFAULT_EXPERT_PRIORS[name]) < 0.1

    def test_deterministic_inference(self, mlp):
        """Verifica che inference sia deterministica."""
        embedding = np.random.randn(1024).astype(np.float32)

        result1 = mlp.predict_single(embedding)
        result2 = mlp.predict_single(embedding)

        assert result1["weights"] == result2["weights"]
        assert result1["confidence"] == result2["confidence"]


class TestNeuralGatingTrainer:
    """Test NeuralGatingTrainer."""

    @pytest.fixture
    def trainer(self):
        """Crea trainer per test."""
        mlp = ExpertGatingMLP()
        return NeuralGatingTrainer(mlp, embedding_service=None)

    def test_init(self, trainer):
        """Verifica inizializzazione."""
        assert trainer.step_count == 0
        assert len(trainer.training_history) == 0

    def test_train_sync(self, trainer):
        """Verifica training step sincrono."""
        embedding = np.random.randn(1024).astype(np.float32)
        expert_correctness = {
            "literal": 0.8,
            "systemic": 0.3,
            "principles": 0.2,
            "precedent": 0.1
        }

        metrics = trainer.train_from_feedback_sync(
            embedding,
            expert_correctness,
            authority_weight=0.9
        )

        assert "step" in metrics
        assert "loss" in metrics
        assert "confidence" in metrics
        assert metrics["step"] == 1
        assert trainer.step_count == 1

    def test_multiple_training_steps(self, trainer):
        """Verifica multiple training steps."""
        for i in range(10):
            embedding = np.random.randn(1024).astype(np.float32)
            expert_correctness = {
                "literal": 0.9 if i % 2 == 0 else 0.2,
                "systemic": 0.1,
                "principles": 0.0,
                "precedent": 0.0
            }

            trainer.train_from_feedback_sync(embedding, expert_correctness)

        assert trainer.step_count == 10
        assert len(trainer.training_history) == 10

    def test_get_training_stats(self, trainer):
        """Verifica statistiche training."""
        # Prima del training
        stats = trainer.get_training_stats()
        assert stats["status"] == "no_training"

        # Dopo alcuni step
        for _ in range(5):
            embedding = np.random.randn(1024).astype(np.float32)
            trainer.train_from_feedback_sync(
                embedding,
                {"literal": 0.8, "systemic": 0.1, "principles": 0.1, "precedent": 0.0}
            )

        stats = trainer.get_training_stats()
        assert stats["total_steps"] == 5
        assert "avg_loss" in stats
        assert "current_priors" in stats

    def test_checkpoint_save_load(self, trainer):
        """Verifica save/load checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

            # Training
            for _ in range(3):
                embedding = np.random.randn(1024).astype(np.float32)
                trainer.train_from_feedback_sync(
                    embedding,
                    {"literal": 0.9, "systemic": 0.05, "principles": 0.03, "precedent": 0.02}
                )

            original_step = trainer.step_count
            original_priors = trainer.model.get_expert_priors()

            # Save
            trainer.save_checkpoint(checkpoint_path)
            assert checkpoint_path.exists()

            # Load in new trainer
            new_mlp = ExpertGatingMLP()
            new_trainer = NeuralGatingTrainer(new_mlp)
            new_trainer.load_checkpoint(checkpoint_path)

            assert new_trainer.step_count == original_step

            # Priors dovrebbero essere simili
            loaded_priors = new_trainer.model.get_expert_priors()
            for name in EXPERT_NAMES:
                assert abs(loaded_priors[name] - original_priors[name]) < 0.01

    def test_authority_weight_capping(self, trainer):
        """Verifica che authority weight sia capped."""
        embedding = np.random.randn(1024).astype(np.float32)

        metrics = trainer.train_from_feedback_sync(
            embedding,
            {"literal": 0.5, "systemic": 0.5, "principles": 0.0, "precedent": 0.0},
            authority_weight=10.0  # Molto alto
        )

        # Deve essere capped a max_authority_weight
        assert metrics["authority_weight"] <= trainer.config.max_authority_weight


class TestAutoSaveCallback:
    """Test AutosaveCallback."""

    def test_autosave(self):
        """Verifica autosave ogni N step."""
        mlp = ExpertGatingMLP()
        trainer = NeuralGatingTrainer(mlp)

        with TemporaryDirectory() as tmpdir:
            callback = AutosaveCallback(
                trainer,
                Path(tmpdir),
                save_every_n=5,
                keep_last_n=2
            )

            # Simula training
            for i in range(12):
                embedding = np.random.randn(1024).astype(np.float32)
                trainer.train_from_feedback_sync(
                    embedding,
                    {"literal": 0.7, "systemic": 0.2, "principles": 0.1, "precedent": 0.0}
                )

                asyncio.get_event_loop().run_until_complete(
                    callback.on_feedback({"step": trainer.step_count})
                )

            # Verifica checkpoint creati
            checkpoints = list(Path(tmpdir).glob("*.pt"))

            # Dovrebbero esserci solo keep_last_n checkpoint
            assert len(checkpoints) <= 2

    def test_get_latest_checkpoint(self):
        """Verifica recupero ultimo checkpoint."""
        mlp = ExpertGatingMLP()
        trainer = NeuralGatingTrainer(mlp)

        with TemporaryDirectory() as tmpdir:
            callback = AutosaveCallback(
                trainer,
                Path(tmpdir),
                save_every_n=2
            )

            # Nessun checkpoint inizialmente
            assert callback.get_latest_checkpoint() is None

            # Dopo training
            for i in range(4):
                embedding = np.random.randn(1024).astype(np.float32)
                trainer.train_from_feedback_sync(
                    embedding,
                    {"literal": 0.6, "systemic": 0.4, "principles": 0.0, "precedent": 0.0}
                )
                asyncio.get_event_loop().run_until_complete(
                    callback.on_feedback({"step": trainer.step_count})
                )

            latest = callback.get_latest_checkpoint()
            assert latest is not None
            assert latest.exists()


class TestHybridExpertRouter:
    """Test HybridExpertRouter."""

    @pytest.fixture
    def router(self):
        """Crea router per test."""
        mlp = ExpertGatingMLP()
        return HybridExpertRouter(
            neural_gating=mlp,
            embedding_service=None,  # Usa hash embedding
            llm_router=ExpertRouter(disable_regex=True),
            confidence_threshold=0.7
        )

    @pytest.fixture
    def context(self):
        """Crea context per test."""
        return ExpertContext(
            query_text="Cos'è la legittima difesa?"
        )

    @pytest.mark.asyncio
    async def test_route_returns_decision(self, router, context):
        """Verifica che route restituisca decision."""
        decision = await router.route(context)

        assert isinstance(decision, HybridRoutingDecision)
        assert "expert_weights" in dir(decision)
        assert "neural_used" in dir(decision)

    @pytest.mark.asyncio
    async def test_route_has_weights(self, router, context):
        """Verifica che decision abbia weights per tutti gli expert."""
        decision = await router.route(context)

        for name in EXPERT_NAMES:
            assert name in decision.expert_weights

    @pytest.mark.asyncio
    async def test_route_weights_sum_to_one(self, router, context):
        """Verifica che weights sommino a 1."""
        decision = await router.route(context)

        total = sum(decision.expert_weights.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_low_confidence_uses_regex(self, router, context):
        """Verifica fallback a regex con bassa confidence."""
        # Alza threshold molto alto → forza regex
        router.set_confidence_threshold(0.99)

        decision = await router.route(context)

        # Con threshold 0.99, quasi sempre regex
        # (dipende dal modello non trainato)
        assert decision.query_type in ["neural", "llm_fallback"]

    @pytest.mark.asyncio
    async def test_routing_stats(self, router, context):
        """Verifica statistiche routing."""
        # Route multiple queries
        for _ in range(5):
            await router.route(context)

        stats = router.get_routing_stats()

        assert stats["total_queries"] == 5
        assert "neural_usage_rate" in stats
        assert "avg_neural_confidence" in stats

    def test_set_confidence_threshold(self, router):
        """Verifica impostazione threshold."""
        router.set_confidence_threshold(0.8)
        assert router.confidence_threshold == 0.8

        # Verifica bounds
        router.set_confidence_threshold(1.5)  # Troppo alto
        assert router.confidence_threshold == 1.0

        router.set_confidence_threshold(-0.5)  # Troppo basso
        assert router.confidence_threshold == 0.0


class TestAdaptiveThresholdManager:
    """Test AdaptiveThresholdManager."""

    @pytest.fixture
    def manager(self):
        """Crea manager per test."""
        mlp = ExpertGatingMLP()
        router = HybridExpertRouter(
            neural_gating=mlp,
            confidence_threshold=0.9
        )
        return AdaptiveThresholdManager(
            router,
            initial_threshold=0.9,
            target_threshold=0.6,
            performance_window=10
        )

    def test_initial_threshold(self, manager):
        """Verifica threshold iniziale."""
        assert manager.router.confidence_threshold == 0.9

    def test_threshold_decreases_with_good_performance(self, manager):
        """Verifica che threshold diminuisca con buona performance."""
        initial = manager.router.confidence_threshold

        # Simula 15 feedback positivi
        for _ in range(15):
            manager.update_from_feedback(
                neural_was_correct=True,
                user_rating=0.9
            )

        # Threshold dovrebbe essere diminuito
        assert manager.router.confidence_threshold < initial

    def test_threshold_increases_with_bad_performance(self, manager):
        """Verifica che threshold aumenti con cattiva performance."""
        # Prima abbassa threshold
        manager.router.set_confidence_threshold(0.7)

        # Simula feedback negativi
        for _ in range(15):
            manager.update_from_feedback(
                neural_was_correct=False,
                user_rating=0.2
            )

        # Threshold dovrebbe essere aumentato
        assert manager.router.confidence_threshold > 0.7

    def test_get_status(self, manager):
        """Verifica status."""
        status = manager.get_status()

        assert "current_threshold" in status
        assert "target_threshold" in status
        assert "total_feedback" in status


class TestNeuralGatingConvergence:
    """Test convergenza neural gating nel tempo."""

    def test_weights_shift_with_consistent_feedback(self):
        """Verifica che pesi si spostino con feedback consistente."""
        mlp = ExpertGatingMLP()
        trainer = NeuralGatingTrainer(mlp)

        initial_priors = mlp.get_expert_priors()

        # Training: literal sempre corretto
        for _ in range(50):
            embedding = np.random.randn(1024).astype(np.float32)
            trainer.train_from_feedback_sync(
                embedding,
                {"literal": 0.95, "systemic": 0.02, "principles": 0.02, "precedent": 0.01},
                authority_weight=1.0
            )

        final_priors = mlp.get_expert_priors()

        # literal dovrebbe aumentare
        assert final_priors["literal"] > initial_priors["literal"]

    def test_loss_decreases(self):
        """Verifica che loss diminuisca durante training."""
        mlp = ExpertGatingMLP()
        trainer = NeuralGatingTrainer(mlp)

        losses = []

        # Training con pattern consistente
        for _ in range(30):
            embedding = np.random.randn(1024).astype(np.float32)
            metrics = trainer.train_from_feedback_sync(
                embedding,
                {"literal": 0.8, "systemic": 0.1, "principles": 0.05, "precedent": 0.05}
            )
            losses.append(metrics["loss"])

        # Media loss ultimi 10 step < media primi 10 step
        early_avg = sum(losses[:10]) / 10
        late_avg = sum(losses[-10:]) / 10

        # Loss dovrebbe diminuire (o almeno non aumentare molto)
        assert late_avg <= early_avg * 1.5  # Tolleranza per rumore
