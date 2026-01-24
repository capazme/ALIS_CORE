"""
Test DisagreementTrainer
=========================

Test per training loop con curriculum learning.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

# Skip se torch non disponibile
torch_available = True
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestTrainerConfig:
    """Test per TrainerConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.trainer import TrainerConfig

        config = TrainerConfig()

        assert config.learning_rate == 2e-5
        assert config.batch_size == 16
        assert config.epochs == 50
        assert config.early_stopping_patience == 5
        assert config.curriculum_phase1_epochs == 10

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.trainer import TrainerConfig

        config = TrainerConfig(
            learning_rate=1e-4,
            epochs=100,
            early_stopping_patience=10,
        )

        assert config.learning_rate == 1e-4
        assert config.epochs == 100
        assert config.early_stopping_patience == 10


class TestEpochMetrics:
    """Test per EpochMetrics."""

    def test_creation(self):
        """Test creazione."""
        from merlt.disagreement.trainer import EpochMetrics

        metrics = EpochMetrics(
            epoch=5,
            phase=2,
            train_loss=0.5,
            train_metrics={"accuracy": 0.8},
            val_loss=0.45,
            val_metrics={"accuracy": 0.82},
        )

        assert metrics.epoch == 5
        assert metrics.phase == 2
        assert metrics.train_loss == 0.5


class TestTrainingState:
    """Test per TrainingState."""

    def test_default_state(self):
        """Test stato iniziale."""
        from merlt.disagreement.trainer import TrainingState

        state = TrainingState()

        assert state.current_epoch == 0
        assert state.global_step == 0
        assert state.best_val_loss == float("inf")
        assert state.history == []


class TestCurriculumScheduler:
    """Test per CurriculumScheduler."""

    def test_get_phase(self):
        """Test determinazione fase."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=10, phase2_epochs=20)

        assert scheduler.get_phase(0) == 1
        assert scheduler.get_phase(9) == 1
        assert scheduler.get_phase(10) == 2
        assert scheduler.get_phase(29) == 2
        assert scheduler.get_phase(30) == 3
        assert scheduler.get_phase(100) == 3

    def test_get_task_mask_phase1(self):
        """Test mask fase 1."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=10, phase2_epochs=20)
        mask = scheduler.get_task_mask(epoch=5)

        assert mask["binary"] is True
        assert mask["type"] is False
        assert mask["level"] is False
        assert mask["intensity"] is False

    def test_get_task_mask_phase2(self):
        """Test mask fase 2."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=10, phase2_epochs=20)
        mask = scheduler.get_task_mask(epoch=15)

        assert mask["binary"] is True
        assert mask["type"] is True
        assert mask["level"] is True
        assert mask["intensity"] is False

    def test_get_task_mask_phase3(self):
        """Test mask fase 3."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=10, phase2_epochs=20)
        mask = scheduler.get_task_mask(epoch=35)

        assert mask["binary"] is True
        assert mask["type"] is True
        assert mask["level"] is True
        assert mask["intensity"] is True
        assert mask["pairwise"] is True

    def test_get_task_weights_phase2_ramp(self):
        """Test ramp up pesi in fase 2."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=10, phase2_epochs=20)

        weights_start = scheduler.get_task_weights(10)  # Inizio fase 2
        weights_end = scheduler.get_task_weights(29)    # Fine fase 2

        # Type weight dovrebbe aumentare
        assert weights_end["type"] > weights_start["type"]


class TestDisagreementTrainer:
    """Test per DisagreementTrainer."""

    @pytest.fixture
    def mock_model(self):
        """Crea mock del modello."""
        from merlt.disagreement.heads import HeadsOutput

        model = MagicMock(spec=nn.Module)
        model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
        model.state_dict.return_value = {"weight": torch.randn(10)}
        model.load_state_dict = MagicMock()
        model.to.return_value = model
        model.train.return_value = None
        model.eval.return_value = None
        model.zero_grad.return_value = None

        # Mock forward
        def mock_forward(inputs, *args, **kwargs):
            batch_size = inputs.shape[0] if hasattr(inputs, 'shape') else 4
            return HeadsOutput(
                binary_logits=torch.randn(batch_size, 2),
                binary_probs=torch.softmax(torch.randn(batch_size, 2), dim=-1),
                type_logits=torch.randn(batch_size, 6),
                type_probs=torch.softmax(torch.randn(batch_size, 6), dim=-1),
                level_logits=torch.randn(batch_size, 4),
                level_probs=torch.softmax(torch.randn(batch_size, 4), dim=-1),
                intensity=torch.rand(batch_size, 1),
                resolvability=torch.rand(batch_size, 1),
                pairwise_matrix=torch.rand(batch_size, 4, 4),
                confidence=torch.rand(batch_size, 1),
            )

        model.return_value = mock_forward(torch.randn(4, 4, 768))
        model.__call__ = mock_forward

        return model

    @pytest.fixture
    def mock_loss_fn(self):
        """Crea mock loss function."""
        def loss_fn(outputs, targets, mask=None):
            return {
                "total": torch.tensor(0.5, requires_grad=True),
                "binary": torch.tensor(0.3, requires_grad=True),
                "metrics": {"binary_accuracy": 0.8},
            }
        return loss_fn

    @pytest.fixture
    def mock_dataloader(self):
        """Crea mock dataloader."""
        batch = {
            "expert_embeddings": torch.randn(4, 4, 768),
            "binary_target": torch.tensor([0, 1, 1, 0]),
            "type_target": torch.tensor([-1, 2, 3, -1]),
            "level_target": torch.tensor([-1, 1, 2, -1]),
            "intensity_target": torch.tensor([0.0, 0.7, 0.8, 0.0]),
            "resolvability_target": torch.tensor([0.5, 0.4, 0.3, 0.5]),
            "conflicting_pairs": None,
        }
        return [batch, batch]  # 2 batches

    def test_initialization(self, mock_model, mock_loss_fn):
        """Test inizializzazione trainer."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            assert trainer.model is mock_model
            assert trainer.optimizer is not None
            assert trainer.curriculum is not None

    def test_batch_to_device(self, mock_model, mock_loss_fn):
        """Test spostamento batch su device."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            batch = {
                "tensor": torch.randn(4, 10),
                "list": [1, 2, 3],
                "string": "test",
            }

            result = trainer._batch_to_device(batch)

            assert isinstance(result["tensor"], torch.Tensor)
            assert result["list"] == [1, 2, 3]
            assert result["string"] == "test"

    def test_prepare_targets_phase1(self, mock_model, mock_loss_fn):
        """Test preparazione targets fase 1."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            batch = {
                "binary_target": torch.tensor([0, 1]),
                "type_target": torch.tensor([2, 3]),
                "level_target": torch.tensor([1, 2]),
                "intensity_target": torch.tensor([0.5, 0.7]),
                "resolvability_target": torch.tensor([0.4, 0.3]),
            }

            task_mask = {
                "binary": True,
                "type": False,
                "level": False,
                "intensity": False,
                "resolvability": False,
                "pairwise": False,
            }

            targets = trainer._prepare_targets(batch, task_mask)

            assert targets["binary"] is not None
            assert targets["type"] is None
            assert targets["level"] is None

    def test_save_checkpoint(self, mock_model, mock_loss_fn):
        """Test salvataggio checkpoint."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            path = trainer.save_checkpoint("test_checkpoint.pt")

            assert Path(path).exists()

    def test_load_checkpoint(self, mock_model, mock_loss_fn):
        """Test caricamento checkpoint."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            # Modifica stato
            trainer.state.current_epoch = 5
            trainer.state.best_val_loss = 0.3

            # Salva
            path = trainer.save_checkpoint("test.pt")

            # Nuovo trainer
            trainer2 = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            # Carica
            trainer2.load_checkpoint(path)

            assert trainer2.state.current_epoch == 5
            assert trainer2.state.best_val_loss == 0.3

    def test_get_training_summary(self, mock_model, mock_loss_fn):
        """Test summary training."""
        from merlt.disagreement.trainer import DisagreementTrainer, TrainerConfig

        with TemporaryDirectory() as tmpdir:
            config = TrainerConfig(checkpoint_dir=tmpdir)
            trainer = DisagreementTrainer(
                model=mock_model,
                loss_fn=mock_loss_fn,
                config=config,
            )

            summary = trainer.get_training_summary()

            assert "current_epoch" in summary
            assert "global_step" in summary
            assert "best_val_loss" in summary
            assert "current_phase" in summary


class TestCurriculumProgression:
    """Test integrazione curriculum nel training."""

    def test_phase_progression_in_training(self):
        """Test che le fasi cambino durante il training."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=5, phase2_epochs=10)

        phases = []
        for epoch in range(20):
            phases.append(scheduler.get_phase(epoch))

        # Verifica progressione corretta
        assert phases[0] == 1   # Epoch 0
        assert phases[4] == 1   # Epoch 4 (ultimo phase 1)
        assert phases[5] == 2   # Epoch 5 (primo phase 2)
        assert phases[14] == 2  # Epoch 14 (ultimo phase 2)
        assert phases[15] == 3  # Epoch 15 (primo phase 3)

    def test_task_weights_progression(self):
        """Test che i pesi task cambino correttamente."""
        from merlt.disagreement.trainer import CurriculumScheduler

        scheduler = CurriculumScheduler(phase1_epochs=5, phase2_epochs=10)

        # Phase 1: solo binary
        w1 = scheduler.get_task_weights(0)
        assert "binary" in w1
        assert w1["binary"] == 1.0

        # Phase 2: binary + type + level
        w2 = scheduler.get_task_weights(7)
        assert "type" in w2
        assert "level" in w2

        # Phase 3: tutti
        w3 = scheduler.get_task_weights(20)
        assert "intensity" in w3
        assert "pairwise" in w3
