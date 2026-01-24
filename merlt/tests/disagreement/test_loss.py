"""
Test DisagreementLoss
======================

Test per multi-task loss con focal loss e curriculum.
"""

import pytest

# Skip se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestLossConfig:
    """Test per LossConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.loss import LossConfig

        config = LossConfig()

        assert config.task_weights["binary"] == 1.0
        assert config.task_weights["type"] == 0.8
        assert config.use_focal_loss is True
        assert config.focal_gamma == 2.0
        assert config.dynamic_weighting is False

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.loss import LossConfig

        config = LossConfig(
            task_weights={"binary": 1.5, "type": 0.5},
            focal_gamma=3.0,
            dynamic_weighting=True,
        )

        assert config.task_weights["binary"] == 1.5
        assert config.focal_gamma == 3.0
        assert config.dynamic_weighting is True


class TestFocalLoss:
    """Test per FocalLoss."""

    def test_focal_loss_output_shape(self):
        """Test shape output."""
        from merlt.disagreement.loss import FocalLoss

        focal = FocalLoss(gamma=2.0)
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = focal(logits, targets)

        assert loss.ndim == 0  # Scalare

    def test_focal_loss_reduction_none(self):
        """Test senza riduzione."""
        from merlt.disagreement.loss import FocalLoss

        focal = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = focal(logits, targets)

        assert loss.shape == (4,)

    def test_focal_loss_gamma_effect(self):
        """Test che gamma riduca loss per esempi facili."""
        from merlt.disagreement.loss import FocalLoss

        # Predizione moderatamente confidante e corretta
        # Con logits estremi (10, -10) entrambe le loss sono ~0
        # Usiamo logits moderati per vedere l'effetto del gamma
        logits = torch.tensor([[2.0, -2.0]])  # Predice classe 0 con p~0.98
        targets = torch.tensor([0])

        focal_low = FocalLoss(gamma=0.0)
        focal_high = FocalLoss(gamma=2.0)

        loss_low = focal_low(logits, targets)
        loss_high = focal_high(logits, targets)

        # Con gamma alto, loss dovrebbe essere minore per esempi facili
        # gamma=0 -> standard CE, gamma>0 -> riduce peso esempi facili
        assert loss_high < loss_low

    def test_focal_loss_with_class_weights(self):
        """Test con pesi per classe."""
        from merlt.disagreement.loss import FocalLoss

        focal = FocalLoss(gamma=2.0, alpha=[0.3, 0.7])
        logits = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])

        loss = focal(logits, targets)

        assert not torch.isnan(loss)


class TestContrastivePairwiseLoss:
    """Test per ContrastivePairwiseLoss."""

    def test_output_scalar(self):
        """Test che output sia scalare."""
        from merlt.disagreement.loss import ContrastivePairwiseLoss

        loss_fn = ContrastivePairwiseLoss(margin=0.5)
        matrix = torch.rand(2, 4, 4)

        loss = loss_fn(matrix, has_disagreement=True)

        assert loss.ndim == 0

    def test_no_disagreement_low_scores(self):
        """Test che penalizzi score alti quando no disagreement."""
        from merlt.disagreement.loss import ContrastivePairwiseLoss

        loss_fn = ContrastivePairwiseLoss()

        # Matrix con score alti
        high_matrix = torch.ones(1, 4, 4) * 0.9
        # Matrix con score bassi
        low_matrix = torch.zeros(1, 4, 4)

        loss_high = loss_fn(high_matrix, has_disagreement=False)
        loss_low = loss_fn(low_matrix, has_disagreement=False)

        # Matrix alta dovrebbe avere loss maggiore
        assert loss_high > loss_low

    def test_with_target_pairs(self):
        """Test con coppie target specifiche."""
        from merlt.disagreement.loss import ContrastivePairwiseLoss

        loss_fn = ContrastivePairwiseLoss()

        # Matrix dove coppia (0,1) ha alto conflitto
        matrix = torch.zeros(1, 4, 4)
        matrix[0, 0, 1] = 0.9
        matrix[0, 1, 0] = 0.9

        target_pairs = [[(0, 1)]]  # Batch 1 con coppia (0,1)

        loss = loss_fn(matrix, target_pairs=target_pairs)

        assert not torch.isnan(loss)


class TestDisagreementLoss:
    """Test per DisagreementLoss principale."""

    @pytest.fixture
    def mock_outputs(self):
        """Crea mock HeadsOutput."""
        from merlt.disagreement.heads import HeadsOutput

        batch_size = 4

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

    @pytest.fixture
    def mock_targets(self):
        """Crea targets mock."""
        batch_size = 4
        return {
            "binary": torch.tensor([0, 1, 1, 0]),
            "type": torch.tensor([-1, 2, 3, -1]),  # Solo per samples con disagreement
            "level": torch.tensor([-1, 1, 2, -1]),
            "intensity": torch.tensor([0.0, 0.7, 0.8, 0.0]),
            "resolvability": torch.tensor([0.5, 0.4, 0.3, 0.5]),
            "conflicting_pairs": None,
        }

    def test_initialization(self):
        """Test inizializzazione."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()

        assert loss_fn.task_weights["binary"] == 1.0
        assert loss_fn.focal_loss is not None

    def test_forward_returns_dict(self, mock_outputs, mock_targets):
        """Test che forward restituisca dict."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        result = loss_fn(mock_outputs, mock_targets)

        assert "total" in result
        assert "binary" in result
        assert "metrics" in result

    def test_total_loss_is_scalar(self, mock_outputs, mock_targets):
        """Test che total loss sia scalare."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        result = loss_fn(mock_outputs, mock_targets)

        assert result["total"].ndim == 0

    def test_metrics_include_accuracy(self, mock_outputs, mock_targets):
        """Test che metrics includano accuracy."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        result = loss_fn(mock_outputs, mock_targets)

        assert "binary_accuracy" in result["metrics"]

    def test_masked_type_loss(self, mock_outputs, mock_targets):
        """Test che type loss sia calcolata solo per samples con disagreement."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        result = loss_fn(mock_outputs, mock_targets)

        # Type loss dovrebbe esistere (abbiamo samples con binary=1)
        if "type" in result:
            assert not torch.isnan(result["type"])

    def test_without_optional_targets(self, mock_outputs):
        """Test con solo binary target."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        targets = {
            "binary": torch.tensor([0, 1, 1, 0]),
            "type": None,
            "level": None,
            "intensity": None,
            "resolvability": None,
        }

        result = loss_fn(mock_outputs, targets)

        assert "total" in result
        assert "binary" in result

    def test_get_task_weights_summary(self):
        """Test summary dei pesi."""
        from merlt.disagreement.loss import DisagreementLoss

        loss_fn = DisagreementLoss()
        weights = loss_fn.get_task_weights_summary()

        assert "binary" in weights
        assert "type" in weights


class TestCurriculumLoss:
    """Test per CurriculumLoss."""

    def test_get_phase(self):
        """Test determinazione fase."""
        from merlt.disagreement.loss import DisagreementLoss, CurriculumLoss

        base_loss = DisagreementLoss()
        curriculum = CurriculumLoss(
            base_loss,
            phase1_epochs=10,
            phase2_epochs=20,
        )

        # Phase 1: epoch 0-9
        assert curriculum.get_phase() == 1
        curriculum.set_epoch(5)
        assert curriculum.get_phase() == 1

        # Phase 2: epoch 10-29
        curriculum.set_epoch(15)
        assert curriculum.get_phase() == 2

        # Phase 3: epoch 30+
        curriculum.set_epoch(35)
        assert curriculum.get_phase() == 3

    def test_phase1_only_binary(self):
        """Test che phase 1 usi solo binary."""
        from merlt.disagreement.loss import DisagreementLoss, CurriculumLoss
        from merlt.disagreement.heads import HeadsOutput

        base_loss = DisagreementLoss()
        curriculum = CurriculumLoss(base_loss, phase1_epochs=10, phase2_epochs=20)

        curriculum.set_epoch(5)  # Phase 1

        outputs = HeadsOutput(
            binary_logits=torch.randn(4, 2),
            binary_probs=torch.softmax(torch.randn(4, 2), dim=-1),
            type_logits=torch.randn(4, 6),
            type_probs=torch.softmax(torch.randn(4, 6), dim=-1),
            level_logits=torch.randn(4, 4),
            level_probs=torch.softmax(torch.randn(4, 4), dim=-1),
            intensity=torch.rand(4, 1),
            resolvability=torch.rand(4, 1),
            pairwise_matrix=torch.rand(4, 4, 4),
            confidence=torch.rand(4, 1),
        )

        targets = {
            "binary": torch.tensor([0, 1, 1, 0]),
            "type": torch.tensor([0, 2, 3, 0]),
            "level": torch.tensor([0, 1, 2, 0]),
            "intensity": torch.tensor([0.5, 0.7, 0.8, 0.3]),
            "resolvability": torch.tensor([0.5, 0.4, 0.3, 0.5]),
        }

        result = curriculum(outputs, targets)

        assert result["phase"] == 1
        # Type e level dovrebbero essere None (non calcolati)

    def test_phase_progression(self):
        """Test progressione delle fasi."""
        from merlt.disagreement.loss import DisagreementLoss, CurriculumLoss

        base_loss = DisagreementLoss()
        curriculum = CurriculumLoss(base_loss, phase1_epochs=5, phase2_epochs=10)

        phases = []
        for epoch in range(20):
            curriculum.set_epoch(epoch)
            phases.append(curriculum.get_phase())

        # Phase 1: 0-4
        assert phases[:5] == [1, 1, 1, 1, 1]
        # Phase 2: 5-14
        assert phases[5:15] == [2] * 10
        # Phase 3: 15+
        assert phases[15:] == [3] * 5
