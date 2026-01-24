"""
Test Active Learning
====================

Test per modulo active learning del disagreement detection.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

# Skip se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestActiveLearningConfig:
    """Test per ActiveLearningConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.active_learning import ActiveLearningConfig

        config = ActiveLearningConfig()

        assert config.strategy == "combined"
        assert config.uncertainty_method == "entropy"
        assert config.diversity_method == "coreset"
        assert config.uncertainty_weight == 0.6
        assert config.diversity_weight == 0.4
        assert config.batch_size == 10

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.active_learning import ActiveLearningConfig

        config = ActiveLearningConfig(
            strategy="uncertainty",
            uncertainty_method="margin",
            batch_size=20,
        )

        assert config.strategy == "uncertainty"
        assert config.uncertainty_method == "margin"
        assert config.batch_size == 20


class TestUncertaintyEstimator:
    """Test per UncertaintyEstimator."""

    def test_entropy(self):
        """Test entropy uncertainty."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="entropy")

        # Probabilita' uniforme -> alta uncertainty
        uniform_probs = torch.tensor([[0.5, 0.5]])
        uniform_uncertainty = estimator.compute(uniform_probs)

        # Probabilita' concentrate -> bassa uncertainty
        confident_probs = torch.tensor([[0.99, 0.01]])
        confident_uncertainty = estimator.compute(confident_probs)

        assert uniform_uncertainty > confident_uncertainty

    def test_margin(self):
        """Test margin uncertainty."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="margin")

        # Margine piccolo -> alta uncertainty
        close_probs = torch.tensor([[0.51, 0.49]])
        close_uncertainty = estimator.compute(close_probs)

        # Margine grande -> bassa uncertainty
        wide_probs = torch.tensor([[0.9, 0.1]])
        wide_uncertainty = estimator.compute(wide_probs)

        assert close_uncertainty > wide_uncertainty

    def test_least_confident(self):
        """Test least confident uncertainty."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="least_confident")

        # Bassa confidence -> alta uncertainty
        uncertain_probs = torch.tensor([[0.4, 0.3, 0.3]])
        uncertain_score = estimator.compute(uncertain_probs)

        # Alta confidence -> bassa uncertainty
        confident_probs = torch.tensor([[0.95, 0.03, 0.02]])
        confident_score = estimator.compute(confident_probs)

        assert uncertain_score > confident_score

    def test_batch_computation(self):
        """Test calcolo su batch."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="entropy")

        batch_probs = torch.tensor([
            [0.5, 0.5],
            [0.9, 0.1],
            [0.7, 0.3],
        ])

        uncertainties = estimator.compute(batch_probs)

        assert uncertainties.shape == (3,)
        # Primo (uniforme) piu' incerto del secondo (confident)
        assert uncertainties[0] > uncertainties[1]


class TestDiversitySampler:
    """Test per DiversitySampler."""

    def test_random_selection(self):
        """Test selezione random."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="random")

        embeddings = torch.randn(100, 768)
        selected = sampler.select(embeddings, n_select=10)

        assert len(selected) == 10
        assert len(set(selected)) == 10  # Tutti unici

    def test_random_with_already_selected(self):
        """Test random evitando gia' selezionati."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="random")

        embeddings = torch.randn(20, 768)
        already = [0, 1, 2, 3, 4]
        selected = sampler.select(embeddings, n_select=5, already_selected=already)

        assert len(selected) == 5
        # Nessuna sovrapposizione
        assert len(set(selected) & set(already)) == 0

    def test_coreset_selection(self):
        """Test coreset selection."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="coreset")

        # Crea embeddings con cluster distinti
        cluster1 = torch.randn(10, 64)
        cluster2 = torch.randn(10, 64) + 10  # Offset per separare
        embeddings = torch.cat([cluster1, cluster2], dim=0)

        selected = sampler.select(embeddings, n_select=2)

        assert len(selected) == 2
        # Dovrebbe selezionare da cluster diversi
        # (uno < 10 e uno >= 10 se la selezione funziona)

    def test_coreset_respects_already_selected(self):
        """Test coreset evita gia' selezionati."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="coreset")

        embeddings = torch.randn(20, 64)
        already = [0, 5, 10]
        selected = sampler.select(embeddings, n_select=3, already_selected=already)

        assert len(selected) == 3
        assert len(set(selected) & set(already)) == 0

    def test_kmeans_selection(self):
        """Test kmeans selection."""
        pytest.importorskip("sklearn")

        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="kmeans")

        embeddings = torch.randn(50, 64)
        selected = sampler.select(embeddings, n_select=5)

        assert len(selected) <= 5


class TestActiveLearningScoringIntegration:
    """Test integrazione scoring."""

    def test_uncertainty_scores_range(self):
        """Test che uncertainty scores siano in [0, 1]."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        for method in ["entropy", "margin", "least_confident"]:
            estimator = UncertaintyEstimator(method=method)

            probs = torch.softmax(torch.randn(100, 5), dim=-1)
            scores = estimator.compute(probs)

            assert (scores >= 0).all()
            assert (scores <= 1).all()

    def test_combined_strategy_weights(self):
        """Test che pesi combined sommino a 1."""
        from merlt.disagreement.active_learning import ActiveLearningConfig

        config = ActiveLearningConfig()

        total = config.uncertainty_weight + config.diversity_weight
        assert abs(total - 1.0) < 0.01


class TestAnnotationCandidate:
    """Test per AnnotationCandidate dataclass."""

    def test_annotation_candidate_creation(self):
        """Test creazione AnnotationCandidate."""
        from merlt.disagreement.types import AnnotationCandidate, DisagreementSample, ExpertResponseData

        sample = DisagreementSample(
            sample_id="test",
            query="Test query",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="Test",
                    confidence=0.8,
                ),
            },
        )

        candidate = AnnotationCandidate(
            sample=sample,
            uncertainty=0.75,
            diversity_score=0.6,
            priority_score=0.69,
        )

        assert candidate.uncertainty == 0.75
        assert candidate.diversity_score == 0.6
        assert candidate.priority_score == 0.69


class TestAnnotation:
    """Test per Annotation dataclass."""

    def test_annotation_creation(self):
        """Test creazione Annotation."""
        from merlt.disagreement.types import (
            Annotation,
            DisagreementType,
            DisagreementLevel,
        )

        annotation = Annotation(
            sample_id="sample_123",
            annotator_id="user_123",
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
        )

        assert annotation.has_disagreement is True
        assert annotation.disagreement_type == DisagreementType.METHODOLOGICAL
        assert annotation.intensity == 0.7


class TestUncertaintyMethods:
    """Test dettagliati per i metodi di uncertainty."""

    def test_entropy_normalization(self):
        """Test normalizzazione entropy."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="entropy")

        # 2 classi, distribuzione uniforme: entropy massima
        uniform_2 = torch.tensor([[0.5, 0.5]])
        entropy_2 = estimator.compute(uniform_2)

        # 4 classi, distribuzione uniforme: entropy massima (normalizzata)
        uniform_4 = torch.tensor([[0.25, 0.25, 0.25, 0.25]])
        entropy_4 = estimator.compute(uniform_4)

        # Entrambe dovrebbero essere circa 1.0 (normalizzate)
        assert abs(entropy_2.item() - 1.0) < 0.01
        assert abs(entropy_4.item() - 1.0) < 0.01

    def test_margin_extreme_cases(self):
        """Test margin in casi estremi."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="margin")

        # Probabilita' identiche per top 2 -> margin = 0 -> uncertainty = 1
        equal_top2 = torch.tensor([[0.5, 0.5, 0.0]])
        assert estimator.compute(equal_top2).item() == 1.0

        # Una classe domina -> margin = 1 -> uncertainty = 0
        one_dominant = torch.tensor([[1.0, 0.0, 0.0]])
        assert estimator.compute(one_dominant).item() == 0.0

    def test_least_confident_extreme_cases(self):
        """Test least confident in casi estremi."""
        from merlt.disagreement.active_learning import UncertaintyEstimator

        estimator = UncertaintyEstimator(method="least_confident")

        # Max prob = 1 -> uncertainty = 0
        certain = torch.tensor([[1.0, 0.0]])
        assert estimator.compute(certain).item() == 0.0

        # Max prob bassa
        uncertain = torch.tensor([[0.3, 0.3, 0.4]])
        assert abs(estimator.compute(uncertain).item() - 0.6) < 1e-5  # 1 - 0.4


class TestDiversityEdgeCases:
    """Test edge cases per diversity sampling."""

    def test_select_more_than_available(self):
        """Test selezione piu' samples di quelli disponibili."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="random")

        embeddings = torch.randn(5, 64)
        selected = sampler.select(embeddings, n_select=10)

        # Dovrebbe selezionare al massimo 5
        assert len(selected) <= 5

    def test_empty_available_set(self):
        """Test con tutti gia' selezionati."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="random")

        embeddings = torch.randn(5, 64)
        already = [0, 1, 2, 3, 4]
        selected = sampler.select(embeddings, n_select=3, already_selected=already)

        # Nessuno disponibile
        assert len(selected) == 0

    def test_coreset_single_point(self):
        """Test coreset con singolo punto."""
        from merlt.disagreement.active_learning import DiversitySampler

        sampler = DiversitySampler(method="coreset")

        embeddings = torch.randn(1, 64)
        selected = sampler.select(embeddings, n_select=1)

        assert selected == [0]
