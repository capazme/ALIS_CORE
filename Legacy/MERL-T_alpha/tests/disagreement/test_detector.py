"""
Test LegalDisagreementNet Detector
==================================

Test per modulo detector del disagreement detection.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

# Skip se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestDetectorConfig:
    """Test per DetectorConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.detector import DetectorConfig

        config = DetectorConfig()

        assert config.hidden_size == 768
        assert config.num_types == 6
        assert config.num_levels == 4
        assert config.num_experts == 4
        assert config.dropout == 0.1
        assert config.binary_threshold == 0.5

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.detector import DetectorConfig

        config = DetectorConfig(
            hidden_size=512,
            num_types=5,
            dropout=0.2,
            binary_threshold=0.6,
        )

        assert config.hidden_size == 512
        assert config.num_types == 5
        assert config.dropout == 0.2
        assert config.binary_threshold == 0.6

    def test_from_env(self):
        """Test creazione da env."""
        from merlt.disagreement.detector import DetectorConfig
        import os

        # Set env var
        original = os.environ.get("DISAGREEMENT_THRESHOLD")
        os.environ["DISAGREEMENT_THRESHOLD"] = "0.7"

        try:
            config = DetectorConfig.from_env()
            assert config.binary_threshold == 0.7
        finally:
            # Restore
            if original:
                os.environ["DISAGREEMENT_THRESHOLD"] = original
            else:
                os.environ.pop("DISAGREEMENT_THRESHOLD", None)


class TestDisagreementAnalysisCreation:
    """Test creazione DisagreementAnalysis."""

    def test_create_analysis_with_disagreement(self):
        """Test creazione analisi con disagreement."""
        from merlt.disagreement.types import (
            DisagreementAnalysis,
            DisagreementType,
            DisagreementLevel,
            ExpertPairConflict,
        )

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.75,
            resolvability=0.4,
            confidence=0.85,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="literal",
                    expert_b="principles",
                    conflict_score=0.8,
                )
            ],
        )

        assert analysis.has_disagreement is True
        assert analysis.disagreement_type == DisagreementType.METHODOLOGICAL
        assert analysis.intensity == 0.75
        assert len(analysis.conflicting_pairs) == 1

    def test_create_analysis_without_disagreement(self):
        """Test creazione analisi senza disagreement."""
        from merlt.disagreement.types import DisagreementAnalysis

        analysis = DisagreementAnalysis(
            has_disagreement=False,
            confidence=0.9,
        )

        assert analysis.has_disagreement is False
        assert analysis.disagreement_type is None
        assert analysis.confidence == 0.9


class TestDetectorHelpers:
    """Test helper methods del detector."""

    def test_expert_pair_conflict_creation(self):
        """Test creazione ExpertPairConflict."""
        from merlt.disagreement.types import ExpertPairConflict

        conflict = ExpertPairConflict(
            expert_a="literal",
            expert_b="systemic",
            conflict_score=0.75,
            contention_point="Interpretazione dell'art. 1372 c.c.",
        )

        assert conflict.expert_a == "literal"
        assert conflict.expert_b == "systemic"
        assert conflict.conflict_score == 0.75
        assert "art. 1372" in conflict.contention_point

    def test_expert_pair_conflict_serialization(self):
        """Test serializzazione ExpertPairConflict."""
        from merlt.disagreement.types import ExpertPairConflict

        conflict = ExpertPairConflict(
            expert_a="literal",
            expert_b="principles",
            conflict_score=0.8,
        )

        d = conflict.to_dict()

        assert d["expert_a"] == "literal"
        assert d["expert_b"] == "principles"
        assert d["conflict_score"] == 0.8


class TestDetectorOutputProcessing:
    """Test processing output del detector."""

    def test_heads_output_creation(self):
        """Test creazione HeadsOutput."""
        from merlt.disagreement.heads import HeadsOutput

        batch_size = 2

        output = HeadsOutput(
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

        assert output.binary_logits.shape == (batch_size, 2)
        assert output.type_probs.shape == (batch_size, 6)
        assert output.pairwise_matrix.shape == (batch_size, 4, 4)

    def test_heads_output_to_dict(self):
        """Test serializzazione HeadsOutput."""
        from merlt.disagreement.heads import HeadsOutput

        output = HeadsOutput(
            binary_logits=torch.tensor([[1.0, -1.0]]),
            binary_probs=torch.tensor([[0.88, 0.12]]),
            type_logits=torch.randn(1, 6),
            type_probs=torch.softmax(torch.randn(1, 6), dim=-1),
            level_logits=torch.randn(1, 4),
            level_probs=torch.softmax(torch.randn(1, 4), dim=-1),
            intensity=torch.tensor([[0.7]]),
            resolvability=torch.tensor([[0.4]]),
            pairwise_matrix=torch.rand(1, 4, 4),
            confidence=torch.tensor([[0.85]]),
        )

        d = output.to_dict()

        assert "binary_probs" in d
        assert "intensity" in d
        assert "confidence" in d


class TestDetectorTypeMappings:
    """Test mappings tipi e livelli."""

    def test_disagreement_type_values(self):
        """Test valori DisagreementType."""
        from merlt.disagreement.types import DisagreementType

        # Test alcuni valori principali
        assert DisagreementType.ANTINOMY.value == "ANT"
        assert DisagreementType.METHODOLOGICAL.value == "MET"
        assert DisagreementType.INTERPRETIVE_GAP.value == "LAC"
        assert DisagreementType.HIERARCHICAL.value == "GER"
        assert DisagreementType.OVERRULING.value == "OVR"
        assert DisagreementType.SPECIALIZATION.value == "SPE"

    def test_disagreement_level_values(self):
        """Test valori DisagreementLevel."""
        from merlt.disagreement.types import DisagreementLevel

        assert DisagreementLevel.SEMANTIC.value == "SEM"
        assert DisagreementLevel.SYSTEMIC.value == "SIS"
        assert DisagreementLevel.TELEOLOGICAL.value == "TEL"

    def test_expert_names(self):
        """Test lista EXPERT_NAMES."""
        from merlt.disagreement.types import EXPERT_NAMES

        assert "literal" in EXPERT_NAMES
        assert "systemic" in EXPERT_NAMES
        assert "principles" in EXPERT_NAMES
        assert "precedent" in EXPERT_NAMES
        assert len(EXPERT_NAMES) == 4


class TestDetectorIntegration:
    """Test integrazione detector."""

    @pytest.fixture
    def mock_encoder(self):
        """Crea mock encoder."""
        encoder = MagicMock()
        encoder.encode.return_value = torch.randn(1, 4, 768)
        encoder.eval.return_value = None
        encoder.initialize.return_value = None
        return encoder

    @pytest.fixture
    def mock_heads(self):
        """Crea mock heads."""
        from merlt.disagreement.heads import HeadsOutput

        heads = MagicMock()
        heads.return_value = HeadsOutput(
            binary_logits=torch.tensor([[1.5, -1.5]]),
            binary_probs=torch.tensor([[0.95, 0.05]]),
            type_logits=torch.randn(1, 6),
            type_probs=torch.softmax(torch.randn(1, 6), dim=-1),
            level_logits=torch.randn(1, 4),
            level_probs=torch.softmax(torch.randn(1, 4), dim=-1),
            intensity=torch.tensor([[0.7]]),
            resolvability=torch.tensor([[0.4]]),
            pairwise_matrix=torch.rand(1, 4, 4),
            confidence=torch.tensor([[0.85]]),
        )
        return heads

    def test_analysis_confidence_range(self):
        """Test che confidence sia in [0, 1]."""
        from merlt.disagreement.types import DisagreementAnalysis

        # Test vari valori
        for conf in [0.0, 0.5, 1.0, 0.85]:
            analysis = DisagreementAnalysis(
                has_disagreement=True,
                confidence=conf,
            )
            assert 0 <= analysis.confidence <= 1

    def test_analysis_intensity_range(self):
        """Test che intensity sia in [0, 1]."""
        from merlt.disagreement.types import DisagreementAnalysis

        # Con intensity esplicita
        analysis1 = DisagreementAnalysis(
            has_disagreement=True,
            intensity=0.75,
            confidence=0.8,
        )
        assert 0 <= analysis1.intensity <= 1

        # Senza intensity esplicita (ha default)
        analysis2 = DisagreementAnalysis(
            has_disagreement=False,
            confidence=0.9,
        )
        assert 0 <= analysis2.intensity <= 1  # Default value


class TestDetectorInputValidation:
    """Test validazione input."""

    def test_empty_expert_responses(self):
        """Test con expert responses vuote."""
        # Il detector dovrebbe gestire gracefully input vuoti
        from merlt.disagreement.types import DisagreementSample, ExpertResponseData

        sample = DisagreementSample(
            sample_id="empty_test",
            query="Test query",
            expert_responses={},
        )

        assert len(sample.expert_responses) == 0

    def test_partial_expert_responses(self):
        """Test con solo alcuni expert."""
        from merlt.disagreement.types import DisagreementSample, ExpertResponseData

        sample = DisagreementSample(
            sample_id="partial_test",
            query="Test query",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="Solo questo expert",
                    confidence=0.8,
                ),
            },
        )

        assert len(sample.expert_responses) == 1
        assert "literal" in sample.expert_responses


class TestDetectorEdgeCases:
    """Test edge cases."""

    def test_very_high_confidence(self):
        """Test con confidence molto alta."""
        from merlt.disagreement.types import DisagreementAnalysis

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            confidence=0.99,
        )

        assert analysis.confidence == 0.99

    def test_very_low_confidence(self):
        """Test con confidence molto bassa."""
        from merlt.disagreement.types import DisagreementAnalysis

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            confidence=0.51,
        )

        assert analysis.confidence == 0.51

    def test_multiple_conflicting_pairs(self):
        """Test con multiple coppie in conflitto."""
        from merlt.disagreement.types import (
            DisagreementAnalysis,
            ExpertPairConflict,
        )

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            confidence=0.8,
            conflicting_pairs=[
                ExpertPairConflict("literal", "systemic", 0.7),
                ExpertPairConflict("literal", "principles", 0.8),
                ExpertPairConflict("systemic", "precedent", 0.65),
            ],
        )

        assert len(analysis.conflicting_pairs) == 3
