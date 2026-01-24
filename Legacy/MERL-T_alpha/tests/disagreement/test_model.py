"""
Test LegalDisagreementNet Model
================================

Test per encoder, heads e detector del modello di disagreement.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import sys

# Skip tests se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


class TestEncoderConfig:
    """Test per EncoderConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.encoder import EncoderConfig

        config = EncoderConfig()

        assert config.model_name == "dlicari/Italian-Legal-BERT"
        assert config.hidden_size == 768
        assert config.max_length == 512
        assert config.use_lora is True
        assert config.lora_rank == 8
        assert config.pooling_strategy == "cls"

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.encoder import EncoderConfig

        config = EncoderConfig(
            model_name="dbmdz/bert-base-italian-xxl-cased",
            use_lora=False,
            max_length=256,
        )

        assert config.model_name == "dbmdz/bert-base-italian-xxl-cased"
        assert config.use_lora is False
        assert config.max_length == 256


class TestPredictionHeads:
    """Test per PredictionHeads."""

    @pytest.fixture
    def mock_embeddings(self):
        """Crea embeddings mock."""
        import torch
        # [batch=2, num_experts=4, hidden=768]
        return torch.randn(2, 4, 768)

    def test_heads_initialization(self):
        """Test inizializzazione heads."""
        from merlt.disagreement.heads import PredictionHeads

        heads = PredictionHeads(
            hidden_size=768,
            num_types=6,
            num_levels=4,
            num_experts=4,
        )

        assert heads.hidden_size == 768
        assert heads.num_experts == 4

    def test_heads_forward(self, mock_embeddings):
        """Test forward pass."""
        from merlt.disagreement.heads import PredictionHeads, HeadsOutput

        heads = PredictionHeads(hidden_size=768)
        outputs = heads(mock_embeddings)

        assert isinstance(outputs, HeadsOutput)

        # Binary
        assert outputs.binary_logits.shape == (2, 2)
        assert outputs.binary_probs.shape == (2, 2)

        # Type
        assert outputs.type_logits.shape == (2, 6)
        assert outputs.type_probs.shape == (2, 6)

        # Level
        assert outputs.level_logits.shape == (2, 4)
        assert outputs.level_probs.shape == (2, 4)

        # Regression
        assert outputs.intensity.shape == (2, 1)
        assert outputs.resolvability.shape == (2, 1)

        # Pairwise
        assert outputs.pairwise_matrix.shape == (2, 4, 4)

        # Confidence
        assert outputs.confidence.shape == (2, 1)

    def test_heads_output_to_dict(self, mock_embeddings):
        """Test serializzazione output."""
        from merlt.disagreement.heads import PredictionHeads

        heads = PredictionHeads(hidden_size=768)
        outputs = heads(mock_embeddings)

        d = outputs.to_dict()

        assert "binary_probs" in d
        assert "type_probs" in d
        assert "intensity" in d
        assert isinstance(d["binary_probs"], list)

    def test_heads_parameters(self):
        """Test che parameters restituisce parametri trainabili."""
        from merlt.disagreement.heads import PredictionHeads

        heads = PredictionHeads(hidden_size=768)
        params = list(heads.parameters())

        assert len(params) > 0
        for p in params:
            assert isinstance(p, torch.nn.Parameter) or hasattr(p, 'requires_grad')


class TestCrossExpertAttention:
    """Test per CrossExpertAttention."""

    @pytest.fixture
    def mock_embeddings(self):
        """Crea embeddings mock."""
        import torch
        return torch.randn(2, 4, 768)

    def test_attention_initialization(self):
        """Test inizializzazione."""
        from merlt.disagreement.heads import CrossExpertAttention

        attention = CrossExpertAttention(
            hidden_size=768,
            num_experts=4,
            num_heads=4,
        )

        assert attention.hidden_size == 768
        assert attention.num_experts == 4

    def test_attention_forward(self, mock_embeddings):
        """Test forward pass."""
        from merlt.disagreement.heads import CrossExpertAttention

        attention = CrossExpertAttention(hidden_size=768)
        outputs = attention(mock_embeddings)

        assert "attended" in outputs
        assert "attention_weights" in outputs
        assert "contrast_features" in outputs
        assert "aggregate" in outputs

        assert outputs["attended"].shape == (2, 4, 768)
        assert outputs["aggregate"].shape == (2, 768)
        assert outputs["contrast_features"].shape == (2, 4, 4, 768)


class TestBinaryHead:
    """Test per BinaryHead."""

    def test_binary_head(self):
        """Test binary head."""
        import torch
        from merlt.disagreement.heads import BinaryHead

        head = BinaryHead(hidden_size=768)
        x = torch.randn(2, 768)

        output = head(x)

        assert output.shape == (2, 2)


class TestTypeHead:
    """Test per TypeHead."""

    def test_type_head(self):
        """Test type head."""
        import torch
        from merlt.disagreement.heads import TypeHead

        head = TypeHead(hidden_size=768, num_types=6)
        x = torch.randn(2, 768)

        output = head(x)

        assert output.shape == (2, 6)


class TestIntensityHead:
    """Test per IntensityHead."""

    def test_intensity_head(self):
        """Test intensity head."""
        import torch
        from merlt.disagreement.heads import IntensityHead

        head = IntensityHead(hidden_size=768)
        x = torch.randn(2, 768)

        output = head(x)

        assert output.shape == (2, 1)
        # Sigmoid constrains to [0, 1]
        assert (output >= 0).all() and (output <= 1).all()


class TestPairwiseHead:
    """Test per PairwiseHead."""

    def test_pairwise_head(self):
        """Test pairwise head."""
        import torch
        from merlt.disagreement.heads import PairwiseHead

        head = PairwiseHead(hidden_size=768, num_experts=4)
        x = torch.randn(2, 4, 768)

        output = head(x)

        assert output.shape == (2, 4, 4)
        # Matrice simmetrica
        assert torch.allclose(output, output.transpose(1, 2), atol=1e-5)


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
        assert config.binary_threshold == 0.5


class TestLegalDisagreementNetUnit:
    """Test unitari per LegalDisagreementNet (senza modello reale)."""

    def test_model_creation(self):
        """Test creazione modello."""
        from merlt.disagreement.detector import LegalDisagreementNet

        # Crea senza inizializzare encoder (lazy)
        model = LegalDisagreementNet(
            encoder_model="test-model",
            use_lora=False,
            device="cpu",
        )

        assert model.device == "cpu"
        assert model._initialized is False

    def test_prepare_inputs(self):
        """Test preparazione input."""
        from merlt.disagreement.detector import LegalDisagreementNet

        model = LegalDisagreementNet(device="cpu")

        expert_responses = {
            "LiteralExpert": "Interpretazione letterale...",
            "PrinciplesExpert": "Secondo i principi...",
        }

        prepared = model._prepare_inputs(expert_responses, query="Test query")

        assert "literal" in prepared
        assert "principles" in prepared
        assert "systemic" in prepared  # Default per mancanti
        assert "precedent" in prepared

        assert "Test query" in prepared["literal"]
        assert "Interpretazione letterale" in prepared["literal"]

    def test_extract_conflicts(self):
        """Test estrazione conflitti da matrice."""
        import torch
        from merlt.disagreement.detector import LegalDisagreementNet

        model = LegalDisagreementNet(device="cpu")

        # Matrice con un conflitto alto
        pairwise = torch.zeros(4, 4)
        pairwise[0, 1] = 0.8  # literal vs systemic
        pairwise[1, 0] = 0.8
        pairwise[2, 3] = 0.3  # principles vs precedent (sotto soglia)
        pairwise[3, 2] = 0.3

        conflicts = model._extract_conflicts(
            pairwise,
            expert_responses={},
            threshold=0.5,
        )

        assert len(conflicts) == 1
        assert conflicts[0].expert_a == "literal"
        assert conflicts[0].expert_b == "systemic"
        assert conflicts[0].conflict_score == pytest.approx(0.8)

    def test_parameters(self):
        """Test che parameters restituisce parametri."""
        from merlt.disagreement.detector import LegalDisagreementNet

        model = LegalDisagreementNet(device="cpu")

        # Solo heads parameters senza inizializzare encoder
        params = list(model.heads.parameters())
        assert len(params) > 0


class TestHeadsOutputIntegration:
    """Test integrazione HeadsOutput con DisagreementAnalysis."""

    def test_outputs_to_analysis(self):
        """Test conversione outputs in analysis."""
        import torch
        from merlt.disagreement.detector import LegalDisagreementNet
        from merlt.disagreement.heads import HeadsOutput

        model = LegalDisagreementNet(device="cpu")

        # Mock outputs
        outputs = HeadsOutput(
            binary_logits=torch.tensor([[0.2, 0.8]]),  # Disagreement
            binary_probs=torch.tensor([[0.2, 0.8]]),
            type_logits=torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.05, 0.05]]),  # MET
            type_probs=torch.softmax(torch.tensor([[0.1, 0.1, 0.6, 0.1, 0.05, 0.05]]), dim=-1),
            level_logits=torch.tensor([[0.1, 0.1, 0.7, 0.1]]),  # TEL
            level_probs=torch.softmax(torch.tensor([[0.1, 0.1, 0.7, 0.1]]), dim=-1),
            intensity=torch.tensor([[0.72]]),
            resolvability=torch.tensor([[0.45]]),
            pairwise_matrix=torch.zeros(1, 4, 4),
            confidence=torch.tensor([[0.88]]),
        )

        analysis = model._outputs_to_analysis(outputs, {})

        assert analysis.has_disagreement is True
        assert analysis.disagreement_type is not None
        assert analysis.intensity == pytest.approx(0.72)
        assert analysis.resolvability == pytest.approx(0.45)
        assert analysis.confidence == pytest.approx(0.88)


class TestFactoryFunction:
    """Test per factory function get_disagreement_detector."""

    def test_singleton(self):
        """Test che restituisce singleton."""
        from merlt.disagreement.detector import get_disagreement_detector

        # Reset singleton
        import merlt.disagreement.detector as detector_module
        detector_module._detector_instance = None

        d1 = get_disagreement_detector()
        d2 = get_disagreement_detector()

        assert d1 is d2

    def test_force_new(self):
        """Test force_new crea nuova istanza."""
        from merlt.disagreement.detector import get_disagreement_detector

        d1 = get_disagreement_detector()
        d2 = get_disagreement_detector(force_new=True)

        assert d1 is not d2
