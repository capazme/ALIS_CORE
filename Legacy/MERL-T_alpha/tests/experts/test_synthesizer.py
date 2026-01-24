"""
Test AdaptiveSynthesizer
=========================

Test per il sintetizzatore adattivo con modalita' convergent/divergent.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime

from merlt.experts.synthesizer import (
    AdaptiveSynthesizer,
    SynthesisMode,
    SynthesisConfig,
    SynthesisResult,
)
from merlt.experts.base import ExpertResponse, LegalSource, ReasoningStep
from merlt.disagreement.types import (
    DisagreementType,
    DisagreementLevel,
    DisagreementAnalysis,
    ExpertPairConflict,
)


class TestSynthesisConfig:
    """Test per SynthesisConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        config = SynthesisConfig()

        assert config.mode == SynthesisMode.AUTO
        assert config.convergent_threshold == 0.5
        assert config.resolvability_weight == 0.3
        assert config.include_disagreement_explanation is True
        assert config.max_alternatives == 3

    def test_custom_config(self):
        """Test configurazione custom."""
        config = SynthesisConfig(
            mode=SynthesisMode.DIVERGENT,
            convergent_threshold=0.3,
            max_alternatives=5,
        )

        assert config.mode == SynthesisMode.DIVERGENT
        assert config.convergent_threshold == 0.3
        assert config.max_alternatives == 5


class TestSynthesisMode:
    """Test per SynthesisMode enum."""

    def test_modes_defined(self):
        """Verifica che tutti i modi siano definiti."""
        assert SynthesisMode.CONVERGENT.value == "convergent"
        assert SynthesisMode.DIVERGENT.value == "divergent"
        assert SynthesisMode.AUTO.value == "auto"


class TestAdaptiveSynthesizer:
    """Test per AdaptiveSynthesizer."""

    @pytest.fixture
    def sample_responses(self):
        """Crea risposte expert di esempio."""
        return [
            ExpertResponse(
                expert_type="literal",
                interpretation="Secondo l'art. 1453 c.c., il contraente puo' risolvere il contratto...",
                confidence=0.85,
                legal_basis=[
                    LegalSource(
                        source_id="urn:cc:art1453",
                        source_type="articolo",
                        citation="Art. 1453 c.c.",
                        excerpt="La parte adempiente puo'..."
                    )
                ],
            ),
            ExpertResponse(
                expert_type="principles",
                interpretation="Il principio di buona fede contrattuale impone che...",
                confidence=0.78,
                legal_basis=[
                    LegalSource(
                        source_id="urn:cc:art1375",
                        source_type="articolo",
                        citation="Art. 1375 c.c.",
                        excerpt="Il contratto deve essere eseguito..."
                    )
                ],
            ),
        ]

    @pytest.fixture
    def high_disagreement_responses(self):
        """Risposte con alto disagreement."""
        return [
            ExpertResponse(
                expert_type="literal",
                interpretation="No, il recesso non e' consentito secondo il testo dell'art. 1372 c.c.",
                confidence=0.9,
                legal_basis=[],
            ),
            ExpertResponse(
                expert_type="principles",
                interpretation="Si, il recesso e' giustificato dal principio di buona fede.",
                confidence=0.85,
                legal_basis=[],
            ),
        ]

    def test_initialization(self):
        """Test inizializzazione."""
        synthesizer = AdaptiveSynthesizer()

        assert synthesizer.config.mode == SynthesisMode.AUTO
        assert synthesizer.ai_service is None

    def test_initialization_with_config(self):
        """Test inizializzazione con config custom."""
        config = SynthesisConfig(mode=SynthesisMode.CONVERGENT)
        synthesizer = AdaptiveSynthesizer(config=config)

        assert synthesizer.config.mode == SynthesisMode.CONVERGENT

    @pytest.mark.asyncio
    async def test_synthesize_convergent(self, sample_responses):
        """Test sintesi convergente."""
        config = SynthesisConfig(mode=SynthesisMode.CONVERGENT)
        synthesizer = AdaptiveSynthesizer(config=config)

        result = await synthesizer.synthesize(
            query="Quando si risolve un contratto?",
            responses=sample_responses,
        )

        assert isinstance(result, SynthesisResult)
        assert result.mode == SynthesisMode.CONVERGENT
        assert len(result.synthesis) > 0
        assert len(result.expert_contributions) == 2

    @pytest.mark.asyncio
    async def test_synthesize_divergent(self, high_disagreement_responses):
        """Test sintesi divergente."""
        config = SynthesisConfig(mode=SynthesisMode.DIVERGENT)
        synthesizer = AdaptiveSynthesizer(config=config)

        result = await synthesizer.synthesize(
            query="Il venditore puo' recedere?",
            responses=high_disagreement_responses,
        )

        assert result.mode == SynthesisMode.DIVERGENT
        assert len(result.alternatives) > 0
        assert "literal" in result.expert_contributions
        assert "principles" in result.expert_contributions

    @pytest.mark.asyncio
    async def test_synthesize_auto_mode(self, sample_responses):
        """Test modalita' auto."""
        synthesizer = AdaptiveSynthesizer()

        result = await synthesizer.synthesize(
            query="Test query",
            responses=sample_responses,
        )

        # Auto mode decide in base al disagreement
        assert result.mode in [SynthesisMode.CONVERGENT, SynthesisMode.DIVERGENT]

    def test_determine_mode_no_disagreement(self):
        """Test che mode = convergent senza disagreement."""
        synthesizer = AdaptiveSynthesizer()

        analysis = DisagreementAnalysis(
            has_disagreement=False,
            intensity=0.0,
        )

        mode = synthesizer._determine_mode(analysis)
        assert mode == SynthesisMode.CONVERGENT

    def test_determine_mode_high_disagreement(self):
        """Test che mode = divergent con alto disagreement."""
        synthesizer = AdaptiveSynthesizer()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.ANTINOMY,
            intensity=0.9,
            resolvability=0.2,
        )

        mode = synthesizer._determine_mode(analysis)
        assert mode == SynthesisMode.DIVERGENT

    def test_determine_mode_low_disagreement(self):
        """Test che mode = convergent con basso disagreement."""
        synthesizer = AdaptiveSynthesizer()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            intensity=0.2,
            resolvability=0.8,
        )

        mode = synthesizer._determine_mode(analysis)
        assert mode == SynthesisMode.CONVERGENT

    def test_heuristic_disagreement_low(self, sample_responses):
        """Test euristica con basso disagreement."""
        synthesizer = AdaptiveSynthesizer()

        analysis = synthesizer._heuristic_disagreement(sample_responses)

        assert isinstance(analysis, DisagreementAnalysis)
        # Confidenze simili -> basso disagreement
        assert analysis.intensity < 0.5

    def test_heuristic_disagreement_high(self):
        """Test euristica con alto disagreement (confidenze diverse)."""
        synthesizer = AdaptiveSynthesizer()

        responses = [
            ExpertResponse(expert_type="literal", interpretation="A", confidence=0.95),
            ExpertResponse(expert_type="principles", interpretation="B", confidence=0.3),
        ]

        analysis = synthesizer._heuristic_disagreement(responses)

        # Alta varianza nelle confidenze
        assert analysis.has_disagreement is True

    def test_get_reasoning_type(self):
        """Test mapping reasoning type."""
        synthesizer = AdaptiveSynthesizer()

        assert "letterale" in synthesizer._get_reasoning_type("literal").lower()
        assert "sistematica" in synthesizer._get_reasoning_type("systemic").lower()
        assert "teleologica" in synthesizer._get_reasoning_type("principles").lower()
        assert "giurisprudenza" in synthesizer._get_reasoning_type("precedent").lower()

    def test_generate_disagreement_explanation(self, sample_responses):
        """Test generazione spiegazione disagreement."""
        synthesizer = AdaptiveSynthesizer()

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="literal",
                    expert_b="principles",
                    conflict_score=0.8,
                )
            ],
        )

        explanation = synthesizer._generate_disagreement_explanation(analysis, sample_responses)

        assert "METODOLOGICO" in explanation or "Divergenza Metodologica" in explanation
        assert "0.7" in explanation or "70%" in explanation  # Intensity
        assert "literal" in explanation
        assert "principles" in explanation

    def test_simple_convergent_synthesis(self, sample_responses):
        """Test sintesi convergente semplice."""
        synthesizer = AdaptiveSynthesizer()
        weights = {"literal": 0.6, "principles": 0.4}

        synthesis = synthesizer._simple_convergent_synthesis(
            sample_responses, weights, None
        )

        assert "Sintesi" in synthesis
        assert "literal" in synthesis.lower()
        assert "principles" in synthesis.lower()

    def test_simple_divergent_synthesis(self):
        """Test sintesi divergente semplice."""
        synthesizer = AdaptiveSynthesizer()

        alternatives = [
            {
                "expert": "literal",
                "position": "Posizione letterale...",
                "confidence": 0.9,
                "legal_basis": ["Art. 1453 c.c."],
                "reasoning_type": "Interpretazione letterale",
            },
            {
                "expert": "principles",
                "position": "Posizione teleologica...",
                "confidence": 0.8,
                "legal_basis": ["Art. 1375 c.c."],
                "reasoning_type": "Interpretazione teleologica",
            },
        ]

        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            intensity=0.7,
        )

        synthesis = synthesizer._simple_divergent_synthesis(alternatives, analysis, None)

        assert "Alternative" in synthesis
        assert "Posizione 1" in synthesis
        assert "Posizione 2" in synthesis
        assert "literal" in synthesis.lower()


class TestSynthesisResult:
    """Test per SynthesisResult dataclass."""

    def test_creation(self):
        """Test creazione."""
        result = SynthesisResult(
            synthesis="Testo sintesi",
            mode=SynthesisMode.CONVERGENT,
            confidence=0.85,
        )

        assert result.synthesis == "Testo sintesi"
        assert result.mode == SynthesisMode.CONVERGENT
        assert result.confidence == 0.85
        assert result.alternatives == []

    def test_to_dict(self):
        """Test serializzazione."""
        result = SynthesisResult(
            synthesis="Test",
            mode=SynthesisMode.DIVERGENT,
            alternatives=[{"expert": "literal", "position": "..."}],
            confidence=0.7,
            explanation="Spiegazione del disagreement",
        )

        d = result.to_dict()

        assert d["synthesis"] == "Test"
        assert d["mode"] == "divergent"
        assert len(d["alternatives"]) == 1
        assert d["confidence"] == 0.7
        assert d["explanation"] is not None


class TestSynthesizerWithMockDetector:
    """Test con detector mockato."""

    @pytest.fixture
    def mock_detector(self):
        """Crea mock del detector."""
        detector = AsyncMock()
        detector.detect = AsyncMock(return_value=DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
        ))
        return detector

    @pytest.mark.asyncio
    async def test_synthesize_with_detector(self, mock_detector):
        """Test sintesi con detector reale."""
        synthesizer = AdaptiveSynthesizer(detector=mock_detector)

        responses = [
            ExpertResponse(expert_type="literal", interpretation="A", confidence=0.8),
            ExpertResponse(expert_type="principles", interpretation="B", confidence=0.75),
        ]

        result = await synthesizer.synthesize(
            query="Test",
            responses=responses,
        )

        # Detector should have been called
        mock_detector.detect.assert_called_once()

        # With intensity=0.7, resolvability=0.4, should be divergent
        assert result.mode == SynthesisMode.DIVERGENT
        assert result.disagreement_analysis is not None
        assert result.disagreement_analysis.disagreement_type == DisagreementType.METHODOLOGICAL
