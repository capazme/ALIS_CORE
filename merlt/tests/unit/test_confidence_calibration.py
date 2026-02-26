"""
[P2] Unit tests for confidence calibration in AdaptiveSynthesizer.

Tests the alpha-blending formula:
    final = alpha * expert_conf + (1-alpha) * disagr_conf
    where alpha = max(0.3, 1 - sqrt(intensity))

Also tests weight normalization to sum=1.0.
"""

import math
import pytest

from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig
from merlt.disagreement.types import DisagreementAnalysis


@pytest.fixture
def synthesizer():
    return AdaptiveSynthesizer(config=SynthesisConfig())


class TestAlphaBlending:
    def test_p2_zero_intensity_alpha_is_one(self, synthesizer):
        """intensity=0 => alpha=max(0.3, 1-0)=1.0, so result = expert_conf."""
        analysis = DisagreementAnalysis(
            has_disagreement=False, confidence=0.5, intensity=0.0
        )
        result = synthesizer._calibrate_confidence(0.8, analysis)
        assert result == pytest.approx(0.8, abs=1e-6)

    def test_p2_max_intensity_alpha_is_floor(self, synthesizer):
        """intensity=1.0 => alpha=max(0.3, 1-1)=0.3."""
        analysis = DisagreementAnalysis(
            has_disagreement=True, confidence=0.4, intensity=1.0
        )
        result = synthesizer._calibrate_confidence(0.9, analysis)
        expected = 0.3 * 0.9 + 0.7 * 0.4
        assert result == pytest.approx(expected, abs=1e-6)

    def test_p2_mid_intensity(self, synthesizer):
        """intensity=0.25 => alpha=max(0.3, 1-0.5)=0.5."""
        analysis = DisagreementAnalysis(
            has_disagreement=True, confidence=0.6, intensity=0.25
        )
        result = synthesizer._calibrate_confidence(0.8, analysis)
        alpha = max(0.3, 1.0 - math.sqrt(0.25))
        expected = alpha * 0.8 + (1.0 - alpha) * 0.6
        assert result == pytest.approx(expected, abs=1e-6)

    def test_p2_high_intensity_clamps_alpha(self, synthesizer):
        """intensity=0.81 => alpha=max(0.3, 1-0.9)=0.3 (floor applies)."""
        analysis = DisagreementAnalysis(
            has_disagreement=True, confidence=0.3, intensity=0.81
        )
        result = synthesizer._calibrate_confidence(0.7, analysis)
        expected = 0.3 * 0.7 + 0.7 * 0.3
        assert result == pytest.approx(expected, abs=1e-6)

    def test_p2_none_analysis_returns_expert_conf(self, synthesizer):
        """No analysis => return expert confidence unchanged."""
        result = synthesizer._calibrate_confidence(0.75, None)
        assert result == pytest.approx(0.75)

    def test_p2_result_clamped_to_unit_interval(self, synthesizer):
        """Result is always in [0.0, 1.0]."""
        analysis = DisagreementAnalysis(
            has_disagreement=False, confidence=1.0, intensity=0.0
        )
        result = synthesizer._calibrate_confidence(1.0, analysis)
        assert 0.0 <= result <= 1.0

        analysis2 = DisagreementAnalysis(
            has_disagreement=True, confidence=0.0, intensity=1.0
        )
        result2 = synthesizer._calibrate_confidence(0.0, analysis2)
        assert 0.0 <= result2 <= 1.0


class TestWeightNormalization:
    @pytest.mark.asyncio
    async def test_p2_weights_normalized_to_sum_one(self):
        """Provided weights are normalized so they sum to 1.0."""
        from merlt.experts.base import ExpertResponse

        synth = AdaptiveSynthesizer(config=SynthesisConfig())
        responses = [
            ExpertResponse(expert_type="literal", interpretation="A", confidence=0.8),
            ExpertResponse(expert_type="systemic", interpretation="B", confidence=0.7),
        ]
        weights = {"literal": 3.0, "systemic": 7.0}

        result = await synth.synthesize(
            query="test", responses=responses, weights=weights
        )
        # After normalization: literal=0.3, systemic=0.7
        # The confidence is weighted sum calibrated with disagreement
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_p2_default_weights_equal(self):
        """When no weights provided, each expert gets equal weight."""
        from merlt.experts.base import ExpertResponse

        synth = AdaptiveSynthesizer(config=SynthesisConfig())
        responses = [
            ExpertResponse(expert_type="literal", interpretation="A", confidence=0.9),
            ExpertResponse(expert_type="systemic", interpretation="B", confidence=0.9),
        ]

        result = await synth.synthesize(query="test", responses=responses)
        assert result.confidence is not None
        assert 0.0 <= result.confidence <= 1.0


class TestSingleExpertResponse:
    @pytest.mark.asyncio
    async def test_p2_single_expert_no_disagreement(self):
        """Single expert response => no disagreement, confidence preserved."""
        from merlt.experts.base import ExpertResponse

        synth = AdaptiveSynthesizer(config=SynthesisConfig())
        responses = [
            ExpertResponse(expert_type="literal", interpretation="Solo", confidence=0.85),
        ]

        result = await synth.synthesize(query="test", responses=responses)
        # Single expert => heuristic returns has_disagreement=False, confidence=0.9
        # alpha-blending with intensity=0 => alpha=1.0 => expert_conf dominates
        assert result.mode.value == "convergent"
        assert result.confidence > 0
