"""
Tests for GatingNetwork.

Tests cover:
- AC1: Combines outputs with query type, confidence, RLCF, user profile weights
- AC2: Merges without losing individual Expert identity, preserves traceability
- AC3: Low-confidence Expert contribution is minimized
- AC4: Structure includes all Expert outputs with weights, rationale logged for F7
"""

import pytest
from typing import Any, Dict, List, Optional

from visualex.experts import (
    GatingNetwork,
    GatingConfig,
    AggregatedResponse,
    ExpertContribution,
    AggregationMethod,
    DEFAULT_EXPERT_WEIGHTS,
    USER_PROFILE_MODIFIERS,
    GATING_SYNTHESIS_PROMPT,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Sintesi LLM test"):
        self.response = response
        self.generate_calls: List[Dict] = []

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        self.generate_calls.append({
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        return self.response


# =============================================================================
# Fixtures
# =============================================================================


def create_expert_response(
    expert_type: str,
    interpretation: str,
    confidence: float,
    section_header: str = "",
    legal_basis: Optional[List[LegalSource]] = None,
    reasoning_steps: Optional[List[ReasoningStep]] = None,
) -> ExpertResponse:
    """Helper to create ExpertResponse for tests."""
    return ExpertResponse(
        expert_type=expert_type,
        section_header=section_header or expert_type.title(),
        interpretation=interpretation,
        legal_basis=legal_basis or [],
        reasoning_steps=reasoning_steps or [],
        confidence=confidence,
        confidence_factors=ConfidenceFactors(),
        trace_id="test_trace",
    )


@pytest.fixture
def sample_expert_responses():
    """Sample responses from multiple experts."""
    return [
        create_expert_response(
            expert_type="literal",
            section_header="Interpretazione Letterale",
            interpretation="L'art. 1453 c.c. prevede che...",
            confidence=0.85,
            legal_basis=[
                LegalSource(
                    source_type="norm",
                    source_id="art_1453",
                    citation="Art. 1453 c.c.",
                    excerpt="Nei contratti con prestazioni corrispettive...",
                )
            ],
            reasoning_steps=[
                ReasoningStep(step_number=1, description="Analisi testuale")
            ],
        ),
        create_expert_response(
            expert_type="systemic",
            section_header="Interpretazione Sistematica",
            interpretation="Nel sistema del codice civile...",
            confidence=0.75,
            legal_basis=[
                LegalSource(
                    source_type="norm",
                    source_id="art_1453",
                    citation="Art. 1453 c.c.",
                ),
                LegalSource(
                    source_type="norm",
                    source_id="art_1455",
                    citation="Art. 1455 c.c.",
                ),
            ],
            reasoning_steps=[
                ReasoningStep(step_number=1, description="Collegamento norme")
            ],
        ),
        create_expert_response(
            expert_type="precedent",
            section_header="Giurisprudenza",
            interpretation="La Cassazione ha stabilito che...",
            confidence=0.70,
            legal_basis=[
                LegalSource(
                    source_type="jurisprudence",
                    source_id="cass_123",
                    citation="Cass. 123/2023",
                ),
            ],
        ),
    ]


@pytest.fixture
def sample_responses_with_conflict():
    """Expert responses with significant confidence divergence."""
    return [
        create_expert_response(
            expert_type="literal",
            interpretation="Interpretazione A",
            confidence=0.95,
            legal_basis=[
                LegalSource(source_type="norm", source_id="src_a", citation="A")
            ],
        ),
        create_expert_response(
            expert_type="systemic",
            interpretation="Interpretazione B diversa",
            confidence=0.40,
            legal_basis=[
                LegalSource(source_type="norm", source_id="src_b", citation="B")
            ],
        ),
    ]


@pytest.fixture
def mock_llm():
    """Mock LLM service."""
    return MockLLMService("Sintesi coerente delle interpretazioni degli Expert.")


@pytest.fixture
def gating_with_llm(mock_llm):
    """GatingNetwork with LLM service."""
    return GatingNetwork(llm_service=mock_llm)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestGatingConfiguration:
    """Tests for GatingNetwork configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GatingConfig()

        assert config.method == AggregationMethod.WEIGHTED_AVERAGE
        assert config.enable_f7_feedback is True
        assert config.confidence_divergence_threshold == 0.4
        assert len(config.default_weights) > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = GatingConfig(
            method=AggregationMethod.BEST_CONFIDENCE,
            enable_f7_feedback=False,
        )

        assert config.method == AggregationMethod.BEST_CONFIDENCE
        assert config.enable_f7_feedback is False

    def test_default_expert_weights_exist(self):
        """Test that default weights are defined."""
        assert "literal" in DEFAULT_EXPERT_WEIGHTS
        assert "systemic" in DEFAULT_EXPERT_WEIGHTS
        assert "principles" in DEFAULT_EXPERT_WEIGHTS
        assert "precedent" in DEFAULT_EXPERT_WEIGHTS

    def test_weights_sum_approximately_one(self):
        """Test that default weights sum to ~1."""
        total = sum(DEFAULT_EXPERT_WEIGHTS.values())
        assert 0.99 <= total <= 1.01

    def test_default_weights_immutable(self):
        """Test that DEFAULT_EXPERT_WEIGHTS is immutable."""
        with pytest.raises(TypeError):
            DEFAULT_EXPERT_WEIGHTS["new_expert"] = 0.5

    def test_user_profile_modifiers_exist(self):
        """Test that user profile modifiers are defined."""
        assert "analysis" in USER_PROFILE_MODIFIERS
        assert "quick" in USER_PROFILE_MODIFIERS
        assert "academic" in USER_PROFILE_MODIFIERS


# =============================================================================
# Weight Combination Tests (AC1)
# =============================================================================


class TestWeightCombination:
    """Tests for combining Expert outputs with weights (AC1)."""

    @pytest.mark.asyncio
    async def test_combines_with_default_weights(self, sample_expert_responses):
        """Test combination uses default weights."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.confidence > 0
        assert len(result.expert_contributions) == 3

    @pytest.mark.asyncio
    async def test_respects_custom_weights(self, sample_expert_responses):
        """Test combination respects custom weights."""
        gating = GatingNetwork()
        weights = {"literal": 0.8, "systemic": 0.1, "precedent": 0.1}

        result = await gating.aggregate(
            sample_expert_responses,
            weights=weights,
            trace_id="test",
        )

        # Literal should have highest weighted confidence
        assert result.confidence_breakdown.get("literal", 0) > result.confidence_breakdown.get("systemic", 0)

    @pytest.mark.asyncio
    async def test_normalizes_weights_for_present_experts(self):
        """Test weight normalization when not all experts present."""
        gating = GatingNetwork()
        responses = [
            create_expert_response("literal", "Test", 0.8),
            create_expert_response("systemic", "Test", 0.7),
        ]
        weights = {"literal": 0.5, "systemic": 0.3, "principles": 0.2}

        result = await gating.aggregate(responses, weights=weights, trace_id="test")

        # Weights should be normalized for present experts only
        used_weights = result.metadata.get("weights_used", {})
        total = sum(used_weights.values())
        assert 0.99 <= total <= 1.01

    @pytest.mark.asyncio
    async def test_user_profile_modifies_weights(self):
        """Test that user profile modifies weights."""
        gating = GatingNetwork()
        responses = [
            create_expert_response("literal", "Test", 0.8),
            create_expert_response("principles", "Test", 0.7),
        ]

        # Academic profile should boost principles
        result = await gating.aggregate(
            responses,
            trace_id="test",
            user_profile="academic",
        )

        used_weights = result.metadata.get("weights_used", {})
        # After academic modifier, principles should have higher relative weight
        # Default: literal=0.35, principles=0.20
        # Academic modifier: literal*0.9=0.315, principles*1.3=0.26
        # Normalized: literal~0.548, principles~0.452
        assert used_weights.get("principles", 0) > 0.4  # Significantly boosted

    @pytest.mark.asyncio
    async def test_unknown_user_profile_ignored(self):
        """Test that unknown user profile is ignored."""
        gating = GatingNetwork()
        responses = [create_expert_response("literal", "Test", 0.8)]

        result = await gating.aggregate(
            responses,
            trace_id="test",
            user_profile="unknown_profile",
        )

        # Should complete without error
        assert result.confidence > 0


# =============================================================================
# Traceability Tests (AC2)
# =============================================================================


class TestTraceability:
    """Tests for preserving Expert identity and traceability (AC2)."""

    @pytest.mark.asyncio
    async def test_preserves_expert_contributions(self, sample_expert_responses):
        """Test that individual Expert contributions are preserved."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        # Each expert should have a contribution entry
        for resp in sample_expert_responses:
            assert resp.expert_type in result.expert_contributions

    @pytest.mark.asyncio
    async def test_contribution_contains_original_interpretation(self, sample_expert_responses):
        """Test contributions contain original interpretations."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        for resp in sample_expert_responses:
            contrib = result.expert_contributions.get(resp.expert_type)
            assert contrib is not None
            assert contrib.interpretation == resp.interpretation

    @pytest.mark.asyncio
    async def test_combines_legal_sources_without_duplicates(self, sample_expert_responses):
        """Test legal sources are combined and deduplicated."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        # art_1453 is in both literal and systemic, should appear once
        source_ids = [lb.source_id for lb in result.combined_legal_basis]
        assert len(source_ids) == len(set(source_ids))  # No duplicates

    @pytest.mark.asyncio
    async def test_combines_reasoning_steps_with_expert_prefix(self, sample_expert_responses):
        """Test reasoning steps include expert attribution."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        # Reasoning steps should include [expert_type] prefix
        for rs in result.combined_reasoning:
            assert "[" in rs.description and "]" in rs.description


# =============================================================================
# Low Confidence Handling Tests (AC3)
# =============================================================================


class TestLowConfidenceHandling:
    """Tests for minimizing low-confidence Expert contribution (AC3)."""

    @pytest.mark.asyncio
    async def test_low_confidence_gets_lower_weighted_contribution(self):
        """Test that low-confidence experts have lower weighted contribution."""
        gating = GatingNetwork()
        responses = [
            create_expert_response("literal", "High confidence", 0.95),
            create_expert_response("systemic", "Low confidence", 0.30),
        ]
        weights = {"literal": 0.5, "systemic": 0.5}

        result = await gating.aggregate(responses, weights=weights, trace_id="test")

        literal_contrib = result.confidence_breakdown.get("literal", 0)
        systemic_contrib = result.confidence_breakdown.get("systemic", 0)

        # Even with equal weights, low confidence should result in lower contribution
        assert literal_contrib > systemic_contrib

    @pytest.mark.asyncio
    async def test_best_confidence_selects_highest(self):
        """Test best_confidence method selects highest confidence expert."""
        config = GatingConfig(method=AggregationMethod.BEST_CONFIDENCE)
        gating = GatingNetwork(config=config)

        responses = [
            create_expert_response("literal", "Low", 0.50),
            create_expert_response("systemic", "High", 0.90),
            create_expert_response("precedent", "Medium", 0.70),
        ]

        result = await gating.aggregate(responses, trace_id="test")

        # Systemic should be selected
        assert result.expert_contributions["systemic"].selected is True
        assert result.expert_contributions["literal"].selected is False


# =============================================================================
# Output Structure Tests (AC4)
# =============================================================================


class TestOutputStructure:
    """Tests for output structure with F7 feedback (AC4)."""

    @pytest.mark.asyncio
    async def test_includes_all_expert_outputs_with_weights(self, sample_expert_responses):
        """Test all experts have outputs with weights."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        for exp_type, contrib in result.expert_contributions.items():
            assert contrib.weight >= 0
            assert contrib.confidence >= 0
            assert contrib.interpretation

    @pytest.mark.asyncio
    async def test_includes_f7_feedback_hook(self, sample_expert_responses):
        """Test F7 feedback hook is included."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.feedback_hook is not None
        assert result.feedback_hook.feedback_type == "F7"
        assert result.feedback_hook.expert_type == "gating"

    @pytest.mark.asyncio
    async def test_f7_feedback_can_be_disabled(self, sample_expert_responses):
        """Test F7 feedback can be disabled."""
        config = GatingConfig(enable_f7_feedback=False)
        gating = GatingNetwork(config=config)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.feedback_hook is None

    @pytest.mark.asyncio
    async def test_metadata_contains_weights_used(self, sample_expert_responses):
        """Test metadata logs weights for RLCF."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert "weights_used" in result.metadata
        assert "expert_count" in result.metadata
        assert result.metadata["expert_count"] == 3


# =============================================================================
# Aggregation Method Tests
# =============================================================================


class TestAggregationMethods:
    """Tests for different aggregation methods."""

    @pytest.mark.asyncio
    async def test_weighted_average_method(self, sample_expert_responses):
        """Test weighted average aggregation."""
        config = GatingConfig(method=AggregationMethod.WEIGHTED_AVERAGE)
        gating = GatingNetwork(config=config)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.aggregation_method == "weighted_average"

    @pytest.mark.asyncio
    async def test_best_confidence_method(self, sample_expert_responses):
        """Test best confidence aggregation."""
        config = GatingConfig(method=AggregationMethod.BEST_CONFIDENCE)
        gating = GatingNetwork(config=config)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.aggregation_method == "best_confidence"
        # Only one expert should be selected
        selected = [c for c in result.expert_contributions.values() if c.selected]
        assert len(selected) == 1

    @pytest.mark.asyncio
    async def test_consensus_method(self, sample_expert_responses):
        """Test consensus aggregation."""
        config = GatingConfig(method=AggregationMethod.CONSENSUS)
        gating = GatingNetwork(config=config)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.aggregation_method == "consensus"

    @pytest.mark.asyncio
    async def test_ensemble_method(self, sample_expert_responses):
        """Test ensemble aggregation."""
        config = GatingConfig(method=AggregationMethod.ENSEMBLE)
        gating = GatingNetwork(config=config)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        assert result.aggregation_method == "ensemble"
        # Ensemble should preserve all interpretations in synthesis
        for resp in sample_expert_responses:
            header = resp.section_header or resp.expert_type.title()
            assert header in result.synthesis


# =============================================================================
# Conflict Detection Tests
# =============================================================================


class TestConflictDetection:
    """Tests for conflict detection between experts."""

    @pytest.mark.asyncio
    async def test_detects_confidence_divergence(self, sample_responses_with_conflict):
        """Test detection of significant confidence divergence."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_responses_with_conflict, trace_id="test")

        # Should detect divergence (0.95 vs 0.40)
        assert len(result.conflicts) > 0
        assert any("divergenza" in c.lower() for c in result.conflicts)

    @pytest.mark.asyncio
    async def test_detects_low_source_overlap(self, sample_responses_with_conflict):
        """Test detection of low source overlap."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_responses_with_conflict, trace_id="test")

        # Sources are completely different
        assert any("fonti" in c.lower() for c in result.conflicts)

    @pytest.mark.asyncio
    async def test_no_conflict_with_similar_confidence(self, sample_expert_responses):
        """Test no false positive with similar confidences."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        # Sample responses have similar confidences (0.85, 0.75, 0.70)
        confidence_conflicts = [c for c in result.conflicts if "divergenza" in c.lower()]
        assert len(confidence_conflicts) == 0


# =============================================================================
# LLM Synthesis Tests
# =============================================================================


class TestLLMSynthesis:
    """Tests for LLM-based synthesis."""

    @pytest.mark.asyncio
    async def test_calls_llm_service(self, gating_with_llm, sample_expert_responses, mock_llm):
        """Test that LLM service is called for synthesis."""
        await gating_with_llm.aggregate(sample_expert_responses, trace_id="test")

        assert len(mock_llm.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_uses_llm_response_as_synthesis(self, gating_with_llm, sample_expert_responses, mock_llm):
        """Test that LLM response becomes synthesis."""
        result = await gating_with_llm.aggregate(sample_expert_responses, trace_id="test")

        assert result.synthesis == mock_llm.response

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, sample_expert_responses):
        """Test fallback synthesis without LLM."""
        gating = GatingNetwork(llm_service=None)

        result = await gating.aggregate(sample_expert_responses, trace_id="test")

        # Should have synthesis without LLM note
        assert "senza AI" in result.synthesis or len(result.synthesis) > 0


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_responses_list(self):
        """Test handling of empty responses list."""
        gating = GatingNetwork()

        result = await gating.aggregate([], trace_id="test")

        assert result.confidence == 0.0
        assert result.metadata.get("expert_count") == 0

    @pytest.mark.asyncio
    async def test_single_expert_response(self):
        """Test handling of single expert response."""
        gating = GatingNetwork()
        responses = [create_expert_response("literal", "Solo Expert", 0.8)]

        result = await gating.aggregate(responses, trace_id="test")

        assert result.confidence > 0
        assert len(result.expert_contributions) == 1

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, sample_expert_responses):
        """Test AggregatedResponse serializes correctly."""
        gating = GatingNetwork()

        result = await gating.aggregate(sample_expert_responses, trace_id="test")
        d = result.to_dict()

        assert "synthesis" in d
        assert "expert_contributions" in d
        assert "confidence" in d
        assert "aggregation_method" in d


# =============================================================================
# Dataclass Tests
# =============================================================================


class TestExpertContribution:
    """Tests for ExpertContribution dataclass."""

    def test_creation(self):
        """Test ExpertContribution creation."""
        contrib = ExpertContribution(
            expert_type="literal",
            interpretation="Test interpretation",
            confidence=0.8,
            weight=0.5,
            weighted_confidence=0.4,
        )

        assert contrib.expert_type == "literal"
        assert contrib.weighted_confidence == 0.4

    def test_to_dict(self):
        """Test ExpertContribution serialization."""
        contrib = ExpertContribution(
            expert_type="literal",
            interpretation="Test",
            confidence=0.8,
            weight=0.5,
            weighted_confidence=0.4,
            selected=True,
        )

        d = contrib.to_dict()

        assert d["expert_type"] == "literal"
        assert d["selected"] is True


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for synthesis prompt template."""

    def test_prompt_template_exists(self):
        """Test that synthesis prompt template is defined."""
        assert GATING_SYNTHESIS_PROMPT
        assert len(GATING_SYNTHESIS_PROMPT) > 100

    def test_prompt_template_has_placeholder(self):
        """Test prompt template has required placeholder."""
        assert "{expert_sections}" in GATING_SYNTHESIS_PROMPT

    def test_prompt_template_italian(self):
        """Test prompt template is in Italian."""
        assert "italiano" in GATING_SYNTHESIS_PROMPT.lower() or "giurista" in GATING_SYNTHESIS_PROMPT.lower()
