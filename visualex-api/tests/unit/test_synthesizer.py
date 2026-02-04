"""
Tests for Synthesizer.

Tests cover:
- AC1: Generates unified response answering original question, integrates insights
- AC2: Profile-aware formatting (consulenza, ricerca, analisi, contributore)
- AC3: Disagreement noted explicitly, Devil's Advocate flagged
- AC4: Main answer, Expert Accordion, source links, confidence indicator, F7 feedback
"""

import pytest
from typing import Any, Dict, List, Optional

from visualex.experts import (
    Synthesizer,
    SynthesizerConfig,
    SynthesizedResponse,
    AccordionSection,
    UserProfile,
    SynthesisMode,
    SYNTHESIS_PROMPT_TEMPLATE,
    PROFILE_INSTRUCTIONS,
    AggregatedResponse,
    ExpertContribution,
    LegalSource,
    FeedbackHook,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Risposta sintetizzata dal LLM."):
        self.response = response
        self.generate_calls: List[Dict] = []

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2500,
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


def create_expert_contribution(
    expert_type: str,
    interpretation: str,
    confidence: float,
    weight: float,
) -> ExpertContribution:
    """Helper to create ExpertContribution."""
    return ExpertContribution(
        expert_type=expert_type,
        interpretation=interpretation,
        confidence=confidence,
        weight=weight,
        weighted_confidence=confidence * weight,
    )


def create_aggregated_response(
    expert_contributions: Dict[str, ExpertContribution],
    synthesis: str = "",
    confidence: float = 0.75,
    conflicts: Optional[List[str]] = None,
    legal_basis: Optional[List[LegalSource]] = None,
) -> AggregatedResponse:
    """Helper to create AggregatedResponse."""
    return AggregatedResponse(
        synthesis=synthesis,
        expert_contributions=expert_contributions,
        combined_legal_basis=legal_basis or [],
        confidence=confidence,
        conflicts=conflicts or [],
        aggregation_method="weighted_average",
        trace_id="test_trace",
    )


@pytest.fixture
def sample_contributions():
    """Sample expert contributions."""
    return {
        "literal": create_expert_contribution(
            expert_type="literal",
            interpretation="L'art. 1453 c.c. prevede letteralmente che...",
            confidence=0.85,
            weight=0.35,
        ),
        "systemic": create_expert_contribution(
            expert_type="systemic",
            interpretation="Nel sistema del codice civile, la risoluzione...",
            confidence=0.75,
            weight=0.30,
        ),
        "principles": create_expert_contribution(
            expert_type="principles",
            interpretation="La ratio legis della norma è tutelare...",
            confidence=0.70,
            weight=0.20,
        ),
    }


@pytest.fixture
def sample_aggregated(sample_contributions):
    """Sample aggregated response."""
    return create_aggregated_response(
        expert_contributions=sample_contributions,
        synthesis="Sintesi integrata delle interpretazioni.",
        confidence=0.78,
        legal_basis=[
            LegalSource(
                source_type="norm",
                source_id="urn:nir:stato:codice.civile:art_1453",
                citation="Art. 1453 c.c.",
            ),
            LegalSource(
                source_type="norm",
                source_id="urn:nir:stato:codice.civile:art_1455",
                citation="Art. 1455 c.c.",
            ),
        ],
    )


@pytest.fixture
def aggregated_with_conflict(sample_contributions):
    """Aggregated response with conflicts."""
    return create_aggregated_response(
        expert_contributions=sample_contributions,
        synthesis="Sintesi con conflitti.",
        confidence=0.65,
        conflicts=["Divergenza significativa: literal (0.85) vs principles (0.70)"],
    )


@pytest.fixture
def mock_llm():
    """Mock LLM service."""
    return MockLLMService("Risposta chiara e completa basata sulle interpretazioni.")


@pytest.fixture
def synthesizer_with_llm(mock_llm):
    """Synthesizer with LLM service."""
    return Synthesizer(llm_service=mock_llm)


# =============================================================================
# Configuration Tests
# =============================================================================


class TestSynthesizerConfiguration:
    """Tests for Synthesizer configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SynthesizerConfig()

        assert config.default_profile == UserProfile.RICERCA
        assert config.enable_f7_feedback is True
        assert config.disagreement_threshold == 0.4

    def test_custom_config(self):
        """Test custom configuration."""
        config = SynthesizerConfig(
            default_profile=UserProfile.ANALISI,
            enable_f7_feedback=False,
        )

        assert config.default_profile == UserProfile.ANALISI
        assert config.enable_f7_feedback is False

    def test_user_profile_enum(self):
        """Test UserProfile enum values."""
        assert UserProfile.CONSULENZA.value == "consulenza"
        assert UserProfile.RICERCA.value == "ricerca"
        assert UserProfile.ANALISI.value == "analisi"
        assert UserProfile.CONTRIBUTORE.value == "contributore"

    def test_profile_instructions_exist(self):
        """Test that profile instructions are defined."""
        for profile in UserProfile:
            assert profile in PROFILE_INSTRUCTIONS


# =============================================================================
# Unified Response Tests (AC1)
# =============================================================================


class TestUnifiedResponse:
    """Tests for generating unified response (AC1)."""

    @pytest.mark.asyncio
    async def test_generates_main_answer(self, synthesizer_with_llm, sample_aggregated):
        """Test that synthesizer generates main answer."""
        result = await synthesizer_with_llm.synthesize(
            query="Cos'è la risoluzione del contratto?",
            aggregated=sample_aggregated,
        )

        assert result.main_answer
        assert len(result.main_answer) > 10

    @pytest.mark.asyncio
    async def test_integrates_expert_insights(self, synthesizer_with_llm, sample_aggregated, mock_llm):
        """Test that LLM prompt includes expert insights."""
        await synthesizer_with_llm.synthesize(
            query="Test query",
            aggregated=sample_aggregated,
        )

        # Check prompt contains expert interpretations
        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "LITERAL" in prompt.upper()
        assert "SYSTEMIC" in prompt.upper()

    @pytest.mark.asyncio
    async def test_includes_confidence_indicator(self, synthesizer_with_llm, sample_aggregated):
        """Test that response includes confidence indicator."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert result.confidence_indicator in ["alta", "media", "bassa"]
        assert result.confidence_value > 0


# =============================================================================
# Profile-Aware Formatting Tests (AC2)
# =============================================================================


class TestProfileFormatting:
    """Tests for profile-aware formatting (AC2)."""

    @pytest.mark.asyncio
    async def test_consulenza_no_accordion(self, synthesizer_with_llm, sample_aggregated):
        """Test that consulenza profile has no accordion."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="consulenza",
        )

        assert result.user_profile == "consulenza"
        assert len(result.expert_accordion) == 0

    @pytest.mark.asyncio
    async def test_ricerca_has_collapsed_accordion(self, synthesizer_with_llm, sample_aggregated):
        """Test that ricerca profile has collapsed accordion."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="ricerca",
        )

        assert result.user_profile == "ricerca"
        assert len(result.expert_accordion) > 0
        # Sections should be collapsed
        for section in result.expert_accordion:
            assert section.is_expanded is False

    @pytest.mark.asyncio
    async def test_analisi_has_expanded_accordion(self, synthesizer_with_llm, sample_aggregated):
        """Test that analisi profile has expanded accordion."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="analisi",
        )

        assert result.user_profile == "analisi"
        # Sections should be expanded
        for section in result.expert_accordion:
            assert section.is_expanded is True

    @pytest.mark.asyncio
    async def test_contributore_has_feedback_hook(self, sample_aggregated):
        """Test that contributore profile has feedback hook."""
        synthesizer = Synthesizer()

        result = await synthesizer.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="contributore",
        )

        assert result.feedback_hook is not None
        assert result.feedback_hook.feedback_type == "F7"

    @pytest.mark.asyncio
    async def test_unknown_profile_uses_default(self, synthesizer_with_llm, sample_aggregated):
        """Test that unknown profile falls back to default."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="unknown_profile",
        )

        assert result.user_profile == "ricerca"  # Default


# =============================================================================
# Disagreement Handling Tests (AC3)
# =============================================================================


class TestDisagreementHandling:
    """Tests for disagreement handling (AC3)."""

    @pytest.mark.asyncio
    async def test_detects_conflict_from_aggregated(self, synthesizer_with_llm, aggregated_with_conflict):
        """Test that conflicts from gating are detected."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=aggregated_with_conflict,
        )

        assert result.has_disagreement is True
        assert result.synthesis_mode == "divergent"

    @pytest.mark.asyncio
    async def test_disagreement_note_generated(self, synthesizer_with_llm, aggregated_with_conflict):
        """Test that disagreement note is generated."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=aggregated_with_conflict,
        )

        assert result.disagreement_note
        assert "divergen" in result.disagreement_note.lower()

    @pytest.mark.asyncio
    async def test_devils_advocate_flagged(self, synthesizer_with_llm, aggregated_with_conflict):
        """Test that Devil's Advocate is flagged on disagreement."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=aggregated_with_conflict,
        )

        assert result.devils_advocate_flag is True

    @pytest.mark.asyncio
    async def test_no_disagreement_convergent_mode(self, synthesizer_with_llm, sample_aggregated):
        """Test convergent mode when no disagreement."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert result.has_disagreement is False
        assert result.synthesis_mode == "convergent"
        assert result.devils_advocate_flag is False


# =============================================================================
# Output Structure Tests (AC4)
# =============================================================================


class TestOutputStructure:
    """Tests for output structure (AC4)."""

    @pytest.mark.asyncio
    async def test_includes_main_answer(self, synthesizer_with_llm, sample_aggregated):
        """Test that main answer is always visible."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert result.main_answer
        assert isinstance(result.main_answer, str)

    @pytest.mark.asyncio
    async def test_includes_expert_accordion(self, synthesizer_with_llm, sample_aggregated):
        """Test that Expert accordion is included (for ricerca+)."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="ricerca",
        )

        assert len(result.expert_accordion) > 0
        for section in result.expert_accordion:
            assert isinstance(section, AccordionSection)
            assert section.header
            assert section.content

    @pytest.mark.asyncio
    async def test_includes_source_links(self, synthesizer_with_llm, sample_aggregated):
        """Test that source links are included."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert len(result.source_links) > 0
        for link in result.source_links:
            assert "source_id" in link
            assert "citation" in link

    @pytest.mark.asyncio
    async def test_includes_confidence_indicator(self, synthesizer_with_llm, sample_aggregated):
        """Test that confidence indicator is included."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert result.confidence_indicator in ["alta", "media", "bassa"]

    @pytest.mark.asyncio
    async def test_f7_feedback_for_eligible_profiles(self, sample_aggregated):
        """Test F7 feedback for analisi and contributore profiles."""
        synthesizer = Synthesizer()

        for profile in ["analisi", "contributore"]:
            result = await synthesizer.synthesize(
                query="Test",
                aggregated=sample_aggregated,
                user_profile=profile,
            )
            assert result.feedback_hook is not None

    @pytest.mark.asyncio
    async def test_no_f7_feedback_for_consulenza(self, sample_aggregated):
        """Test no F7 feedback for consulenza profile."""
        synthesizer = Synthesizer()

        result = await synthesizer.synthesize(
            query="Test",
            aggregated=sample_aggregated,
            user_profile="consulenza",
        )

        assert result.feedback_hook is None


# =============================================================================
# LLM Synthesis Tests
# =============================================================================


class TestLLMSynthesis:
    """Tests for LLM-based synthesis."""

    @pytest.mark.asyncio
    async def test_calls_llm_service(self, synthesizer_with_llm, sample_aggregated, mock_llm):
        """Test that LLM service is called."""
        await synthesizer_with_llm.synthesize(
            query="Test query",
            aggregated=sample_aggregated,
        )

        assert len(mock_llm.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_prompt_includes_query(self, synthesizer_with_llm, sample_aggregated, mock_llm):
        """Test that prompt includes original query."""
        await synthesizer_with_llm.synthesize(
            query="Cos'è la risoluzione?",
            aggregated=sample_aggregated,
        )

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "Cos'è la risoluzione?" in prompt

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, sample_aggregated):
        """Test fallback synthesis without LLM."""
        synthesizer = Synthesizer(llm_service=None)

        result = await synthesizer.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        assert result.main_answer
        # Check case-insensitive
        assert "senza sintesi ai" in result.main_answer.lower()


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for response serialization."""

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, synthesizer_with_llm, sample_aggregated):
        """Test SynthesizedResponse serializes correctly."""
        result = await synthesizer_with_llm.synthesize(
            query="Test",
            aggregated=sample_aggregated,
        )

        d = result.to_dict()

        assert "main_answer" in d
        assert "expert_accordion" in d
        assert "source_links" in d
        assert "confidence_indicator" in d
        assert "synthesis_mode" in d

    def test_accordion_section_to_dict(self):
        """Test AccordionSection serialization."""
        section = AccordionSection(
            expert_type="literal",
            header="Interpretazione Letterale",
            content="Test content",
            confidence=0.85,
            is_expanded=True,
        )

        d = section.to_dict()

        assert d["expert_type"] == "literal"
        assert d["is_expanded"] is True


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for synthesis prompt template."""

    def test_prompt_template_exists(self):
        """Test that synthesis prompt template is defined."""
        assert SYNTHESIS_PROMPT_TEMPLATE
        assert len(SYNTHESIS_PROMPT_TEMPLATE) > 100

    def test_prompt_template_has_placeholders(self):
        """Test prompt template has required placeholders."""
        assert "{query}" in SYNTHESIS_PROMPT_TEMPLATE
        assert "{expert_interpretations}" in SYNTHESIS_PROMPT_TEMPLATE
        assert "{sources}" in SYNTHESIS_PROMPT_TEMPLATE

    def test_prompt_template_italian(self):
        """Test prompt template is in Italian."""
        assert "italiano" in SYNTHESIS_PROMPT_TEMPLATE.lower() or "giurista" in SYNTHESIS_PROMPT_TEMPLATE.lower()
