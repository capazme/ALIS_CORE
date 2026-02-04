"""
Tests for LiteralExpert.

Tests cover:
- AC1: Retrieve norm text chunks via Bridge Table
- AC2: Analyze literal meaning using LLM with legal prompt
- AC3: Include definitions from Concetto/Definizione nodes
- AC4: Output includes correct structure (section header, text, confidence, time)
- AC5: Handle queries with no clear norm reference (low confidence)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, Dict, List, Optional

from visualex.experts import (
    LiteralExpert,
    LiteralConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    LITERAL_PROMPT_TEMPLATE,
)
from visualex.experts.base import BaseExpert, ExpertConfig


# =============================================================================
# Mock Classes
# =============================================================================


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, norm_chunks: List[Dict[str, Any]] = None, definition_chunks: List[Dict[str, Any]] = None):
        self.norm_chunks = norm_chunks or []
        self.definition_chunks = definition_chunks or []
        self.retrieve_calls: List[Dict] = []

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        self.retrieve_calls.append({
            "query": query,
            "filters": filters,
            "limit": limit,
        })

        source_type = filters.get("source_type") if filters else None
        if source_type == "norm":
            return self.norm_chunks[:limit]
        elif source_type == "definition":
            return self.definition_chunks[:limit]
        return self.norm_chunks[:limit]


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Test interpretation response"):
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


@pytest.fixture
def sample_norm_chunks():
    """Sample norm chunks for testing."""
    return [
        {
            "id": "chunk1",
            "urn": "urn:nir:stato:codice.civile:art_1453",
            "citation": "Art. 1453 c.c.",
            "title": "Risolubilità del contratto per inadempimento",
            "text": "Nei contratti con prestazioni corrispettive, quando uno dei contraenti non adempie le sue obbligazioni, l'altro può a sua scelta chiedere l'adempimento o la risoluzione del contratto...",
            "score": 0.85,
        },
        {
            "id": "chunk2",
            "urn": "urn:nir:stato:codice.civile:art_1454",
            "citation": "Art. 1454 c.c.",
            "title": "Diffida ad adempiere",
            "text": "Alla parte inadempiente l'altra può intimare per iscritto di adempiere in un congruo termine, con dichiarazione che, decorso inutilmente detto termine, il contratto s'intenderà senz'altro risoluto.",
            "score": 0.78,
        },
    ]


@pytest.fixture
def sample_definition_chunks():
    """Sample definition chunks for testing."""
    return [
        {
            "id": "def1",
            "urn": "urn:nir:stato:codice.civile:art_1321",
            "citation": "Art. 1321 c.c.",
            "text": "Il contratto è l'accordo di due o più parti per costituire, regolare o estinguere tra loro un rapporto giuridico patrimoniale.",
            "score": 0.90,
        },
    ]


@pytest.fixture
def mock_retriever(sample_norm_chunks, sample_definition_chunks):
    """Create mock retriever with sample data."""
    return MockRetriever(
        norm_chunks=sample_norm_chunks,
        definition_chunks=sample_definition_chunks,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService(
        response="L'art. 1453 c.c. disciplina la risoluzione del contratto per inadempimento. "
        "In base al significato letterale della norma, il contraente non inadempiente può scegliere "
        "tra chiedere l'adempimento o la risoluzione del contratto."
    )


@pytest.fixture
def expert_with_mocks(mock_retriever, mock_llm):
    """Create LiteralExpert with mock dependencies."""
    return LiteralExpert(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )


@pytest.fixture
def expert_no_llm(mock_retriever):
    """Create LiteralExpert without LLM service."""
    return LiteralExpert(retriever=mock_retriever)


@pytest.fixture
def expert_no_retriever():
    """Create LiteralExpert without retriever."""
    return LiteralExpert()


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return ExpertContext(
        query_text="Cos'è la risoluzione del contratto?",
        entities={
            "norm_references": ["urn:nir:stato:codice.civile:art_1453"],
            "legal_concepts": ["contratto", "risoluzione"],
        },
    )


# =============================================================================
# Base Class Tests
# =============================================================================


class TestBaseExpertConfiguration:
    """Tests for BaseExpert configuration."""

    def test_literal_expert_type(self):
        """Test that LiteralExpert has correct type."""
        expert = LiteralExpert()
        assert expert.expert_type == "literal"

    def test_literal_section_header(self):
        """Test that LiteralExpert has correct Italian header."""
        expert = LiteralExpert()
        assert expert.section_header == "Interpretazione Letterale"

    def test_literal_description(self):
        """Test that LiteralExpert has description."""
        expert = LiteralExpert()
        assert "art. 12" in expert.description.lower()

    def test_expert_requires_type(self):
        """Test that BaseExpert requires expert_type."""

        class BadExpert(BaseExpert):
            expert_type = ""
            section_header = "Test"
            description = "Test"

            async def analyze(self, context):
                pass

        with pytest.raises(ValueError, match="expert_type"):
            BadExpert()

    def test_expert_requires_section_header(self):
        """Test that BaseExpert requires section_header."""

        class BadExpert(BaseExpert):
            expert_type = "test"
            section_header = ""
            description = "Test"

            async def analyze(self, context):
                pass

        with pytest.raises(ValueError, match="section_header"):
            BadExpert()


class TestLiteralConfig:
    """Tests for LiteralConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LiteralConfig()

        assert config.temperature == 0.3
        assert config.literal_temperature == 0.2
        assert config.chunk_limit == 10
        assert config.include_definitions is True
        assert config.min_norm_chunks == 1

    def test_custom_config(self):
        """Test custom configuration."""
        config = LiteralConfig(
            chunk_limit=5,
            literal_temperature=0.1,
            include_definitions=False,
        )

        assert config.chunk_limit == 5
        assert config.literal_temperature == 0.1
        assert config.include_definitions is False


# =============================================================================
# Chunk Retrieval Tests (AC1)
# =============================================================================


class TestChunkRetrieval:
    """Tests for chunk retrieval via Bridge Table (AC1)."""

    @pytest.mark.asyncio
    async def test_retrieves_norm_chunks(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that expert retrieves norm chunks."""
        await expert_with_mocks.analyze(sample_context)

        # Should have called retriever for norms
        norm_calls = [c for c in mock_retriever.retrieve_calls if c["filters"].get("source_type") == "norm"]
        assert len(norm_calls) > 0

    @pytest.mark.asyncio
    async def test_retrieves_definition_chunks(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that expert retrieves definition chunks (AC3)."""
        await expert_with_mocks.analyze(sample_context)

        # Should have called retriever for definitions
        def_calls = [c for c in mock_retriever.retrieve_calls if c["filters"].get("source_type") == "definition"]
        assert len(def_calls) > 0

    @pytest.mark.asyncio
    async def test_uses_expert_affinity_filter(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that retrieval uses expert_affinity filter."""
        await expert_with_mocks.analyze(sample_context)

        # All calls should include expert_affinity=literal
        for call in mock_retriever.retrieve_calls:
            assert call["filters"].get("expert_affinity") == "literal"

    @pytest.mark.asyncio
    async def test_includes_urn_filters(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that URN filters are passed when available."""
        await expert_with_mocks.analyze(sample_context)

        # Norm retrieval should include URN filter
        norm_calls = [c for c in mock_retriever.retrieve_calls if c["filters"].get("source_type") == "norm"]
        if norm_calls:
            assert "urns" in norm_calls[0]["filters"]

    @pytest.mark.asyncio
    async def test_handles_no_retriever(self, expert_no_retriever, sample_context):
        """Test handling when no retriever is available."""
        response = await expert_no_retriever.analyze(sample_context)

        # Should return low confidence
        assert response.confidence < 0.3
        assert response.is_low_confidence()


# =============================================================================
# LLM Analysis Tests (AC2)
# =============================================================================


class TestLLMAnalysis:
    """Tests for LLM-based analysis (AC2)."""

    @pytest.mark.asyncio
    async def test_calls_llm_service(self, expert_with_mocks, sample_context, mock_llm):
        """Test that LLM service is called."""
        await expert_with_mocks.analyze(sample_context)

        assert len(mock_llm.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_llm_prompt_includes_query(self, expert_with_mocks, sample_context, mock_llm):
        """Test that prompt includes user query."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert sample_context.query_text in prompt

    @pytest.mark.asyncio
    async def test_llm_prompt_includes_norms(self, expert_with_mocks, sample_context, mock_llm, sample_norm_chunks):
        """Test that prompt includes norm texts."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        # Should include citation from norm chunk
        assert "Art. 1453 c.c." in prompt

    @pytest.mark.asyncio
    async def test_uses_literal_temperature(self, expert_with_mocks, sample_context, mock_llm):
        """Test that LLM uses literal-specific temperature."""
        await expert_with_mocks.analyze(sample_context)

        temp = mock_llm.generate_calls[0]["temperature"]
        assert temp == 0.2  # LiteralConfig default

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, expert_no_llm, sample_context):
        """Test fallback interpretation without LLM."""
        response = await expert_no_llm.analyze(sample_context)

        # Should still produce response
        assert response.interpretation
        assert response.tokens_used == 0


# =============================================================================
# Output Structure Tests (AC4)
# =============================================================================


class TestOutputStructure:
    """Tests for output structure (AC4)."""

    @pytest.mark.asyncio
    async def test_includes_section_header(self, expert_with_mocks, sample_context):
        """Test that response includes Italian section header."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.section_header == "Interpretazione Letterale"

    @pytest.mark.asyncio
    async def test_includes_interpretation(self, expert_with_mocks, sample_context):
        """Test that response includes interpretation text."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.interpretation
        assert len(response.interpretation) > 10

    @pytest.mark.asyncio
    async def test_includes_legal_basis(self, expert_with_mocks, sample_context):
        """Test that response includes legal sources."""
        response = await expert_with_mocks.analyze(sample_context)

        assert len(response.legal_basis) > 0
        assert all(isinstance(s, LegalSource) for s in response.legal_basis)

    @pytest.mark.asyncio
    async def test_legal_basis_has_urn(self, expert_with_mocks, sample_context):
        """Test that legal sources include URN."""
        response = await expert_with_mocks.analyze(sample_context)

        for source in response.legal_basis:
            assert source.source_id
            # Norm sources should have URN format
            if source.source_type == "norm":
                assert "urn:" in source.source_id

    @pytest.mark.asyncio
    async def test_includes_confidence_score(self, expert_with_mocks, sample_context):
        """Test that response includes confidence score."""
        response = await expert_with_mocks.analyze(sample_context)

        assert 0.0 <= response.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_includes_confidence_factors(self, expert_with_mocks, sample_context):
        """Test that response includes confidence breakdown."""
        response = await expert_with_mocks.analyze(sample_context)

        factors = response.confidence_factors
        assert isinstance(factors, ConfidenceFactors)
        assert 0.0 <= factors.norm_clarity <= 1.0
        assert 0.0 <= factors.source_availability <= 1.0

    @pytest.mark.asyncio
    async def test_includes_execution_time(self, expert_with_mocks, sample_context):
        """Test that response includes processing time."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_includes_reasoning_steps(self, expert_with_mocks, sample_context):
        """Test that response includes reasoning steps."""
        response = await expert_with_mocks.analyze(sample_context)

        assert len(response.reasoning_steps) > 0
        assert all(isinstance(s, ReasoningStep) for s in response.reasoning_steps)

    @pytest.mark.asyncio
    async def test_reasoning_steps_numbered(self, expert_with_mocks, sample_context):
        """Test that reasoning steps are numbered."""
        response = await expert_with_mocks.analyze(sample_context)

        for i, step in enumerate(response.reasoning_steps, 1):
            assert step.step_number == i

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, expert_with_mocks, sample_context):
        """Test that response can be serialized to dict."""
        response = await expert_with_mocks.analyze(sample_context)

        d = response.to_dict()

        assert d["expert_type"] == "literal"
        assert d["section_header"] == "Interpretazione Letterale"
        assert "interpretation" in d
        assert "confidence" in d
        assert "execution_time_ms" in d


# =============================================================================
# Low Confidence Handling Tests (AC5)
# =============================================================================


class TestLowConfidenceHandling:
    """Tests for handling queries with no clear norm reference (AC5)."""

    @pytest.mark.asyncio
    async def test_low_confidence_no_norms(self, sample_context):
        """Test low confidence when no norms found."""
        empty_retriever = MockRetriever(norm_chunks=[], definition_chunks=[])
        expert = LiteralExpert(retriever=empty_retriever)

        response = await expert.analyze(sample_context)

        assert response.confidence < 0.3
        assert response.is_low_confidence()

    @pytest.mark.asyncio
    async def test_suggests_clarification(self, sample_context):
        """Test that low confidence suggests clarification."""
        empty_retriever = MockRetriever(norm_chunks=[], definition_chunks=[])
        expert = LiteralExpert(retriever=empty_retriever)

        response = await expert.analyze(sample_context)

        # Should suggest specifying article
        assert response.suggestions
        assert "articolo" in response.suggestions.lower() or "specificare" in response.suggestions.lower()

    @pytest.mark.asyncio
    async def test_identifies_limitations(self, sample_context):
        """Test that limitations are identified."""
        empty_retriever = MockRetriever(norm_chunks=[], definition_chunks=[])
        expert = LiteralExpert(retriever=empty_retriever)

        response = await expert.analyze(sample_context)

        assert response.limitations

    @pytest.mark.asyncio
    async def test_no_norm_refs_lower_confidence(self, expert_with_mocks):
        """Test that missing norm references results in lower confidence."""
        context_no_refs = ExpertContext(
            query_text="Una domanda generica sul diritto",
            entities={},  # No norm references
        )

        response = await expert_with_mocks.analyze(context_no_refs)

        # Should have lower confidence due to missing specific references
        assert response.confidence_factors.contextual_ambiguity > 0.3


# =============================================================================
# Definition Integration Tests (AC3)
# =============================================================================


class TestDefinitionIntegration:
    """Tests for definition integration (AC3)."""

    @pytest.mark.asyncio
    async def test_includes_definitions_in_sources(self, expert_with_mocks, sample_context):
        """Test that definitions are included in legal sources."""
        response = await expert_with_mocks.analyze(sample_context)

        definition_sources = [s for s in response.legal_basis if s.source_type == "definition"]
        assert len(definition_sources) > 0

    @pytest.mark.asyncio
    async def test_definition_coverage_affects_confidence(self, mock_llm, sample_context):
        """Test that definition coverage affects confidence."""
        # Expert with no definitions
        no_def_retriever = MockRetriever(
            norm_chunks=[{
                "id": "chunk1",
                "urn": "urn:test",
                "citation": "Art. 1 test",
                "text": "Test norm",
                "score": 0.8,
            }],
            definition_chunks=[],
        )
        expert_no_def = LiteralExpert(retriever=no_def_retriever, llm_service=mock_llm)

        # Expert with definitions
        with_def_retriever = MockRetriever(
            norm_chunks=[{
                "id": "chunk1",
                "urn": "urn:test",
                "citation": "Art. 1 test",
                "text": "Test norm",
                "score": 0.8,
            }],
            definition_chunks=[{
                "id": "def1",
                "urn": "urn:test:def",
                "citation": "Definition",
                "text": "Test definition",
                "score": 0.9,
            }],
        )
        expert_with_def = LiteralExpert(retriever=with_def_retriever, llm_service=mock_llm)

        response_no_def = await expert_no_def.analyze(sample_context)
        response_with_def = await expert_with_def.analyze(sample_context)

        # Definition coverage should be higher with definitions
        assert (
            response_with_def.confidence_factors.definition_coverage >=
            response_no_def.confidence_factors.definition_coverage
        )

    @pytest.mark.asyncio
    async def test_can_disable_definitions(self, sample_norm_chunks, mock_llm, sample_context):
        """Test that definition retrieval can be disabled."""
        retriever = MockRetriever(norm_chunks=sample_norm_chunks)
        config = LiteralConfig(include_definitions=False)
        expert = LiteralExpert(retriever=retriever, llm_service=mock_llm, config=config)

        await expert.analyze(sample_context)

        # Should not have called retriever for definitions
        def_calls = [c for c in retriever.retrieve_calls if c["filters"].get("source_type") == "definition"]
        assert len(def_calls) == 0


# =============================================================================
# Confidence Computation Tests
# =============================================================================


class TestConfidenceComputation:
    """Tests for confidence score computation."""

    def test_confidence_factors_compute_overall(self):
        """Test overall confidence computation."""
        factors = ConfidenceFactors(
            norm_clarity=0.8,
            source_availability=0.9,
            contextual_ambiguity=0.2,
            definition_coverage=0.7,
        )

        overall = factors.compute_overall()

        assert 0.0 <= overall <= 1.0
        # High positive factors, low ambiguity should give high confidence
        assert overall > 0.5

    def test_high_ambiguity_reduces_confidence(self):
        """Test that high ambiguity reduces confidence."""
        low_ambiguity = ConfidenceFactors(
            norm_clarity=0.8,
            source_availability=0.8,
            contextual_ambiguity=0.1,
            definition_coverage=0.8,
        )
        high_ambiguity = ConfidenceFactors(
            norm_clarity=0.8,
            source_availability=0.8,
            contextual_ambiguity=0.9,
            definition_coverage=0.8,
        )

        assert low_ambiguity.compute_overall() > high_ambiguity.compute_overall()

    @pytest.mark.asyncio
    async def test_high_score_chunks_increase_confidence(self, mock_llm, sample_context):
        """Test that high-score chunks increase confidence."""
        high_score_retriever = MockRetriever(
            norm_chunks=[{
                "id": "chunk1",
                "urn": "urn:test",
                "citation": "Art. 1",
                "text": "Test",
                "score": 0.95,
            }],
        )
        low_score_retriever = MockRetriever(
            norm_chunks=[{
                "id": "chunk1",
                "urn": "urn:test",
                "citation": "Art. 1",
                "text": "Test",
                "score": 0.4,
            }],
        )

        expert_high = LiteralExpert(retriever=high_score_retriever, llm_service=mock_llm)
        expert_low = LiteralExpert(retriever=low_score_retriever, llm_service=mock_llm)

        response_high = await expert_high.analyze(sample_context)
        response_low = await expert_low.analyze(sample_context)

        # Norm clarity should be higher with high-score chunks
        assert (
            response_high.confidence_factors.norm_clarity >
            response_low.confidence_factors.norm_clarity
        )


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_empty_query(self, expert_with_mocks):
        """Test handling empty query."""
        context = ExpertContext(query_text="")

        response = await expert_with_mocks.analyze(context)

        # Should still return valid response
        assert response.expert_type == "literal"
        assert isinstance(response.confidence, float)

    @pytest.mark.asyncio
    async def test_very_long_query(self, expert_with_mocks):
        """Test handling very long query."""
        context = ExpertContext(query_text="Domanda " * 1000)

        response = await expert_with_mocks.analyze(context)

        assert response.expert_type == "literal"

    @pytest.mark.asyncio
    async def test_uses_pre_retrieved_chunks(self, expert_no_retriever, mock_llm, sample_norm_chunks):
        """Test that expert can use pre-retrieved chunks from context."""
        expert = LiteralExpert(llm_service=mock_llm)  # No retriever
        context = ExpertContext(
            query_text="Test query",
            retrieved_chunks=sample_norm_chunks,  # Pre-retrieved
        )

        response = await expert.analyze(context)

        # Should use pre-retrieved chunks
        assert len(response.legal_basis) > 0

    @pytest.mark.asyncio
    async def test_metadata_includes_chunk_counts(self, expert_with_mocks, sample_context):
        """Test that metadata includes chunk counts."""
        response = await expert_with_mocks.analyze(sample_context)

        assert "norm_chunks_count" in response.metadata
        assert "definition_chunks_count" in response.metadata

    @pytest.mark.asyncio
    async def test_trace_id_propagated(self, expert_with_mocks):
        """Test that trace_id is propagated to response."""
        context = ExpertContext(
            query_text="Test",
            trace_id="test_trace_123",
        )

        response = await expert_with_mocks.analyze(context)

        assert response.trace_id == "test_trace_123"


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for prompt template."""

    def test_prompt_template_exists(self):
        """Test that prompt template is defined."""
        assert LITERAL_PROMPT_TEMPLATE
        assert len(LITERAL_PROMPT_TEMPLATE) > 100

    def test_prompt_template_placeholders(self):
        """Test that prompt template has required placeholders."""
        assert "{query}" in LITERAL_PROMPT_TEMPLATE
        assert "{norm_texts}" in LITERAL_PROMPT_TEMPLATE
        assert "{definitions}" in LITERAL_PROMPT_TEMPLATE

    def test_prompt_template_italian(self):
        """Test that prompt template is in Italian."""
        # Should contain Italian keywords
        assert "italiano" in LITERAL_PROMPT_TEMPLATE.lower() or "norma" in LITERAL_PROMPT_TEMPLATE.lower()


# =============================================================================
# F3 Feedback Hook Tests (RLCF Integration)
# =============================================================================


class TestF3FeedbackHook:
    """Tests for F3 feedback hook integration."""

    @pytest.fixture
    def expert_with_mocks(self, mock_retriever, mock_llm):
        """Create LiteralExpert with mock dependencies."""
        return LiteralExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
        )

    @pytest.fixture
    def mock_retriever(self, sample_norm_chunks, sample_definition_chunks):
        """Create mock retriever with sample data."""
        return MockRetriever(
            norm_chunks=sample_norm_chunks,
            definition_chunks=sample_definition_chunks,
        )

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM service."""
        return MockLLMService(response="Test interpretation")

    @pytest.fixture
    def sample_norm_chunks(self):
        """Sample norm chunks for testing."""
        return [{
            "id": "chunk1",
            "urn": "urn:nir:stato:codice.civile:art_1453",
            "citation": "Art. 1453 c.c.",
            "text": "Test norm text",
            "score": 0.85,
        }]

    @pytest.fixture
    def sample_definition_chunks(self):
        """Sample definition chunks for testing."""
        return []

    @pytest.fixture
    def sample_context(self):
        """Sample context for testing."""
        return ExpertContext(query_text="Test query")

    @pytest.mark.asyncio
    async def test_includes_f3_feedback_hook(self, expert_with_mocks, sample_context):
        """Test that response includes F3 feedback hook."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook is not None
        assert isinstance(response.feedback_hook, FeedbackHook)

    @pytest.mark.asyncio
    async def test_f3_hook_has_correct_type(self, expert_with_mocks, sample_context):
        """Test that F3 hook has correct feedback type."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.feedback_type == "F3"
        assert response.feedback_hook.expert_type == "literal"

    @pytest.mark.asyncio
    async def test_f3_hook_links_to_response(self, expert_with_mocks, sample_context):
        """Test that F3 hook links to response via trace_id."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.response_id == response.trace_id

    @pytest.mark.asyncio
    async def test_f3_hook_enabled_by_default(self, expert_with_mocks, sample_context):
        """Test that F3 hook is enabled by default."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.enabled is True

    @pytest.mark.asyncio
    async def test_f3_hook_can_be_disabled(self, sample_norm_chunks, mock_llm, sample_context):
        """Test that F3 feedback can be disabled via config."""
        retriever = MockRetriever(norm_chunks=sample_norm_chunks)
        config = LiteralConfig(enable_f3_feedback=False)
        expert = LiteralExpert(retriever=retriever, llm_service=mock_llm, config=config)

        response = await expert.analyze(sample_context)

        assert response.feedback_hook is None

    @pytest.mark.asyncio
    async def test_f3_hook_serialization(self, expert_with_mocks, sample_context):
        """Test that F3 hook serializes correctly in to_dict."""
        response = await expert_with_mocks.analyze(sample_context)

        d = response.to_dict()

        assert "feedback_hook" in d
        assert d["feedback_hook"]["feedback_type"] == "F3"
        assert d["feedback_hook"]["expert_type"] == "literal"
        assert d["feedback_hook"]["enabled"] is True
