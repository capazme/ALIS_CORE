"""
Tests for PrinciplesExpert.

Tests cover:
- AC1: Identifies legal principles, legislative intent, constitutional values, doctrine
- AC2: LLM explains how principles guide interpretation, highlights tensions
- AC3: Output includes section header, principles, ratio legis, doctrine, confidence
- AC4: Handles doctrinal disagreement with multiple views
"""

import pytest
from typing import Any, Dict, List, Optional

from visualex.experts import (
    PrinciplesExpert,
    PrinciplesConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    IdentifiedPrinciple,
    PRINCIPLES_PROMPT_TEMPLATE,
    LEGAL_PRINCIPLES,
    CONSTITUTIONAL_PRINCIPLES,
)
from visualex.experts.base import BaseExpert


# =============================================================================
# Mock Classes
# =============================================================================


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(
        self,
        norm_chunks: List[Dict[str, Any]] = None,
        doctrine_chunks: List[Dict[str, Any]] = None,
    ):
        self.norm_chunks = norm_chunks or []
        self.doctrine_chunks = doctrine_chunks or []
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

        # Return based on source_type filter
        source_type = (filters or {}).get("source_type", "")
        if source_type == "doctrine":
            return self.doctrine_chunks[:limit]
        elif source_type == "norm":
            return self.norm_chunks[:limit]
        return []


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Test principles interpretation"):
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
    """Sample main norm chunks."""
    return [
        {
            "id": "chunk1",
            "urn": "urn:nir:stato:codice.civile:art_1453",
            "citation": "Art. 1453 c.c.",
            "text": "Nei contratti con prestazioni corrispettive...",
            "score": 0.85,
        },
    ]


@pytest.fixture
def sample_doctrine_chunks():
    """Sample doctrine chunks."""
    return [
        {
            "id": "doctrine1",
            "author": "Bianca",
            "citation": "Bianca, Diritto civile, III",
            "text": "La risoluzione per inadempimento tutela l'interesse del creditore alla corretta esecuzione del contratto. Il principio di buona fede impone di valutare l'importanza dell'inadempimento.",
            "score": 0.82,
        },
        {
            "id": "doctrine2",
            "author": "Gazzoni",
            "citation": "Gazzoni, Manuale di diritto privato",
            "text": "La ratio legis dell'art. 1453 è quella di liberare la parte fedele dal vincolo contrattuale quando l'altra parte non adempie. Si tratta di un rimedio sinallagmatico.",
            "score": 0.78,
        },
    ]


@pytest.fixture
def sample_doctrine_with_disagreement():
    """Sample doctrine with disagreement."""
    return [
        {
            "id": "doctrine1",
            "author": "Autore A",
            "text": "La dottrina prevalente ritiene che...",
            "score": 0.80,
        },
        {
            "id": "doctrine2",
            "author": "Autore B",
            "text": "Tuttavia, in senso contrario si è sostenuto che diversamente...",
            "score": 0.75,
        },
    ]


@pytest.fixture
def mock_retriever(sample_norm_chunks, sample_doctrine_chunks):
    """Create mock retriever with both norms and doctrine."""
    return MockRetriever(
        norm_chunks=sample_norm_chunks,
        doctrine_chunks=sample_doctrine_chunks,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService(
        response="La ratio legis dell'art. 1453 c.c. è quella di tutelare l'equilibrio "
        "contrattuale. Il principio di buona fede (art. 1375 c.c.) guida l'interpretazione "
        "richiedendo che l'inadempimento sia di non scarsa importanza."
    )


@pytest.fixture
def expert_with_mocks(mock_retriever, mock_llm):
    """Create PrinciplesExpert with all mocks."""
    return PrinciplesExpert(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )


@pytest.fixture
def expert_no_retriever(mock_llm):
    """Create PrinciplesExpert without retriever."""
    return PrinciplesExpert(
        llm_service=mock_llm,
    )


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return ExpertContext(
        query_text="Qual è la ratio legis dell'art. 1453 c.c. sulla risoluzione per inadempimento?",
        entities={
            "norm_references": ["urn:nir:stato:codice.civile:art_1453"],
            "legal_concepts": ["risoluzione", "inadempimento"],
        },
    )


@pytest.fixture
def context_with_principle():
    """Context mentioning a specific principle."""
    return ExpertContext(
        query_text="Come si applica il principio di buona fede nella risoluzione contrattuale?",
        entities={
            "legal_concepts": ["buona fede", "risoluzione"],
        },
    )


# =============================================================================
# Base Configuration Tests
# =============================================================================


class TestPrinciplesExpertConfiguration:
    """Tests for PrinciplesExpert configuration."""

    def test_expert_type(self):
        """Test that PrinciplesExpert has correct type."""
        expert = PrinciplesExpert()
        assert expert.expert_type == "principles"

    def test_section_header(self):
        """Test that PrinciplesExpert has correct Italian header."""
        expert = PrinciplesExpert()
        assert expert.section_header == "Interpretazione Teleologica"

    def test_description(self):
        """Test that PrinciplesExpert has description."""
        expert = PrinciplesExpert()
        assert "art. 12" in expert.description.lower()
        assert "teleologica" in expert.description.lower()


class TestPrinciplesConfig:
    """Tests for PrinciplesConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrinciplesConfig()

        assert config.max_doctrine_chunks == 8
        assert config.min_doctrine_score == 0.4
        assert config.include_constitutional is True
        assert config.enable_f5_feedback is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PrinciplesConfig(
            max_doctrine_chunks=5,
            enable_f5_feedback=False,
        )

        assert config.max_doctrine_chunks == 5
        assert config.enable_f5_feedback is False


# =============================================================================
# Principle Identification Tests (AC1)
# =============================================================================


class TestPrincipleIdentification:
    """Tests for legal principle identification (AC1)."""

    @pytest.mark.asyncio
    async def test_identifies_principle_from_query(self, expert_with_mocks, context_with_principle):
        """Test that principles are identified from query text."""
        response = await expert_with_mocks.analyze(context_with_principle)

        # Should identify buona_fede from query
        identified = response.metadata.get("identified_principles", [])
        principle_ids = [p["principle_id"] for p in identified]
        assert "buona_fede" in principle_ids

    @pytest.mark.asyncio
    async def test_identifies_multiple_principles(self, expert_with_mocks):
        """Test that multiple principles can be identified."""
        context = ExpertContext(
            query_text="Come interagiscono buona fede e autonomia privata nel contratto?",
            entities={"legal_concepts": ["buona fede", "autonomia privata"]},
        )

        response = await expert_with_mocks.analyze(context)

        identified = response.metadata.get("identified_principles", [])
        assert len(identified) >= 2

    @pytest.mark.asyncio
    async def test_principle_has_definition(self, expert_with_mocks, context_with_principle):
        """Test that identified principles include definition."""
        response = await expert_with_mocks.analyze(context_with_principle)

        identified = response.metadata.get("identified_principles", [])
        if identified:
            assert "definition" in identified[0]
            assert len(identified[0]["definition"]) > 10

    @pytest.mark.asyncio
    async def test_principle_has_articles(self, expert_with_mocks, context_with_principle):
        """Test that identified principles include related articles."""
        response = await expert_with_mocks.analyze(context_with_principle)

        identified = response.metadata.get("identified_principles", [])
        if identified:
            buona_fede = next((p for p in identified if p["principle_id"] == "buona_fede"), None)
            if buona_fede:
                assert len(buona_fede["articles"]) > 0
                assert any("1175" in art or "1375" in art for art in buona_fede["articles"])


# =============================================================================
# Doctrine Retrieval Tests (AC1)
# =============================================================================


class TestDoctrineRetrieval:
    """Tests for doctrine retrieval (AC1)."""

    @pytest.mark.asyncio
    async def test_retrieves_doctrine_chunks(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that doctrine chunks are retrieved."""
        await expert_with_mocks.analyze(sample_context)

        # Should have called retrieve with doctrine filter
        doctrine_calls = [
            c for c in mock_retriever.retrieve_calls
            if c.get("filters", {}).get("source_type") == "doctrine"
        ]
        assert len(doctrine_calls) > 0

    @pytest.mark.asyncio
    async def test_doctrine_in_legal_basis(self, expert_with_mocks, sample_context):
        """Test that doctrine appears in legal basis."""
        response = await expert_with_mocks.analyze(sample_context)

        doctrine_sources = [s for s in response.legal_basis if s.source_type == "doctrine"]
        assert len(doctrine_sources) > 0


# =============================================================================
# LLM Analysis Tests (AC2)
# =============================================================================


class TestLLMAnalysis:
    """Tests for LLM-based synthesis (AC2)."""

    @pytest.mark.asyncio
    async def test_calls_llm_service(self, expert_with_mocks, sample_context, mock_llm):
        """Test that LLM service is called."""
        await expert_with_mocks.analyze(sample_context)

        assert len(mock_llm.generate_calls) == 1

    @pytest.mark.asyncio
    async def test_prompt_includes_principles(self, expert_with_mocks, context_with_principle, mock_llm):
        """Test that prompt includes identified principles."""
        await expert_with_mocks.analyze(context_with_principle)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "PRINCIPI GIURIDICI" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_doctrine(self, expert_with_mocks, sample_context, mock_llm):
        """Test that prompt includes doctrine."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "DOTTRINA" in prompt

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, mock_retriever, sample_context):
        """Test fallback interpretation without LLM."""
        expert = PrinciplesExpert(retriever=mock_retriever)

        response = await expert.analyze(sample_context)

        assert response.interpretation
        assert response.tokens_used == 0


# =============================================================================
# Output Structure Tests (AC3)
# =============================================================================


class TestOutputStructure:
    """Tests for output structure (AC3)."""

    @pytest.mark.asyncio
    async def test_includes_section_header(self, expert_with_mocks, sample_context):
        """Test that response includes Italian section header."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.section_header == "Interpretazione Teleologica"

    @pytest.mark.asyncio
    async def test_includes_interpretation(self, expert_with_mocks, sample_context):
        """Test that response includes interpretation."""
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
    async def test_principles_in_legal_basis(self, expert_with_mocks, context_with_principle):
        """Test that identified principles appear in legal basis."""
        response = await expert_with_mocks.analyze(context_with_principle)

        principle_sources = [s for s in response.legal_basis if s.source_type == "principle"]
        assert len(principle_sources) > 0

    @pytest.mark.asyncio
    async def test_includes_confidence_score(self, expert_with_mocks, sample_context):
        """Test that response includes confidence score."""
        response = await expert_with_mocks.analyze(sample_context)

        assert 0.0 <= response.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_includes_reasoning_steps(self, expert_with_mocks, sample_context):
        """Test that response includes reasoning steps."""
        response = await expert_with_mocks.analyze(sample_context)

        assert len(response.reasoning_steps) > 0

    @pytest.mark.asyncio
    async def test_includes_execution_time(self, expert_with_mocks, sample_context):
        """Test that response includes execution time."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_metadata_includes_principles_count(self, expert_with_mocks, sample_context):
        """Test that metadata includes principles count."""
        response = await expert_with_mocks.analyze(sample_context)

        assert "principles_identified" in response.metadata
        assert isinstance(response.metadata["principles_identified"], int)

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, expert_with_mocks, sample_context):
        """Test that response serializes correctly."""
        response = await expert_with_mocks.analyze(sample_context)

        d = response.to_dict()

        assert d["expert_type"] == "principles"
        assert d["section_header"] == "Interpretazione Teleologica"
        assert "identified_principles" in d["metadata"]


# =============================================================================
# Doctrinal Disagreement Tests (AC4)
# =============================================================================


class TestDoctrinalDisagreement:
    """Tests for doctrinal disagreement handling (AC4)."""

    @pytest.mark.asyncio
    async def test_detects_disagreement(self, sample_norm_chunks, sample_doctrine_with_disagreement, mock_llm):
        """Test that doctrinal disagreement is detected."""
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            doctrine_chunks=sample_doctrine_with_disagreement,
        )
        expert = PrinciplesExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test query")
        response = await expert.analyze(context)

        assert response.metadata.get("has_doctrinal_disagreement") is True

    @pytest.mark.asyncio
    async def test_no_disagreement_when_single_source(self, mock_retriever, mock_llm, sample_context):
        """Test no disagreement with single doctrine source."""
        # Modify to have single doctrine
        mock_retriever.doctrine_chunks = mock_retriever.doctrine_chunks[:1]
        expert = PrinciplesExpert(retriever=mock_retriever, llm_service=mock_llm)

        response = await expert.analyze(sample_context)

        assert response.metadata.get("has_doctrinal_disagreement") is False

    @pytest.mark.asyncio
    async def test_disagreement_noted_in_limitations(self, sample_norm_chunks, sample_doctrine_with_disagreement, mock_llm):
        """Test that disagreement is noted in limitations."""
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            doctrine_chunks=sample_doctrine_with_disagreement,
        )
        expert = PrinciplesExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test query")
        response = await expert.analyze(context)

        assert "discordanti" in response.limitations.lower()


# =============================================================================
# F5 Feedback Hook Tests
# =============================================================================


class TestF5FeedbackHook:
    """Tests for F5 feedback hook integration."""

    @pytest.mark.asyncio
    async def test_includes_f5_feedback_hook(self, expert_with_mocks, sample_context):
        """Test that response includes F5 feedback hook."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook is not None
        assert isinstance(response.feedback_hook, FeedbackHook)

    @pytest.mark.asyncio
    async def test_f5_hook_has_correct_type(self, expert_with_mocks, sample_context):
        """Test that F5 hook has correct feedback type."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.feedback_type == "F5"
        assert response.feedback_hook.expert_type == "principles"

    @pytest.mark.asyncio
    async def test_f5_hook_can_be_disabled(self, mock_retriever, mock_llm, sample_context):
        """Test that F5 feedback can be disabled."""
        config = PrinciplesConfig(enable_f5_feedback=False)
        expert = PrinciplesExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
            config=config,
        )

        response = await expert.analyze(sample_context)

        assert response.feedback_hook is None


# =============================================================================
# No Doctrine Edge Case Tests
# =============================================================================


class TestNoDoctrine:
    """Tests for handling when no doctrine is available."""

    @pytest.mark.asyncio
    async def test_handles_no_doctrine(self, mock_llm):
        """Test handling when no doctrine is available."""
        retriever = MockRetriever(norm_chunks=[], doctrine_chunks=[])
        expert = PrinciplesExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Query senza principi specifici")
        response = await expert.analyze(context)

        assert response.expert_type == "principles"
        assert response.confidence <= 0.4
        assert "non disponibil" in response.limitations.lower() or "non" in response.limitations.lower()

    @pytest.mark.asyncio
    async def test_no_doctrine_metadata_flag(self, mock_llm):
        """Test that no doctrine case has metadata flag."""
        retriever = MockRetriever(norm_chunks=[], doctrine_chunks=[])
        expert = PrinciplesExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Query generica")
        response = await expert.analyze(context)

        assert response.metadata.get("no_doctrine") is True or response.metadata.get("principles_identified") == 0


# =============================================================================
# IdentifiedPrinciple Tests
# =============================================================================


class TestIdentifiedPrinciple:
    """Tests for IdentifiedPrinciple dataclass."""

    def test_creation(self):
        """Test IdentifiedPrinciple creation."""
        principle = IdentifiedPrinciple(
            principle_id="buona_fede",
            name="Buona fede",
            definition="Test definition",
            articles=["art. 1175 c.c."],
            relevance_score=0.9,
            source="query",
        )

        assert principle.principle_id == "buona_fede"
        assert principle.name == "Buona fede"
        assert principle.relevance_score == 0.9

    def test_to_dict(self):
        """Test IdentifiedPrinciple serialization."""
        principle = IdentifiedPrinciple(
            principle_id="test",
            name="Test",
            definition="Definition",
        )

        d = principle.to_dict()

        assert d["principle_id"] == "test"
        assert d["name"] == "Test"
        assert d["definition"] == "Definition"
        assert d["articles"] == []
        assert d["source"] == "inference"


# =============================================================================
# Legal Principles Taxonomy Tests
# =============================================================================


class TestLegalPrinciplesTaxonomy:
    """Tests for LEGAL_PRINCIPLES constant."""

    def test_taxonomy_exists(self):
        """Test that taxonomy is defined."""
        assert LEGAL_PRINCIPLES
        assert len(LEGAL_PRINCIPLES) > 5

    def test_taxonomy_structure(self):
        """Test that each principle has required fields."""
        for principle_id, data in LEGAL_PRINCIPLES.items():
            assert "name" in data
            assert "definition" in data
            assert "articles" in data
            assert isinstance(data["articles"], list)
            assert "category" in data  # New field

    def test_buona_fede_present(self):
        """Test that buona fede is in taxonomy."""
        assert "buona_fede" in LEGAL_PRINCIPLES
        assert "1175" in str(LEGAL_PRINCIPLES["buona_fede"]["articles"])

    def test_legal_principles_immutable(self):
        """Test that LEGAL_PRINCIPLES cannot be modified."""
        with pytest.raises(TypeError):
            LEGAL_PRINCIPLES["new_principle"] = {"name": "Test"}

    def test_constitutional_principles_exist(self):
        """Test that constitutional principles are defined."""
        assert CONSTITUTIONAL_PRINCIPLES
        assert len(CONSTITUTIONAL_PRINCIPLES) >= 3

    def test_constitutional_principles_structure(self):
        """Test that constitutional principles have required fields."""
        for principle_id, data in CONSTITUTIONAL_PRINCIPLES.items():
            assert "name" in data
            assert "definition" in data
            assert "articles" in data
            assert data["category"] == "constitutional"

    def test_constitutional_principles_immutable(self):
        """Test that CONSTITUTIONAL_PRINCIPLES cannot be modified."""
        with pytest.raises(TypeError):
            CONSTITUTIONAL_PRINCIPLES["new_principle"] = {"name": "Test"}


# =============================================================================
# Constitutional Principles Integration Tests (Code Review Fix)
# =============================================================================


class TestConstitutionalPrinciples:
    """Tests for constitutional principles integration."""

    @pytest.mark.asyncio
    async def test_identifies_constitutional_principle(self, mock_retriever, mock_llm):
        """Test that constitutional principles are identified when enabled."""
        config = PrinciplesConfig(include_constitutional=True)
        expert = PrinciplesExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
            config=config,
        )

        context = ExpertContext(
            query_text="Come si applica il principio di uguaglianza nel contratto?",
            entities={"legal_concepts": ["uguaglianza"]},
        )

        response = await expert.analyze(context)

        identified = response.metadata.get("identified_principles", [])
        principle_ids = [p["principle_id"] for p in identified]
        assert "uguaglianza" in principle_ids

    @pytest.mark.asyncio
    async def test_constitutional_disabled_by_config(self, mock_retriever, mock_llm):
        """Test that constitutional principles are NOT identified when disabled."""
        config = PrinciplesConfig(include_constitutional=False)
        expert = PrinciplesExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
            config=config,
        )

        context = ExpertContext(
            query_text="Come si applica il principio di uguaglianza nel contratto?",
            entities={"legal_concepts": ["uguaglianza"]},
        )

        response = await expert.analyze(context)

        identified = response.metadata.get("identified_principles", [])
        principle_ids = [p["principle_id"] for p in identified]
        # Should NOT contain constitutional principle when disabled
        assert "uguaglianza" not in principle_ids

    @pytest.mark.asyncio
    async def test_mixed_civil_and_constitutional(self, mock_retriever, mock_llm):
        """Test identifying both civil and constitutional principles together."""
        config = PrinciplesConfig(include_constitutional=True)
        expert = PrinciplesExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
            config=config,
        )

        context = ExpertContext(
            query_text="La buona fede e l'uguaglianza nel contratto",
            entities={"legal_concepts": ["buona fede", "uguaglianza"]},
        )

        response = await expert.analyze(context)

        identified = response.metadata.get("identified_principles", [])
        principle_ids = [p["principle_id"] for p in identified]

        # Should have both civil and constitutional
        assert "buona_fede" in principle_ids
        assert "uguaglianza" in principle_ids


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for prompt template."""

    def test_prompt_template_exists(self):
        """Test that prompt template is defined."""
        assert PRINCIPLES_PROMPT_TEMPLATE
        assert len(PRINCIPLES_PROMPT_TEMPLATE) > 100

    def test_prompt_template_placeholders(self):
        """Test that prompt template has required placeholders."""
        assert "{query}" in PRINCIPLES_PROMPT_TEMPLATE
        assert "{main_norm}" in PRINCIPLES_PROMPT_TEMPLATE
        assert "{principles}" in PRINCIPLES_PROMPT_TEMPLATE
        assert "{doctrine}" in PRINCIPLES_PROMPT_TEMPLATE

    def test_prompt_template_italian(self):
        """Test that prompt template is in Italian."""
        assert "italiano" in PRINCIPLES_PROMPT_TEMPLATE.lower() or "teleologica" in PRINCIPLES_PROMPT_TEMPLATE.lower()
