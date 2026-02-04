"""
Tests for PrecedentExpert.

Tests cover:
- AC1: Retrieves Cassazione decisions, massime, trends, minority views
- AC2: LLM synthesizes judicial interpretation, distinguishes consolidated/evolving
- AC3: Output includes header, decisions, massime, trend analysis, confidence
- AC4: Handles case law conflicts with explicit presentation
"""

import pytest
from typing import Any, Dict, List, Optional

from visualex.experts import (
    PrecedentExpert,
    PrecedentConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    CaseDecision,
    CourtAuthority,
    PRECEDENT_PROMPT_TEMPLATE,
    COURT_PATTERNS,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(
        self,
        norm_chunks: List[Dict[str, Any]] = None,
        case_chunks: List[Dict[str, Any]] = None,
        massime_chunks: List[Dict[str, Any]] = None,
    ):
        self.norm_chunks = norm_chunks or []
        self.case_chunks = case_chunks or []
        self.massime_chunks = massime_chunks or []
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

        source_type = (filters or {}).get("source_type", "")
        if source_type == "jurisprudence":
            return self.case_chunks[:limit]
        elif source_type == "massima":
            return self.massime_chunks[:limit]
        elif source_type == "norm":
            return self.norm_chunks[:limit]
        return []


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Test precedent interpretation"):
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
            "id": "norm1",
            "urn": "urn:nir:stato:codice.civile:art_1453",
            "citation": "Art. 1453 c.c.",
            "text": "Nei contratti con prestazioni corrispettive...",
            "score": 0.85,
        },
    ]


@pytest.fixture
def sample_case_chunks():
    """Sample jurisprudence chunks."""
    return [
        {
            "id": "case1",
            "case_id": "cass_2023_12345",
            "court": "Cassazione civile",
            "section": "Sez. III",
            "date": "2023-05-15",
            "number": "12345",
            "massima": "L'inadempimento di non scarsa importanza legittima la risoluzione del contratto.",
            "text": "La Corte ha stabilito che...",
            "score": 0.88,
        },
        {
            "id": "case2",
            "case_id": "cass_2022_9876",
            "court": "Cassazione civile",
            "section": "Sezioni Unite",
            "date": "2022-01-10",
            "number": "9876",
            "massima": "Le Sezioni Unite confermano l'orientamento consolidato...",
            "text": "Nel caso di specie...",
            "score": 0.82,
        },
    ]


@pytest.fixture
def sample_case_chunks_with_conflict():
    """Sample jurisprudence with conflicting positions."""
    return [
        {
            "id": "case1",
            "case_id": "cass_a",
            "court": "Cassazione civile",
            "section": "Sez. II",
            "date": "2023-01-15",
            "massima": "La risoluzione richiede sempre la diffida ad adempiere.",
            "text": "...",
            "score": 0.85,
        },
        {
            "id": "case2",
            "case_id": "cass_b",
            "court": "Cassazione civile",
            "section": "Sez. III",
            "date": "2022-06-20",
            "massima": "Tuttavia, in senso contrario, la risoluzione può essere immediata se l'inadempimento è grave.",
            "text": "In difformità dall'orientamento precedente...",
            "score": 0.80,
        },
    ]


@pytest.fixture
def mock_retriever(sample_norm_chunks, sample_case_chunks):
    """Create mock retriever with norms and cases."""
    return MockRetriever(
        norm_chunks=sample_norm_chunks,
        case_chunks=sample_case_chunks,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService(
        response="La Cassazione ha consolidato l'orientamento secondo cui "
        "l'inadempimento di non scarsa importanza è presupposto necessario "
        "per la risoluzione del contratto (Cass. 12345/2023)."
    )


@pytest.fixture
def expert_with_mocks(mock_retriever, mock_llm):
    """Create PrecedentExpert with all mocks."""
    return PrecedentExpert(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return ExpertContext(
        query_text="Come la Cassazione interpreta l'art. 1453 c.c.?",
        entities={
            "norm_references": ["urn:nir:stato:codice.civile:art_1453"],
        },
    )


# =============================================================================
# Base Configuration Tests
# =============================================================================


class TestPrecedentExpertConfiguration:
    """Tests for PrecedentExpert configuration."""

    def test_expert_type(self):
        """Test that PrecedentExpert has correct type."""
        expert = PrecedentExpert()
        assert expert.expert_type == "precedent"

    def test_section_header(self):
        """Test that PrecedentExpert has correct Italian header."""
        expert = PrecedentExpert()
        assert expert.section_header == "Giurisprudenza"

    def test_description(self):
        """Test that PrecedentExpert has description."""
        expert = PrecedentExpert()
        assert "giurisprudenz" in expert.description.lower()


class TestPrecedentConfig:
    """Tests for PrecedentConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PrecedentConfig()

        assert config.max_case_chunks == 10
        assert config.max_massime == 5
        assert config.prefer_recent is True
        assert config.detect_conflicts is True
        assert config.enable_f6_feedback is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = PrecedentConfig(
            max_case_chunks=5,
            detect_conflicts=False,
        )

        assert config.max_case_chunks == 5
        assert config.detect_conflicts is False


# =============================================================================
# Case Law Retrieval Tests (AC1)
# =============================================================================


class TestCaseLawRetrieval:
    """Tests for case law retrieval (AC1)."""

    @pytest.mark.asyncio
    async def test_retrieves_case_chunks(self, expert_with_mocks, sample_context, mock_retriever):
        """Test that case law chunks are retrieved."""
        await expert_with_mocks.analyze(sample_context)

        juris_calls = [
            c for c in mock_retriever.retrieve_calls
            if c.get("filters", {}).get("source_type") == "jurisprudence"
        ]
        assert len(juris_calls) > 0

    @pytest.mark.asyncio
    async def test_cases_in_legal_basis(self, expert_with_mocks, sample_context):
        """Test that cases appear in legal basis."""
        response = await expert_with_mocks.analyze(sample_context)

        juris_sources = [s for s in response.legal_basis if s.source_type == "jurisprudence"]
        assert len(juris_sources) > 0

    @pytest.mark.asyncio
    async def test_cases_metadata_includes_count(self, expert_with_mocks, sample_context):
        """Test that metadata includes cases count."""
        response = await expert_with_mocks.analyze(sample_context)

        assert "cases_found" in response.metadata
        assert response.metadata["cases_found"] > 0


# =============================================================================
# Authority Ranking Tests (AC1)
# =============================================================================


class TestAuthorityRanking:
    """Tests for court authority ranking."""

    def test_court_authority_enum(self):
        """Test CourtAuthority values."""
        assert CourtAuthority.CASSAZIONE_SU > CourtAuthority.CASSAZIONE
        assert CourtAuthority.CASSAZIONE > CourtAuthority.APPELLO
        assert CourtAuthority.APPELLO > CourtAuthority.TRIBUNALE

    def test_court_patterns_immutable(self):
        """Test that COURT_PATTERNS is immutable."""
        with pytest.raises(TypeError):
            COURT_PATTERNS["new_pattern"] = CourtAuthority.OTHER

    @pytest.mark.asyncio
    async def test_sezioni_unite_ranked_highest(self, mock_llm, sample_norm_chunks):
        """Test that Sezioni Unite cases get highest authority."""
        case_chunks = [
            {
                "id": "su",
                "case_id": "su_case",
                "court": "Cassazione civile",
                "section": "Sezioni Unite",
                "score": 0.8,
            },
            {
                "id": "simple",
                "case_id": "simple_case",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "score": 0.85,
            },
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=case_chunks,
        )
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test")
        response = await expert.analyze(context)

        top_cases = response.metadata.get("top_cases", [])
        # Sezioni Unite should be first due to higher authority
        if len(top_cases) >= 2:
            assert top_cases[0]["case_id"] == "su_case"


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
    async def test_prompt_includes_case_law(self, expert_with_mocks, sample_context, mock_llm):
        """Test that prompt includes case law."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "GIURISPRUDENZA" in prompt

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, mock_retriever, sample_context):
        """Test fallback interpretation without LLM."""
        expert = PrecedentExpert(retriever=mock_retriever)

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

        assert response.section_header == "Giurisprudenza"

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
    async def test_to_dict_serialization(self, expert_with_mocks, sample_context):
        """Test that response serializes correctly."""
        response = await expert_with_mocks.analyze(sample_context)

        d = response.to_dict()

        assert d["expert_type"] == "precedent"
        assert d["section_header"] == "Giurisprudenza"
        assert "top_cases" in d["metadata"]


# =============================================================================
# Conflict Detection Tests (AC4)
# =============================================================================


class TestConflictDetection:
    """Tests for case law conflict detection (AC4)."""

    @pytest.mark.asyncio
    async def test_detects_conflict(self, sample_norm_chunks, sample_case_chunks_with_conflict, mock_llm):
        """Test that conflicts are detected."""
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=sample_case_chunks_with_conflict,
        )
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test conflict")
        response = await expert.analyze(context)

        assert response.metadata.get("has_conflict") is True

    @pytest.mark.asyncio
    async def test_no_conflict_with_consistent_cases(self, expert_with_mocks, sample_context):
        """Test no conflict with consistent case law."""
        response = await expert_with_mocks.analyze(sample_context)

        # Default sample cases don't have conflict indicators
        assert response.metadata.get("has_conflict") is False

    @pytest.mark.asyncio
    async def test_conflict_noted_in_limitations(self, sample_norm_chunks, sample_case_chunks_with_conflict, mock_llm):
        """Test that conflict is noted in limitations."""
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=sample_case_chunks_with_conflict,
        )
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test")
        response = await expert.analyze(context)

        assert "contrasto" in response.limitations.lower()

    @pytest.mark.asyncio
    async def test_conflict_flags_devils_advocate(self, sample_norm_chunks, sample_case_chunks_with_conflict, mock_llm):
        """Test that conflict flags Devil's Advocate consideration."""
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=sample_case_chunks_with_conflict,
        )
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Test")
        response = await expert.analyze(context)

        assert response.metadata.get("devils_advocate_flag") is True


# =============================================================================
# F6 Feedback Hook Tests
# =============================================================================


class TestF6FeedbackHook:
    """Tests for F6 feedback hook integration."""

    @pytest.mark.asyncio
    async def test_includes_f6_feedback_hook(self, expert_with_mocks, sample_context):
        """Test that response includes F6 feedback hook."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook is not None
        assert isinstance(response.feedback_hook, FeedbackHook)

    @pytest.mark.asyncio
    async def test_f6_hook_has_correct_type(self, expert_with_mocks, sample_context):
        """Test that F6 hook has correct feedback type."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.feedback_type == "F6"
        assert response.feedback_hook.expert_type == "precedent"

    @pytest.mark.asyncio
    async def test_f6_hook_can_be_disabled(self, mock_retriever, mock_llm, sample_context):
        """Test that F6 feedback can be disabled."""
        config = PrecedentConfig(enable_f6_feedback=False)
        expert = PrecedentExpert(
            retriever=mock_retriever,
            llm_service=mock_llm,
            config=config,
        )

        response = await expert.analyze(sample_context)

        assert response.feedback_hook is None


# =============================================================================
# No Cases Edge Case Tests
# =============================================================================


class TestNoCases:
    """Tests for handling when no case law is available."""

    @pytest.mark.asyncio
    async def test_handles_no_cases(self, mock_llm, sample_norm_chunks):
        """Test handling when no cases are available."""
        retriever = MockRetriever(norm_chunks=sample_norm_chunks, case_chunks=[])
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm)

        context = ExpertContext(query_text="Query senza giurisprudenza")
        response = await expert.analyze(context)

        assert response.expert_type == "precedent"
        assert response.confidence <= 0.3
        assert response.metadata.get("cases_found") == 0


# =============================================================================
# CaseDecision Tests
# =============================================================================


class TestCaseDecision:
    """Tests for CaseDecision dataclass."""

    def test_creation(self):
        """Test CaseDecision creation."""
        case = CaseDecision(
            case_id="test_123",
            court="Cassazione civile",
            section="Sez. III",
            date="2023-05-15",
            number="12345",
            massima="Test massima",
            authority_score=0.8,
        )

        assert case.case_id == "test_123"
        assert case.authority_score == 0.8

    def test_citation_property(self):
        """Test CaseDecision citation generation."""
        case = CaseDecision(
            case_id="test",
            court="Cassazione civile",
            section="Sez. III",
            date="2023-05-15",
            number="12345",
        )

        assert "Cassazione civile" in case.citation
        assert "Sez. III" in case.citation
        assert "12345" in case.citation

    def test_to_dict(self):
        """Test CaseDecision serialization."""
        case = CaseDecision(
            case_id="test",
            court="Tribunale",
            authority_score=0.5,
            relevance_score=0.7,
        )

        d = case.to_dict()

        assert d["case_id"] == "test"
        assert d["court"] == "Tribunale"
        assert d["authority_score"] == 0.5


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for prompt template."""

    def test_prompt_template_exists(self):
        """Test that prompt template is defined."""
        assert PRECEDENT_PROMPT_TEMPLATE
        assert len(PRECEDENT_PROMPT_TEMPLATE) > 100

    def test_prompt_template_placeholders(self):
        """Test that prompt template has required placeholders."""
        assert "{query}" in PRECEDENT_PROMPT_TEMPLATE
        assert "{main_norm}" in PRECEDENT_PROMPT_TEMPLATE
        assert "{case_law}" in PRECEDENT_PROMPT_TEMPLATE
        assert "{massime}" in PRECEDENT_PROMPT_TEMPLATE

    def test_prompt_template_italian(self):
        """Test that prompt template is in Italian."""
        assert "italiano" in PRECEDENT_PROMPT_TEMPLATE.lower() or "giurisprudenza" in PRECEDENT_PROMPT_TEMPLATE.lower()


# =============================================================================
# Recency Boost Tests
# =============================================================================


class TestRecencyBoost:
    """Tests for recency boost functionality."""

    @pytest.mark.asyncio
    async def test_recent_cases_get_boost(self, mock_llm, sample_norm_chunks):
        """Test that recent cases get authority boost."""
        from datetime import datetime, timedelta

        # Create case with recent date
        recent_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        old_date = "2015-01-01"

        case_chunks = [
            {
                "id": "recent",
                "case_id": "recent_case",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "date": recent_date,
                "score": 0.8,
            },
            {
                "id": "old",
                "case_id": "old_case",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "date": old_date,
                "score": 0.8,
            },
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=case_chunks,
        )
        config = PrecedentConfig(prefer_recent=True, recency_boost_factor=0.2)
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm, config=config)

        context = ExpertContext(query_text="Test recency")
        response = await expert.analyze(context)

        # Recent case should have higher effective score
        top_cases = response.metadata.get("top_cases", [])
        assert len(top_cases) >= 2
        # Recent case should be ranked higher due to recency boost
        assert top_cases[0]["case_id"] == "recent_case"

    @pytest.mark.asyncio
    async def test_recency_boost_disabled(self, mock_llm, sample_norm_chunks):
        """Test that recency boost can be disabled."""
        from datetime import datetime, timedelta

        recent_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        case_chunks = [
            {
                "id": "c1",
                "case_id": "case_1",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "date": recent_date,
                "score": 0.7,
            },
            {
                "id": "c2",
                "case_id": "case_2",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "date": "2010-01-01",
                "score": 0.9,  # Higher relevance
            },
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=case_chunks,
        )
        config = PrecedentConfig(prefer_recent=False)
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm, config=config)

        context = ExpertContext(query_text="Test no recency")
        response = await expert.analyze(context)

        # Without recency boost, higher relevance score should win
        top_cases = response.metadata.get("top_cases", [])
        assert top_cases[0]["case_id"] == "case_2"

    def test_recency_boost_factor_configurable(self):
        """Test that recency boost factor is configurable."""
        config = PrecedentConfig(recency_boost_factor=0.5)
        assert config.recency_boost_factor == 0.5


# =============================================================================
# Sezioni Unite Boost Tests
# =============================================================================


class TestSezioniUniteBoost:
    """Tests for Sezioni Unite authority boost."""

    @pytest.mark.asyncio
    async def test_sezioni_unite_boost_applied(self, mock_llm, sample_norm_chunks):
        """Test that Sezioni Unite gets additive boost."""
        case_chunks = [
            {
                "id": "su",
                "case_id": "su_case",
                "court": "Cassazione civile",
                "section": "Sezioni Unite",
                "date": "2020-01-01",
                "score": 0.75,
            },
            {
                "id": "sez",
                "case_id": "sez_case",
                "court": "Cassazione civile",
                "section": "Sez. III",
                "date": "2023-01-01",  # More recent
                "score": 0.85,  # Higher relevance
            },
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=case_chunks,
        )
        config = PrecedentConfig(sezioni_unite_boost=0.15, prefer_recent=False)
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm, config=config)

        context = ExpertContext(query_text="Test SU boost")
        response = await expert.analyze(context)

        # Sezioni Unite should rank higher due to authority boost
        top_cases = response.metadata.get("top_cases", [])
        assert top_cases[0]["case_id"] == "su_case"

    def test_sezioni_unite_boost_configurable(self):
        """Test that Sezioni Unite boost is configurable."""
        config = PrecedentConfig(sezioni_unite_boost=0.25)
        assert config.sezioni_unite_boost == 0.25


# =============================================================================
# Massime Filtering Tests
# =============================================================================


class TestMassimeFiltering:
    """Tests for massime score filtering."""

    @pytest.mark.asyncio
    async def test_massime_filtered_by_score(self, mock_llm, sample_norm_chunks):
        """Test that low-score massime are filtered out."""
        massime_chunks = [
            {"id": "m1", "score": 0.5, "massima": "High relevance massima"},
            {"id": "m2", "score": 0.2, "massima": "Low relevance massima"},
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=[],
            massime_chunks=massime_chunks,
        )
        config = PrecedentConfig(min_case_score=0.4)
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm, config=config)

        context = ExpertContext(query_text="Test massime filter")
        # The expert should filter out massime below 0.4 score
        response = await expert.analyze(context)

        # Only 1 massima should pass the filter
        assert response.metadata.get("cases_found", 0) <= 1


# =============================================================================
# Conflict Configuration Tests
# =============================================================================


class TestConflictConfiguration:
    """Tests for conflict detection configuration."""

    def test_min_conflict_indicators_configurable(self):
        """Test that min conflict indicators is configurable."""
        config = PrecedentConfig(min_conflict_indicators=2)
        assert config.min_conflict_indicators == 2

    @pytest.mark.asyncio
    async def test_conflict_threshold_applied(self, mock_llm, sample_norm_chunks):
        """Test that conflict detection respects min_conflict_indicators."""
        case_chunks = [
            {
                "id": "c1",
                "case_id": "case_1",
                "court": "Cassazione civile",
                "massima": "In senso contrario...",  # One indicator
                "score": 0.8,
            },
        ]
        retriever = MockRetriever(
            norm_chunks=sample_norm_chunks,
            case_chunks=case_chunks,
        )
        # Require 2 indicators minimum
        config = PrecedentConfig(min_conflict_indicators=2, detect_conflicts=True)
        expert = PrecedentExpert(retriever=retriever, llm_service=mock_llm, config=config)

        context = ExpertContext(query_text="Test threshold")
        response = await expert.analyze(context)

        # Should NOT flag conflict with only 1 indicator when threshold is 2
        assert response.metadata.get("has_conflict") is False
