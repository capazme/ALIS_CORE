"""
Tests for SystemicExpert.

Tests cover:
- AC1: Traverse Knowledge Graph (RIFERIMENTO, MODIFICA, ATTUA edges)
- AC2: LLM synthesizes how related norms affect interpretation
- AC3: Output includes section header, graph data, synthesis, confidence
- AC4: Low confidence for isolated norms
"""

import pytest
from typing import Any, Dict, List, Optional

from visualex.experts import (
    SystemicExpert,
    SystemicConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    GraphRelation,
    SYSTEMIC_PROMPT_TEMPLATE,
)
from visualex.experts.base import BaseExpert


# =============================================================================
# Mock Classes
# =============================================================================


class MockRetriever:
    """Mock retriever for testing."""

    def __init__(self, chunks: List[Dict[str, Any]] = None):
        self.chunks = chunks or []
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
        return self.chunks[:limit]


class MockGraphTraverser:
    """Mock graph traverser for testing."""

    def __init__(
        self,
        neighbors: List[Dict[str, Any]] = None,
        modifications: List[Dict[str, Any]] = None,
    ):
        self.neighbors = neighbors or []
        self.modifications = modifications or []
        self.get_neighbors_calls: List[Dict] = []
        self.get_modifications_calls: List[Dict] = []

    async def get_neighbors(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        depth: int = 1,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        self.get_neighbors_calls.append({
            "urn": urn,
            "relation_types": relation_types,
            "depth": depth,
            "limit": limit,
        })
        return self.neighbors[:limit]

    async def get_modifications(
        self,
        urn: str,
        as_of_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.get_modifications_calls.append({
            "urn": urn,
            "as_of_date": as_of_date,
        })
        return self.modifications


class MockLLMService:
    """Mock LLM service for testing."""

    def __init__(self, response: str = "Test systemic interpretation"):
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
def sample_main_norm_chunks():
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
def sample_related_norms():
    """Sample related norms from graph traversal."""
    return [
        {
            "urn": "urn:nir:stato:codice.civile:art_1454",
            "citation": "Art. 1454 c.c.",
            "text": "Diffida ad adempiere...",
            "relation_type": "riferimento",
            "edge_metadata": {"context": "rinvio normativo"},
        },
        {
            "urn": "urn:nir:stato:codice.civile:art_1455",
            "citation": "Art. 1455 c.c.",
            "text": "Importanza dell'inadempimento...",
            "relation_type": "riferimento",
        },
        {
            "urn": "urn:nir:unione.europea:direttiva:2011-83",
            "citation": "Dir. 2011/83/UE",
            "text": "Direttiva sui diritti dei consumatori...",
            "relation_type": "attua",
        },
    ]


@pytest.fixture
def sample_modifications():
    """Sample modification history."""
    return [
        {
            "data_effetto": "2014-06-13",
            "tipo_modifica": "modifica",
            "norma_modificante": "D.Lgs. 21/2014",
        },
    ]


@pytest.fixture
def mock_retriever(sample_main_norm_chunks):
    """Create mock retriever."""
    return MockRetriever(chunks=sample_main_norm_chunks)


@pytest.fixture
def mock_graph(sample_related_norms, sample_modifications):
    """Create mock graph traverser."""
    return MockGraphTraverser(
        neighbors=sample_related_norms,
        modifications=sample_modifications,
    )


@pytest.fixture
def mock_llm():
    """Create mock LLM service."""
    return MockLLMService(
        response="L'art. 1453 c.c. si inserisce nel sistema delle obbligazioni contrattuali. "
        "È strettamente connesso all'art. 1454 (diffida) e all'art. 1455 (gravità). "
        "La normativa è stata influenzata dalla Dir. 2011/83/UE."
    )


@pytest.fixture
def expert_with_mocks(mock_retriever, mock_graph, mock_llm):
    """Create SystemicExpert with all mocks."""
    return SystemicExpert(
        retriever=mock_retriever,
        graph_traverser=mock_graph,
        llm_service=mock_llm,
    )


@pytest.fixture
def expert_no_graph(mock_retriever, mock_llm):
    """Create SystemicExpert without graph traverser."""
    return SystemicExpert(
        retriever=mock_retriever,
        llm_service=mock_llm,
    )


@pytest.fixture
def sample_context():
    """Sample context for testing."""
    return ExpertContext(
        query_text="Come si collega l'art. 1453 c.c. alle altre norme sulla risoluzione?",
        entities={
            "norm_references": ["urn:nir:stato:codice.civile:art_1453"],
            "legal_concepts": ["risoluzione", "inadempimento"],
        },
    )


# =============================================================================
# Base Configuration Tests
# =============================================================================


class TestSystemicExpertConfiguration:
    """Tests for SystemicExpert configuration."""

    def test_expert_type(self):
        """Test that SystemicExpert has correct type."""
        expert = SystemicExpert()
        assert expert.expert_type == "systemic"

    def test_section_header(self):
        """Test that SystemicExpert has correct Italian header."""
        expert = SystemicExpert()
        assert expert.section_header == "Interpretazione Sistematica"

    def test_description(self):
        """Test that SystemicExpert has description."""
        expert = SystemicExpert()
        assert "art. 12" in expert.description.lower()
        assert "art. 14" in expert.description.lower()


class TestSystemicConfig:
    """Tests for SystemicConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SystemicConfig()

        assert config.max_traversal_depth == 2
        assert config.max_related_norms == 10
        assert config.include_historical is True
        assert config.enable_f4_feedback is True
        assert config.relation_weights is not None

    def test_default_relation_weights(self):
        """Test default relation weights."""
        config = SystemicConfig()

        assert config.relation_weights["riferimento"] == 0.9
        assert config.relation_weights["modifica"] == 0.95
        assert config.relation_weights["attua"] == 0.85

    def test_custom_config(self):
        """Test custom configuration."""
        config = SystemicConfig(
            max_traversal_depth=3,
            include_historical=False,
        )

        assert config.max_traversal_depth == 3
        assert config.include_historical is False


# =============================================================================
# Graph Traversal Tests (AC1)
# =============================================================================


class TestGraphTraversal:
    """Tests for Knowledge Graph traversal (AC1)."""

    @pytest.mark.asyncio
    async def test_traverses_graph(self, expert_with_mocks, sample_context, mock_graph):
        """Test that expert traverses the graph."""
        await expert_with_mocks.analyze(sample_context)

        assert len(mock_graph.get_neighbors_calls) > 0

    @pytest.mark.asyncio
    async def test_uses_correct_relation_types(self, expert_with_mocks, sample_context, mock_graph):
        """Test that traversal uses correct relation types."""
        await expert_with_mocks.analyze(sample_context)

        call = mock_graph.get_neighbors_calls[0]
        relation_types = call["relation_types"]

        assert "riferimento" in relation_types
        assert "modifica" in relation_types
        assert "attua" in relation_types

    @pytest.mark.asyncio
    async def test_includes_related_norms_in_response(self, expert_with_mocks, sample_context):
        """Test that related norms appear in response."""
        response = await expert_with_mocks.analyze(sample_context)

        # Check metadata for related norms count
        assert response.metadata["related_norms_count"] > 0

    @pytest.mark.asyncio
    async def test_graph_visualization_data(self, expert_with_mocks, sample_context):
        """Test that response includes graph visualization data."""
        response = await expert_with_mocks.analyze(sample_context)

        graph_data = response.metadata.get("graph_data", {})
        assert "nodes" in graph_data
        assert "edges" in graph_data

    @pytest.mark.asyncio
    async def test_handles_no_graph_traverser(self, expert_no_graph, sample_context):
        """Test handling when no graph traverser available."""
        response = await expert_no_graph.analyze(sample_context)

        # Should still work but with limited results
        assert response.expert_type == "systemic"
        assert "Knowledge Graph non disponibile" in response.limitations


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
    async def test_prompt_includes_main_norm(self, expert_with_mocks, sample_context, mock_llm):
        """Test that prompt includes main norm."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "NORMA PRINCIPALE" in prompt

    @pytest.mark.asyncio
    async def test_prompt_includes_related_norms(self, expert_with_mocks, sample_context, mock_llm):
        """Test that prompt includes related norms."""
        await expert_with_mocks.analyze(sample_context)

        prompt = mock_llm.generate_calls[0]["prompt"]
        assert "NORME CORRELATE" in prompt

    @pytest.mark.asyncio
    async def test_fallback_without_llm(self, mock_retriever, mock_graph, sample_context):
        """Test fallback interpretation without LLM."""
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=mock_graph,
        )

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

        assert response.section_header == "Interpretazione Sistematica"

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
    async def test_legal_sources_include_relation_type(self, expert_with_mocks, sample_context):
        """Test that legal sources indicate relation type."""
        response = await expert_with_mocks.analyze(sample_context)

        # Related norms should have relation type in relevance
        related_sources = [s for s in response.legal_basis if "Connessa via" in s.relevance]
        assert len(related_sources) > 0

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
        assert any("Knowledge Graph" in s.description for s in response.reasoning_steps)

    @pytest.mark.asyncio
    async def test_includes_execution_time(self, expert_with_mocks, sample_context):
        """Test that response includes execution time."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_to_dict_serialization(self, expert_with_mocks, sample_context):
        """Test that response serializes correctly."""
        response = await expert_with_mocks.analyze(sample_context)

        d = response.to_dict()

        assert d["expert_type"] == "systemic"
        assert d["section_header"] == "Interpretazione Sistematica"
        assert "graph_data" in d["metadata"]


# =============================================================================
# Isolated Norm Tests (AC4)
# =============================================================================


class TestIsolatedNorm:
    """Tests for isolated norm handling (AC4)."""

    @pytest.mark.asyncio
    async def test_low_confidence_for_isolated_norm(self, mock_retriever, mock_llm, sample_context):
        """Test that isolated norms get low confidence."""
        # Graph with no neighbors
        empty_graph = MockGraphTraverser(neighbors=[], modifications=[])
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=empty_graph,
            llm_service=mock_llm,
        )

        response = await expert.analyze(sample_context)

        assert response.confidence <= 0.4

    @pytest.mark.asyncio
    async def test_isolated_norm_notes_limited_context(self, mock_retriever, mock_llm, sample_context):
        """Test that isolated norm response notes limited context."""
        empty_graph = MockGraphTraverser(neighbors=[], modifications=[])
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=empty_graph,
            llm_service=mock_llm,
        )

        response = await expert.analyze(sample_context)

        # Should mention limited context
        assert "isolata" in response.interpretation.lower() or "poche" in response.limitations.lower()

    @pytest.mark.asyncio
    async def test_isolated_norm_metadata_flag(self, mock_retriever, mock_llm, sample_context):
        """Test that isolated norm has metadata flag."""
        empty_graph = MockGraphTraverser(neighbors=[], modifications=[])
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=empty_graph,
            llm_service=mock_llm,
        )

        response = await expert.analyze(sample_context)

        # Check for isolated flag or zero related norms
        assert (
            response.metadata.get("isolated_norm", False) or
            response.metadata.get("related_norms_count", 0) == 0
        )


# =============================================================================
# F4 Feedback Hook Tests
# =============================================================================


class TestF4FeedbackHook:
    """Tests for F4 feedback hook integration."""

    @pytest.mark.asyncio
    async def test_includes_f4_feedback_hook(self, expert_with_mocks, sample_context):
        """Test that response includes F4 feedback hook."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook is not None
        assert isinstance(response.feedback_hook, FeedbackHook)

    @pytest.mark.asyncio
    async def test_f4_hook_has_correct_type(self, expert_with_mocks, sample_context):
        """Test that F4 hook has correct feedback type."""
        response = await expert_with_mocks.analyze(sample_context)

        assert response.feedback_hook.feedback_type == "F4"
        assert response.feedback_hook.expert_type == "systemic"

    @pytest.mark.asyncio
    async def test_f4_hook_can_be_disabled(self, mock_retriever, mock_graph, mock_llm, sample_context):
        """Test that F4 feedback can be disabled."""
        config = SystemicConfig(enable_f4_feedback=False)
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=mock_graph,
            llm_service=mock_llm,
            config=config,
        )

        response = await expert.analyze(sample_context)

        assert response.feedback_hook is None


# =============================================================================
# Historical Context Tests
# =============================================================================


class TestHistoricalContext:
    """Tests for historical modification context."""

    @pytest.mark.asyncio
    async def test_includes_historical_context(self, expert_with_mocks, sample_context, mock_graph):
        """Test that historical modifications are retrieved."""
        await expert_with_mocks.analyze(sample_context)

        assert len(mock_graph.get_modifications_calls) > 0

    @pytest.mark.asyncio
    async def test_can_disable_historical(self, mock_retriever, mock_graph, mock_llm, sample_context):
        """Test that historical retrieval can be disabled."""
        config = SystemicConfig(include_historical=False)
        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=mock_graph,
            llm_service=mock_llm,
            config=config,
        )

        await expert.analyze(sample_context)

        assert len(mock_graph.get_modifications_calls) == 0


# =============================================================================
# GraphRelation Tests
# =============================================================================


class TestGraphRelation:
    """Tests for GraphRelation dataclass."""

    def test_creation(self):
        """Test GraphRelation creation."""
        relation = GraphRelation(
            relation_type="riferimento",
            source_urn="urn:source",
            target_urn="urn:target",
            metadata={"context": "test"},
        )

        assert relation.relation_type == "riferimento"
        assert relation.source_urn == "urn:source"
        assert relation.target_urn == "urn:target"

    def test_to_dict(self):
        """Test GraphRelation serialization."""
        relation = GraphRelation(
            relation_type="modifica",
            source_urn="urn:a",
            target_urn="urn:b",
        )

        d = relation.to_dict()

        assert d["relation_type"] == "modifica"
        assert d["source_urn"] == "urn:a"
        assert d["target_urn"] == "urn:b"
        assert d["metadata"] == {}

    def test_default_metadata_is_empty_dict(self):
        """Test that default metadata is an empty dict, not None."""
        relation = GraphRelation(
            relation_type="attua",
            source_urn="urn:x",
            target_urn="urn:y",
        )

        assert relation.metadata == {}
        assert isinstance(relation.metadata, dict)


# =============================================================================
# Deduplication Tests (Code Review Fix)
# =============================================================================


class TestDeduplication:
    """Tests for related norm deduplication."""

    @pytest.mark.asyncio
    async def test_deduplicates_related_norms(self, mock_retriever, mock_llm, sample_context):
        """Test that duplicate norms from multiple traversals are deduplicated."""
        # Graph that returns same norm from different starting URNs
        duplicate_neighbors = [
            {
                "urn": "urn:nir:stato:codice.civile:art_1454",
                "citation": "Art. 1454 c.c.",
                "text": "Diffida ad adempiere...",
                "relation_type": "riferimento",
            },
        ]
        graph_with_duplicates = MockGraphTraverser(
            neighbors=duplicate_neighbors,
            modifications=[],
        )

        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=graph_with_duplicates,
            llm_service=mock_llm,
        )

        # Context with multiple norm references that would both traverse to same neighbor
        context = ExpertContext(
            query_text="Test query",
            entities={
                "norm_references": [
                    "urn:nir:stato:codice.civile:art_1453",
                    "urn:nir:stato:codice.civile:art_1455",
                ],
            },
        )

        response = await expert.analyze(context)

        # Should only have 1 related norm despite 2 traversals finding the same URN
        # (excluding main norms in legal_basis)
        related_sources = [s for s in response.legal_basis if "Connessa via" in s.relevance]
        assert len(related_sources) == 1


# =============================================================================
# Relation Weights Tests (Code Review Fix)
# =============================================================================


class TestRelationWeights:
    """Tests for relation weight ranking."""

    @pytest.mark.asyncio
    async def test_legal_sources_include_weight(self, expert_with_mocks, sample_context):
        """Test that legal sources include relation weight in relevance."""
        response = await expert_with_mocks.analyze(sample_context)

        related_sources = [s for s in response.legal_basis if "Connessa via" in s.relevance]
        assert len(related_sources) > 0

        # Check that at least one source has weight info
        has_weight = any("peso:" in s.relevance for s in related_sources)
        assert has_weight

    @pytest.mark.asyncio
    async def test_sources_sorted_by_weight(self, mock_retriever, mock_llm, sample_context):
        """Test that related sources are sorted by relation weight (descending)."""
        # Create neighbors with different relation types
        mixed_neighbors = [
            {
                "urn": "urn:attua",
                "citation": "Attua",
                "text": "...",
                "relation_type": "attua",  # weight 0.85
            },
            {
                "urn": "urn:modifica",
                "citation": "Modifica",
                "text": "...",
                "relation_type": "modifica",  # weight 0.95 (highest)
            },
            {
                "urn": "urn:rinvia",
                "citation": "Rinvia",
                "text": "...",
                "relation_type": "rinvia",  # weight 0.80
            },
        ]
        graph = MockGraphTraverser(neighbors=mixed_neighbors, modifications=[])

        expert = SystemicExpert(
            retriever=mock_retriever,
            graph_traverser=graph,
            llm_service=mock_llm,
        )

        response = await expert.analyze(sample_context)

        # Get related sources (excluding main norm)
        related_sources = [s for s in response.legal_basis if "Connessa via" in s.relevance]

        # First related should be MODIFICA (highest weight 0.95)
        assert "MODIFICA" in related_sources[0].relevance


# =============================================================================
# Prompt Template Tests
# =============================================================================


class TestPromptTemplate:
    """Tests for prompt template."""

    def test_prompt_template_exists(self):
        """Test that prompt template is defined."""
        assert SYSTEMIC_PROMPT_TEMPLATE
        assert len(SYSTEMIC_PROMPT_TEMPLATE) > 100

    def test_prompt_template_placeholders(self):
        """Test that prompt template has required placeholders."""
        assert "{query}" in SYSTEMIC_PROMPT_TEMPLATE
        assert "{main_norm}" in SYSTEMIC_PROMPT_TEMPLATE
        assert "{related_norms}" in SYSTEMIC_PROMPT_TEMPLATE
        assert "{historical_context}" in SYSTEMIC_PROMPT_TEMPLATE

    def test_prompt_template_italian(self):
        """Test that prompt template is in Italian."""
        assert "italiano" in SYSTEMIC_PROMPT_TEMPLATE.lower() or "sistematica" in SYSTEMIC_PROMPT_TEMPLATE.lower()
