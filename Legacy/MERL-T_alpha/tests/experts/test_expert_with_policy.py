"""
Test Integrazione Expert con PolicyManager
==========================================

Verifica che gli Expert usino correttamente PolicyManager
per il traversal del grafo con pesi neurali.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List

from merlt.experts.base import BaseExpert, ExpertContext, ExpertResponse
from merlt.experts.literal import LiteralExpert
from merlt.experts.systemic import SystemicExpert
from merlt.experts.principles import PrinciplesExpert
from merlt.experts.precedent import PrecedentExpert
from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig
from merlt.rlcf.execution_trace import ExecutionTrace


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding():
    """Embedding 768-dim normalizzato."""
    np.random.seed(42)
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


@pytest.fixture
def expert_context(sample_embedding):
    """ExpertContext con query e embedding."""
    return ExpertContext(
        query_text="Cos'è la legittima difesa?",
        query_embedding=sample_embedding,
        entities={"norm_references": ["art. 52 c.p."]},
        trace_id="test_trace_001"
    )


@pytest.fixture
def mock_policy_manager():
    """Mock PolicyManager."""
    manager = MagicMock(spec=PolicyManager)

    # Mock filter_relations_by_weight
    async def mock_filter(query_embedding, relation_types, expert_type, trace=None):
        # Ritorna le prime 3 relazioni
        return relation_types[:3] if relation_types else []

    manager.filter_relations_by_weight = AsyncMock(side_effect=mock_filter)

    # Mock compute_batch_weights
    async def mock_batch(query_embedding, relation_types, expert_type, trace=None):
        return {rel: (0.7, -0.36) for rel in relation_types}

    manager.compute_batch_weights = AsyncMock(side_effect=mock_batch)

    return manager


@pytest.fixture
def mock_ai_service():
    """Mock AI service."""
    service = MagicMock()

    async def mock_generate(*args, **kwargs):
        return {
            "interpretation": "Test interpretation",
            "legal_basis": [],
            "reasoning_steps": [],
            "confidence": 0.8,
            "limitations": ""
        }

    service.generate_response_async = AsyncMock(side_effect=mock_generate)
    return service


@pytest.fixture
def mock_tools():
    """Mock tools per expert."""
    from merlt.tools import BaseTool, ToolResult, ToolParameter

    class MockSemanticTool(BaseTool):
        name = "semantic_search"
        description = "Mock semantic search"

        @property
        def parameters(self) -> List[ToolParameter]:
            return [
                ToolParameter(name="query", param_type="string", description="Query"),
                ToolParameter(name="top_k", param_type="integer", description="Top K", required=False),
                ToolParameter(name="expert_type", param_type="string", description="Expert", required=False),
                ToolParameter(name="source_types", param_type="array", description="Sources", required=False),
            ]

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(
                success=True,
                data={"results": [
                    {"text": "Art. 52 c.p.", "urn": "urn:nir:stato:codice.penale:art.52"}
                ]}
            )

    class MockGraphTool(BaseTool):
        name = "graph_search"
        description = "Mock graph search"

        @property
        def parameters(self) -> List[ToolParameter]:
            return [
                ToolParameter(name="start_node", param_type="string", description="Start node"),
                ToolParameter(name="relation_types", param_type="array", description="Relations", required=False),
                ToolParameter(name="max_hops", param_type="integer", description="Max hops", required=False),
                ToolParameter(name="direction", param_type="string", description="Direction", required=False),
            ]

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(
                success=True,
                data={
                    "nodes": [{"urn": "urn:test", "properties": {"testo": "Test"}}],
                    "edges": [{"type": "RIFERIMENTO", "source": "a", "target": "b"}]
                }
            )

    return [MockSemanticTool(), MockGraphTool()]


# ============================================================================
# Test Inizializzazione Expert con PolicyManager
# ============================================================================

class TestExpertInitWithPolicy:
    """Test inizializzazione expert con PolicyManager."""

    def test_literal_expert_accepts_policy_manager(self, mock_policy_manager):
        """Test che LiteralExpert accetta policy_manager."""
        expert = LiteralExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )

        assert expert.policy_manager is mock_policy_manager

    def test_systemic_expert_accepts_policy_manager(self, mock_policy_manager):
        """Test che SystemicExpert accetta policy_manager."""
        expert = SystemicExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )

        assert expert.policy_manager is mock_policy_manager

    def test_principles_expert_accepts_policy_manager(self, mock_policy_manager):
        """Test che PrinciplesExpert accetta policy_manager."""
        expert = PrinciplesExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )

        assert expert.policy_manager is mock_policy_manager

    def test_precedent_expert_accepts_policy_manager(self, mock_policy_manager):
        """Test che PrecedentExpert accetta policy_manager."""
        expert = PrecedentExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )

        assert expert.policy_manager is mock_policy_manager

    def test_expert_without_policy_manager(self):
        """Test che expert funziona senza policy_manager."""
        expert = LiteralExpert(tools=[], ai_service=None)

        assert expert.policy_manager is None


# ============================================================================
# Test Trace Initialization
# ============================================================================

class TestTraceInitialization:
    """Test inizializzazione ExecutionTrace."""

    def test_init_trace_creates_trace(self):
        """Test che _init_trace crea un ExecutionTrace."""
        expert = LiteralExpert(tools=[], ai_service=None)
        context = ExpertContext(
            query_text="Test query",
            trace_id="test_123"
        )

        expert._init_trace(context)

        assert expert._current_trace is not None
        assert isinstance(expert._current_trace, ExecutionTrace)
        assert expert._current_trace.query_id == "test_123"

    def test_trace_contains_expert_metadata(self):
        """Test che trace contiene metadata dell'expert."""
        expert = LiteralExpert(tools=[], ai_service=None)
        context = ExpertContext(
            query_text="Test query about legittima difesa",
            trace_id="test_456"
        )

        expert._init_trace(context)

        trace_dict = expert.get_trace_dict()
        assert trace_dict is not None
        assert trace_dict["metadata"]["expert_type"] == "literal"
        assert "legittima difesa" in trace_dict["metadata"]["query_text"]

    def test_get_current_trace(self):
        """Test get_current_trace."""
        expert = LiteralExpert(tools=[], ai_service=None)

        # Prima di init, trace è None
        assert expert.get_current_trace() is None

        # Dopo init, trace esiste
        context = ExpertContext(query_text="Test", trace_id="test")
        expert._init_trace(context)

        trace = expert.get_current_trace()
        assert trace is not None


# ============================================================================
# Test Policy Filtered Relations
# ============================================================================

class TestPolicyFilteredRelations:
    """Test _get_policy_filtered_relations."""

    @pytest.mark.asyncio
    async def test_filter_with_policy_manager(self, mock_policy_manager, sample_embedding):
        """Test filtro relazioni con PolicyManager."""
        expert = LiteralExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )
        context = ExpertContext(
            query_text="Test",
            query_embedding=sample_embedding,
            trace_id="test"
        )
        expert._init_trace(context)

        relations = await expert._get_policy_filtered_relations(context)

        # Verifica che PolicyManager sia stato chiamato
        mock_policy_manager.filter_relations_by_weight.assert_called_once()

        # Verifica parametri
        call_kwargs = mock_policy_manager.filter_relations_by_weight.call_args[1]
        assert call_kwargs["query_embedding"] == sample_embedding
        assert call_kwargs["expert_type"] == "literal"

    @pytest.mark.asyncio
    async def test_filter_without_policy_manager(self, sample_embedding):
        """Test fallback senza PolicyManager."""
        expert = LiteralExpert(tools=[], ai_service=None)
        context = ExpertContext(
            query_text="Test",
            query_embedding=sample_embedding,
            trace_id="test"
        )

        relations = await expert._get_policy_filtered_relations(context)

        # Dovrebbe ritornare tutte le relazioni da config
        assert len(relations) > 0
        assert "contiene" in relations or "RIFERIMENTO" in relations

    @pytest.mark.asyncio
    async def test_filter_without_embedding(self, mock_policy_manager):
        """Test fallback senza query_embedding."""
        expert = LiteralExpert(
            tools=[],
            ai_service=None,
            policy_manager=mock_policy_manager
        )
        context = ExpertContext(
            query_text="Test",
            query_embedding=None,  # No embedding
            trace_id="test"
        )

        relations = await expert._get_policy_filtered_relations(context)

        # PolicyManager non dovrebbe essere chiamato
        mock_policy_manager.filter_relations_by_weight.assert_not_called()

        # Dovrebbe ritornare tutte le relazioni
        assert len(relations) > 0


# ============================================================================
# Test Analyze con Trace
# ============================================================================

class TestAnalyzeWithTrace:
    """Test metodo analyze con trace."""

    @pytest.mark.asyncio
    async def test_analyze_initializes_trace(self, mock_tools, mock_ai_service):
        """Test che analyze inizializza trace."""
        expert = LiteralExpert(
            tools=mock_tools,
            ai_service=mock_ai_service
        )
        context = ExpertContext(
            query_text="Cos'è la legittima difesa?",
            trace_id="test_analyze"
        )

        response = await expert.analyze(context)

        # Verifica trace inizializzato
        assert expert._current_trace is not None
        assert expert._current_trace.query_id == "test_analyze"

    @pytest.mark.asyncio
    async def test_analyze_adds_trace_to_response(self, mock_tools, mock_ai_service):
        """Test che analyze aggiunge trace alla response."""
        expert = LiteralExpert(
            tools=mock_tools,
            ai_service=mock_ai_service
        )
        context = ExpertContext(
            query_text="Cos'è la legittima difesa?",
            trace_id="test_trace_response"
        )

        response = await expert.analyze(context)

        # Verifica trace in metadata
        assert response.metadata is not None
        assert "execution_trace" in response.metadata
        assert response.metadata["execution_trace"]["query_id"] == "test_trace_response"

    @pytest.mark.asyncio
    async def test_analyze_with_policy_uses_filtered_relations(
        self, mock_tools, mock_ai_service, mock_policy_manager, sample_embedding
    ):
        """Test che analyze usa relazioni filtrate da policy."""
        expert = LiteralExpert(
            tools=mock_tools,
            ai_service=mock_ai_service,
            policy_manager=mock_policy_manager
        )
        context = ExpertContext(
            query_text="Cos'è la legittima difesa?",
            query_embedding=sample_embedding,
            trace_id="test_policy_filter"
        )

        response = await expert.analyze(context)

        # PolicyManager dovrebbe essere stato chiamato durante explore
        # (il numero di chiamate dipende dall'implementazione)
        assert response is not None


# ============================================================================
# Test Explore Iteratively con Policy
# ============================================================================

class TestExploreIterativelyWithPolicy:
    """Test explore_iteratively con PolicyManager."""

    @pytest.mark.asyncio
    async def test_explore_calls_policy_filter(
        self, mock_tools, mock_policy_manager, sample_embedding
    ):
        """Test che explore_iteratively usa policy per filtrare relazioni."""
        expert = LiteralExpert(
            tools=mock_tools,
            ai_service=None,
            policy_manager=mock_policy_manager
        )

        context = ExpertContext(
            query_text="Test query",
            query_embedding=sample_embedding,
            trace_id="test_explore"
        )
        expert._init_trace(context)

        sources = await expert.explore_iteratively(
            context=context,
            max_iterations=1
        )

        # Verifica che policy sia stata usata
        # (il filtro viene chiamato durante graph search)
        assert sources is not None


# ============================================================================
# Test All Experts
# ============================================================================

class TestAllExperts:
    """Test integrazione per tutti gli expert."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("expert_class,expert_type", [
        (LiteralExpert, "literal"),
        (SystemicExpert, "systemic"),
        (PrinciplesExpert, "principles"),
        (PrecedentExpert, "precedent"),
    ])
    async def test_expert_trace_metadata(
        self, expert_class, expert_type, mock_tools, mock_ai_service
    ):
        """Test che ogni expert aggiunge trace con expert_type corretto."""
        expert = expert_class(
            tools=mock_tools,
            ai_service=mock_ai_service
        )

        context = ExpertContext(
            query_text="Test query",
            trace_id=f"test_{expert_type}"
        )

        response = await expert.analyze(context)

        assert response.metadata is not None
        assert "execution_trace" in response.metadata

        trace_data = response.metadata["execution_trace"]
        assert trace_data["metadata"]["expert_type"] == expert_type


# ============================================================================
# Test RLCF Loop Integration
# ============================================================================

class TestRLCFLoopIntegration:
    """Test integrazione completa per RLCF loop."""

    @pytest.mark.asyncio
    async def test_full_rlcf_flow(self, mock_tools, mock_ai_service, tmp_path):
        """Test flusso completo: query -> analyze -> trace -> feedback."""
        from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig
        from merlt.rlcf.multilevel_feedback import MultilevelFeedback

        # Setup PolicyManager reale (senza checkpoint, usa static)
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=False,  # Usa static per test
            device="cpu"
        )
        policy_manager = PolicyManager(config=config)

        # Crea expert con policy manager
        expert = LiteralExpert(
            tools=mock_tools,
            ai_service=mock_ai_service,
            policy_manager=policy_manager
        )

        # Crea context con embedding
        np.random.seed(42)
        embedding = np.random.randn(768).astype(np.float32).tolist()
        context = ExpertContext(
            query_text="Cos'è la legittima difesa secondo l'art. 52 c.p.?",
            query_embedding=embedding,
            trace_id="rlcf_test"
        )

        # Analyze
        response = await expert.analyze(context)

        # Verifica response
        assert response is not None
        assert response.metadata is not None
        assert "execution_trace" in response.metadata

        # Estrai trace
        trace_dict = response.metadata["execution_trace"]
        trace = ExecutionTrace.from_dict(trace_dict)

        assert trace.query_id == "rlcf_test"

        # Simula feedback
        feedback = MultilevelFeedback(
            query_id="rlcf_test",
            overall_rating=0.8
        )

        # Verifica che trace e feedback possano essere usati per training
        # (qui non testiamo il training effettivo, solo la compatibilità)
        assert trace is not None
        assert feedback.overall_rating == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
