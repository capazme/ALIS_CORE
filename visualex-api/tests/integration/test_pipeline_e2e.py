"""
End-to-End Integration Tests for Pipeline Orchestrator (Story 5.0).

Tests the complete pipeline flow with real or mock LLM service.
Requires OPENROUTER_API_KEY for live tests.

Run with: pytest tests/integration/test_pipeline_e2e.py -v -s
"""

import asyncio
import os
import pytest
from datetime import datetime
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

from visualex.experts import (
    PipelineOrchestrator,
    PipelineRequest,
    PipelineResult,
    OrchestratorConfig,
    LiteralExpert,
    SystemicExpert,
    PrinciplesExpert,
    PrecedentExpert,
    LiteralConfig,
    SystemicConfig,
    PrinciplesConfig,
    PrecedentConfig,
    ExpertRouter,
    GatingNetwork,
    Synthesizer,
    LLMProviderFactory,
    FailoverLLMService,
    PipelineStage,
)
from visualex.ner import NERService


# =============================================================================
# Mock Retriever and Graph
# =============================================================================


class MockChunkRetriever:
    """Mock retriever that returns predefined chunks."""

    def __init__(self, chunks: Optional[List[dict]] = None):
        self.chunks = chunks or [
            {
                "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
                "text": "Art. 1453 - Risolubilità del contratto per inadempimento. "
                       "Nei contratti con prestazioni corrispettive, quando uno dei "
                       "contraenti non adempie le sue obbligazioni, l'altro può a sua "
                       "scelta chiedere l'adempimento o la risoluzione del contratto.",
                "articolo": "1453",
                "tipo_atto": "regio.decreto",
            },
            {
                "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1455",
                "text": "Art. 1455 - Importanza dell'inadempimento. "
                       "Il contratto non si può risolvere se l'inadempimento di una "
                       "delle parti ha scarsa importanza, avuto riguardo all'interesse "
                       "dell'altra.",
                "articolo": "1455",
                "tipo_atto": "regio.decreto",
            },
        ]

    async def retrieve(self, query: str, top_k: int = 5) -> List[dict]:
        """Return mock chunks."""
        return self.chunks[:top_k]


class MockGraphTraverser:
    """Mock graph traverser for systemic expert (implements GraphTraverser protocol)."""

    async def get_related(
        self,
        urn: str,
        relation_types: Optional[List[str]] = None,
        max_depth: int = 1,
        limit: int = 10,
    ) -> List[dict]:
        """Return mock related norms."""
        return [
            {
                "source_urn": urn,
                "target_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1454",
                "relation_type": "MODIFICA",
                "weight": 0.8,
            },
            {
                "source_urn": urn,
                "target_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1456",
                "relation_type": "RIFERIMENTO",
                "weight": 0.7,
            },
        ]

    async def get_history(self, urn: str) -> List[dict]:
        """Return mock historical versions."""
        return []


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_retriever():
    """Create mock chunk retriever."""
    return MockChunkRetriever()


@pytest.fixture
def mock_graph_traverser():
    """Create mock graph traverser."""
    return MockGraphTraverser()


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service for unit tests."""
    service = AsyncMock()
    service.generate = AsyncMock(return_value="Risposta mock dell'LLM.")
    return service


@pytest.fixture
def orchestrator_with_mocks(mock_retriever, mock_graph_traverser, mock_llm_service):
    """Create orchestrator with all mocked dependencies."""
    config = OrchestratorConfig(
        expert_timeout_ms=30000.0,
        total_timeout_ms=120000.0,
        parallel_execution=True,
        enable_tracing=True,
    )

    # Create experts with mock dependencies
    literal_expert = LiteralExpert(
        retriever=mock_retriever,
        llm_service=mock_llm_service,
        config=LiteralConfig(),
    )

    systemic_expert = SystemicExpert(
        retriever=mock_retriever,
        graph_traverser=mock_graph_traverser,
        llm_service=mock_llm_service,
        config=SystemicConfig(),
    )

    principles_expert = PrinciplesExpert(
        retriever=mock_retriever,
        llm_service=mock_llm_service,
        config=PrinciplesConfig(),
    )

    precedent_expert = PrecedentExpert(
        retriever=mock_retriever,
        llm_service=mock_llm_service,
        config=PrecedentConfig(),
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        ner_service=NERService(),
        router=ExpertRouter(),
        gating=GatingNetwork(llm_service=mock_llm_service),
        synthesizer=Synthesizer(llm_service=mock_llm_service),
    )

    # Register experts
    orchestrator.register_expert("literal", literal_expert)
    orchestrator.register_expert("systemic", systemic_expert)
    orchestrator.register_expert("principles", principles_expert)
    orchestrator.register_expert("precedent", precedent_expert)

    return orchestrator


# =============================================================================
# Unit Integration Tests (with mocks)
# =============================================================================


class TestPipelineWithMocks:
    """Tests with mocked LLM service."""

    @pytest.mark.asyncio
    async def test_full_pipeline_execution(self, orchestrator_with_mocks):
        """Test complete pipeline execution with all stages."""
        request = PipelineRequest(
            query="Cos'è la risoluzione del contratto per inadempimento?",
            user_profile="ricerca",
        )

        result = await orchestrator_with_mocks.process_query(request)

        # Verify success
        assert result.success is True
        assert result.error is None

        # Verify response structure
        assert result.response is not None
        assert result.response.main_answer is not None

        # Verify trace
        assert result.trace is not None
        assert result.trace.trace_id is not None
        assert result.trace.query_text == request.query
        assert result.trace.ner_result is not None
        assert result.trace.routing_decision is not None

        # Verify metrics
        assert result.metrics is not None
        assert result.metrics.total_time_ms > 0
        assert result.metrics.ner_time_ms > 0
        assert result.metrics.routing_time_ms > 0

    @pytest.mark.asyncio
    async def test_pipeline_with_different_profiles(self, orchestrator_with_mocks):
        """Test pipeline with different user profiles."""
        profiles = ["consulenza", "ricerca", "analisi", "contributore"]

        for profile in profiles:
            request = PipelineRequest(
                query="Cos'è la risoluzione?",
                user_profile=profile,
            )

            result = await orchestrator_with_mocks.process_query(request)

            assert result.success is True
            assert result.response.user_profile == profile

    @pytest.mark.asyncio
    async def test_pipeline_trace_serialization(self, orchestrator_with_mocks):
        """Test that trace can be serialized to JSON."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator_with_mocks.process_query(request)

        # Trace should be JSON serializable
        json_str = result.trace.to_json()
        assert isinstance(json_str, str)
        assert result.trace.trace_id in json_str

    @pytest.mark.asyncio
    async def test_pipeline_with_override_weights(self, orchestrator_with_mocks):
        """Test pipeline with custom expert weights."""
        request = PipelineRequest(
            query="Test query",
            override_weights={"literal": 0.8, "systemic": 0.1, "principles": 0.05, "precedent": 0.05},
        )

        result = await orchestrator_with_mocks.process_query(request)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_pipeline_with_bypass_experts(self, orchestrator_with_mocks):
        """Test pipeline with bypassed experts."""
        request = PipelineRequest(
            query="Test query",
            bypass_experts=["precedent", "principles"],
        )

        result = await orchestrator_with_mocks.process_query(request)

        assert result.success is True
        assert "precedent" in result.metrics.experts_skipped
        assert "principles" in result.metrics.experts_skipped

    @pytest.mark.asyncio
    async def test_pipeline_metrics_stage_times(self, orchestrator_with_mocks):
        """Test that stage times are recorded in metrics."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator_with_mocks.process_query(request)

        # Check stage times in trace
        assert PipelineStage.NER in result.trace.stage_times_ms
        assert PipelineStage.ROUTING in result.trace.stage_times_ms
        assert PipelineStage.GATING in result.trace.stage_times_ms
        assert PipelineStage.SYNTHESIS in result.trace.stage_times_ms

    @pytest.mark.asyncio
    async def test_pipeline_feedback_hooks_collected(self, orchestrator_with_mocks):
        """Test that feedback hooks are collected from all stages."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator_with_mocks.process_query(request)

        # Should have feedback hooks
        assert len(result.feedback_hooks) > 0

        # Check feedback types
        hook_types = {fh.feedback_type for fh in result.feedback_hooks}
        # At minimum, should have gating (F7) or synthesis (F7) hooks
        assert len(hook_types) > 0


# =============================================================================
# Live Integration Tests (require API key)
# =============================================================================


@pytest.fixture
def has_api_key():
    """Check if OpenRouter API key is available."""
    return bool(os.getenv("OPENROUTER_API_KEY"))


@pytest.fixture
def live_llm_service():
    """Create live LLM service."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        pytest.skip("OPENROUTER_API_KEY not set")

    factory = LLMProviderFactory()
    provider = factory.create("openrouter")
    return FailoverLLMService(providers=[provider])


@pytest.fixture
def live_orchestrator(mock_retriever, mock_graph_traverser, live_llm_service):
    """Create orchestrator with live LLM service."""
    config = OrchestratorConfig(
        expert_timeout_ms=60000.0,  # 60 seconds for live LLM
        total_timeout_ms=180000.0,  # 3 minutes total
        parallel_execution=True,
        enable_tracing=True,
    )

    # Create experts with live LLM
    literal_expert = LiteralExpert(
        retriever=mock_retriever,
        llm_service=live_llm_service,
        config=LiteralConfig(),
    )

    systemic_expert = SystemicExpert(
        retriever=mock_retriever,
        graph_traverser=mock_graph_traverser,
        llm_service=live_llm_service,
        config=SystemicConfig(),
    )

    principles_expert = PrinciplesExpert(
        retriever=mock_retriever,
        llm_service=live_llm_service,
        config=PrinciplesConfig(),
    )

    precedent_expert = PrecedentExpert(
        retriever=mock_retriever,
        llm_service=live_llm_service,
        config=PrecedentConfig(),
    )

    # Create orchestrator
    orchestrator = PipelineOrchestrator(
        config=config,
        ner_service=NERService(),
        router=ExpertRouter(),
        gating=GatingNetwork(llm_service=live_llm_service),
        synthesizer=Synthesizer(llm_service=live_llm_service),
    )

    # Register experts
    orchestrator.register_expert("literal", literal_expert)
    orchestrator.register_expert("systemic", systemic_expert)
    orchestrator.register_expert("principles", principles_expert)
    orchestrator.register_expert("precedent", precedent_expert)

    return orchestrator


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set"
)
class TestPipelineLive:
    """Live integration tests with real LLM."""

    @pytest.mark.asyncio
    async def test_live_pipeline_definition_query(self, live_orchestrator):
        """Test live pipeline with definition query."""
        request = PipelineRequest(
            query="Cos'è la risoluzione del contratto per inadempimento?",
            user_profile="ricerca",
        )

        result = await live_orchestrator.process_query(request)

        # Verify success
        assert result.success is True, f"Pipeline failed: {result.error}"

        # Verify response
        assert result.response is not None
        assert len(result.response.main_answer) > 100  # Should have substantial answer

        # Verify Italian content
        answer_lower = result.response.main_answer.lower()
        assert any(word in answer_lower for word in ["risoluzione", "contratto", "inadempimento"])

        # Print for manual verification
        print(f"\n{'='*60}")
        print(f"Query: {request.query}")
        print(f"Profile: {request.user_profile}")
        print(f"{'='*60}")
        print(f"Response ({len(result.response.main_answer)} chars):")
        print(result.response.main_answer[:500] + "...")
        print(f"\nMetrics:")
        print(f"  - Total time: {result.metrics.total_time_ms:.0f}ms")
        print(f"  - Total tokens: {result.metrics.total_tokens}")
        print(f"  - Experts activated: {result.metrics.experts_activated}")
        print(f"  - Degraded: {result.metrics.degraded}")

    @pytest.mark.asyncio
    async def test_live_pipeline_application_query(self, live_orchestrator):
        """Test live pipeline with application query."""
        request = PipelineRequest(
            query="Quando posso risolvere un contratto per inadempimento?",
            user_profile="analisi",
        )

        result = await live_orchestrator.process_query(request)

        assert result.success is True, f"Pipeline failed: {result.error}"
        assert len(result.response.main_answer) > 100

        # Print metrics
        print(f"\nTotal time: {result.metrics.total_time_ms:.0f}ms")
        print(f"Experts: {result.metrics.experts_activated}")

    @pytest.mark.asyncio
    async def test_live_pipeline_trace_complete(self, live_orchestrator):
        """Test that live pipeline produces complete trace."""
        request = PipelineRequest(query="Cos'è la clausola risolutiva espressa?")

        result = await live_orchestrator.process_query(request)

        assert result.success is True

        # Verify trace completeness
        trace = result.trace
        assert trace.ner_result.get("entities") is not None
        assert trace.routing_decision.get("query_type") is not None
        assert len(trace.expert_executions) > 0
        assert trace.gating_result.get("synthesis") is not None
        assert trace.synthesis_result.get("main_answer") is not None

        # Verify timing
        assert trace.total_time_ms > 0
        assert len(trace.stage_times_ms) >= 4  # NER, routing, gating, synthesis

        print(f"\nTrace ID: {trace.trace_id}")
        print(f"Total time: {trace.total_time_ms:.0f}ms")
        print(f"Stage times: {trace.stage_times_ms}")
