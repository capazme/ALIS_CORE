"""
Unit tests for Pipeline Orchestrator (Story 5.0 Tasks 2-6).

Tests PipelineOrchestrator class with mocked dependencies.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from visualex.experts.pipeline import PipelineOrchestrator
from visualex.experts.pipeline_types import (
    PipelineRequest,
    PipelineResult,
    OrchestratorConfig,
    PipelineValidationError,
    PipelineTimeoutError,
)
from visualex.experts.base import (
    ExpertContext,
    ExpertResponse,
    FeedbackHook,
    ConfidenceFactors,
)
from visualex.experts.router import RoutingDecision, QueryType, ExpertWeight, ExpertType
from visualex.experts.gating import AggregatedResponse, ExpertContribution
from visualex.experts.synthesizer import SynthesizedResponse
from visualex.ner.entities import ExtractionResult, ExtractedEntity, EntityType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_ner_result():
    """Create mock NER result."""
    return ExtractionResult(
        text="Cos'è la risoluzione del contratto?",
        entities=[
            ExtractedEntity(
                text="risoluzione",
                entity_type=EntityType.LEGAL_CONCEPT,
                start=9,
                end=20,
                confidence=0.9,
            ),
            ExtractedEntity(
                text="contratto",
                entity_type=EntityType.LEGAL_CONCEPT,
                start=25,
                end=34,
                confidence=0.95,
            ),
        ],
        processing_time_ms=50.0,
    )


@pytest.fixture
def mock_routing_decision():
    """Create mock routing decision."""
    return RoutingDecision(
        query_type=QueryType.DEFINITION,
        expert_weights=[
            ExpertWeight(expert=ExpertType.LITERAL, weight=0.5, is_primary=True),
            ExpertWeight(expert=ExpertType.SYSTEMIC, weight=0.25),
            ExpertWeight(expert=ExpertType.PRINCIPLES, weight=0.15),
            ExpertWeight(expert=ExpertType.PRECEDENT, weight=0.10),
        ],
        confidence=0.85,
        rationale="Query classificata come DEFINITION.",
    )


@pytest.fixture
def mock_expert_response():
    """Create mock expert response."""
    return ExpertResponse(
        expert_type="literal",
        section_header="Interpretazione Letterale",
        interpretation="La risoluzione del contratto è lo scioglimento...",
        confidence=0.85,
        confidence_factors=ConfidenceFactors(
            norm_clarity=0.9,
            source_availability=0.8,
        ),
        execution_time_ms=1500.0,
        tokens_used=300,
        trace_id="test-trace",
        feedback_hook=FeedbackHook(
            feedback_type="F3",
            expert_type="literal",
            response_id="test-trace",
        ),
    )


@pytest.fixture
def mock_aggregated_response(mock_expert_response):
    """Create mock aggregated response."""
    return AggregatedResponse(
        synthesis="Sintesi delle interpretazioni...",
        expert_contributions={
            "literal": ExpertContribution(
                expert_type="literal",
                interpretation=mock_expert_response.interpretation,
                confidence=0.85,
                weight=0.5,
                weighted_confidence=0.425,
            ),
        },
        confidence=0.85,
        aggregation_method="weighted_average",
        trace_id="test-trace",
        execution_time_ms=500.0,
        feedback_hook=FeedbackHook(
            feedback_type="F7",
            expert_type="gating",
            response_id="test-trace",
        ),
    )


@pytest.fixture
def mock_synthesized_response():
    """Create mock synthesized response."""
    return SynthesizedResponse(
        main_answer="La risoluzione del contratto è...",
        confidence_indicator="alta",
        confidence_value=0.85,
        synthesis_mode="convergent",
        user_profile="ricerca",
        trace_id="test-trace",
        execution_time_ms=800.0,
        feedback_hook=FeedbackHook(
            feedback_type="F7",
            expert_type="synthesizer",
            response_id="test-trace",
        ),
    )


@pytest.fixture
def mock_ner_service(mock_ner_result):
    """Create mock NER service."""
    service = AsyncMock()
    service.extract = AsyncMock(return_value=mock_ner_result)
    return service


@pytest.fixture
def mock_router(mock_routing_decision):
    """Create mock router."""
    router = AsyncMock()
    router.route = AsyncMock(return_value=mock_routing_decision)
    return router


@pytest.fixture
def mock_expert(mock_expert_response):
    """Create mock expert."""
    expert = AsyncMock()
    expert.analyze = AsyncMock(return_value=mock_expert_response)
    expert.expert_type = "literal"
    return expert


@pytest.fixture
def mock_gating(mock_aggregated_response):
    """Create mock gating network."""
    gating = AsyncMock()
    gating.aggregate = AsyncMock(return_value=mock_aggregated_response)
    return gating


@pytest.fixture
def mock_synthesizer(mock_synthesized_response):
    """Create mock synthesizer."""
    synthesizer = AsyncMock()
    synthesizer.synthesize = AsyncMock(return_value=mock_synthesized_response)
    return synthesizer


@pytest.fixture
def orchestrator(
    mock_ner_service,
    mock_router,
    mock_expert,
    mock_gating,
    mock_synthesizer,
):
    """Create orchestrator with mock dependencies."""
    orch = PipelineOrchestrator(
        config=OrchestratorConfig(
            expert_timeout_ms=5000.0,
            total_timeout_ms=30000.0,
        ),
        ner_service=mock_ner_service,
        router=mock_router,
        gating=mock_gating,
        synthesizer=mock_synthesizer,
    )
    # Register mock expert
    orch.register_expert("literal", mock_expert)
    return orch


# =============================================================================
# Tests: Initialization
# =============================================================================


class TestOrchestratorInit:
    """Tests for orchestrator initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        orch = PipelineOrchestrator()

        assert orch.config is not None
        assert orch.config.expert_timeout_ms == 30000.0
        assert orch.config.parallel_execution is True

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = OrchestratorConfig(
            expert_timeout_ms=10000.0,
            parallel_execution=False,
        )
        orch = PipelineOrchestrator(config=config)

        assert orch.config.expert_timeout_ms == 10000.0
        assert orch.config.parallel_execution is False

    def test_register_expert(self):
        """Test expert registration."""
        orch = PipelineOrchestrator()
        mock_expert = MagicMock()
        mock_expert.expert_type = "literal"

        orch.register_expert("literal", mock_expert)

        assert "literal" in orch._experts
        assert orch._experts["literal"] == mock_expert

    def test_set_llm_service(self):
        """Test setting LLM service."""
        orch = PipelineOrchestrator()
        mock_llm = MagicMock()

        orch.set_llm_service(mock_llm)

        assert orch._llm_service == mock_llm


# =============================================================================
# Tests: Request Validation
# =============================================================================


class TestRequestValidation:
    """Tests for request validation."""

    @pytest.mark.asyncio
    async def test_empty_query_raises_error(self, orchestrator):
        """Test that empty query raises validation error."""
        request = PipelineRequest(query="")

        with pytest.raises(PipelineValidationError) as exc_info:
            await orchestrator.process_query(request)

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_whitespace_query_raises_error(self, orchestrator):
        """Test that whitespace-only query raises validation error."""
        request = PipelineRequest(query="   ")

        with pytest.raises(PipelineValidationError) as exc_info:
            await orchestrator.process_query(request)

        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_too_long_query_raises_error(self, orchestrator):
        """Test that too long query raises validation error."""
        request = PipelineRequest(query="a" * 10001)

        with pytest.raises(PipelineValidationError) as exc_info:
            await orchestrator.process_query(request)

        assert "too long" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_profile_raises_error(self, orchestrator):
        """Test that invalid profile raises validation error."""
        request = PipelineRequest(
            query="Test query",
            user_profile="invalid_profile",
        )

        with pytest.raises(PipelineValidationError) as exc_info:
            await orchestrator.process_query(request)

        assert "user_profile" in str(exc_info.value).lower()


# =============================================================================
# Tests: Pipeline Execution
# =============================================================================


class TestPipelineExecution:
    """Tests for pipeline execution."""

    @pytest.mark.asyncio
    async def test_successful_pipeline_execution(self, orchestrator):
        """Test successful end-to-end pipeline execution."""
        request = PipelineRequest(
            query="Cos'è la risoluzione del contratto?",
            user_profile="ricerca",
        )

        result = await orchestrator.process_query(request)

        assert isinstance(result, PipelineResult)
        assert result.success is True
        assert result.error is None
        assert result.response is not None
        assert result.trace is not None
        assert result.metrics is not None

    @pytest.mark.asyncio
    async def test_pipeline_generates_trace_id(self, orchestrator):
        """Test that pipeline generates trace_id if not provided."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        assert result.trace.trace_id is not None
        assert len(result.trace.trace_id) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_pipeline_uses_provided_trace_id(self, orchestrator):
        """Test that pipeline uses provided trace_id."""
        request = PipelineRequest(
            query="Test query",
            trace_id="custom-trace-123",
        )

        result = await orchestrator.process_query(request)

        assert result.trace.trace_id == "custom-trace-123"

    @pytest.mark.asyncio
    async def test_pipeline_collects_metrics(self, orchestrator):
        """Test that pipeline collects metrics."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        assert result.metrics.total_time_ms > 0
        assert result.metrics.ner_time_ms >= 0
        assert result.metrics.routing_time_ms >= 0

    @pytest.mark.asyncio
    async def test_pipeline_collects_feedback_hooks(self, orchestrator):
        """Test that pipeline collects feedback hooks."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        assert len(result.feedback_hooks) > 0
        hook_types = [fh.feedback_type for fh in result.feedback_hooks]
        assert "F3" in hook_types or "F7" in hook_types

    @pytest.mark.asyncio
    async def test_pipeline_trace_contains_all_stages(self, orchestrator):
        """Test that trace contains all stage data."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        assert result.trace.ner_result is not None
        assert result.trace.routing_decision is not None
        assert result.trace.gating_result is not None
        assert result.trace.synthesis_result is not None


# =============================================================================
# Tests: Expert Execution
# =============================================================================


class TestExpertExecution:
    """Tests for expert execution."""

    @pytest.mark.asyncio
    async def test_experts_activated_tracked(self, orchestrator):
        """Test that activated experts are tracked in metrics."""
        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        assert "literal" in result.metrics.experts_activated

    @pytest.mark.asyncio
    async def test_expert_bypass(self, orchestrator, mock_expert):
        """Test that bypassed experts are skipped."""
        # Register more experts
        for exp_type in ["systemic", "principles", "precedent"]:
            mock_exp = AsyncMock()
            mock_exp.analyze = AsyncMock(return_value=mock_expert.analyze.return_value)
            orchestrator.register_expert(exp_type, mock_exp)

        request = PipelineRequest(
            query="Test query",
            bypass_experts=["systemic", "precedent"],
        )

        result = await orchestrator.process_query(request)

        assert "systemic" in result.metrics.experts_skipped
        assert "precedent" in result.metrics.experts_skipped

    @pytest.mark.asyncio
    async def test_weight_threshold_skips_low_weight_experts(self, orchestrator):
        """Test that experts below weight threshold are skipped."""
        # Routing gives precedent low weight (0.10)
        # With threshold 0.1, it should still be considered
        # But if we set threshold higher, it should be skipped
        orchestrator.config.min_confidence_threshold = 0.15

        request = PipelineRequest(query="Test query")

        result = await orchestrator.process_query(request)

        # Precedent (weight 0.10) should be skipped
        assert "precedent" in result.metrics.experts_skipped


# =============================================================================
# Tests: Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_expert_failure_handled_gracefully(
        self,
        mock_ner_service,
        mock_router,
        mock_gating,
        mock_synthesizer,
        mock_ner_result,
        mock_routing_decision,
    ):
        """Test that expert failure doesn't crash pipeline."""
        # Create expert that fails
        failing_expert = AsyncMock()
        failing_expert.analyze = AsyncMock(side_effect=Exception("Expert error"))

        orch = PipelineOrchestrator(
            ner_service=mock_ner_service,
            router=mock_router,
            gating=mock_gating,
            synthesizer=mock_synthesizer,
        )
        orch.register_expert("literal", failing_expert)

        request = PipelineRequest(query="Test query")

        # Should not raise, should return degraded result
        result = await orch.process_query(request)

        assert result.metrics.degraded is True
        assert "literal" in result.metrics.experts_failed

    @pytest.mark.asyncio
    async def test_pipeline_timeout_handled(self, mock_ner_service):
        """Test that total pipeline timeout is handled."""
        # Create NER service that takes too long
        async def slow_ner(*args, **kwargs):
            await asyncio.sleep(10)  # 10 seconds
            return mock_ner_service.extract.return_value

        slow_ner_service = AsyncMock()
        slow_ner_service.extract = slow_ner

        orch = PipelineOrchestrator(
            config=OrchestratorConfig(total_timeout_ms=100),  # 100ms timeout
            ner_service=slow_ner_service,
        )

        request = PipelineRequest(query="Test query")

        with pytest.raises(PipelineTimeoutError):
            await orch.process_query(request)


# =============================================================================
# Tests: Degradation
# =============================================================================


class TestDegradation:
    """Tests for graceful degradation."""

    @pytest.mark.asyncio
    async def test_degradation_reason_set_on_failure(
        self,
        mock_ner_service,
        mock_router,
        mock_gating,
        mock_synthesizer,
    ):
        """Test that degradation reason is set when experts fail."""
        # Create expert that fails
        failing_expert = AsyncMock()
        failing_expert.analyze = AsyncMock(side_effect=Exception("Error"))

        orch = PipelineOrchestrator(
            ner_service=mock_ner_service,
            router=mock_router,
            gating=mock_gating,
            synthesizer=mock_synthesizer,
        )
        orch.register_expert("literal", failing_expert)

        request = PipelineRequest(query="Test query")
        result = await orch.process_query(request)

        assert result.metrics.degraded is True
        assert result.metrics.degradation_reason is not None
        assert "literal" in result.metrics.degradation_reason


# =============================================================================
# Tests: Circuit Breaker Integration
# =============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    def test_get_circuit_breaker_stats(self, orchestrator):
        """Test getting circuit breaker statistics."""
        stats = orchestrator.get_circuit_breaker_stats()

        assert isinstance(stats, dict)

    def test_reset_circuit_breakers(self, orchestrator):
        """Test resetting circuit breakers."""
        # Should not raise
        orchestrator.reset_circuit_breakers()
