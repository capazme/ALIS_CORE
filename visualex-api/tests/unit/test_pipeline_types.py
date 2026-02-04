"""
Unit tests for Pipeline Types (Story 5.0 Task 1).

Tests data structures: PipelineRequest, PipelineTrace, PipelineMetrics,
PipelineResult, OrchestratorConfig, ExpertExecution.
"""

import json
import pytest
from datetime import datetime

from visualex.experts.pipeline_types import (
    PipelineRequest,
    PipelineTrace,
    PipelineMetrics,
    PipelineResult,
    OrchestratorConfig,
    ExpertExecution,
    PipelineStage,
    PipelineError,
    PipelineValidationError,
    PipelineTimeoutError,
    ExpertExecutionError,
    generate_trace_id,
)
from visualex.experts.base import FeedbackHook


class TestPipelineRequest:
    """Tests for PipelineRequest dataclass."""

    def test_create_minimal_request(self):
        """Test creating request with only required fields."""
        request = PipelineRequest(query="Cos'è la risoluzione?")

        assert request.query == "Cos'è la risoluzione?"
        assert request.user_profile == "ricerca"  # Default
        assert request.trace_id is None
        assert request.user_id is None
        assert request.override_weights is None
        assert request.bypass_experts is None

    def test_create_full_request(self):
        """Test creating request with all fields."""
        request = PipelineRequest(
            query="Art. 1453 c.c.",
            user_profile="analisi",
            trace_id="test-trace-123",
            user_id="user-456",
            override_weights={"literal": 0.5, "systemic": 0.3},
            bypass_experts=["precedent"],
            context={"source": "api"},
        )

        assert request.query == "Art. 1453 c.c."
        assert request.user_profile == "analisi"
        assert request.trace_id == "test-trace-123"
        assert request.user_id == "user-456"
        assert request.override_weights == {"literal": 0.5, "systemic": 0.3}
        assert request.bypass_experts == ["precedent"]
        assert request.context == {"source": "api"}

    def test_to_dict(self):
        """Test serialization to dictionary."""
        request = PipelineRequest(
            query="Test query",
            user_profile="consulenza",
        )
        d = request.to_dict()

        assert d["query"] == "Test query"
        assert d["user_profile"] == "consulenza"
        assert "trace_id" in d
        assert "override_weights" in d


class TestExpertExecution:
    """Tests for ExpertExecution dataclass."""

    def test_create_successful_execution(self):
        """Test creating successful expert execution."""
        now = datetime.now()
        execution = ExpertExecution(
            expert_type="literal",
            started_at=now,
            completed_at=now,
            duration_ms=1500.0,
            success=True,
            tokens_used=500,
            confidence=0.85,
            output={"interpretation": "Test interpretation"},
        )

        assert execution.expert_type == "literal"
        assert execution.success is True
        assert execution.duration_ms == 1500.0
        assert execution.tokens_used == 500
        assert execution.confidence == 0.85
        assert execution.error is None

    def test_create_failed_execution(self):
        """Test creating failed expert execution."""
        execution = ExpertExecution(
            expert_type="systemic",
            success=False,
            error="Connection timeout",
            circuit_breaker_state="open",
        )

        assert execution.success is False
        assert execution.error == "Connection timeout"
        assert execution.circuit_breaker_state == "open"

    def test_create_skipped_execution(self):
        """Test creating skipped expert execution."""
        execution = ExpertExecution(
            expert_type="precedent",
            skipped=True,
            skip_reason="Weight below threshold",
        )

        assert execution.skipped is True
        assert execution.skip_reason == "Weight below threshold"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        now = datetime.now()
        execution = ExpertExecution(
            expert_type="principles",
            started_at=now,
            completed_at=now,
            duration_ms=2000.0,
            success=True,
            confidence=0.75,
        )
        d = execution.to_dict()

        assert d["expert_type"] == "principles"
        assert d["duration_ms"] == 2000.0
        assert d["success"] is True
        assert d["started_at"] == now.isoformat()


class TestPipelineTrace:
    """Tests for PipelineTrace dataclass."""

    def test_create_trace(self):
        """Test creating pipeline trace."""
        trace = PipelineTrace(
            trace_id="trace-123",
            query_text="Test query",
        )

        assert trace.trace_id == "trace-123"
        assert trace.query_text == "Test query"
        assert isinstance(trace.timestamp, datetime)
        assert trace.ner_result == {}
        assert trace.total_time_ms == 0.0

    def test_trace_with_full_data(self):
        """Test trace with all stage data."""
        trace = PipelineTrace(
            trace_id="trace-456",
            query_text="Cos'è il contratto?",
            ner_result={"entities": [{"text": "contratto", "type": "LEGAL_CONCEPT"}]},
            routing_decision={"query_type": "DEFINITION", "weights": {"literal": 0.6}},
            expert_executions=[
                {"expert_type": "literal", "success": True, "duration_ms": 1000}
            ],
            gating_result={"synthesis": "Test synthesis"},
            synthesis_result={"main_answer": "Test answer"},
            total_time_ms=5000.0,
            stage_times_ms={"ner": 100, "routing": 50, "expert_literal": 1000},
            total_tokens=1500,
        )

        assert trace.total_tokens == 1500
        assert len(trace.expert_executions) == 1
        assert trace.stage_times_ms["ner"] == 100

    def test_to_dict(self):
        """Test serialization to dictionary."""
        trace = PipelineTrace(
            trace_id="trace-789",
            query_text="Test",
            total_time_ms=3000.5,
        )
        d = trace.to_dict()

        assert d["trace_id"] == "trace-789"
        assert d["total_time_ms"] == 3000.5
        assert "timestamp" in d

    def test_to_json(self):
        """Test JSON serialization for PostgreSQL storage."""
        trace = PipelineTrace(
            trace_id="trace-json",
            query_text="Test JSON",
            total_tokens=500,
        )
        json_str = trace.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["trace_id"] == "trace-json"
        assert parsed["total_tokens"] == 500

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "trace_id": "trace-restore",
            "query_text": "Restored query",
            "timestamp": "2026-02-02T12:00:00",
            "total_time_ms": 2500.0,
            "total_tokens": 800,
            "ner_result": {},
            "routing_decision": {},
            "expert_executions": [],
            "gating_result": {},
            "synthesis_result": {},
            "stage_times_ms": {},
        }
        trace = PipelineTrace.from_dict(data)

        assert trace.trace_id == "trace-restore"
        assert trace.total_time_ms == 2500.0
        assert trace.total_tokens == 800


class TestPipelineMetrics:
    """Tests for PipelineMetrics dataclass."""

    def test_create_empty_metrics(self):
        """Test creating metrics with defaults."""
        metrics = PipelineMetrics()

        assert metrics.total_time_ms == 0.0
        assert metrics.experts_activated == []
        assert metrics.experts_failed == []
        assert metrics.degraded is False

    def test_create_full_metrics(self):
        """Test creating metrics with all data."""
        metrics = PipelineMetrics(
            total_time_ms=10000.0,
            ner_time_ms=100.0,
            routing_time_ms=50.0,
            expert_times_ms={
                "literal": 2000.0,
                "systemic": 2500.0,
                "principles": 2200.0,
                "precedent": 0.0,
            },
            gating_time_ms=500.0,
            synthesis_time_ms=1000.0,
            total_tokens=3000,
            experts_activated=["literal", "systemic", "principles"],
            experts_failed=["precedent"],
            experts_skipped=[],
            circuit_breaker_events=[{"expert": "precedent", "state": "open"}],
            degraded=True,
            degradation_reason="1 expert failed",
        )

        assert metrics.total_time_ms == 10000.0
        assert len(metrics.experts_activated) == 3
        assert "precedent" in metrics.experts_failed
        assert metrics.degraded is True

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = PipelineMetrics(
            total_time_ms=5000.123,
            expert_times_ms={"literal": 1000.456},
        )
        d = metrics.to_dict()

        assert d["total_time_ms"] == 5000.12  # Rounded to 2 decimals
        assert d["expert_times_ms"]["literal"] == 1000.46


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""

    def test_create_successful_result(self):
        """Test creating successful pipeline result."""
        trace = PipelineTrace(trace_id="result-trace", query_text="Test")
        metrics = PipelineMetrics(total_time_ms=5000.0)
        feedback_hooks = [
            FeedbackHook(
                feedback_type="F3",
                expert_type="literal",
                response_id="result-trace",
            )
        ]

        # Mock response object
        class MockResponse:
            def to_dict(self):
                return {"main_answer": "Test answer"}

        result = PipelineResult(
            response=MockResponse(),
            trace=trace,
            metrics=metrics,
            feedback_hooks=feedback_hooks,
            success=True,
        )

        assert result.success is True
        assert result.error is None
        assert len(result.feedback_hooks) == 1

    def test_create_failed_result(self):
        """Test creating failed pipeline result."""
        trace = PipelineTrace(trace_id="failed-trace", query_text="Test")
        metrics = PipelineMetrics()

        result = PipelineResult(
            response=None,
            trace=trace,
            metrics=metrics,
            success=False,
            error="Pipeline timeout",
        )

        assert result.success is False
        assert result.error == "Pipeline timeout"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        trace = PipelineTrace(trace_id="dict-trace", query_text="Test")
        metrics = PipelineMetrics()

        class MockResponse:
            def to_dict(self):
                return {"main_answer": "Answer"}

        result = PipelineResult(
            response=MockResponse(),
            trace=trace,
            metrics=metrics,
        )
        d = result.to_dict()

        assert d["success"] is True
        assert "response" in d
        assert "trace" in d
        assert "metrics" in d


class TestOrchestratorConfig:
    """Tests for OrchestratorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OrchestratorConfig()

        assert config.expert_timeout_ms == 30000.0
        assert config.total_timeout_ms == 120000.0
        assert config.parallel_execution is True
        assert config.min_confidence_threshold == 0.2
        assert config.enable_tracing is True
        assert config.enable_metrics is True
        assert config.degradation_confidence_penalty == 0.10
        assert config.llm_provider == "openrouter"

    def test_custom_config(self):
        """Test custom configuration values."""
        config = OrchestratorConfig(
            expert_timeout_ms=15000.0,
            parallel_execution=False,
            min_confidence_threshold=0.2,
            llm_provider="anthropic",
            llm_model="claude-3-sonnet",
        )

        assert config.expert_timeout_ms == 15000.0
        assert config.parallel_execution is False
        assert config.min_confidence_threshold == 0.2
        assert config.llm_provider == "anthropic"
        assert config.llm_model == "claude-3-sonnet"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = OrchestratorConfig(
            expert_timeout_ms=20000.0,
        )
        d = config.to_dict()

        assert d["expert_timeout_ms"] == 20000.0
        assert "parallel_execution" in d
        assert "enable_tracing" in d


class TestPipelineStage:
    """Tests for PipelineStage constants."""

    def test_stage_names(self):
        """Test pipeline stage name constants."""
        assert PipelineStage.NER == "ner"
        assert PipelineStage.ROUTING == "routing"
        assert PipelineStage.EXPERT_LITERAL == "expert_literal"
        assert PipelineStage.EXPERT_SYSTEMIC == "expert_systemic"
        assert PipelineStage.EXPERT_PRINCIPLES == "expert_principles"
        assert PipelineStage.EXPERT_PRECEDENT == "expert_precedent"
        assert PipelineStage.GATING == "gating"
        assert PipelineStage.SYNTHESIS == "synthesis"


class TestPipelineExceptions:
    """Tests for pipeline exception classes."""

    def test_pipeline_error(self):
        """Test base pipeline error."""
        error = PipelineError("Test error", stage="ner")
        assert str(error) == "Test error"
        assert error.stage == "ner"

    def test_pipeline_validation_error(self):
        """Test validation error."""
        error = PipelineValidationError("Invalid query", stage="validation")
        assert isinstance(error, PipelineError)
        assert "Invalid query" in str(error)

    def test_pipeline_timeout_error(self):
        """Test timeout error."""
        error = PipelineTimeoutError("Pipeline exceeded 2 minutes")
        assert isinstance(error, PipelineError)

    def test_expert_execution_error(self):
        """Test expert execution error with details."""
        error = ExpertExecutionError(
            message="2 experts failed",
            experts_failed=["systemic", "precedent"],
            partial_results=[{"expert_type": "literal", "success": True}],
        )

        assert isinstance(error, PipelineError)
        assert error.experts_failed == ["systemic", "precedent"]
        assert len(error.partial_results) == 1
        assert error.stage == "expert_execution"


class TestGenerateTraceId:
    """Tests for trace ID generation."""

    def test_generates_uuid(self):
        """Test that trace ID is a valid UUID."""
        trace_id = generate_trace_id()

        # Should be a string
        assert isinstance(trace_id, str)
        # Should be UUID format (36 characters with hyphens)
        assert len(trace_id) == 36
        assert trace_id.count("-") == 4

    def test_generates_unique_ids(self):
        """Test that each call generates a unique ID."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100  # All unique
