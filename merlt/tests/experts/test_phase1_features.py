"""
Tests for Phase 1 features with zero existing coverage.

Tests per:
- FeedbackHook (F3-F7 dataclass)
- CircuitBreaker (state machine)
- create_unavailable_response
- CircuitBreakerRegistry
- AggregationMethod + GatingConfig
- DEFAULT_EXPERT_WEIGHTS
- USER_PROFILE_MODIFIERS
- ExpertContribution
- UserProfile + AccordionSection
- PipelineTypes (PipelineRequest, PipelineTrace, PipelineResult)
- OrchestratorConfig new fields
"""

import asyncio
import pytest
from datetime import datetime

from merlt.experts.base import (
    ExpertResponse,
    FeedbackHook,
)
from merlt.experts.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    CircuitOpenError,
    CircuitState,
    create_unavailable_response,
)
from merlt.experts.gating import (
    AggregationMethod,
    ExpertContribution,
    GatingConfig,
    DEFAULT_EXPERT_WEIGHTS,
    USER_PROFILE_MODIFIERS,
)
from merlt.experts.synthesizer import (
    AccordionSection,
    UserProfile,
)
from merlt.experts.pipeline_types import (
    PipelineRequest,
    PipelineTrace,
    PipelineResult,
    PipelineMetrics,
    OrchestratorConfig as PipelineOrchestratorConfig,
)
from merlt.experts.orchestrator import OrchestratorConfig


# ============================================================================
# FeedbackHook Tests
# ============================================================================


class TestFeedbackHook:
    """Test per FeedbackHook F3-F7 dataclass."""

    def test_create_f3(self):
        """Crea hook F3 con correction_options e context_snapshot."""
        hook = FeedbackHook(
            feedback_type="F3",
            expert_type="literal",
            response_id="trace_001",
            enabled=True,
            correction_options={
                "interpretation_quality": ["accurate", "inaccurate", "partially_correct"],
            },
            context_snapshot={
                "confidence": 0.8,
                "sources_used": 3,
            },
        )

        assert hook.feedback_type == "F3"
        assert hook.expert_type == "literal"
        assert hook.response_id == "trace_001"
        assert hook.enabled is True
        assert "interpretation_quality" in hook.correction_options
        assert hook.context_snapshot["confidence"] == 0.8

    def test_to_dict(self):
        """Serializzazione to_dict."""
        hook = FeedbackHook(
            feedback_type="F7",
            expert_type="gating",
            response_id="trace_002",
            correction_options={"weight": ["ok", "adjust"]},
            context_snapshot={"expert_count": 4},
        )

        data = hook.to_dict()

        assert data["feedback_type"] == "F7"
        assert data["expert_type"] == "gating"
        assert data["response_id"] == "trace_002"
        assert data["enabled"] is True
        assert data["correction_options"]["weight"] == ["ok", "adjust"]
        assert data["context_snapshot"]["expert_count"] == 4

    def test_expert_response_carries_hook(self):
        """ExpertResponse porta il FeedbackHook correttamente."""
        hook = FeedbackHook(
            feedback_type="F3",
            expert_type="literal",
            response_id="trace_003",
        )

        response = ExpertResponse(
            expert_type="literal",
            interpretation="Test interpretation",
            confidence=0.8,
            feedback_hook=hook,
        )

        assert response.feedback_hook is not None
        assert response.feedback_hook.feedback_type == "F3"

        data = response.to_dict()
        assert data["feedback_hook"]["feedback_type"] == "F3"


# ============================================================================
# CircuitBreaker Tests
# ============================================================================


class TestCircuitBreaker:
    """Test per CircuitBreaker state machine."""

    def test_initial_state_closed(self):
        """Stato iniziale CLOSED."""
        breaker = CircuitBreaker(name="test_expert")

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    def test_stays_closed_after_successes(self):
        """Rimane CLOSED dopo successi."""
        breaker = CircuitBreaker(name="test_expert")

        breaker.record_success()
        breaker.record_success()
        breaker.record_success()

        assert breaker.state == CircuitState.CLOSED

    def test_opens_after_threshold_failures(self):
        """Apre dopo N fallimenti (failure_threshold)."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker(name="test_expert", config=config)

        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail 1"))
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail 2"))
        assert breaker.state == CircuitState.CLOSED

        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail 3"))
        assert breaker.state == CircuitState.OPEN

    def test_open_rejects_calls(self):
        """OPEN rifiuta chiamate con can_execute() == False."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout_seconds=9999)
        breaker = CircuitBreaker(name="test_expert", config=config)

        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail 1"))
        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail 2"))

        assert breaker.state == CircuitState.OPEN
        assert breaker.can_execute() is False

    @pytest.mark.asyncio
    async def test_open_raises_circuit_open_error(self):
        """OPEN solleva CircuitOpenError tramite context manager."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout_seconds=9999)
        breaker = CircuitBreaker(name="test_expert", config=config)

        # Trip the breaker
        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail"))
        assert breaker.is_open

        with pytest.raises(CircuitOpenError):
            async with breaker:
                pass  # Should not reach here

    def test_reset_returns_to_closed(self):
        """reset() riporta a CLOSED."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker(name="test_expert", config=config)

        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail"))
        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.can_execute() is True

    def test_get_stats(self):
        """get_stats() restituisce CircuitBreakerStats."""
        breaker = CircuitBreaker(name="test_expert")

        breaker.record_success()
        breaker.record_failure(exc_type=RuntimeError, exc_val=RuntimeError("fail"))

        stats = breaker.get_stats()

        assert isinstance(stats, CircuitBreakerStats)
        assert stats.name == "test_expert"
        assert stats.state == CircuitState.CLOSED
        assert stats.total_successes == 1
        assert stats.total_failures == 1


# ============================================================================
# create_unavailable_response Tests
# ============================================================================


class TestCreateUnavailableResponse:
    """Test per create_unavailable_response."""

    def test_returns_expert_response(self):
        """Restituisce ExpertResponse (non dict)."""
        response = create_unavailable_response("literal", trace_id="test_trace")

        assert isinstance(response, ExpertResponse)

    def test_correct_fields(self):
        """expert_type, trace_id, confidence=0.0 corretti."""
        response = create_unavailable_response("systemic", trace_id="trace_123")

        assert response.expert_type == "systemic"
        assert response.trace_id == "trace_123"
        assert response.confidence == 0.0

    def test_is_degraded_metadata(self):
        """metadata['is_degraded'] == True."""
        response = create_unavailable_response("literal")

        assert response.metadata.get("is_degraded") is True


# ============================================================================
# CircuitBreakerRegistry Tests
# ============================================================================


class TestCircuitBreakerRegistry:
    """Test per CircuitBreakerRegistry."""

    def test_get_or_create_returns_same_instance(self):
        """get_or_create() restituisce stessa istanza per stesso nome."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("literal_expert")
        cb2 = registry.get_or_create("literal_expert")

        assert cb1 is cb2

    def test_get_or_create_different_names(self):
        """get_or_create() restituisce istanze diverse per nomi diversi."""
        registry = CircuitBreakerRegistry()

        cb1 = registry.get_or_create("literal_expert")
        cb2 = registry.get_or_create("systemic_expert")

        assert cb1 is not cb2

    def test_get_all_stats(self):
        """get_all_stats() include tutti i breaker registrati."""
        registry = CircuitBreakerRegistry()

        registry.get_or_create("expert_a")
        registry.get_or_create("expert_b")

        stats = registry.get_all_stats()

        assert "expert_a" in stats
        assert "expert_b" in stats
        assert isinstance(stats["expert_a"], CircuitBreakerStats)


# ============================================================================
# AggregationMethod + GatingConfig Tests
# ============================================================================


class TestAggregationMethod:
    """Test per AggregationMethod enum."""

    def test_values(self):
        """Valori enum: WEIGHTED_AVERAGE, BEST_CONFIDENCE, CONSENSUS, ENSEMBLE."""
        assert AggregationMethod.WEIGHTED_AVERAGE == "weighted_average"
        assert AggregationMethod.BEST_CONFIDENCE == "best_confidence"
        assert AggregationMethod.CONSENSUS == "consensus"
        assert AggregationMethod.ENSEMBLE == "ensemble"

    def test_all_four_members(self):
        """Exactly 4 members."""
        assert len(AggregationMethod) == 4


class TestGatingConfig:
    """Test per GatingConfig defaults."""

    def test_defaults(self):
        """GatingConfig() defaults corretti."""
        config = GatingConfig()

        assert config.method == AggregationMethod.WEIGHTED_AVERAGE
        assert config.confidence_divergence_threshold == 0.4
        assert config.source_overlap_threshold == 0.2
        assert config.max_legal_sources == 10
        assert config.max_reasoning_steps == 15
        assert config.enable_f7_feedback is True

    def test_custom(self):
        """GatingConfig con parametri custom."""
        config = GatingConfig(
            method=AggregationMethod.ENSEMBLE,
            confidence_divergence_threshold=0.6,
        )

        assert config.method == AggregationMethod.ENSEMBLE
        assert config.confidence_divergence_threshold == 0.6


# ============================================================================
# DEFAULT_EXPERT_WEIGHTS Tests
# ============================================================================


class TestDefaultExpertWeights:
    """Test per DEFAULT_EXPERT_WEIGHTS."""

    def test_sum_equals_one(self):
        """Somma pesi = 1.0."""
        total = sum(DEFAULT_EXPERT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.001

    def test_immutable(self):
        """Immutabile (TypeError su assegnamento)."""
        with pytest.raises(TypeError):
            DEFAULT_EXPERT_WEIGHTS["literal"] = 0.99

    def test_contains_all_experts(self):
        """Contiene tutti e 4 gli expert."""
        assert "literal" in DEFAULT_EXPERT_WEIGHTS
        assert "systemic" in DEFAULT_EXPERT_WEIGHTS
        assert "principles" in DEFAULT_EXPERT_WEIGHTS
        assert "precedent" in DEFAULT_EXPERT_WEIGHTS


# ============================================================================
# USER_PROFILE_MODIFIERS Tests
# ============================================================================


class TestUserProfileModifiers:
    """Test per USER_PROFILE_MODIFIERS."""

    def test_profiles_present(self):
        """Profili 'analysis', 'quick', 'academic' presenti."""
        assert "analysis" in USER_PROFILE_MODIFIERS
        assert "quick" in USER_PROFILE_MODIFIERS
        assert "academic" in USER_PROFILE_MODIFIERS

    def test_each_profile_has_four_experts(self):
        """Ogni profilo ha tutti e 4 gli expert."""
        for profile_name, modifiers in USER_PROFILE_MODIFIERS.items():
            assert "literal" in modifiers, f"{profile_name} missing literal"
            assert "systemic" in modifiers, f"{profile_name} missing systemic"
            assert "principles" in modifiers, f"{profile_name} missing principles"
            assert "precedent" in modifiers, f"{profile_name} missing precedent"


# ============================================================================
# ExpertContribution Tests
# ============================================================================


class TestExpertContribution:
    """Test per ExpertContribution dataclass."""

    def test_create_and_serialize(self):
        """Creazione e serializzazione."""
        contrib = ExpertContribution(
            expert_type="literal",
            interpretation="Test interpretation text",
            confidence=0.85,
            weight=0.5,
            weighted_confidence=0.425,
        )

        assert contrib.expert_type == "literal"
        assert contrib.confidence == 0.85

        data = contrib.to_dict()
        assert data["expert_type"] == "literal"
        assert data["confidence"] == 0.85
        assert data["weight"] == 0.5
        assert data["weighted_confidence"] == 0.425

    def test_interpretation_preview_truncated(self):
        """interpretation_preview troncato a 200 char."""
        long_text = "A" * 300
        contrib = ExpertContribution(
            expert_type="literal",
            interpretation=long_text,
            confidence=0.8,
            weight=1.0,
            weighted_confidence=0.8,
        )

        data = contrib.to_dict()
        assert len(data["interpretation_preview"]) == 200

    def test_selected_flag(self):
        """selected flag per best_confidence."""
        contrib = ExpertContribution(
            expert_type="systemic",
            interpretation="Best",
            confidence=0.95,
            weight=1.0,
            weighted_confidence=0.95,
            selected=True,
        )

        assert contrib.selected is True
        assert contrib.to_dict()["selected"] is True


# ============================================================================
# UserProfile + AccordionSection Tests
# ============================================================================


class TestUserProfile:
    """Test per UserProfile enum."""

    def test_enum_values(self):
        """Enum values: CONSULENZA, RICERCA, ANALISI, CONTRIBUTORE."""
        assert UserProfile.CONSULENZA == "consulenza"
        assert UserProfile.RICERCA == "ricerca"
        assert UserProfile.ANALISI == "analisi"
        assert UserProfile.CONTRIBUTORE == "contributore"


class TestAccordionSection:
    """Test per AccordionSection dataclass."""

    def test_create_with_to_dict(self):
        """Creazione con to_dict()."""
        section = AccordionSection(
            expert_type="literal",
            header="Interpretazione Letterale",
            content="Contenuto della sezione",
            confidence=0.85,
            is_expanded=True,
        )

        assert section.expert_type == "literal"
        assert section.is_expanded is True

        data = section.to_dict()
        assert data["expert_type"] == "literal"
        assert data["header"] == "Interpretazione Letterale"
        assert data["content"] == "Contenuto della sezione"
        assert data["confidence"] == 0.85
        assert data["is_expanded"] is True

    def test_default_not_expanded(self):
        """Default is_expanded = False."""
        section = AccordionSection(
            expert_type="systemic",
            header="Test",
            content="Test content",
            confidence=0.7,
        )

        assert section.is_expanded is False


# ============================================================================
# Pipeline Types Tests
# ============================================================================


class TestPipelineRequest:
    """Test per PipelineRequest."""

    def test_create_and_to_dict(self):
        """PipelineRequest creazione e to_dict()."""
        req = PipelineRequest(
            query="Cos'e' la legittima difesa?",
            user_profile="ricerca",
            trace_id="trace_001",
        )

        assert req.query == "Cos'e' la legittima difesa?"
        assert req.user_profile == "ricerca"

        data = req.to_dict()
        assert data["query"] == "Cos'e' la legittima difesa?"
        assert data["trace_id"] == "trace_001"

    def test_defaults(self):
        """Defaults: user_profile='ricerca'."""
        req = PipelineRequest(query="Test")

        assert req.user_profile == "ricerca"
        assert req.trace_id is None
        assert req.override_weights is None


class TestPipelineTrace:
    """Test per PipelineTrace."""

    def test_roundtrip_to_dict_from_dict(self):
        """PipelineTrace roundtrip to_dict() -> from_dict()."""
        trace = PipelineTrace(
            trace_id="trace_001",
            query_text="Test query",
            ner_result={"entities": ["art_52"]},
            routing_decision={"query_type": "definitional"},
            total_time_ms=150.5,
            stage_times_ms={"ner": 10.2, "routing": 5.1},
            total_tokens=500,
        )

        data = trace.to_dict()
        restored = PipelineTrace.from_dict(data)

        assert restored.trace_id == "trace_001"
        assert restored.query_text == "Test query"
        assert restored.ner_result == {"entities": ["art_52"]}
        assert restored.routing_decision == {"query_type": "definitional"}
        assert restored.total_time_ms == 150.5
        assert restored.total_tokens == 500

    def test_to_json(self):
        """to_json() serializes to valid JSON string."""
        trace = PipelineTrace(
            trace_id="trace_002",
            query_text="Test",
        )

        import json
        json_str = trace.to_json()
        data = json.loads(json_str)

        assert data["trace_id"] == "trace_002"


class TestPipelineResult:
    """Test per PipelineResult."""

    def test_with_success_flag(self):
        """PipelineResult con success flag."""
        trace = PipelineTrace(trace_id="t1", query_text="Test")
        metrics = PipelineMetrics(total_time_ms=100.0)

        result = PipelineResult(
            response={"synthesis": "Test response"},
            trace=trace,
            metrics=metrics,
            success=True,
        )

        assert result.success is True
        assert result.error is None

    def test_with_failure(self):
        """PipelineResult con errore."""
        trace = PipelineTrace(trace_id="t2", query_text="Test")
        metrics = PipelineMetrics(total_time_ms=50.0)

        result = PipelineResult(
            response=None,
            trace=trace,
            metrics=metrics,
            success=False,
            error="Expert timeout",
        )

        assert result.success is False
        assert result.error == "Expert timeout"


# ============================================================================
# OrchestratorConfig Tests
# ============================================================================


class TestOrchestratorConfigNewFields:
    """Test per OrchestratorConfig nuovi campi Phase 1."""

    def test_enable_circuit_breaker_default(self):
        """enable_circuit_breaker default True."""
        config = OrchestratorConfig()

        assert config.enable_circuit_breaker is True

    def test_expert_weight_threshold_default(self):
        """expert_weight_threshold default 0.1."""
        config = OrchestratorConfig()

        assert config.expert_weight_threshold == 0.1

    def test_custom_values(self):
        """Valori custom per nuovi campi."""
        config = OrchestratorConfig(
            enable_circuit_breaker=False,
            expert_weight_threshold=0.2,
        )

        assert config.enable_circuit_breaker is False
        assert config.expert_weight_threshold == 0.2


class TestPipelineOrchestratorConfig:
    """Test per pipeline_types.OrchestratorConfig."""

    def test_expert_weight_threshold_default(self):
        """expert_weight_threshold default 0.1 (pipeline_types version)."""
        config = PipelineOrchestratorConfig()

        assert config.expert_weight_threshold == 0.1

    def test_to_dict(self):
        """to_dict() includes all fields."""
        config = PipelineOrchestratorConfig()
        data = config.to_dict()

        assert "expert_weight_threshold" in data
        assert "parallel_execution" in data
        assert "enable_tracing" in data
