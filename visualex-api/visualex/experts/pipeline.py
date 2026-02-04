"""
Pipeline Orchestrator for MERL-T Analysis Pipeline.

Coordinates the complete expert analysis flow:
NER → Router → Experts (parallel) → Gating → Synthesis

Story 5.0: Expert Pipeline Orchestrator
"""

import asyncio
import time
import structlog
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..ner import NERService, ExtractionResult
from .base import (
    BaseExpert,
    ExpertContext,
    ExpertResponse,
    FeedbackHook,
    LLMService,
)
from .router import ExpertRouter, RoutingDecision, ExpertType
from .gating import GatingNetwork, AggregatedResponse
from .synthesizer import Synthesizer, SynthesizedResponse
from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerRegistry,
    CircuitBreakerConfig,
    CircuitOpenError,
    get_expert_circuit_breaker,
)
from .pipeline_types import (
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

log = structlog.get_logger()


class PipelineOrchestrator:
    """
    Central orchestrator for the MERL-T expert pipeline.

    Coordinates all Epic 4 components into a cohesive analysis flow:
    1. NER extraction (entities, confidence)
    2. Router decision (expert weights, query type)
    3. Expert dispatch (4 experts with weights, parallel execution)
    4. Gating aggregation (weighted synthesis)
    5. Final synthesis (profile-aware formatting)

    Returns PipelineResult containing:
    - response: SynthesizedResponse for UI
    - trace: Complete execution trace (Epic 5.1 requirement)
    - metrics: Timing and token usage per stage
    - feedback_hooks: F1-F7 feedback opportunities

    Example:
        >>> orchestrator = PipelineOrchestrator()
        >>> request = PipelineRequest(query="Cos'è la risoluzione del contratto?")
        >>> result = await orchestrator.process_query(request)
        >>> print(result.response.main_answer)
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        ner_service: Optional[NERService] = None,
        router: Optional[ExpertRouter] = None,
        experts: Optional[Dict[str, BaseExpert]] = None,
        gating: Optional[GatingNetwork] = None,
        synthesizer: Optional[Synthesizer] = None,
        llm_service: Optional[LLMService] = None,
        circuit_breaker_registry: Optional[CircuitBreakerRegistry] = None,
    ):
        """
        Initialize Pipeline Orchestrator.

        Args:
            config: Orchestrator configuration
            ner_service: NER service for entity extraction
            router: Expert router for weight assignment
            experts: Dictionary of expert instances by type
            gating: Gating network for aggregation
            synthesizer: Synthesizer for final response
            llm_service: LLM service for all components
            circuit_breaker_registry: Circuit breaker registry
        """
        self.config = config or OrchestratorConfig()

        # Initialize components (lazy init if not provided)
        self._ner_service = ner_service
        self._router = router
        self._experts = experts or {}
        self._gating = gating
        self._synthesizer = synthesizer
        self._llm_service = llm_service

        # Circuit breaker registry
        self._cb_registry = circuit_breaker_registry or CircuitBreakerRegistry.get_instance()

        log.info(
            "pipeline_orchestrator_initialized",
            config=self.config.to_dict(),
            experts_provided=list(self._experts.keys()) if self._experts else [],
        )

    @property
    def ner_service(self) -> NERService:
        """Get or create NER service."""
        if self._ner_service is None:
            self._ner_service = NERService()
        return self._ner_service

    @property
    def router(self) -> ExpertRouter:
        """Get or create router."""
        if self._router is None:
            self._router = ExpertRouter()
        return self._router

    @property
    def gating(self) -> GatingNetwork:
        """Get or create gating network."""
        if self._gating is None:
            self._gating = GatingNetwork(llm_service=self._llm_service)
        return self._gating

    @property
    def synthesizer(self) -> Synthesizer:
        """Get or create synthesizer."""
        if self._synthesizer is None:
            self._synthesizer = Synthesizer(llm_service=self._llm_service)
        return self._synthesizer

    def set_llm_service(self, llm_service: LLMService) -> None:
        """
        Set LLM service for all components.

        Args:
            llm_service: LLM service instance
        """
        self._llm_service = llm_service

        # Update components that need LLM
        if self._gating:
            self._gating.llm_service = llm_service
        if self._synthesizer:
            self._synthesizer.llm_service = llm_service
        for expert in self._experts.values():
            if hasattr(expert, "llm_service"):
                expert.llm_service = llm_service

    def register_expert(self, expert_type: str, expert: BaseExpert) -> None:
        """
        Register an expert instance.

        Args:
            expert_type: Expert type (literal, systemic, principles, precedent)
            expert: Expert instance
        """
        self._experts[expert_type] = expert
        log.info("expert_registered", expert_type=expert_type)

    async def process_query(self, request: PipelineRequest) -> PipelineResult:
        """
        Process a query through the complete pipeline.

        Args:
            request: Pipeline request with query and configuration

        Returns:
            PipelineResult with response, trace, metrics, and feedback hooks

        Raises:
            PipelineValidationError: If request is invalid
            PipelineTimeoutError: If pipeline exceeds total timeout
            PipelineError: For other pipeline failures
        """
        start_time = time.perf_counter()
        trace_id = request.trace_id or generate_trace_id()

        # Bind trace_id to all subsequent logs
        log_ctx = log.bind(trace_id=trace_id)
        log_ctx.info("pipeline_started", query_length=len(request.query))

        # Initialize trace and metrics
        trace = PipelineTrace(
            trace_id=trace_id,
            query_text=request.query,
            timestamp=datetime.now(),
        )
        metrics = PipelineMetrics()
        feedback_hooks: List[FeedbackHook] = []

        try:
            # Validate request
            self._validate_request(request)

            # Execute pipeline with total timeout
            timeout_sec = self.config.total_timeout_ms / 1000.0

            try:
                result = await asyncio.wait_for(
                    self._execute_pipeline(
                        request=request,
                        trace_id=trace_id,
                        trace=trace,
                        metrics=metrics,
                        feedback_hooks=feedback_hooks,
                        log_ctx=log_ctx,
                    ),
                    timeout=timeout_sec,
                )
                return result

            except asyncio.TimeoutError:
                elapsed = (time.perf_counter() - start_time) * 1000
                log_ctx.error(
                    "pipeline_timeout",
                    elapsed_ms=elapsed,
                    timeout_ms=self.config.total_timeout_ms,
                )
                raise PipelineTimeoutError(
                    f"Pipeline exceeded total timeout of {self.config.total_timeout_ms}ms",
                    stage="total",
                )

        except PipelineError:
            raise

        except Exception as e:
            elapsed = (time.perf_counter() - start_time) * 1000
            log_ctx.error(
                "pipeline_unexpected_error",
                error=str(e),
                elapsed_ms=elapsed,
            )

            # Return partial result with error
            metrics.total_time_ms = elapsed
            trace.total_time_ms = elapsed

            return PipelineResult(
                response=self._create_error_response(str(e), trace_id),
                trace=trace,
                metrics=metrics,
                feedback_hooks=feedback_hooks,
                success=False,
                error=str(e),
            )

    async def _execute_pipeline(
        self,
        request: PipelineRequest,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> PipelineResult:
        """
        Execute the pipeline stages.

        Returns:
            PipelineResult with complete execution results
        """
        start_time = time.perf_counter()

        # Stage 1: NER Extraction
        ner_result = await self._execute_ner(
            request.query,
            trace_id,
            trace,
            metrics,
            feedback_hooks,
            log_ctx,
        )

        # Stage 2: Expert Routing
        routing_decision = await self._execute_routing(
            request.query,
            ner_result,
            trace_id,
            trace,
            metrics,
            log_ctx,
        )

        # Stage 3: Build Expert Context
        expert_context = self._build_expert_context(
            request.query,
            ner_result,
            trace_id,
        )

        # Stage 4: Parallel Expert Execution
        expert_responses = await self._execute_experts_parallel(
            expert_context=expert_context,
            routing_decision=routing_decision,
            request=request,
            trace_id=trace_id,
            trace=trace,
            metrics=metrics,
            feedback_hooks=feedback_hooks,
            log_ctx=log_ctx,
        )

        # Stage 5: Gating Aggregation
        aggregated = await self._execute_gating(
            expert_responses=expert_responses,
            routing_decision=routing_decision,
            request=request,
            trace_id=trace_id,
            trace=trace,
            metrics=metrics,
            feedback_hooks=feedback_hooks,
            log_ctx=log_ctx,
        )

        # Stage 6: Synthesis
        synthesized = await self._execute_synthesis(
            query=request.query,
            aggregated=aggregated,
            user_profile=request.user_profile,
            trace_id=trace_id,
            trace=trace,
            metrics=metrics,
            feedback_hooks=feedback_hooks,
            log_ctx=log_ctx,
        )

        # Calculate totals
        elapsed = (time.perf_counter() - start_time) * 1000
        metrics.total_time_ms = elapsed
        trace.total_time_ms = elapsed

        # Calculate total tokens
        total_tokens = sum(
            exec_trace.get("tokens_used", 0)
            for exec_trace in trace.expert_executions
        )
        metrics.total_tokens = total_tokens
        trace.total_tokens = total_tokens

        log_ctx.info(
            "pipeline_completed",
            total_time_ms=elapsed,
            total_tokens=total_tokens,
            experts_activated=len(metrics.experts_activated),
            degraded=metrics.degraded,
        )

        return PipelineResult(
            response=synthesized,
            trace=trace,
            metrics=metrics,
            feedback_hooks=feedback_hooks,
            success=True,
        )

    def _validate_request(self, request: PipelineRequest) -> None:
        """Validate pipeline request."""
        if not request.query or not request.query.strip():
            raise PipelineValidationError("Query cannot be empty")

        if len(request.query) > 10000:
            raise PipelineValidationError("Query too long (max 10000 characters)")

        valid_profiles = {"consulenza", "ricerca", "analisi", "contributore"}
        if request.user_profile not in valid_profiles:
            raise PipelineValidationError(
                f"Invalid user_profile: {request.user_profile}. "
                f"Valid profiles: {valid_profiles}"
            )

    async def _execute_ner(
        self,
        query: str,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> ExtractionResult:
        """Execute NER extraction stage."""
        stage_start = time.perf_counter()

        ner_result = await self.ner_service.extract(query)

        stage_time = (time.perf_counter() - stage_start) * 1000
        metrics.ner_time_ms = stage_time
        trace.stage_times_ms[PipelineStage.NER] = stage_time
        trace.ner_result = ner_result.to_dict()

        # F1 feedback hook for ambiguous entities
        if self.config.enable_feedback_hooks and ner_result.ambiguous_entities:
            feedback_hooks.append(
                FeedbackHook(
                    feedback_type="F1",
                    expert_type="ner",
                    response_id=trace_id,
                    enabled=True,
                )
            )

        log_ctx.debug(
            "ner_completed",
            entity_count=len(ner_result.entities),
            ambiguous_count=len(ner_result.ambiguous_entities),
            time_ms=stage_time,
        )

        return ner_result

    async def _execute_routing(
        self,
        query: str,
        ner_result: ExtractionResult,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        log_ctx: Any,
    ) -> RoutingDecision:
        """Execute routing stage."""
        stage_start = time.perf_counter()

        routing_decision = await self.router.route(query, ner_result)

        stage_time = (time.perf_counter() - stage_start) * 1000
        metrics.routing_time_ms = stage_time
        trace.stage_times_ms[PipelineStage.ROUTING] = stage_time
        trace.routing_decision = routing_decision.to_dict()

        log_ctx.debug(
            "routing_completed",
            query_type=routing_decision.query_type.value,
            primary_expert=routing_decision.get_primary_expert(),
            time_ms=stage_time,
        )

        return routing_decision

    def _build_expert_context(
        self,
        query: str,
        ner_result: ExtractionResult,
        trace_id: str,
    ) -> ExpertContext:
        """Build context for expert execution."""
        # Extract entities by type
        entities = {
            "norm_references": [e.text for e in ner_result.article_refs],
            "legal_concepts": [e.text for e in ner_result.legal_concepts],
            "temporal_refs": [e.text for e in ner_result.temporal_refs],
            "party_refs": [e.text for e in ner_result.party_refs],
        }

        return ExpertContext(
            query_text=query,
            entities=entities,
            trace_id=trace_id,
            metadata={
                "ner_entity_count": len(ner_result.entities),
                "has_ambiguous": len(ner_result.ambiguous_entities) > 0,
            },
        )

    async def _execute_experts_parallel(
        self,
        expert_context: ExpertContext,
        routing_decision: RoutingDecision,
        request: PipelineRequest,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> List[ExpertResponse]:
        """
        Execute experts in parallel with circuit breaker protection.

        Returns:
            List of ExpertResponse (may be partial if some experts fail)
        """
        # Get weights, potentially with overrides
        weights = {}
        for ew in routing_decision.expert_weights:
            weights[ew.expert.value] = ew.weight

        if request.override_weights:
            weights.update(request.override_weights)

        # Determine which experts to activate
        experts_to_run: List[Tuple[str, float]] = []
        for expert_type, weight in weights.items():
            # Skip if weight below weight threshold
            if weight < self.config.expert_weight_threshold:
                trace.expert_executions.append(
                    ExpertExecution(
                        expert_type=expert_type,
                        skipped=True,
                        skip_reason=f"Weight {weight:.2f} below threshold {self.config.expert_weight_threshold}",
                    ).to_dict()
                )
                metrics.experts_skipped.append(expert_type)
                continue

            # Skip if in bypass list
            if request.bypass_experts and expert_type in request.bypass_experts:
                trace.expert_executions.append(
                    ExpertExecution(
                        expert_type=expert_type,
                        skipped=True,
                        skip_reason="Bypassed by request",
                    ).to_dict()
                )
                metrics.experts_skipped.append(expert_type)
                continue

            # Skip if expert not registered
            if expert_type not in self._experts:
                trace.expert_executions.append(
                    ExpertExecution(
                        expert_type=expert_type,
                        skipped=True,
                        skip_reason="Expert not registered",
                    ).to_dict()
                )
                metrics.experts_skipped.append(expert_type)
                continue

            experts_to_run.append((expert_type, weight))

        if not experts_to_run:
            log_ctx.warning("no_experts_to_run")
            return []

        # Execute in parallel or sequential
        if self.config.parallel_execution:
            results = await self._run_experts_parallel(
                experts_to_run=experts_to_run,
                expert_context=expert_context,
                trace_id=trace_id,
                trace=trace,
                metrics=metrics,
                feedback_hooks=feedback_hooks,
                log_ctx=log_ctx,
            )
        else:
            results = await self._run_experts_sequential(
                experts_to_run=experts_to_run,
                expert_context=expert_context,
                trace_id=trace_id,
                trace=trace,
                metrics=metrics,
                feedback_hooks=feedback_hooks,
                log_ctx=log_ctx,
            )

        # Calculate degradation
        if metrics.experts_failed:
            metrics.degraded = True
            penalty = min(
                len(metrics.experts_failed) * self.config.degradation_confidence_penalty,
                self.config.max_degradation_penalty,
            )
            metrics.degradation_reason = (
                f"{len(metrics.experts_failed)} expert(s) failed: "
                f"{', '.join(metrics.experts_failed)}. "
                f"Confidence penalty: -{penalty:.0%}"
            )

        return results

    async def _run_experts_parallel(
        self,
        experts_to_run: List[Tuple[str, float]],
        expert_context: ExpertContext,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> List[ExpertResponse]:
        """Run experts in parallel using asyncio.gather."""
        tasks = []
        expert_types = []

        for expert_type, weight in experts_to_run:
            task = self._execute_single_expert(
                expert_type=expert_type,
                expert_context=expert_context,
                trace_id=trace_id,
                log_ctx=log_ctx,
            )
            tasks.append(task)
            expert_types.append(expert_type)

        # Execute all with return_exceptions=True to handle failures gracefully
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        responses: List[ExpertResponse] = []
        for i, result in enumerate(results):
            expert_type = expert_types[i]
            execution = ExpertExecution(expert_type=expert_type)

            if isinstance(result, Exception):
                # Expert failed
                execution.success = False
                execution.error = str(result)
                if isinstance(result, CircuitOpenError):
                    execution.circuit_breaker_state = "open"
                metrics.experts_failed.append(expert_type)
                log_ctx.warning(
                    "expert_failed",
                    expert_type=expert_type,
                    error=str(result),
                )
            elif result is not None:
                # Expert succeeded
                execution.success = True
                execution.output = result.to_dict()
                execution.tokens_used = result.tokens_used
                execution.confidence = result.confidence
                execution.duration_ms = result.execution_time_ms
                metrics.experts_activated.append(expert_type)
                metrics.expert_times_ms[expert_type] = result.execution_time_ms
                responses.append(result)

                # Add feedback hook
                if self.config.enable_feedback_hooks and result.feedback_hook:
                    feedback_hooks.append(result.feedback_hook)

            trace.expert_executions.append(execution.to_dict())

        return responses

    async def _run_experts_sequential(
        self,
        experts_to_run: List[Tuple[str, float]],
        expert_context: ExpertContext,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> List[ExpertResponse]:
        """Run experts sequentially."""
        responses: List[ExpertResponse] = []

        for expert_type, weight in experts_to_run:
            execution = ExpertExecution(expert_type=expert_type)

            try:
                result = await self._execute_single_expert(
                    expert_type=expert_type,
                    expert_context=expert_context,
                    trace_id=trace_id,
                    log_ctx=log_ctx,
                )

                if result is not None:
                    execution.success = True
                    execution.output = result.to_dict()
                    execution.tokens_used = result.tokens_used
                    execution.confidence = result.confidence
                    execution.duration_ms = result.execution_time_ms
                    metrics.experts_activated.append(expert_type)
                    metrics.expert_times_ms[expert_type] = result.execution_time_ms
                    responses.append(result)

                    if self.config.enable_feedback_hooks and result.feedback_hook:
                        feedback_hooks.append(result.feedback_hook)

            except Exception as e:
                execution.success = False
                execution.error = str(e)
                if isinstance(e, CircuitOpenError):
                    execution.circuit_breaker_state = "open"
                metrics.experts_failed.append(expert_type)
                log_ctx.warning(
                    "expert_failed",
                    expert_type=expert_type,
                    error=str(e),
                )

            trace.expert_executions.append(execution.to_dict())

        return responses

    async def _execute_single_expert(
        self,
        expert_type: str,
        expert_context: ExpertContext,
        trace_id: str,
        log_ctx: Any,
    ) -> Optional[ExpertResponse]:
        """
        Execute a single expert with circuit breaker and timeout.

        Returns:
            ExpertResponse or None if failed
        """
        expert = self._experts.get(expert_type)
        if not expert:
            return None

        # Get circuit breaker
        cb = get_expert_circuit_breaker(expert_type)

        try:
            async with cb:
                # Execute with timeout
                timeout_sec = self.config.expert_timeout_ms / 1000.0
                result = await asyncio.wait_for(
                    expert.analyze(expert_context),
                    timeout=timeout_sec,
                )
                return result

        except asyncio.TimeoutError:
            log_ctx.warning(
                "expert_timeout",
                expert_type=expert_type,
                timeout_ms=self.config.expert_timeout_ms,
            )
            raise

        except CircuitOpenError:
            log_ctx.warning(
                "expert_circuit_open",
                expert_type=expert_type,
            )
            raise

    async def _execute_gating(
        self,
        expert_responses: List[ExpertResponse],
        routing_decision: RoutingDecision,
        request: PipelineRequest,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> AggregatedResponse:
        """Execute gating aggregation stage."""
        stage_start = time.perf_counter()

        # Get weights from routing decision
        weights = {ew.expert.value: ew.weight for ew in routing_decision.expert_weights}
        if request.override_weights:
            weights.update(request.override_weights)

        aggregated = await self.gating.aggregate(
            responses=expert_responses,
            weights=weights,
            trace_id=trace_id,
            user_profile=request.user_profile,
        )

        stage_time = (time.perf_counter() - stage_start) * 1000
        metrics.gating_time_ms = stage_time
        trace.stage_times_ms[PipelineStage.GATING] = stage_time
        trace.gating_result = aggregated.to_dict()

        # Add gating feedback hook
        if self.config.enable_feedback_hooks and aggregated.feedback_hook:
            feedback_hooks.append(aggregated.feedback_hook)

        log_ctx.debug(
            "gating_completed",
            method=aggregated.aggregation_method,
            confidence=aggregated.confidence,
            conflicts=len(aggregated.conflicts),
            time_ms=stage_time,
        )

        return aggregated

    async def _execute_synthesis(
        self,
        query: str,
        aggregated: AggregatedResponse,
        user_profile: str,
        trace_id: str,
        trace: PipelineTrace,
        metrics: PipelineMetrics,
        feedback_hooks: List[FeedbackHook],
        log_ctx: Any,
    ) -> SynthesizedResponse:
        """Execute synthesis stage."""
        stage_start = time.perf_counter()

        synthesized = await self.synthesizer.synthesize(
            query=query,
            aggregated=aggregated,
            user_profile=user_profile,
            trace_id=trace_id,
        )

        stage_time = (time.perf_counter() - stage_start) * 1000
        metrics.synthesis_time_ms = stage_time
        trace.stage_times_ms[PipelineStage.SYNTHESIS] = stage_time
        trace.synthesis_result = synthesized.to_dict()

        # Add synthesis feedback hook
        if self.config.enable_feedback_hooks and synthesized.feedback_hook:
            feedback_hooks.append(synthesized.feedback_hook)

        log_ctx.debug(
            "synthesis_completed",
            mode=synthesized.synthesis_mode,
            profile=user_profile,
            has_disagreement=synthesized.has_disagreement,
            time_ms=stage_time,
        )

        return synthesized

    def _create_error_response(
        self,
        error_message: str,
        trace_id: str,
    ) -> SynthesizedResponse:
        """Create error response when pipeline fails."""
        return SynthesizedResponse(
            main_answer=f"Si è verificato un errore durante l'elaborazione: {error_message}",
            confidence_indicator="bassa",
            confidence_value=0.0,
            synthesis_mode="error",
            trace_id=trace_id,
            metadata={"error": error_message},
        )

    def get_circuit_breaker_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics for all experts."""
        return {
            name: cb.get_stats().to_dict()
            for name, cb in self._cb_registry._breakers.items()
        }

    def reset_circuit_breakers(self) -> None:
        """Reset all circuit breakers."""
        self._cb_registry.reset_all()
        log.info("circuit_breakers_reset")
