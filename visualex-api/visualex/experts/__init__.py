"""
Expert System Module for MERL-T Analysis Pipeline.

Implements the Art. 12 Preleggi interpretation hierarchy:
1. LiteralExpert - Interpretazione letterale
2. SystemicExpert - Interpretazione sistematica
3. PrinciplesExpert - Ratio legis / Principi
4. PrecedentExpert - Giurisprudenza

Example:
    >>> from visualex.experts import ExpertRouter, QueryType, LiteralExpert
    >>> from visualex.ner import NERService
    >>>
    >>> ner = NERService()
    >>> router = ExpertRouter()
    >>>
    >>> ner_result = await ner.extract("Cos'è la risoluzione del contratto?")
    >>> decision = await router.route("Cos'è la risoluzione del contratto?", ner_result)
    >>> print(decision.query_type)
    DEFINITION
    >>> print(decision.get_primary_expert())
    literal
    >>>
    >>> # Use LiteralExpert for detailed analysis
    >>> expert = LiteralExpert(retriever=my_retriever, llm_service=my_llm)
    >>> context = ExpertContext(query_text="Cos'è la risoluzione?")
    >>> response = await expert.analyze(context)
    >>> print(response.section_header)
    "Interpretazione Letterale"
"""

from .router import (
    ExpertRouter,
    RouterConfig,
    RoutingDecision,
    ExpertWeight,
    QueryType,
    ExpertType,
    QUERY_PATTERNS,
    DEFAULT_WEIGHTS,
)

from .base import (
    BaseExpert,
    ExpertConfig,
    ExpertContext,
    ExpertResponse,
    LegalSource,
    ReasoningStep,
    ConfidenceFactors,
    FeedbackHook,
    ChunkRetriever,
    LLMService,
)

from .literal import (
    LiteralExpert,
    LiteralConfig,
    LITERAL_PROMPT_TEMPLATE,
)

from .systemic import (
    SystemicExpert,
    SystemicConfig,
    GraphRelation,
    SYSTEMIC_PROMPT_TEMPLATE,
)

from .principles import (
    PrinciplesExpert,
    PrinciplesConfig,
    IdentifiedPrinciple,
    PRINCIPLES_PROMPT_TEMPLATE,
    LEGAL_PRINCIPLES,
    CONSTITUTIONAL_PRINCIPLES,
)

from .precedent import (
    PrecedentExpert,
    PrecedentConfig,
    CaseDecision,
    CourtAuthority,
    PRECEDENT_PROMPT_TEMPLATE,
    COURT_PATTERNS,
)

from .gating import (
    GatingNetwork,
    GatingConfig,
    AggregatedResponse,
    ExpertContribution,
    AggregationMethod,
    DEFAULT_EXPERT_WEIGHTS,
    USER_PROFILE_MODIFIERS,
    GATING_SYNTHESIS_PROMPT,
)

from .synthesizer import (
    Synthesizer,
    SynthesizerConfig,
    SynthesizedResponse,
    AccordionSection,
    UserProfile,
    SynthesisMode,
    SYNTHESIS_PROMPT_TEMPLATE,
    PROFILE_INSTRUCTIONS,
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitBreakerStats,
    CircuitState,
    CircuitOpenError,
    CircuitBreakerError,
    EXPERT_CIRCUIT_BREAKERS,
    get_expert_circuit_breaker,
    create_unavailable_response,
)

from .llm import (
    # Config
    LLMConfig,
    ProviderConfig,
    ModelVersionInfo,
    DEFAULT_PROVIDER_CONFIGS,
    # Base
    BaseLLMProvider,
    LLMResponse,
    LLMUsage,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    # Providers
    OpenAIProvider,
    AnthropicProvider,
    OllamaProvider,
    # Factory
    LLMProviderFactory,
    # Failover
    FailoverLLMService,
    FailoverConfig,
    FailoverEvent,
)

from .pipeline import PipelineOrchestrator

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

__all__ = [
    # Router
    "ExpertRouter",
    "RouterConfig",
    "RoutingDecision",
    "ExpertWeight",
    # Enums
    "QueryType",
    "ExpertType",
    # Constants
    "QUERY_PATTERNS",
    "DEFAULT_WEIGHTS",
    # Base classes
    "BaseExpert",
    "ExpertConfig",
    "ExpertContext",
    "ExpertResponse",
    "LegalSource",
    "ReasoningStep",
    "ConfidenceFactors",
    "FeedbackHook",
    "ChunkRetriever",
    "LLMService",
    # Experts
    "LiteralExpert",
    "LiteralConfig",
    "LITERAL_PROMPT_TEMPLATE",
    "SystemicExpert",
    "SystemicConfig",
    "GraphRelation",
    "SYSTEMIC_PROMPT_TEMPLATE",
    "PrinciplesExpert",
    "PrinciplesConfig",
    "IdentifiedPrinciple",
    "PRINCIPLES_PROMPT_TEMPLATE",
    "LEGAL_PRINCIPLES",
    "CONSTITUTIONAL_PRINCIPLES",
    "PrecedentExpert",
    "PrecedentConfig",
    "CaseDecision",
    "CourtAuthority",
    "PRECEDENT_PROMPT_TEMPLATE",
    "COURT_PATTERNS",
    # Gating Network
    "GatingNetwork",
    "GatingConfig",
    "AggregatedResponse",
    "ExpertContribution",
    "AggregationMethod",
    "DEFAULT_EXPERT_WEIGHTS",
    "USER_PROFILE_MODIFIERS",
    "GATING_SYNTHESIS_PROMPT",
    # Synthesizer
    "Synthesizer",
    "SynthesizerConfig",
    "SynthesizedResponse",
    "AccordionSection",
    "UserProfile",
    "SynthesisMode",
    "SYNTHESIS_PROMPT_TEMPLATE",
    "PROFILE_INSTRUCTIONS",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerRegistry",
    "CircuitBreakerStats",
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreakerError",
    "EXPERT_CIRCUIT_BREAKERS",
    "get_expert_circuit_breaker",
    "create_unavailable_response",
    # LLM Providers
    "LLMConfig",
    "ProviderConfig",
    "ModelVersionInfo",
    "DEFAULT_PROVIDER_CONFIGS",
    "BaseLLMProvider",
    "LLMResponse",
    "LLMUsage",
    "ProviderError",
    "RateLimitError",
    "AuthenticationError",
    "ModelNotFoundError",
    "OpenAIProvider",
    "AnthropicProvider",
    "OllamaProvider",
    "LLMProviderFactory",
    "FailoverLLMService",
    "FailoverConfig",
    "FailoverEvent",
    # Pipeline Orchestrator
    "PipelineOrchestrator",
    "PipelineRequest",
    "PipelineTrace",
    "PipelineMetrics",
    "PipelineResult",
    "OrchestratorConfig",
    "ExpertExecution",
    "PipelineStage",
    "PipelineError",
    "PipelineValidationError",
    "PipelineTimeoutError",
    "ExpertExecutionError",
    "generate_trace_id",
]
