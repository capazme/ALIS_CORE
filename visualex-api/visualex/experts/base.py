"""
Expert Base Classes for MERL-T Analysis Pipeline.

Provides foundational structures for the Art. 12 Preleggi Expert system:
- LiteralExpert: "significato proprio delle parole" (art. 12, I)
- SystemicExpert: "connessione di esse" (art. 12, I)
- PrinciplesExpert: "intenzione del legislatore" (art. 12, II)
- PrecedentExpert: Giurisprudenza applicata

Architecture:
    Query → Router → [Expert1, Expert2, ...] → GatingNetwork → Synthesizer
"""

import structlog
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol

log = structlog.get_logger()


@dataclass
class LegalSource:
    """
    Legal source cited in reasoning.

    Tracks provenance of each assertion.

    Attributes:
        source_type: Type (norm, jurisprudence, doctrine, constitutional)
        source_id: URN or unique identifier
        citation: Formal citation (e.g., "Art. 1321 c.c.")
        excerpt: Relevant excerpt
        relevance: Why this source is relevant (human-readable)
        relevance_score: Normalized relevance score [0.0-1.0] for ranking/training
    """

    source_type: str  # norm, jurisprudence, doctrine, constitutional
    source_id: str
    citation: str
    excerpt: str = ""
    relevance: str = ""
    relevance_score: float = 0.0  # Normalized 0-1 score for training

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "source_type": self.source_type,
            "source_id": self.source_id,
            "citation": self.citation,
            "excerpt": self.excerpt,
            "relevance": self.relevance,
            "relevance_score": self.relevance_score,
        }


@dataclass
class ReasoningStep:
    """
    Single step in the reasoning chain.

    Attributes:
        step_number: Progressive number
        description: Step description
        sources: IDs of sources used
    """

    step_number: int
    description: str
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "sources": self.sources,
        }


@dataclass
class ConfidenceFactors:
    """
    Breakdown of confidence score.

    Attributes:
        norm_clarity: Clarity of the norm [0-1]
        source_availability: Source availability [0-1]
        contextual_ambiguity: Contextual ambiguity [0-1] (1 = very ambiguous)
        definition_coverage: Coverage of defined terms [0-1]
    """

    norm_clarity: float = 0.5
    source_availability: float = 0.5
    contextual_ambiguity: float = 0.5
    definition_coverage: float = 0.5

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for API response."""
        return {
            "norm_clarity": round(self.norm_clarity, 3),
            "source_availability": round(self.source_availability, 3),
            "contextual_ambiguity": round(self.contextual_ambiguity, 3),
            "definition_coverage": round(self.definition_coverage, 3),
        }

    def compute_overall(self) -> float:
        """
        Compute overall confidence from factors.

        Ambiguity reduces confidence, others increase it.
        """
        positive = (self.norm_clarity + self.source_availability + self.definition_coverage) / 3
        penalty = self.contextual_ambiguity * 0.3
        return max(0.0, min(1.0, positive - penalty))


@dataclass
class ExpertContext:
    """
    Input context for Expert analysis.

    Contains all information needed for legal reasoning.

    Attributes:
        query_text: Original user query
        query_embedding: Query embedding for retrieval (optional)
        entities: Extracted entities (norms, concepts, etc.)
        retrieved_chunks: Already retrieved chunks (optional)
        metadata: Additional metadata
        trace_id: ID for tracing
    """

    query_text: str
    query_embedding: Optional[List[float]] = None
    entities: Dict[str, List[str]] = field(default_factory=dict)
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trace_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S_%f"))

    @property
    def norm_references(self) -> List[str]:
        """Extracted norm references."""
        return self.entities.get("norm_references", [])

    @property
    def legal_concepts(self) -> List[str]:
        """Extracted legal concepts."""
        return self.entities.get("legal_concepts", [])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "query_text": self.query_text,
            "entities": self.entities,
            "retrieved_chunks_count": len(self.retrieved_chunks),
            "metadata": self.metadata,
            "trace_id": self.trace_id,
        }


@dataclass
class FeedbackHook:
    """
    F3-F6 Feedback hook for RLCF integration.

    Allows users to provide feedback on Expert outputs for learning.

    Attributes:
        feedback_type: Type of feedback (F3=Literal, F4=Systemic, F5=Principles, F6=Precedent)
        expert_type: The expert this feedback is for
        response_id: Unique ID linking to the response
        enabled: Whether feedback collection is enabled
        correction_options: Available correction choices for each feedback dimension
        context_snapshot: Snapshot of key context for training (sources, confidence, etc.)
    """

    feedback_type: str  # F3, F4, F5, F6, F7
    expert_type: str
    response_id: str
    enabled: bool = True
    correction_options: Dict[str, List[str]] = field(default_factory=dict)
    context_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "feedback_type": self.feedback_type,
            "expert_type": self.expert_type,
            "response_id": self.response_id,
            "enabled": self.enabled,
            "correction_options": self.correction_options,
            "context_snapshot": self.context_snapshot,
        }


@dataclass
class ExpertResponse:
    """
    Structured output from an Expert.

    Attributes:
        expert_type: Type of expert (literal, systemic, principles, precedent)
        section_header: Italian header for UI display
        interpretation: Main interpretation (in Italian)
        legal_basis: Legal sources cited
        reasoning_steps: Reasoning chain steps
        confidence: Confidence score [0-1]
        confidence_factors: Breakdown of confidence
        limitations: What the expert could not consider
        suggestions: Suggestions for clarification (if low confidence)
        trace_id: ID for tracing
        execution_time_ms: Execution time in milliseconds
        tokens_used: LLM tokens used
        feedback_hook: F3-F6 feedback hook for RLCF
        metadata: Additional metadata
    """

    expert_type: str
    section_header: str
    interpretation: str
    legal_basis: List[LegalSource] = field(default_factory=list)
    reasoning_steps: List[ReasoningStep] = field(default_factory=list)
    confidence: float = 0.5
    confidence_factors: ConfidenceFactors = field(default_factory=ConfidenceFactors)
    limitations: str = ""
    suggestions: str = ""
    trace_id: str = ""
    execution_time_ms: float = 0.0
    tokens_used: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    feedback_hook: Optional[FeedbackHook] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        result = {
            "expert_type": self.expert_type,
            "section_header": self.section_header,
            "interpretation": self.interpretation,
            "legal_basis": [lb.to_dict() for lb in self.legal_basis],
            "reasoning_steps": [rs.to_dict() for rs in self.reasoning_steps],
            "confidence": round(self.confidence, 3),
            "confidence_factors": self.confidence_factors.to_dict(),
            "limitations": self.limitations,
            "suggestions": self.suggestions,
            "trace_id": self.trace_id,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "tokens_used": self.tokens_used,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
        if self.feedback_hook:
            result["feedback_hook"] = self.feedback_hook.to_dict()
        return result

    def is_low_confidence(self, threshold: float = 0.3) -> bool:
        """Check if response has low confidence."""
        return self.confidence < threshold


class ChunkRetriever(Protocol):
    """Protocol for chunk retrieval from Bridge Table."""

    async def retrieve(
        self,
        query: str,
        query_embedding: Optional[List[float]] = None,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks.

        Args:
            query: Query text
            query_embedding: Query embedding (optional)
            filters: Filters to apply (e.g., expert_affinity, source_type)
            limit: Maximum number of chunks to return

        Returns:
            List of chunk dictionaries with text, metadata, score
        """
        ...


class LLMService(Protocol):
    """Protocol for LLM service."""

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            Generated text
        """
        ...


@dataclass
class ExpertConfig:
    """Configuration for an Expert."""

    temperature: float = 0.3
    max_tokens: int = 2000
    chunk_limit: int = 10
    min_chunk_score: float = 0.5
    record_reasoning: bool = True


class BaseExpert(ABC):
    """
    Abstract base class for all Experts.

    Each Expert:
    - Has a specific type (literal, systemic, principles, precedent)
    - Has an Italian section header for UI display
    - Retrieves relevant chunks via Bridge Table
    - Produces structured ExpertResponse

    Subclasses must implement:
    - expert_type: Class attribute
    - section_header: Class attribute
    - description: Class attribute
    - analyze(): Main analysis method
    """

    expert_type: str = ""
    section_header: str = ""
    description: str = ""

    def __init__(
        self,
        retriever: Optional[ChunkRetriever] = None,
        llm_service: Optional[LLMService] = None,
        config: Optional[ExpertConfig] = None,
    ):
        """
        Initialize Expert.

        Args:
            retriever: Chunk retriever for Bridge Table access
            llm_service: LLM service for analysis
            config: Expert configuration
        """
        if not self.expert_type:
            raise ValueError("Expert must have an expert_type")
        if not self.section_header:
            raise ValueError("Expert must have a section_header")
        if not self.description:
            raise ValueError("Expert must have a description")

        self.retriever = retriever
        self.llm_service = llm_service
        self.config = config or ExpertConfig()

        log.info(
            "expert_initialized",
            expert_type=self.expert_type,
            has_retriever=self.retriever is not None,
            has_llm=self.llm_service is not None,
        )

    @abstractmethod
    async def analyze(self, context: ExpertContext) -> ExpertResponse:
        """
        Analyze query and produce interpretation.

        Args:
            context: Input context with query and entities

        Returns:
            ExpertResponse with interpretation, sources, confidence
        """
        pass

    def _create_low_confidence_response(
        self,
        context: ExpertContext,
        reason: str,
        suggestion: str,
        execution_time_ms: float = 0.0,
    ) -> ExpertResponse:
        """
        Create low-confidence response when analysis cannot proceed.

        Args:
            context: Input context
            reason: Why confidence is low
            suggestion: Suggestion for user
            execution_time_ms: Execution time

        Returns:
            Low-confidence ExpertResponse
        """
        return ExpertResponse(
            expert_type=self.expert_type,
            section_header=self.section_header,
            interpretation=f"Non è stato possibile fornire un'interpretazione affidabile. {reason}",
            confidence=0.1,
            confidence_factors=ConfidenceFactors(
                norm_clarity=0.1,
                source_availability=0.1,
                contextual_ambiguity=0.9,
                definition_coverage=0.1,
            ),
            limitations=reason,
            suggestions=suggestion,
            trace_id=context.trace_id,
            execution_time_ms=execution_time_ms,
        )

    def _build_reasoning_step(
        self,
        step_number: int,
        description: str,
        source_ids: Optional[List[str]] = None,
    ) -> ReasoningStep:
        """Build a reasoning step."""
        return ReasoningStep(
            step_number=step_number,
            description=description,
            sources=source_ids or [],
        )
