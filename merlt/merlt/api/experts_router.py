"""
Expert System Q&A Router
=========================

FastAPI router for Expert System Q&A with multi-level feedback.

Endpoints:
- POST /api/experts/query: Submit query to MultiExpertOrchestrator
- POST /api/experts/feedback/inline: Quick thumbs up/down
- POST /api/experts/feedback/detailed: 3-dimension feedback form
- POST /api/experts/feedback/source: Per-source rating
- POST /api/experts/feedback/refine: Conversational follow-up

Usage:
    from merlt.api.experts_router import router as experts_router
    app.include_router(experts_router)
"""

import structlog
import time
from typing import Optional, List, Dict, Any
from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from merlt.experts.orchestrator import MultiExpertOrchestrator
from merlt.experts.synthesizer import SynthesisMode
from merlt.experts.models import QATrace, QAFeedback
from merlt.rlcf.database import get_async_session_dep
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

log = structlog.get_logger()

router = APIRouter(prefix="/api/experts", tags=["experts"])


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ExpertQueryRequest(BaseModel):
    """Request for Expert Q&A query."""
    query: str = Field(..., min_length=5, description="Legal query in natural language")
    user_id: str = Field(..., description="User ID for tracking and authority")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context (entities, article_urn, etc.)")
    max_experts: Optional[int] = Field(4, ge=1, le=4, description="Max number of experts to invoke")
    include_trace: bool = Field(False, description="Include full pipeline trace in response")


class SourceReference(BaseModel):
    """Legal source reference in response."""
    article_urn: str
    expert: str
    relevance: float = Field(..., ge=0.0, le=1.0)
    excerpt: Optional[str] = None


class ExpertQueryResponse(BaseModel):
    """Response from Expert Q&A system."""
    trace_id: str = Field(..., description="Unique trace ID for feedback")
    synthesis: str = Field(..., description="Final synthesis text")
    mode: str = Field(..., description="convergent or divergent")
    alternatives: Optional[List[Dict[str, Any]]] = Field(None, description="Alternative interpretations (divergent mode)")
    sources: List[SourceReference] = Field(default_factory=list, description="Legal sources cited")
    experts_used: List[str] = Field(default_factory=list, description="Experts that analyzed the query")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    execution_time_ms: int = Field(..., description="Execution time in milliseconds")
    pipeline_trace: Optional[Dict[str, Any]] = Field(None, description="Full pipeline trace (when include_trace=True)")
    pipeline_metrics: Optional[Dict[str, Any]] = Field(None, description="Pipeline metrics (when include_trace=True)")


class InlineFeedbackRequest(BaseModel):
    """Quick thumbs feedback."""
    trace_id: str
    user_id: str
    rating: int = Field(..., ge=1, le=5, description="1=thumbs down, 5=thumbs up")


class DetailedFeedbackRequest(BaseModel):
    """Detailed 3-dimension feedback."""
    trace_id: str
    user_id: str
    retrieval_score: float = Field(..., ge=0.0, le=1.0, description="Quality of retrieved sources")
    reasoning_score: float = Field(..., ge=0.0, le=1.0, description="Quality of reasoning")
    synthesis_score: float = Field(..., ge=0.0, le=1.0, description="Quality of synthesis")
    comment: Optional[str] = Field(None, description="Optional textual comment")
    user_authority: Optional[float] = None


class SourceFeedbackRequest(BaseModel):
    """Per-source rating feedback."""
    trace_id: str
    user_id: str
    source_id: str = Field(..., description="article URN")
    relevance: int = Field(..., ge=1, le=5, description="1-5 stars")
    user_authority: Optional[float] = None


class RefineFeedbackRequest(BaseModel):
    """Conversational refinement feedback."""
    trace_id: str
    user_id: str
    follow_up_query: str = Field(..., min_length=5)


class ExpertPreferenceFeedbackRequest(BaseModel):
    """
    Feedback for divergent interpretations.

    When mode=divergent, user can indicate which expert
    provided the most useful interpretation.
    """
    trace_id: str
    user_id: str
    preferred_expert: str = Field(..., description="Expert type (literal, systemic, principles, precedent)")
    comment: Optional[str] = Field(None, description="Optional comment explaining preference")


class FeedbackResponse(BaseModel):
    """Generic feedback response."""
    success: bool
    feedback_id: Optional[int] = None
    message: str


# ============================================================================
# DEPENDENCY INJECTION
# ============================================================================

# Global orchestrator instance (initialized on startup)
_orchestrator: Optional[MultiExpertOrchestrator] = None


def get_orchestrator() -> MultiExpertOrchestrator:
    """
    Get MultiExpertOrchestrator instance.

    This should be initialized in the app startup event.
    For now, returns None and will be injected via dependency.
    """
    global _orchestrator
    if _orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Expert System not initialized. Call initialize_expert_system() on startup."
        )
    return _orchestrator


def initialize_expert_system(orchestrator: MultiExpertOrchestrator):
    """
    Initialize expert system with orchestrator.

    Should be called in FastAPI startup event:

    @app.on_event("startup")
    async def startup():
        orchestrator = create_orchestrator()
        initialize_expert_system(orchestrator)
    """
    global _orchestrator
    _orchestrator = orchestrator
    log.info("Expert System initialized")


# ============================================================================
# ENDPOINTS
# ============================================================================

@router.post("/query", response_model=ExpertQueryResponse)
async def query_experts(
    request: ExpertQueryRequest,
    session: AsyncSession = Depends(get_async_session_dep),
    orchestrator: MultiExpertOrchestrator = Depends(get_orchestrator)
):
    """
    Submit query to MultiExpertOrchestrator and save trace.

    Flow:
    1. Run MultiExpertOrchestrator.process()
    2. Extract sources from SynthesisResult
    3. Save QATrace to database
    4. Return response with trace_id

    Example:
        POST /api/experts/query
        {
            "query": "Cos'è la legittima difesa?",
            "user_id": "user123"
        }

        Response:
        {
            "trace_id": "trace_abc123",
            "synthesis": "La legittima difesa è...",
            "mode": "convergent",
            "sources": [
                {"article_urn": "urn:...:art52", "expert": "literal", "relevance": 0.95}
            ],
            "experts_used": ["literal", "systemic"],
            "confidence": 0.85,
            "execution_time_ms": 2450
        }
    """
    start_time = time.time()

    log.info(
        "Expert query received",
        query=request.query[:50],
        user_id=request.user_id,
        max_experts=request.max_experts
    )

    try:
        # Run orchestrator
        result = await orchestrator.process(
            query=request.query,
            entities=request.context.get("entities") if request.context else None,
            retrieved_chunks=request.context.get("retrieved_chunks") if request.context else None,
            metadata={"user_id": request.user_id},
            include_trace=request.include_trace
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Generate trace ID
        trace_id = f"trace_{uuid4().hex[:12]}"

        # Extract sources from combined_legal_basis
        sources = []
        for legal_source in result.combined_legal_basis:
            sources.append(SourceReference(
                article_urn=legal_source.source_id,  # LegalSource uses source_id, not urn
                expert="combined",  # LegalSource doesn't track individual expert, it's combined
                relevance=0.9,  # Default high relevance for combined sources
                excerpt=legal_source.excerpt[:200] if legal_source.excerpt else None
            ))

        # Extract experts used from expert_contributions
        experts_used = list(result.expert_contributions.keys())

        # Extract pipeline trace from result metadata
        pipeline_trace_data = result.metadata.get("pipeline_trace") if request.include_trace else None
        pipeline_metrics_data = result.metadata.get("pipeline_metrics") if request.include_trace else None

        # Inject trace_id into pipeline_trace for correlation
        if pipeline_trace_data:
            pipeline_trace_data["trace_id"] = trace_id

        # Save trace to database
        trace = QATrace(
            trace_id=trace_id,
            user_id=request.user_id,
            query=request.query,
            selected_experts=experts_used,
            synthesis_mode=result.mode.value,
            synthesis_text=result.synthesis,
            sources=[s.dict() for s in sources],  # Store as JSONB
            execution_time_ms=execution_time_ms,
            full_trace=pipeline_trace_data,
        )
        session.add(trace)
        await session.commit()

        log.info(
            "Expert query completed",
            trace_id=trace_id,
            mode=result.mode.value,
            experts_count=len(experts_used),
            sources_count=len(sources),
            execution_time_ms=execution_time_ms,
            has_trace=pipeline_trace_data is not None
        )

        # Return response
        return ExpertQueryResponse(
            trace_id=trace_id,
            synthesis=result.synthesis,
            mode=result.mode.value,
            alternatives=result.alternatives if result.mode == SynthesisMode.DIVERGENT else None,
            sources=sources,
            experts_used=experts_used,
            confidence=result.confidence,
            execution_time_ms=execution_time_ms,
            pipeline_trace=pipeline_trace_data,
            pipeline_metrics=pipeline_metrics_data,
        )

    except Exception as e:
        log.error("Expert query failed", error=str(e), query=request.query[:50])
        raise HTTPException(status_code=500, detail=f"Expert query failed: {str(e)}")


@router.post("/feedback/inline", response_model=FeedbackResponse)
async def submit_inline_feedback(
    request: InlineFeedbackRequest,
    session: AsyncSession = Depends(get_async_session_dep)
):
    """
    Submit quick thumbs up/down feedback.

    Example:
        POST /api/experts/feedback/inline
        {
            "trace_id": "trace_abc123",
            "user_id": "user456",
            "rating": 5  // thumbs up
        }
    """
    log.info(
        "Inline feedback received",
        trace_id=request.trace_id,
        user_id=request.user_id,
        rating=request.rating
    )

    try:
        # Verify trace exists
        result = await session.execute(
            select(QATrace).where(QATrace.trace_id == request.trace_id)
        )
        trace = result.scalar_one_or_none()

        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")

        # Create feedback
        feedback = QAFeedback(
            trace_id=request.trace_id,
            user_id=request.user_id,
            inline_rating=request.rating
        )
        session.add(feedback)
        await session.commit()

        log.info(
            "Inline feedback saved",
            feedback_id=feedback.id,
            trace_id=request.trace_id,
            rating=request.rating
        )

        return FeedbackResponse(
            success=True,
            feedback_id=feedback.id,
            message="Inline feedback saved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to save inline feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/detailed", response_model=FeedbackResponse)
async def submit_detailed_feedback(
    request: DetailedFeedbackRequest,
    session: AsyncSession = Depends(get_async_session_dep)
):
    """
    Submit detailed 3-dimension feedback.

    Example:
        POST /api/experts/feedback/detailed
        {
            "trace_id": "trace_abc123",
            "user_id": "user456",
            "retrieval_score": 0.8,
            "reasoning_score": 0.9,
            "synthesis_score": 0.7,
            "comment": "Buona risposta ma sintesi migliorabile"
        }
    """
    log.info(
        "Detailed feedback received",
        trace_id=request.trace_id,
        user_id=request.user_id,
        avg_score=(request.retrieval_score + request.reasoning_score + request.synthesis_score) / 3
    )

    try:
        # Verify trace exists
        result = await session.execute(
            select(QATrace).where(QATrace.trace_id == request.trace_id)
        )
        trace = result.scalar_one_or_none()

        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")

        # Create feedback
        feedback = QAFeedback(
            trace_id=request.trace_id,
            user_id=request.user_id,
            retrieval_score=request.retrieval_score,
            reasoning_score=request.reasoning_score,
            synthesis_score=request.synthesis_score,
            detailed_comment=request.comment,
            user_authority=request.user_authority
        )
        session.add(feedback)
        await session.commit()

        log.info(
            "Detailed feedback saved",
            feedback_id=feedback.id,
            trace_id=request.trace_id,
            retrieval=request.retrieval_score,
            reasoning=request.reasoning_score,
            synthesis=request.synthesis_score
        )

        return FeedbackResponse(
            success=True,
            feedback_id=feedback.id,
            message="Detailed feedback saved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to save detailed feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/source", response_model=FeedbackResponse)
async def submit_source_feedback(
    request: SourceFeedbackRequest,
    session: AsyncSession = Depends(get_async_session_dep)
):
    """
    Submit per-source rating feedback.

    Example:
        POST /api/experts/feedback/source
        {
            "trace_id": "trace_abc123",
            "user_id": "user456",
            "source_id": "urn:nir:stato:codice.civile:1942;art1453",
            "relevance": 5  // 5 stars
        }
    """
    log.info(
        "Source feedback received",
        trace_id=request.trace_id,
        source_id=request.source_id,
        relevance=request.relevance
    )

    try:
        # Verify trace exists
        result = await session.execute(
            select(QATrace).where(QATrace.trace_id == request.trace_id)
        )
        trace = result.scalar_one_or_none()

        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")

        # Create feedback
        feedback = QAFeedback(
            trace_id=request.trace_id,
            user_id=request.user_id,
            source_id=request.source_id,
            source_relevance=request.relevance,
            user_authority=request.user_authority
        )
        session.add(feedback)
        await session.commit()

        log.info(
            "Source feedback saved",
            feedback_id=feedback.id,
            source_id=request.source_id,
            relevance=request.relevance
        )

        return FeedbackResponse(
            success=True,
            feedback_id=feedback.id,
            message="Source feedback saved successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to save source feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/preference", response_model=FeedbackResponse)
async def submit_expert_preference_feedback(
    request: ExpertPreferenceFeedbackRequest,
    session: AsyncSession = Depends(get_async_session_dep)
):
    """
    Submit expert preference feedback for divergent interpretations.

    When mode=divergent, users can indicate which expert's interpretation
    was most useful. This feedback is used for:
    - RLCF training
    - Expert weight optimization
    - Response synthesis improvement

    Example:
        POST /api/experts/feedback/preference
        {
            "trace_id": "trace_abc123",
            "user_id": "user456",
            "preferred_expert": "systemic",
            "comment": "L'interpretazione sistematica e' piu' completa"
        }
    """
    log.info(
        "Expert preference feedback received",
        trace_id=request.trace_id,
        user_id=request.user_id,
        preferred_expert=request.preferred_expert
    )

    try:
        # Verify trace exists and was divergent
        result = await session.execute(
            select(QATrace).where(QATrace.trace_id == request.trace_id)
        )
        trace = result.scalar_one_or_none()

        if not trace:
            raise HTTPException(status_code=404, detail=f"Trace {request.trace_id} not found")

        if trace.synthesis_mode != "divergent":
            log.warning(
                "Preference feedback for non-divergent trace",
                trace_id=request.trace_id,
                mode=trace.synthesis_mode
            )
            # Still accept feedback but log warning

        # Validate expert type
        valid_experts = ["literal", "systemic", "principles", "precedent"]
        if request.preferred_expert not in valid_experts:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid expert type '{request.preferred_expert}'. Valid: {valid_experts}"
            )

        # Create feedback with expert preference
        feedback = QAFeedback(
            trace_id=request.trace_id,
            user_id=request.user_id,
            preferred_expert=request.preferred_expert,
            detailed_comment=request.comment
        )
        session.add(feedback)
        await session.commit()

        log.info(
            "Expert preference feedback saved",
            feedback_id=feedback.id,
            trace_id=request.trace_id,
            preferred_expert=request.preferred_expert
        )

        return FeedbackResponse(
            success=True,
            feedback_id=feedback.id,
            message=f"Preference feedback saved: {request.preferred_expert}"
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to save expert preference feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save feedback: {str(e)}")


@router.post("/feedback/refine", response_model=ExpertQueryResponse)
async def submit_refine_feedback(
    request: RefineFeedbackRequest,
    session: AsyncSession = Depends(get_async_session_dep),
    orchestrator: MultiExpertOrchestrator = Depends(get_orchestrator)
):
    """
    Submit conversational refinement feedback and get new response.

    This endpoint:
    1. Saves follow-up query as feedback
    2. Re-runs orchestrator with context from original trace
    3. Links new trace to original via refined_trace_id

    Example:
        POST /api/experts/feedback/refine
        {
            "trace_id": "trace_abc123",
            "user_id": "user456",
            "follow_up_query": "Puoi spiegare meglio il requisito della proporzione?"
        }

        Returns: ExpertQueryResponse (same as /query)
    """
    start_time = time.time()

    log.info(
        "Refine feedback received",
        trace_id=request.trace_id,
        user_id=request.user_id,
        follow_up=request.follow_up_query[:50]
    )

    try:
        # Verify original trace exists
        result = await session.execute(
            select(QATrace).where(QATrace.trace_id == request.trace_id)
        )
        original_trace = result.scalar_one_or_none()

        if not original_trace:
            raise HTTPException(status_code=404, detail=f"Original trace {request.trace_id} not found")

        # Re-run orchestrator with follow-up query
        result = await orchestrator.process(
            query=request.follow_up_query,
            metadata={
                "user_id": request.user_id,
                "refine_from": request.trace_id,
                "original_query": original_trace.query
            }
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Generate new trace ID
        new_trace_id = f"trace_{uuid4().hex[:12]}"

        # Extract sources
        sources = []
        for legal_source in result.combined_legal_basis:
            sources.append(SourceReference(
                article_urn=legal_source.urn,
                expert=legal_source.expert_type,
                relevance=legal_source.relevance,
                excerpt=legal_source.text_excerpt[:200] if legal_source.text_excerpt else None
            ))

        experts_used = list(result.expert_contributions.keys())

        # Save new trace
        new_trace = QATrace(
            trace_id=new_trace_id,
            user_id=request.user_id,
            query=request.follow_up_query,
            selected_experts=experts_used,
            synthesis_mode=result.mode.value,
            synthesis_text=result.synthesis,
            sources=[s.dict() for s in sources],
            execution_time_ms=execution_time_ms
        )
        session.add(new_trace)

        # Save refinement feedback with link to new trace
        feedback = QAFeedback(
            trace_id=request.trace_id,
            user_id=request.user_id,
            follow_up_query=request.follow_up_query,
            refined_trace_id=new_trace_id
        )
        session.add(feedback)

        await session.commit()

        log.info(
            "Refine feedback processed",
            original_trace_id=request.trace_id,
            new_trace_id=new_trace_id,
            execution_time_ms=execution_time_ms
        )

        return ExpertQueryResponse(
            trace_id=new_trace_id,
            synthesis=result.synthesis,
            mode=result.mode.value,
            alternatives=result.alternatives if result.mode == SynthesisMode.DIVERGENT else None,
            sources=sources,
            experts_used=experts_used,
            confidence=result.confidence,
            execution_time_ms=execution_time_ms
        )

    except HTTPException:
        raise
    except Exception as e:
        log.error("Failed to process refine feedback", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process refinement: {str(e)}")


@router.get("/trace/{trace_id}")
async def get_trace(
    trace_id: str,
    session: AsyncSession = Depends(get_async_session_dep)
):
    """
    Recupera il trace completo di una query precedente.

    Returns the full pipeline trace JSON stored during query execution.
    Only available for queries executed with include_trace=True.

    Example:
        GET /api/experts/trace/trace_abc123def456
    """
    result = await session.execute(
        select(QATrace).where(QATrace.trace_id == trace_id)
    )
    qa_trace = result.scalar_one_or_none()

    if not qa_trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    if not qa_trace.full_trace:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline trace available for {trace_id}. Was the query executed with include_trace=True?"
        )

    return qa_trace.full_trace
