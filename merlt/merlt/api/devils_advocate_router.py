"""
Devil's Advocate Router
========================

API wrapper for DevilsAdvocateAssigner.
Triggered on high-consensus queries (disagreement < 0.1).

Endpoints:
- POST /devils-advocate/check: Check if DA should trigger for a trace
- POST /devils-advocate/feedback: Submit DA feedback
- GET /devils-advocate/effectiveness: Aggregate effectiveness metrics
"""

import structlog
from collections import deque
from threading import Lock
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field

from merlt.rlcf.devils_advocate import (
    DevilsAdvocateAssigner,
    TaskType,
)

log = structlog.get_logger()

router = APIRouter(prefix="/devils-advocate", tags=["devils-advocate"])

# Singleton assigner
_assigner = DevilsAdvocateAssigner()

# Threshold: only trigger DA when disagreement is very low (high consensus)
TRIGGER_THRESHOLD = 0.1


# =============================================================================
# MODELS
# =============================================================================


class DACheckResponse(BaseModel):
    triggered: bool
    critical_prompt: Optional[str] = None
    message: str


class DAFeedbackRequest(BaseModel):
    trace_id: str
    feedback_text: str = Field(..., min_length=5, description="Devil's advocate response text")
    assessment: str = Field(
        "interesting",
        description="Assessment: valid, weak, interesting",
    )


class DAFeedbackResponse(BaseModel):
    received: bool
    engagement_score: float
    critical_keywords_found: int


class DAEffectivenessResponse(BaseModel):
    total_triggers: int = 0
    total_feedbacks: int = 0
    avg_engagement: float = 0.0
    avg_keywords: float = 0.0


# =============================================================================
# STATE (in-memory, bounded, thread-safe)
# =============================================================================

_state_lock = Lock()
_trigger_count = 0
_feedback_entries: deque = deque(maxlen=1000)


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/check", response_model=DACheckResponse)
async def check_devils_advocate(
    trace_id: str = Query(..., description="Trace ID to check"),
    disagreement_score: float = Query(..., ge=0.0, le=1.0, description="Current disagreement score"),
):
    """
    Check if Devil's Advocate should trigger for a trace.

    Triggered when disagreement_score < 0.1 (high consensus, potential groupthink).
    """
    global _trigger_count

    if disagreement_score >= TRIGGER_THRESHOLD:
        return DACheckResponse(
            triggered=False,
            critical_prompt=None,
            message=f"Disagreement {disagreement_score:.2f} >= threshold {TRIGGER_THRESHOLD}. No DA needed.",
        )

    # Generate critical prompt
    prompt = _assigner.generate_critical_prompt(task_type=TaskType.QA)
    with _state_lock:
        _trigger_count += 1

    log.info(
        "Devils advocate triggered",
        trace_id=trace_id,
        disagreement=disagreement_score,
    )

    return DACheckResponse(
        triggered=True,
        critical_prompt=prompt,
        message=f"High consensus detected (disagreement={disagreement_score:.2f}). Critical review recommended.",
    )


@router.post("/feedback", response_model=DAFeedbackResponse)
async def submit_da_feedback(request: DAFeedbackRequest):
    """
    Submit Devil's Advocate feedback and analyze engagement.
    """
    # Analyze critical engagement
    engagement_score, keywords_found = _assigner.analyze_critical_engagement(request.feedback_text)

    with _state_lock:
        _feedback_entries.append({
            "trace_id": request.trace_id,
            "assessment": request.assessment,
            "engagement_score": engagement_score,
            "keywords_found": keywords_found,
        })

    log.info(
        "DA feedback received",
        trace_id=request.trace_id,
        engagement=round(engagement_score, 3),
        keywords=keywords_found,
    )

    return DAFeedbackResponse(
        received=True,
        engagement_score=round(engagement_score, 4),
        critical_keywords_found=keywords_found,
    )


@router.get("/effectiveness", response_model=DAEffectivenessResponse)
async def get_da_effectiveness():
    """Aggregate Devil's Advocate effectiveness metrics."""
    with _state_lock:
        entries = list(_feedback_entries)
        triggers = _trigger_count

    if not entries:
        return DAEffectivenessResponse(
            total_triggers=triggers,
            total_feedbacks=0,
            avg_engagement=0.0,
            avg_keywords=0.0,
        )

    avg_eng = sum(e["engagement_score"] for e in entries) / len(entries)
    avg_kw = sum(e["keywords_found"] for e in entries) / len(entries)

    return DAEffectivenessResponse(
        total_triggers=triggers,
        total_feedbacks=len(entries),
        avg_engagement=round(avg_eng, 4),
        avg_keywords=round(avg_kw, 2),
    )
