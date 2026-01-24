"""
Expert Metrics Router
=====================

Endpoints REST per metriche e statistiche del sistema Expert.

Questo router espone API per:
- Performance per expert (accuracy, latency, usage)
- Statistiche classificazione query
- Reasoning trace visualizzazione
- Aggregation stats (agreement rate, divergence)

NOTA: Attualmente restituisce dati vuoti/default.
      Per popolare i dati, implementare tracking delle metriche nel sistema Expert.

Endpoints:
- GET /expert-metrics/performance - Metriche performance per expert
- GET /expert-metrics/queries/stats - Statistiche classificazione query
- GET /expert-metrics/queries/recent - Query recenti con trace
- GET /expert-metrics/trace/{trace_id} - Reasoning trace singolo
- GET /expert-metrics/aggregation - Statistiche aggregazione

Example:
    >>> response = await client.get("/api/v1/expert-metrics/performance")
    >>> performance = response.json()
    >>> for expert in performance["experts"]:
    ...     print(f"{expert['name']}: {expert['accuracy']}%")
"""

from datetime import datetime
from typing import List, Optional

import structlog
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

log = structlog.get_logger()

router = APIRouter(prefix="/expert-metrics", tags=["expert-metrics"])


# =============================================================================
# MODELS
# =============================================================================


class ExpertPerformance(BaseModel):
    """Performance metrics per singolo expert."""
    name: str
    display_name: str
    accuracy: float = Field(0.0, ge=0.0, le=100.0, description="Accuracy %")
    accuracy_ci: tuple[float, float] = Field((0.0, 0.0), description="95% CI")
    latency_ms: float = Field(0.0, description="Latenza media in ms")
    latency_p95: float = Field(0.0, description="Latenza p95 in ms")
    usage_percentage: float = Field(0.0, ge=0.0, le=100.0, description="% query gestite")
    feedback_score: float = Field(0.0, ge=-1.0, le=1.0, description="Score feedback medio")
    feedback_count: int = Field(0, description="Numero feedback ricevuti")
    queries_handled: int = Field(0, description="Query totali gestite")


class ExpertPerformanceResponse(BaseModel):
    """Response con performance di tutti gli expert."""
    experts: List[ExpertPerformance] = Field(default_factory=list)
    period_days: int = 7
    total_queries: int = 0
    last_updated: Optional[str] = None


class QueryTypeStats(BaseModel):
    """Statistiche per tipo di query."""
    type: str
    count: int = 0
    percentage: float = 0.0
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0


class QueryStatsResponse(BaseModel):
    """Response con statistiche query."""
    total_queries: int = 0
    by_type: List[QueryTypeStats] = Field(default_factory=list)
    avg_latency_ms: float = 0.0
    avg_confidence: float = 0.0
    period_days: int = 7


class RecentQuery(BaseModel):
    """Query recente con summary."""
    trace_id: str
    query: str
    timestamp: str  # ISO format
    experts_used: List[str]
    confidence: float
    latency_ms: int
    mode: str  # convergent, divergent
    feedback_received: bool = False


class RecentQueriesResponse(BaseModel):
    """Response con query recenti."""
    queries: List[RecentQuery] = Field(default_factory=list)
    total_count: int = 0
    has_more: bool = False


class ExpertContribution(BaseModel):
    """Contributo di un singolo expert nella risposta."""
    expert_name: str
    confidence: float = 0.0
    weight: float = 0.0
    sources_cited: int = 0
    key_points: List[str] = Field(default_factory=list)
    excerpt: Optional[str] = None


class ReasoningTrace(BaseModel):
    """Trace completo del reasoning per una query."""
    trace_id: str
    query: str
    timestamp: str  # ISO format

    # Experts contributions
    contributions: List[ExpertContribution] = Field(default_factory=list)

    # Aggregation
    aggregation_method: str = ""
    final_confidence: float = 0.0
    confidence_ci: tuple[float, float] = (0.0, 0.0)

    # Output
    synthesis: str = ""
    mode: str = ""
    has_alternatives: bool = False

    # Performance
    total_latency_ms: int = 0
    sources_count: int = 0


class AggregationStats(BaseModel):
    """Statistiche di aggregazione delle risposte."""
    method: str = "weighted_consensus"
    total_responses: int = 0
    agreement_rate: float = 0.0  # % quando tutti concordano
    divergence_count: int = 0  # Query con divergenza
    divergence_rate: float = 0.0
    avg_confidence: float = 0.0
    confidence_ci: tuple[float, float] = (0.0, 0.0)
    avg_experts_per_query: float = 0.0


# =============================================================================
# REAL DATA ACCESS
# =============================================================================


def _get_expert_list() -> List[ExpertPerformance]:
    """
    Ritorna lista expert con valori default.

    TODO: Implementare tracking metriche nel sistema Expert.
    """
    # Ritorna gli expert con valori default (0)
    return [
        ExpertPerformance(
            name="literal",
            display_name="LiteralExpert",
        ),
        ExpertPerformance(
            name="systemic",
            display_name="SystemicExpert",
        ),
        ExpertPerformance(
            name="principles",
            display_name="PrinciplesExpert",
        ),
        ExpertPerformance(
            name="precedent",
            display_name="PrecedentExpert",
        ),
    ]


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.get("/performance", response_model=ExpertPerformanceResponse)
async def get_expert_performance(
    period_days: int = Query(7, ge=1, le=90, description="Periodo in giorni"),
) -> ExpertPerformanceResponse:
    """
    Recupera metriche performance per ogni expert.

    NOTA: Attualmente restituisce valori default.
          Per popolare i dati, implementare tracking metriche.

    Args:
        period_days: Periodo di riferimento in giorni

    Returns:
        ExpertPerformanceResponse con performance di tutti gli expert

    Example:
        >>> GET /api/v1/expert-metrics/performance?period_days=7
        {
          "experts": [
            {"name": "literal", "accuracy": 0.0, "latency_ms": 0.0, ...},
            ...
          ],
          "total_queries": 0
        }
    """
    log.info("Getting expert performance", period_days=period_days)

    experts = _get_expert_list()

    return ExpertPerformanceResponse(
        experts=experts,
        period_days=period_days,
        total_queries=0,
        last_updated=datetime.now().isoformat(),
    )


@router.get("/queries/stats", response_model=QueryStatsResponse)
async def get_query_stats(
    period_days: int = Query(7, ge=1, le=90, description="Periodo in giorni"),
) -> QueryStatsResponse:
    """
    Recupera statistiche classificazione query.

    NOTA: Attualmente restituisce valori vuoti.

    Args:
        period_days: Periodo di riferimento

    Returns:
        QueryStatsResponse con breakdown per tipo

    Example:
        >>> GET /api/v1/expert-metrics/queries/stats
        {
          "total_queries": 0,
          "by_type": []
        }
    """
    log.info("Getting query stats", period_days=period_days)

    return QueryStatsResponse(
        total_queries=0,
        by_type=[],
        avg_latency_ms=0.0,
        avg_confidence=0.0,
        period_days=period_days,
    )


@router.get("/queries/recent", response_model=RecentQueriesResponse)
async def get_recent_queries(
    limit: int = Query(10, ge=1, le=50, description="Numero massimo di query"),
    offset: int = Query(0, ge=0, description="Offset per paginazione"),
) -> RecentQueriesResponse:
    """
    Recupera query recenti con summary.

    NOTA: Attualmente restituisce lista vuota.

    Args:
        limit: Numero massimo di query
        offset: Offset per paginazione

    Returns:
        RecentQueriesResponse con lista query

    Example:
        >>> GET /api/v1/expert-metrics/queries/recent?limit=5
        {
          "queries": [],
          "total_count": 0,
          "has_more": false
        }
    """
    log.info("Getting recent queries", limit=limit, offset=offset)

    return RecentQueriesResponse(
        queries=[],
        total_count=0,
        has_more=False,
    )


@router.get("/trace/{trace_id}", response_model=ReasoningTrace)
async def get_reasoning_trace(trace_id: str) -> ReasoningTrace:
    """
    Recupera reasoning trace completo per una query.

    NOTA: Attualmente restituisce trace vuoto.
          Per popolare i dati, implementare persistenza dei trace.

    Args:
        trace_id: ID del trace

    Returns:
        ReasoningTrace con contributi di ogni expert

    Raises:
        HTTPException: Se trace non trovato

    Example:
        >>> GET /api/v1/expert-metrics/trace/trace_abc123
        {
          "trace_id": "trace_abc123",
          "query": "",
          "contributions": [],
          "final_confidence": 0.0
        }
    """
    log.info("Getting reasoning trace", trace_id=trace_id)

    # TODO: Implementare persistenza trace nel sistema Expert
    # Per ora ritorna un trace vuoto con l'ID richiesto
    return ReasoningTrace(
        trace_id=trace_id,
        query="",
        timestamp=datetime.now().isoformat(),
        contributions=[],
        aggregation_method="weighted_consensus",
        final_confidence=0.0,
        confidence_ci=(0.0, 0.0),
        synthesis="",
        mode="",
        has_alternatives=False,
        total_latency_ms=0,
        sources_count=0,
    )


@router.get("/aggregation", response_model=AggregationStats)
async def get_aggregation_stats() -> AggregationStats:
    """
    Recupera statistiche di aggregazione delle risposte.

    NOTA: Attualmente restituisce valori default.

    Returns:
        AggregationStats con agreement rate, divergence, confidence

    Example:
        >>> GET /api/v1/expert-metrics/aggregation
        {
          "method": "weighted_consensus",
          "agreement_rate": 0.0,
          "divergence_count": 0,
          "avg_confidence": 0.0
        }
    """
    log.info("Getting aggregation stats")

    return AggregationStats(
        method="weighted_consensus",
        total_responses=0,
        agreement_rate=0.0,
        divergence_count=0,
        divergence_rate=0.0,
        avg_confidence=0.0,
        confidence_ci=(0.0, 0.0),
        avg_experts_per_query=0.0,
    )
