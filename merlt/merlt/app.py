"""
MERL-T FastAPI Application
===========================

Entry point principale per l'API MERL-T.

Features:
- Live enrichment con validazione community
- Document upload & parsing
- Amendment submission (multivigenza)
- Multi-expert Q&A system
- RLCF feedback collection

Usage:
    # Development
    uvicorn merlt.app:app --reload --port 8000

    # Production
    uvicorn merlt.app:app --host 0.0.0.0 --port 8000 --workers 4
"""

import structlog
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load environment variables
env_file = Path(__file__).parent.parent / ".env"
load_dotenv(env_file)
log = structlog.get_logger()
log.info("Environment variables loaded", env_file=str(env_file))

from merlt.storage.enrichment import init_db, close_db
from merlt.api import (
    ingestion_router,
    feedback_router,
    auth_router,
    experts_router,
    enrichment_router,
    document_router,
    amendments_router,
    graph_router,
    pipeline_router,
    training_router,
    trace_router,
    validity_router,
    citation_router,
    dashboard_router,
    profile_router,
    statistics_router,
    rlcf_router,
    expert_metrics_router,
    ws_router,
    tracking_router,
    policy_evolution_router,
    export_router,
    devils_advocate_router,
    audit_router,
    circuit_breaker_router,
    regression_router,
    schedule_router,
    quarantine_router,
    api_keys_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown.

    Startup:
    - Initialize PostgreSQL connection pool (enrichment DB)
    - Log application startup

    Shutdown:
    - Close database connections
    """
    # Startup
    log.info("=" * 60)
    log.info("MERL-T API Starting...")
    log.info("=" * 60)

    ai_service = None
    try:
        # Initialize enrichment database
        await init_db(echo=False)  # Set echo=True for SQL logging in dev
        log.info("✅ Enrichment database initialized")

        # Initialize Expert System (MultiExpertOrchestrator)
        try:
            from merlt.rlcf.ai_service import OpenRouterService
            from merlt.experts.synthesizer import AdaptiveSynthesizer, SynthesisConfig
            from merlt.experts.orchestrator import MultiExpertOrchestrator, OrchestratorConfig
            from merlt.api.experts_router import initialize_expert_system

            ai_service = OpenRouterService()
            synthesizer = AdaptiveSynthesizer(
                config=SynthesisConfig(
                    convergent_threshold=0.5,
                    resolvability_weight=0.3,
                    include_disagreement_explanation=True,
                    max_alternatives=3,
                ),
                ai_service=ai_service,
            )
            orchestrator = MultiExpertOrchestrator(
                synthesizer=synthesizer,
                tools=[],
                ai_service=ai_service,
                config=OrchestratorConfig(
                    max_experts=4,
                    timeout_seconds=60,
                    parallel_execution=True,
                ),
            )
            initialize_expert_system(orchestrator)
            log.info("✅ Expert System initialized")
        except Exception as e:
            log.error("Failed to initialize Expert System", error=str(e), exc_info=True)
            log.warning("Expert System endpoints will return 503 errors")

        log.info("=" * 60)
        log.info("MERL-T API Ready")
        log.info("=" * 60)

        yield

    finally:
        # Shutdown
        log.info("=" * 60)
        log.info("MERL-T API Shutting down...")
        log.info("=" * 60)

        if ai_service is not None:
            await ai_service.close()
            log.info("✅ AI service connections closed")

        await close_db()
        log.info("✅ Database connections closed")

        log.info("=" * 60)
        log.info("MERL-T API Stopped")
        log.info("=" * 60)


# Create FastAPI app
app = FastAPI(
    title="MERL-T API",
    description="Multi-Expert Reasoning with Legal Texts",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware (configure for your frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        # Add production origins
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingestion_router, prefix="/api/v1", tags=["ingestion"])
app.include_router(feedback_router, prefix="/api/v1", tags=["feedback"])
app.include_router(auth_router, prefix="/api/v1", tags=["auth"])
app.include_router(experts_router, prefix="/api/v1", tags=["experts"])
app.include_router(enrichment_router, prefix="/api/v1", tags=["enrichment"])
app.include_router(document_router, prefix="/api/v1", tags=["documents"])
app.include_router(amendments_router, prefix="/api/v1", tags=["amendments"])
app.include_router(graph_router, prefix="/api/v1", tags=["graph"])
app.include_router(pipeline_router, prefix="/api/v1", tags=["pipeline"])
app.include_router(training_router, prefix="/api/v1", tags=["training"])
app.include_router(trace_router, prefix="/api/v1", tags=["traces"])
app.include_router(validity_router, prefix="/api/v1", tags=["validity"])
app.include_router(citation_router, prefix="/api/v1", tags=["citations"])
app.include_router(dashboard_router, prefix="/api/v1", tags=["dashboard"])
app.include_router(profile_router, prefix="/api/v1", tags=["profile"])
app.include_router(statistics_router, prefix="/api/v1", tags=["statistics"])
app.include_router(rlcf_router, prefix="/api/v1", tags=["rlcf"])
app.include_router(expert_metrics_router, prefix="/api/v1", tags=["expert-metrics"])
app.include_router(ws_router, prefix="/api/v1", tags=["websocket"])
app.include_router(tracking_router, prefix="/api/v1", tags=["tracking"])
app.include_router(policy_evolution_router, prefix="/api/v1", tags=["policy-evolution"])
app.include_router(export_router, prefix="/api/v1", tags=["export"])
app.include_router(devils_advocate_router, prefix="/api/v1", tags=["devils-advocate"])
app.include_router(audit_router, prefix="/api/v1", tags=["audit"])
app.include_router(circuit_breaker_router, prefix="/api/v1", tags=["circuit-breaker"])
app.include_router(regression_router, prefix="/api/v1", tags=["regression"])
app.include_router(schedule_router, prefix="/api/v1", tags=["ingestion-schedules"])
app.include_router(quarantine_router, prefix="/api/v1", tags=["feedback-quarantine"])
app.include_router(api_keys_router, prefix="/api/v1", tags=["api-keys"])


# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    from merlt.storage.enrichment import check_db_health

    db_healthy = await check_db_health()

    return {
        "status": "healthy" if db_healthy else "degraded",
        "database": "healthy" if db_healthy else "unhealthy",
        "version": "1.0.0",
    }


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "MERL-T API",
        "description": "Multi-Expert Reasoning with Legal Texts",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "ingestion": "/api/v1/ingestion",
            "feedback": "/api/v1/feedback",
            "auth": "/api/v1/auth",
            "experts": "/api/v1/experts",
            "enrichment": "/api/v1/enrichment",
            "documents": "/api/v1/documents",
            "amendments": "/api/v1/amendments",
            "graph": "/api/v1/graph",
            "pipeline": "/api/v1/pipeline",
            "training": "/api/v1/training",
            "traces": "/api/v1/traces",
            "validity": "/api/v1/validity",
            "citations": "/api/v1/citations",
            "dashboard": "/api/v1/dashboard",
            "profile": "/api/v1/profile",
            "statistics": "/api/v1/statistics",
            "rlcf": "/api/v1/rlcf",
            "expert-metrics": "/api/v1/expert-metrics",
            "policy-evolution": "/api/v1/policy-evolution",
            "export": "/api/v1/export",
            "devils-advocate": "/api/v1/devils-advocate",
            "audit": "/api/v1/audit",
            "circuit-breaker": "/api/v1/circuit-breaker",
            "regression": "/api/v1/regression",
            "ingestion-schedules": "/api/v1/ingestion/schedules",
            "feedback-quarantine": "/api/v1/feedback",
            "api-keys": "/api/v1/api-keys",
        },
    }


# Export for uvicorn
__all__ = ["app"]
