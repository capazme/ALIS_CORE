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

    try:
        # Initialize enrichment database
        await init_db(echo=False)  # Set echo=True for SQL logging in dev
        log.info("✅ Enrichment database initialized")

        # TODO: Initialize FalkorDB connection pool (if needed)
        # TODO: Initialize Qdrant client (if needed)

        log.info("=" * 60)
        log.info("MERL-T API Ready")
        log.info("=" * 60)

        yield

    finally:
        # Shutdown
        log.info("=" * 60)
        log.info("MERL-T API Shutting down...")
        log.info("=" * 60)

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
        },
    }


# Export for uvicorn
__all__ = ["app"]
