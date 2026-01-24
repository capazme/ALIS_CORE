"""
MERL-T API Module
=================

FastAPI routers per integrazioni esterne.

Routers:
- ingestion_api: Ingestion da fonti esterne (VisuaLex)
- feedback_api: Ricezione feedback RLCF
- auth_api: Sincronizzazione authority utente
- experts_router: Query multi-expert system
- enrichment_router: Live enrichment con validazione community
- document_router: Upload e parsing documenti utente
- amendments_router: Submission e gestione amendments (multivigenza)
- graph_router: API per visualizzazione Knowledge Graph
- pipeline_router: Monitoring pipeline ingestion/enrichment

Esempio:
    >>> from fastapi import FastAPI
    >>> from merlt.api import (
    ...     ingestion_router,
    ...     feedback_router,
    ...     auth_router,
    ...     enrichment_router,
    ...     document_router,
    ...     amendments_router,
    ...     graph_router,
    ...     pipeline_router,
    ... )
    >>>
    >>> app = FastAPI(title="MERL-T API")
    >>> app.include_router(ingestion_router, prefix="/api/v1")
    >>> app.include_router(feedback_router, prefix="/api/v1")
    >>> app.include_router(auth_router, prefix="/api/v1")
    >>> app.include_router(enrichment_router, prefix="/api/v1")
    >>> app.include_router(document_router, prefix="/api/v1")
    >>> app.include_router(amendments_router, prefix="/api/v1")
    >>> app.include_router(graph_router, prefix="/api/v1")
    >>> app.include_router(pipeline_router, prefix="/api/v1")
"""

from merlt.api.ingestion_api import router as ingestion_router
from merlt.api.feedback_api import router as feedback_router
from merlt.api.auth_api import router as auth_router
from merlt.api.experts_router import router as experts_router
from merlt.api.enrichment_router import router as enrichment_router
from merlt.api.document_router import router as document_router, amendments_router
from merlt.api.graph_router import router as graph_router
from merlt.api.pipeline_router import router as pipeline_router
from merlt.api.training_router import router as training_router

__all__ = [
    "ingestion_router",
    "feedback_router",
    "auth_router",
    "experts_router",
    "enrichment_router",
    "document_router",
    "amendments_router",
    "graph_router",
    "pipeline_router",
    "training_router",
]
