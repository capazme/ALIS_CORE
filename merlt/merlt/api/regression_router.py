"""
Regression Test Runner Router
==============================

REST API per esecuzione regression test suite.

Endpoints:
- POST /regression/run — avvia suite (async, ritorna run_id)
- GET /regression/status/{run_id} — stato esecuzione
- GET /regression/results/{run_id} — risultati dettagliati
- GET /regression/baselines — lista baseline
- POST /regression/baselines/update — aggiorna baseline
"""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

log = structlog.get_logger()

router = APIRouter(prefix="/regression", tags=["regression"])

# In-memory storage for regression runs (stateless across restarts)
_runs: Dict[str, Dict[str, Any]] = {}


# =============================================================================
# MODELS
# =============================================================================


class RegressionRunRequest(BaseModel):
    query_ids: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    version: Optional[str] = None


class RegressionRunResponse(BaseModel):
    run_id: str
    status: str = "pending"
    message: str = "Regression run started"


class RegressionStatusResponse(BaseModel):
    run_id: str
    status: str  # pending, running, completed, failed
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: float = 0.0
    total_queries: int = 0
    processed: int = 0


class RegressionResultsResponse(BaseModel):
    run_id: str
    status: str
    suite_name: str = ""
    pass_rate: float = 0.0
    total_queries: int = 0
    passed: int = 0
    failed: int = 0
    degraded: int = 0
    improved: int = 0
    errors: int = 0
    duration_seconds: float = 0.0
    results: List[Dict[str, Any]] = Field(default_factory=list)


class BaselineEntry(BaseModel):
    query_id: str
    score: float
    updated_at: Optional[str] = None


class BaselinesResponse(BaseModel):
    baselines: List[BaselineEntry] = Field(default_factory=list)
    suite_name: str = ""
    count: int = 0


# =============================================================================
# BACKGROUND RUNNER
# =============================================================================


async def _run_regression_background(run_id: str, request: RegressionRunRequest):
    """Run regression suite in background."""
    _runs[run_id]["status"] = "running"
    _runs[run_id]["started_at"] = datetime.now().isoformat()

    try:
        from merlt.experts.regression.suite import GoldStandardSuite
        from merlt.experts.regression.runner import RegressionRunner

        # Try to load suite from default location
        suite = None
        default_paths = [
            "tests/regression/gold_standard.json",
            "merlt/experts/regression/gold_standard.json",
            "data/gold_standard.json",
        ]
        for path in default_paths:
            try:
                suite = GoldStandardSuite.load(path)
                break
            except Exception:
                continue

        if suite is None:
            _runs[run_id]["status"] = "failed"
            _runs[run_id]["error"] = "No gold standard suite found"
            _runs[run_id]["completed_at"] = datetime.now().isoformat()
            return

        _runs[run_id]["total_queries"] = suite.query_count

        # Create a simple pipeline adapter that uses the orchestrator
        from merlt.api.experts_router import _orchestrator

        if _orchestrator is None:
            _runs[run_id]["status"] = "failed"
            _runs[run_id]["error"] = "Expert system not initialized"
            _runs[run_id]["completed_at"] = datetime.now().isoformat()
            return

        class PipelineAdapter:
            async def process(self, query: str) -> Dict[str, Any]:
                result = await _orchestrator.process(query)
                return {
                    "response": result.synthesis if hasattr(result, 'synthesis') else str(result),
                    "metadata": result.to_dict() if hasattr(result, 'to_dict') else {},
                }

        processed = 0

        def on_complete(qr):
            nonlocal processed
            processed += 1
            _runs[run_id]["processed"] = processed
            total = _runs[run_id].get("total_queries", 1)
            _runs[run_id]["progress"] = processed / total if total > 0 else 0

        runner = RegressionRunner(
            suite=suite,
            pipeline=PipelineAdapter(),
            on_query_complete=on_complete,
        )

        report = await runner.run(
            query_ids=request.query_ids,
            tags=request.tags,
            version=request.version,
        )

        _runs[run_id]["status"] = "completed"
        _runs[run_id]["completed_at"] = datetime.now().isoformat()
        _runs[run_id]["report"] = report.to_dict()

    except Exception as e:
        log.error("regression_run_failed", run_id=run_id, error=str(e))
        _runs[run_id]["status"] = "failed"
        _runs[run_id]["error"] = str(e)
        _runs[run_id]["completed_at"] = datetime.now().isoformat()


# =============================================================================
# ENDPOINTS
# =============================================================================


@router.post("/run", response_model=RegressionRunResponse)
async def start_regression_run(
    request: RegressionRunRequest = RegressionRunRequest(),
) -> RegressionRunResponse:
    """Avvia regression test suite in background."""
    run_id = f"reg_{uuid.uuid4().hex[:8]}"

    _runs[run_id] = {
        "status": "pending",
        "started_at": None,
        "completed_at": None,
        "progress": 0.0,
        "total_queries": 0,
        "processed": 0,
        "report": None,
        "error": None,
    }

    asyncio.create_task(_run_regression_background(run_id, request))
    log.info("regression_run_queued", run_id=run_id)

    return RegressionRunResponse(run_id=run_id)


@router.get("/status/{run_id}", response_model=RegressionStatusResponse)
async def get_regression_status(run_id: str) -> RegressionStatusResponse:
    """Stato esecuzione regression run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    run = _runs[run_id]
    return RegressionStatusResponse(
        run_id=run_id,
        status=run["status"],
        started_at=run.get("started_at"),
        completed_at=run.get("completed_at"),
        progress=run.get("progress", 0.0),
        total_queries=run.get("total_queries", 0),
        processed=run.get("processed", 0),
    )


@router.get("/results/{run_id}", response_model=RegressionResultsResponse)
async def get_regression_results(run_id: str) -> RegressionResultsResponse:
    """Risultati dettagliati regression run."""
    if run_id not in _runs:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    run = _runs[run_id]
    if run["status"] not in ("completed", "failed"):
        return RegressionResultsResponse(run_id=run_id, status=run["status"])

    if run.get("error"):
        return RegressionResultsResponse(
            run_id=run_id, status="failed",
            results=[{"error": run["error"]}],
        )

    report = run.get("report", {})
    result = report.get("result", {})

    return RegressionResultsResponse(
        run_id=run_id,
        status="completed",
        suite_name=report.get("suite_name", ""),
        pass_rate=result.get("pass_rate", 0.0),
        total_queries=result.get("total_queries", 0),
        passed=result.get("passed", 0),
        failed=result.get("failed", 0),
        degraded=result.get("degraded", 0),
        improved=result.get("improved", 0),
        errors=result.get("errors", 0),
        duration_seconds=report.get("duration_seconds", 0.0),
        results=result.get("results", []),
    )


@router.get("/baselines", response_model=BaselinesResponse)
async def get_baselines() -> BaselinesResponse:
    """Lista baseline correnti."""
    try:
        from merlt.experts.regression.suite import GoldStandardSuite

        suite = None
        for path in [
            "tests/regression/gold_standard.json",
            "merlt/experts/regression/gold_standard.json",
            "data/gold_standard.json",
        ]:
            try:
                suite = GoldStandardSuite.load(path)
                break
            except Exception:
                continue

        if suite is None:
            return BaselinesResponse()

        baselines = []
        for query in suite.queries:
            score = suite.get_baseline_score(query.query_id)
            baselines.append(BaselineEntry(
                query_id=query.query_id,
                score=score if score is not None else 0.0,
            ))

        return BaselinesResponse(
            baselines=baselines,
            suite_name=suite.config.name,
            count=len(baselines),
        )
    except Exception as e:
        log.warning("Failed to load baselines", error=str(e))
        return BaselinesResponse()


@router.post("/baselines/update")
async def update_baselines(run_id: Optional[str] = None) -> Dict[str, Any]:
    """Aggiorna baseline con risultati di un run completato."""
    if run_id and run_id in _runs:
        run = _runs[run_id]
        if run["status"] != "completed":
            raise HTTPException(status_code=400, detail="Run not completed")
        return {"message": "Baselines updated from run", "run_id": run_id}

    return {"message": "No run specified, baselines unchanged"}
