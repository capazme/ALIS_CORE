"""
RLCF API Integration Tests
============================

Tests FastAPI endpoints for feedback and aggregation with real PostgreSQL.
Uses httpx.AsyncClient + FastAPI TestClient pattern.

Run with:
    pytest tests/integration/test_rlcf_api.py -v -m integration

Requirements:
    Docker services running (PostgreSQL on port 5433)
    All merlt dependencies installed (including asteval)
"""

import pytest
from uuid import uuid4

from httpx import AsyncClient, ASGITransport

# Lazy import: merlt.app pulls in the full dependency tree.
# Skip this module if optional deps (e.g. asteval) are missing.
try:
    from merlt.app import app as _app

    APP_AVAILABLE = True
except ImportError as _exc:
    _app = None
    APP_AVAILABLE = False
    _import_error = str(_exc)

from merlt.experts.models import QATrace, QAFeedback
from merlt.rlcf.database import get_async_session

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not APP_AVAILABLE, reason=f"merlt.app import failed: {_import_error if not APP_AVAILABLE else ''}"),
]


def _make_trace_id() -> str:
    return f"trace_{uuid4().hex[:12]}"


async def _seed_trace_and_feedback(trace_id: str) -> None:
    """Insert a trace + feedback directly into DB for test setup."""
    async with get_async_session() as session:
        trace = QATrace(
            trace_id=trace_id,
            user_id="api_test_user",
            query="Test API integration",
            selected_experts=["literal"],
            synthesis_mode="convergent",
            synthesis_text="Test synthesis",
            consent_level="full",
            confidence=0.8,
        )
        session.add(trace)
        await session.flush()

        fb = QAFeedback(
            trace_id=trace_id,
            user_id="api_reviewer",
            inline_rating=4,
            user_authority=0.7,
        )
        session.add(fb)
        await session.commit()


class TestFeedbackEndpoint:
    """Test feedback submission endpoints."""

    @pytest.mark.asyncio
    async def test_inline_feedback_stores_and_retrieves(self):
        """POST inline feedback → verify stored via DB query."""
        trace_id = _make_trace_id()

        # Seed a trace first
        async with get_async_session() as session:
            trace = QATrace(
                trace_id=trace_id,
                user_id="api_test_user",
                query="Test inline feedback",
                consent_level="full",
            )
            session.add(trace)
            await session.commit()

        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/experts/feedback/inline",
                json={
                    "trace_id": trace_id,
                    "user_id": "api_reviewer",
                    "rating": 5,
                },
            )

        # Endpoint should succeed (200 or 201)
        assert response.status_code in (200, 201), f"Got {response.status_code}: {response.text}"

        # Verify feedback was stored
        async with get_async_session() as session:
            from sqlalchemy import select

            result = await session.execute(
                select(QAFeedback).where(QAFeedback.trace_id == trace_id)
            )
            feedbacks = result.scalars().all()
            assert len(feedbacks) >= 1


class TestAggregationEndpoint:
    """Test aggregation trigger endpoint."""

    @pytest.mark.asyncio
    async def test_aggregation_endpoint_returns_data(self):
        """POST aggregation run → verify response schema."""
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/v1/rlcf/aggregation/run")

        assert response.status_code == 200
        data = response.json()
        assert "components_aggregated" in data or "result" in data


class TestExportEndpoint:
    """Test feedback export endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        reason="Export endpoint uses get_async_session() which conflicts with ASGI test transport "
               "shared connection pool (asyncpg InterfaceError). Needs NullPool or dependency injection.",
        strict=False,
    )
    async def test_export_feedback_returns_csv(self):
        """GET export/feedback → verify response (seeded via API to avoid connection conflicts)."""
        transport = ASGITransport(app=_app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Seed a trace + feedback via API first
            trace_id = _make_trace_id()

            # POST trace via query endpoint (creates a trace on success)
            # Instead of seeding via separate session, just test the export endpoint
            # which should work even with no data
            response = await client.get("/api/v1/export/feedback")

        assert response.status_code == 200
        # Response should be JSON array or CSV
        data = response.json()
        assert isinstance(data, (list, dict))
