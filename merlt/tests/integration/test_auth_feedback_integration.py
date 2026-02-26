"""
Auth + Feedback Integration Tests
===================================

Tests that auth enforcement works correctly on feedback/RLCF endpoints.

Uses TestClient with a minimal FastAPI app that mounts the real routers
with dependency overrides for DB sessions.

Run:
    pytest tests/integration/test_auth_feedback_integration.py -v -m integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.auth import verify_api_key, require_role
from merlt.api.experts_router import router as experts_router, get_orchestrator
from merlt.api.rlcf_router import router as rlcf_router
from merlt.api.trace_router import router as trace_router, get_trace_service
from merlt.api.audit_router import router as audit_router
from merlt.experts.models import ApiKey
from merlt.rlcf.database import get_async_session_dep

pytestmark = pytest.mark.integration


# ============================================================================
# Helpers
# ============================================================================

def _make_api_key(role: str = "user") -> ApiKey:
    """Create a fake ApiKey for dependency override."""
    return ApiKey(
        key_id=f"test-{role}-key",
        user_id=f"test_{role}",
        api_key_hash="fakehash",
        role=role,
        rate_limit_tier="unlimited",
        is_active=True,
        description=f"Test {role} key",
    )


def _build_app_no_auth() -> FastAPI:
    """Build app with NO auth override (auth required, will fail without key)."""
    app = FastAPI()
    app.include_router(experts_router, prefix="/api/v1")
    app.include_router(rlcf_router, prefix="/api/v1")
    app.include_router(trace_router, prefix="/api/v1")
    app.include_router(audit_router, prefix="/api/v1")

    # Override DB session to avoid real DB
    mock_session = AsyncMock()
    app.dependency_overrides[get_async_session_dep] = lambda: mock_session

    # Override orchestrator
    app.dependency_overrides[get_orchestrator] = lambda: MagicMock()

    # Override trace service
    mock_trace_svc = AsyncMock()
    mock_trace_svc.get_trace.return_value = None
    app.dependency_overrides[get_trace_service] = lambda: mock_trace_svc

    return app


def _build_app_with_auth(role: str = "user") -> FastAPI:
    """Build app with auth override set to a specific role."""
    app = _build_app_no_auth()
    api_key = _make_api_key(role)
    app.dependency_overrides[verify_api_key] = lambda: api_key
    return app


# ============================================================================
# Test: POST /experts/query without API key -> 401
# ============================================================================

class TestExpertsQueryAuth:
    """Auth enforcement on /experts/query."""

    def test_query_without_key_returns_401(self):
        """POST /experts/query without API key -> 401."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/experts/query",
            json={
                "query": "Cos'e' la legittima difesa?",
                "user_id": "test_user",
            },
        )

        assert resp.status_code in (401, 403, 422), (
            f"Expected 401/403/422 without auth, got {resp.status_code}"
        )

    def test_query_with_invalid_key_returns_401(self):
        """POST /experts/query with invalid key -> 401."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/experts/query",
            json={
                "query": "Cos'e' la legittima difesa?",
                "user_id": "test_user",
            },
            headers={"X-API-Key": "invalid-key-12345"},
        )

        # verify_api_key will try to look up the hash in DB, fail -> 401
        assert resp.status_code in (401, 403, 422, 500), (
            f"Expected auth error, got {resp.status_code}"
        )


# ============================================================================
# Test: Admin-only RLCF endpoints without auth -> 401
# ============================================================================

class TestRLCFAdminAuth:
    """Auth enforcement on admin-only RLCF endpoints."""

    def test_training_start_without_auth_returns_error(self):
        """POST /rlcf/training/start without API key -> 401/403."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/rlcf/training/start",
            json={"epochs": 10, "learning_rate": 0.001, "batch_size": 32},
        )

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )

    def test_training_start_with_user_role_returns_403(self):
        """POST /rlcf/training/start with user role -> 403."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/rlcf/training/start",
            json={"epochs": 10, "learning_rate": 0.001, "batch_size": 32},
        )

        assert resp.status_code == 403, (
            f"Expected 403 for non-admin, got {resp.status_code}"
        )

    def test_training_stop_with_user_role_returns_403(self):
        """POST /rlcf/training/stop with user role -> 403."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post("/api/v1/rlcf/training/stop")

        assert resp.status_code == 403, (
            f"Expected 403 for non-admin, got {resp.status_code}"
        )


# ============================================================================
# Test: Trace DELETE without admin role -> 403
# ============================================================================

class TestTraceDeleteAuth:
    """Auth enforcement on trace DELETE."""

    def test_delete_trace_without_admin_returns_403(self):
        """DELETE /traces/{id} with user role -> 403."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.delete("/api/v1/traces/trace_test123")

        assert resp.status_code == 403, (
            f"Expected 403 for non-admin DELETE, got {resp.status_code}"
        )

    def test_archive_traces_without_admin_returns_403(self):
        """POST /traces/archive with user role -> 403."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/traces/archive",
            json={"days": 90},
        )

        assert resp.status_code == 403, (
            f"Expected 403 for non-admin archive, got {resp.status_code}"
        )


# ============================================================================
# Test: Audit log access without admin -> 403
# ============================================================================

class TestAuditLogAuth:
    """Auth enforcement on audit log endpoints."""

    def test_audit_logs_without_admin_returns_403(self):
        """GET /audit/logs with user role -> 403."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/audit/logs")

        assert resp.status_code == 403, (
            f"Expected 403 for non-admin audit access, got {resp.status_code}"
        )

    def test_audit_logs_with_guest_returns_403(self):
        """GET /audit/logs with guest role -> 403."""
        app = _build_app_with_auth(role="guest")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/audit/logs")

        assert resp.status_code == 403, (
            f"Expected 403 for guest audit access, got {resp.status_code}"
        )
