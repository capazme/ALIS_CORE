"""
Graph Auth Integration Tests
==============================

Tests that auth enforcement works correctly on graph endpoints.

Uses TestClient with a minimal FastAPI app that mounts the real graph router
with dependency overrides for DB sessions and graph client.

Run:
    pytest tests/integration/test_graph_auth_integration.py -v -m integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.auth import verify_api_key
from merlt.api.graph_router import router as graph_router
from merlt.experts.models import ApiKey
from merlt.storage.enrichment import get_db_session_dependency

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
    """Build app with graph router but NO auth override."""
    app = FastAPI()
    app.include_router(graph_router, prefix="/api/v1")

    # Override DB session
    mock_session = AsyncMock()
    app.dependency_overrides[get_db_session_dependency] = lambda: mock_session

    return app


def _build_app_with_auth(role: str = "user") -> FastAPI:
    """Build app with graph router and auth set to a specific role."""
    app = _build_app_no_auth()
    api_key = _make_api_key(role)
    app.dependency_overrides[verify_api_key] = lambda: api_key
    return app


# ============================================================================
# Test: Graph endpoints without API key -> 401/422
# ============================================================================

class TestGraphEndpointsNoAuth:
    """Graph endpoints without API key should reject requests."""

    def test_check_article_without_key(self):
        """GET /graph/check-article without API key -> 401/422."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/api/v1/graph/check-article",
            params={"article_urn": "urn:test:art1"},
        )

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )

    def test_node_details_without_key(self):
        """GET /graph/node/{id} without API key -> 401/422."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/graph/node/art:1218:cc")

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )

    def test_subgraph_without_key(self):
        """GET /graph/subgraph without API key -> 401/422."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/api/v1/graph/subgraph",
            params={"root_urn": "urn:test:art1"},
        )

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )

    def test_graph_search_without_key(self):
        """POST /graph/search without API key -> 401/422."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.post(
            "/api/v1/graph/search",
            json={"query": "responsabilita contrattuale", "limit": 5},
        )

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )

    def test_overview_without_key(self):
        """GET /graph/overview without API key -> 401/422."""
        app = _build_app_no_auth()
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get("/api/v1/graph/overview")

        assert resp.status_code in (401, 403, 422), (
            f"Expected auth error, got {resp.status_code}"
        )


# ============================================================================
# Test: Graph endpoints with valid key -> not 401
# ============================================================================

class TestGraphEndpointsWithAuth:
    """Graph endpoints with valid API key should not return 401."""

    def test_check_article_with_key_not_401(self):
        """GET /graph/check-article with valid key -> not 401."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/api/v1/graph/check-article",
            params={"article_urn": "urn:test:art1"},
        )

        # May be 500 (no real graph), but should NOT be 401/403
        assert resp.status_code not in (401, 403), (
            f"Valid key should not get auth error, got {resp.status_code}"
        )

    def test_entity_search_with_key_not_401(self):
        """GET /graph/entities/search with valid key -> not 401."""
        app = _build_app_with_auth(role="user")
        client = TestClient(app, raise_server_exceptions=False)

        resp = client.get(
            "/api/v1/graph/entities/search",
            params={"q": "buona fede"},
        )

        assert resp.status_code not in (401, 403), (
            f"Valid key should not get auth error, got {resp.status_code}"
        )
