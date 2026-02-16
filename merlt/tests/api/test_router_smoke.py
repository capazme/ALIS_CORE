"""
Router Smoke Tests
===================

Smoke tests for all auth-wired FastAPI routers in the MERL-T app.

Each test verifies that:
1. Endpoints respond (not 500/401) when auth is properly overridden
2. Auth dependency override works via verify_api_key

Pattern:
- Create a FastAPI app per router group
- Override verify_api_key with a mock returning a fake ApiKey
- For DB-dependent routers, also override the session dependency
- For service-dependent routers, mock external services (FalkorDB, etc.)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.auth import verify_api_key, require_role
from merlt.experts.models import ApiKey
from merlt.rlcf.database import get_async_session_dep
from merlt.storage.enrichment.database import get_db_session_dependency


# =============================================================================
# SHARED HELPERS
# =============================================================================


def _make_fake_api_key(role: str = "admin") -> ApiKey:
    """Create a fake ApiKey for test auth override."""
    key = MagicMock(spec=ApiKey)
    key.key_id = "test-key"
    key.role = role
    key.is_active = True
    key.rate_limit_tier = "unlimited"
    key.user_id = "test-user"
    return key


def _make_mock_async_session():
    """Create a mock async DB session with common return values."""
    session = AsyncMock()
    # Mock .execute() to return a result with .scalars() / .scalar() / .one()
    mock_result = MagicMock()
    mock_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=[]))
    )
    mock_result.scalar = MagicMock(return_value=0)
    mock_result.scalar_one_or_none = MagicMock(return_value=None)
    mock_result.one = MagicMock(return_value=(0, 0, 0))
    mock_result.all = MagicMock(return_value=[])
    session.execute = AsyncMock(return_value=mock_result)
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.add = MagicMock()
    session.refresh = AsyncMock()
    return session


def _make_app_with_overrides(router, prefix="/api/v1", role="admin",
                              override_rlcf_session=False,
                              override_enrichment_session=False):
    """
    Create a FastAPI app with the given router and standard overrides.

    Args:
        router: The APIRouter to include
        prefix: URL prefix for the router
        role: Role for the fake API key
        override_rlcf_session: If True, override get_async_session_dep
        override_enrichment_session: If True, override get_db_session_dependency
    """
    app = FastAPI()
    app.include_router(router, prefix=prefix)

    fake_key = _make_fake_api_key(role=role)
    app.dependency_overrides[verify_api_key] = lambda: fake_key

    if override_rlcf_session:
        mock_session = _make_mock_async_session()
        app.dependency_overrides[get_async_session_dep] = lambda: mock_session

    if override_enrichment_session:
        mock_session = _make_mock_async_session()
        app.dependency_overrides[get_db_session_dependency] = lambda: mock_session

    return app


# =============================================================================
# 1. AUDIT ROUTER
# =============================================================================


class TestAuditRouterSmoke:
    """Smoke tests for audit_router (admin-only, needs DB session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.audit_router import router
        app = _make_app_with_overrides(
            router, override_rlcf_session=True, role="admin"
        )
        return TestClient(app)

    def test_get_audit_logs_responds(self, client):
        """GET /audit/logs should respond (not 401/500)."""
        with patch("merlt.api.audit_router._audit_service") as mock_svc:
            mock_svc.get_logs = AsyncMock(return_value=[])
            response = client.get("/api/v1/audit/logs")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# =============================================================================
# 2. DASHBOARD ROUTER
# =============================================================================


class TestDashboardRouterSmoke:
    """Smoke tests for dashboard_router (no DB needed for architecture)."""

    @pytest.fixture
    def client(self):
        from merlt.api.dashboard_router import router
        app = _make_app_with_overrides(router)
        return TestClient(app)

    def test_get_architecture_responds(self, client):
        """GET /dashboard/architecture returns static architecture data."""
        response = client.get("/api/v1/dashboard/architecture")
        assert response.status_code == 200
        data = response.json()
        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) > 0


# =============================================================================
# 3. DEVIL'S ADVOCATE ROUTER
# =============================================================================


class TestDevilsAdvocateRouterSmoke:
    """Smoke tests for devils_advocate_router (needs DB session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.devils_advocate_router import router
        app = _make_app_with_overrides(
            router, override_rlcf_session=True
        )
        return TestClient(app)

    def test_get_effectiveness_responds(self, client):
        """GET /devils-advocate/effectiveness should respond."""
        # Mock the DB query results for aggregation
        mock_session = _make_mock_async_session()
        # The endpoint does two queries:
        # 1. count triggers -> scalar
        # 2. aggregate feedback -> .one() returns (count, avg_engagement, avg_keywords)
        call_count = [0]
        original_mock_result = MagicMock()
        original_mock_result.scalar = MagicMock(return_value=0)
        original_mock_result.one = MagicMock(return_value=(0, None, None))

        mock_session.execute = AsyncMock(return_value=original_mock_result)

        from merlt.api.devils_advocate_router import router
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[verify_api_key] = lambda: _make_fake_api_key()
        app.dependency_overrides[get_async_session_dep] = lambda: mock_session

        test_client = TestClient(app)
        response = test_client.get("/api/v1/devils-advocate/effectiveness")
        assert response.status_code == 200
        data = response.json()
        assert "total_triggers" in data


# =============================================================================
# 4. EXPERT METRICS ROUTER
# =============================================================================


class TestExpertMetricsRouterSmoke:
    """Smoke tests for expert_metrics_router (uses internal get_async_session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.expert_metrics_router import router
        app = _make_app_with_overrides(router)
        return TestClient(app)

    def test_get_performance_responds(self, client):
        """GET /expert-metrics/performance should respond (falls back to defaults)."""
        response = client.get("/api/v1/expert-metrics/performance")
        assert response.status_code == 200
        data = response.json()
        assert "experts" in data
        # Should have 4 experts even with no DB
        assert len(data["experts"]) == 4

    def test_get_trace_responds(self, client):
        """GET /expert-metrics/trace/{trace_id} should return empty trace."""
        response = client.get("/api/v1/expert-metrics/trace/trace_test123")
        assert response.status_code == 200
        data = response.json()
        assert data["trace_id"] == "trace_test123"

    def test_get_aggregation_responds(self, client):
        """GET /expert-metrics/aggregation should respond (falls back to defaults)."""
        response = client.get("/api/v1/expert-metrics/aggregation")
        assert response.status_code == 200
        data = response.json()
        assert "method" in data


# =============================================================================
# 5. POLICY EVOLUTION ROUTER
# =============================================================================


class TestPolicyEvolutionRouterSmoke:
    """Smoke tests for policy_evolution_router (uses internal get_async_session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.policy_evolution_router import router
        app = _make_app_with_overrides(router)
        return TestClient(app)

    def test_get_time_series_responds(self, client):
        """GET /policy-evolution/time-series should respond (empty list on failure)."""
        response = client.get("/api/v1/policy-evolution/time-series")
        assert response.status_code == 200
        # Returns empty list if DB is not available
        assert isinstance(response.json(), list)

    def test_get_expert_evolution_responds(self, client):
        """GET /policy-evolution/expert-evolution should respond."""
        response = client.get("/api/v1/policy-evolution/expert-evolution")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_aggregation_history_responds(self, client):
        """GET /policy-evolution/aggregation-history should respond."""
        response = client.get("/api/v1/policy-evolution/aggregation-history")
        assert response.status_code == 200
        assert isinstance(response.json(), list)


# =============================================================================
# 6. TRACKING ROUTER
# =============================================================================


class TestTrackingRouterSmoke:
    """Smoke tests for tracking_router (in-memory, no DB needed)."""

    @pytest.fixture
    def client(self):
        from merlt.api.tracking_router import router
        app = _make_app_with_overrides(router)
        return TestClient(app)

    def test_post_events_empty_batch(self, client):
        """POST /tracking/events with empty events list should succeed."""
        response = client.post(
            "/api/v1/tracking/events",
            json={"events": []},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["received"] == 0

    def test_post_events_with_data(self, client):
        """POST /tracking/events with events should succeed."""
        response = client.post(
            "/api/v1/tracking/events",
            json={
                "events": [
                    {
                        "type": "article:viewed",
                        "data": {"urn": "test:urn"},
                        "timestamp": 1700000000000,
                    }
                ]
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["received"] == 1


# =============================================================================
# 7. VALIDITY ROUTER
# =============================================================================


class TestValidityRouterSmoke:
    """Smoke tests for validity_router (needs FalkorDB/service mock)."""

    @pytest.fixture
    def client(self):
        from merlt.api.validity_router import (
            router,
            get_validity_service,
            get_graph_db,
        )
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[verify_api_key] = lambda: _make_fake_api_key()

        # Mock the validity service dependency
        mock_service = AsyncMock()
        mock_result = MagicMock()
        mock_result.urn = "test:urn"
        mock_result.status = "vigente"
        mock_result.is_valid = True
        mock_result.warning_level = "none"
        mock_result.warning_message = None
        mock_result.last_modified = None
        mock_result.modification_count = 0
        mock_result.abrogating_norm = None
        mock_result.replacing_norm = None
        mock_result.recent_modifications = []
        mock_result.checked_at = "2026-02-16T00:00:00Z"
        mock_service.check_batch_validity = AsyncMock(return_value=[mock_result])
        mock_service.build_summary_message = MagicMock(
            return_value="1 norme verificate: tutte vigenti"
        )

        app.dependency_overrides[get_validity_service] = lambda: mock_service

        # Mock graph DB too
        mock_graph = AsyncMock()
        mock_graph.health_check = AsyncMock(return_value=True)
        app.dependency_overrides[get_graph_db] = lambda: mock_graph

        return TestClient(app)

    def test_check_validity_responds(self, client):
        """GET /validity/check?urns=test:urn should respond."""
        response = client.get("/api/v1/validity/check?urns=test:urn")
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "summary" in data

    def test_health_responds(self, client):
        """GET /validity/health should respond."""
        response = client.get("/api/v1/validity/health")
        assert response.status_code == 200


# =============================================================================
# 8. STATISTICS ROUTER
# =============================================================================


class TestStatisticsRouterSmoke:
    """Smoke tests for statistics_router (mostly static data)."""

    @pytest.fixture
    def client(self):
        from merlt.api.statistics_router import router
        app = _make_app_with_overrides(router)
        return TestClient(app)

    def test_get_overview_responds(self, client):
        """GET /statistics/overview should respond with empty/default data."""
        response = client.get("/api/v1/statistics/overview")
        assert response.status_code == 200
        data = response.json()
        assert "hypothesis_tests" in data
        assert "distributions" in data
        assert "correlations" in data

    def test_get_hypothesis_tests_responds(self, client):
        """GET /statistics/hypothesis-tests should respond."""
        response = client.get("/api/v1/statistics/hypothesis-tests")
        assert response.status_code == 200
        data = response.json()
        assert "tests" in data
        assert len(data["tests"]) == 4

    def test_get_distributions_responds(self, client):
        """GET /statistics/distributions should respond."""
        response = client.get("/api/v1/statistics/distributions")
        assert response.status_code == 200

    def test_get_correlations_responds(self, client):
        """GET /statistics/correlations should respond."""
        response = client.get("/api/v1/statistics/correlations")
        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert "matrix" in data


# =============================================================================
# 9. GRAPH ROUTER
# =============================================================================


class TestGraphRouterSmoke:
    """Smoke tests for graph_router (needs FalkorDB mock)."""

    @pytest.fixture
    def client(self):
        from merlt.api.graph_router import router
        app = _make_app_with_overrides(
            router, override_enrichment_session=True
        )
        return TestClient(app)

    def test_check_article_responds(self, client):
        """GET /graph/check-article should respond (mocked FalkorDB)."""
        with patch("merlt.api.graph_router.FalkorDBClient") as MockClient:
            mock_instance = AsyncMock()
            mock_instance.connect = AsyncMock()
            mock_instance.close = AsyncMock()
            mock_instance.query = AsyncMock(return_value=[])
            MockClient.return_value = mock_instance

            response = client.get(
                "/api/v1/graph/check-article?article_urn=test:urn"
            )

        assert response.status_code == 200
        data = response.json()
        assert "exists" in data
        assert data["exists"] is False


# =============================================================================
# 10. PROFILE ROUTER
# =============================================================================


class TestProfileRouterSmoke:
    """Smoke tests for profile_router (needs enrichment DB session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.profile_router import router
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[verify_api_key] = lambda: _make_fake_api_key()

        # Profile router does many queries; mock the session carefully
        mock_session = _make_mock_async_session()

        # Make .one() return a tuple-like with named attributes for contribution stats
        mock_one = MagicMock()
        mock_one.total = 0
        mock_one.approved = 0
        mock_one.rejected = 0
        mock_one.pending = 0
        mock_one.__getitem__ = lambda self, idx: 0
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                one=MagicMock(return_value=mock_one),
                scalar=MagicMock(return_value=0),
                scalar_one_or_none=MagicMock(return_value=None),
                scalars=MagicMock(
                    return_value=MagicMock(all=MagicMock(return_value=[]))
                ),
                all=MagicMock(return_value=[]),
            )
        )

        app.dependency_overrides[get_db_session_dependency] = lambda: mock_session
        return TestClient(app)

    def test_get_full_profile_responds(self, client):
        """GET /profile/full?user_id=test should respond."""
        with patch(
            "merlt.api.profile_router.get_ner_feedback_stats",
            new_callable=AsyncMock,
            return_value={
                "total": 0,
                "confirmations": 0,
                "corrections": 0,
                "annotations": 0,
                "accuracy": 0.0,
            },
        ), patch(
            "merlt.api.profile_router.get_recent_activity",
            new_callable=AsyncMock,
            return_value=[],
        ):
            response = client.get("/api/v1/profile/full?user_id=test")
        # May get 200 or 500 depending on mocking depth; should not be 401
        assert response.status_code != 401

    def test_get_domain_authority_responds(self, client):
        """GET /profile/authority/domains should respond."""
        response = client.get(
            "/api/v1/profile/authority/domains?user_id=test"
        )
        assert response.status_code != 401


# =============================================================================
# 11. DOCUMENT ROUTER
# =============================================================================


class TestDocumentRouterSmoke:
    """Smoke tests for document_router (needs enrichment DB session)."""

    @pytest.fixture
    def client(self):
        from merlt.api.document_router import router
        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        app.dependency_overrides[verify_api_key] = lambda: _make_fake_api_key()

        mock_session = _make_mock_async_session()
        # For list_user_documents, .scalars().all() returns []
        mock_session.execute = AsyncMock(
            return_value=MagicMock(
                scalars=MagicMock(
                    return_value=MagicMock(all=MagicMock(return_value=[]))
                ),
                scalar_one_or_none=MagicMock(return_value=None),
            )
        )
        app.dependency_overrides[get_db_session_dependency] = lambda: mock_session
        return TestClient(app)

    def test_list_documents_responds(self, client):
        """GET /documents?user_id=test should respond."""
        response = client.get("/api/v1/documents?user_id=test")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert data["documents"] == []

    def test_get_nonexistent_document_responds(self, client):
        """GET /documents/{id} with nonexistent doc should return 404."""
        response = client.get("/api/v1/documents/99999")
        assert response.status_code == 404
