"""
Consent Filtering Tests (P1-CONSENT-1)
========================================

Tests that pipeline/trace endpoints respect consent_level parameter.

Verifies:
- anonymous: query text AND user_id are redacted
- basic: query text is redacted, user_id visible
- full: all data visible, no redaction
- Stored consent_level takes effect even without caller_consent
- Most-restrictive-wins rule between stored and caller consent

Run:
    pytest tests/api/test_consent_filtering.py -v
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.auth import verify_api_key
from merlt.api.trace_router import router as trace_router, get_trace_service
from merlt.experts.models import ApiKey
from merlt.storage.trace.trace_service import (
    TraceStorageService,
    TraceSummary,
)


# ============================================================================
# Helpers
# ============================================================================

def _make_fake_api_key(role: str = "admin") -> ApiKey:
    """Create a fake ApiKey for dependency override."""
    return ApiKey(
        key_id="test-consent-key",
        user_id="test_admin",
        api_key_hash="fakehash",
        role=role,
        rate_limit_tier="unlimited",
        is_active=True,
        description="Test consent key",
        created_at=datetime.now(UTC),
        expires_at=None,
        last_used_at=None,
    )


def _make_trace_dict(
    consent_level: str = "full",
    query: str = "Cos'e' la legittima difesa?",
    user_id: str = "user_abc",
) -> dict:
    """
    Build a trace dict as returned by TraceStorageService.get_trace
    after consent filtering has been applied internally.
    """
    return {
        "trace_id": "trace_test001",
        "user_id": user_id,
        "query": query,
        "selected_experts": ["literal", "systemic"],
        "synthesis_mode": "convergent",
        "synthesis_text": "La legittima difesa e' ...",
        "sources": [{"article_urn": "urn:nir:stato:codice.penale", "relevance": 0.9}],
        "execution_time_ms": 1200,
        "full_trace": {"steps": []},
        "consent_level": consent_level,
        "query_type": "definitional",
        "confidence": 0.85,
        "routing_method": "neural",
        "is_archived": False,
        "archived_at": None,
        "created_at": datetime.now(UTC).isoformat(),
    }


def _make_trace_summary(
    query_preview: str = "Cos'e' la legittima difesa?",
    user_id: str = "user_abc",
) -> TraceSummary:
    """Build a TraceSummary for list endpoint mocks."""
    return TraceSummary(
        trace_id="trace_test001",
        user_id=user_id,
        query_preview=query_preview,
        query_type="definitional",
        synthesis_mode="convergent",
        confidence=0.85,
        execution_time_ms=1200,
        created_at=datetime.now(UTC),
        is_archived=False,
    )


def _build_app(mock_service: AsyncMock) -> FastAPI:
    """
    Build a minimal FastAPI app with trace router mounted and
    dependencies overridden.
    """
    app = FastAPI()
    app.include_router(trace_router, prefix="/api")

    # Override auth dependency
    app.dependency_overrides[verify_api_key] = lambda: _make_fake_api_key()

    # Override trace service dependency
    async def _get_mock_service():
        return mock_service

    app.dependency_overrides[get_trace_service] = _get_mock_service

    return app


# ============================================================================
# Unit tests for _apply_consent_filter (the core logic)
# ============================================================================

class TestApplyConsentFilter:
    """Direct tests on TraceStorageService._apply_consent_filter."""

    def setup_method(self):
        self.service = TraceStorageService.__new__(TraceStorageService)

    def _make_trace_orm(self, consent_level: str = "full"):
        """Create a mock QATrace ORM object."""
        trace = MagicMock()
        trace.trace_id = "trace_unit001"
        trace.user_id = "user_xyz"
        trace.query = "Qual e' la ratio dell'art. 2043 c.c.?"
        trace.selected_experts = ["literal"]
        trace.synthesis_mode = "convergent"
        trace.synthesis_text = "La ratio e' ..."
        trace.sources = []
        trace.execution_time_ms = 500
        trace.full_trace = {}
        trace.consent_level = consent_level
        trace.query_type = "interpretive"
        trace.confidence = 0.9
        trace.routing_method = "neural"
        trace.is_archived = False
        trace.archived_at = None
        trace.created_at = datetime.now(UTC)
        return trace

    def test_full_consent_no_redaction(self):
        """consent_level=full, caller_consent=full -> no redaction."""
        trace = self._make_trace_orm("full")
        result = self.service._apply_consent_filter(trace, "full")

        assert result["query"] == trace.query
        assert result["user_id"] == trace.user_id

    def test_anonymous_consent_redacts_query_and_user(self):
        """consent_level=anonymous -> query AND user_id redacted."""
        trace = self._make_trace_orm("anonymous")
        result = self.service._apply_consent_filter(trace, "full")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == "[REDACTED]"

    def test_basic_consent_redacts_query_only(self):
        """consent_level=basic -> query redacted, user_id visible."""
        trace = self._make_trace_orm("basic")
        result = self.service._apply_consent_filter(trace, "full")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == trace.user_id

    def test_caller_anonymous_overrides_full_stored(self):
        """stored=full but caller=anonymous -> anonymous wins (most restrictive)."""
        trace = self._make_trace_orm("full")
        result = self.service._apply_consent_filter(trace, "anonymous")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == "[REDACTED]"

    def test_caller_basic_overrides_full_stored(self):
        """stored=full but caller=basic -> basic wins."""
        trace = self._make_trace_orm("full")
        result = self.service._apply_consent_filter(trace, "basic")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == trace.user_id

    def test_none_caller_defaults_to_max_visibility(self):
        """caller_consent=None -> no caller restriction, stored level applies."""
        trace = self._make_trace_orm("full")
        result = self.service._apply_consent_filter(trace, None)

        assert result["query"] == trace.query
        assert result["user_id"] == trace.user_id

    def test_none_caller_with_anonymous_stored(self):
        """caller_consent=None, stored=anonymous -> anonymous applies."""
        trace = self._make_trace_orm("anonymous")
        result = self.service._apply_consent_filter(trace, None)

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == "[REDACTED]"

    def test_invalid_caller_consent_treated_as_anonymous(self):
        """Invalid caller_consent value defaults to most restrictive (anonymous)."""
        trace = self._make_trace_orm("full")
        result = self.service._apply_consent_filter(trace, "invalid_value")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == "[REDACTED]"

    def test_synthesis_text_always_visible(self):
        """synthesis_text is NOT redacted regardless of consent level."""
        trace = self._make_trace_orm("anonymous")
        result = self.service._apply_consent_filter(trace, "anonymous")

        assert result["synthesis_text"] == trace.synthesis_text

    def test_metadata_preserved_under_all_levels(self):
        """Non-PII fields (confidence, execution_time_ms, etc.) always present."""
        for level in ("anonymous", "basic", "full"):
            trace = self._make_trace_orm(level)
            result = self.service._apply_consent_filter(trace, level)

            assert result["trace_id"] == trace.trace_id
            assert result["confidence"] == trace.confidence
            assert result["execution_time_ms"] == trace.execution_time_ms
            assert result["consent_level"] == level


# ============================================================================
# API endpoint tests: GET /api/traces/{trace_id}
# ============================================================================

class TestGetTraceConsentEndpoint:
    """Test consent filtering through the GET /{trace_id} endpoint."""

    def test_get_trace_full_consent_returns_query(self):
        """GET with caller_consent=full on a full trace returns raw query."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        trace_data = _make_trace_dict(consent_level="full")
        mock_svc.get_trace.return_value = trace_data

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/trace_test001",
            params={"caller_consent": "full"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == trace_data["query"]
        assert body["user_id"] == "user_abc"

        # Verify service was called with validated consent
        mock_svc.get_trace.assert_awaited_once_with("trace_test001", consent_level="full")

    def test_get_trace_anonymous_consent_redacted(self):
        """GET with caller_consent=anonymous returns redacted data from service."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        # Service already applies filtering and returns redacted dict
        redacted = _make_trace_dict(consent_level="anonymous")
        redacted["query"] = "[REDACTED]"
        redacted["user_id"] = "[REDACTED]"
        mock_svc.get_trace.return_value = redacted

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/trace_test001",
            params={"caller_consent": "anonymous"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "[REDACTED]"
        assert body["user_id"] == "[REDACTED]"

    def test_get_trace_basic_consent_redacts_query(self):
        """GET with caller_consent=basic returns redacted query, visible user_id."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        redacted = _make_trace_dict(consent_level="basic")
        redacted["query"] = "[REDACTED]"
        mock_svc.get_trace.return_value = redacted

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/trace_test001",
            params={"caller_consent": "basic"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["query"] == "[REDACTED]"
        assert body["user_id"] == "user_abc"

    def test_get_trace_no_caller_consent(self):
        """GET without caller_consent passes None to service."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.get_trace.return_value = _make_trace_dict(consent_level="full")

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/trace_test001",
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        # Validate consent_level=None was passed
        mock_svc.get_trace.assert_awaited_once_with("trace_test001", consent_level=None)

    def test_get_trace_invalid_consent_treated_as_none(self):
        """GET with invalid caller_consent -> _validate_consent returns None."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.get_trace.return_value = _make_trace_dict(consent_level="full")

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/trace_test001",
            params={"caller_consent": "INVALID"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        # _validate_consent("INVALID") returns None
        mock_svc.get_trace.assert_awaited_once_with("trace_test001", consent_level=None)

    def test_get_trace_not_found_returns_404(self):
        """GET for non-existent trace returns 404."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.get_trace.return_value = None

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces/nonexistent",
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 404


# ============================================================================
# API endpoint tests: GET /api/traces (list)
# ============================================================================

class TestListTracesConsentEndpoint:
    """Test consent filtering in the list traces endpoint."""

    def test_list_traces_full_consent_shows_query_preview(self):
        """List with caller_consent=full shows query preview."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.list_traces.return_value = [
            _make_trace_summary(query_preview="Cos'e' la legittima difesa?")
        ]
        mock_svc.count_traces.return_value = 1

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces",
            params={"caller_consent": "full"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert body["traces"][0]["query_preview"] == "Cos'e' la legittima difesa?"

    def test_list_traces_anonymous_consent_redacted_preview(self):
        """List with caller_consent=anonymous shows redacted preview."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.list_traces.return_value = [
            _make_trace_summary(
                query_preview="[REDACTED]",
                user_id="[REDACTED]",
            )
        ]
        mock_svc.count_traces.return_value = 1

        app = _build_app(mock_svc)
        client = TestClient(app)

        resp = client.get(
            "/api/traces",
            params={"caller_consent": "anonymous"},
            headers={"X-API-Key": "fake"},
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body["traces"][0]["query_preview"] == "[REDACTED]"
        assert body["traces"][0]["user_id"] == "[REDACTED]"

    def test_list_traces_passes_validated_consent_to_service(self):
        """Verify the validated consent level is forwarded to service.list_traces."""
        mock_svc = AsyncMock(spec=TraceStorageService)
        mock_svc.list_traces.return_value = []
        mock_svc.count_traces.return_value = 0

        app = _build_app(mock_svc)
        client = TestClient(app)

        client.get(
            "/api/traces",
            params={"caller_consent": "basic"},
            headers={"X-API-Key": "fake"},
        )

        # list_traces should have been called with consent_level="basic"
        call_kwargs = mock_svc.list_traces.call_args
        assert call_kwargs.kwargs.get("consent_level") == "basic"


# ============================================================================
# Stored consent scenario: trace created with anonymous stays redacted
# ============================================================================

class TestStoredConsentPersistence:
    """
    Verify that a trace stored with consent_level='anonymous'
    does NOT expose the raw query even when the caller requests full.
    """

    def test_anonymous_stored_trace_redacted_regardless_of_caller(self):
        """
        Stored consent=anonymous + caller consent=full
        -> most restrictive wins -> query and user_id are redacted.
        """
        service = TraceStorageService.__new__(TraceStorageService)
        trace = MagicMock()
        trace.trace_id = "trace_anon_stored"
        trace.user_id = "secret_user"
        trace.query = "Secret legal question"
        trace.selected_experts = ["literal"]
        trace.synthesis_mode = "convergent"
        trace.synthesis_text = "Answer text"
        trace.sources = []
        trace.execution_time_ms = 300
        trace.full_trace = {}
        trace.consent_level = "anonymous"
        trace.query_type = "definitional"
        trace.confidence = 0.7
        trace.routing_method = "neural"
        trace.is_archived = False
        trace.archived_at = None
        trace.created_at = datetime.now(UTC)

        result = service._apply_consent_filter(trace, "full")

        assert result["query"] == "[REDACTED]", (
            "Query must be redacted for anonymous stored trace even with caller_consent=full"
        )
        assert result["user_id"] == "[REDACTED]", (
            "user_id must be redacted for anonymous stored trace even with caller_consent=full"
        )
        # Non-PII fields remain
        assert result["synthesis_text"] == "Answer text"
        assert result["confidence"] == 0.7

    def test_basic_stored_trace_redacts_query_with_full_caller(self):
        """
        Stored consent=basic + caller consent=full
        -> most restrictive wins -> query redacted, user_id visible.
        """
        service = TraceStorageService.__new__(TraceStorageService)
        trace = MagicMock()
        trace.trace_id = "trace_basic_stored"
        trace.user_id = "visible_user"
        trace.query = "A legal question"
        trace.selected_experts = ["systemic"]
        trace.synthesis_mode = "divergent"
        trace.synthesis_text = "Divergent answer"
        trace.sources = []
        trace.execution_time_ms = 400
        trace.full_trace = {}
        trace.consent_level = "basic"
        trace.query_type = "comparative"
        trace.confidence = 0.6
        trace.routing_method = "llm_fallback"
        trace.is_archived = False
        trace.archived_at = None
        trace.created_at = datetime.now(UTC)

        result = service._apply_consent_filter(trace, "full")

        assert result["query"] == "[REDACTED]"
        assert result["user_id"] == "visible_user"


# ============================================================================
# Validate consent helper
# ============================================================================

class TestValidateConsentHelper:
    """Test the _validate_consent helper from trace_router."""

    def test_valid_levels(self):
        from merlt.api.trace_router import _validate_consent

        assert _validate_consent("anonymous") == "anonymous"
        assert _validate_consent("basic") == "basic"
        assert _validate_consent("full") == "full"

    def test_none_returns_none(self):
        from merlt.api.trace_router import _validate_consent

        assert _validate_consent(None) is None

    def test_invalid_returns_none(self):
        from merlt.api.trace_router import _validate_consent

        assert _validate_consent("research") is None
        assert _validate_consent("FULL") is None
        assert _validate_consent("") is None
        assert _validate_consent("anything_else") is None
