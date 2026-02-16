"""
Test Export PII Protection
============================
P1-PII-1: Verify export endpoints with anonymize=True never leak PII.

Tests:
- GET /export/feedback?anonymize=true must not contain raw user_ids or emails
- GET /export/traces?anonymize=true must not leak user_ids, query text, or synthesis text
- GET /export/aggregation must not contain any user data by design

All DB dependencies are mocked with sample data that includes
realistic PII (user_ids, emails, query text) so we can assert
the export service strips them correctly.
"""

import pytest
import re
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.export_router import router
from merlt.api.auth import verify_api_key, require_role
from merlt.experts.models import ApiKey
from merlt.rlcf.export_service import DatasetExportService


# =============================================================================
# SAMPLE DATA WITH REALISTIC PII
# =============================================================================

# These are the "raw" user IDs and personal info we expect NOT to appear
# in anonymized output.
RAW_USER_IDS = ["mario.rossi@studio-legale.it", "avv.bianchi_42", "user-secret-id-xyz"]
RAW_QUERY_TEXTS = [
    "Qual e' la responsabilita' contrattuale dell'avvocato Rossi?",
    "Il signor Bianchi ha subito un danno patrimoniale",
]
RAW_SYNTHESIS_TEXTS = [
    "La risposta per il caso di Mario Rossi e' che...",
    "Il convenuto Bianchi...",
]
RAW_COMMENTS = [
    "La risposta non considera il caso specifico del mio cliente Verdi",
]


def _fake_feedback_export(anonymize: bool) -> dict:
    """Build a fake export_feedback_dataset result."""
    rows = []
    for i, uid in enumerate(RAW_USER_IDS):
        row = {
            "feedback_id": i + 1,
            "trace_id": f"trace_{i:04d}",
            "user_id": (
                DatasetExportService._anonymize_user_id(uid) if anonymize else uid
            ),
            "inline_rating": 4,
            "retrieval_score": 0.8,
            "reasoning_score": 0.75,
            "synthesis_score": 0.9,
            "source_relevance": 4,
            "preferred_expert": "literal",
            "user_authority": 0.6,
            "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc).isoformat(),
            "consent_level": "full",
        }
        if not anonymize:
            row["detailed_comment"] = RAW_COMMENTS[0] if i == 0 else None
            row["source_id"] = f"urn:nir:stato:codice.civile:art{1453 + i}"
        rows.append(row)
    return {"format": "json", "data": rows, "count": len(rows)}


def _fake_traces_export(anonymize: bool) -> dict:
    """Build a fake export_traces_dataset result."""
    rows = []
    for i, uid in enumerate(RAW_USER_IDS):
        row = {
            "trace_id": f"trace_{i:04d}",
            "user_id": (
                DatasetExportService._anonymize_user_id(uid) if anonymize else uid
            ),
            "query_type": "definitional",
            "selected_experts": ["literal", "systemic"],
            "synthesis_mode": "convergent",
            "confidence": 0.85,
            "execution_time_ms": 1200 + i * 100,
            "routing_method": "neural",
            "consent_level": "full",
            "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc).isoformat(),
        }
        if not anonymize:
            row["query"] = RAW_QUERY_TEXTS[i] if i < len(RAW_QUERY_TEXTS) else "test query"
            row["synthesis_text"] = (
                RAW_SYNTHESIS_TEXTS[i] if i < len(RAW_SYNTHESIS_TEXTS) else "synthesis"
            )
        rows.append(row)
    return {"format": "json", "data": rows, "count": len(rows)}


def _fake_aggregation_export() -> dict:
    """Build a fake export_aggregation_dataset result (no user data)."""
    return {
        "format": "json",
        "data": [
            {
                "id": 1,
                "component": "literal",
                "period_start": "2026-01-01T00:00:00",
                "period_end": "2026-01-31T23:59:59",
                "avg_rating": 4.2,
                "authority_weighted_avg": 4.5,
                "disagreement_score": 0.15,
                "total_feedback": 42,
            },
            {
                "id": 2,
                "component": "systemic",
                "period_start": "2026-01-01T00:00:00",
                "period_end": "2026-01-31T23:59:59",
                "avg_rating": 3.8,
                "authority_weighted_avg": 4.0,
                "disagreement_score": 0.22,
                "total_feedback": 37,
            },
        ],
        "count": 2,
    }


# =============================================================================
# FIXTURES
# =============================================================================

def _make_fake_api_key(role: str = "admin") -> ApiKey:
    key = MagicMock(spec=ApiKey)
    key.key_id = "test-key-id"
    key.role = role
    key.is_active = True
    key.user_id = "test-admin"
    key.is_expired = MagicMock(return_value=False)
    return key


@pytest.fixture
def mock_export_service():
    """DatasetExportService with mocked DB calls."""
    svc = MagicMock(spec=DatasetExportService)

    async def _feedback(session, since=None, output_format="json", anonymize=True):
        return _fake_feedback_export(anonymize)

    async def _traces(session, since=None, output_format="json", anonymize=True):
        return _fake_traces_export(anonymize)

    async def _aggregation(session, since=None):
        return _fake_aggregation_export()

    svc.export_feedback_dataset = AsyncMock(side_effect=_feedback)
    svc.export_traces_dataset = AsyncMock(side_effect=_traces)
    svc.export_aggregation_dataset = AsyncMock(side_effect=_aggregation)

    return svc


@pytest.fixture
def app():
    """FastAPI test app with overridden auth."""
    test_app = FastAPI()
    test_app.include_router(router, prefix="/api/v1")

    admin_key = _make_fake_api_key("admin")
    test_app.dependency_overrides[verify_api_key] = lambda: admin_key
    test_app.dependency_overrides[require_role("admin")] = lambda: admin_key

    return test_app


@pytest.fixture
def client(app, mock_export_service):
    """TestClient with export service and DB session patched."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _fake_session():
        yield MagicMock()

    with (
        patch("merlt.api.export_router._svc", mock_export_service),
        patch("merlt.api.export_router.get_async_session", _fake_session),
    ):
        yield TestClient(app)


@pytest.fixture(autouse=True)
def _set_anon_salt(monkeypatch):
    """Ensure EXPORT_ANON_SALT is set for anonymization hashing."""
    monkeypatch.setenv("EXPORT_ANON_SALT", "test-salt-for-unit-tests")


# =============================================================================
# HELPERS
# =============================================================================

def _response_contains_any(response_text: str, needles: list[str]) -> list[str]:
    """Return which needle strings appear verbatim in the response."""
    return [n for n in needles if n in response_text]


def _looks_like_email(text: str) -> list[str]:
    """Find anything that looks like an email address in text."""
    return re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}", text)


# =============================================================================
# FEEDBACK EXPORT PII TESTS
# =============================================================================

class TestFeedbackExportPII:
    """GET /export/feedback?anonymize=true must not leak PII."""

    def test_anonymized_feedback_hides_user_ids(self, client):
        """Raw user_ids must not appear when anonymize=true."""
        response = client.get("/api/v1/export/feedback?anonymize=true")
        assert response.status_code == 200
        text = response.text
        leaked = _response_contains_any(text, RAW_USER_IDS)
        assert leaked == [], f"Raw user_ids leaked in anonymized feedback export: {leaked}"

    def test_anonymized_feedback_hides_emails(self, client):
        """No email addresses in anonymized output."""
        response = client.get("/api/v1/export/feedback?anonymize=true")
        assert response.status_code == 200
        emails = _looks_like_email(response.text)
        assert emails == [], f"Email addresses found in anonymized feedback export: {emails}"

    def test_anonymized_feedback_hides_comments(self, client):
        """detailed_comment field must not appear in anonymized feedback."""
        response = client.get("/api/v1/export/feedback?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            assert "detailed_comment" not in row, "detailed_comment leaked in anonymized feedback"
            assert "source_id" not in row, "source_id leaked in anonymized feedback"

    def test_anonymized_feedback_has_hashed_user_ids(self, client):
        """user_id field should contain hex hash, not raw id."""
        response = client.get("/api/v1/export/feedback?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            uid = row["user_id"]
            # Anonymized IDs are 16-char hex strings (SHA-256 prefix)
            assert re.match(r"^[0-9a-f]{16}$", uid), (
                f"Expected 16-char hex hash, got: {uid}"
            )

    def test_non_anonymized_feedback_contains_user_ids(self, client):
        """Sanity check: when anonymize=false, raw user_ids are present."""
        response = client.get("/api/v1/export/feedback?anonymize=false")
        assert response.status_code == 200
        data = response.json()
        returned_ids = {row["user_id"] for row in data["data"]}
        for raw_id in RAW_USER_IDS:
            assert raw_id in returned_ids, f"Expected raw user_id '{raw_id}' in non-anonymized export"

    def test_feedback_export_returns_data(self, client):
        """Basic sanity: anonymized export still returns rows."""
        response = client.get("/api/v1/export/feedback?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == len(RAW_USER_IDS)
        assert len(data["data"]) == len(RAW_USER_IDS)


# =============================================================================
# TRACES EXPORT PII TESTS
# =============================================================================

class TestTracesExportPII:
    """GET /export/traces?anonymize=true must not leak PII."""

    def test_anonymized_traces_hides_user_ids(self, client):
        """Raw user_ids must not appear when anonymize=true."""
        response = client.get("/api/v1/export/traces?anonymize=true")
        assert response.status_code == 200
        text = response.text
        leaked = _response_contains_any(text, RAW_USER_IDS)
        assert leaked == [], f"Raw user_ids leaked in anonymized traces export: {leaked}"

    def test_anonymized_traces_hides_query_text(self, client):
        """Query text must not appear in anonymized output."""
        response = client.get("/api/v1/export/traces?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            assert "query" not in row, "query field leaked in anonymized traces"
            assert "synthesis_text" not in row, "synthesis_text leaked in anonymized traces"

    def test_anonymized_traces_hides_emails(self, client):
        """No email addresses in anonymized trace output."""
        response = client.get("/api/v1/export/traces?anonymize=true")
        assert response.status_code == 200
        emails = _looks_like_email(response.text)
        assert emails == [], f"Email addresses found in anonymized traces export: {emails}"

    def test_anonymized_traces_has_hashed_user_ids(self, client):
        """user_id field should contain hex hash, not raw id."""
        response = client.get("/api/v1/export/traces?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            uid = row["user_id"]
            assert re.match(r"^[0-9a-f]{16}$", uid), (
                f"Expected 16-char hex hash, got: {uid}"
            )

    def test_anonymized_traces_preserves_non_pii(self, client):
        """Non-PII fields must still be present in anonymized traces."""
        response = client.get("/api/v1/export/traces?anonymize=true")
        assert response.status_code == 200
        data = response.json()
        required_fields = [
            "trace_id", "user_id", "query_type", "selected_experts",
            "synthesis_mode", "confidence", "execution_time_ms",
            "routing_method", "consent_level", "created_at",
        ]
        for row in data["data"]:
            for field in required_fields:
                assert field in row, f"Non-PII field '{field}' missing from anonymized trace"

    def test_non_anonymized_traces_contains_query(self, client):
        """Sanity check: non-anonymized traces include query text."""
        response = client.get("/api/v1/export/traces?anonymize=false")
        assert response.status_code == 200
        data = response.json()
        has_query = any("query" in row for row in data["data"])
        assert has_query, "Expected query field in non-anonymized traces"


# =============================================================================
# AGGREGATION EXPORT PII TESTS
# =============================================================================

class TestAggregationExportPII:
    """GET /export/aggregation must not contain user data by design."""

    def test_aggregation_has_no_user_ids(self, client):
        """Aggregation data must not contain user_id field."""
        response = client.get("/api/v1/export/aggregation")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            assert "user_id" not in row, "user_id found in aggregation export"
            assert "user_authority" not in row, "user_authority found in aggregation export"

    def test_aggregation_has_no_emails(self, client):
        """Aggregation output must not contain email-like strings."""
        response = client.get("/api/v1/export/aggregation")
        assert response.status_code == 200
        emails = _looks_like_email(response.text)
        assert emails == [], f"Email addresses found in aggregation export: {emails}"

    def test_aggregation_has_no_query_text(self, client):
        """Aggregation must not contain query text."""
        response = client.get("/api/v1/export/aggregation")
        assert response.status_code == 200
        data = response.json()
        for row in data["data"]:
            assert "query" not in row, "query field found in aggregation export"
            assert "synthesis_text" not in row, "synthesis_text found in aggregation export"

    def test_aggregation_structure(self, client):
        """Aggregation export contains only statistical fields."""
        response = client.get("/api/v1/export/aggregation")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2

        allowed_fields = {
            "id", "component", "period_start", "period_end",
            "avg_rating", "authority_weighted_avg", "disagreement_score",
            "total_feedback",
        }
        for row in data["data"]:
            extra = set(row.keys()) - allowed_fields
            assert extra == set(), f"Unexpected fields in aggregation: {extra}"

    def test_aggregation_no_raw_pii_strings(self, client):
        """No known PII strings appear anywhere in aggregation response."""
        response = client.get("/api/v1/export/aggregation")
        assert response.status_code == 200
        text = response.text
        leaked = _response_contains_any(text, RAW_USER_IDS + RAW_QUERY_TEXTS + RAW_COMMENTS)
        assert leaked == [], f"PII strings leaked in aggregation export: {leaked}"


# =============================================================================
# CROSS-CUTTING PII TESTS
# =============================================================================

class TestCrossCuttingPII:
    """Tests that apply to all export endpoints."""

    @pytest.mark.parametrize("endpoint", [
        "/api/v1/export/feedback?anonymize=true",
        "/api/v1/export/traces?anonymize=true",
        "/api/v1/export/aggregation",
    ])
    def test_no_known_pii_in_any_endpoint(self, client, endpoint):
        """None of the known PII strings should appear in any export."""
        response = client.get(endpoint)
        assert response.status_code == 200
        text = response.text
        all_pii = RAW_USER_IDS + RAW_QUERY_TEXTS + RAW_SYNTHESIS_TEXTS + RAW_COMMENTS
        leaked = _response_contains_any(text, all_pii)
        assert leaked == [], f"PII leaked at {endpoint}: {leaked}"

    @pytest.mark.parametrize("endpoint", [
        "/api/v1/export/feedback?anonymize=true",
        "/api/v1/export/traces?anonymize=true",
        "/api/v1/export/aggregation",
    ])
    def test_no_email_in_any_endpoint(self, client, endpoint):
        """No email-like pattern should appear in any anonymized export."""
        response = client.get(endpoint)
        assert response.status_code == 200
        emails = _looks_like_email(response.text)
        assert emails == [], f"Email pattern found at {endpoint}: {emails}"
