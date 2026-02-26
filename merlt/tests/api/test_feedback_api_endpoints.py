"""
Test Feedback API Endpoints
=============================

Test per gli endpoint FastAPI del feedback router:
- /feedback/interaction
- /feedback/batch
- /feedback/explicit
- /feedback/session
- /feedback/mappings
- Integration tests
- Error handling
- Edge cases
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from fastapi import FastAPI

from merlt.api.feedback_api import (
    router,
    InteractionModel,
    get_adapter,
)
from merlt.rlcf.external_feedback import (
    ExternalFeedbackAdapter,
    VisualexInteraction,
    PartialFeedback,
    FeedbackLevel,
    MultilevelFeedback,
    RetrievalFeedback,
    ReasoningFeedback,
    SynthesisFeedback,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_adapter():
    """Mock ExternalFeedbackAdapter."""
    adapter = MagicMock(spec=ExternalFeedbackAdapter)

    # Configure IMPLICIT_MAPPINGS for get_mappings endpoint
    adapter.IMPLICIT_MAPPINGS = {
        "bookmark_add": (FeedbackLevel.RETRIEVAL, "precision", 0.1),
        "highlight_create": (FeedbackLevel.RETRIEVAL, "precision", 0.2),
        "quicknorm_save": (FeedbackLevel.SYNTHESIS, "usefulness", 0.2),
    }

    return adapter


@pytest.fixture
def app(mock_adapter):
    """FastAPI test app with mocked adapter."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override the dependency
    app.dependency_overrides[get_adapter] = lambda: mock_adapter

    return app


@pytest.fixture
def client(app):
    """Test client."""
    return TestClient(app)


# =============================================================================
# INTERACTION ENDPOINT TESTS
# =============================================================================

class TestInteractionEndpoint:
    """Test /feedback/interaction endpoint."""

    def test_register_bookmark_interaction(self, client, mock_adapter):
        """Test registrazione interazione bookmark."""
        # Mock convert_interaction
        mock_adapter.convert_interaction.return_value = PartialFeedback(
            trace_id=None,
            level=FeedbackLevel.RETRIEVAL,
            field="precision",
            delta=0.1,
            source="implicit",
            timestamp=datetime.now(timezone.utc),
        )

        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "bookmark_add",
                "article_urn": "urn:norma:cc:art1337",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["partial_feedback"]["level"] == "retrieval"
        assert data["partial_feedback"]["field"] == "precision"

    def test_register_unknown_interaction(self, client, mock_adapter):
        """Test interazione non mappata."""
        mock_adapter.convert_interaction.return_value = None

        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "unknown_type",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["partial_feedback"] is None

    def test_register_interaction_with_timestamp(self, client, mock_adapter):
        """Test interazione con timestamp specifico."""
        mock_adapter.convert_interaction.return_value = PartialFeedback(
            trace_id=None,
            level=FeedbackLevel.SYNTHESIS,
            field="usefulness",
            delta=0.2,
            source="implicit",
            timestamp=datetime.now(timezone.utc),
        )

        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "quicknorm_save",
                "timestamp": "2024-01-15T10:30:00Z",
            },
        )

        assert response.status_code == 200


# =============================================================================
# BATCH ENDPOINT TESTS
# =============================================================================

class TestBatchEndpoint:
    """Test /feedback/batch endpoint."""

    def test_register_batch_success(self, client, mock_adapter):
        """Test registrazione batch di interazioni."""
        # Mock aggregate_session
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.8),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(usefulness=0.9),
            user_id="user-123",
            trace_id="trace-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/batch",
            json={
                "user_id": "user-123",
                "user_authority": 0.7,
                "trace_id": "trace-123",
                "interactions": [
                    {"user_id": "user-123", "interaction_type": "bookmark_add"},
                    {"user_id": "user-123", "interaction_type": "highlight_create"},
                ],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["interaction_count"] == 2
        assert data["has_retrieval"] is True
        assert data["has_synthesis"] is True

    def test_register_empty_batch(self, client, mock_adapter):
        """Test batch vuoto."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/batch",
            json={
                "user_id": "user-123",
                "interactions": [],
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["interaction_count"] == 0


# =============================================================================
# EXPLICIT FEEDBACK ENDPOINT TESTS
# =============================================================================

class TestExplicitEndpoint:
    """Test /feedback/explicit endpoint."""

    def test_register_explicit_success(self, client, mock_adapter):
        """Test registrazione feedback esplicito."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.9),
            reasoning=ReasoningFeedback(legal_soundness=0.85),
            synthesis=SynthesisFeedback(clarity=0.95, usefulness=0.9),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/explicit",
            json={
                "user_id": "user-123",
                "user_authority": 0.8,
                "precision": 0.9,
                "legal_soundness": 0.85,
                "clarity": 0.95,
                "usefulness": 0.9,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_register_explicit_partial(self, client, mock_adapter):
        """Test feedback esplicito parziale."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(clarity=0.8),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/explicit",
            json={
                "user_id": "user-123",
                "clarity": 0.8,
            },
        )

        assert response.status_code == 200


# =============================================================================
# SESSION ENDPOINT TESTS
# =============================================================================

class TestSessionEndpoint:
    """Test /feedback/session endpoint."""

    def test_finalize_session_success(self, client, mock_adapter):
        """Test finalizzazione sessione."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.85),
            reasoning=ReasoningFeedback(legal_soundness=0.8),
            synthesis=SynthesisFeedback(clarity=0.9, usefulness=0.95),
            user_id="user-123",
            trace_id="trace-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/session",
            json={
                "session_id": "session-abc",
                "user_id": "user-123",
                "user_authority": 0.7,
                "trace_id": "trace-123",
                "interactions": [
                    {"user_id": "user-123", "interaction_type": "bookmark_add"},
                ],
                "explicit_feedback": {"clarity": 0.9},
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["has_retrieval"] is True
        assert data["has_reasoning"] is True
        assert data["has_synthesis"] is True

    def test_finalize_session_implicit_only(self, client, mock_adapter):
        """Test sessione solo implicita."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.6),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/session",
            json={
                "session_id": "session-def",
                "user_id": "user-123",
                "interactions": [
                    {"user_id": "user-123", "interaction_type": "bookmark_add"},
                ],
            },
        )

        assert response.status_code == 200


# =============================================================================
# MAPPINGS ENDPOINT TESTS
# =============================================================================

class TestMappingsEndpoint:
    """Test /feedback/mappings endpoint."""

    def test_get_mappings(self, client, mock_adapter):
        """Test recupero mapping."""
        response = client.get("/api/v1/feedback/mappings")

        assert response.status_code == 200
        data = response.json()
        assert "mappings" in data
        assert "levels" in data
        assert "total_mappings" in data
        # The mock has 3 mappings configured
        assert data["total_mappings"] == 3

    def test_mappings_structure(self, client, mock_adapter):
        """Test struttura mapping."""
        response = client.get("/api/v1/feedback/mappings")

        data = response.json()
        bookmark_mapping = data["mappings"]["bookmark_add"]
        assert bookmark_mapping["level"] == "retrieval"
        assert bookmark_mapping["field"] == "precision"
        assert bookmark_mapping["delta"] == 0.1


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Test di integrazione."""

    def test_full_feedback_flow(self, client, mock_adapter):
        """Test flusso completo feedback."""
        # 1. Registra interazioni singole
        mock_adapter.convert_interaction.return_value = PartialFeedback(
            trace_id=None,
            level=FeedbackLevel.RETRIEVAL,
            field="precision",
            delta=0.1,
            source="implicit",
            timestamp=datetime.now(timezone.utc),
        )

        response1 = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "bookmark_add",
            },
        )

        assert response1.status_code == 200

        # 2. Finalizza sessione
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.8),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(clarity=0.9),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response2 = client.post(
            "/api/v1/feedback/session",
            json={
                "session_id": "session-xyz",
                "user_id": "user-123",
                "interactions": [
                    {"user_id": "user-123", "interaction_type": "bookmark_add"},
                ],
                "explicit_feedback": {"clarity": 0.9},
            },
        )

        assert response2.status_code == 200
        data = response2.json()
        assert data["success"] is True

    def test_multiple_users_sessions(self, client, mock_adapter):
        """Test sessioni multiple utenti."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-1",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        # User 1
        r1 = client.post(
            "/api/v1/feedback/batch",
            json={
                "user_id": "user-1",
                "interactions": [],
            },
        )
        # User 2
        r2 = client.post(
            "/api/v1/feedback/batch",
            json={
                "user_id": "user-2",
                "interactions": [],
            },
        )

        assert r1.status_code == 200
        assert r2.status_code == 200


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Test gestione errori."""

    def test_interaction_missing_user_id(self, client):
        """Test interazione senza user_id."""
        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "interaction_type": "bookmark_add",
            },
        )
        assert response.status_code == 422  # Validation error

    def test_interaction_missing_type(self, client):
        """Test interazione senza tipo."""
        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
            },
        )
        assert response.status_code == 422

    def test_explicit_invalid_score(self, client):
        """Test score invalido nel feedback esplicito."""
        response = client.post(
            "/api/v1/feedback/explicit",
            json={
                "user_id": "user-123",
                "precision": 2.0,  # Invalid: > 1.0
            },
        )
        assert response.status_code == 422

    def test_session_missing_session_id(self, client):
        """Test sessione senza ID."""
        response = client.post(
            "/api/v1/feedback/session",
            json={
                "user_id": "user-123",
            },
        )
        assert response.status_code == 422


# =============================================================================
# EDGE CASES TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_empty_interactions_list(self, client, mock_adapter):
        """Test batch con lista vuota."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/batch",
            json={
                "user_id": "user-123",
                "interactions": [],
            },
        )

        assert response.status_code == 200
        assert response.json()["interaction_count"] == 0

    def test_all_feedback_none(self, client, mock_adapter):
        """Test feedback tutto None."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        response = client.post(
            "/api/v1/feedback/explicit",
            json={
                "user_id": "user-123",
            },
        )

        assert response.status_code == 200

    def test_very_long_session_id(self, client, mock_adapter):
        """Test session_id molto lungo."""
        mock_feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(),
            reasoning=ReasoningFeedback(),
            synthesis=SynthesisFeedback(),
            user_id="user-123",
        )
        mock_adapter.aggregate_session.return_value = mock_feedback

        long_session_id = "s" * 1000

        response = client.post(
            "/api/v1/feedback/session",
            json={
                "session_id": long_session_id,
                "user_id": "user-123",
                "interactions": [],
            },
        )

        assert response.status_code == 200

    def test_invalid_timestamp_format(self, client, mock_adapter):
        """Test timestamp con formato invalido (fallback a now)."""
        mock_adapter.convert_interaction.return_value = None

        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "bookmark_add",
                "timestamp": "invalid-timestamp",
            },
        )

        # Should succeed with fallback to current time
        assert response.status_code == 200

    def test_unicode_in_metadata(self, client, mock_adapter):
        """Test metadata con caratteri Unicode."""
        mock_adapter.convert_interaction.return_value = None

        response = client.post(
            "/api/v1/feedback/interaction",
            json={
                "user_id": "user-123",
                "interaction_type": "bookmark_add",
                "metadata": {"note": "Annotazione con àèìòù é caratteri speciali"},
            },
        )

        assert response.status_code == 200
