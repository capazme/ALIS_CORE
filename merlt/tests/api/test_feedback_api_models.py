"""
Test Feedback API Models
=========================

Test per i modelli Pydantic dell'API feedback:
- InteractionModel
- BatchInteractionsModel
- ExplicitFeedbackModel
- SessionFinalizeModel
"""

import pytest
from datetime import datetime, timezone

from merlt.api.feedback_api import (
    InteractionModel,
    BatchInteractionsModel,
    ExplicitFeedbackModel,
    SessionFinalizeModel,
)


# =============================================================================
# INTERACTION MODEL TESTS
# =============================================================================

class TestInteractionModel:
    """Test InteractionModel validation."""

    def test_valid_interaction(self):
        """Test creazione interazione valida."""
        interaction = InteractionModel(
            user_id="user-123",
            interaction_type="bookmark_add",
            article_urn="urn:norma:cc:art1337",
        )
        assert interaction.user_id == "user-123"
        assert interaction.interaction_type == "bookmark_add"
        assert interaction.article_urn == "urn:norma:cc:art1337"

    def test_interaction_with_timestamp(self):
        """Test interazione con timestamp."""
        interaction = InteractionModel(
            user_id="user-123",
            interaction_type="highlight_create",
            timestamp="2024-01-15T10:30:00+00:00",
        )
        assert interaction.timestamp == "2024-01-15T10:30:00+00:00"

    def test_interaction_with_metadata(self):
        """Test interazione con metadata."""
        interaction = InteractionModel(
            user_id="user-123",
            interaction_type="cross_ref_found",
            metadata={"source_page": 5, "highlight_color": "yellow"},
        )
        assert interaction.metadata["source_page"] == 5

    def test_interaction_with_trace_id(self):
        """Test interazione con trace_id."""
        interaction = InteractionModel(
            user_id="user-123",
            interaction_type="doctrine_read",
            trace_id="trace-456",
        )
        assert interaction.trace_id == "trace-456"


# =============================================================================
# BATCH INTERACTIONS MODEL TESTS
# =============================================================================

class TestBatchInteractionsModel:
    """Test BatchInteractionsModel validation."""

    def test_valid_batch(self):
        """Test batch valido."""
        batch = BatchInteractionsModel(
            user_id="user-123",
            user_authority=0.7,
            interactions=[
                InteractionModel(user_id="user-123", interaction_type="bookmark_add"),
                InteractionModel(user_id="user-123", interaction_type="highlight_create"),
            ],
        )
        assert len(batch.interactions) == 2
        assert batch.user_authority == 0.7

    def test_batch_default_authority(self):
        """Test authority default."""
        batch = BatchInteractionsModel(
            user_id="user-123",
            interactions=[],
        )
        assert batch.user_authority == 0.5

    def test_batch_with_trace_id(self):
        """Test batch con trace_id."""
        batch = BatchInteractionsModel(
            user_id="user-123",
            trace_id="trace-789",
            interactions=[],
        )
        assert batch.trace_id == "trace-789"

    def test_batch_authority_bounds(self):
        """Test authority deve essere tra 0 e 1."""
        with pytest.raises(ValueError):
            BatchInteractionsModel(
                user_id="user-123",
                user_authority=1.5,
                interactions=[],
            )


# =============================================================================
# EXPLICIT FEEDBACK MODEL TESTS
# =============================================================================

class TestExplicitFeedbackModel:
    """Test ExplicitFeedbackModel validation."""

    def test_valid_explicit_feedback(self):
        """Test feedback esplicito valido."""
        feedback = ExplicitFeedbackModel(
            user_id="user-123",
            user_authority=0.8,
            precision=0.9,
            clarity=0.85,
            usefulness=0.95,
        )
        assert feedback.precision == 0.9
        assert feedback.clarity == 0.85
        assert feedback.usefulness == 0.95

    def test_explicit_feedback_partial(self):
        """Test feedback con solo alcuni campi."""
        feedback = ExplicitFeedbackModel(
            user_id="user-123",
            precision=0.7,
        )
        assert feedback.precision == 0.7
        assert feedback.clarity is None
        assert feedback.usefulness is None

    def test_explicit_feedback_all_fields(self):
        """Test feedback con tutti i campi."""
        feedback = ExplicitFeedbackModel(
            user_id="user-123",
            precision=0.8,
            recall=0.7,
            missing_sources=["urn:norma:cc:art1338"],
            ranking_quality=0.9,
            legal_soundness=0.85,
            logical_coherence=0.9,
            citation_quality=0.8,
            clarity=0.95,
            completeness=0.9,
            usefulness=0.92,
            user_satisfaction=0.88,
        )
        assert feedback.missing_sources == ["urn:norma:cc:art1338"]

    def test_explicit_feedback_bounds(self):
        """Test valori devono essere tra 0 e 1."""
        with pytest.raises(ValueError):
            ExplicitFeedbackModel(
                user_id="user-123",
                precision=1.5,
            )


# =============================================================================
# SESSION FINALIZE MODEL TESTS
# =============================================================================

class TestSessionFinalizeModel:
    """Test SessionFinalizeModel validation."""

    def test_valid_session(self):
        """Test sessione valida."""
        session = SessionFinalizeModel(
            session_id="session-123",
            user_id="user-456",
            user_authority=0.7,
        )
        assert session.session_id == "session-123"
        assert session.user_id == "user-456"

    def test_session_with_interactions(self):
        """Test sessione con interazioni."""
        session = SessionFinalizeModel(
            session_id="session-123",
            user_id="user-456",
            interactions=[
                InteractionModel(user_id="user-456", interaction_type="bookmark_add"),
            ],
        )
        assert len(session.interactions) == 1

    def test_session_with_explicit_feedback(self):
        """Test sessione con feedback esplicito."""
        session = SessionFinalizeModel(
            session_id="session-123",
            user_id="user-456",
            explicit_feedback={"clarity": 0.9, "usefulness": 0.85},
        )
        assert session.explicit_feedback["clarity"] == 0.9
