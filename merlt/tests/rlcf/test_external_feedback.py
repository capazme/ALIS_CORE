"""
Test suite per External Feedback Adapter
=========================================

Test per:
- MultilevelFeedback dataclass
- ExternalFeedbackAdapter
- FeedbackAccumulator
- Mapping interazioni

Esempio:
    pytest tests/rlcf/test_external_feedback.py -v
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any


# =============================================================================
# TEST DATACLASS
# =============================================================================

class TestRetrievalFeedback:
    """Test per RetrievalFeedback."""

    def test_retrieval_feedback_creation(self):
        """Test creazione RetrievalFeedback."""
        from merlt.rlcf.external_feedback import RetrievalFeedback

        feedback = RetrievalFeedback(
            precision=0.8,
            recall=0.7,
            ranking_quality=0.9,
        )

        assert feedback.precision == 0.8
        assert feedback.recall == 0.7
        assert feedback.ranking_quality == 0.9

    def test_retrieval_feedback_to_dict(self):
        """Test serializzazione."""
        from merlt.rlcf.external_feedback import RetrievalFeedback

        feedback = RetrievalFeedback(
            precision=0.8,
            missing_sources=["urn:art1", "urn:art2"],
        )

        result = feedback.to_dict()

        assert result["precision"] == 0.8
        assert len(result["missing_sources"]) == 2


class TestReasoningFeedback:
    """Test per ReasoningFeedback."""

    def test_reasoning_feedback_creation(self):
        """Test creazione ReasoningFeedback."""
        from merlt.rlcf.external_feedback import ReasoningFeedback

        feedback = ReasoningFeedback(
            legal_soundness=0.85,
            logical_coherence=0.9,
        )

        assert feedback.legal_soundness == 0.85
        assert feedback.logical_coherence == 0.9


class TestSynthesisFeedback:
    """Test per SynthesisFeedback."""

    def test_synthesis_feedback_creation(self):
        """Test creazione SynthesisFeedback."""
        from merlt.rlcf.external_feedback import SynthesisFeedback

        feedback = SynthesisFeedback(
            clarity=0.9,
            usefulness=0.85,
            user_satisfaction=0.8,
        )

        assert feedback.clarity == 0.9
        assert feedback.usefulness == 0.85


class TestMultilevelFeedback:
    """Test per MultilevelFeedback."""

    def test_multilevel_feedback_creation(self):
        """Test creazione MultilevelFeedback."""
        from merlt.rlcf.external_feedback import (
            MultilevelFeedback,
            RetrievalFeedback,
            ReasoningFeedback,
            SynthesisFeedback,
        )

        feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.8),
            reasoning=ReasoningFeedback(legal_soundness=0.9),
            synthesis=SynthesisFeedback(clarity=0.85),
            user_id="user-123",
            user_authority=0.7,
        )

        assert feedback.user_id == "user-123"
        assert feedback.user_authority == 0.7
        assert feedback.retrieval.precision == 0.8

    def test_multilevel_feedback_has_feedback_properties(self):
        """Test proprietà has_*_feedback."""
        from merlt.rlcf.external_feedback import (
            MultilevelFeedback,
            RetrievalFeedback,
            ReasoningFeedback,
            SynthesisFeedback,
        )

        # Empty feedback
        empty = MultilevelFeedback()
        assert empty.has_retrieval_feedback is False
        assert empty.has_reasoning_feedback is False
        assert empty.has_synthesis_feedback is False

        # With retrieval
        with_retrieval = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.8)
        )
        assert with_retrieval.has_retrieval_feedback is True
        assert with_retrieval.has_reasoning_feedback is False

        # With reasoning
        with_reasoning = MultilevelFeedback(
            reasoning=ReasoningFeedback(legal_soundness=0.9)
        )
        assert with_reasoning.has_reasoning_feedback is True

        # With synthesis
        with_synthesis = MultilevelFeedback(
            synthesis=SynthesisFeedback(user_satisfaction=0.85)
        )
        assert with_synthesis.has_synthesis_feedback is True

    def test_multilevel_feedback_to_dict(self):
        """Test serializzazione."""
        from merlt.rlcf.external_feedback import (
            MultilevelFeedback,
            RetrievalFeedback,
        )

        feedback = MultilevelFeedback(
            retrieval=RetrievalFeedback(precision=0.8),
            user_id="user-123",
            trace_id="trace-456",
        )

        result = feedback.to_dict()

        assert result["retrieval"]["precision"] == 0.8
        assert result["user_id"] == "user-123"
        assert result["trace_id"] == "trace-456"
        assert "timestamp" in result


# =============================================================================
# TEST INTERACTION
# =============================================================================

class TestVisualexInteraction:
    """Test per VisualexInteraction."""

    def test_interaction_creation(self):
        """Test creazione interaction."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
            article_urn="urn:art1453",
        )

        assert interaction.user_id == "user-123"
        assert interaction.interaction_type == "bookmark_add"
        assert interaction.article_urn == "urn:art1453"


class TestPartialFeedback:
    """Test per PartialFeedback."""

    def test_partial_feedback_creation(self):
        """Test creazione partial feedback."""
        from merlt.rlcf.external_feedback import PartialFeedback, FeedbackLevel

        partial = PartialFeedback(
            trace_id="trace-123",
            level=FeedbackLevel.RETRIEVAL,
            field="precision",
            delta=0.1,
        )

        assert partial.level == FeedbackLevel.RETRIEVAL
        assert partial.field == "precision"
        assert partial.delta == 0.1


# =============================================================================
# TEST ADAPTER
# =============================================================================

class TestExternalFeedbackAdapter:
    """Test per ExternalFeedbackAdapter."""

    @pytest.fixture
    def adapter(self):
        """Fixture per adapter."""
        from merlt.rlcf.external_feedback import ExternalFeedbackAdapter
        return ExternalFeedbackAdapter()

    def test_convert_bookmark_interaction(self, adapter):
        """Test conversione bookmark → precision."""
        from merlt.rlcf.external_feedback import (
            VisualexInteraction,
            FeedbackLevel,
        )

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
            article_urn="urn:art1453",
        )

        partial = adapter.convert_interaction(interaction)

        assert partial is not None
        assert partial.level == FeedbackLevel.RETRIEVAL
        assert partial.field == "precision"
        assert partial.delta == 0.1

    def test_convert_highlight_interaction(self, adapter):
        """Test conversione highlight → precision (higher delta)."""
        from merlt.rlcf.external_feedback import (
            VisualexInteraction,
            FeedbackLevel,
        )

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="highlight_create",
        )

        partial = adapter.convert_interaction(interaction)

        assert partial is not None
        assert partial.level == FeedbackLevel.RETRIEVAL
        assert partial.field == "precision"
        assert partial.delta == 0.2  # Higher than bookmark

    def test_convert_skip_results_negative(self, adapter):
        """Test conversione skip_results → ranking_quality (negative)."""
        from merlt.rlcf.external_feedback import (
            VisualexInteraction,
            FeedbackLevel,
        )

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="skip_results",
        )

        partial = adapter.convert_interaction(interaction)

        assert partial is not None
        assert partial.level == FeedbackLevel.RETRIEVAL
        assert partial.field == "ranking_quality"
        assert partial.delta < 0  # Negative

    def test_convert_synthesis_interaction(self, adapter):
        """Test conversione quicknorm_save → usefulness."""
        from merlt.rlcf.external_feedback import (
            VisualexInteraction,
            FeedbackLevel,
        )

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="quicknorm_save",
        )

        partial = adapter.convert_interaction(interaction)

        assert partial is not None
        assert partial.level == FeedbackLevel.SYNTHESIS
        assert partial.field == "usefulness"
        assert partial.delta == 0.2

    def test_convert_unknown_interaction(self, adapter):
        """Test tipo interazione non mappato."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="unknown_type",
        )

        partial = adapter.convert_interaction(interaction)

        assert partial is None

    def test_aggregate_session_empty(self, adapter):
        """Test aggregazione sessione vuota."""
        feedback = adapter.aggregate_session(
            interactions=[],
            user_id="user-123",
        )

        assert feedback.user_id == "user-123"
        assert feedback.has_retrieval_feedback is False

    def test_aggregate_session_single_interaction(self, adapter):
        """Test aggregazione con singola interazione."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
        )

        feedback = adapter.aggregate_session(
            interactions=[interaction],
            user_id="user-123",
        )

        # Baseline 0.5 + delta 0.1 = 0.6
        assert feedback.retrieval.precision == 0.6

    def test_aggregate_session_multiple_interactions(self, adapter):
        """Test aggregazione con multiple interazioni stesso campo."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interactions = [
            VisualexInteraction(
                user_id="user-123",
                interaction_type="bookmark_add",  # +0.1
            ),
            VisualexInteraction(
                user_id="user-123",
                interaction_type="highlight_create",  # +0.2
            ),
        ]

        feedback = adapter.aggregate_session(
            interactions=interactions,
            user_id="user-123",
        )

        # Baseline 0.5 + avg(0.1, 0.2) = 0.5 + 0.15 = 0.65
        assert feedback.retrieval.precision == 0.65

    def test_aggregate_session_with_explicit(self, adapter):
        """Test aggregazione con feedback esplicito."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
        )

        feedback = adapter.aggregate_session(
            interactions=[interaction],
            explicit_feedback={"clarity": 0.9},
            user_id="user-123",
        )

        # Explicit ha priorità
        assert feedback.synthesis.clarity == 0.9
        # Implicit ancora presente
        assert feedback.retrieval.precision == 0.6

    def test_aggregate_session_explicit_overrides_implicit(self, adapter):
        """Test che explicit overrides implicit per stesso campo."""
        from merlt.rlcf.external_feedback import VisualexInteraction

        interaction = VisualexInteraction(
            user_id="user-123",
            interaction_type="quicknorm_save",  # usefulness +0.2
        )

        feedback = adapter.aggregate_session(
            interactions=[interaction],
            explicit_feedback={"usefulness": 0.95},  # Override
            user_id="user-123",
        )

        # Explicit wins
        assert feedback.synthesis.usefulness == 0.95

    def test_convert_rating_to_score(self, adapter):
        """Test conversione rating → score."""
        assert adapter.convert_rating_to_score(5, 5) == 1.0
        assert adapter.convert_rating_to_score(1, 5) == 0.2
        assert adapter.convert_rating_to_score(3, 5) == 0.6


# =============================================================================
# TEST ACCUMULATOR
# =============================================================================

class TestFeedbackAccumulator:
    """Test per FeedbackAccumulator."""

    def test_accumulator_creation(self):
        """Test creazione accumulator."""
        from merlt.rlcf.external_feedback import FeedbackAccumulator

        acc = FeedbackAccumulator(
            user_id="user-123",
            user_authority=0.7,
        )

        assert acc.user_id == "user-123"
        assert acc.user_authority == 0.7
        assert acc.interaction_count == 0

    def test_accumulator_add_interaction(self):
        """Test aggiunta interazioni."""
        from merlt.rlcf.external_feedback import (
            FeedbackAccumulator,
            VisualexInteraction,
        )

        acc = FeedbackAccumulator(user_id="user-123")

        acc.add_interaction(VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
        ))
        acc.add_interaction(VisualexInteraction(
            user_id="user-123",
            interaction_type="highlight_create",
        ))

        assert acc.interaction_count == 2

    def test_accumulator_add_explicit(self):
        """Test aggiunta feedback esplicito."""
        from merlt.rlcf.external_feedback import FeedbackAccumulator

        acc = FeedbackAccumulator(user_id="user-123")
        acc.add_explicit({"clarity": 0.9})
        acc.add_explicit({"usefulness": 0.85})

        assert acc.explicit_feedback["clarity"] == 0.9
        assert acc.explicit_feedback["usefulness"] == 0.85

    def test_accumulator_finalize(self):
        """Test finalizzazione."""
        from merlt.rlcf.external_feedback import (
            FeedbackAccumulator,
            VisualexInteraction,
        )

        acc = FeedbackAccumulator(
            user_id="user-123",
            user_authority=0.7,
            trace_id="trace-456",
        )

        acc.add_interaction(VisualexInteraction(
            user_id="user-123",
            interaction_type="bookmark_add",
        ))
        acc.add_explicit({"clarity": 0.9})

        feedback = acc.finalize()

        assert feedback.user_id == "user-123"
        assert feedback.user_authority == 0.7
        assert feedback.trace_id == "trace-456"
        assert feedback.retrieval.precision == 0.6
        assert feedback.synthesis.clarity == 0.9

    def test_accumulator_cannot_add_after_finalize(self):
        """Test che non si può aggiungere dopo finalize."""
        from merlt.rlcf.external_feedback import (
            FeedbackAccumulator,
            VisualexInteraction,
        )

        acc = FeedbackAccumulator(user_id="user-123")
        acc.finalize()

        with pytest.raises(RuntimeError):
            acc.add_interaction(VisualexInteraction(
                user_id="user-123",
                interaction_type="bookmark_add",
            ))

    def test_accumulator_cannot_finalize_twice(self):
        """Test che non si può finalizzare due volte."""
        from merlt.rlcf.external_feedback import FeedbackAccumulator

        acc = FeedbackAccumulator(user_id="user-123")
        acc.finalize()

        with pytest.raises(RuntimeError):
            acc.finalize()


# =============================================================================
# TEST MAPPINGS
# =============================================================================

class TestMappings:
    """Test per mapping interazioni → feedback."""

    def test_all_mappings_have_valid_levels(self):
        """Verifica che tutti i mapping abbiano livelli validi."""
        from merlt.rlcf.external_feedback import (
            ExternalFeedbackAdapter,
            FeedbackLevel,
        )

        adapter = ExternalFeedbackAdapter()

        for interaction_type, (level, field, delta) in adapter.IMPLICIT_MAPPINGS.items():
            assert level in FeedbackLevel
            assert isinstance(field, str)
            assert isinstance(delta, float)

    def test_retrieval_mappings(self):
        """Test mapping retrieval."""
        from merlt.rlcf.external_feedback import (
            ExternalFeedbackAdapter,
            FeedbackLevel,
        )

        adapter = ExternalFeedbackAdapter()

        retrieval_types = [
            "bookmark_add", "highlight_create", "first_result_click",
            "skip_results", "cross_ref_found", "cross_ref_missing",
        ]

        for t in retrieval_types:
            if t in adapter.IMPLICIT_MAPPINGS:
                level, _, _ = adapter.IMPLICIT_MAPPINGS[t]
                assert level == FeedbackLevel.RETRIEVAL

    def test_synthesis_mappings(self):
        """Test mapping synthesis."""
        from merlt.rlcf.external_feedback import (
            ExternalFeedbackAdapter,
            FeedbackLevel,
        )

        adapter = ExternalFeedbackAdapter()

        synthesis_types = [
            "quicknorm_save", "dossier_add", "long_read",
            "quick_close", "search_after_ai",
        ]

        for t in synthesis_types:
            if t in adapter.IMPLICIT_MAPPINGS:
                level, _, _ = adapter.IMPLICIT_MAPPINGS[t]
                assert level == FeedbackLevel.SYNTHESIS
