"""
Test suite for FeedbackAggregationService
==========================================

Tests for:
- aggregate_trace_feedback
- aggregate_component_feedback
- run_periodic_aggregation

Pattern: class-based, @pytest.mark.asyncio, AsyncMock with side_effect.

Example:
    pytest tests/rlcf/test_feedback_aggregation_service.py -v
"""

import pytest
from datetime import datetime, timedelta, UTC
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.rlcf.feedback_aggregation_service import (
    FeedbackAggregationService,
    AggregatedTraceResult,
    ComponentAggregation,
    AggregationRunResult,
    VALID_COMPONENTS,
)


def _make_feedback(**kwargs):
    """Factory for mock QAFeedback objects."""
    fb = MagicMock()
    fb.inline_rating = kwargs.get("inline_rating", None)
    fb.retrieval_score = kwargs.get("retrieval_score", None)
    fb.reasoning_score = kwargs.get("reasoning_score", None)
    fb.synthesis_score = kwargs.get("synthesis_score", None)
    fb.source_id = kwargs.get("source_id", None)
    fb.source_relevance = kwargs.get("source_relevance", None)
    fb.preferred_expert = kwargs.get("preferred_expert", None)
    fb.detailed_comment = kwargs.get("detailed_comment", None)
    fb.user_authority = kwargs.get("user_authority", None)
    fb.created_at = kwargs.get("created_at", datetime.now(UTC))
    return fb


def _mock_session(feedbacks):
    """Create a mock AsyncSession returning given feedbacks."""
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = feedbacks
    session.execute = AsyncMock(return_value=result_mock)
    return session


# =============================================================================
# TEST AGGREGATE TRACE FEEDBACK
# =============================================================================


class TestAggregateTraceFeedback:
    """Tests for FeedbackAggregationService.aggregate_trace_feedback."""

    @pytest.mark.asyncio
    async def test_empty_feedback(self):
        """Empty feedback → all-None AggregatedTraceResult."""
        svc = FeedbackAggregationService()
        session = _mock_session([])

        result = await svc.aggregate_trace_feedback(session, "trace_1")

        assert isinstance(result, AggregatedTraceResult)
        assert result.trace_id == "trace_1"
        assert result.total_feedback == 0
        assert result.avg_inline_rating is None
        assert result.avg_retrieval_score is None
        assert result.avg_reasoning_score is None
        assert result.avg_synthesis_score is None
        assert result.authority_weighted_inline is None

    @pytest.mark.asyncio
    async def test_single_inline_feedback(self):
        """Single inline rating → avg=4.0, weighted=4.0."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(inline_rating=4, user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_trace_feedback(session, "trace_2")

        assert result.total_feedback == 1
        assert result.avg_inline_rating == 4.0
        assert result.authority_weighted_inline == 4.0

    @pytest.mark.asyncio
    async def test_multiple_inline_ratings(self):
        """Multiple inline ratings → verify weighted average math."""
        svc = FeedbackAggregationService()
        fb1 = _make_feedback(inline_rating=5, user_authority=2.0)
        fb2 = _make_feedback(inline_rating=3, user_authority=1.0)
        session = _mock_session([fb1, fb2])

        result = await svc.aggregate_trace_feedback(session, "trace_3")

        assert result.total_feedback == 2
        assert result.avg_inline_rating == 4.0  # (5+3)/2
        # weighted: (5*2 + 3*1) / (2+1) = 13/3 ≈ 4.333
        assert abs(result.authority_weighted_inline - 13 / 3) < 0.001

    @pytest.mark.asyncio
    async def test_mixed_score_types(self):
        """Mixed scores → selective averaging."""
        svc = FeedbackAggregationService()
        fb1 = _make_feedback(
            inline_rating=4, retrieval_score=0.8, user_authority=1.0
        )
        fb2 = _make_feedback(
            retrieval_score=0.6, reasoning_score=0.9, user_authority=1.0
        )
        session = _mock_session([fb1, fb2])

        result = await svc.aggregate_trace_feedback(session, "trace_4")

        assert result.total_feedback == 2
        assert result.avg_inline_rating == 4.0  # only fb1
        assert abs(result.avg_retrieval_score - 0.7) < 0.001  # (0.8+0.6)/2
        assert result.avg_reasoning_score == 0.9  # only fb2
        assert result.avg_synthesis_score is None

    @pytest.mark.asyncio
    async def test_zero_authority(self):
        """Zero authority → defaults to 1.0."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(inline_rating=3, user_authority=None)
        session = _mock_session([fb])

        result = await svc.aggregate_trace_feedback(session, "trace_5")

        # user_authority None → defaults to 1.0 in code (f.user_authority or 1.0)
        assert result.authority_weighted_inline == 3.0


# =============================================================================
# TEST AGGREGATE COMPONENT FEEDBACK
# =============================================================================


class TestAggregateComponentFeedback:
    """Tests for FeedbackAggregationService.aggregate_component_feedback."""

    @pytest.mark.asyncio
    async def test_expert_component_filter(self):
        """Expert component (literal) filters by preferred_expert or tag."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(inline_rating=4, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "literal")

        assert isinstance(result, ComponentAggregation)
        assert result.component == "literal"
        assert result.total_feedback == 1

    @pytest.mark.asyncio
    async def test_router_filter(self):
        """Router component filters by [router] tag."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(
            inline_rating=5, detailed_comment="[router] good", user_authority=1.0
        )
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "router")

        assert result.component == "router"
        assert result.total_feedback == 1

    @pytest.mark.asyncio
    async def test_synthesizer_filter(self):
        """Synthesizer component filters by synthesis_score IS NOT NULL."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(synthesis_score=0.8, user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "synthesizer")

        assert result.component == "synthesizer"
        assert result.total_feedback == 1

    @pytest.mark.asyncio
    async def test_bridge_filter(self):
        """Bridge component filters by source_id IS NOT NULL."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(source_id="urn:test", source_relevance=4, user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "bridge")

        assert result.component == "bridge"
        assert result.total_feedback == 1

    @pytest.mark.asyncio
    async def test_ner_filter(self):
        """NER component filters by [ner] tag."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(
            inline_rating=3, detailed_comment="[ner] inaccurate", user_authority=1.0
        )
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "ner")

        assert result.component == "ner"
        assert result.total_feedback == 1

    @pytest.mark.asyncio
    async def test_empty_feedback_component(self):
        """Empty feedback for component → zeros."""
        svc = FeedbackAggregationService()
        session = _mock_session([])

        result = await svc.aggregate_component_feedback(session, "literal")

        assert result.avg_rating == 0.0
        assert result.total_feedback == 0
        assert result.authority_weighted_avg == 0.0
        assert result.disagreement_score == 0.0
        assert result.variance == 0.0

    @pytest.mark.asyncio
    async def test_rating_normalization_inline(self):
        """Inline rating 3 → normalized to 0.5."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(inline_rating=3, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "literal")

        # inline_rating=3 → (3-1)/4 = 0.5
        assert result.avg_rating == 0.5

    @pytest.mark.asyncio
    async def test_rating_normalization_fallback(self):
        """No inline/synthesis/source → fallback 0.5."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(
            detailed_comment="[ner] test", user_authority=1.0
        )
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "ner")

        assert result.avg_rating == 0.5

    @pytest.mark.asyncio
    async def test_disagreement_one_bin(self):
        """All ratings in same bin → disagreement 0.0."""
        svc = FeedbackAggregationService()
        # Both ratings are high (0.75 and 1.0), same bin
        fb1 = _make_feedback(inline_rating=4, preferred_expert="literal", user_authority=1.0)
        fb2 = _make_feedback(inline_rating=5, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb1, fb2])

        result = await svc.aggregate_component_feedback(session, "literal")

        # Both (4-1)/4=0.75 and (5-1)/4=1.0 are > 0.66 → "high" bin → 1 non-zero bin → 0.0
        assert result.disagreement_score == 0.0

    @pytest.mark.asyncio
    async def test_disagreement_two_bins(self):
        """Ratings in 2 bins → calls calculate_disagreement."""
        svc = FeedbackAggregationService()
        # Low bin: inline_rating=1 → (1-1)/4=0.0 → "low"
        # High bin: inline_rating=5 → (5-1)/4=1.0 → "high"
        fb1 = _make_feedback(inline_rating=1, preferred_expert="literal", user_authority=1.0)
        fb2 = _make_feedback(inline_rating=5, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb1, fb2])

        result = await svc.aggregate_component_feedback(session, "literal")

        # 2 non-zero bins → calculate_disagreement called, > 0
        assert result.disagreement_score > 0.0

    @pytest.mark.asyncio
    async def test_disagreement_three_bins(self):
        """Ratings in 3 bins → max entropy."""
        svc = FeedbackAggregationService()
        # Low: 1 → 0.0, Mid: 3 → 0.5, High: 5 → 1.0
        fb1 = _make_feedback(inline_rating=1, preferred_expert="literal", user_authority=1.0)
        fb2 = _make_feedback(inline_rating=3, preferred_expert="literal", user_authority=1.0)
        fb3 = _make_feedback(inline_rating=5, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb1, fb2, fb3])

        result = await svc.aggregate_component_feedback(session, "literal")

        # Equal distribution across 3 bins → maximum entropy → 1.0
        assert abs(result.disagreement_score - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_variance_single(self):
        """Single feedback → variance 0.0."""
        svc = FeedbackAggregationService()
        fb = _make_feedback(inline_rating=4, preferred_expert="literal", user_authority=1.0)
        session = _mock_session([fb])

        result = await svc.aggregate_component_feedback(session, "literal")

        assert result.variance == 0.0


# =============================================================================
# TEST RUN PERIODIC AGGREGATION
# =============================================================================


class TestRunPeriodicAggregation:
    """Tests for FeedbackAggregationService.run_periodic_aggregation."""

    @pytest.mark.asyncio
    async def test_all_components_aggregated(self):
        """All components get aggregated."""
        svc = FeedbackAggregationService()

        fb = _make_feedback(
            inline_rating=4,
            preferred_expert="literal",
            synthesis_score=0.8,
            source_id="urn:test",
            source_relevance=4,
            detailed_comment="[ner][router] test",
            user_authority=1.0,
        )

        session = _mock_session([fb])

        result = await svc.run_periodic_aggregation(session)

        assert isinstance(result, AggregationRunResult)
        assert result.components_aggregated > 0
        assert result.total_feedback_processed > 0
        assert isinstance(result.high_disagreement_components, list)

    @pytest.mark.asyncio
    async def test_exception_per_component(self):
        """Exception in one component → continues with others."""
        svc = FeedbackAggregationService()

        call_count = 0

        async def mock_aggregate(session, component, since=None):
            nonlocal call_count
            call_count += 1
            if component == "ner":
                raise RuntimeError("DB error")
            return ComponentAggregation(
                component=component,
                avg_rating=0.5,
                total_feedback=1,
                authority_weighted_avg=0.5,
                disagreement_score=0.1,
                variance=0.01,
                period_start=datetime.now(UTC) - timedelta(days=30),
                period_end=datetime.now(UTC),
            )

        svc.aggregate_component_feedback = mock_aggregate
        session = AsyncMock()

        result = await svc.run_periodic_aggregation(session)

        # Should still process other components even though ner failed
        assert result.components_aggregated == len(VALID_COMPONENTS) - 1

    @pytest.mark.asyncio
    async def test_high_disagreement_threshold(self):
        """Disagreement > 0.4 → added to high_disagreement_components."""
        svc = FeedbackAggregationService()

        async def mock_aggregate(session, component, since=None):
            return ComponentAggregation(
                component=component,
                avg_rating=0.5,
                total_feedback=5,
                authority_weighted_avg=0.5,
                disagreement_score=0.6 if component == "router" else 0.1,
                variance=0.01,
                period_start=datetime.now(UTC) - timedelta(days=30),
                period_end=datetime.now(UTC),
            )

        svc.aggregate_component_feedback = mock_aggregate
        session = AsyncMock()

        result = await svc.run_periodic_aggregation(session)

        assert "router" in result.high_disagreement_components
        assert len(result.high_disagreement_components) == 1

    @pytest.mark.asyncio
    async def test_empty_results(self):
        """All components return empty feedback → 0 aggregated."""
        svc = FeedbackAggregationService()
        session = _mock_session([])

        result = await svc.run_periodic_aggregation(session)

        assert result.components_aggregated == 0
        assert result.total_feedback_processed == 0
        assert result.high_disagreement_components == []
