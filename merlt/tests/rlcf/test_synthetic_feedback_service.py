"""Tests for synthetic feedback generator."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from merlt.rlcf.synthetic_feedback_service import (
    SyntheticFeedbackService,
    TEMPLATE_QUERIES,
)


@pytest.fixture
def svc():
    return SyntheticFeedbackService()


class TestSyntheticFeedbackService:
    """Test synthetic feedback generation."""

    def test_sample_rating_in_range(self, svc):
        for _ in range(100):
            rating = svc._sample_rating()
            assert 1 <= rating <= 5

    def test_sample_rating_distribution(self, svc):
        """Beta(3,2) should produce more 3-4 ratings than 1-2."""
        ratings = [svc._sample_rating() for _ in range(1000)]
        avg = sum(ratings) / len(ratings)
        # Beta(3,2) mapped to 1-5 should average around 3.4
        assert 2.5 < avg < 4.5

    def test_templates_exist(self):
        assert len(TEMPLATE_QUERIES) >= 5
        for t in TEMPLATE_QUERIES:
            assert "query" in t
            assert "domain" in t
            assert "experts" in t
            assert "difficulty" in t

    @pytest.mark.asyncio
    async def test_compute_weight_no_real_feedback(self, svc):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)

        weight = await svc._compute_weight_factor(session)
        assert weight == 1.0

    @pytest.mark.asyncio
    async def test_compute_weight_half_threshold(self, svc):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 250  # half of 500 threshold
        session.execute = AsyncMock(return_value=mock_result)

        weight = await svc._compute_weight_factor(session)
        assert weight == 0.5

    @pytest.mark.asyncio
    async def test_compute_weight_above_threshold(self, svc):
        session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1000  # above 500 threshold
        session.execute = AsyncMock(return_value=mock_result)

        weight = await svc._compute_weight_factor(session)
        assert weight == svc.MIN_SYNTHETIC_WEIGHT

    @pytest.mark.asyncio
    async def test_generate_batch(self, svc):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        # Mock weight computation
        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)

        result = await svc.generate_feedback_batch(session, count=5)
        assert result.traces_created == 5
        assert result.feedbacks_created >= 5  # at least inline + maybe detailed
        assert result.weight_factor == 1.0
        assert session.add.call_count >= 10  # 5 traces + 5+ feedbacks

    @pytest.mark.asyncio
    async def test_generate_batch_with_domain_filter(self, svc):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalar.return_value = 0
        session.execute = AsyncMock(return_value=mock_result)

        result = await svc.generate_feedback_batch(
            session, count=3, domain="civile"
        )
        assert result.traces_created == 3

    def test_to_dict(self):
        from merlt.rlcf.synthetic_feedback_service import SyntheticBatchResult
        r = SyntheticBatchResult(traces_created=10, feedbacks_created=15, weight_factor=0.75)
        d = r.to_dict()
        assert d["traces_created"] == 10
        assert d["feedbacks_created"] == 15
        assert d["weight_factor"] == 0.75
