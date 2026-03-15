"""Tests for QuarantineService."""
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, call

from merlt.rlcf.quarantine_service import QuarantineService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_feedback(
    fid=1,
    status="approved",
    inline_rating=3,
    user_authority=0.5,
    trace_id="trace_x",
    user_id="user_1",
):
    fb = MagicMock()
    fb.id = fid
    fb.trace_id = trace_id
    fb.user_id = user_id
    fb.inline_rating = inline_rating
    fb.status = status
    fb.quarantine_reason = None
    fb.flagged_at = None
    fb.flagged_by = None
    fb.reviewed_at = None
    fb.reviewed_by = None
    fb.user_authority = user_authority
    fb.created_at = datetime(2025, 3, 1, 10, 0, 0)
    return fb


def _scalar_result(obj):
    """Return a mock execute result wrapping a single object."""
    r = MagicMock()
    r.scalar_one_or_none = MagicMock(return_value=obj)
    return r


def _scalars_result(items):
    """Return a mock execute result wrapping a list via .scalars().all()."""
    r = MagicMock()
    r.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=items)))
    return r


def _count_result(count):
    r = MagicMock()
    r.scalar = MagicMock(return_value=count)
    return r


# ---------------------------------------------------------------------------
# QuarantineService._to_dict
# ---------------------------------------------------------------------------


class TestToDict:
    def test_all_fields_present(self):
        fb = _make_feedback(fid=42, status="flagged")
        d = QuarantineService._to_dict(fb)
        assert d["id"] == 42
        assert d["status"] == "flagged"
        assert d["trace_id"] == "trace_x"
        assert d["user_id"] == "user_1"
        assert d["inline_rating"] == 3
        assert d["user_authority"] == 0.5
        assert d["quarantine_reason"] is None

    def test_timestamps_serialized(self):
        fb = _make_feedback()
        fb.flagged_at = datetime(2025, 1, 10, 9, 0, 0)
        fb.reviewed_at = datetime(2025, 1, 11, 9, 0, 0)
        d = QuarantineService._to_dict(fb)
        assert isinstance(d["flagged_at"], str)
        assert isinstance(d["reviewed_at"], str)

    def test_none_timestamps_remain_none(self):
        fb = _make_feedback()
        fb.flagged_at = None
        fb.reviewed_at = None
        d = QuarantineService._to_dict(fb)
        assert d["flagged_at"] is None
        assert d["reviewed_at"] is None


# ---------------------------------------------------------------------------
# QuarantineService.flag_feedback
# ---------------------------------------------------------------------------


class TestFlagFeedback:
    @pytest.mark.asyncio
    async def test_flags_existing_feedback(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=10)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        result = await svc.flag_feedback(
            session, feedback_id=10, reason="Suspicious rating"
        )

        assert result is not None
        assert fb.status == "flagged"
        assert fb.quarantine_reason == "Suspicious rating"
        assert fb.flagged_at is not None
        assert fb.flagged_by == "admin"
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        svc = QuarantineService()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(None))

        result = await svc.flag_feedback(session, feedback_id=999, reason="x")

        assert result is None
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_flagged_by(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=5)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        await svc.flag_feedback(
            session, feedback_id=5, reason="r", flagged_by="moderator_1"
        )

        assert fb.flagged_by == "moderator_1"

    @pytest.mark.asyncio
    async def test_returns_dict_with_id(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=7)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        result = await svc.flag_feedback(session, feedback_id=7, reason="test")

        assert isinstance(result, dict)
        assert result["id"] == 7


# ---------------------------------------------------------------------------
# QuarantineService.quarantine_feedback
# ---------------------------------------------------------------------------


class TestQuarantineFeedback:
    @pytest.mark.asyncio
    async def test_quarantines_existing_feedback(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=20, status="flagged")
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        result = await svc.quarantine_feedback(
            session, feedback_id=20, reason="Confirmed spam"
        )

        assert result is not None
        assert fb.status == "quarantined"
        assert fb.quarantine_reason == "Confirmed spam"
        assert fb.reviewed_at is not None
        assert fb.reviewed_by == "admin"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        svc = QuarantineService()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(None))

        result = await svc.quarantine_feedback(session, feedback_id=0, reason="x")

        assert result is None

    @pytest.mark.asyncio
    async def test_custom_reviewed_by(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=21)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        await svc.quarantine_feedback(
            session, feedback_id=21, reason="r", reviewed_by="admin_2"
        )

        assert fb.reviewed_by == "admin_2"


# ---------------------------------------------------------------------------
# QuarantineService.approve_feedback
# ---------------------------------------------------------------------------


class TestApproveFeedback:
    @pytest.mark.asyncio
    async def test_approves_existing_feedback(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=30, status="flagged")
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        result = await svc.approve_feedback(session, feedback_id=30)

        assert result is not None
        assert fb.status == "approved"
        assert fb.reviewed_at is not None
        assert fb.reviewed_by == "admin"

    @pytest.mark.asyncio
    async def test_returns_none_when_not_found(self):
        svc = QuarantineService()
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(None))

        result = await svc.approve_feedback(session, feedback_id=9999)

        assert result is None

    @pytest.mark.asyncio
    async def test_custom_reviewed_by(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=31)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalar_result(fb))
        session.commit = AsyncMock()

        await svc.approve_feedback(
            session, feedback_id=31, reviewed_by="reviewer_xyz"
        )

        assert fb.reviewed_by == "reviewer_xyz"


# ---------------------------------------------------------------------------
# QuarantineService._list_by_status (via get_flagged / get_quarantined)
# ---------------------------------------------------------------------------


class TestListByStatus:
    @pytest.mark.asyncio
    async def test_get_flagged_returns_structure(self):
        svc = QuarantineService()
        fb1 = _make_feedback(fid=1, status="flagged")
        fb2 = _make_feedback(fid=2, status="flagged")
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[
                _count_result(2),
                _scalars_result([fb1, fb2]),
            ]
        )

        result = await svc.get_flagged(session, limit=10, offset=0)

        assert result["total"] == 2
        assert len(result["items"]) == 2
        assert result["limit"] == 10
        assert result["offset"] == 0
        assert result["has_more"] is False

    @pytest.mark.asyncio
    async def test_get_quarantined_returns_structure(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=50, status="quarantined")
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[
                _count_result(1),
                _scalars_result([fb]),
            ]
        )

        result = await svc.get_quarantined(session)

        assert result["total"] == 1
        assert len(result["items"]) == 1
        assert result["items"][0]["status"] == "quarantined"

    @pytest.mark.asyncio
    async def test_has_more_true_when_total_exceeds_page(self):
        svc = QuarantineService()
        fbs = [_make_feedback(fid=i, status="flagged") for i in range(5)]
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[
                _count_result(20),
                _scalars_result(fbs),
            ]
        )

        result = await svc.get_flagged(session, limit=5, offset=0)

        assert result["has_more"] is True

    @pytest.mark.asyncio
    async def test_empty_result(self):
        svc = QuarantineService()
        session = AsyncMock()
        session.execute = AsyncMock(
            side_effect=[
                _count_result(0),
                _scalars_result([]),
            ]
        )

        result = await svc.get_flagged(session)

        assert result["total"] == 0
        assert result["items"] == []
        assert result["has_more"] is False


# ---------------------------------------------------------------------------
# QuarantineService.auto_detect_outliers
# ---------------------------------------------------------------------------


class TestAutoDetectOutliers:
    @pytest.mark.asyncio
    async def test_flags_extreme_low_authority_items(self):
        svc = QuarantineService()

        fb1 = _make_feedback(fid=100, inline_rating=1, user_authority=0.1, status="approved")
        fb2 = _make_feedback(fid=101, inline_rating=5, user_authority=0.15, status="approved")

        session = AsyncMock()
        session.execute = AsyncMock(
            return_value=_scalars_result([fb1, fb2])
        )
        session.commit = AsyncMock()

        result = await svc.auto_detect_outliers(session)

        assert result["flagged_count"] == 2
        assert result["flagged_by"] == "auto_detect"
        assert fb1.status == "flagged"
        assert fb2.status == "flagged"
        assert fb1.flagged_at is not None
        assert "Auto-detect" in fb1.quarantine_reason
        session.commit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_outliers_no_commit(self):
        svc = QuarantineService()

        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalars_result([]))
        session.commit = AsyncMock()

        result = await svc.auto_detect_outliers(session)

        assert result["flagged_count"] == 0
        session.commit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_flagged_by(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=200, inline_rating=1, user_authority=0.05)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalars_result([fb]))
        session.commit = AsyncMock()

        result = await svc.auto_detect_outliers(session, flagged_by="scheduler")

        assert result["flagged_by"] == "scheduler"
        assert fb.flagged_by == "scheduler"

    @pytest.mark.asyncio
    async def test_quarantine_reason_contains_rating_and_authority(self):
        svc = QuarantineService()
        fb = _make_feedback(fid=300, inline_rating=5, user_authority=0.1)
        session = AsyncMock()
        session.execute = AsyncMock(return_value=_scalars_result([fb]))
        session.commit = AsyncMock()

        await svc.auto_detect_outliers(session)

        reason = fb.quarantine_reason
        assert "5" in reason  # inline_rating
        assert "0.10" in reason  # user_authority formatted
