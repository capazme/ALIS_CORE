"""
Test suite for AffinityUpdateService
======================================

Tests for:
- _clamp and _default_affinity
- update_from_source_feedback
- update_implicit_from_expert_feedback
- get_affinity_stats
- _find_experts_for_source

Pattern: class-based, @pytest.mark.asyncio, AsyncMock with side_effect.

Example:
    pytest tests/rlcf/test_affinity_service.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from merlt.rlcf.affinity_service import AffinityUpdateService, EXPERT_TYPES


def _make_trace(**kwargs):
    """Factory for mock QATrace objects."""
    trace = MagicMock()
    trace.full_trace = kwargs.get("full_trace", None)
    trace.selected_experts = kwargs.get("selected_experts", None)
    trace.sources = kwargs.get("sources", None)
    return trace


def _make_feedback(**kwargs):
    """Factory for mock QAFeedback objects."""
    fb = MagicMock()
    fb.source_id = kwargs.get("source_id", None)
    fb.source_relevance = kwargs.get("source_relevance", None)
    fb.inline_rating = kwargs.get("inline_rating", None)
    fb.retrieval_score = kwargs.get("retrieval_score", None)
    fb.user_authority = kwargs.get("user_authority", None)
    return fb


def _make_bridge_entry(urn, affinity=None):
    """Factory for mock BridgeTableEntry objects."""
    entry = MagicMock()
    entry.graph_node_urn = urn
    entry.expert_affinity = affinity
    return entry


def _mock_session_with_entries(entries):
    """Create a mock AsyncSession returning given bridge entries."""
    session = AsyncMock()
    result_mock = MagicMock()
    result_mock.scalars.return_value.all.return_value = entries
    session.execute = AsyncMock(return_value=result_mock)
    session.flush = AsyncMock()
    return session


# =============================================================================
# TEST CLAMP AND DEFAULTS
# =============================================================================


class TestClampAndDefaults:
    """Tests for _clamp and _default_affinity."""

    def test_clamp_within(self):
        svc = AffinityUpdateService()
        assert svc._clamp(0.5) == 0.5

    def test_clamp_below(self):
        svc = AffinityUpdateService()
        assert svc._clamp(0.01) == 0.1

    def test_clamp_above(self):
        svc = AffinityUpdateService()
        assert svc._clamp(1.5) == 0.95

    def test_default_affinity(self):
        svc = AffinityUpdateService()
        result = svc._default_affinity()
        assert len(result) == 4
        for expert in EXPERT_TYPES:
            assert result[expert] == 0.5


# =============================================================================
# TEST UPDATE FROM SOURCE FEEDBACK
# =============================================================================


class TestUpdateFromSourceFeedback:
    """Tests for update_from_source_feedback."""

    @pytest.mark.asyncio
    async def test_no_source_urn(self):
        """No source_id → returns None."""
        svc = AffinityUpdateService()
        trace = _make_trace()
        fb = _make_feedback(source_id=None, source_relevance=4)
        session = AsyncMock()

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is None

    @pytest.mark.asyncio
    async def test_no_relevance(self):
        """No source_relevance → returns None."""
        svc = AffinityUpdateService()
        trace = _make_trace()
        fb = _make_feedback(source_id="urn:test", source_relevance=None)
        session = AsyncMock()

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is None

    @pytest.mark.asyncio
    @patch("merlt.rlcf.affinity_service.AffinityUpdateService._find_experts_for_source")
    async def test_no_bridge_entries(self, mock_find):
        """No bridge entries → returns None (but no error)."""
        mock_find.return_value = ["literal"]
        svc = AffinityUpdateService()
        trace = _make_trace(full_trace={"expert_results": {}})
        fb = _make_feedback(source_id="urn:test", source_relevance=4)
        session = _mock_session_with_entries([])

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is None

    @pytest.mark.asyncio
    async def test_single_entry_update(self):
        """Single entry → formula: new = old + 0.3*(target - old)."""
        svc = AffinityUpdateService()
        initial_affinity = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:test", dict(initial_affinity))

        trace = _make_trace(
            full_trace={
                "expert_results": {
                    "literal": {"sources": [{"source_id": "urn:test"}]},
                }
            }
        )
        fb = _make_feedback(source_id="urn:test", source_relevance=5)  # target = 1.0
        session = _mock_session_with_entries([entry])

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is not None
        # old=0.5, target=1.0, lr=0.3 → new = 0.5 + 0.3*(1.0 - 0.5) = 0.65
        assert abs(result["literal"] - 0.65) < 0.001
        # systemic not used → stays 0.5
        assert result["systemic"] == 0.5

    @pytest.mark.asyncio
    async def test_multiple_entries(self):
        """Multiple bridge entries → both updated."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry1 = _make_bridge_entry("urn:test", dict(initial))
        entry2 = _make_bridge_entry("urn:test", dict(initial))

        trace = _make_trace(
            full_trace={
                "expert_results": {
                    "literal": {"sources": [{"source_id": "urn:test"}]}
                }
            }
        )
        fb = _make_feedback(source_id="urn:test", source_relevance=5)
        session = _mock_session_with_entries([entry1, entry2])

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is not None
        # Both entries should have been updated
        assert entry1.expert_affinity["literal"] != 0.5
        assert entry2.expert_affinity["literal"] != 0.5

    @pytest.mark.asyncio
    async def test_experts_from_trace(self):
        """Correct experts extracted from trace."""
        result = AffinityUpdateService._find_experts_for_source(
            {
                "expert_results": {
                    "literal": {"sources": [{"source_id": "urn:art1"}]},
                    "systemic": {"sources": [{"source_id": "urn:art2"}]},
                    "principles": {"sources": [{"article_urn": "urn:art1"}]},
                }
            },
            "urn:art1",
        )
        assert "literal" in result
        assert "principles" in result
        assert "systemic" not in result

    @pytest.mark.asyncio
    async def test_fallback_to_selected_experts(self):
        """No expert found in trace → fallback to selected_experts."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:test", dict(initial))

        trace = _make_trace(
            full_trace={"expert_results": {}},
            selected_experts=["systemic", "principles"],
        )
        fb = _make_feedback(source_id="urn:test", source_relevance=5)
        session = _mock_session_with_entries([entry])

        result = await svc.update_from_source_feedback(session, trace, fb)

        assert result is not None
        # systemic and principles should be updated, not literal
        assert result["systemic"] != 0.5
        assert result["principles"] != 0.5
        assert result["literal"] == 0.5

    @pytest.mark.asyncio
    async def test_clamping_high(self):
        """target=1.0 with high old → cap 0.95."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.9, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:test", dict(initial))

        trace = _make_trace(
            full_trace={
                "expert_results": {
                    "literal": {"sources": [{"source_id": "urn:test"}]}
                }
            }
        )
        fb = _make_feedback(source_id="urn:test", source_relevance=5)  # target=1.0
        session = _mock_session_with_entries([entry])

        result = await svc.update_from_source_feedback(session, trace, fb)

        # old=0.9, target=1.0, lr=0.3 → new = 0.9 + 0.3*(1.0 - 0.9) = 0.93
        assert result["literal"] <= 0.95
        assert abs(result["literal"] - 0.93) < 0.001

    @pytest.mark.asyncio
    async def test_clamping_low(self):
        """target=0.0 with low old → cap 0.1."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.15, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:test", dict(initial))

        trace = _make_trace(
            full_trace={
                "expert_results": {
                    "literal": {"sources": [{"source_id": "urn:test"}]}
                }
            }
        )
        fb = _make_feedback(source_id="urn:test", source_relevance=1)  # target=0.0
        session = _mock_session_with_entries([entry])

        result = await svc.update_from_source_feedback(session, trace, fb)

        # old=0.15, target=0.0, lr=0.3 → new = 0.15 + 0.3*(0.0 - 0.15) = 0.105
        assert result["literal"] >= 0.1
        assert abs(result["literal"] - 0.105) < 0.001


# =============================================================================
# TEST UPDATE IMPLICIT FROM EXPERT FEEDBACK
# =============================================================================


class TestUpdateImplicitFromExpertFeedback:
    """Tests for update_implicit_from_expert_feedback."""

    @pytest.mark.asyncio
    async def test_no_sources(self):
        """No trace.sources → early return."""
        svc = AffinityUpdateService()
        trace = _make_trace(sources=None)
        fb = _make_feedback(inline_rating=5)
        session = AsyncMock()

        await svc.update_implicit_from_expert_feedback(session, trace, fb, "literal")

        # Should not call session.execute
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_target_from_inline_rating(self):
        """inline_rating=4 → target=0.75."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:art1", dict(initial))

        trace = _make_trace(sources=[{"article_urn": "urn:art1"}])
        fb = _make_feedback(inline_rating=4)  # target = (4-1)/4 = 0.75

        session = _mock_session_with_entries([entry])

        await svc.update_implicit_from_expert_feedback(session, trace, fb, "literal")

        # target=0.75, old=0.5, lr=0.1 → new = 0.5 + 0.1*(0.75 - 0.5) = 0.525
        assert abs(entry.expert_affinity["literal"] - 0.525) < 0.001

    @pytest.mark.asyncio
    async def test_target_from_retrieval_score(self):
        """retrieval_score used when no inline_rating."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:art1", dict(initial))

        trace = _make_trace(sources=[{"article_urn": "urn:art1"}])
        fb = _make_feedback(inline_rating=None, retrieval_score=0.9)

        session = _mock_session_with_entries([entry])

        await svc.update_implicit_from_expert_feedback(session, trace, fb, "literal")

        # target=0.9, old=0.5, lr=0.1 → new = 0.5 + 0.1*(0.9 - 0.5) = 0.54
        assert abs(entry.expert_affinity["literal"] - 0.54) < 0.001

    @pytest.mark.asyncio
    async def test_neutrality_filter(self):
        """target=0.52 → abs(0.52 - 0.5) < 0.1 → skip."""
        svc = AffinityUpdateService()
        trace = _make_trace(sources=[{"article_urn": "urn:art1"}])
        fb = _make_feedback(inline_rating=None, retrieval_score=0.52)

        session = AsyncMock()

        await svc.update_implicit_from_expert_feedback(session, trace, fb, "literal")

        # Should not perform any DB operations
        session.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_implicit_lr(self):
        """Implicit uses base LR (0.1), NOT 0.3."""
        svc = AffinityUpdateService()
        initial = {"literal": 0.5, "systemic": 0.5, "principles": 0.5, "precedent": 0.5}
        entry = _make_bridge_entry("urn:art1", dict(initial))

        trace = _make_trace(sources=[{"article_urn": "urn:art1"}])
        fb = _make_feedback(inline_rating=5)  # target = (5-1)/4 = 1.0

        session = _mock_session_with_entries([entry])

        await svc.update_implicit_from_expert_feedback(session, trace, fb, "literal")

        # target=1.0, old=0.5, lr=0.1 → new = 0.5 + 0.1*(1.0 - 0.5) = 0.55
        # If it were explicit (lr=0.3), it would be 0.65
        assert abs(entry.expert_affinity["literal"] - 0.55) < 0.001


# =============================================================================
# TEST GET AFFINITY STATS
# =============================================================================


class TestGetAffinityStats:
    """Tests for get_affinity_stats."""

    @pytest.mark.asyncio
    async def test_entry_found(self):
        """Entry with affinity → returns it."""
        svc = AffinityUpdateService()
        expected = {"literal": 0.7, "systemic": 0.3, "principles": 0.5, "precedent": 0.6}

        session = AsyncMock()
        result_mock = MagicMock()
        entry = MagicMock()
        entry.expert_affinity = expected
        result_mock.scalar_one_or_none.return_value = entry
        session.execute = AsyncMock(return_value=result_mock)

        result = await svc.get_affinity_stats(session, "urn:test")

        assert result == expected

    @pytest.mark.asyncio
    async def test_no_entry(self):
        """No entry found → default affinity."""
        svc = AffinityUpdateService()
        session = AsyncMock()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        session.execute = AsyncMock(return_value=result_mock)

        result = await svc.get_affinity_stats(session, "urn:nonexistent")

        assert result == svc._default_affinity()

    @pytest.mark.asyncio
    async def test_none_affinity(self):
        """Entry found but affinity is None → default."""
        svc = AffinityUpdateService()
        session = AsyncMock()
        result_mock = MagicMock()
        entry = MagicMock()
        entry.expert_affinity = None
        result_mock.scalar_one_or_none.return_value = entry
        session.execute = AsyncMock(return_value=result_mock)

        result = await svc.get_affinity_stats(session, "urn:test")

        assert result == svc._default_affinity()


# =============================================================================
# TEST FIND EXPERTS FOR SOURCE
# =============================================================================


class TestFindExpertsForSource:
    """Tests for _find_experts_for_source static method."""

    def test_empty_trace(self):
        result = AffinityUpdateService._find_experts_for_source(None, "urn:test")
        assert result == []

    def test_matching_source(self):
        trace = {
            "expert_results": {
                "literal": {
                    "sources": [{"source_id": "urn:art1"}, {"source_id": "urn:art2"}]
                }
            }
        }
        result = AffinityUpdateService._find_experts_for_source(trace, "urn:art1")
        assert result == ["literal"]

    def test_multiple_experts(self):
        trace = {
            "expert_results": {
                "literal": {"sources": [{"source_id": "urn:art1"}]},
                "systemic": {"sources": [{"urn": "urn:art1"}]},
                "principles": {"sources": [{"source_id": "urn:art2"}]},
            }
        }
        result = AffinityUpdateService._find_experts_for_source(trace, "urn:art1")
        assert "literal" in result
        assert "systemic" in result
        assert "principles" not in result
