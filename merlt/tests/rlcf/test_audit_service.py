"""Tests for audit service."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from merlt.rlcf.audit_service import AuditService


@pytest.fixture
def svc():
    return AuditService(salt="test_salt")


class TestAuditService:
    """Test audit service hashing and event creation."""

    def test_hash_actor_deterministic(self, svc):
        h1 = svc._hash_actor("user123")
        h2 = svc._hash_actor("user123")
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_actor_different_users(self, svc):
        h1 = svc._hash_actor("user123")
        h2 = svc._hash_actor("user456")
        assert h1 != h2

    def test_hash_content_none(self, svc):
        assert svc._hash_content(None) is None

    def test_hash_content_deterministic(self, svc):
        details = {"feedback_type": "inline", "rating": 5}
        h1 = svc._hash_content(details)
        h2 = svc._hash_content(details)
        assert h1 == h2
        assert len(h1) == 64

    def test_hash_content_order_independent(self, svc):
        h1 = svc._hash_content({"a": 1, "b": 2})
        h2 = svc._hash_content({"b": 2, "a": 1})
        assert h1 == h2  # sort_keys=True

    def test_chain_hash_includes_prev(self, svc):
        svc._prev_hash = None
        h1 = svc._compute_chain_hash("data1")
        svc._prev_hash = "abc123"
        h2 = svc._compute_chain_hash("data1")
        assert h1 != h2  # different prev_hash

    @pytest.mark.asyncio
    async def test_log_event(self, svc):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        entry = await svc.log_event(
            session,
            action="CREATE",
            actor_id="user123",
            resource_type="feedback",
            resource_id="42",
            details={"rating": 5},
        )

        session.add.assert_called_once()
        session.flush.assert_awaited_once()
        assert entry.action == "CREATE"
        assert entry.resource_type == "feedback"
        assert entry.resource_id == "42"
        assert entry.actor_hash == svc._hash_actor("user123")

    @pytest.mark.asyncio
    async def test_log_event_updates_prev_hash(self, svc):
        session = AsyncMock()
        session.add = MagicMock()
        session.flush = AsyncMock()

        assert svc._prev_hash is None
        await svc.log_event(session, "CREATE", "u1", "feedback", "1")
        assert svc._prev_hash is not None

        first_prev = svc._prev_hash
        await svc.log_event(session, "CREATE", "u2", "feedback", "2")
        assert svc._prev_hash != first_prev  # chain progresses
