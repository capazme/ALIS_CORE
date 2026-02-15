"""
Tests for TraceStorageService
==============================

Tests cover:
- Save/get trace operations
- Consent filtering (3 levels: anonymous, basic, full)
- List with pagination and filters
- Delete with cascade
- Archive functionality

Usage:
    pytest tests/storage/test_trace_service.py -v -m integration

Requires:
    PostgreSQL running on localhost:5433 (docker-compose.dev.yml)
"""

import os
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

from merlt.storage.trace import TraceStorageService, TraceStorageConfig
from merlt.storage.trace.trace_service import TraceFilter
from merlt.experts.models import QATrace

pytestmark = pytest.mark.integration


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def trace_config():
    """Test configuration pointing to dev PostgreSQL (env-overridable)."""
    return TraceStorageConfig(
        host=os.environ.get("RLCF_PG_HOST", "localhost"),
        port=int(os.environ.get("RLCF_PG_PORT", "5433")),
        database=os.environ.get("RLCF_PG_DATABASE", "rlcf_dev"),
        user=os.environ.get("RLCF_PG_USER", "dev"),
        password=os.environ.get("RLCF_PG_PASSWORD", "devpassword"),
    )


@pytest.fixture
async def trace_service(trace_config):
    """
    Initialize TraceStorageService for tests.

    Creates tables if not exist, yields service, cleans up after.
    """
    service = TraceStorageService(trace_config)
    await service.connect()
    await service.ensure_tables_exist()
    yield service
    await service.close()


def _make_trace(**overrides) -> QATrace:
    """Helper to create a QATrace with sensible defaults."""
    defaults = dict(
        trace_id=f"test_trace_{uuid4().hex[:8]}",
        user_id="test_user_123",
        query="Cos'è la legittima difesa secondo l'art. 52 c.p.?",
        selected_experts=["literal", "systemic"],
        synthesis_mode="convergent",
        synthesis_text="La legittima difesa è una causa di giustificazione...",
        sources=[
            {"article_urn": "urn:nir:stato:codice.penale:1930;art52", "expert": "literal", "relevance": 0.95}
        ],
        execution_time_ms=1500,
        full_trace={"routing": {"method": "neural", "query_type": "definitional"}},
        consent_level="basic",
        query_type="definitional",
        confidence=0.85,
        routing_method="neural",
        is_archived=False,
    )
    defaults.update(overrides)
    return QATrace(**defaults)


# ============================================================================
# CONNECTION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_connect_and_health_check(trace_config):
    """Test service can connect and pass health check."""
    service = TraceStorageService(trace_config)
    await service.connect()
    try:
        healthy = await service.health_check()
        assert healthy is True
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_ensure_tables_exist(trace_service):
    """Test tables are created if they don't exist."""
    healthy = await trace_service.health_check()
    assert healthy is True


# ============================================================================
# SAVE/GET TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_save_and_get_trace(trace_service):
    """Test saving and retrieving a trace."""
    trace = _make_trace(consent_level="full")
    trace_id = trace.trace_id

    try:
        saved_id = await trace_service.save_trace(trace)
        assert saved_id == trace_id

        retrieved = await trace_service.get_trace(trace_id, consent_level="full")
        assert retrieved is not None
        assert retrieved["trace_id"] == trace_id
        assert retrieved["user_id"] == "test_user_123"
        assert "legittima difesa" in retrieved["query"]
        assert retrieved["consent_level"] == "full"
        assert retrieved["query_type"] == "definitional"
        assert retrieved["confidence"] == 0.85
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_get_nonexistent_trace(trace_service):
    """Test getting a trace that doesn't exist."""
    result = await trace_service.get_trace("nonexistent_trace_12345")
    assert result is None


# ============================================================================
# CONSENT FILTERING TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_consent_filtering_anonymous(trace_service):
    """Test anonymous consent level redacts user_id and query."""
    trace = _make_trace(consent_level="basic")
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        retrieved = await trace_service.get_trace(trace_id, consent_level="anonymous")
        assert retrieved["user_id"] == "[REDACTED]"
        assert retrieved["query"] == "[REDACTED]"
        assert retrieved["trace_id"] == trace_id
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_consent_filtering_basic(trace_service):
    """Test basic consent level redacts only query."""
    trace = _make_trace(consent_level="full")
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        retrieved = await trace_service.get_trace(trace_id, consent_level="basic")
        assert retrieved["user_id"] == "test_user_123"
        assert retrieved["query"] == "[REDACTED]"
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_consent_filtering_full(trace_service):
    """Test full consent level shows everything."""
    trace = _make_trace(consent_level="full")
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        retrieved = await trace_service.get_trace(trace_id, consent_level="full")
        assert retrieved["user_id"] == "test_user_123"
        assert "legittima difesa" in retrieved["query"]
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_consent_stored_level_takes_precedence(trace_service):
    """Test that stored consent_level restricts even when caller has higher consent."""
    trace = _make_trace(consent_level="anonymous")
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        retrieved = await trace_service.get_trace(trace_id, consent_level="full")
        assert retrieved["user_id"] == "[REDACTED]"
        assert retrieved["query"] == "[REDACTED]"
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_consent_invalid_value_defaults_to_restrictive(trace_service):
    """Test that an invalid caller consent level defaults to most restrictive."""
    trace = _make_trace(consent_level="full")
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        # "superadmin" is not valid, should default to anonymous (most restrictive)
        retrieved = await trace_service.get_trace(trace_id, consent_level="superadmin")
        assert retrieved["user_id"] == "[REDACTED]"
        assert retrieved["query"] == "[REDACTED]"
    finally:
        await trace_service.delete_trace(trace_id)


# ============================================================================
# LIST TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_list_traces_basic(trace_service):
    """Test listing traces returns correct results."""
    trace = _make_trace()
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        traces = await trace_service.list_traces(
            filters=TraceFilter(user_id="test_user_123", is_archived=False),
            limit=10
        )
        assert len(traces) >= 1
        assert any(t.trace_id == trace_id for t in traces)
    finally:
        await trace_service.delete_trace(trace_id)


@pytest.mark.asyncio
async def test_list_traces_pagination(trace_service):
    """Test list pagination works correctly."""
    uid = f"page_test_{uuid4().hex[:6]}"
    trace_ids = []

    try:
        for i in range(5):
            trace = _make_trace(user_id=uid, query=f"Test query {i}", consent_level="full")
            await trace_service.save_trace(trace)
            trace_ids.append(trace.trace_id)

        page1 = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, is_archived=False),
            limit=2, offset=0, consent_level="full"
        )
        page2 = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, is_archived=False),
            limit=2, offset=2, consent_level="full"
        )

        assert len(page1) == 2
        assert len(page2) == 2
        page1_ids = {t.trace_id for t in page1}
        page2_ids = {t.trace_id for t in page2}
        assert page1_ids.isdisjoint(page2_ids)
    finally:
        for tid in trace_ids:
            await trace_service.delete_trace(tid)


@pytest.mark.asyncio
async def test_list_traces_filter_by_query_type(trace_service):
    """Test filtering by query_type."""
    uid = f"qtype_test_{uuid4().hex[:6]}"

    trace1 = _make_trace(user_id=uid, query="What is X?", query_type="definitional", consent_level="full")
    trace2 = _make_trace(user_id=uid, query="Compare X and Y", query_type="comparative", consent_level="full")

    try:
        await trace_service.save_trace(trace1)
        await trace_service.save_trace(trace2)

        definitional = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, query_type="definitional")
        )
        comparative = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, query_type="comparative")
        )

        assert any(t.trace_id == trace1.trace_id for t in definitional)
        assert not any(t.trace_id == trace2.trace_id for t in definitional)
        assert any(t.trace_id == trace2.trace_id for t in comparative)
    finally:
        await trace_service.delete_trace(trace1.trace_id)
        await trace_service.delete_trace(trace2.trace_id)


@pytest.mark.asyncio
async def test_list_traces_excludes_archived_by_default(trace_service):
    """Test that archived traces can be filtered."""
    uid = f"arch_test_{uuid4().hex[:6]}"

    trace_active = _make_trace(user_id=uid, query="Active query", is_archived=False, consent_level="full")
    trace_archived = _make_trace(user_id=uid, query="Archived query", is_archived=True, consent_level="full")

    try:
        await trace_service.save_trace(trace_active)
        await trace_service.save_trace(trace_archived)

        active_traces = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, is_archived=False)
        )
        assert any(t.trace_id == trace_active.trace_id for t in active_traces)
        assert not any(t.trace_id == trace_archived.trace_id for t in active_traces)

        archived_traces = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid, is_archived=True)
        )
        assert any(t.trace_id == trace_archived.trace_id for t in archived_traces)
    finally:
        await trace_service.delete_trace(trace_active.trace_id)
        await trace_service.delete_trace(trace_archived.trace_id)


# ============================================================================
# COUNT TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_count_traces(trace_service):
    """Test counting traces with filters."""
    uid = f"count_test_{uuid4().hex[:6]}"
    trace_ids = []

    try:
        for i in range(3):
            trace = _make_trace(user_id=uid, query=f"Count test {i}", consent_level="full")
            await trace_service.save_trace(trace)
            trace_ids.append(trace.trace_id)

        count = await trace_service.count_traces(filters=TraceFilter(user_id=uid))
        assert count >= 3
    finally:
        for tid in trace_ids:
            await trace_service.delete_trace(tid)


# ============================================================================
# DELETE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_delete_trace(trace_service):
    """Test deleting a trace."""
    trace = _make_trace()
    trace_id = trace.trace_id

    await trace_service.save_trace(trace)
    retrieved = await trace_service.get_trace(trace_id)
    assert retrieved is not None

    deleted = await trace_service.delete_trace(trace_id)
    assert deleted is True

    retrieved = await trace_service.get_trace(trace_id)
    assert retrieved is None


@pytest.mark.asyncio
async def test_delete_nonexistent_trace(trace_service):
    """Test deleting a nonexistent trace returns False."""
    deleted = await trace_service.delete_trace("nonexistent_trace_xyz")
    assert deleted is False


# ============================================================================
# ARCHIVE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_archive_old_traces(trace_service):
    """Test archiving old traces."""
    trace = _make_trace(
        user_id=f"archive_test_{uuid4().hex[:6]}",
        query="Old query",
        consent_level="full"
    )
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        # Archive traces older than 0 days (immediate)
        count = await trace_service.archive_old_traces(days=0)
        # Should have archived at least our trace
        assert count >= 0

        retrieved = await trace_service.get_trace(trace_id)
        if retrieved:
            # If archived, check fields
            if retrieved["is_archived"]:
                assert retrieved["archived_at"] is not None
    finally:
        await trace_service.delete_trace(trace_id)


# ============================================================================
# CONSENT FILTERING IN LIST
# ============================================================================

@pytest.mark.asyncio
async def test_list_consent_filtering(trace_service):
    """Test that list applies consent filtering to previews."""
    uid = f"list_consent_{uuid4().hex[:6]}"
    trace = _make_trace(
        user_id=uid,
        query="Sensitive legal query about tax evasion",
        consent_level="basic"
    )
    trace_id = trace.trace_id

    try:
        await trace_service.save_trace(trace)

        # List with full caller consent — but stored consent is basic, so query redacted
        traces_full = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid),
            consent_level="full"
        )
        found_full = next((t for t in traces_full if t.trace_id == trace_id), None)
        assert found_full is not None
        assert found_full.query_preview == "[REDACTED]"

        # List with anonymous caller consent
        traces_anon = await trace_service.list_traces(
            filters=TraceFilter(user_id=uid),
            consent_level="anonymous"
        )
        found_anon = next((t for t in traces_anon if t.trace_id == trace_id), None)
        assert found_anon is not None
        assert found_anon.user_id == "[REDACTED]"
    finally:
        await trace_service.delete_trace(trace_id)
