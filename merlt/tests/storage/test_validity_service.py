"""
Tests for TemporalValidityService
===================================

Tests cover:
- Check vigente norm -> status="vigente", is_valid=True
- Check abrogated norm -> status="abrogato", warning_level="critical"
- Check modified norm -> status="modificato", warning con data
- Check replaced norm -> status="sostituito", warning_level="critical"
- Check unknown URN -> status="unknown"
- Batch validity check
- Cache hit/miss/expiry
- as_of_date filtering (modifications, abrogation, sostituzione)
- Summary aggregation with unknown_count
- Warning messages in italian
- as_of_date validation

Usage:
    pytest tests/storage/test_validity_service.py -v
"""

import time
import pytest
from unittest.mock import AsyncMock

from merlt.storage.temporal.validity_service import (
    TemporalValidityService,
    ValidityResult,
    ValiditySummary,
    CACHE_TTL_SECONDS,
    validate_as_of_date,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_graph_db():
    """Mock FalkorDBClient for unit tests."""
    mock = AsyncMock()
    mock.query = AsyncMock(return_value=[])
    mock.health_check = AsyncMock(return_value=True)
    return mock


@pytest.fixture
def validity_service(mock_graph_db):
    """TemporalValidityService with mock graph."""
    return TemporalValidityService(graph_db=mock_graph_db)


def _norm_status_row(
    is_abrogated=False,
    is_current=True,
    mod_count=0,
    last_modified=None,
    effective_since=None,
    abr_urn=None,
    abr_estremi=None,
    abr_date=None,
    sost_urn=None,
    sost_estremi=None,
    sost_date=None,
):
    """Helper to build a norm status query result row."""
    return {
        "is_abrogated": is_abrogated,
        "is_current": is_current,
        "mod_count": mod_count,
        "last_modified": last_modified,
        "effective_since": effective_since,
        "abr_urn": abr_urn,
        "abr_estremi": abr_estremi,
        "abr_date": abr_date,
        "sost_urn": sost_urn,
        "sost_estremi": sost_estremi,
        "sost_date": sost_date,
    }


# ============================================================================
# VIGENTE NORM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_check_vigente_norm(validity_service, mock_graph_db):
    """Norma vigente senza modifiche -> status='vigente', is_valid=True."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    result = await validity_service.check_validity("urn:nir:stato:codice.civile:1942;art1")

    assert result.urn == "urn:nir:stato:codice.civile:1942;art1"
    assert result.status == "vigente"
    assert result.is_valid is True
    assert result.warning_level == "none"
    assert result.warning_message is None
    assert result.checked_at != ""


# ============================================================================
# ABROGATED NORM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_check_abrogated_norm(validity_service, mock_graph_db):
    """Norma abrogata -> status='abrogato', warning_level='critical'."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            is_abrogated=True,
            is_current=False,
            mod_count=1,
            abr_urn="urn:nir:stato:legge:2020-12-30;178",
            abr_estremi="L. 178/2020",
            abr_date="2021-01-01",
        )
    ])

    result = await validity_service.check_validity("urn:nir:stato:codice.penale:1930;art528")

    assert result.status == "abrogato"
    assert result.is_valid is False
    assert result.warning_level == "critical"
    assert "abrogata" in result.warning_message
    assert "L. 178/2020" in result.warning_message
    assert result.abrogating_norm is not None
    assert result.abrogating_norm["urn"] == "urn:nir:stato:legge:2020-12-30;178"


# ============================================================================
# MODIFIED NORM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_check_modified_norm(validity_service, mock_graph_db):
    """Norma modificata -> status='modificato', warning con data."""
    mock_graph_db.query = AsyncMock(side_effect=[
        [_norm_status_row(is_current=True, mod_count=3, last_modified="2024-06-15")],
        [
            {"event_type": "modifica", "by_urn": "urn:dl:2024-01", "by_estremi": "D.L. 1/2024", "event_date": "2024-06-15"},
            {"event_type": "modifica", "by_urn": "urn:dl:2023-05", "by_estremi": "D.L. 5/2023", "event_date": "2023-03-01"},
        ],
    ])

    result = await validity_service.check_validity("urn:nir:stato:codice.civile:1942;art1453")

    assert result.status == "modificato"
    assert result.is_valid is True
    assert result.warning_level == "warning"
    assert "modificata" in result.warning_message
    assert "2024-06-15" in result.warning_message
    assert result.modification_count == 3
    assert len(result.recent_modifications) == 2


# ============================================================================
# REPLACED NORM TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_check_replaced_norm(validity_service, mock_graph_db):
    """Norma sostituita -> status='sostituito', warning_level='critical'."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            is_current=False,
            mod_count=1,
            sost_urn="urn:nir:stato:legge:2023-01-01;10",
            sost_estremi="L. 10/2023",
            sost_date="2023-06-01",
        )
    ])

    result = await validity_service.check_validity("urn:nir:stato:codice.civile:1942;art100")

    assert result.status == "sostituito"
    assert result.is_valid is False
    assert result.warning_level == "critical"
    assert "sostituita" in result.warning_message
    assert "L. 10/2023" in result.warning_message
    assert result.replacing_norm is not None
    assert result.replacing_norm["urn"] == "urn:nir:stato:legge:2023-01-01;10"


# ============================================================================
# UNKNOWN URN TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_check_unknown_urn(validity_service, mock_graph_db):
    """URN inesistente -> status='unknown'."""
    mock_graph_db.query = AsyncMock(return_value=[])

    result = await validity_service.check_validity("urn:nir:stato:norma:inesistente")

    assert result.status == "unknown"
    assert result.is_valid is False
    assert result.warning_level == "info"
    assert "non verificabile" in result.warning_message


# ============================================================================
# BATCH VALIDITY TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_batch_validity(validity_service, mock_graph_db):
    """Batch di URN -> risultati per ciascuna."""
    mock_graph_db.query = AsyncMock(side_effect=[
        [_norm_status_row(is_current=True, mod_count=0)],
        [],  # unknown
        [_norm_status_row(is_current=True, mod_count=0)],
    ])

    results = await validity_service.check_batch_validity([
        "urn:a", "urn:b", "urn:c"
    ])

    assert len(results) == 3
    assert results[0].status == "vigente"
    assert results[1].status == "unknown"
    assert results[2].status == "vigente"


# ============================================================================
# CACHE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_cache_hit(validity_service, mock_graph_db):
    """Seconda chiamata stessa URN -> from cache (no query)."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    result1 = await validity_service.check_validity("urn:cached")
    assert result1.status == "vigente"

    mock_graph_db.query.reset_mock()

    result2 = await validity_service.check_validity("urn:cached")
    assert result2.status == "vigente"
    mock_graph_db.query.assert_not_called()


@pytest.mark.asyncio
async def test_cache_expiry(validity_service, mock_graph_db):
    """Dopo TTL -> re-fetch."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    await validity_service.check_validity("urn:expiry")

    # Manually expire cache entry
    cache_key = "urn:expiry:current"
    result, _ = validity_service._cache[cache_key]
    validity_service._cache[cache_key] = (result, time.time() - CACHE_TTL_SECONDS - 1)

    mock_graph_db.query.reset_mock()
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    await validity_service.check_validity("urn:expiry")
    mock_graph_db.query.assert_called()


@pytest.mark.asyncio
async def test_clear_cache(validity_service, mock_graph_db):
    """clear_cache() svuota la cache."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    await validity_service.check_validity("urn:clear")
    assert len(validity_service._cache) > 0

    validity_service.clear_cache()
    assert len(validity_service._cache) == 0


# ============================================================================
# AS_OF_DATE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_as_of_date_filtering_modifications(validity_service, mock_graph_db):
    """Modifiche post-as_of_date -> warning, pre -> ok."""
    mock_graph_db.query = AsyncMock(side_effect=[
        [_norm_status_row(is_current=True, mod_count=2, last_modified="2024-06-15")],
        [
            {"event_type": "modifica", "by_urn": "urn:a", "by_estremi": "D.L. 1/2024", "event_date": "2024-06-15"},
            {"event_type": "modifica", "by_urn": "urn:b", "by_estremi": "D.L. 2/2023", "event_date": "2023-03-01"},
        ],
    ])

    # as_of_date AFTER all modifications -> no relevant mods -> vigente
    result = await validity_service.check_validity(
        "urn:date_test", as_of_date="2025-01-01"
    )
    assert result.status == "vigente"
    assert result.warning_level == "none"

    validity_service.clear_cache()

    mock_graph_db.query = AsyncMock(side_effect=[
        [_norm_status_row(is_current=True, mod_count=2, last_modified="2024-06-15")],
        [
            {"event_type": "modifica", "by_urn": "urn:a", "by_estremi": "D.L. 1/2024", "event_date": "2024-06-15"},
            {"event_type": "modifica", "by_urn": "urn:b", "by_estremi": "D.L. 2/2023", "event_date": "2023-03-01"},
        ],
    ])

    # as_of_date BEFORE some modifications -> warning
    result2 = await validity_service.check_validity(
        "urn:date_test", as_of_date="2024-01-01"
    )
    assert result2.status == "modificato"
    assert result2.warning_level == "warning"


@pytest.mark.asyncio
async def test_as_of_date_abrogation_after_date(validity_service, mock_graph_db):
    """Norma abrogata DOPO as_of_date -> vigente a quella data."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            is_abrogated=True,
            abr_urn="urn:nir:stato:legge:2024;100",
            abr_estremi="L. 100/2024",
            abr_date="2024-06-01",
            mod_count=0,
        )
    ])

    # as_of_date BEFORE abrogation -> norm was vigente at that date
    result = await validity_service.check_validity(
        "urn:abr_date_test", as_of_date="2023-01-01"
    )
    assert result.status == "vigente"
    assert result.warning_level == "none"


@pytest.mark.asyncio
async def test_as_of_date_abrogation_before_date(validity_service, mock_graph_db):
    """Norma abrogata PRIMA di as_of_date -> abrogata."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            is_abrogated=True,
            abr_urn="urn:nir:stato:legge:2020;50",
            abr_estremi="L. 50/2020",
            abr_date="2020-06-01",
            mod_count=0,
        )
    ])

    result = await validity_service.check_validity(
        "urn:abr_before_test", as_of_date="2023-01-01"
    )
    assert result.status == "abrogato"
    assert result.warning_level == "critical"


@pytest.mark.asyncio
async def test_as_of_date_sostituzione_after_date(validity_service, mock_graph_db):
    """Norma sostituita DOPO as_of_date -> vigente a quella data."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            sost_urn="urn:nir:stato:legge:2024;200",
            sost_estremi="L. 200/2024",
            sost_date="2024-09-01",
            mod_count=0,
        )
    ])

    result = await validity_service.check_validity(
        "urn:sost_date_test", as_of_date="2023-01-01"
    )
    assert result.status == "vigente"
    assert result.warning_level == "none"


# ============================================================================
# SUMMARY AGGREGATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_summary_aggregation(validity_service, mock_graph_db):
    """Mix vigenti/abrogati/unknown -> summary corretto con unknown_count."""
    mock_graph_db.query = AsyncMock(side_effect=[
        # Norm 1: vigente
        [_norm_status_row(is_current=True, mod_count=0)],
        # Norm 2: abrogato
        [_norm_status_row(is_abrogated=True, abr_urn="urn:x", abr_estremi="L.1", abr_date="2020-01-01")],
        # Norm 3: modificato (status query)
        [_norm_status_row(is_current=True, mod_count=2, last_modified="2024-01-01")],
        # Norm 3: modifications query
        [{"event_type": "modifica", "by_urn": "urn:y", "by_estremi": "D.L. 1", "event_date": "2024-01-01"}],
        # Norm 4: unknown
        [],
    ])

    mock_trace_service = AsyncMock()
    mock_trace_service.get_trace = AsyncMock(return_value={
        "trace_id": "test_trace_1",
        "sources": [
            {"article_urn": "urn:vigente"},
            {"article_urn": "urn:abrogato"},
            {"article_urn": "urn:modificato"},
            {"article_urn": "urn:sconosciuto"},
        ]
    })

    summary = await validity_service.check_trace_validity(
        trace_id="test_trace_1",
        trace_service=mock_trace_service
    )

    assert summary.trace_id == "test_trace_1"
    assert summary.total_sources == 4
    assert summary.valid_count == 1
    assert summary.critical_count == 1
    assert summary.warning_count == 1
    assert summary.unknown_count == 1
    assert summary.summary_message is not None
    assert "non più in vigore" in summary.summary_message
    assert "modifiche recenti" in summary.summary_message
    assert "non verificabile" in summary.summary_message


@pytest.mark.asyncio
async def test_summary_all_valid(validity_service, mock_graph_db):
    """Tutte le fonti vigenti -> nessun summary_message."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    mock_trace_service = AsyncMock()
    mock_trace_service.get_trace = AsyncMock(return_value={
        "trace_id": "test_ok",
        "sources": [
            {"article_urn": "urn:ok1"},
            {"article_urn": "urn:ok2"},
        ]
    })

    summary = await validity_service.check_trace_validity(
        trace_id="test_ok",
        trace_service=mock_trace_service
    )

    assert summary.valid_count == 2
    assert summary.critical_count == 0
    assert summary.warning_count == 0
    assert summary.unknown_count == 0
    assert summary.summary_message is None


@pytest.mark.asyncio
async def test_trace_not_found(validity_service):
    """Trace inesistente -> ValueError."""
    mock_trace_service = AsyncMock()
    mock_trace_service.get_trace = AsyncMock(return_value=None)

    with pytest.raises(ValueError, match="not found"):
        await validity_service.check_trace_validity(
            trace_id="nonexistent",
            trace_service=mock_trace_service
        )


@pytest.mark.asyncio
async def test_trace_no_sources(validity_service):
    """Trace senza sources -> summary vuoto."""
    mock_trace_service = AsyncMock()
    mock_trace_service.get_trace = AsyncMock(return_value={
        "trace_id": "empty_trace",
        "sources": []
    })

    summary = await validity_service.check_trace_validity(
        trace_id="empty_trace",
        trace_service=mock_trace_service
    )

    assert summary.total_sources == 0
    assert summary.valid_count == 0
    assert summary.summary_message is None


# ============================================================================
# WARNING MESSAGE TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_warning_messages_it(validity_service, mock_graph_db):
    """Warning messages sono in italiano e contengono i dettagli corretti."""

    # Test abrogato message
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            is_abrogated=True,
            abr_urn="urn:nir:stato:decreto.legislativo:2018-05-01;51",
            abr_estremi="D.Lgs. 51/2018",
            abr_date="2018-05-04",
        )
    ])

    result = await validity_service.check_validity("urn:test_msg_abr")
    assert "Norma abrogata il 2018-05-04 da D.Lgs. 51/2018" == result.warning_message

    validity_service.clear_cache()

    # Test modificato message
    mock_graph_db.query = AsyncMock(side_effect=[
        [_norm_status_row(is_current=True, mod_count=1, last_modified="2023-12-31")],
        [{"event_type": "modifica", "by_urn": "urn:x", "by_estremi": "L. 1", "event_date": "2023-12-31"}],
    ])

    result = await validity_service.check_validity("urn:test_msg_mod")
    assert "Norma modificata (ultima modifica: 2023-12-31) - verificare vigenza attuale" == result.warning_message

    validity_service.clear_cache()

    # Test sostituito message
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(
            sost_urn="urn:nir:stato:legge:2020;120",
            sost_estremi="L. 120/2020",
            sost_date="2020-09-01",
        )
    ])

    result = await validity_service.check_validity("urn:test_msg_sost")
    assert "Norma sostituita il 2020-09-01 da L. 120/2020" == result.warning_message


# ============================================================================
# DEDUPLICATION TESTS
# ============================================================================

@pytest.mark.asyncio
async def test_trace_deduplicates_urns(validity_service, mock_graph_db):
    """Duplicate URNs in sources are deduplicated."""
    mock_graph_db.query = AsyncMock(return_value=[
        _norm_status_row(is_current=True, mod_count=0)
    ])

    mock_trace_service = AsyncMock()
    mock_trace_service.get_trace = AsyncMock(return_value={
        "trace_id": "dedup_trace",
        "sources": [
            {"article_urn": "urn:same"},
            {"article_urn": "urn:same"},
            {"article_urn": "urn:same"},
        ]
    })

    summary = await validity_service.check_trace_validity(
        trace_id="dedup_trace",
        trace_service=mock_trace_service
    )

    assert summary.total_sources == 1


# ============================================================================
# SERIALIZATION TESTS
# ============================================================================

def test_validity_result_to_dict():
    """ValidityResult.to_dict() serializes correctly."""
    result = ValidityResult(
        urn="urn:test",
        status="vigente",
        is_valid=True,
        warning_level="none",
        checked_at="2024-01-01T00:00:00+00:00",
    )
    d = result.to_dict()
    assert d["urn"] == "urn:test"
    assert d["status"] == "vigente"
    assert d["is_valid"] is True
    assert d["warning_level"] == "none"
    assert d["warning_message"] is None
    assert d["recent_modifications"] == []


def test_validity_summary_to_dict():
    """ValiditySummary.to_dict() serializes correctly."""
    summary = ValiditySummary(
        trace_id="t1",
        as_of_date=None,
        total_sources=2,
        valid_count=1,
        warning_count=1,
        critical_count=0,
        unknown_count=0,
        results=[
            ValidityResult(
                urn="urn:a", status="vigente", is_valid=True,
                warning_level="none", checked_at="2024-01-01T00:00:00+00:00",
            ),
        ],
        summary_message="test",
    )
    d = summary.to_dict()
    assert d["trace_id"] == "t1"
    assert d["total_sources"] == 2
    assert d["unknown_count"] == 0
    assert len(d["results"]) == 1
    assert d["results"][0]["urn"] == "urn:a"


# ============================================================================
# VALIDATE AS_OF_DATE TESTS
# ============================================================================

def test_validate_as_of_date_valid():
    """Valid ISO date passes."""
    assert validate_as_of_date("2024-01-15") == "2024-01-15"
    assert validate_as_of_date("2023-12-31") == "2023-12-31"


def test_validate_as_of_date_none():
    """None returns None."""
    assert validate_as_of_date(None) is None


def test_validate_as_of_date_invalid():
    """Invalid formats raise ValueError."""
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_as_of_date("not-a-date")

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_as_of_date("01/15/2024")

    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        validate_as_of_date("2024-1-5")


# ============================================================================
# BUILD_SUMMARY_MESSAGE TESTS (public method)
# ============================================================================

def test_build_summary_message_no_issues(validity_service):
    """No issues -> None."""
    msg = validity_service.build_summary_message(5, 0, 0, 0)
    assert msg is None


def test_build_summary_message_with_unknown(validity_service):
    """Unknown sources included in message."""
    msg = validity_service.build_summary_message(2, 0, 0, 1)
    assert msg is not None
    assert "non verificabile" in msg
    assert "3 fonti citate" in msg


def test_build_summary_message_all_categories(validity_service):
    """All categories present in message."""
    msg = validity_service.build_summary_message(1, 2, 3, 4)
    assert "non più in vigore" in msg
    assert "modifiche recenti" in msg
    assert "non verificabile" in msg
    assert "10 fonti citate" in msg
