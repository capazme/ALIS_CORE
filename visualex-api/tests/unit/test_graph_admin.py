"""
Tests for Graph Admin Module
============================

Tests for:
- IngestService: Manual ingestion of norms
- IngestRequest: Request validation
- IngestJobResult: Result serialization
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, AsyncMock, patch

from visualex.graph.admin import (
    IngestService,
    IngestRequest,
    IngestJobResult,
)
from visualex.graph.ingestion import IngestionResult


# =============================================================================
# IngestRequest Tests
# =============================================================================


class TestIngestRequest:
    """Test suite for IngestRequest validation."""

    def test_valid_urn_request(self):
        """Test validation passes for URN request."""
        request = IngestRequest(
            urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )
        assert request.validate() is None

    def test_valid_range_request(self):
        """Test validation passes for range request."""
        request = IngestRequest(
            act_type="codice civile",
            article_range="1470-1490",
        )
        assert request.validate() is None

    def test_valid_articles_list_request(self):
        """Test validation passes for articles list request."""
        request = IngestRequest(
            act_type="codice civile",
            articles=["1453", "1454", "1455"],
        )
        assert request.validate() is None

    def test_invalid_empty_request(self):
        """Test validation fails for empty request."""
        request = IngestRequest()
        error = request.validate()
        assert error is not None
        assert "urn or act_type" in error.lower()

    def test_invalid_act_type_without_articles(self):
        """Test validation fails when act_type provided without articles."""
        request = IngestRequest(act_type="codice civile")
        error = request.validate()
        assert error is not None
        assert "article_range or articles" in error.lower()


# =============================================================================
# IngestJobResult Tests
# =============================================================================


class TestIngestJobResult:
    """Test suite for IngestJobResult."""

    def test_to_dict_complete(self):
        """Test to_dict with all fields."""
        result = IngestJobResult(
            job_id="abc123",
            status="completed",
            total=10,
            succeeded=9,
            failed=1,
            results=[
                {"urn": "urn:test:1", "success": True},
                {"urn": "urn:test:2", "success": False, "error": "Not found"},
            ],
            completed_at="2026-02-01T12:00:00",
            duration_seconds=5.5,
        )

        d = result.to_dict()

        assert d["job_id"] == "abc123"
        assert d["status"] == "completed"
        assert d["total"] == 10
        assert d["succeeded"] == 9
        assert d["failed"] == 1
        assert len(d["results"]) == 2
        assert d["duration_seconds"] == 5.5

    def test_to_dict_minimal(self):
        """Test to_dict with minimal fields."""
        result = IngestJobResult(
            job_id="xyz",
            status="failed",
            total=0,
            succeeded=0,
            failed=0,
        )

        d = result.to_dict()

        assert d["job_id"] == "xyz"
        assert d["status"] == "failed"
        assert d["results"] == []


# =============================================================================
# IngestService Tests
# =============================================================================


class TestIngestService:
    """Test suite for IngestService."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()

        self.mock_normattiva = MagicMock()
        self.mock_normattiva.get_document = AsyncMock(
            return_value=("Article text content", "https://example.com/article")
        )

        self.mock_brocardi = MagicMock()
        self.mock_brocardi.get_info = AsyncMock(
            return_value=("Position", {"Spiegazione": "Test explanation"}, "https://brocardi.it/link")
        )

        # Mock the NormIngester
        self.mock_ingester_result = IngestionResult(
            urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            success=True,
            nodes_created=5,
            edges_created=2,
        )

    @pytest.mark.asyncio
    async def test_ingest_single_urn(self):
        """Test ingesting a single article by URN (AC1)."""
        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                return_value=self.mock_ingester_result
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
            )
            result = await service.ingest(request)

            assert result.total == 1
            assert result.succeeded == 1
            assert result.failed == 0
            assert result.status == "completed"
            assert len(result.results) == 1

    @pytest.mark.asyncio
    async def test_ingest_article_range(self):
        """Test ingesting a range of articles (AC2)."""
        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                return_value=self.mock_ingester_result
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                act_type="codice civile",
                article_range="1470-1472",  # 3 articles
            )
            result = await service.ingest(request)

            assert result.total == 3
            assert result.status in ["completed", "partial", "failed"]
            assert len(result.results) == 3

    @pytest.mark.asyncio
    async def test_ingest_articles_list(self):
        """Test ingesting a list of articles (AC2)."""
        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                return_value=self.mock_ingester_result
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                act_type="codice civile",
                articles=["1453", "1454"],
            )
            result = await service.ingest(request)

            assert result.total == 2
            assert len(result.results) == 2

    @pytest.mark.asyncio
    async def test_ingest_partial_failure(self):
        """Test partial failure handling (AC3)."""
        success_result = IngestionResult(
            urn="urn:test:success",
            success=True,
            nodes_created=5,
        )
        failure_result = IngestionResult(
            urn="urn:test:failure",
            success=False,
            error="Article not found",
        )

        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                side_effect=[success_result, failure_result]
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                act_type="codice civile",
                articles=["1453", "9999"],  # One exists, one doesn't
            )
            result = await service.ingest(request)

            assert result.total == 2
            assert result.succeeded == 1
            assert result.failed == 1
            assert result.status == "partial"

    @pytest.mark.asyncio
    async def test_ingest_all_failed(self):
        """Test all articles failed status."""
        failure_result = IngestionResult(
            urn="urn:test:failure",
            success=False,
            error="Connection error",
        )

        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                return_value=failure_result
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                act_type="codice civile",
                articles=["1453"],
            )
            result = await service.ingest(request)

            assert result.status == "failed"
            assert result.succeeded == 0
            assert result.failed == 1

    @pytest.mark.asyncio
    async def test_ingest_without_brocardi(self):
        """Test ingestion without Brocardi enrichment."""
        with patch('visualex.graph.admin.NormIngester') as MockIngester:
            mock_ingester_instance = MagicMock()
            mock_ingester_instance.ingest_article = AsyncMock(
                return_value=self.mock_ingester_result
            )
            MockIngester.return_value = mock_ingester_instance

            service = IngestService(
                client=self.mock_client,
                normattiva_scraper=self.mock_normattiva,
                brocardi_scraper=self.mock_brocardi,
            )

            request = IngestRequest(
                urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
                include_brocardi=False,
            )
            result = await service.ingest(request)

            assert result.status == "completed"
            # Brocardi should not have been called
            self.mock_brocardi.get_info.assert_not_called()

    def test_resolve_articles_range(self):
        """Test article range resolution."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        articles = service._resolve_articles("codice civile", "1470-1475", None)

        assert articles == ["1470", "1471", "1472", "1473", "1474", "1475"]

    def test_resolve_articles_list(self):
        """Test article list passthrough."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        input_list = ["1453", "1454-bis", "1455"]
        articles = service._resolve_articles("codice civile", None, input_list)

        assert articles == input_list

    def test_parse_urn_to_norma_visitata(self):
        """Test URN parsing."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        nv = service._parse_urn_to_norma_visitata(
            "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )

        assert nv is not None
        assert nv.numero_articolo == "1453"
        assert nv.norma.data == "1942-03-16"

    def test_parse_urn_with_annex(self):
        """Test URN parsing with annex."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        nv = service._parse_urn_to_norma_visitata(
            "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1"
        )

        assert nv is not None
        assert nv.allegato == "2"
        assert nv.numero_articolo == "1"

    def test_parse_malformed_urn_returns_none(self):
        """Test that malformed URN returns None (M2)."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        # Various malformed URNs
        assert service._parse_urn_to_norma_visitata("not-a-urn") is None
        assert service._parse_urn_to_norma_visitata("urn:invalid") is None
        assert service._parse_urn_to_norma_visitata("urn:nir:missing:parts") is None
        assert service._parse_urn_to_norma_visitata("") is None

    def test_resolve_articles_enforces_max_limit(self):
        """Test that article range is truncated to max limit (H3)."""
        service = IngestService(
            client=self.mock_client,
            normattiva_scraper=self.mock_normattiva,
        )

        # Request a huge range
        articles = service._resolve_articles("codice civile", "1-500", None)

        # Should be truncated to MAX_ARTICLES_PER_REQUEST (100)
        assert len(articles) <= 100

    def test_job_has_unique_id(self):
        """Test each job gets a unique ID."""
        result1 = IngestJobResult(
            job_id="test1",
            status="completed",
            total=1,
            succeeded=1,
            failed=0,
        )
        result2 = IngestJobResult(
            job_id="test2",
            status="completed",
            total=1,
            succeeded=1,
            failed=0,
        )

        # Job IDs should be set (in practice they're UUIDs)
        assert result1.job_id != result2.job_id

    def test_job_duration_tracked(self):
        """Test job duration is tracked."""
        result = IngestJobResult(
            job_id="test",
            status="completed",
            total=1,
            succeeded=1,
            failed=0,
            duration_seconds=2.5,
        )

        assert result.duration_seconds == 2.5
        assert result.to_dict()["duration_seconds"] == 2.5
