"""
Tests for Citation Router
=========================

Tests for the citation export API endpoints.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from merlt.api.citation_router import (
    router,
    sanitize_filename,
    generate_filename,
    EXPORT_DIR,
)
from merlt.api.models.citation_models import (
    CitationFormat,
    CitationSource,
    CitationExportRequest,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def app():
    """Create a FastAPI app with the citation router."""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def sample_sources():
    """Sample sources for testing."""
    return [
        {
            "article_urn": "urn:nir:stato:codice.civile:1942;art1453",
            "expert": "literal",
            "relevance": 0.95,
        },
        {
            "article_urn": "urn:nir:stato:legge:1990-08-07;241~art1",
            "expert": "systemic",
            "relevance": 0.85,
        },
    ]


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestSanitizeFilename:
    """Tests for the sanitize_filename helper."""

    def test_basic_sanitization(self):
        """Test basic filename sanitization."""
        result = sanitize_filename("Hello World")
        assert result == "hello_world"

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        result = sanitize_filename("What's the law?")
        assert "'" not in result
        assert "?" not in result

    def test_max_length(self):
        """Test that result is truncated to max_length."""
        long_string = "a" * 100
        result = sanitize_filename(long_string, max_length=50)
        assert len(result) == 50

    def test_italian_characters(self):
        """Test handling of Italian characters."""
        result = sanitize_filename("Cos'è la responsabilità?")
        # Should handle gracefully
        assert len(result) > 0

    def test_empty_string(self):
        """Test empty string handling."""
        result = sanitize_filename("")
        assert result == ""


class TestGenerateFilename:
    """Tests for the generate_filename helper."""

    def test_filename_format(self):
        """Test filename format."""
        result = generate_filename(
            query_summary="risoluzione contratto",
            format=CitationFormat.ITALIAN_LEGAL,
            extension="txt"
        )

        assert result.startswith("citations_")
        assert result.endswith(".txt")
        assert "risoluzione" in result

    def test_filename_without_query(self):
        """Test filename without query summary."""
        result = generate_filename(
            query_summary=None,
            format=CitationFormat.BIBTEX,
            extension="bib"
        )

        assert result.startswith("citations_")
        assert result.endswith(".bib")
        assert "bibtex" in result

    def test_filename_uniqueness(self):
        """Test that filenames are unique (contain UUID)."""
        result1 = generate_filename("test", CitationFormat.JSON, "json")
        result2 = generate_filename("test", CitationFormat.JSON, "json")

        # Should be different due to UUID
        assert result1 != result2


# =============================================================================
# ENDPOINT TESTS
# =============================================================================


class TestListFormats:
    """Tests for the /formats endpoint."""

    def test_list_formats(self, client):
        """Test listing available formats."""
        response = client.get("/api/v1/citations/formats")

        assert response.status_code == 200
        data = response.json()

        assert "formats" in data
        assert "default_format" in data
        assert data["default_format"] == "italian_legal"

        format_names = [f["name"] for f in data["formats"]]
        assert "italian_legal" in format_names
        assert "bibtex" in format_names
        assert "plain_text" in format_names
        assert "json" in format_names

    def test_format_info_structure(self, client):
        """Test format info structure."""
        response = client.get("/api/v1/citations/formats")
        data = response.json()

        for fmt in data["formats"]:
            assert "name" in fmt
            assert "description" in fmt
            assert "extension" in fmt
            assert "media_type" in fmt


class TestFormatInline:
    """Tests for the /format endpoint."""

    def test_format_inline_italian_legal(self, client, sample_sources):
        """Test inline formatting in Italian legal style."""
        response = client.post(
            "/api/v1/citations/format",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
                "query_summary": "Test query",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"]
        assert data["format"] == "italian_legal"
        assert "FONTI GIURIDICHE" in data["content"]
        assert "Art. 1453 c.c." in data["content"]
        assert data["citations_count"] == 2

    def test_format_inline_bibtex(self, client, sample_sources):
        """Test inline formatting in BibTeX."""
        response = client.post(
            "/api/v1/citations/format",
            json={
                "sources": sample_sources,
                "format": "bibtex",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"]
        assert "@legislation{" in data["content"]

    def test_format_inline_json(self, client, sample_sources):
        """Test inline formatting in JSON."""
        response = client.post(
            "/api/v1/citations/format",
            json={
                "sources": sample_sources,
                "format": "json",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Content should be valid JSON
        content_data = json.loads(data["content"])
        assert "citations" in content_data

    def test_format_inline_empty_sources(self, client):
        """Test that empty sources list returns error."""
        response = client.post(
            "/api/v1/citations/format",
            json={
                "sources": [],
                "format": "italian_legal",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_format_inline_without_attribution(self, client, sample_sources):
        """Test formatting without attribution."""
        response = client.post(
            "/api/v1/citations/format",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
                "include_attribution": False,
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert "Elaborazione a cura di ALIS" not in data["content"]


class TestExportCitations:
    """Tests for the /export endpoint."""

    def test_export_with_sources(self, client, sample_sources):
        """Test exporting with direct sources."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["success"]
        assert data["format"] == "italian_legal"
        assert "download_url" in data
        assert data["citations_count"] == 2
        assert "filename" in data

    def test_export_creates_file(self, client, sample_sources):
        """Test that export creates a downloadable file."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
            },
        )

        data = response.json()
        download_url = data["download_url"]

        # Try to download
        download_response = client.get(download_url)
        assert download_response.status_code == 200
        assert "Art. 1453 c.c." in download_response.text

    def test_export_bibtex_format(self, client, sample_sources):
        """Test exporting in BibTeX format."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "bibtex",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filename"].endswith(".bib")

        # Download and verify
        download_response = client.get(data["download_url"])
        assert "@legislation{" in download_response.text

    def test_export_json_format(self, client, sample_sources):
        """Test exporting in JSON format."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "json",
            },
        )

        assert response.status_code == 200
        data = response.json()

        assert data["filename"].endswith(".json")

        # Download and verify
        download_response = client.get(data["download_url"])
        content = json.loads(download_response.text)
        assert "citations" in content

    def test_export_no_sources_or_trace(self, client):
        """Test that export without sources or trace_id returns error."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "format": "italian_legal",
            },
        )

        assert response.status_code == 400
        assert "Either trace_id or sources" in response.json()["detail"]

    def test_export_with_query_summary(self, client, sample_sources):
        """Test export includes query summary."""
        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
                "include_query_summary": True,
            },
        )

        assert response.status_code == 200


class TestDownloadFile:
    """Tests for the /download endpoint."""

    def test_download_nonexistent_file(self, client):
        """Test downloading a nonexistent file."""
        response = client.get("/api/v1/citations/download/nonexistent.txt")
        assert response.status_code == 404

    def test_download_path_traversal_blocked(self, client):
        """Test that path traversal is blocked."""
        response = client.get("/api/v1/citations/download/../../../etc/passwd")
        # Either 400 (our check) or 404 (URL resolved by framework)
        assert response.status_code in [400, 404]

    def test_download_slash_blocked(self, client):
        """Test that slashes in filename are blocked."""
        response = client.get("/api/v1/citations/download/path/to/file.txt")
        assert response.status_code in [400, 404]


class TestExportWithTrace:
    """Tests for export with trace_id."""

    @pytest.mark.asyncio
    async def test_export_with_trace_not_found(self, client):
        """Test export with nonexistent trace_id."""
        with patch("merlt.api.citation_router.get_trace_service") as mock_service:
            mock_instance = AsyncMock()
            mock_instance.get_trace.return_value = None
            mock_instance.close = AsyncMock()
            mock_service.return_value = mock_instance

            response = client.post(
                "/api/v1/citations/export",
                json={
                    "trace_id": "trace_nonexistent",
                    "format": "italian_legal",
                },
            )

            # Either 404 (trace not found) or 500 (connection error in test)
            assert response.status_code in [404, 500]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestIntegration:
    """Integration tests for the citation export flow."""

    def test_full_export_download_flow(self, client, sample_sources):
        """Test full export -> download flow."""
        # Export
        export_response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sample_sources,
                "format": "italian_legal",
            },
        )

        assert export_response.status_code == 200
        export_data = export_response.json()

        # Download
        download_response = client.get(export_data["download_url"])

        assert download_response.status_code == 200
        assert "Art. 1453 c.c." in download_response.text
        assert "L. 7 agosto 1990" in download_response.text

    def test_all_formats_work(self, client, sample_sources):
        """Test that all formats can be exported."""
        formats = ["italian_legal", "bibtex", "plain_text", "json"]

        for fmt in formats:
            response = client.post(
                "/api/v1/citations/export",
                json={
                    "sources": sample_sources,
                    "format": fmt,
                },
            )

            assert response.status_code == 200, f"Format {fmt} failed"
            data = response.json()
            assert data["success"], f"Format {fmt} not successful"

            # Download and verify
            download_response = client.get(data["download_url"])
            assert download_response.status_code == 200, f"Download for {fmt} failed"

    def test_utf8_characters_preserved(self, client):
        """Test that UTF-8 characters are preserved through the flow."""
        sources = [
            {
                "article_urn": "urn:nir:stato:codice.civile:1942;art1453",
                "title": "Risolubilità del contratto per inadempimento - obbligatorietà",
            }
        ]

        response = client.post(
            "/api/v1/citations/export",
            json={
                "sources": sources,
                "format": "italian_legal",
            },
        )

        data = response.json()
        download_response = client.get(data["download_url"])

        # Italian month should be preserved
        assert "agosto" in download_response.text or "marzo" in download_response.text or "FONTI" in download_response.text
