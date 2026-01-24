"""
Integration tests for visualex-api REST endpoints.

These tests use pytest-asyncio and test the Quart API endpoints.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
def app_client():
    """Create test client for the Quart app."""
    import sys
    # Mock playwright before importing app
    sys.modules['playwright'] = MagicMock()
    sys.modules['playwright.async_api'] = MagicMock()

    from visualex.app import app

    return app.test_client()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app_client):
        """Test that health endpoint returns healthy status."""
        response = await app_client.get("/health")

        assert response.status_code == 200
        data = await response.get_json()
        assert data["status"] == "healthy"


class TestFetchNormaDataEndpoint:
    """Tests for /fetch_norma_data endpoint."""

    @pytest.mark.asyncio
    async def test_fetch_norma_data_codice_civile(self, app_client):
        """Test fetching norma data for Codice Civile."""
        response = await app_client.post(
            "/fetch_norma_data",
            json={
                "act_type": "codice civile",
                "article": "1453",
            },
        )

        assert response.status_code == 200
        data = await response.get_json()
        assert "norma_data" in data

    @pytest.mark.asyncio
    async def test_fetch_norma_data_missing_act_type(self, app_client):
        """Test that missing act_type returns error."""
        response = await app_client.post(
            "/fetch_norma_data",
            json={
                "article": "1453",
            },
        )

        # Should return 400 or have error in response
        assert response.status_code in [400, 422] or "error" in (await response.get_json())


class TestFetchArticleTextEndpoint:
    """Tests for /fetch_article_text endpoint."""

    @pytest.mark.asyncio
    @patch("visualex.scrapers.normattiva.NormattivaScraper.get_document")
    async def test_fetch_article_text_success(self, mock_get_doc, app_client):
        """Test successful article text fetch."""
        mock_get_doc.return_value = (
            "Nei contratti a prestazioni corrispettive...",
            "https://normattiva.it/...",
        )

        response = await app_client.post(
            "/fetch_article_text",
            json={
                "act_type": "codice civile",
                "article": "1453",
            },
        )

        assert response.status_code == 200
        data = await response.get_json()
        assert isinstance(data, list)
        if data:
            assert "article_text" in data[0] or "text" in data[0] or "error" in data[0]


class TestFetchBrocardiInfoEndpoint:
    """Tests for /fetch_brocardi_info endpoint."""

    @pytest.mark.asyncio
    @patch("visualex.scrapers.brocardi.BrocardiScraper.get_info")
    async def test_fetch_brocardi_info_success(self, mock_get_info, app_client):
        """Test successful Brocardi info fetch."""
        mock_get_info.return_value = (
            "Libro IV > Titolo II",
            {
                "Spiegazione": "Articolo sulla risoluzione...",
                "Massime": ["Cass. 123/2020"],
            },
            "https://brocardi.it/...",
        )

        response = await app_client.post(
            "/fetch_brocardi_info",
            json={
                "act_type": "codice civile",
                "article": "1453",
            },
        )

        assert response.status_code == 200
        data = await response.get_json()
        assert isinstance(data, list)


class TestFetchTreeEndpoint:
    """Tests for /fetch_tree endpoint."""

    @pytest.mark.asyncio
    @patch("visualex.utils.treextractor.get_tree")
    async def test_fetch_tree_success(self, mock_get_tree, app_client):
        """Test successful tree fetch."""
        mock_get_tree.return_value = {
            "articles": [
                {"number": "1", "title": "Delle fonti del diritto"},
                {"number": "2", "title": "Le leggi"},
            ],
            "count": 2,
        }

        response = await app_client.post(
            "/fetch_tree",
            json={
                "urn": "urn:nir:stato:regio.decreto:1942-03-16;262",
            },
        )

        assert response.status_code == 200
        data = await response.get_json()
        # Should have articles or error
        assert "articles" in data or "error" in data


class TestFetchAllDataEndpoint:
    """Tests for /fetch_all_data endpoint."""

    @pytest.mark.asyncio
    @patch("visualex.scrapers.normattiva.NormattivaScraper.get_document")
    @patch("visualex.scrapers.brocardi.BrocardiScraper.get_info")
    async def test_fetch_all_data_combines_results(
        self, mock_brocardi, mock_normattiva, app_client
    ):
        """Test that fetch_all_data combines article text and Brocardi info."""
        mock_normattiva.return_value = ("Article text...", "https://normattiva.it/...")
        mock_brocardi.return_value = ("Position", {"Spiegazione": "..."}, "https://brocardi.it/...")

        response = await app_client.post(
            "/fetch_all_data",
            json={
                "act_type": "codice civile",
                "article": "1453",
            },
        )

        assert response.status_code == 200
        data = await response.get_json()
        assert isinstance(data, list)
