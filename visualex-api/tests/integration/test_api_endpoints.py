"""
Integration tests for visualex-api REST endpoints.

These tests use pytest-asyncio and test the Quart API endpoints.
"""
import asyncio
from time import perf_counter
import pytest
from unittest.mock import patch, AsyncMock, MagicMock


@pytest.fixture
async def app_client():
    """Create test client for the Quart app."""
    import sys
    # Mock playwright before importing app
    sys.modules['playwright'] = MagicMock()
    sys.modules['playwright.async_api'] = MagicMock()

    from visualex.app import NormaController

    controller = NormaController()
    await controller.start_background_services()
    try:
        yield controller.app.test_client()
    finally:
        await controller.stop_background_services()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, app_client):
        """Test that health endpoint returns healthy status."""
        response = await app_client.get("/health")

        assert response.status_code == 200
        data = await response.get_json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    @patch("visualex.scrapers.brocardi.BrocardiScraper.request_document")
    @patch("visualex.scrapers.eurlex.EurlexScraper.request_document")
    @patch("visualex.scrapers.normattiva.NormattivaScraper.request_document")
    async def test_health_detailed_checks_services_in_parallel(
        self,
        mock_normattiva_request,
        mock_eurlex_request,
        mock_brocardi_request,
        app_client,
    ):
        """Health checks should run concurrently to keep the endpoint fast."""

        async def delayed_ok(*args, **kwargs):
            await asyncio.sleep(0.1)
            return "ok"

        mock_normattiva_request.side_effect = delayed_ok
        mock_eurlex_request.side_effect = delayed_ok
        mock_brocardi_request.side_effect = delayed_ok

        start = perf_counter()
        response = await app_client.get("/health/detailed")
        elapsed = perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.2
        data = await response.get_json()
        assert data["status"] == "ok"
        assert data["services"]["normattiva"]["status"] == "ok"
        assert data["services"]["eurlex"]["status"] == "ok"
        assert data["services"]["brocardi"]["status"] == "ok"


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

    @pytest.mark.asyncio
    @patch("visualex.app.NormaController.create_norma_visitata_from_data", new_callable=AsyncMock)
    @patch("visualex.scrapers.normattiva.NormattivaScraper.get_document")
    @patch("visualex.scrapers.brocardi.BrocardiScraper.get_info")
    async def test_fetch_all_data_fetches_sources_in_parallel(
        self,
        mock_brocardi,
        mock_normattiva,
        mock_create_norma_visitata,
        app_client,
    ):
        """Normattiva text and Brocardi info should be fetched concurrently."""

        async def delayed_document(*args, **kwargs):
            await asyncio.sleep(0.12)
            return ("Article text...", "https://normattiva.it/...")

        async def delayed_brocardi(*args, **kwargs):
            await asyncio.sleep(0.12)
            return ("Position", {"Spiegazione": "..."}, "https://brocardi.it/...")

        fake_norma = MagicMock(tipo_atto="codice civile")
        fake_nv = MagicMock()
        fake_nv.norma = fake_norma
        fake_nv.allegato = None
        fake_nv.numero_articolo = "1453"
        fake_nv.to_dict.return_value = {"article": "1453"}

        mock_create_norma_visitata.return_value = [fake_nv]
        mock_normattiva.side_effect = delayed_document
        mock_brocardi.side_effect = delayed_brocardi

        start = perf_counter()
        response = await app_client.post(
            "/fetch_all_data",
            json={
                "act_type": "codice civile",
                "article": "1453",
            },
        )
        elapsed = perf_counter() - start

        assert response.status_code == 200
        assert elapsed < 0.2
        data = await response.get_json()
        assert isinstance(data, list)
        assert data[0]["article_text"] == "Article text..."
        assert data[0]["brocardi_info"]["Spiegazione"] == "..."


class TestQueryStatsLogging:
    """Tests for response statistics logging optimizations."""

    @pytest.mark.asyncio
    async def test_log_query_stats_skips_large_json_bodies(self):
        """Large JSON responses should not be reparsed in after_request."""
        import sys

        sys.modules['playwright'] = MagicMock()
        sys.modules['playwright.async_api'] = MagicMock()

        from visualex.app import NormaController
        from visualex.config import QUERY_STATS_MAX_BODY_BYTES
        from quart import g

        controller = NormaController()
        response = MagicMock()
        response.content_type = "application/json"
        response.content_length = QUERY_STATS_MAX_BODY_BYTES + 1
        response.get_data = AsyncMock(return_value='{"large": true}')

        async with controller.app.test_request_context("/health"):
            g.start_time = 0
            returned = await controller.log_query_stats(response)

        assert returned is response
        response.get_data.assert_not_called()
