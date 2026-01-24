"""
Pytest configuration and shared fixtures for visualex-api tests.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_playwright():
    """Mock playwright for tests that don't need real browser."""
    with patch("visualex.utils.sys_op.get_playwright_manager") as mock:
        manager = MagicMock()
        context = AsyncMock()
        page = AsyncMock()

        manager.new_context = AsyncMock(return_value=context)
        context.new_page = AsyncMock(return_value=page)
        context.close = AsyncMock()

        mock.return_value = manager
        yield page


@pytest.fixture
def sample_norma_dict():
    """Sample norma dictionary for testing."""
    return {
        "tipo_atto": "legge",
        "data": "2020-01-15",
        "numero_atto": "123",
        "numero_articolo": "1",
    }


@pytest.fixture
def sample_codice_civile_dict():
    """Sample codice civile dictionary for testing."""
    return {
        "tipo_atto": "codice civile",
        "numero_articolo": "1453",
    }


@pytest.fixture
def sample_article_response():
    """Sample article response from scraper."""
    return {
        "text": "Il contratto è l'accordo di due o più parti...",
        "url": "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:regio.decreto:1942-03-16;262~art1321",
    }


@pytest.fixture
def sample_brocardi_response():
    """Sample Brocardi info response."""
    return {
        "position": "Libro IV > Titolo II > Capo I",
        "link": "https://www.brocardi.it/codice-civile/libro-quarto/titolo-ii/capo-i/art1453.html",
        "Brocardi": "Risoluzione per inadempimento",
        "Ratio": "La ratio della norma è quella di tutelare...",
        "Spiegazione": "L'art. 1453 c.c. disciplina la risoluzione...",
        "Massime": [
            "Cass. civ., sez. II, 15/03/2021, n. 7234",
            "Cass. civ., sez. III, 22/05/2020, n. 9456",
        ],
    }
