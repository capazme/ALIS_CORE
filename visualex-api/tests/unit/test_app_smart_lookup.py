import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def controller_instance():
    """Build a controller instance with browser dependencies mocked."""
    sys.modules["playwright"] = MagicMock()
    sys.modules["playwright.async_api"] = MagicMock()

    from visualex.app import NormaController

    return NormaController()


@pytest.mark.asyncio
async def test_create_norma_visitata_skips_tree_lookup_for_explicit_dispositivo(controller_instance):
    """Explicit dispositivo requests should skip smart annex lookup entirely."""
    with (
        patch("visualex.app.parse_article_input", new=AsyncMock(return_value=["1453"])),
        patch("visualex.app.get_tree", new=AsyncMock()) as mock_get_tree,
    ):
        result = await controller_instance.create_norma_visitata_from_data(
            {
                "act_type": "codice civile",
                "article": "1453",
                "annex": "",
            }
        )

    mock_get_tree.assert_not_awaited()
    assert len(result) == 1
    assert result[0].allegato is None


@pytest.mark.asyncio
async def test_create_norma_visitata_uses_article_lookup_for_smart_redirect(controller_instance):
    """Indexed metadata should redirect directly to the best annex."""
    with (
        patch("visualex.app.NORMATTIVA_URN_CODICI", {}),
        patch("visualex.app.parse_article_input", new=AsyncMock(return_value=["1453"])),
        patch(
            "visualex.app.get_tree",
            new=AsyncMock(
                return_value=(
                    [],
                    3,
                    {
                        "article_lookup": {
                            "1453": {
                                "in_dispositivo": False,
                                "best_annex": "2",
                            }
                        }
                    },
                )
            ),
        ) as mock_get_tree,
    ):
        result = await controller_instance.create_norma_visitata_from_data(
            {
                "act_type": "codice civile",
                "article": "1453",
            }
        )

    mock_get_tree.assert_awaited_once()
    assert len(result) == 1
    assert result[0].allegato == "2"


@pytest.mark.asyncio
async def test_create_norma_visitata_keeps_dispositivo_when_lookup_says_so(controller_instance):
    """The dispositivo must win even when an annex also contains the article."""
    with (
        patch("visualex.app.NORMATTIVA_URN_CODICI", {}),
        patch("visualex.app.parse_article_input", new=AsyncMock(return_value=["1453"])),
        patch(
            "visualex.app.get_tree",
            new=AsyncMock(
                return_value=(
                    [],
                    3,
                    {
                        "article_lookup": {
                            "1453": {
                                "in_dispositivo": True,
                                "best_annex": "2",
                            }
                        }
                    },
                )
            ),
        ),
    ):
        result = await controller_instance.create_norma_visitata_from_data(
            {
                "act_type": "codice civile",
                "article": "1453",
            }
        )

    assert len(result) == 1
    assert result[0].allegato is None


@pytest.mark.asyncio
async def test_create_norma_visitata_falls_back_to_annex_scan_for_legacy_cache(controller_instance):
    """Older cached metadata without article_lookup should still behave identically."""
    legacy_metadata = {
        "annexes": [
            {
                "number": None,
                "article_count": 1,
                "article_numbers": ["1"],
            },
            {
                "number": "1",
                "article_count": 1,
                "article_numbers": ["1453"],
            },
            {
                "number": "2",
                "article_count": 3,
                "article_numbers": ["1453", "1454", "1455"],
            },
        ]
    }

    with (
        patch("visualex.app.NORMATTIVA_URN_CODICI", {}),
        patch("visualex.app.parse_article_input", new=AsyncMock(return_value=["1453"])),
        patch(
            "visualex.app.get_tree",
            new=AsyncMock(return_value=([], 4, legacy_metadata)),
        ),
    ):
        result = await controller_instance.create_norma_visitata_from_data(
            {
                "act_type": "codice civile",
                "article": "1453",
            }
        )

    assert len(result) == 1
    assert result[0].allegato == "2"
