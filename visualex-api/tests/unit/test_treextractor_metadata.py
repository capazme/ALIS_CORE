import pytest
from bs4 import BeautifulSoup

from visualex.utils.treextractor import _parse_normattiva_tree


@pytest.mark.asyncio
async def test_parse_normattiva_tree_builds_article_lookup_metadata():
    """Metadata should include an O(1) lookup for smart annex resolution."""
    html = """
    <div id="albero">
      <ul>
        <li class="box_articoli">Articoli</li>
        <li><a class="numero_articolo">Art. 1</a></li>
        <li class="box_articoli">Allegati</li>
        <li class="box_allegati"><span>Disposizioni sulla legge in generale</span></li>
        <li><a class="numero_articolo">Art. 2</a></li>
        <li class="box_allegati_small">CODICE CIVILE</li>
        <li><a class="numero_articolo">Art. 1453</a></li>
        <li><a class="numero_articolo">Art. 2</a></li>
      </ul>
    </div>
    """

    soup = BeautifulSoup(html, "html.parser")

    _, count, metadata = await _parse_normattiva_tree(
        soup,
        "https://www.normattiva.it/fake",
        link=False,
        details=False,
        return_metadata=True,
    )

    assert count == 4
    assert metadata["annexes"][0]["number"] is None
    assert metadata["annexes"][1]["number"] == "1"
    assert metadata["annexes"][2]["number"] == "2"
    assert metadata["article_lookup"]["1"] == {
        "in_dispositivo": True,
        "best_annex": None,
    }
    assert metadata["article_lookup"]["1453"] == {
        "in_dispositivo": False,
        "best_annex": "2",
    }
    assert metadata["article_lookup"]["2"] == {
        "in_dispositivo": False,
        "best_annex": "2",
    }
