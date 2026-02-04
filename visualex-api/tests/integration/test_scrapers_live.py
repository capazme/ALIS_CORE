"""
Live integration tests for scrapers - these actually hit real websites.

Run with: pytest tests/integration/test_scrapers_live.py -v --run-live
Skip with: pytest (default - skips live tests)

These tests verify that scrapers work against real websites.
They are slow and depend on external services, so they're skipped by default.
"""
import pytest

# Mark all tests in this module as live tests
pytestmark = pytest.mark.live


class TestNormattivaScraperLive:
    """Live tests for NormattivaScraper against real Normattiva.it."""

    @pytest.fixture
    def scraper(self):
        """Create a real NormattivaScraper instance."""
        from visualex.scrapers.normattiva import NormattivaScraper
        return NormattivaScraper()

    @pytest.mark.asyncio
    async def test_fetch_codice_civile_art1453(self, scraper):
        """Test fetching Art. 1453 from Codice Civile (real request).

        Note: Codice Civile is allegato 2 of R.D. 262/1942.
        Without allegato="2", we'd get the dispositivo (law of enactment).
        """
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        # allegato="2" is required for actual Codice Civile content (not dispositivo)
        nv = NormaVisitata(norma=norma, numero_articolo="1453", allegato="2")

        text, url = await scraper.get_document(nv)

        assert text is not None
        assert len(text) > 100
        # Art. 1453 is about contract resolution
        assert "risoluzione" in text.lower() or "contratto" in text.lower()
        assert "normattiva" in url.lower()

    @pytest.mark.asyncio
    async def test_fetch_codice_civile_art2043(self, scraper):
        """Test fetching Art. 2043 from Codice Civile (tort law)."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        # allegato="2" is required for actual Codice Civile content
        nv = NormaVisitata(norma=norma, numero_articolo="2043", allegato="2")

        text, url = await scraper.get_document(nv)

        assert text is not None
        assert len(text) > 50
        # Art. 2043 is about tort/damages
        assert "danno" in text.lower() or "risarcimento" in text.lower()

    @pytest.mark.asyncio
    async def test_fetch_costituzione_art1(self, scraper):
        """Test fetching Art. 1 from Costituzione."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="costituzione")
        nv = NormaVisitata(norma=norma, numero_articolo="1")

        text, url = await scraper.get_document(nv)

        assert text is not None
        # Art. 1 Constitution: "L'Italia è una Repubblica democratica, fondata sul lavoro"
        assert "italia" in text.lower() or "repubblica" in text.lower() or "lavoro" in text.lower()


class TestBrocardiScraperLive:
    """Live tests for BrocardiScraper against real Brocardi.it."""

    @pytest.fixture
    def scraper(self):
        """Create a real BrocardiScraper instance."""
        from visualex.scrapers.brocardi import BrocardiScraper
        return BrocardiScraper()

    @pytest.mark.asyncio
    async def test_fetch_codice_civile_art1453_info(self, scraper):
        """Test fetching commentary for Art. 1453 from Brocardi."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(norma=norma, numero_articolo="1453")

        # First check if Brocardi knows about this article
        url = await scraper.do_know(nv)
        if url is None:
            pytest.skip("Brocardi does not have this article in knowledge base")

        result = await scraper.get_info(nv)

        assert result is not None
        # get_info returns tuple: (position, info_dict, url)
        assert isinstance(result, tuple)
        assert len(result) == 3
        position, info, url = result
        assert isinstance(info, dict)
        # Brocardi should have at least some sections
        assert any(key in info for key in ['Spiegazione', 'Ratio', 'Brocardi', 'Massime'])

    @pytest.mark.asyncio
    async def test_fetch_codice_penale_art640(self, scraper):
        """Test fetching commentary for Art. 640 from Codice Penale (fraud)."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice penale")
        nv = NormaVisitata(norma=norma, numero_articolo="640")

        url = await scraper.do_know(nv)
        if url is None:
            pytest.skip("Brocardi does not have this article in knowledge base")

        result = await scraper.get_info(nv)

        assert result is not None
        # get_info returns tuple: (position, info_dict, url)
        assert isinstance(result, tuple)
        position, info, url = result
        assert isinstance(info, dict)


class TestEurlexScraperLive:
    """Live tests for EurlexScraper against real EUR-Lex."""

    @pytest.fixture
    def scraper(self):
        """Create a real EurlexScraper instance."""
        from visualex.scrapers.eurlex import EurlexScraper
        return EurlexScraper()

    @pytest.mark.asyncio
    async def test_fetch_gdpr_article5(self, scraper):
        """Test fetching Article 5 from GDPR (Regulation 2016/679)."""
        text, url = await scraper.get_document(
            act_type="regolamento ue",
            year="2016",
            num="679",
            article="5"
        )

        assert text is not None
        assert len(text) > 100
        # Article 5 GDPR is about principles of data processing
        assert "eur-lex" in url.lower()

    @pytest.mark.asyncio
    async def test_fetch_consumer_rights_directive(self, scraper):
        """Test fetching Consumer Rights Directive 2011/83/EU."""
        text, url = await scraper.get_document(
            act_type="direttiva ue",
            year="2011",
            num="83",
            article="5"
        )

        assert text is not None
        assert len(text) > 50
        assert "eur-lex" in url.lower()

    @pytest.mark.asyncio
    async def test_fetch_tfue_article101(self, scraper):
        """Test fetching TFUE Article 101 (competition law)."""
        text, url = await scraper.get_document(
            act_type="tfue",
            year=None,
            num=None,
            article="101"
        )

        assert text is not None
        # Article 101 TFUE is about competition
        assert "eur-lex" in url.lower()

    @pytest.mark.asyncio
    async def test_uri_generation_regulation(self, scraper):
        """Test that URI is generated correctly for regulations."""
        uri = scraper.get_uri(act_type="regolamento ue", year="2016", num="679")

        assert uri is not None
        assert "eur-lex" in uri.lower()
        assert "2016" in uri
        assert "679" in uri

    @pytest.mark.asyncio
    async def test_uri_generation_directive(self, scraper):
        """Test that URI is generated correctly for directives."""
        uri = scraper.get_uri(act_type="direttiva ue", year="2019", num="1937")

        assert uri is not None
        assert "eur-lex" in uri.lower()
        assert "2019" in uri
        assert "1937" in uri
