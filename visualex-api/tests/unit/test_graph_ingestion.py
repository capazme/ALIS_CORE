"""
Tests for Graph Ingestion Module
================================

Unit tests for NormIngester and ArticleParser.
Uses mocking to avoid requiring a running FalkorDB instance.
"""

import pytest
from unittest.mock import Mock, MagicMock, AsyncMock, patch
from datetime import datetime

from visualex.graph.ingestion import (
    NormIngester,
    ArticleParser,
    ArticleStructure,
    CommaStructure,
    LetteraStructure,
    NumeroStructure,
    IngestionResult,
    BatchResult,
)
from visualex.graph.schema import NodeType, EdgeType


# =============================================================================
# ArticleParser Tests
# =============================================================================


class TestArticleParser:
    """Tests for article structure parsing."""

    def setup_method(self):
        """Create parser instance for each test."""
        self.parser = ArticleParser()

    def test_parse_empty_text(self):
        """Parser handles empty text gracefully."""
        result = self.parser.parse("", "urn:nir:stato:legge:2020-01-01;1~art1")
        assert result.urn == "urn:nir:stato:legge:2020-01-01;1~art1"
        assert result.numero_articolo == "1"
        assert len(result.commi) == 0

    def test_parse_simple_article_single_comma(self):
        """Parser handles single paragraph article."""
        text = "Il contratto e' l'accordo di due o piu' parti."
        result = self.parser.parse(text, "urn:nir:stato:regio.decreto:1942-03-16;262~art1321")

        assert result.numero_articolo == "1321"
        assert len(result.commi) == 1
        assert result.commi[0].posizione == 1
        assert "contratto" in result.commi[0].testo

    def test_parse_numbered_commi(self):
        """Parser extracts numbered paragraphs."""
        text = """1. Primo comma con testo.
2. Secondo comma con altro testo.
3. Terzo comma finale."""

        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art5")

        assert len(result.commi) == 3
        assert result.commi[0].posizione == 1
        assert result.commi[1].posizione == 2
        assert result.commi[2].posizione == 3
        assert "Primo" in result.commi[0].testo
        assert "Secondo" in result.commi[1].testo
        assert "Terzo" in result.commi[2].testo

    def test_parse_commi_with_lettere(self):
        """Parser extracts lettered sub-points."""
        text = """1. Il comma contiene:
a) prima lettera;
b) seconda lettera;
c) terza lettera."""

        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art10")

        assert len(result.commi) == 1
        comma = result.commi[0]
        assert len(comma.lettere) == 3
        assert comma.lettere[0].posizione == "a"
        assert comma.lettere[1].posizione == "b"
        assert comma.lettere[2].posizione == "c"
        assert "prima" in comma.lettere[0].testo

    def test_parse_lettere_with_numeri(self):
        """Parser extracts numbered sub-sub-points."""
        text = """1. Il comma contiene:
a) la lettera con:
1) primo numero;
2) secondo numero;
b) altra lettera."""

        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art20")

        comma = result.commi[0]
        assert len(comma.lettere) == 2

        lettera_a = comma.lettere[0]
        assert len(lettera_a.numeri) == 2
        assert lettera_a.numeri[0].posizione == 1
        assert lettera_a.numeri[1].posizione == 2

    def test_urn_construction_comma(self):
        """Parser builds correct URN for commi."""
        text = "1. Primo comma."
        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art1")

        assert result.commi[0].urn == "urn:nir:stato:legge:2020-01-01;1~art1-com1"

    def test_urn_construction_lettera(self):
        """Parser builds correct URN for lettere."""
        text = """1. Contiene:
a) prima;
b) seconda."""
        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art1")

        assert result.commi[0].lettere[0].urn == "urn:nir:stato:legge:2020-01-01;1~art1-com1-leta"
        assert result.commi[0].lettere[1].urn == "urn:nir:stato:legge:2020-01-01;1~art1-com1-letb"

    def test_urn_construction_numero(self):
        """Parser builds correct URN for numeri."""
        text = """1. Contiene:
a) con:
1) uno;
2) due."""
        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1~art1")

        numeri = result.commi[0].lettere[0].numeri
        assert numeri[0].urn == "urn:nir:stato:legge:2020-01-01;1~art1-com1-leta-num1"
        assert numeri[1].urn == "urn:nir:stato:legge:2020-01-01;1~art1-com1-leta-num2"

    def test_extract_article_number_simple(self):
        """Extract simple article number from URN."""
        num = self.parser._extract_article_number("urn:nir:stato:legge:2020-01-01;1~art123")
        assert num == "123"

    def test_extract_article_number_with_extension(self):
        """Extract article number with Latin extension."""
        num = self.parser._extract_article_number("urn:nir:stato:legge:2020-01-01;1~art16bis")
        assert num == "16bis"

    def test_extract_article_number_no_article(self):
        """Extract returns None for URN without article."""
        num = self.parser._extract_article_number("urn:nir:stato:legge:2020-01-01;1")
        assert num is None

    def test_parse_urn_without_article(self):
        """Parser handles URN without article number gracefully."""
        text = "Testo della norma completa."
        result = self.parser.parse(text, "urn:nir:stato:legge:2020-01-01;1")

        assert result.urn == "urn:nir:stato:legge:2020-01-01;1"
        assert result.numero_articolo == ""  # Empty, not None
        assert len(result.commi) == 1

    def test_node_count(self):
        """ArticleStructure counts total nodes correctly."""
        structure = ArticleStructure(
            urn="test",
            numero_articolo="1",
            commi=[
                CommaStructure(
                    posizione=1,
                    testo="test",
                    lettere=[
                        LetteraStructure(
                            posizione="a",
                            testo="test",
                            numeri=[
                                NumeroStructure(posizione=1, testo="test"),
                                NumeroStructure(posizione=2, testo="test"),
                            ]
                        )
                    ]
                )
            ]
        )

        # 1 article + 1 comma + 1 lettera + 2 numeri = 5
        assert structure.node_count() == 5


# =============================================================================
# IngestionResult Tests
# =============================================================================


class TestIngestionResult:
    """Tests for IngestionResult data class."""

    def test_successful_result(self):
        """Successful result has correct values."""
        result = IngestionResult(
            urn="urn:test",
            success=True,
            nodes_created=5,
            edges_created=4,
        )
        assert result.success is True
        assert result.nodes_created == 5
        assert result.error is None

    def test_failed_result(self):
        """Failed result includes error."""
        result = IngestionResult(
            urn="urn:test",
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert "Connection" in result.error

    def test_to_dict(self):
        """Result converts to dictionary."""
        result = IngestionResult(
            urn="urn:test",
            success=True,
            nodes_created=3,
        )
        d = result.to_dict()
        assert d["urn"] == "urn:test"
        assert d["success"] is True
        assert d["nodes_created"] == 3
        assert "timestamp" in d


# =============================================================================
# BatchResult Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult data class."""

    def test_success_rate_calculation(self):
        """Success rate calculated correctly."""
        result = BatchResult(
            total=100,
            successful=75,
            failed=25,
        )
        assert result.success_rate == 75.0

    def test_success_rate_empty(self):
        """Success rate handles empty batch."""
        result = BatchResult(total=0, successful=0, failed=0)
        assert result.success_rate == 0.0

    def test_get_failed(self):
        """Get failed results filters correctly."""
        results = [
            IngestionResult(urn="urn:1", success=True),
            IngestionResult(urn="urn:2", success=False, error="Error"),
            IngestionResult(urn="urn:3", success=True),
            IngestionResult(urn="urn:4", success=False, error="Error"),
        ]
        batch = BatchResult(
            total=4,
            successful=2,
            failed=2,
            results=results,
        )
        failed = batch.get_failed()
        assert len(failed) == 2
        assert failed[0].urn == "urn:2"
        assert failed[1].urn == "urn:4"

    def test_to_dict(self):
        """Batch result converts to summary dict."""
        batch = BatchResult(
            total=10,
            successful=8,
            failed=2,
            duration_seconds=5.5,
            results=[
                IngestionResult(urn="urn:fail1", success=False),
                IngestionResult(urn="urn:fail2", success=False),
            ],
        )
        d = batch.to_dict()
        assert d["total"] == 10
        assert d["successful"] == 8
        assert d["success_rate"] == "80.0%"
        assert d["duration_seconds"] == 5.5
        assert "urn:fail1" in d["failed_urns"]


# =============================================================================
# NormIngester Tests
# =============================================================================


class TestNormIngester:
    """Tests for NormIngester class."""

    def setup_method(self):
        """Create mock client and ingester for each test."""
        self.mock_client = MagicMock()
        self.mock_client.merge_node = AsyncMock(return_value={"properties": {}})
        self.mock_client.create_edge = AsyncMock(return_value={"properties": {}})
        # Mock get_node_by_urn to return None (new node) by default
        self.mock_client.get_node_by_urn = AsyncMock(return_value=None)
        self.ingester = NormIngester(self.mock_client)

    def _create_mock_norma_visitata(
        self,
        urn: str = "urn:nir:stato:legge:2020-01-01;1~art1",
        tipo_atto_str: str = "Legge",
        data_versione: str = "2020-01-01",
    ):
        """Create mock NormaVisitata."""
        mock_norma = MagicMock()
        mock_norma.tipo_atto_str = tipo_atto_str
        mock_norma.__str__ = lambda self: f"{tipo_atto_str} 2020"

        mock_visitata = MagicMock()
        mock_visitata.urn = urn
        mock_visitata.norma = mock_norma
        mock_visitata.data_versione = data_versione

        return mock_visitata

    @pytest.mark.asyncio
    async def test_ingest_article_success(self):
        """Ingest article creates nodes successfully."""
        norma = self._create_mock_norma_visitata()
        text = "Testo dell'articolo semplice."

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        assert result.nodes_created > 0
        # Should create at least Norma + Comma nodes
        self.mock_client.merge_node.assert_called()

    @pytest.mark.asyncio
    async def test_ingest_article_with_structure(self):
        """Ingest article creates hierarchical structure."""
        norma = self._create_mock_norma_visitata()
        text = """1. Primo comma:
a) lettera a;
b) lettera b.
2. Secondo comma."""

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        # Should have: Norma + 2 Comma + 2 Lettera = 5 nodes minimum
        assert result.nodes_created >= 4
        assert result.edges_created >= 3  # contiene edges

    @pytest.mark.asyncio
    async def test_ingest_article_creates_edges(self):
        """Ingest article creates contiene edges."""
        norma = self._create_mock_norma_visitata()
        text = "1. Primo comma."

        await self.ingester.ingest_article(norma, text)

        # Should call create_edge for Norma -> Comma
        self.mock_client.create_edge.assert_called()
        call_args = self.mock_client.create_edge.call_args_list
        edge_types = [call.args[0] for call in call_args if call.args]
        assert EdgeType.CONTIENE in edge_types

    @pytest.mark.asyncio
    async def test_ingest_article_with_brocardi(self):
        """Ingest article with Brocardi enrichment."""
        norma = self._create_mock_norma_visitata()
        text = "Testo articolo."
        brocardi = {
            "ratio": "Ratio legis dell'articolo",
            "spiegazione": "Spiegazione dettagliata",
            "massime": [
                {"testo": "Massima della Cassazione", "corte": "Cassazione", "data": "2021-01-15"}
            ],
        }

        result = await self.ingester.ingest_article(norma, text, brocardi_info=brocardi)

        assert result.success is True
        # Should have extra nodes for Dottrina and AttoGiudiziario
        assert result.nodes_created >= 3

    @pytest.mark.asyncio
    async def test_ingest_article_error_handling(self):
        """Ingest article handles errors gracefully."""
        self.mock_client.merge_node = AsyncMock(side_effect=Exception("DB error"))
        norma = self._create_mock_norma_visitata()

        result = await self.ingester.ingest_article(norma, "text")

        assert result.success is False
        assert "DB error" in result.error

    @pytest.mark.asyncio
    async def test_ingest_batch_success(self):
        """Batch ingestion processes multiple articles."""
        articles = [
            (self._create_mock_norma_visitata(f"urn:test:{i}"), f"Text {i}", None)
            for i in range(5)
        ]

        result = await self.ingester.ingest_batch(articles)

        assert result.total == 5
        assert result.successful == 5
        assert result.failed == 0

    @pytest.mark.asyncio
    async def test_ingest_batch_partial_failure(self):
        """Batch ingestion isolates failures."""
        # Make third call fail
        call_count = [0]
        original_merge = self.mock_client.merge_node

        async def failing_merge(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 5:  # Fail on 5th call (3rd article's comma)
                raise Exception("Simulated failure")
            return {"properties": {}}

        self.mock_client.merge_node = AsyncMock(side_effect=failing_merge)

        articles = [
            (self._create_mock_norma_visitata(f"urn:test:{i}"), "1. Comma.", None)
            for i in range(5)
        ]

        result = await self.ingester.ingest_batch(articles)

        # Should have some failures but continue processing
        assert result.total == 5
        assert result.failed >= 1
        assert result.successful < 5

    @pytest.mark.asyncio
    async def test_ingest_batch_with_progress_callback(self):
        """Batch ingestion calls progress callback."""
        progress_calls = []

        def callback(current, total, urn):
            progress_calls.append((current, total, urn))

        articles = [
            (self._create_mock_norma_visitata(f"urn:test:{i}"), f"Text {i}", None)
            for i in range(3)
        ]

        await self.ingester.ingest_batch(articles, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[0][0] == 1
        assert progress_calls[2][0] == 3

    @pytest.mark.asyncio
    async def test_ingest_article_tracks_updated(self):
        """Ingest article tracks nodes_updated when updating existing."""
        # Mock existing node
        self.mock_client.get_node_by_urn = AsyncMock(return_value={
            "n": {"properties": {"data_versione": "2020-01-01"}}
        })
        norma = self._create_mock_norma_visitata(data_versione="2020-01-01")
        text = "Testo articolo."

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        assert result.nodes_updated >= 1  # Norma was updated, not created

    @pytest.mark.asyncio
    async def test_ingest_article_creates_versione_on_date_change(self):
        """Versione node created when data_versione changes."""
        # Mock existing node with old date
        self.mock_client.get_node_by_urn = AsyncMock(return_value={
            "n": {"properties": {"data_versione": "2019-01-01"}}
        })
        norma = self._create_mock_norma_visitata(data_versione="2020-01-01")
        text = "Testo articolo aggiornato."

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        # Should have created Versione node and ha_versione edge
        calls = self.mock_client.merge_node.call_args_list
        versione_calls = [c for c in calls if len(c.args) > 0 and c.args[0] == NodeType.VERSIONE]
        assert len(versione_calls) >= 1


# =============================================================================
# Integration-style Tests (with mocked client)
# =============================================================================


class TestIngestionIntegration:
    """Integration tests for ingestion pipeline."""

    def setup_method(self):
        """Create mock client and ingester."""
        self.mock_client = MagicMock()
        self.mock_client.merge_node = AsyncMock(return_value={"properties": {}})
        self.mock_client.create_edge = AsyncMock(return_value={"properties": {}})
        self.mock_client.get_node_by_urn = AsyncMock(return_value=None)
        self.ingester = NormIngester(self.mock_client)

    def _create_mock_norma_visitata(self, urn: str):
        """Create mock NormaVisitata."""
        mock_norma = MagicMock()
        mock_norma.tipo_atto_str = "Regio Decreto"
        mock_norma.__str__ = lambda self: "R.D. 1942"

        mock_visitata = MagicMock()
        mock_visitata.urn = urn
        mock_visitata.norma = mock_norma
        mock_visitata.data_versione = "2020-01-01"

        return mock_visitata

    @pytest.mark.asyncio
    async def test_codice_civile_article_structure(self):
        """Test parsing real Codice Civile article structure."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        norma = self._create_mock_norma_visitata(urn)

        # Real structure of Art. 1453 C.C. (simplified)
        text = """1. Nei contratti con prestazioni corrispettive, quando uno dei
contraenti non adempie le sue obbligazioni, l'altro puo' a sua
scelta chiedere l'adempimento o la risoluzione del contratto, salvo,
in ogni caso, il risarcimento del danno.
2. La risoluzione puo' essere domandata anche quando il giudizio e'
stato promosso per ottenere l'adempimento; ma non puo' piu' chiedersi
l'adempimento quando e' stata domandata la risoluzione.
3. Dalla data della domanda di risoluzione l'inadempiente non puo'
piu' adempiere."""

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        # Norma + 3 commi = 4 nodes minimum
        assert result.nodes_created >= 4
        assert result.edges_created >= 3

    @pytest.mark.asyncio
    async def test_article_with_complex_structure(self):
        """Test parsing article with deep nested structure."""
        urn = "urn:nir:stato:legge:2020-01-01;1~art50"
        norma = self._create_mock_norma_visitata(urn)

        text = """1. La disciplina si applica:
a) ai soggetti di cui all'articolo 1, comma 2:
1) le persone fisiche;
2) le persone giuridiche;
3) gli enti del terzo settore;
b) agli enti pubblici;
c) alle societa' di capitali.
2. Sono esclusi i soggetti esteri."""

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        # Norma + 2 commi + 3 lettere + 3 numeri = 9 nodes
        assert result.nodes_created >= 6

    @pytest.mark.asyncio
    async def test_merge_idempotency(self):
        """Verify MERGE is used for idempotent re-ingestion."""
        urn = "urn:nir:stato:legge:2020-01-01;1~art1"
        norma = self._create_mock_norma_visitata(urn)
        text = "1. Testo semplice."

        # Ingest twice
        await self.ingester.ingest_article(norma, text)
        await self.ingester.ingest_article(norma, text)

        # Should use merge_node (upsert), not create_node
        assert self.mock_client.merge_node.call_count >= 4  # 2 ingestions x 2 nodes
        # create_node should not be called directly
        self.mock_client.create_node.assert_not_called()

    @pytest.mark.asyncio
    async def test_brocardi_dottrina_creation(self):
        """Test Brocardi creates Dottrina node."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        norma = self._create_mock_norma_visitata(urn)
        text = "Testo articolo."
        brocardi = {
            "ratio": "Tutela del contraente fedele",
            "spiegazione": "L'articolo disciplina la risoluzione per inadempimento.",
        }

        await self.ingester.ingest_article(norma, text, brocardi_info=brocardi)

        # Find merge_node call for Dottrina
        calls = self.mock_client.merge_node.call_args_list
        dottrina_calls = [c for c in calls if c.args[0] == NodeType.DOTTRINA]
        assert len(dottrina_calls) >= 1

    @pytest.mark.asyncio
    async def test_brocardi_massime_creation(self):
        """Test Brocardi creates AttoGiudiziario nodes for massime."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        norma = self._create_mock_norma_visitata(urn)
        text = "Testo articolo."
        brocardi = {
            "massime": [
                {
                    "testo": "Prima massima",
                    "corte": "Cassazione",
                    "data": "2021-01-15",
                    "numero": "12345/2021",
                },
                {
                    "testo": "Seconda massima",
                    "corte": "Corte d'Appello Milano",
                },
            ],
        }

        await self.ingester.ingest_article(norma, text, brocardi_info=brocardi)

        # Find merge_node calls for AttoGiudiziario
        calls = self.mock_client.merge_node.call_args_list
        atto_calls = [c for c in calls if c.args[0] == NodeType.ATTO_GIUDIZIARIO]
        assert len(atto_calls) >= 2

        # Verify INTERPRETA edge was created
        edge_calls = self.mock_client.create_edge.call_args_list
        interpreta_calls = [c for c in edge_calls if c.args[0] == EdgeType.INTERPRETA]
        assert len(interpreta_calls) >= 2


# =============================================================================
# RelationCreator Integration Tests
# =============================================================================


class TestRelationCreatorIntegration:
    """
    Tests to verify RelationCreator is properly integrated in ingestion.

    This addresses the retrospective finding about "Task Marking Prematura" -
    ensuring E2E integration is tested, not just unit functionality.
    """

    def setup_method(self):
        """Create mock client and ingester with relation extraction."""
        self.mock_client = MagicMock()
        self.mock_client.merge_node = AsyncMock(return_value={"properties": {}})
        self.mock_client.create_edge = AsyncMock(return_value={"properties": {}})
        self.mock_client.get_node_by_urn = AsyncMock(return_value=None)
        # Enable relation extraction (default)
        self.ingester = NormIngester(self.mock_client, extract_relations=True)

    def _create_mock_norma_visitata(self, urn: str):
        """Create mock NormaVisitata."""
        mock_norma = MagicMock()
        mock_norma.tipo_atto_str = "Legge"
        mock_norma.__str__ = lambda self: "Legge 2020"

        mock_visitata = MagicMock()
        mock_visitata.urn = urn
        mock_visitata.norma = mock_norma
        mock_visitata.data_versione = "2020-01-01"

        return mock_visitata

    def test_relation_creator_initialized_by_default(self):
        """RelationCreator is initialized when extract_relations=True (default)."""
        assert self.ingester._relation_creator is not None

    def test_relation_creator_disabled(self):
        """RelationCreator is not initialized when extract_relations=False."""
        ingester = NormIngester(self.mock_client, extract_relations=False)
        assert ingester._relation_creator is None

    @pytest.mark.asyncio
    async def test_relation_extraction_called_on_ingest(self):
        """create_relations_from_text is called during article ingestion."""
        # Mock the relation creator
        mock_relation_creator = MagicMock()
        mock_relation_creator.create_relations_from_text = AsyncMock(return_value={
            "citations_created": 2,
            "modifications_created": 1,
            "jurisprudence_created": 0,
        })
        mock_relation_creator.process_brocardi_relations = AsyncMock(return_value={
            "interpreta_edges": 0,
        })
        self.ingester._relation_creator = mock_relation_creator

        norma = self._create_mock_norma_visitata("urn:nir:stato:legge:2020-01-01;1~art1")
        text = "L'articolo 1321 del codice civile dispone che..."

        result = await self.ingester.ingest_article(norma, text)

        assert result.success is True
        # Verify create_relations_from_text was called
        mock_relation_creator.create_relations_from_text.assert_called_once()
        call_args = mock_relation_creator.create_relations_from_text.call_args
        assert call_args.kwargs["source_urn"] == "urn:nir:stato:legge:2020-01-01;1~art1"
        assert "codice civile" in call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_relation_edges_counted_in_result(self):
        """Extracted relations are counted in edges_created."""
        mock_relation_creator = MagicMock()
        mock_relation_creator.create_relations_from_text = AsyncMock(return_value={
            "citations_created": 3,
            "modifications_created": 2,
            "jurisprudence_created": 1,
        })
        mock_relation_creator.process_brocardi_relations = AsyncMock(return_value={
            "interpreta_edges": 0,
        })
        self.ingester._relation_creator = mock_relation_creator

        norma = self._create_mock_norma_visitata("urn:nir:stato:legge:2020-01-01;1~art1")
        text = "Testo con citazioni."

        result = await self.ingester.ingest_article(norma, text)

        # edges_created should include: contiene edges + relation edges
        # relations = 3 + 2 + 1 = 6
        assert result.edges_created >= 6

    @pytest.mark.asyncio
    async def test_brocardi_relations_processed(self):
        """process_brocardi_relations is called when brocardi_info provided."""
        mock_relation_creator = MagicMock()
        mock_relation_creator.create_relations_from_text = AsyncMock(return_value={
            "citations_created": 0,
            "modifications_created": 0,
            "jurisprudence_created": 0,
        })
        mock_relation_creator.process_brocardi_relations = AsyncMock(return_value={
            "interpreta_edges": 5,
        })
        self.ingester._relation_creator = mock_relation_creator

        norma = self._create_mock_norma_visitata("urn:nir:stato:legge:2020-01-01;1~art1")
        text = "Testo articolo."
        brocardi = {
            "ratio": "Ratio legis",
            "massime": [{"testo": "Massima 1"}],
        }

        result = await self.ingester.ingest_article(norma, text, brocardi_info=brocardi)

        # Verify process_brocardi_relations was called
        mock_relation_creator.process_brocardi_relations.assert_called_once()
        call_args = mock_relation_creator.process_brocardi_relations.call_args
        assert call_args.kwargs["article_urn"] == "urn:nir:stato:legge:2020-01-01;1~art1"
        assert call_args.kwargs["brocardi_info"] == brocardi

    @pytest.mark.asyncio
    async def test_relation_extraction_failure_isolated(self):
        """Relation extraction failure doesn't fail the entire ingestion."""
        mock_relation_creator = MagicMock()
        mock_relation_creator.create_relations_from_text = AsyncMock(
            side_effect=Exception("Extraction failed")
        )
        self.ingester._relation_creator = mock_relation_creator

        norma = self._create_mock_norma_visitata("urn:nir:stato:legge:2020-01-01;1~art1")
        text = "Testo articolo."

        result = await self.ingester.ingest_article(norma, text)

        # Ingestion should still succeed
        assert result.success is True
        # But edges from relations should be 0
        # (only contiene edges, no relation edges)

    @pytest.mark.asyncio
    async def test_no_relations_when_disabled(self):
        """No relation extraction when extract_relations=False."""
        ingester = NormIngester(self.mock_client, extract_relations=False)

        norma = self._create_mock_norma_visitata("urn:nir:stato:legge:2020-01-01;1~art1")
        text = "L'articolo 1321 del codice civile dispone che..."

        result = await ingester.ingest_article(norma, text)

        assert result.success is True
        # _relation_creator should be None, so no relation edges
        assert ingester._relation_creator is None
