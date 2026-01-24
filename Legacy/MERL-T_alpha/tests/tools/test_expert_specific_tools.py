"""
Test per i tools degli Expert (Fase 4 v2 Recovery).

Sezione 1: Tools originali
1. SemanticSearchTool - ricerca semantica ibrida
2. GraphSearchTool - traversal del grafo
3. ArticleFetchTool - fetch da Normattiva
4. DefinitionLookupTool - ricerca definizioni
5. HierarchyNavigationTool - navigazione gerarchia
6. VerificationTool - verifica esistenza fonti

Sezione 2: Nuovi tools specifici Expert (6 tools)
7. ExternalSourceTool - cascata graph→normattiva→brocardi
8. TextualReferenceTool - rinvii normativi
9. HistoricalEvolutionTool - storia norma
10. PrincipleLookupTool - principi giuridici
11. ConstitutionalBasisTool - gerarchia Kelseniana
12. CitationChainTool - catena citazioni

Ogni tool ha almeno 4 test per coprire i casi d'uso principali.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from merlt.tools import (
    SemanticSearchTool,
    GraphSearchTool,
    ArticleFetchTool,
    DefinitionLookupTool,
    HierarchyNavigationTool,
    VerificationTool,
    ToolResult,
    # Nuovi tools (Fase 4)
    ExternalSourceTool,
    TextualReferenceTool,
    HistoricalEvolutionTool,
    PrincipleLookupTool,
    ConstitutionalBasisTool,
    CitationChainTool,
)


# ===================================
# Fixtures
# ===================================


@pytest.fixture
def mock_graph_db():
    """
    Mock FalkorDBClient.

    Simula risposte del grafo per testing senza database reale.
    """
    mock = AsyncMock()

    # Default response: articolo del codice civile
    mock.query.return_value = [
        {
            "URN": "urn:norma:cc:art1453",
            "estremi": "Art. 1453 c.c.",
            "testo_vigente": "La risoluzione del contratto per inadempimento...",
            "rubrica": "Risoluzione per inadempimento",
            "numero_articolo": "1453"
        }
    ]

    return mock


@pytest.fixture
def mock_retriever():
    """
    Mock GraphAwareRetriever.

    Simula risultati di ricerca semantica.
    """
    mock = AsyncMock()

    # Mock result object
    class MockResult:
        def __init__(self, chunk_id, text, similarity_score, graph_score):
            self.chunk_id = chunk_id
            self.text = text
            self.similarity_score = similarity_score
            self.graph_score = graph_score
            self.final_score = (similarity_score + graph_score) / 2
            self.linked_nodes = []
            self.metadata = {"article_urn": "urn:norma:cc:art1453"}

    mock.retrieve.return_value = [
        MockResult(
            chunk_id="chunk-001",
            text="Art. 1453 c.c. - La risoluzione del contratto...",
            similarity_score=0.85,
            graph_score=0.75
        ),
        MockResult(
            chunk_id="chunk-002",
            text="Art. 1218 c.c. - Il debitore che non esegue...",
            similarity_score=0.72,
            graph_score=0.68
        )
    ]

    return mock


@pytest.fixture
def mock_embeddings():
    """
    Mock EmbeddingService.

    Simula encoding di query in embedding.
    """
    mock = MagicMock()

    # encode_query ritorna numpy array
    import numpy as np
    mock.encode_query.return_value = np.random.randn(1024).astype(np.float32)

    return mock


@pytest.fixture
def mock_scraper():
    """
    Mock NormattivaScraper.

    Simula fetch di documenti da Normattiva.
    """
    mock = AsyncMock()

    # get_document ritorna (testo, urn)
    mock.get_document.return_value = (
        "Art. 1453 c.c. - La risoluzione del contratto per inadempimento...",
        "urn:norma:cc:1942-03-16;262!vig=;art1453"
    )

    return mock


@pytest.fixture
def mock_bridge():
    """
    Mock BridgeTable.

    Simula mappatura chunk ↔ nodo grafo.
    """
    mock = AsyncMock()

    # get_chunks_for_node ritorna lista di chunk IDs
    mock.get_chunks_for_node.return_value = [
        "chunk-001",
        "chunk-002",
        "chunk-003"
    ]

    return mock


# ===================================
# SemanticSearchTool Tests
# ===================================


class TestSemanticSearchTool:
    """Test SemanticSearchTool - ricerca semantica ibrida."""

    @pytest.mark.asyncio
    async def test_semantic_search_finds_results(self, mock_retriever, mock_embeddings):
        """Verifica che semantic search trovi risultati."""
        tool = SemanticSearchTool(
            retriever=mock_retriever,
            embeddings=mock_embeddings
        )

        result = await tool(
            query="Cos'è la risoluzione del contratto?",
            top_k=5
        )

        assert result.success
        assert "results" in result.data
        assert len(result.data["results"]) == 2
        assert result.data["results"][0]["final_score"] > 0

        # Verifica chiamate
        mock_embeddings.encode_query.assert_called_once()
        mock_retriever.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_semantic_search_filters_by_source_type(self, mock_retriever, mock_embeddings):
        """Verifica che source_types filtri correttamente."""
        tool = SemanticSearchTool(
            retriever=mock_retriever,
            embeddings=mock_embeddings
        )

        result = await tool(
            query="legittima difesa",
            top_k=3,
            source_types=["norma"]
        )

        assert result.success

        # Verifica che retrieve sia stato chiamato con source_types
        call_kwargs = mock_retriever.retrieve.call_args.kwargs
        assert call_kwargs.get("source_types") == ["norma"]

    @pytest.mark.asyncio
    async def test_semantic_search_respects_min_score(self, mock_retriever, mock_embeddings):
        """Verifica che min_score filtri risultati con score basso."""
        tool = SemanticSearchTool(
            retriever=mock_retriever,
            embeddings=mock_embeddings
        )

        result = await tool(
            query="test query",
            top_k=10,
            min_score=0.9  # Molto alto - escluderà risultati
        )

        assert result.success
        # Tutti i risultati dovrebbero avere final_score >= 0.9
        for item in result.data["results"]:
            assert item["final_score"] >= 0.9 or len(result.data["results"]) == 0

    @pytest.mark.asyncio
    async def test_semantic_search_handles_no_embeddings(self, mock_retriever):
        """Verifica gestione mancanza EmbeddingService."""
        tool = SemanticSearchTool(
            retriever=mock_retriever,
            embeddings=None  # Nessun embedding service
        )

        result = await tool(query="test")

        assert not result.success
        assert "EmbeddingService non configurato" in result.error


# ===================================
# GraphSearchTool Tests
# ===================================


class TestGraphSearchTool:
    """Test GraphSearchTool - traversal del knowledge graph."""

    @pytest.mark.asyncio
    async def test_graph_search_traverses_relations(self, mock_graph_db):
        """Verifica che graph search faccia traversal."""
        # Mock result con nodi e relazioni
        mock_graph_db.query.return_value = [
            {
                "node": {
                    "URN": "urn:norma:cc:art1218",
                    "estremi": "Art. 1218 c.c.",
                    "properties": {"testo_vigente": "Il debitore..."}
                },
                "rel": {"type": "DISCIPLINA"}
            }
        ]

        tool = GraphSearchTool(graph_db=mock_graph_db)

        result = await tool(
            start_node="urn:norma:cc:art1453",
            relation_types=["disciplina", "rinvia"],
            max_hops=2
        )

        assert result.success
        assert "nodes" in result.data
        assert "edges" in result.data

        # Verifica query Cypher costruita
        mock_graph_db.query.assert_called_once()
        call_args = mock_graph_db.query.call_args
        cypher_query = call_args[0][0]
        assert "MATCH" in cypher_query
        assert "start_urn" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_graph_search_max_depth(self, mock_graph_db):
        """Verifica che max_hops sia rispettato."""
        tool = GraphSearchTool(graph_db=mock_graph_db, default_max_hops=3)

        await tool(
            start_node="urn:norma:cc:art1453",
            max_hops=2
        )

        # Verifica che query contenga max_hops=2
        cypher_query = mock_graph_db.query.call_args[0][0]
        assert "*1..2" in cypher_query

    @pytest.mark.asyncio
    async def test_graph_search_direction(self, mock_graph_db):
        """Verifica direzione del traversal."""
        tool = GraphSearchTool(graph_db=mock_graph_db)

        # Test outgoing (default)
        await tool(
            start_node="urn:norma:cc:art1453",
            direction="outgoing"
        )
        cypher_outgoing = mock_graph_db.query.call_args[0][0]
        assert "->" in cypher_outgoing

        # Test incoming
        await tool(
            start_node="urn:norma:cc:art1453",
            direction="incoming"
        )
        cypher_incoming = mock_graph_db.query.call_args[0][0]
        assert "<-" in cypher_incoming

    @pytest.mark.asyncio
    async def test_graph_search_no_graph_db(self):
        """Verifica gestione mancanza FalkorDB."""
        tool = GraphSearchTool(graph_db=None)

        result = await tool(start_node="urn:norma:cc:art1453")

        assert not result.success
        assert "FalkorDB client non configurato" in result.error


# ===================================
# ArticleFetchTool Tests
# ===================================


class TestArticleFetchTool:
    """Test ArticleFetchTool - fetch da Normattiva."""

    @pytest.mark.asyncio
    async def test_article_fetch_retrieves_text(self, mock_scraper):
        """Verifica che ArticleFetchTool recuperi testo articolo."""
        tool = ArticleFetchTool(scraper=mock_scraper)

        result = await tool(
            tipo_atto="codice civile",
            numero_articolo="1453"
        )

        assert result.success
        assert "text" in result.data
        assert "urn" in result.data
        assert "Art. 1453" in result.data["text"]

        # Verifica chiamata scraper
        mock_scraper.get_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_article_fetch_handles_decreto(self, mock_scraper):
        """Verifica fetch per decreti/leggi con data e numero."""
        tool = ArticleFetchTool(scraper=mock_scraper)

        result = await tool(
            tipo_atto="decreto legislativo",
            numero_articolo="52",
            data_atto="2006-04-12",
            numero_atto="163"
        )

        assert result.success
        # Verifica che NormaVisitata sia creato correttamente
        mock_scraper.get_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_article_fetch_returns_urn(self, mock_scraper):
        """Verifica che tool ritorni URN corretto."""
        tool = ArticleFetchTool(scraper=mock_scraper)

        result = await tool(
            tipo_atto="codice penale",
            numero_articolo="52"
        )

        assert result.success
        assert result.data["urn"].startswith("urn:norma")
        assert result.data["source"] == "normattiva"

    @pytest.mark.asyncio
    async def test_article_fetch_handles_error(self, mock_scraper):
        """Verifica gestione errori durante fetch."""
        mock_scraper.get_document.side_effect = Exception("Network error")

        tool = ArticleFetchTool(scraper=mock_scraper)

        result = await tool(
            tipo_atto="codice civile",
            numero_articolo="9999"
        )

        assert not result.success
        assert "Impossibile recuperare articolo" in result.error


# ===================================
# DefinitionLookupTool Tests
# ===================================


class TestDefinitionLookupTool:
    """Test DefinitionLookupTool - ricerca definizioni legali."""

    @pytest.mark.asyncio
    async def test_definition_lookup_finds_definitions(self, mock_graph_db):
        """Verifica che definition lookup trovi definizioni."""
        # Mock risposta con definizione via DEFINISCE
        mock_graph_db.query.return_value = [
            {
                "term": "contratto",
                "source_urn": "urn:norma:cc:art1321",
                "source_type": "Norma",
                "source_estremi": "Art. 1321 c.c.",
                "definition_text": "Il contratto è l'accordo di due o più parti...",
                "context": "Titolo II - Dei contratti in generale"
            }
        ]

        tool = DefinitionLookupTool(graph_db=mock_graph_db)

        result = await tool(
            term="contratto",
            exact_match=False
        )

        assert result.success
        assert "definitions" in result.data
        assert len(result.data["definitions"]) > 0
        assert result.data["definitions"][0]["term"] == "contratto"
        assert result.data["definitions"][0]["confidence"] == 1.0  # DEFINISCE relation

    @pytest.mark.asyncio
    async def test_definition_lookup_fuzzy_match(self, mock_graph_db):
        """Verifica che ricerca fuzzy funzioni."""
        mock_graph_db.query.return_value = [
            {
                "term": "legittima difesa",
                "source_urn": "urn:norma:cp:art52",
                "source_type": "Norma",
                "source_estremi": "Art. 52 c.p.",
                "definition_text": "Non è punibile chi ha commesso il fatto..."
            }
        ]

        tool = DefinitionLookupTool(graph_db=mock_graph_db)

        # Cerca con term parziale
        result = await tool(
            term="difesa",  # Parziale
            exact_match=False
        )

        assert result.success

        # Verifica query Cypher usa CONTAINS
        cypher = mock_graph_db.query.call_args[0][0]
        assert "toLower" in cypher
        assert "CONTAINS" in cypher

    @pytest.mark.asyncio
    async def test_definition_lookup_filters_source_types(self, mock_graph_db):
        """Verifica filtro per tipo fonte."""
        tool = DefinitionLookupTool(graph_db=mock_graph_db)

        await tool(
            term="responsabilità",
            source_types=["Norma"]
        )

        # Verifica che query filtri per Norma
        cypher = mock_graph_db.query.call_args[0][0]
        assert ":Norma" in cypher or "source_filter" in cypher

    @pytest.mark.asyncio
    async def test_definition_lookup_empty_results(self, mock_graph_db):
        """Verifica gestione nessuna definizione trovata."""
        mock_graph_db.query.return_value = []

        tool = DefinitionLookupTool(graph_db=mock_graph_db)

        result = await tool(term="termine_inesistente")

        assert result.success
        assert result.data["total"] == 0
        assert len(result.data["definitions"]) == 0


# ===================================
# HierarchyNavigationTool Tests
# ===================================


class TestHierarchyNavigationTool:
    """Test HierarchyNavigationTool - navigazione gerarchia normativa."""

    @pytest.mark.asyncio
    async def test_hierarchy_navigation_finds_ancestors(self, mock_graph_db):
        """Verifica che hierarchy tool trovi antenati."""
        # Mock: find_start_node
        mock_graph_db.query.side_effect = [
            # First call: find start node
            [
                {
                    "urn": "urn:norma:cc:art1453",
                    "tipo": "Articolo",
                    "estremi": "Art. 1453 c.c.",
                    "rubrica": "Risoluzione per inadempimento",
                    "numero": "1453"
                }
            ],
            # Second call: ancestors
            [
                {
                    "urn": "urn:norma:cc:capo14",
                    "tipo": "Capo",
                    "estremi": "Capo XIV",
                    "rubrica": "Della risoluzione del contratto",
                    "depth": 1
                },
                {
                    "urn": "urn:norma:cc:titolo1",
                    "tipo": "Titolo",
                    "estremi": "Titolo I",
                    "rubrica": "Dei contratti in generale",
                    "depth": 2
                }
            ]
        ]

        tool = HierarchyNavigationTool(graph_db=mock_graph_db)

        result = await tool(
            start_node="urn:norma:cc:art1453",
            direction="ancestors",
            max_depth=5
        )

        assert result.success
        assert "hierarchy" in result.data
        assert len(result.data["hierarchy"]) == 2
        assert result.data["hierarchy"][0]["tipo"] == "Capo"

    @pytest.mark.asyncio
    async def test_hierarchy_navigation_finds_descendants(self, mock_graph_db):
        """Verifica che hierarchy tool trovi discendenti."""
        mock_graph_db.query.side_effect = [
            # find_start_node
            [{"urn": "urn:norma:cc:libro4", "tipo": "Libro", "estremi": "Libro IV"}],
            # descendants
            [
                {"urn": "urn:norma:cc:titolo1", "tipo": "Titolo", "estremi": "Titolo I", "depth": 1},
                {"urn": "urn:norma:cc:capo1", "tipo": "Capo", "estremi": "Capo I", "depth": 2}
            ]
        ]

        tool = HierarchyNavigationTool(graph_db=mock_graph_db)

        result = await tool(
            start_node="urn:norma:cc:libro4",
            direction="descendants",
            max_depth=2
        )

        assert result.success
        assert len(result.data["hierarchy"]) == 2

    @pytest.mark.asyncio
    async def test_hierarchy_navigation_finds_siblings(self, mock_graph_db):
        """Verifica che hierarchy tool trovi fratelli."""
        mock_graph_db.query.side_effect = [
            # find_start_node
            [{"urn": "urn:norma:cc:art1453", "tipo": "Articolo", "estremi": "Art. 1453 c.c."}],
            # siblings
            [
                {"urn": "urn:norma:cc:art1454", "tipo": "Articolo", "estremi": "Art. 1454 c.c.", "order_num": "1454"},
                {"urn": "urn:norma:cc:art1455", "tipo": "Articolo", "estremi": "Art. 1455 c.c.", "order_num": "1455"}
            ]
        ]

        tool = HierarchyNavigationTool(graph_db=mock_graph_db)

        result = await tool(
            start_node="urn:norma:cc:art1453",
            direction="siblings"
        )

        assert result.success
        assert len(result.data["hierarchy"]) == 2
        assert all(n["tipo"] == "Articolo" for n in result.data["hierarchy"])

    @pytest.mark.asyncio
    async def test_hierarchy_navigation_context_mode(self, mock_graph_db):
        """Verifica modalità context (ancestors + siblings + descendants)."""
        mock_graph_db.query.side_effect = [
            # find_start_node
            [{"urn": "urn:norma:cc:art1453", "tipo": "Articolo"}],
            # ancestors
            [{"urn": "urn:norma:cc:capo14", "tipo": "Capo", "depth": 1}],
            # siblings
            [{"urn": "urn:norma:cc:art1454", "tipo": "Articolo"}],
            # descendants (1 level)
            []
        ]

        tool = HierarchyNavigationTool(graph_db=mock_graph_db)

        result = await tool(
            start_node="urn:norma:cc:art1453",
            direction="context"
        )

        assert result.success
        # Dovrebbe contenere ancestors + siblings
        assert any(n.get("relation") == "ancestor" for n in result.data["hierarchy"])
        assert any(n.get("relation") == "sibling" for n in result.data["hierarchy"])


# ===================================
# VerificationTool Tests
# ===================================


class TestVerificationTool:
    """Test VerificationTool - verifica esistenza fonti."""

    @pytest.mark.asyncio
    async def test_verification_verifies_existing_source(self, mock_graph_db, mock_bridge):
        """Verifica che verification tool confermi fonte esistente."""
        # Mock: source esiste nel grafo
        mock_graph_db.query.return_value = [
            {
                "node_type": "Norma",
                "urn": "urn:norma:cc:art1453"
            }
        ]

        # Mock: source ha chunks collegati
        mock_bridge.get_chunks_for_node.return_value = ["chunk-001", "chunk-002"]

        tool = VerificationTool(
            graph_db=mock_graph_db,
            bridge=mock_bridge,
            require_chunks=True
        )

        result = await tool(
            source_ids=["urn:norma:cc:art1453"],
            strict_mode=True
        )

        assert result.success
        assert result.data["all_verified"]
        assert "urn:norma:cc:art1453" in result.data["verified"]
        assert len(result.data["unverified"]) == 0

    @pytest.mark.asyncio
    async def test_verification_detects_nonexistent_source(self, mock_graph_db, mock_bridge):
        """Verifica che verification tool rilevi fonte inesistente."""
        # Mock: source NON esiste nel grafo
        mock_graph_db.query.return_value = []

        tool = VerificationTool(
            graph_db=mock_graph_db,
            bridge=mock_bridge
        )

        result = await tool(
            source_ids=["urn:norma:fake:art9999"],
            strict_mode=True
        )

        assert result.success
        assert not result.data["all_verified"]
        assert "urn:norma:fake:art9999" in result.data["unverified"]

    @pytest.mark.asyncio
    async def test_verification_handles_partial_verification(self, mock_graph_db, mock_bridge):
        """Verifica fonte nel grafo MA senza chunks (partial)."""
        # Mock: source esiste nel grafo
        mock_graph_db.query.return_value = [
            {"node_type": "Norma", "urn": "urn:norma:cc:art1453"}
        ]

        # Mock: NO chunks collegati
        mock_bridge.get_chunks_for_node.return_value = []

        tool = VerificationTool(
            graph_db=mock_graph_db,
            bridge=mock_bridge,
            require_chunks=True
        )

        result = await tool(
            source_ids=["urn:norma:cc:art1453"],
            strict_mode=True
        )

        assert result.success
        assert not result.data["all_verified"]
        assert "urn:norma:cc:art1453" in result.data["partial"]

    @pytest.mark.asyncio
    async def test_verification_non_strict_mode(self, mock_graph_db, mock_bridge):
        """Verifica che non-strict mode richieda solo esistenza grafo."""
        # Mock: source esiste nel grafo
        mock_graph_db.query.return_value = [
            {"node_type": "Norma", "urn": "urn:norma:cc:art1453"}
        ]

        # Mock: NO chunks (ma in non-strict va bene)
        mock_bridge.get_chunks_for_node.return_value = []

        tool = VerificationTool(
            graph_db=mock_graph_db,
            bridge=mock_bridge
        )

        result = await tool(
            source_ids=["urn:norma:cc:art1453"],
            strict_mode=False  # Non-strict: basta esistenza grafo
        )

        assert result.success
        assert result.data["all_verified"]
        assert "urn:norma:cc:art1453" in result.data["verified"]


# ===================================
# Integration Tests
# ===================================


class TestToolsIntegration:
    """Test integrazione tra tools."""

    @pytest.mark.asyncio
    async def test_semantic_then_verification(self, mock_retriever, mock_embeddings, mock_graph_db, mock_bridge):
        """
        Test workflow: semantic search + verifica fonti.

        Flow tipico di un Expert:
        1. Cerca con semantic search
        2. Verifica che le fonti trovate esistano
        """
        # Step 1: Semantic search
        search_tool = SemanticSearchTool(
            retriever=mock_retriever,
            embeddings=mock_embeddings
        )

        search_result = await search_tool(query="contratto")

        assert search_result.success

        # Step 2: Estrai source IDs dai risultati
        source_ids = [
            r.get("metadata", {}).get("article_urn", r.get("chunk_id"))
            for r in search_result.data["results"]
        ]

        # Step 3: Verifica fonti
        mock_graph_db.query.return_value = [{"node_type": "Norma", "urn": source_ids[0]}]
        mock_bridge.get_chunks_for_node.return_value = ["chunk-001"]

        verify_tool = VerificationTool(
            graph_db=mock_graph_db,
            bridge=mock_bridge
        )

        verify_result = await verify_tool(source_ids=source_ids)

        assert verify_result.success

    @pytest.mark.asyncio
    async def test_definition_then_hierarchy(self, mock_graph_db):
        """
        Test workflow: cerca definizione + naviga gerarchia.

        Flow: trova definizione → esplora struttura gerarchica
        """
        # Step 1: Cerca definizione
        # DefinitionLookupTool fa 3 strategie di ricerca (DEFINISCE, concept, text)
        # Non include_related=True, quindi strategy 4 non eseguita
        mock_graph_db.query.side_effect = [
            # definition_lookup: strategy 1 - DEFINISCE relation
            [
                {
                    "term": "contratto",
                    "source_urn": "urn:norma:cc:art1321",
                    "source_type": "Norma",
                    "source_estremi": "Art. 1321 c.c.",
                    "definition_text": "Il contratto è l'accordo...",
                    "context": "Titolo II"
                }
            ],
            # definition_lookup: strategy 2 - concept nodes
            [],
            # definition_lookup: strategy 3 - text search
            []
        ]

        definition_tool = DefinitionLookupTool(graph_db=mock_graph_db)
        def_result = await definition_tool(term="contratto")

        assert def_result.success

        # Step 2: Reset mock per HierarchyNavigationTool
        source_urn = def_result.data["definitions"][0]["source_urn"]

        mock_graph_db.query.side_effect = [
            # hierarchy: find_start_node
            [
                {"urn": "urn:norma:cc:art1321", "tipo": "Articolo", "estremi": "Art. 1321 c.c."}
            ],
            # hierarchy: ancestors
            [
                {"urn": "urn:norma:cc:titolo2", "tipo": "Titolo", "estremi": "Titolo II", "depth": 1}
            ]
        ]

        hierarchy_tool = HierarchyNavigationTool(graph_db=mock_graph_db)
        hier_result = await hierarchy_tool(
            start_node=source_urn,
            direction="ancestors"
        )

        assert hier_result.success
        assert len(hier_result.data["hierarchy"]) > 0


# ===================================
# ExternalSourceTool Tests
# ===================================


class TestExternalSourceTool:
    """Test ExternalSourceTool - cascata graph→normattiva→brocardi."""

    @pytest.mark.asyncio
    async def test_external_source_finds_in_graph(self, mock_graph_db):
        """Verifica che trovi prima nel grafo locale."""
        mock_graph_db.query.return_value = [
            {
                "text": "Art. 1453 c.c. - La risoluzione del contratto...",
                "urn": "urn:norma:cc:art1453",
                "estremi": "Art. 1453 c.c.",
                "numero": "1453"
            }
        ]

        tool = ExternalSourceTool(graph_db=mock_graph_db)

        result = await tool(query="art. 1453 c.c.")

        assert result.success
        assert result.data["source"] == "graph"
        assert not result.data["fallback_used"]

    @pytest.mark.asyncio
    async def test_external_source_fallback_to_normattiva(self, mock_graph_db, mock_scraper):
        """Verifica fallback a Normattiva quando grafo vuoto."""
        # Grafo non trova nulla
        mock_graph_db.query.return_value = []

        tool = ExternalSourceTool(
            graph_db=mock_graph_db,
            normattiva_scraper=mock_scraper
        )

        result = await tool(query="art. 1453 c.c.")

        assert result.success
        assert result.data["source"] == "normattiva"
        assert result.data["fallback_used"]

    @pytest.mark.asyncio
    async def test_external_source_require_official(self, mock_graph_db, mock_scraper):
        """Verifica che require_official escluda Brocardi."""
        mock_graph_db.query.return_value = []

        tool = ExternalSourceTool(
            graph_db=mock_graph_db,
            normattiva_scraper=mock_scraper
        )

        result = await tool(
            query="art. 52 c.p.",
            require_official=True
        )

        assert result.success
        # Non deve usare brocardi
        assert result.data["source"] in ["graph", "normattiva"]

    @pytest.mark.asyncio
    async def test_external_source_no_sources_available(self):
        """Verifica gestione nessuna fonte disponibile."""
        # ExternalSourceTool con graph vuoto e scraper mock che fallisce
        mock_graph = AsyncMock()
        mock_graph.query.return_value = []  # Grafo non trova nulla

        # Mock scraper che non trova nulla
        mock_normattiva = AsyncMock()
        mock_normattiva.get_document.side_effect = Exception("Not found")

        mock_brocardi = AsyncMock()
        mock_brocardi.search.side_effect = Exception("Not found")

        tool = ExternalSourceTool(
            graph_db=mock_graph,
            normattiva_scraper=mock_normattiva,
            brocardi_scraper=mock_brocardi
        )

        result = await tool(query="atto inesistente xyz123")

        assert not result.success
        assert "non trovata" in result.error.lower()


# ===================================
# TextualReferenceTool Tests
# ===================================


class TestTextualReferenceTool:
    """Test TextualReferenceTool - rinvii normativi."""

    @pytest.mark.asyncio
    async def test_textual_reference_finds_references(self, mock_graph_db):
        """Verifica che trovi rinvii normativi."""
        mock_graph_db.query.return_value = [
            {
                "from_urn": "urn:norma:cc:art1453",
                "to_urn": "urn:norma:cc:art1455",
                "to_estremi": "Art. 1455 c.c.",
                "excerpt": "Importanza dell'inadempimento...",
                "depth": 1,
                "relation_types": ["RINVIA"]
            }
        ]

        tool = TextualReferenceTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art1453", max_depth=2)

        assert result.success
        assert len(result.data["references"]) > 0
        assert result.data["references"][0]["reference_type"] == "RINVIA"

    @pytest.mark.asyncio
    async def test_textual_reference_detects_circular(self, mock_graph_db):
        """Verifica rilevamento riferimenti circolari."""
        # Simula ciclo: A → B → A
        mock_graph_db.query.return_value = [
            {
                "from_urn": "urn:norma:cc:art1453",
                "to_urn": "urn:norma:cc:art1455",
                "to_estremi": "Art. 1455 c.c.",
                "depth": 1,
                "relation_types": ["RINVIA"]
            },
            {
                "from_urn": "urn:norma:cc:art1455",
                "to_urn": "urn:norma:cc:art1453",  # Torna a origine
                "to_estremi": "Art. 1453 c.c.",
                "depth": 2,
                "relation_types": ["richiama"]
            }
        ]

        tool = TextualReferenceTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art1453", max_depth=3)

        assert result.success
        assert result.data["circular_detected"]

    @pytest.mark.asyncio
    async def test_textual_reference_filters_types(self, mock_graph_db):
        """Verifica filtro per tipi di relazione."""
        tool = TextualReferenceTool(graph_db=mock_graph_db)

        await tool(
            article_urn="urn:norma:cc:art1453",
            reference_types=["RINVIA"]  # Solo RINVIA
        )

        # Verifica che query Cypher contenga solo RINVIA
        cypher = mock_graph_db.query.call_args[0][0]
        assert "RINVIA" in cypher

    @pytest.mark.asyncio
    async def test_textual_reference_no_graph_db(self):
        """Verifica gestione mancanza FalkorDB."""
        tool = TextualReferenceTool(graph_db=None)

        result = await tool(article_urn="urn:norma:cc:art1453")

        assert not result.success
        assert "FalkorDB client non configurato" in result.error


# ===================================
# HistoricalEvolutionTool Tests
# ===================================


class TestHistoricalEvolutionTool:
    """Test HistoricalEvolutionTool - storia della norma."""

    @pytest.mark.asyncio
    async def test_historical_evolution_finds_history(self, mock_graph_db):
        """Verifica che trovi storia della norma."""
        mock_graph_db.query.side_effect = [
            # _get_timeline query
            [
                {
                    "event_type": "modifica",
                    "by_urn": "urn:norma:legge:2019:123",
                    "by_estremi": "L. 123/2019",
                    "event_date": "2019-06-15",
                    "description": "Modifica rubrica"
                }
            ],
            # _get_current_status query
            [{"is_vigente": True, "is_abrogato": False, "is_sostituito": False}]
        ]

        tool = HistoricalEvolutionTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art1453")

        assert result.success
        assert len(result.data["timeline"]) > 0
        assert result.data["timeline"][0]["event"] == "modifica"

    @pytest.mark.asyncio
    async def test_historical_evolution_detects_abrogation(self, mock_graph_db):
        """Verifica rilevamento abrogazione."""
        mock_graph_db.query.side_effect = [
            # _get_timeline query: include abroga
            [
                {
                    "event_type": "abroga",
                    "by_urn": "urn:norma:legge:2020:45",
                    "by_estremi": "L. 45/2020",
                    "event_date": "2020-01-01",
                    "description": "Abrogazione"
                }
            ],
            # _get_current_status query: is_abrogato = True
            [{"is_vigente": False, "is_abrogato": True, "is_sostituito": False}]
        ]

        tool = HistoricalEvolutionTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art2000")

        assert result.success
        assert result.data["current_status"] == "abrogato"

    @pytest.mark.asyncio
    async def test_historical_evolution_orders_by_date(self, mock_graph_db):
        """Verifica ordinamento timeline per data."""
        mock_graph_db.query.side_effect = [
            # _get_timeline - already ordered by query ORDER BY event_date ASC
            [
                {"event_type": "modifica", "by_urn": "b", "by_estremi": "B", "event_date": "2015-01-01"},
                {"event_type": "modifica", "by_urn": "a", "by_estremi": "A", "event_date": "2020-01-01"},
                {"event_type": "modifica", "by_urn": "c", "by_estremi": "C", "event_date": "2022-01-01"}
            ],
            # _get_current_status
            [{"is_vigente": True, "is_abrogato": False, "is_sostituito": False}]
        ]

        tool = HistoricalEvolutionTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art1453")

        assert result.success
        dates = [e["date"] for e in result.data["timeline"]]
        # Timeline ordinata ascendente (dalla più vecchia alla più recente)
        assert dates == sorted(dates)

    @pytest.mark.asyncio
    async def test_historical_evolution_article_not_found(self, mock_graph_db):
        """Verifica gestione articolo non trovato (restituisce timeline vuota)."""
        # HistoricalEvolutionTool non fallisce se articolo non trovato,
        # restituisce timeline vuota con status "unknown"
        mock_graph_db.query.side_effect = [
            [],  # _get_timeline: nessun evento
            []   # _get_current_status: nessun risultato
        ]

        tool = HistoricalEvolutionTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:fake:art9999")

        assert result.success  # Tool non fallisce
        assert result.data["current_status"] == "unknown"
        assert result.data["total_events"] == 0


# ===================================
# PrincipleLookupTool Tests
# ===================================


class TestPrincipleLookupTool:
    """Test PrincipleLookupTool - principi giuridici."""

    @pytest.mark.asyncio
    async def test_principle_lookup_finds_principles(self, mock_graph_db):
        """Verifica che trovi principi giuridici."""
        # PrincipleLookupTool usa 3 strategie, quindi side_effect per multiple chiamate
        mock_graph_db.query.side_effect = [
            # Strategy 1: ESPRIME_PRINCIPIO relations
            [
                {
                    "nome": "pacta sunt servanda",
                    "description": "I contratti hanno forza di legge tra le parti",
                    "level": "generale",
                    "fondamento": "Art. 1372 c.c.",
                    "norme_attuative": ["Art. 1372 c.c."],
                    "norme_urns": ["urn:norma:cc:art1372"]
                }
            ],
            # Strategy 2: PrincipioGiuridico nodes
            [],
            # Strategy 3: text search
            []
        ]

        tool = PrincipleLookupTool(graph_db=mock_graph_db)

        result = await tool(query="pacta sunt servanda")

        assert result.success
        assert len(result.data["principles"]) > 0
        assert result.data["principles"][0]["nome"] == "pacta sunt servanda"

    @pytest.mark.asyncio
    async def test_principle_lookup_filters_by_level(self, mock_graph_db):
        """Verifica filtro per livello (costituzionale, generale, europeo)."""
        # PrincipleLookupTool usa principle_level, non level
        mock_graph_db.query.side_effect = [
            [
                {
                    "nome": "uguaglianza",
                    "description": "Tutti i cittadini sono uguali",
                    "level": "costituzionale",
                    "fondamento": "Art. 3 Cost.",
                    "norme_attuative": ["Art. 3 Cost."],
                    "norme_urns": ["urn:norma:cost:art3"]
                }
            ],
            [],
            []
        ]

        tool = PrincipleLookupTool(graph_db=mock_graph_db)

        result = await tool(
            query="uguaglianza",
            principle_level=["costituzionale"]  # Parametro corretto
        )

        assert result.success
        # Verifica che query sia stata chiamata
        assert mock_graph_db.query.called

    @pytest.mark.asyncio
    async def test_principle_lookup_includes_sources(self, mock_graph_db):
        """Verifica che includa fonti del principio."""
        mock_graph_db.query.side_effect = [
            [
                {
                    "nome": "buona fede",
                    "description": "Obbligo di comportarsi secondo buona fede",
                    "level": "generale",
                    "fondamento": "Art. 1175 c.c.",
                    "norme_attuative": ["Art. 1175 c.c.", "Art. 1375 c.c."],
                    "norme_urns": ["urn:norma:cc:art1175", "urn:norma:cc:art1375"]
                }
            ],
            [],
            []
        ]

        tool = PrincipleLookupTool(graph_db=mock_graph_db)

        # PrincipleLookupTool non ha parametro include_sources, include sempre
        result = await tool(query="buona fede")

        assert result.success
        assert len(result.data["principles"][0]["norme_attuative"]) >= 2

    @pytest.mark.asyncio
    async def test_principle_lookup_empty_results(self, mock_graph_db):
        """Verifica gestione nessun principio trovato."""
        mock_graph_db.query.side_effect = [[], [], []]  # Tutte le strategie vuote

        tool = PrincipleLookupTool(graph_db=mock_graph_db)

        result = await tool(query="principio_inesistente")

        assert result.success
        assert result.data["total"] == 0


# ===================================
# ConstitutionalBasisTool Tests
# ===================================


class TestConstitutionalBasisTool:
    """Test ConstitutionalBasisTool - gerarchia Kelseniana."""

    @pytest.mark.asyncio
    async def test_constitutional_basis_finds_basis(self, mock_graph_db):
        """Verifica che trovi base costituzionale."""
        mock_graph_db.query.side_effect = [
            # find_article
            [{"urn": "urn:norma:cc:art2043", "tipo": "Articolo", "estremi": "Art. 2043 c.c."}],
            # find_constitutional_basis
            [
                {
                    "norm_urn": "urn:norma:cost:art3",
                    "norm_estremi": "Art. 3 Cost.",
                    "principle": "Principio di uguaglianza",
                    "relation_path": ["urn:norma:cc:art2043", "urn:norma:cost:art3"],
                    "distance": 1,
                    "strength": "diretta"
                }
            ],
            # find_eu_basis
            []
        ]

        tool = ConstitutionalBasisTool(graph_db=mock_graph_db)

        result = await tool(article_urn="urn:norma:cc:art2043")

        assert result.success
        assert len(result.data["constitutional_basis"]) > 0
        assert "Cost." in result.data["constitutional_basis"][0]["norm_estremi"]

    @pytest.mark.asyncio
    async def test_constitutional_basis_includes_eu_law(self, mock_graph_db):
        """Verifica inclusione diritto UE."""
        mock_graph_db.query.side_effect = [
            # find_article
            [{"urn": "urn:norma:gdpr:art5"}],
            # find_constitutional_basis
            [],
            # find_eu_basis
            [
                {
                    "norm_urn": "urn:norma:tfue:art16",
                    "norm_estremi": "Art. 16 TFUE",
                    "principle": "Protezione dati personali",
                    "distance": 1
                }
            ]
        ]

        tool = ConstitutionalBasisTool(graph_db=mock_graph_db)

        result = await tool(
            article_urn="urn:norma:gdpr:art5",
            include_eu_law=True
        )

        assert result.success
        assert result.data["total_eu"] > 0

    @pytest.mark.asyncio
    async def test_constitutional_basis_respects_max_depth(self, mock_graph_db):
        """Verifica che max_depth sia rispettato."""
        mock_graph_db.query.side_effect = [
            [{"urn": "urn:norma:cc:art1453"}],
            [],
            []
        ]

        tool = ConstitutionalBasisTool(graph_db=mock_graph_db, max_depth=2)

        await tool(article_urn="urn:norma:cc:art1453", max_depth=2)

        # Verifica che query Cypher contenga *1..2
        for call in mock_graph_db.query.call_args_list:
            cypher = call[0][0]
            if "ATTUA" in cypher:
                assert "*1..2" in cypher

    @pytest.mark.asyncio
    async def test_constitutional_basis_no_graph_db(self):
        """Verifica gestione mancanza FalkorDB."""
        tool = ConstitutionalBasisTool(graph_db=None)

        result = await tool(article_urn="urn:norma:cc:art1453")

        assert not result.success


# ===================================
# CitationChainTool Tests
# ===================================


class TestCitationChainTool:
    """Test CitationChainTool - catena citazioni giurisprudenziali."""

    @pytest.mark.asyncio
    async def test_citation_chain_finds_citations(self, mock_graph_db):
        """Verifica che trovi catena citazioni."""
        mock_graph_db.query.side_effect = [
            # find_case_node
            [{"urn": "urn:giurisp:cass:2024:12345", "estremi": "Cass. 12345/2024"}],
            # get_citing_chain
            [
                {
                    "from_case": "urn:giurisp:cass:2024:12345",
                    "to_case": "urn:giurisp:cass:2020:5678",
                    "to_estremi": "Cass. 5678/2020",
                    "relation": "cita",
                    "depth": 1
                }
            ],
            # get_cited_by_chain
            [
                {
                    "from_case": "urn:giurisp:cass:2025:999",
                    "to_case": "urn:giurisp:cass:2024:12345",
                    "from_estremi": "Cass. 999/2025",
                    "relation": "conferma",
                    "depth": 1
                }
            ],
            # detect_overruling
            [],
            # find_leading_cases
            []
        ]

        tool = CitationChainTool(graph_db=mock_graph_db)

        result = await tool(
            case_urn="urn:giurisp:cass:2024:12345",
            direction="both"
        )

        assert result.success
        assert result.data["total_citations"] >= 2

    @pytest.mark.asyncio
    async def test_citation_chain_detects_overruling(self, mock_graph_db):
        """Verifica rilevamento overruling."""
        mock_graph_db.query.side_effect = [
            # find_case_node
            [{"urn": "urn:giurisp:cass:2020:1111"}],
            # get_citing_chain
            [],
            # get_cited_by_chain
            [],
            # detect_overruling
            [
                {
                    "old_case": "urn:giurisp:cass:2020:1111",
                    "old_estremi": "Cass. 1111/2020",
                    "new_case": "urn:giurisp:cass:2024:2222",
                    "new_estremi": "Cass. 2222/2024",
                    "overruling_date": "2024-03-15"
                }
            ],
            # find_leading_cases
            []
        ]

        tool = CitationChainTool(graph_db=mock_graph_db)

        result = await tool(
            case_urn="urn:giurisp:cass:2020:1111",
            detect_overruling=True
        )

        assert result.success
        assert len(result.data["overruling_events"]) > 0

    @pytest.mark.asyncio
    async def test_citation_chain_finds_leading_cases(self, mock_graph_db):
        """Verifica identificazione leading cases."""
        mock_graph_db.query.side_effect = [
            [{"urn": "urn:giurisp:cass:2024:12345"}],
            [],
            [],
            [],
            # find_leading_cases
            [
                {
                    "urn": "urn:giurisp:cass:ss:uu:2022:100",
                    "estremi": "Cass. SS.UU. 100/2022",
                    "citation_count": 150
                }
            ]
        ]

        tool = CitationChainTool(graph_db=mock_graph_db)

        result = await tool(
            case_urn="urn:giurisp:cass:2024:12345",
            include_leading_cases=True
        )

        assert result.success
        assert len(result.data["leading_cases"]) > 0
        assert result.data["leading_cases"][0]["citation_count"] == 150

    @pytest.mark.asyncio
    async def test_citation_chain_case_not_found(self, mock_graph_db):
        """Verifica gestione caso non trovato."""
        mock_graph_db.query.return_value = []

        tool = CitationChainTool(graph_db=mock_graph_db)

        result = await tool(case_urn="urn:giurisp:fake:9999")

        assert not result.success
        assert "non trovato" in result.error.lower()
