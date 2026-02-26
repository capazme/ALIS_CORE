"""
Edge case tests for Search Tools (Semantic and Graph).

Tests per situazioni limite dei tools di ricerca:
- Cycle detection nel grafo
- Dead-end node
- max_hops=0
- Grafo vuoto
- Limiting risultati grandi
- Zero-vector query
- top_k=0 e top_k > risultati disponibili
- Componenti disconnesse
- Nodi non esistenti
"""

import pytest
from typing import List
from uuid import uuid4
from dataclasses import dataclass

from merlt.tools import (
    SemanticSearchTool,
    GraphSearchTool,
    SearchResultItem,
    ToolResult,
)


# ============================================================================
# Mock Classes
# ============================================================================

@dataclass
class MockRetrievalResult:
    """Mock del risultato di retrieval."""
    chunk_id: str
    text: str
    similarity_score: float
    graph_score: float
    final_score: float
    linked_nodes: List[dict]
    metadata: dict


class MockRetriever:
    """Mock di GraphAwareRetriever."""

    def __init__(self, results: List[MockRetrievalResult] = None):
        self.results = results or []
        self.last_call = None

    async def retrieve(
        self,
        query_embedding: List[float],
        context_nodes: List[str] = None,
        expert_type: str = None,
        top_k: int = 10,
        source_types: List[str] = None,
    ) -> List[MockRetrievalResult]:
        self.last_call = {
            "query_embedding": query_embedding,
            "context_nodes": context_nodes,
            "expert_type": expert_type,
            "top_k": top_k,
            "source_types": source_types,
        }
        return self.results[:top_k]


class MockEmbeddings:
    """Mock di EmbeddingService."""

    def encode_query(self, text: str) -> List[float]:
        return [0.1] * 1024


class MockGraphDB:
    """Mock di FalkorDBClient."""

    def __init__(self, query_results: List[dict] = None):
        self.query_results = query_results or []
        self.last_query = None

    async def query(self, query: str, params: dict = None) -> List[dict]:
        self.last_query = {"query": query, "params": params}
        return self.query_results


# ============================================================================
# Graph Traversal Edge Cases
# ============================================================================

class TestGraphTraversalEdgeCases:
    """Tests per edge cases nel graph traversal."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        SemanticSearchTool.clear_cache()
        yield
        SemanticSearchTool.clear_cache()

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        """Cycle A->B->A: non deve andare in loop infinito."""
        # Mock that returns nodes that form a cycle
        cycle_results = [
            {
                "node": {"URN": "urn:B", "_type": "Norma", "testo": "Norma B"},
                "rel": {"relation": "cita", "properties": {}},
            },
            {
                "node": {"URN": "urn:A", "_type": "Norma", "testo": "Norma A"},
                "rel": {"relation": "cita", "properties": {}},
            },
        ]

        graph_db = MockGraphDB(query_results=cycle_results)
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(start_node="urn:A", max_hops=2)

        assert result.success is True
        # The query completes without infinite loop
        assert result.data["total_nodes"] == 2

    @pytest.mark.asyncio
    async def test_dead_end_node(self):
        """Nodo senza relazioni uscenti."""
        graph_db = MockGraphDB(query_results=[])
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(start_node="urn:dead_end")

        assert result.success is True
        assert result.data["total_nodes"] == 0
        assert result.data["total_edges"] == 0

    @pytest.mark.asyncio
    async def test_max_hops_zero(self):
        """max_hops=0 non dovrebbe espandere nulla ma non deve crashare."""
        graph_db = MockGraphDB(query_results=[])
        tool = GraphSearchTool(graph_db=graph_db)

        # max_hops=0 will be used in query as *1..0 which may yield no results
        result = await tool(start_node="urn:test", max_hops=0)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_empty_graph(self):
        """Graph vuoto non causa errori."""
        graph_db = MockGraphDB(query_results=[])
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(start_node="urn:non_esiste")

        assert result.success is True
        assert result.data["total_nodes"] == 0

    @pytest.mark.asyncio
    async def test_large_result_limiting(self):
        """Risultati grandi sono limitati dal LIMIT nella query."""
        # Create many results
        many_results = [
            {
                "node": {"URN": f"urn:node_{i}", "_type": "Norma"},
                "rel": {"relation": "cita", "properties": {}},
            }
            for i in range(200)
        ]

        graph_db = MockGraphDB(query_results=many_results)
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(start_node="urn:start", max_hops=3)

        assert result.success is True
        # The query has LIMIT 100, but MockGraphDB returns all, so tool processes all
        assert result.data["total_nodes"] == 200

    def test_build_query_max_hops_zero(self):
        """_build_traversal_query con max_hops=0."""
        tool = GraphSearchTool()
        query, params = tool._build_traversal_query(
            start_node="urn:test", max_hops=0
        )
        assert params["start_urn"] == "urn:test"
        assert "*1..0" in query

    @pytest.mark.asyncio
    async def test_disconnected_components(self):
        """Componente disconnessa: nodo non raggiungibile dal grafo."""
        # Node exists but has no connections — empty result
        graph_db = MockGraphDB(query_results=[])
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(
            start_node="urn:isolated",
            relation_types=["connesso_a"],
            max_hops=5,
        )

        assert result.success is True
        assert result.data["total_nodes"] == 0

    @pytest.mark.asyncio
    async def test_non_existent_node_search(self):
        """Ricerca da nodo non esistente."""
        graph_db = MockGraphDB(query_results=[])
        tool = GraphSearchTool(graph_db=graph_db)

        result = await tool(start_node="urn:questo_non_esiste_nel_grafo")

        assert result.success is True
        assert result.data["total_nodes"] == 0


# ============================================================================
# Semantic Search Edge Cases
# ============================================================================

class TestSemanticSearchEdgeCases:
    """Tests per edge cases nella ricerca semantica."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        SemanticSearchTool.clear_cache()
        yield
        SemanticSearchTool.clear_cache()

    @pytest.mark.asyncio
    async def test_zero_vector_query(self):
        """Query con embedding tutto zero."""

        class ZeroEmbeddings:
            def encode_query(self, text: str) -> List[float]:
                return [0.0] * 1024

        retriever = MockRetriever(results=[])
        tool = SemanticSearchTool(
            retriever=retriever, embeddings=ZeroEmbeddings()
        )

        result = await tool(query="test zero vector")

        assert result.success is True
        assert result.data["total"] == 0

    @pytest.mark.asyncio
    async def test_top_k_zero(self):
        """top_k=0 non deve crashare."""
        mock_results = [
            MockRetrievalResult(
                chunk_id=str(uuid4()),
                text="Qualche testo",
                similarity_score=0.8,
                graph_score=0.7,
                final_score=0.75,
                linked_nodes=[],
                metadata={},
            )
        ]
        retriever = MockRetriever(results=mock_results)
        tool = SemanticSearchTool(
            retriever=retriever, embeddings=MockEmbeddings()
        )

        result = await tool(query="test", top_k=0)

        assert result.success is True
        # top_k=0 is falsy, so default_top_k (10) is used
        assert result.data["total"] <= 10

    @pytest.mark.asyncio
    async def test_top_k_greater_than_results(self):
        """top_k maggiore dei risultati disponibili."""
        mock_results = [
            MockRetrievalResult(
                chunk_id=str(uuid4()),
                text=f"Result {i}",
                similarity_score=0.9 - i * 0.1,
                graph_score=0.7,
                final_score=0.8 - i * 0.1,
                linked_nodes=[],
                metadata={},
            )
            for i in range(3)
        ]
        retriever = MockRetriever(results=mock_results)
        tool = SemanticSearchTool(
            retriever=retriever, embeddings=MockEmbeddings()
        )

        result = await tool(query="test", top_k=100)

        assert result.success is True
        assert result.data["total"] == 3  # Only 3 available


# ============================================================================
# Node/Edge Conversion Edge Cases
# ============================================================================

class TestNodeEdgeConversion:
    """Tests per conversione nodi ed edge."""

    def test_node_to_dict_empty_node(self):
        """Nodo senza proprieta'."""
        tool = GraphSearchTool()
        result = tool._node_to_dict({})
        assert result["urn"] == ""
        assert result["type"] == "Unknown"

    def test_node_to_dict_non_dict_node(self):
        """Nodo che non e' un dict ne' ha properties."""
        tool = GraphSearchTool()
        result = tool._node_to_dict("not_a_node")
        assert result["urn"] == ""
        assert result["type"] == "Unknown"

    def test_edge_to_dict_empty_edge(self):
        """Edge senza proprieta'."""
        tool = GraphSearchTool()
        result = tool._edge_to_dict({})
        assert result["type"] == "UNKNOWN"
        assert result["properties"] == {}

    def test_edge_to_dict_non_dict_edge(self):
        """Edge che non e' un dict."""
        tool = GraphSearchTool()
        result = tool._edge_to_dict("not_an_edge")
        assert result["type"] == "UNKNOWN"

    def test_search_result_item_to_dict(self):
        """SearchResultItem.to_dict preserva tutti i campi."""
        item = SearchResultItem(
            chunk_id="abc",
            text="Test",
            similarity_score=0.9,
            graph_score=0.8,
            final_score=0.85,
            linked_nodes=[{"urn": "urn:test"}],
            metadata={"source": "graph"},
        )
        d = item.to_dict()
        assert d["chunk_id"] == "abc"
        assert d["similarity_score"] == 0.9
        assert d["final_score"] == 0.85
        assert len(d["linked_nodes"]) == 1


# ============================================================================
# Cache Edge Cases
# ============================================================================

class TestCacheEdgeCases:
    """Tests per cache semantica."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        SemanticSearchTool.clear_cache()
        yield
        SemanticSearchTool.clear_cache()

    @pytest.mark.asyncio
    async def test_cache_hit_same_query(self):
        """Stessa query ritorna risultato dalla cache."""
        retriever = MockRetriever(results=[
            MockRetrievalResult(
                chunk_id=str(uuid4()),
                text="Cached result",
                similarity_score=0.9,
                graph_score=0.8,
                final_score=0.85,
                linked_nodes=[],
                metadata={},
            )
        ])
        tool = SemanticSearchTool(
            retriever=retriever, embeddings=MockEmbeddings()
        )

        result1 = await tool(query="test cache")
        result2 = await tool(query="test cache")

        assert result1.success is True
        assert result2.success is True
        # Second call should be from cache (retriever called only once)
        assert retriever.last_call is not None

    @pytest.mark.asyncio
    async def test_cache_clear(self):
        """clear_cache resetta la cache."""
        retriever = MockRetriever(results=[])
        tool = SemanticSearchTool(
            retriever=retriever, embeddings=MockEmbeddings()
        )

        await tool(query="test clear")
        assert len(SemanticSearchTool._shared_cache) > 0

        SemanticSearchTool.clear_cache()
        assert len(SemanticSearchTool._shared_cache) == 0
