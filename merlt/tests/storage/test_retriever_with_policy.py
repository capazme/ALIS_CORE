"""
Test GraphAwareRetriever con PolicyManager
==========================================

Test integrazione retriever con pesi neurali per RLCF.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, AsyncMock, patch
from uuid import UUID
from typing import Dict, Any, List

from merlt.storage.retriever.retriever import GraphAwareRetriever
from merlt.storage.retriever.models import (
    RetrieverConfig,
    GraphPath,
    EXPERT_TRAVERSAL_WEIGHTS
)
from merlt.rlcf.policy_manager import PolicyManager, PolicyConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_embedding():
    """Embedding 768-dim normalizzato."""
    np.random.seed(42)
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / np.linalg.norm(emb)
    return emb.tolist()


@pytest.fixture
def mock_vector_db():
    """Mock Qdrant client."""
    db = MagicMock()

    # Mock response per query_points
    mock_point = MagicMock()
    mock_point.id = UUID("12345678-1234-5678-1234-567812345678")
    mock_point.score = 0.85
    mock_point.payload = {
        "text": "Art. 52 c.p. - Legittima difesa",
        "article_urn": "urn:nir:stato:codice.penale:art.52",
        "source_type": "norma"
    }

    mock_response = MagicMock()
    mock_response.points = [mock_point]
    db.query_points = MagicMock(return_value=mock_response)

    return db


@pytest.fixture
def mock_graph_db():
    """Mock FalkorDB client."""
    db = MagicMock()

    # Mock shortest_path
    async def mock_shortest_path(start_node, end_node, max_hops):
        return {
            "path": {
                "edges": [
                    {"type": "RIFERIMENTO", "source": start_node, "target": "middle"},
                    {"type": "CITATO_DA", "source": "middle", "target": end_node}
                ],
                "length": 2
            }
        }

    db.shortest_path = AsyncMock(side_effect=mock_shortest_path)

    # Mock get_related_nodes_for_article
    async def mock_related(article_urn, max_results):
        return [
            {"node_urn": "urn:concetto:legittima_difesa", "node_label": "Concetto"}
        ]

    db.get_related_nodes_for_article = AsyncMock(side_effect=mock_related)

    return db


@pytest.fixture
def mock_bridge_table():
    """Mock bridge table."""
    bridge = MagicMock()

    async def mock_get_nodes(chunk_id):
        return [{"graph_node_urn": "urn:test:node"}]

    bridge.get_nodes_for_chunk = AsyncMock(side_effect=mock_get_nodes)
    return bridge


@pytest.fixture
def mock_policy_manager():
    """Mock PolicyManager."""
    manager = MagicMock(spec=PolicyManager)

    # Mock compute_batch_weights
    async def mock_batch(query_embedding, relation_types, expert_type, trace=None):
        return {rel: (0.75, -0.288) for rel in relation_types}

    manager.compute_batch_weights = AsyncMock(side_effect=mock_batch)

    return manager


@pytest.fixture
def mock_trace():
    """Mock ExecutionTrace."""
    trace = MagicMock()
    trace.add_graph_traversal = MagicMock()
    return trace


@pytest.fixture
def retriever_config():
    """RetrieverConfig per test."""
    return RetrieverConfig(
        alpha=0.7,
        enable_graph_enrichment=True,
        max_graph_hops=3,
        default_graph_score=0.3,
        over_retrieve_factor=2
    )


# ============================================================================
# Test Inizializzazione con PolicyManager
# ============================================================================

class TestRetrieverInitWithPolicy:
    """Test inizializzazione retriever con PolicyManager."""

    def test_init_with_policy_manager(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager
    ):
        """Test che retriever accetta policy_manager."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        assert retriever.policy_manager is mock_policy_manager

    def test_init_without_policy_manager(
        self, mock_vector_db, mock_graph_db, mock_bridge_table, retriever_config
    ):
        """Test che retriever funziona senza policy_manager."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config
        )

        assert retriever.policy_manager is None


# ============================================================================
# Test _score_path con PolicyManager
# ============================================================================

class TestScorePathWithPolicy:
    """Test _score_path con pesi neurali."""

    @pytest.mark.asyncio
    async def test_score_path_uses_policy_manager(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager, sample_embedding, mock_trace
    ):
        """Test che _score_path usa PolicyManager per pesi."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        path = GraphPath(
            source_node="urn:start",
            target_node="urn:end",
            edges=[
                {"type": "RIFERIMENTO", "source": "urn:start", "target": "urn:middle"},
                {"type": "CITATO_DA", "source": "urn:middle", "target": "urn:end"}
            ],
            length=2
        )

        score = await retriever._score_path(
            path=path,
            expert_type="literal",
            query_embedding=sample_embedding,
            trace=mock_trace
        )

        # PolicyManager dovrebbe essere chiamato
        mock_policy_manager.compute_batch_weights.assert_called_once()

        # Verifica parametri
        call_kwargs = mock_policy_manager.compute_batch_weights.call_args[1]
        assert call_kwargs["query_embedding"] == sample_embedding
        assert "RIFERIMENTO" in call_kwargs["relation_types"]
        assert "CITATO_DA" in call_kwargs["relation_types"]
        assert call_kwargs["expert_type"] == "literal"
        assert call_kwargs["trace"] is mock_trace

        # Score dovrebbe essere calcolato con pesi neurali (0.75 per edge)
        # distance_score = 1/(2+1) = 0.333
        # relation_bonus = 0.75 * 0.75 = 0.5625
        # final = 0.333 * 0.5625 ≈ 0.1875
        assert score == pytest.approx(0.1875, abs=0.01)

    @pytest.mark.asyncio
    async def test_score_path_fallback_without_embedding(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager
    ):
        """Test fallback a pesi statici senza embedding."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        path = GraphPath(
            source_node="urn:start",
            target_node="urn:end",
            edges=[{"type": "RIFERIMENTO"}],
            length=1
        )

        score = await retriever._score_path(
            path=path,
            expert_type="LiteralExpert",
            query_embedding=None  # No embedding
        )

        # PolicyManager non dovrebbe essere chiamato
        mock_policy_manager.compute_batch_weights.assert_not_called()

        # Usa peso statico per RIFERIMENTO in LiteralExpert
        expected_weight = EXPERT_TRAVERSAL_WEIGHTS.get("LiteralExpert", {}).get("RIFERIMENTO", 0.5)
        # distance_score = 1/(1+1) = 0.5
        expected_score = 0.5 * expected_weight

        assert score > 0

    @pytest.mark.asyncio
    async def test_score_path_without_policy_manager(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, sample_embedding
    ):
        """Test _score_path senza PolicyManager usa pesi statici."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=None  # No PolicyManager
        )

        path = GraphPath(
            source_node="urn:start",
            target_node="urn:end",
            edges=[{"type": "RIFERIMENTO"}],
            length=1
        )

        score = await retriever._score_path(
            path=path,
            expert_type="LiteralExpert",
            query_embedding=sample_embedding
        )

        # Calcola expected con pesi statici
        expected_weight = EXPERT_TRAVERSAL_WEIGHTS.get("LiteralExpert", {}).get("RIFERIMENTO", 0.5)
        expected_score = 0.5 * expected_weight  # distance=0.5, relation=weight

        assert score > 0


# ============================================================================
# Test _compute_graph_score con PolicyManager
# ============================================================================

class TestComputeGraphScoreWithPolicy:
    """Test _compute_graph_score con PolicyManager."""

    @pytest.mark.asyncio
    async def test_graph_score_propagates_params(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager, sample_embedding, mock_trace
    ):
        """Test che _compute_graph_score propaga parametri a _score_path."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        score = await retriever._compute_graph_score(
            chunk_nodes=["urn:chunk:node"],
            context_nodes=["urn:context:node"],
            expert_type="systemic",
            query_embedding=sample_embedding,
            trace=mock_trace
        )

        # graph_db.shortest_path dovrebbe essere chiamato
        mock_graph_db.shortest_path.assert_called()

        # PolicyManager dovrebbe essere chiamato
        mock_policy_manager.compute_batch_weights.assert_called()

        # Score > 0
        assert score > 0

    @pytest.mark.asyncio
    async def test_graph_score_default_without_nodes(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager, sample_embedding
    ):
        """Test default score quando non ci sono nodi."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        score = await retriever._compute_graph_score(
            chunk_nodes=[],
            context_nodes=[],
            expert_type="literal",
            query_embedding=sample_embedding
        )

        assert score == retriever_config.default_graph_score


# ============================================================================
# Test retrieve() con trace
# ============================================================================

class TestRetrieveWithTrace:
    """Test retrieve() con ExecutionTrace."""

    @pytest.mark.asyncio
    async def test_retrieve_accepts_trace(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, mock_policy_manager, sample_embedding, mock_trace
    ):
        """Test che retrieve() accetta parametro trace."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=mock_policy_manager
        )

        results = await retriever.retrieve(
            query_embedding=sample_embedding,
            context_nodes=["urn:context:node"],
            expert_type="literal",
            trace=mock_trace
        )

        assert results is not None


# ============================================================================
# Test Error Handling
# ============================================================================

class TestErrorHandling:
    """Test gestione errori con PolicyManager."""

    @pytest.mark.asyncio
    async def test_policy_error_fallback_to_static(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        retriever_config, sample_embedding
    ):
        """Test fallback a pesi statici quando PolicyManager fallisce."""
        # Mock che fallisce
        failing_manager = MagicMock(spec=PolicyManager)
        failing_manager.compute_batch_weights = AsyncMock(
            side_effect=Exception("Test error")
        )

        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=failing_manager
        )

        path = GraphPath(
            source_node="urn:start",
            target_node="urn:end",
            edges=[{"type": "RIFERIMENTO"}],
            length=1
        )

        # Non dovrebbe sollevare eccezione
        score = await retriever._score_path(
            path=path,
            expert_type="LiteralExpert",
            query_embedding=sample_embedding
        )

        # Dovrebbe usare pesi statici (> 0)
        assert score > 0


# ============================================================================
# Test _compute_static_relation_bonus
# ============================================================================

class TestStaticRelationBonus:
    """Test _compute_static_relation_bonus."""

    def test_static_bonus_with_known_expert(
        self, mock_vector_db, mock_graph_db, mock_bridge_table, retriever_config
    ):
        """Test calcolo bonus con expert noto."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config
        )

        edges = [
            {"type": "RIFERIMENTO"},
            {"type": "CITATO_DA"}
        ]

        bonus = retriever._compute_static_relation_bonus(
            edges=edges,
            expert_type="LiteralExpert"
        )

        # Dovrebbe essere prodotto dei pesi statici
        if "LiteralExpert" in EXPERT_TRAVERSAL_WEIGHTS:
            weights = EXPERT_TRAVERSAL_WEIGHTS["LiteralExpert"]
            expected = weights.get("RIFERIMENTO", 0.5) * weights.get("CITATO_DA", 0.5)
            assert bonus == pytest.approx(expected, abs=0.01)
        else:
            # Se expert non in config, bonus = 1.0
            assert bonus == 1.0

    def test_static_bonus_unknown_expert(
        self, mock_vector_db, mock_graph_db, mock_bridge_table, retriever_config
    ):
        """Test calcolo bonus con expert sconosciuto."""
        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config
        )

        edges = [{"type": "QUALUNQUE"}]

        bonus = retriever._compute_static_relation_bonus(
            edges=edges,
            expert_type="ExpertInesistente"
        )

        # Expert sconosciuto → bonus = 1.0 (nessuna modifica)
        assert bonus == 1.0


# ============================================================================
# Test Integrazione Completa
# ============================================================================

class TestFullIntegration:
    """Test integrazione completa retriever + policy."""

    @pytest.mark.asyncio
    async def test_full_retrieve_with_policy(
        self, mock_vector_db, mock_graph_db, mock_bridge_table,
        sample_embedding, tmp_path
    ):
        """Test flusso completo: retrieve → graph score → policy weights."""
        # Crea PolicyManager reale (senza checkpoint, usa static)
        config = PolicyConfig(
            checkpoint_dir=tmp_path,
            enable_traversal_policy=False,  # Usa static per test
            device="cpu"
        )
        policy_manager = PolicyManager(config=config)

        retriever_config = RetrieverConfig(
            alpha=0.7,
            enable_graph_enrichment=True
        )

        retriever = GraphAwareRetriever(
            vector_db=mock_vector_db,
            graph_db=mock_graph_db,
            bridge_table=mock_bridge_table,
            config=retriever_config,
            policy_manager=policy_manager
        )

        results = await retriever.retrieve(
            query_embedding=sample_embedding,
            context_nodes=["urn:context:node"],
            expert_type="literal",
            top_k=5
        )

        # Verifica risultati
        assert results is not None
        assert len(results) > 0

        # Verifica scores
        for result in results:
            assert result.similarity_score >= 0
            assert result.graph_score >= 0
            assert result.final_score >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
