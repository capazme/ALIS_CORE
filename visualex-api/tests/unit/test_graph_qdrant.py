"""
Tests for Qdrant Collection Manager
===================================

Integration tests using real Qdrant instance.
Requires: docker container 'merl-t-qdrant-dev' running on localhost:6333

Tests cover:
- AC1: Collection schema with HNSW and payload indexes
- AC2: Filtered queries by source_type, authority, expert affinity
- AC3: Semantic search with full payload
"""

import pytest
import random
import uuid

from visualex.graph.qdrant import (
    QdrantConfig,
    QdrantCollectionManager,
    SearchResult,
    ExpertType,
    DEFAULT_EXPERT_AFFINITIES,
    HNSW_CONFIG,
)
from visualex.graph.chunking import SourceType


# =============================================================================
# Test Configuration
# =============================================================================


# Use a test-specific collection to avoid conflicts
TEST_COLLECTION = f"test_legal_chunks_{uuid.uuid4().hex[:8]}"
TEST_VECTOR_SIZE = 384  # Smaller for faster tests


@pytest.fixture(scope="module")
def qdrant_config():
    """Create test configuration."""
    return QdrantConfig(
        host="localhost",
        port=6333,
        collection_name=TEST_COLLECTION,
        vector_size=TEST_VECTOR_SIZE,
    )


@pytest.fixture(scope="module")
def manager(qdrant_config):
    """Create manager and clean up after tests."""
    mgr = QdrantCollectionManager(qdrant_config)

    # Create collection for tests
    mgr.create_collection(recreate=True)

    yield mgr

    # Cleanup: delete test collection
    try:
        mgr.delete_collection()
    except Exception:
        pass


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": f"chunk-norm-{uuid.uuid4().hex[:8]}",
            "text": "Art. 1453 c.c. - La risoluzione del contratto per inadempimento.",
            "source_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            "source_type": SourceType.NORM,
            "source_authority": 1.0,
            "article_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            "model_id": "test-model",
        },
        {
            "chunk_id": f"chunk-juris-{uuid.uuid4().hex[:8]}",
            "text": "La Cassazione ha stabilito che l'inadempimento deve essere grave.",
            "source_urn": "urn:cassazione:sent:12345:2024",
            "source_type": SourceType.JURISPRUDENCE,
            "source_authority": 0.8,
            "article_urn": "urn:cassazione:sent:12345:2024",
            "model_id": "test-model",
        },
        {
            "chunk_id": f"chunk-comm-{uuid.uuid4().hex[:8]}",
            "text": "Il commento al codice civile spiega la ratio della norma.",
            "source_urn": "urn:brocardi:art1453:spiegazione",
            "source_type": SourceType.COMMENTARY,
            "source_authority": 0.5,
            "article_urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            "model_id": "test-model",
        },
        {
            "chunk_id": f"chunk-doct-{uuid.uuid4().hex[:8]}",
            "text": "La dottrina elabora i principi generali del contratto.",
            "source_urn": "urn:doctrine:manuale:contratti:cap5",
            "source_type": SourceType.DOCTRINE,
            "source_authority": 0.4,
            "article_urn": "urn:doctrine:manuale:contratti",
            "model_id": "test-model",
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings matching TEST_VECTOR_SIZE."""
    # Seed for reproducibility
    random.seed(42)

    # Generate 4 normalized embeddings
    embeddings = []
    for i in range(4):
        # Create slightly different embeddings
        vec = [random.gauss(0.1 * i, 0.1) for _ in range(TEST_VECTOR_SIZE)]
        # Normalize
        norm = sum(x*x for x in vec) ** 0.5
        vec = [x / norm for x in vec]
        embeddings.append(vec)

    return embeddings


# =============================================================================
# QdrantConfig Tests
# =============================================================================


class TestQdrantConfig:
    """Test suite for QdrantConfig."""

    def test_default_values(self):
        """Test default configuration."""
        config = QdrantConfig()

        assert config.host == "localhost"
        assert config.port == 6333
        assert config.collection_name == "legal_chunks"
        assert config.vector_size == 1024
        assert config.distance == "Cosine"

    def test_from_env(self, monkeypatch):
        """Test configuration from environment."""
        monkeypatch.setenv("QDRANT_HOST", "qdrant-server")
        monkeypatch.setenv("QDRANT_PORT", "6334")
        monkeypatch.setenv("QDRANT_COLLECTION", "my_chunks")
        monkeypatch.setenv("QDRANT_VECTOR_SIZE", "768")

        config = QdrantConfig.from_env()

        assert config.host == "qdrant-server"
        assert config.port == 6334
        assert config.collection_name == "my_chunks"
        assert config.vector_size == 768


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test suite for module constants."""

    def test_expert_type_values(self):
        """Test ExpertType enum values."""
        assert ExpertType.LITERAL.value == "literal"
        assert ExpertType.SYSTEMIC.value == "systemic"
        assert ExpertType.PRINCIPLES.value == "principles"
        assert ExpertType.PRECEDENT.value == "precedent"

    def test_default_expert_affinities(self):
        """Test default expert affinities by source type."""
        # Norm should favor literal expert
        assert DEFAULT_EXPERT_AFFINITIES[SourceType.NORM]["literal"] == 0.9
        assert DEFAULT_EXPERT_AFFINITIES[SourceType.NORM]["precedent"] == 0.3

        # Jurisprudence should favor precedent expert
        assert DEFAULT_EXPERT_AFFINITIES[SourceType.JURISPRUDENCE]["precedent"] == 0.9
        assert DEFAULT_EXPERT_AFFINITIES[SourceType.JURISPRUDENCE]["literal"] == 0.3

        # Doctrine should favor principles expert
        assert DEFAULT_EXPERT_AFFINITIES[SourceType.DOCTRINE]["principles"] == 0.9

    def test_hnsw_config(self):
        """Test HNSW configuration values."""
        assert HNSW_CONFIG["m"] == 16
        assert HNSW_CONFIG["ef_construct"] == 128
        assert HNSW_CONFIG["ef"] == 128


# =============================================================================
# SearchResult Tests
# =============================================================================


class TestSearchResult:
    """Test suite for SearchResult dataclass."""

    def test_creation(self):
        """Test SearchResult creation."""
        result = SearchResult(
            chunk_id="test-chunk",
            score=0.95,
            text="Test text",
            source_urn="urn:test",
            source_type="norm",
            source_authority=1.0,
            article_urn="urn:test:art1",
            expert_affinity={"literal": 0.9},
        )

        assert result.chunk_id == "test-chunk"
        assert result.score == 0.95
        assert result.source_type == "norm"

    def test_to_dict(self):
        """Test SearchResult serialization."""
        result = SearchResult(
            chunk_id="test-chunk",
            score=0.85,
            text="Text",
            source_urn="urn:test",
            source_type="commentary",
            source_authority=0.5,
            article_urn="urn:test:art",
            expert_affinity={"literal": 0.5},
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["chunk_id"] == "test-chunk"
        assert d["score"] == 0.85
        assert d["metadata"]["key"] == "value"


# =============================================================================
# QdrantCollectionManager Tests (Integration)
# =============================================================================


class TestQdrantCollectionManager:
    """Integration tests for QdrantCollectionManager with real Qdrant."""

    def test_collection_creation(self, manager):
        """Test collection exists after creation (AC1)."""
        assert manager.collection_exists() is True

    def test_collection_info(self, manager):
        """Test getting collection info (AC1)."""
        info = manager.get_collection_info()

        assert info is not None
        assert info["name"] == TEST_COLLECTION
        assert "points_count" in info
        assert "indexed_vectors_count" in info
        assert "status" in info

    def test_upsert_points(self, manager, sample_chunks, sample_embeddings):
        """Test inserting points with payload (AC1)."""
        count = manager.upsert_points(sample_chunks, sample_embeddings)

        assert count == 4

        # Verify points are in collection
        info = manager.get_collection_info()
        assert info["points_count"] >= 4

    def test_upsert_validates_length_mismatch(self, manager):
        """Test upsert fails with mismatched lengths."""
        chunks = [{"chunk_id": "c1", "text": "test"}]
        embeddings = [[0.1] * TEST_VECTOR_SIZE, [0.2] * TEST_VECTOR_SIZE]

        with pytest.raises(ValueError, match="must match"):
            manager.upsert_points(chunks, embeddings)

    def test_search_basic(self, manager, sample_chunks, sample_embeddings):
        """Test basic semantic search (AC3)."""
        # Ensure points are inserted
        manager.upsert_points(sample_chunks, sample_embeddings)

        # Search with first embedding
        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
        )

        assert len(results) > 0
        assert isinstance(results[0], SearchResult)
        assert results[0].score > 0
        assert results[0].text != ""

    def test_search_filter_by_source_type(self, manager, sample_chunks, sample_embeddings):
        """Test filtering by source_type (AC2)."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        # Search only norms
        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            source_types=["norm"],
        )

        # All results should be norms
        for r in results:
            assert r.source_type == "norm"

    def test_search_filter_by_multiple_source_types(self, manager, sample_chunks, sample_embeddings):
        """Test filtering by multiple source types (AC2)."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            source_types=["norm", "jurisprudence"],
        )

        for r in results:
            assert r.source_type in ["norm", "jurisprudence"]

    def test_search_filter_by_min_authority(self, manager, sample_chunks, sample_embeddings):
        """Test filtering by minimum authority (AC2)."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        # Search with high authority threshold
        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            min_authority=0.7,
        )

        # All results should have authority >= 0.7
        for r in results:
            assert r.source_authority >= 0.7

    def test_search_with_expert_boost(self, manager, sample_chunks, sample_embeddings):
        """Test expert affinity boosting (AC2)."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        # Search with literal expert boost
        results_literal = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            expert_type="literal",
        )

        # Search with precedent expert boost
        results_precedent = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            expert_type="precedent",
        )

        # Both should return results
        assert len(results_literal) > 0
        assert len(results_precedent) > 0

        # Order might differ due to boosting
        # Norms should rank higher for literal, jurisprudence for precedent

    def test_search_returns_full_payload(self, manager, sample_chunks, sample_embeddings):
        """Test search returns full payload for display (AC3)."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=1,
        )

        assert len(results) >= 1
        result = results[0]

        # Verify all payload fields
        assert result.chunk_id != ""
        assert result.text != ""
        assert result.source_urn != ""
        assert result.source_type in ["norm", "jurisprudence", "commentary", "doctrine"]
        assert 0.0 <= result.source_authority <= 1.0
        assert result.article_urn != ""
        assert isinstance(result.expert_affinity, dict)

    def test_search_with_score_threshold(self, manager, sample_chunks, sample_embeddings):
        """Test search with minimum score threshold."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            score_threshold=0.5,
        )

        for r in results:
            assert r.score >= 0.5

    def test_delete_points(self, manager):
        """Test deleting points by chunk_id."""
        # Insert a test point
        chunk_id = f"delete-test-{uuid.uuid4().hex[:8]}"
        chunks = [{
            "chunk_id": chunk_id,
            "text": "Test for deletion",
            "source_urn": "urn:test:delete",
            "source_type": SourceType.NORM,
            "source_authority": 1.0,
            "article_urn": "urn:test:delete",
        }]
        embeddings = [[0.1] * TEST_VECTOR_SIZE]

        manager.upsert_points(chunks, embeddings)

        # Delete the point
        count = manager.delete_points([chunk_id])
        assert count == 1

    def test_expert_affinity_auto_computed(self, manager):
        """Test expert affinity is auto-computed from source_type."""
        chunk_id = f"affinity-test-{uuid.uuid4().hex[:8]}"
        chunks = [{
            "chunk_id": chunk_id,
            "text": "Test affinity computation",
            "source_urn": "urn:test:affinity",
            "source_type": SourceType.JURISPRUDENCE,  # Should get high precedent affinity
            "source_authority": 0.8,
            "article_urn": "urn:test:affinity",
        }]
        embeddings = [[0.5] * TEST_VECTOR_SIZE]

        manager.upsert_points(chunks, embeddings)

        # Search and verify affinity
        results = manager.search(
            query_embedding=embeddings[0],
            limit=10,
        )

        # Find our inserted chunk
        for r in results:
            if r.chunk_id == chunk_id:
                # Jurisprudence should have high precedent affinity
                assert r.expert_affinity.get("precedent", 0) == 0.9
                assert r.expert_affinity.get("literal", 0) == 0.3
                break

    def test_search_with_invalid_expert_type(self, manager, sample_chunks, sample_embeddings):
        """Test search with invalid expert_type uses default affinity."""
        manager.upsert_points(sample_chunks, sample_embeddings)

        # Search with invalid expert type - should not crash, uses default 0.5
        results = manager.search(
            query_embedding=sample_embeddings[0],
            limit=10,
            expert_type="invalid_expert",
        )

        # Should still return results
        assert len(results) > 0
        # Scores should be boosted with default 0.5 affinity: score * (0.5 + 0.5*0.5) = score * 0.75
        for r in results:
            assert r.score > 0


class TestQdrantCollectionManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_upsert_empty_list(self, manager):
        """Test upserting empty list."""
        count = manager.upsert_points([], [])
        assert count == 0

    def test_search_empty_collection(self, qdrant_config):
        """Test searching empty collection."""
        # Create a fresh empty collection
        empty_collection = f"empty_test_{uuid.uuid4().hex[:8]}"
        config = QdrantConfig(
            host=qdrant_config.host,
            port=qdrant_config.port,
            collection_name=empty_collection,
            vector_size=TEST_VECTOR_SIZE,
        )
        mgr = QdrantCollectionManager(config)
        mgr.create_collection(recreate=True)

        try:
            results = mgr.search(
                query_embedding=[0.1] * TEST_VECTOR_SIZE,
                limit=10,
            )
            assert results == []
        finally:
            mgr.delete_collection()

    def test_collection_recreate(self, qdrant_config):
        """Test recreating existing collection."""
        recreate_collection = f"recreate_test_{uuid.uuid4().hex[:8]}"
        config = QdrantConfig(
            host=qdrant_config.host,
            port=qdrant_config.port,
            collection_name=recreate_collection,
            vector_size=TEST_VECTOR_SIZE,
        )
        mgr = QdrantCollectionManager(config)

        try:
            # Create collection
            result1 = mgr.create_collection(recreate=False)
            assert result1 is True

            # Try to create again without recreate - should return False
            result2 = mgr.create_collection(recreate=False)
            assert result2 is False

            # Recreate - should return True
            result3 = mgr.create_collection(recreate=True)
            assert result3 is True
        finally:
            mgr.delete_collection()
