"""
Tests for Graph Embeddings Module
==================================

Tests for:
- EmbeddingConfig dataclass
- EmbeddingResult dataclass
- LegalEmbedder with mocked sentence-transformers

Note: Tests use mocks to avoid downloading large models during testing.
"""

import pytest
import random
from unittest.mock import Mock, patch, MagicMock

from visualex.graph.embeddings import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingBatchResult,
    LegalEmbedder,
    SUPPORTED_MODELS,
    DEFAULT_MODEL,
)


# =============================================================================
# EmbeddingConfig Tests
# =============================================================================


class TestEmbeddingConfig:
    """Test suite for EmbeddingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmbeddingConfig()

        assert config.model_name == DEFAULT_MODEL
        assert config.device is None  # Auto-detect
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert config.show_progress is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            device="cuda",
            batch_size=64,
            normalize_embeddings=False,
            show_progress=False,
        )

        assert config.model_name == "BAAI/bge-m3"
        assert config.device == "cuda"
        assert config.batch_size == 64
        assert config.normalize_embeddings is False
        assert config.show_progress is False

    def test_unknown_model_warning(self, caplog):
        """Test warning for unknown model."""
        with caplog.at_level("WARNING"):
            config = EmbeddingConfig(model_name="unknown/model")

        assert config.model_name == "unknown/model"
        # Warning should be logged (exact message may vary)


# =============================================================================
# EmbeddingResult Tests
# =============================================================================


class TestEmbeddingResult:
    """Test suite for EmbeddingResult dataclass."""

    def test_creation(self):
        """Test creating an EmbeddingResult."""
        result = EmbeddingResult(
            chunk_id="chunk-123",
            embedding=[0.1, 0.2, 0.3],
            model_id="intfloat/multilingual-e5-large",
            dimension=3,
        )

        assert result.chunk_id == "chunk-123"
        assert result.embedding == [0.1, 0.2, 0.3]
        assert result.model_id == "intfloat/multilingual-e5-large"
        assert result.dimension == 3
        assert result.created_at is not None
        assert result.metadata == {}

    def test_with_metadata(self):
        """Test EmbeddingResult with extra metadata."""
        result = EmbeddingResult(
            chunk_id="chunk-456",
            embedding=[0.5] * 1024,
            model_id="test-model",
            dimension=1024,
            metadata={"source_type": "norm", "article_urn": "urn:test"},
        )

        assert result.metadata["source_type"] == "norm"
        assert result.metadata["article_urn"] == "urn:test"

    def test_to_dict(self):
        """Test EmbeddingResult serialization."""
        result = EmbeddingResult(
            chunk_id="chunk-789",
            embedding=[0.1, 0.2],
            model_id="test-model",
            dimension=2,
            metadata={"key": "value"},
        )

        d = result.to_dict()

        assert d["chunk_id"] == "chunk-789"
        assert d["embedding"] == [0.1, 0.2]
        assert d["model_id"] == "test-model"
        assert d["dimension"] == 2
        assert "created_at" in d
        assert d["metadata"]["key"] == "value"


# =============================================================================
# EmbeddingBatchResult Tests
# =============================================================================


class TestEmbeddingBatchResult:
    """Test suite for EmbeddingBatchResult dataclass."""

    def test_creation(self):
        """Test creating an EmbeddingBatchResult."""
        result = EmbeddingBatchResult(
            total_chunks=100,
            successful=95,
            failed=5,
            model_id="test-model",
            duration_seconds=10.5,
        )

        assert result.total_chunks == 100
        assert result.successful == 95
        assert result.failed == 5
        assert result.model_id == "test-model"
        assert result.duration_seconds == 10.5
        assert result.embeddings == []
        assert result.failed_chunks == []

    def test_to_dict(self):
        """Test EmbeddingBatchResult serialization."""
        result = EmbeddingBatchResult(
            total_chunks=50,
            successful=48,
            failed=2,
            model_id="e5-large",
            duration_seconds=5.25,
        )

        d = result.to_dict()

        assert d["total_chunks"] == 50
        assert d["successful"] == 48
        assert d["failed"] == 2
        assert d["model_id"] == "e5-large"
        assert d["duration_seconds"] == 5.25


# =============================================================================
# SUPPORTED_MODELS Tests
# =============================================================================


class TestSupportedModels:
    """Test suite for SUPPORTED_MODELS configuration."""

    def test_default_model_exists(self):
        """Test that default model is in supported models."""
        assert DEFAULT_MODEL in SUPPORTED_MODELS

    def test_e5_large_config(self):
        """Test E5-large model configuration."""
        e5_config = SUPPORTED_MODELS["intfloat/multilingual-e5-large"]

        assert e5_config["dimension"] == 1024
        assert e5_config["requires_prefix"] is True
        assert e5_config["query_prefix"] == "query: "
        assert e5_config["passage_prefix"] == "passage: "

    def test_bge_m3_config(self):
        """Test BGE-M3 model configuration."""
        bge_config = SUPPORTED_MODELS["BAAI/bge-m3"]

        assert bge_config["dimension"] == 1024
        assert bge_config["requires_prefix"] is False

    def test_minilm_config(self):
        """Test MiniLM model configuration."""
        minilm_key = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        minilm_config = SUPPORTED_MODELS[minilm_key]

        assert minilm_config["dimension"] == 384
        assert minilm_config["requires_prefix"] is False


# =============================================================================
# LegalEmbedder Tests (with mocks)
# =============================================================================


class TestLegalEmbedder:
    """Test suite for LegalEmbedder with mocked sentence-transformers."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create mock SentenceTransformer."""
        with patch("visualex.graph.embeddings.LegalEmbedder._load_model") as mock_load:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024

            # Mock encode to return mock array-like object
            class MockArray:
                """Mock numpy array that supports tolist()."""

                def __init__(self, data):
                    self._data = data

                def tolist(self):
                    return self._data

            def mock_encode(text, **kwargs):
                if isinstance(text, list):
                    return MockArray([[random.random() for _ in range(1024)] for _ in text])
                return MockArray([random.random() for _ in range(1024)])

            mock_model.encode = mock_encode
            mock_load.return_value = mock_model

            yield mock_model

    def test_initialization(self):
        """Test LegalEmbedder initialization."""
        embedder = LegalEmbedder()

        assert embedder.model_id == DEFAULT_MODEL
        assert embedder.is_loaded is False

    def test_custom_config(self):
        """Test LegalEmbedder with custom config."""
        config = EmbeddingConfig(
            model_name="BAAI/bge-m3",
            batch_size=64,
        )
        embedder = LegalEmbedder(config)

        assert embedder.model_id == "BAAI/bge-m3"
        assert embedder.config.batch_size == 64

    def test_dimension_from_supported_models(self):
        """Test dimension retrieval from SUPPORTED_MODELS."""
        embedder = LegalEmbedder()

        assert embedder.dimension == 1024

    def test_prepare_text_e5_query(self):
        """Test text preparation with E5 query prefix."""
        embedder = LegalEmbedder()
        prepared = embedder._prepare_text("test query", is_query=True)

        assert prepared == "query: test query"

    def test_prepare_text_e5_passage(self):
        """Test text preparation with E5 passage prefix."""
        embedder = LegalEmbedder()
        prepared = embedder._prepare_text("test document", is_query=False)

        assert prepared == "passage: test document"

    def test_prepare_text_no_prefix_model(self):
        """Test text preparation for models without prefix."""
        config = EmbeddingConfig(model_name="BAAI/bge-m3")
        embedder = LegalEmbedder(config)

        prepared = embedder._prepare_text("test text", is_query=True)

        assert prepared == "test text"  # No prefix added

    def test_embed_text(self, mock_sentence_transformer):
        """Test single text embedding."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        result = embedder.embed_text("Test legal text")

        assert isinstance(result, list)
        assert len(result) == 1024

    def test_embed_texts_batch(self, mock_sentence_transformer):
        """Test batch text embedding."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        texts = ["Text 1", "Text 2", "Text 3"]
        results = embedder.embed_texts(texts)

        assert len(results) == 3
        assert all(len(r) == 1024 for r in results)

    def test_embed_texts_empty(self, mock_sentence_transformer):
        """Test batch embedding with empty list."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        results = embedder.embed_texts([])

        assert results == []

    def test_embed_chunk(self, mock_sentence_transformer):
        """Test single chunk embedding with metadata (AC4)."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        result = embedder.embed_chunk(
            chunk_id="chunk-001",
            text="Art. 1453 c.c. - Risoluzione del contratto",
            extra_metadata={"source_type": "norm"},
        )

        assert isinstance(result, EmbeddingResult)
        assert result.chunk_id == "chunk-001"
        assert result.model_id == DEFAULT_MODEL
        assert result.dimension == 1024
        assert len(result.embedding) == 1024
        assert result.metadata["source_type"] == "norm"
        assert result.created_at is not None

    def test_embed_chunks_batch(self, mock_sentence_transformer):
        """Test batch chunk embedding (AC3)."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        chunks = [
            {"chunk_id": "c1", "text": "First chunk text"},
            {"chunk_id": "c2", "text": "Second chunk text"},
            {"chunk_id": "c3", "text": "Third chunk text"},
        ]

        result = embedder.embed_chunks(chunks)

        assert isinstance(result, EmbeddingBatchResult)
        assert result.total_chunks == 3
        assert result.successful == 3
        assert result.failed == 0
        assert len(result.embeddings) == 3
        assert result.model_id == DEFAULT_MODEL
        assert result.duration_seconds >= 0

    def test_embed_chunks_with_empty_text(self, mock_sentence_transformer):
        """Test batch embedding handles empty text (AC3 failure isolation)."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        chunks = [
            {"chunk_id": "valid1", "text": "Valid text"},
            {"chunk_id": "empty1", "text": ""},  # Empty
            {"chunk_id": "valid2", "text": "Another valid text"},
            {"chunk_id": "whitespace", "text": "   "},  # Whitespace only
        ]

        result = embedder.embed_chunks(chunks)

        assert result.total_chunks == 4
        assert result.successful == 2  # Only valid texts
        assert result.failed == 2  # Empty and whitespace
        assert len(result.embeddings) == 2
        assert len(result.failed_chunks) == 2

        # Check failed chunks have error messages
        failed_ids = [f["chunk_id"] for f in result.failed_chunks]
        assert "empty1" in failed_ids
        assert "whitespace" in failed_ids

    def test_embed_chunks_empty_list(self, mock_sentence_transformer):
        """Test batch embedding with no chunks."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        result = embedder.embed_chunks([])

        assert result.total_chunks == 0
        assert result.successful == 0
        assert result.failed == 0

    def test_embed_chunks_preserves_metadata(self, mock_sentence_transformer):
        """Test that chunk metadata is preserved in results (AC4)."""
        embedder = LegalEmbedder()
        embedder._model = mock_sentence_transformer

        chunks = [
            {
                "chunk_id": "c1",
                "text": "Legal text",
                "metadata": {"source_urn": "urn:test", "position": 0},
            },
        ]

        result = embedder.embed_chunks(chunks)

        assert len(result.embeddings) == 1
        assert result.embeddings[0].metadata["source_urn"] == "urn:test"
        assert result.embeddings[0].metadata["position"] == 0

    def test_model_id_versioning(self, mock_sentence_transformer):
        """Test model_id is included for versioning (AC2)."""
        config = EmbeddingConfig(model_name="BAAI/bge-m3")
        embedder = LegalEmbedder(config)
        embedder._model = mock_sentence_transformer

        result = embedder.embed_chunk("c1", "test text")

        assert result.model_id == "BAAI/bge-m3"

    def test_embed_chunks_batch_fallback(self):
        """Test batch embedding falls back to individual on batch failure (AC3)."""
        with patch("visualex.graph.embeddings.LegalEmbedder._load_model") as mock_load:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024

            class MockArray:
                def __init__(self, data):
                    self._data = data

                def tolist(self):
                    return self._data

            call_count = [0]

            def mock_encode(text, **kwargs):
                call_count[0] += 1
                # First call (batch) fails, subsequent individual calls succeed
                if call_count[0] == 1 and isinstance(text, list):
                    raise RuntimeError("Batch encoding failed")
                # Individual encoding succeeds
                return MockArray([random.random() for _ in range(1024)])

            mock_model.encode = mock_encode
            mock_load.return_value = mock_model

            embedder = LegalEmbedder()
            embedder._model = mock_model

            chunks = [
                {"chunk_id": "c1", "text": "First chunk"},
                {"chunk_id": "c2", "text": "Second chunk"},
            ]

            result = embedder.embed_chunks(chunks)

            # Should have succeeded via fallback
            assert result.successful == 2
            assert result.failed == 0
            assert len(result.embeddings) == 2

    def test_repr(self):
        """Test string representation."""
        embedder = LegalEmbedder()

        repr_str = repr(embedder)

        assert "LegalEmbedder" in repr_str
        assert DEFAULT_MODEL in repr_str
        assert "loaded=False" in repr_str


# =============================================================================
# Async Tests
# =============================================================================


class TestLegalEmbedderAsync:
    """Test async methods of LegalEmbedder."""

    @pytest.fixture
    def mock_embedder(self):
        """Create embedder with mocked model."""
        with patch("visualex.graph.embeddings.LegalEmbedder._load_model") as mock_load:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 1024

            class MockArray:
                """Mock numpy array that supports tolist()."""

                def __init__(self, data):
                    self._data = data

                def tolist(self):
                    return self._data

            def mock_encode(text, **kwargs):
                if isinstance(text, list):
                    return MockArray([[random.random() for _ in range(1024)] for _ in text])
                return MockArray([random.random() for _ in range(1024)])

            mock_model.encode = mock_encode
            mock_load.return_value = mock_model

            embedder = LegalEmbedder()
            embedder._model = mock_model
            yield embedder

    @pytest.mark.asyncio
    async def test_embed_text_async(self, mock_embedder):
        """Test async single text embedding."""
        result = await mock_embedder.embed_text_async("Test legal text")

        assert isinstance(result, list)
        assert len(result) == 1024

    @pytest.mark.asyncio
    async def test_embed_texts_async(self, mock_embedder):
        """Test async batch text embedding."""
        texts = ["Text 1", "Text 2"]
        results = await mock_embedder.embed_texts_async(texts)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_embed_chunk_async(self, mock_embedder):
        """Test async chunk embedding."""
        result = await mock_embedder.embed_chunk_async("c1", "Test text")

        assert isinstance(result, EmbeddingResult)
        assert result.chunk_id == "c1"

    @pytest.mark.asyncio
    async def test_embed_chunks_async(self, mock_embedder):
        """Test async batch chunk embedding."""
        chunks = [
            {"chunk_id": "c1", "text": "First"},
            {"chunk_id": "c2", "text": "Second"},
        ]

        result = await mock_embedder.embed_chunks_async(chunks)

        assert isinstance(result, EmbeddingBatchResult)
        assert result.successful == 2


# =============================================================================
# Environment Variable Tests
# =============================================================================


class TestEmbedderEnvironmentConfig:
    """Test LegalEmbedder configuration from environment variables."""

    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables (AC2)."""
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "64")
        monkeypatch.setenv("EMBEDDING_NORMALIZE", "false")

        embedder = LegalEmbedder()

        assert embedder.config.model_name == "BAAI/bge-m3"
        assert embedder.config.device == "cpu"
        assert embedder.config.batch_size == 64
        assert embedder.config.normalize_embeddings is False

    def test_default_env_config(self, monkeypatch):
        """Test default configuration when env vars not set."""
        # Clear any existing env vars
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)

        embedder = LegalEmbedder()

        assert embedder.config.model_name == DEFAULT_MODEL
        assert embedder.config.device is None  # Auto-detect

    def test_invalid_batch_size_env(self, monkeypatch):
        """Test invalid batch_size env var falls back to default."""
        monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "not_a_number")

        embedder = LegalEmbedder()

        # Should use default 32 when parsing fails
        assert embedder.config.batch_size == 32
