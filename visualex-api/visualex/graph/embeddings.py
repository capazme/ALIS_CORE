"""
Legal Text Embedding Module
===========================

Configurable embedding generation for legal texts, with model versioning and batch processing.

Supports:
- Multiple embedding models (E5-large, BGE-M3, MiniLM)
- E5-style prefix handling (query/passage)
- Batch processing with progress logging
- GPU/CPU device selection
- Model versioning for reproducibility (NFR-R6)

Each embedding includes metadata for Bridge Table mapping and audit trail.
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Dict, List, Any, Optional

__all__ = [
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingBatchResult",
    "LegalEmbedder",
    "SUPPORTED_MODELS",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Configuration
# =============================================================================


# Supported embedding models with their configurations
SUPPORTED_MODELS: Dict[str, Dict[str, Any]] = {
    "intfloat/multilingual-e5-large": {
        "dimension": 1024,
        "max_tokens": 512,
        "requires_prefix": True,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
        "description": "Best for Italian legal text, multilingual",
    },
    "BAAI/bge-m3": {
        "dimension": 1024,
        "max_tokens": 8192,
        "requires_prefix": False,
        "query_prefix": "",
        "passage_prefix": "",
        "description": "Alternative multilingual, longer context",
    },
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_tokens": 128,
        "requires_prefix": False,
        "query_prefix": "",
        "passage_prefix": "",
        "description": "Lightweight option, faster inference",
    },
}

# Default model
DEFAULT_MODEL = "intfloat/multilingual-e5-large"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model_name: str = DEFAULT_MODEL
    device: Optional[str] = None  # None = auto-detect
    batch_size: int = 32
    normalize_embeddings: bool = True
    show_progress: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.model_name not in SUPPORTED_MODELS:
            logger.warning(
                "Model %s not in SUPPORTED_MODELS, assuming custom model",
                self.model_name,
            )


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""

    chunk_id: str
    embedding: List[float]
    model_id: str
    dimension: int
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chunk_id": self.chunk_id,
            "embedding": self.embedding,
            "model_id": self.model_id,
            "dimension": self.dimension,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }


@dataclass
class EmbeddingBatchResult:
    """Result of batch embedding."""

    total_chunks: int
    successful: int
    failed: int
    embeddings: List[EmbeddingResult] = field(default_factory=list)
    failed_chunks: List[Dict[str, str]] = field(default_factory=list)
    model_id: str = ""
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary."""
        return {
            "total_chunks": self.total_chunks,
            "successful": self.successful,
            "failed": self.failed,
            "model_id": self.model_id,
            "duration_seconds": self.duration_seconds,
        }


# =============================================================================
# Legal Embedder
# =============================================================================


class LegalEmbedder:
    """
    Configurable embedder for legal texts.

    Supports multiple models with automatic prefix handling for E5-style models.
    Provides batch processing with progress logging and failure isolation.

    Example:
        embedder = LegalEmbedder()
        results = embedder.embed_chunks(chunks)

        # Or with custom config
        config = EmbeddingConfig(model_name="BAAI/bge-m3", batch_size=64)
        embedder = LegalEmbedder(config)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize LegalEmbedder.

        Args:
            config: Optional configuration. If None, uses environment variables
                   or defaults.
        """
        self.config = config or self._config_from_env()
        self._model = None
        self._model_info = SUPPORTED_MODELS.get(self.config.model_name, {})
        self._lock = Lock()  # Instance-level lock for thread-safe model loading

        logger.info(
            "LegalEmbedder initialized with model=%s, device=%s, batch_size=%d",
            self.config.model_name,
            self.config.device or "auto",
            self.config.batch_size,
        )

    def _config_from_env(self) -> EmbeddingConfig:
        """Create configuration from environment variables."""
        # Parse batch_size with validation
        try:
            batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
        except ValueError:
            logger.warning("Invalid EMBEDDING_BATCH_SIZE, using default 32")
            batch_size = 32

        return EmbeddingConfig(
            model_name=os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL),
            device=os.getenv("EMBEDDING_DEVICE"),
            batch_size=batch_size,
            normalize_embeddings=os.getenv("EMBEDDING_NORMALIZE", "true").lower()
            == "true",
        )

    def _load_model(self):
        """
        Lazy load the sentence-transformers model.

        Thread-safe loading with double-checked locking.
        """
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        import torch
                    except ImportError as e:
                        raise ImportError(
                            "sentence-transformers and torch are required. "
                            "Install with: pip install sentence-transformers torch"
                        ) from e

                    device = self.config.device
                    if device is None:
                        device = "cuda" if torch.cuda.is_available() else "cpu"

                    logger.info(
                        "Loading embedding model: %s on device: %s",
                        self.config.model_name,
                        device,
                    )

                    self._model = SentenceTransformer(
                        self.config.model_name, device=device
                    )

                    logger.info(
                        "Model loaded. Dimension: %d",
                        self._model.get_sentence_embedding_dimension(),
                    )

        return self._model

    @property
    def model_id(self) -> str:
        """Get model identifier for versioning."""
        return self.config.model_name

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        if self._model_info:
            return self._model_info.get("dimension", 0)
        # If custom model, load to get dimension
        model = self._load_model()
        return model.get_sentence_embedding_dimension()

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def _prepare_text(self, text: str, is_query: bool = False) -> str:
        """
        Prepare text with appropriate prefix for E5-style models.

        Args:
            text: Raw text to prepare
            is_query: If True, use query prefix; else use passage prefix

        Returns:
            Text with prefix (if required by model)
        """
        if not self._model_info.get("requires_prefix", False):
            return text

        prefix = (
            self._model_info.get("query_prefix", "")
            if is_query
            else self._model_info.get("passage_prefix", "")
        )
        return f"{prefix}{text}"

    def embed_text(self, text: str, is_query: bool = False) -> List[float]:
        """
        Embed a single text.

        Args:
            text: Text to embed
            is_query: If True, treat as query (for E5 models)

        Returns:
            Embedding vector as list of floats
        """
        model = self._load_model()
        prepared_text = self._prepare_text(text, is_query)

        embedding = model.encode(
            prepared_text,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_texts(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Embed multiple texts in batch.

        Args:
            texts: List of texts to embed
            is_query: If True, treat as queries (for E5 models)
            show_progress: Override config show_progress setting

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        model = self._load_model()
        prepared_texts = [self._prepare_text(t, is_query) for t in texts]

        show = show_progress if show_progress is not None else self.config.show_progress

        embeddings = model.encode(
            prepared_texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize_embeddings,
            show_progress_bar=show,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_chunk(
        self,
        chunk_id: str,
        text: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """
        Embed a single chunk and return full result with metadata.

        Args:
            chunk_id: Unique identifier for the chunk
            text: Chunk text to embed
            extra_metadata: Additional metadata to include

        Returns:
            EmbeddingResult with embedding and metadata
        """
        embedding = self.embed_text(text, is_query=False)

        return EmbeddingResult(
            chunk_id=chunk_id,
            embedding=embedding,
            model_id=self.model_id,
            dimension=len(embedding),
            metadata=extra_metadata or {},
        )

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> EmbeddingBatchResult:
        """
        Embed a batch of chunks with progress logging and failure isolation.

        Each chunk dict should contain:
        - chunk_id: str (required)
        - text: str (required)
        - metadata: Optional[Dict] (preserved in result)

        Args:
            chunks: List of chunk dictionaries

        Returns:
            EmbeddingBatchResult with all embeddings and failure info
        """
        start_time = datetime.now()
        results: List[EmbeddingResult] = []
        failed: List[Dict[str, str]] = []

        if not chunks:
            return EmbeddingBatchResult(
                total_chunks=0,
                successful=0,
                failed=0,
                model_id=self.model_id,
            )

        # Extract texts for batch processing
        valid_chunks: List[Dict[str, Any]] = []
        for i, chunk in enumerate(chunks):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            text = chunk.get("text", "")

            if not text or not text.strip():
                failed.append({"chunk_id": chunk_id, "error": "Empty text"})
                continue

            valid_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text,
                    "metadata": chunk.get("metadata", {}),
                }
            )

        if not valid_chunks:
            duration = (datetime.now() - start_time).total_seconds()
            return EmbeddingBatchResult(
                total_chunks=len(chunks),
                successful=0,
                failed=len(failed),
                failed_chunks=failed,
                model_id=self.model_id,
                duration_seconds=round(duration, 2),
            )

        # Batch embed all valid texts
        logger.info(
            "Batch embedding %d chunks (model=%s)",
            len(valid_chunks),
            self.model_id,
        )

        try:
            texts = [c["text"] for c in valid_chunks]
            embeddings = self.embed_texts(texts, is_query=False)

            # Create results
            for chunk_data, embedding in zip(valid_chunks, embeddings):
                result = EmbeddingResult(
                    chunk_id=chunk_data["chunk_id"],
                    embedding=embedding,
                    model_id=self.model_id,
                    dimension=len(embedding),
                    metadata=chunk_data["metadata"],
                )
                results.append(result)

        except Exception as e:
            # If batch fails, try individual embedding with isolation
            logger.warning(
                "Batch embedding failed, falling back to individual: %s", str(e)
            )

            for chunk_data in valid_chunks:
                try:
                    embedding = self.embed_text(chunk_data["text"], is_query=False)
                    result = EmbeddingResult(
                        chunk_id=chunk_data["chunk_id"],
                        embedding=embedding,
                        model_id=self.model_id,
                        dimension=len(embedding),
                        metadata=chunk_data["metadata"],
                    )
                    results.append(result)

                except Exception as chunk_error:
                    logger.error(
                        "Failed to embed chunk %s: %s",
                        chunk_data["chunk_id"],
                        str(chunk_error),
                    )
                    failed.append(
                        {
                            "chunk_id": chunk_data["chunk_id"],
                            "error": str(chunk_error),
                        }
                    )

        duration = (datetime.now() - start_time).total_seconds()

        logger.info(
            "Batch embedding complete: %d/%d successful (%.2fs)",
            len(results),
            len(chunks),
            duration,
        )

        return EmbeddingBatchResult(
            total_chunks=len(chunks),
            successful=len(results),
            failed=len(failed),
            embeddings=results,
            failed_chunks=failed,
            model_id=self.model_id,
            duration_seconds=round(duration, 2),
        )

    # =========================================================================
    # Async Wrappers
    # =========================================================================

    async def embed_text_async(self, text: str, is_query: bool = False) -> List[float]:
        """
        Async wrapper for embed_text.

        Runs in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_text, text, is_query)

    async def embed_texts_async(
        self,
        texts: List[str],
        is_query: bool = False,
        show_progress: Optional[bool] = None,
    ) -> List[List[float]]:
        """
        Async wrapper for embed_texts.

        Runs in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.embed_texts(texts, is_query, show_progress)
        )

    async def embed_chunk_async(
        self,
        chunk_id: str,
        text: str,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingResult:
        """
        Async wrapper for embed_chunk.

        Runs in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.embed_chunk(chunk_id, text, extra_metadata)
        )

    async def embed_chunks_async(
        self,
        chunks: List[Dict[str, Any]],
    ) -> EmbeddingBatchResult:
        """
        Async wrapper for embed_chunks.

        Runs in thread pool to avoid blocking event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_chunks, chunks)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"LegalEmbedder(model={self.model_id}, "
            f"dimension={self.dimension}, "
            f"loaded={self.is_loaded})"
        )
