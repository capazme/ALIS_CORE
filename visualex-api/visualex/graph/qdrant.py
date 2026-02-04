"""
Qdrant Collection Manager
=========================

Configures and manages Qdrant collections for legal text chunks with rich payload schema.

Features:
- HNSW index configuration for NFR-P4 (<200ms search)
- Payload indexing for filtered queries (source_type, source_authority)
- Expert affinity boosting for retrieval
- Integration with ChunkResult and EmbeddingResult from Stories 2c-1, 2c-2

Default expert affinities based on source type follow MERL-T specification.
"""

import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

from visualex.graph.chunking import SourceType

__all__ = [
    "QdrantConfig",
    "QdrantCollectionManager",
    "SearchResult",
    "ExpertType",
    "DEFAULT_EXPERT_AFFINITIES",
    "HNSW_CONFIG",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class ExpertType(str, Enum):
    """MERL-T Expert types for affinity-based retrieval."""

    LITERAL = "literal"
    SYSTEMIC = "systemic"
    PRINCIPLES = "principles"
    PRECEDENT = "precedent"


# HNSW index configuration for quality/speed balance (NFR-P4: <200ms)
HNSW_CONFIG = {
    "m": 16,  # Number of connections per node
    "ef_construct": 128,  # Build-time accuracy
    "ef": 128,  # Search-time accuracy (can be overridden per query)
}

# Default expert affinities by source type (MERL-T specification)
DEFAULT_EXPERT_AFFINITIES: Dict[SourceType, Dict[str, float]] = {
    SourceType.NORM: {
        "literal": 0.9,
        "systemic": 0.8,
        "principles": 0.5,
        "precedent": 0.3,
    },
    SourceType.JURISPRUDENCE: {
        "literal": 0.3,
        "systemic": 0.5,
        "principles": 0.6,
        "precedent": 0.9,
    },
    SourceType.COMMENTARY: {
        "literal": 0.5,
        "systemic": 0.6,
        "principles": 0.7,
        "precedent": 0.6,
    },
    SourceType.DOCTRINE: {
        "literal": 0.4,
        "systemic": 0.5,
        "principles": 0.9,
        "precedent": 0.4,
    },
}

# Default collection name
DEFAULT_COLLECTION = "legal_chunks"


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection and collection."""

    host: str = "localhost"
    port: int = 6333
    collection_name: str = DEFAULT_COLLECTION
    vector_size: int = 1024  # E5-large default
    distance: str = "Cosine"  # Cosine similarity for normalized embeddings
    on_disk: bool = False  # Store vectors on disk for large collections
    grpc_port: Optional[int] = None  # gRPC port for faster operations

    @classmethod
    def from_env(cls) -> "QdrantConfig":
        """Create configuration from environment variables."""
        # Validate port with fallback
        try:
            port = int(os.getenv("QDRANT_PORT", "6333"))
        except ValueError:
            logger.warning("Invalid QDRANT_PORT, using default 6333")
            port = 6333

        # Validate vector_size with fallback
        try:
            vector_size = int(os.getenv("QDRANT_VECTOR_SIZE", "1024"))
        except ValueError:
            logger.warning("Invalid QDRANT_VECTOR_SIZE, using default 1024")
            vector_size = 1024

        return cls(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=port,
            collection_name=os.getenv("QDRANT_COLLECTION", DEFAULT_COLLECTION),
            vector_size=vector_size,
        )


@dataclass
class SearchResult:
    """Result from Qdrant semantic search."""

    chunk_id: str
    score: float
    text: str
    source_urn: str
    source_type: str
    source_authority: float
    article_urn: str
    expert_affinity: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "source_urn": self.source_urn,
            "source_type": self.source_type,
            "source_authority": self.source_authority,
            "article_urn": self.article_urn,
            "expert_affinity": self.expert_affinity,
            "metadata": self.metadata,
        }


# =============================================================================
# Qdrant Collection Manager
# =============================================================================


class QdrantCollectionManager:
    """
    Manages Qdrant collections for legal text chunks.

    Provides:
    - Collection creation with HNSW configuration
    - Payload index setup for filtered queries
    - Point insertion with expert affinity computation
    - Semantic search with source type/authority filters

    Example:
        manager = QdrantCollectionManager(config)
        manager.create_collection()
        manager.upsert_points(chunks, embeddings)
        results = manager.search(query_embedding, source_types=["norm"])
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize QdrantCollectionManager.

        Args:
            config: Qdrant configuration. If None, uses environment variables.
        """
        self.config = config or QdrantConfig.from_env()
        self._client = None

        logger.info(
            "QdrantCollectionManager initialized: %s:%d/%s (dim=%d)",
            self.config.host,
            self.config.port,
            self.config.collection_name,
            self.config.vector_size,
        )

    def _get_client(self):
        """
        Lazy load Qdrant client.

        Returns:
            QdrantClient instance
        """
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as e:
                raise ImportError(
                    "qdrant-client is required. Install with: pip install qdrant-client"
                ) from e

            self._client = QdrantClient(
                host=self.config.host,
                port=self.config.port,
                grpc_port=self.config.grpc_port,
            )

        return self._client

    @property
    def client(self):
        """Get Qdrant client."""
        return self._get_client()

    def create_collection(self, recreate: bool = False) -> bool:
        """
        Create collection with HNSW configuration and payload indexes (AC1).

        Args:
            recreate: If True, delete existing collection first

        Returns:
            True if collection was created, False if already exists
        """
        from qdrant_client.models import (
            Distance,
            VectorParams,
            HnswConfigDiff,
            PayloadSchemaType,
        )

        client = self._get_client()

        # Check if collection exists
        collections = client.get_collections().collections
        exists = any(c.name == self.config.collection_name for c in collections)

        if exists:
            if recreate:
                logger.warning(
                    "Deleting existing collection: %s", self.config.collection_name
                )
                client.delete_collection(self.config.collection_name)
            else:
                logger.info(
                    "Collection already exists: %s", self.config.collection_name
                )
                return False

        # Map distance string to enum
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclid": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        distance = distance_map.get(self.config.distance, Distance.COSINE)

        # Create collection with HNSW config
        client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=VectorParams(
                size=self.config.vector_size,
                distance=distance,
                on_disk=self.config.on_disk,
                hnsw_config=HnswConfigDiff(
                    m=HNSW_CONFIG["m"],
                    ef_construct=HNSW_CONFIG["ef_construct"],
                ),
            ),
        )

        logger.info(
            "Created collection %s (dim=%d, distance=%s, HNSW m=%d ef=%d)",
            self.config.collection_name,
            self.config.vector_size,
            self.config.distance,
            HNSW_CONFIG["m"],
            HNSW_CONFIG["ef_construct"],
        )

        # Create payload indexes for filtering
        self._create_payload_indexes()

        return True

    def _create_payload_indexes(self):
        """Create payload field indexes for efficient filtering (AC1)."""
        from qdrant_client.models import PayloadSchemaType

        client = self._get_client()

        # Define indexed fields
        indexes = [
            ("chunk_id", PayloadSchemaType.KEYWORD),
            ("source_urn", PayloadSchemaType.KEYWORD),
            ("source_type", PayloadSchemaType.KEYWORD),
            ("article_urn", PayloadSchemaType.KEYWORD),
            ("model_id", PayloadSchemaType.KEYWORD),
            # Float fields for range queries
            ("source_authority", PayloadSchemaType.FLOAT),
        ]

        for field_name, schema_type in indexes:
            try:
                client.create_payload_index(
                    collection_name=self.config.collection_name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
                logger.debug("Created index on %s (%s)", field_name, schema_type)
            except Exception as e:
                # Index might already exist
                logger.debug("Index creation skipped for %s: %s", field_name, str(e))

        logger.info("Payload indexes configured for %s", self.config.collection_name)

    def collection_exists(self) -> bool:
        """Check if collection exists."""
        client = self._get_client()
        collections = client.get_collections().collections
        return any(c.name == self.config.collection_name for c in collections)

    def get_collection_info(self) -> Optional[Dict[str, Any]]:
        """Get collection statistics."""
        if not self.collection_exists():
            return None

        client = self._get_client()
        info = client.get_collection(self.config.collection_name)

        return {
            "name": self.config.collection_name,
            "points_count": info.points_count or 0,
            "indexed_vectors_count": info.indexed_vectors_count or 0,
            "status": info.status.value if info.status else "unknown",
            "config": {
                "vector_size": info.config.params.vectors.size
                if hasattr(info.config.params.vectors, "size")
                else self.config.vector_size,
            },
        }

    def upsert_points(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Insert or update points in collection (AC1, AC3).

        Args:
            chunks: List of chunk dicts with metadata
            embeddings: List of embedding vectors (must match chunks length)
            batch_size: Batch size for upsert operations

        Returns:
            Number of points upserted
        """
        from qdrant_client.models import PointStruct

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
            )

        if not chunks:
            return 0

        client = self._get_client()
        points = []

        for chunk, embedding in zip(chunks, embeddings):
            # Get source type for expert affinity
            source_type_str = chunk.get("source_type", "norm")
            if isinstance(source_type_str, SourceType):
                source_type = source_type_str
            else:
                try:
                    source_type = SourceType(source_type_str)
                except ValueError:
                    source_type = SourceType.NORM

            # Compute expert affinity based on source type
            expert_affinity = chunk.get(
                "expert_affinity", DEFAULT_EXPERT_AFFINITIES.get(source_type, {})
            )

            # Build payload
            payload = {
                "chunk_id": chunk.get("chunk_id", ""),
                "source_urn": chunk.get("source_urn", ""),
                "source_type": source_type.value,
                "source_authority": chunk.get("source_authority", 0.5),
                "article_urn": chunk.get("article_urn", chunk.get("parent_article_urn", "")),
                "text": chunk.get("text", ""),
                "expert_affinity": expert_affinity,
                "model_id": chunk.get("model_id", ""),
                "created_at": chunk.get("created_at", datetime.now().isoformat()),
            }

            # Add any extra metadata
            if "metadata" in chunk:
                payload["metadata"] = chunk["metadata"]

            # Convert chunk_id to UUID for Qdrant (deterministic via UUID5)
            chunk_id = chunk.get("chunk_id", str(len(points)))
            point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # Upsert in batches
        total_upserted = 0
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=self.config.collection_name,
                points=batch,
            )
            total_upserted += len(batch)

            if total_upserted % 500 == 0:
                logger.info("Upserted %d/%d points", total_upserted, len(points))

        logger.info(
            "Upserted %d points to %s", total_upserted, self.config.collection_name
        )
        return total_upserted

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        source_types: Optional[List[str]] = None,
        min_authority: Optional[float] = None,
        expert_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
        ef: Optional[int] = None,
    ) -> List[SearchResult]:
        """
        Semantic search with filters and expert affinity boosting (AC2, AC3).

        Args:
            query_embedding: Query vector
            limit: Maximum results to return
            source_types: Filter by source types (e.g., ["norm", "jurisprudence"])
            min_authority: Minimum source_authority threshold
            expert_type: Expert type for affinity-based re-ranking
            score_threshold: Minimum similarity score
            ef: Search-time ef parameter (overrides HNSW_CONFIG)

        Returns:
            List of SearchResult sorted by score
        """
        from qdrant_client.models import (
            Filter,
            FieldCondition,
            MatchAny,
            Range,
            SearchParams,
        )

        client = self._get_client()

        # Build filter conditions
        must_conditions = []

        if source_types:
            must_conditions.append(
                FieldCondition(key="source_type", match=MatchAny(any=source_types))
            )

        if min_authority is not None:
            must_conditions.append(
                FieldCondition(
                    key="source_authority", range=Range(gte=min_authority)
                )
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search params with HNSW ef
        search_params = SearchParams(
            hnsw_ef=ef or HNSW_CONFIG["ef"],
        )

        # Perform search using query_points (new API)
        response = client.query_points(
            collection_name=self.config.collection_name,
            query=query_embedding,
            limit=limit,
            query_filter=query_filter,
            search_params=search_params,
            score_threshold=score_threshold,
            with_payload=True,
        )

        # Convert to SearchResult
        search_results = []
        for r in response.points:
            payload = r.payload or {}

            result = SearchResult(
                chunk_id=payload.get("chunk_id", str(r.id)),
                score=r.score,
                text=payload.get("text", ""),
                source_urn=payload.get("source_urn", ""),
                source_type=payload.get("source_type", ""),
                source_authority=payload.get("source_authority", 0.0),
                article_urn=payload.get("article_urn", ""),
                expert_affinity=payload.get("expert_affinity", {}),
                metadata=payload.get("metadata", {}),
            )
            search_results.append(result)

        # Apply expert affinity boosting if specified
        if expert_type and search_results:
            search_results = self._apply_expert_boost(search_results, expert_type)

        logger.debug(
            "Search returned %d results (filters: source_types=%s, min_authority=%s)",
            len(search_results),
            source_types,
            min_authority,
        )

        return search_results

    def _apply_expert_boost(
        self, results: List[SearchResult], expert_type: str
    ) -> List[SearchResult]:
        """
        Re-rank results based on expert affinity (AC2).

        Creates new SearchResult objects with boosted scores to avoid mutating originals.

        Args:
            results: Search results to re-rank
            expert_type: Expert type (literal, systemic, principles, precedent)

        Returns:
            New list of SearchResult with boosted scores, sorted by score descending
        """
        expert_key = expert_type.lower()

        boosted_results = []
        for result in results:
            affinity = result.expert_affinity.get(expert_key, 0.5)
            # Boost score by affinity: boosted = score * (0.5 + 0.5 * affinity)
            # This gives range [0.5 * score, score] based on affinity
            boosted_score = result.score * (0.5 + 0.5 * affinity)

            # Create new SearchResult with boosted score
            boosted_results.append(
                SearchResult(
                    chunk_id=result.chunk_id,
                    score=boosted_score,
                    text=result.text,
                    source_urn=result.source_urn,
                    source_type=result.source_type,
                    source_authority=result.source_authority,
                    article_urn=result.article_urn,
                    expert_affinity=result.expert_affinity,
                    metadata=result.metadata,
                )
            )

        # Sort by boosted score
        boosted_results.sort(key=lambda x: x.score, reverse=True)

        return boosted_results

    def delete_points(self, chunk_ids: List[str]) -> int:
        """
        Delete points by chunk_id.

        Args:
            chunk_ids: List of chunk IDs to delete

        Returns:
            Number of chunk_ids requested for deletion (Qdrant API does not
            return actual deletion count; non-existent IDs are silently ignored)
        """
        from qdrant_client.models import PointIdsList

        if not chunk_ids:
            return 0

        # Convert chunk_ids to UUIDs (same conversion as in upsert)
        point_ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, cid)) for cid in chunk_ids]

        client = self._get_client()
        client.delete(
            collection_name=self.config.collection_name,
            points_selector=PointIdsList(points=point_ids),
        )

        logger.info("Deleted %d points from %s", len(chunk_ids), self.config.collection_name)
        return len(chunk_ids)

    def delete_collection(self) -> bool:
        """
        Delete the entire collection.

        Returns:
            True if deleted, False if didn't exist
        """
        if not self.collection_exists():
            return False

        client = self._get_client()
        client.delete_collection(self.config.collection_name)

        logger.info("Deleted collection: %s", self.config.collection_name)
        return True
