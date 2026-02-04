"""
Graph Module
============

FalkorDB graph database integration for visualex.

Provides schema definitions, client wrapper, index management, and ingestion pipeline.

Schema follows MERL-T Knowledge Graph specification:
- 26 Node Types
- 65 Relation Types
- LKIF Core, Akoma Ntoso, ELI, EuroVoc compliance
"""

from visualex.graph.config import FalkorDBConfig
from visualex.graph.schema import (
    GraphSchema,
    NodeType,
    EdgeType,
    Direction,
    IndexDefinition,
    NODE_PROPERTIES,
    EDGE_PROPERTIES,
    COMMON_EDGE_PROPERTIES,
    INDEXES,
    FULLTEXT_INDEXES,
)
from visualex.graph.client import FalkorDBClient
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
from visualex.graph.relations import (
    CitationExtractor,
    RelationCreator,
    ExtractedCitation,
    ExtractedModification,
    ExtractionResult,
    RelationType,
    extract_citations,
)
from visualex.graph.temporal import (
    TemporalQuery,
    VersionedNorm,
    VersionTimeline,
    VersionDiff,
    DiffSegment,
    NormStatus,
)
from visualex.graph.admin import (
    IngestService,
    IngestRequest,
    IngestJobResult,
)
from visualex.graph.chunking import (
    SourceType,
    ChunkResult,
    ChunkBatchResult,
    LegalChunker,
    ChunkingConfig,
)
from visualex.graph.embeddings import (
    EmbeddingConfig,
    EmbeddingResult,
    EmbeddingBatchResult,
    LegalEmbedder,
    SUPPORTED_MODELS,
)
from visualex.graph.qdrant import (
    QdrantConfig,
    QdrantCollectionManager,
    SearchResult,
    ExpertType,
    DEFAULT_EXPERT_AFFINITIES,
    HNSW_CONFIG,
)
from visualex.graph.bridge import (
    BridgeConfig,
    BridgeTableManager,
    BridgeMapping,
    MappingType,
    DEFAULT_EXPERT_AFFINITIES as BRIDGE_EXPERT_AFFINITIES,
)
from visualex.graph.search import (
    SearchConfig,
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    HybridSearchService,
    SearchMode,
)
from visualex.graph.neighbors import (
    ReferenceType,
    CrossReference,
    CrossReferenceGroup,
    CrossReferenceResponse,
    CrossReferenceService,
)
from visualex.graph.alerts import (
    AlertType,
    ModificationAlert,
    ModificationAlertService,
)
from visualex.graph.stats import (
    KGStats,
    NodeTypeCount,
    EdgeTypeCount,
    CoverageMetrics,
    SourceStatus,
    KGStatsService,
)

__all__ = [
    # Config
    "FalkorDBConfig",
    # Schema
    "GraphSchema",
    "NodeType",
    "EdgeType",
    "Direction",
    "IndexDefinition",
    "NODE_PROPERTIES",
    "EDGE_PROPERTIES",
    "COMMON_EDGE_PROPERTIES",
    "INDEXES",
    "FULLTEXT_INDEXES",
    # Client
    "FalkorDBClient",
    # Ingestion
    "NormIngester",
    "ArticleParser",
    "ArticleStructure",
    "CommaStructure",
    "LetteraStructure",
    "NumeroStructure",
    "IngestionResult",
    "BatchResult",
    # Relations
    "CitationExtractor",
    "RelationCreator",
    "ExtractedCitation",
    "ExtractedModification",
    "ExtractionResult",
    "RelationType",
    "extract_citations",
    # Temporal
    "TemporalQuery",
    "VersionedNorm",
    "VersionTimeline",
    "VersionDiff",
    "DiffSegment",
    "NormStatus",
    # Admin
    "IngestService",
    "IngestRequest",
    "IngestJobResult",
    # Chunking
    "SourceType",
    "ChunkResult",
    "ChunkBatchResult",
    "LegalChunker",
    "ChunkingConfig",
    # Embeddings
    "EmbeddingConfig",
    "EmbeddingResult",
    "EmbeddingBatchResult",
    "LegalEmbedder",
    "SUPPORTED_MODELS",
    # Qdrant
    "QdrantConfig",
    "QdrantCollectionManager",
    "SearchResult",
    "ExpertType",
    "DEFAULT_EXPERT_AFFINITIES",
    "HNSW_CONFIG",
    # Bridge
    "BridgeConfig",
    "BridgeTableManager",
    "BridgeMapping",
    "MappingType",
    "BRIDGE_EXPERT_AFFINITIES",
    # Search
    "SearchConfig",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "HybridSearchService",
    "SearchMode",
    # Neighbors/Cross-References
    "ReferenceType",
    "CrossReference",
    "CrossReferenceGroup",
    "CrossReferenceResponse",
    "CrossReferenceService",
    # Alerts
    "AlertType",
    "ModificationAlert",
    "ModificationAlertService",
    # Stats
    "KGStats",
    "NodeTypeCount",
    "EdgeTypeCount",
    "CoverageMetrics",
    "SourceStatus",
    "KGStatsService",
]
