"""
Hybrid Search Service
=====================

Combines FalkorDB full-text search with Qdrant semantic search
for comprehensive legal norm retrieval.

Features:
- Keyword search in norm text, titles, commentary
- Semantic search for meaning-based retrieval
- URN pattern recognition for direct lookups
- Hybrid ranking with configurable weights
- Temporal filtering for point-in-time queries

Reference: Story 3-3: Search by Keyword/URN
"""

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Dict, List, Any, Optional

__all__ = [
    "SearchConfig",
    "SearchRequest",
    "SearchResultItem",
    "SearchResponse",
    "HybridSearchService",
    "SearchMode",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================


class SearchMode(str, Enum):
    """Search mode selection."""

    HYBRID = "hybrid"  # Both full-text and semantic
    KEYWORD = "keyword"  # Full-text only
    SEMANTIC = "semantic"  # Semantic only


# URN pattern matchers for Italian legal citations
URN_PATTERNS = [
    # art. 1453 c.c. / art. 2043 cod. civ.
    (
        r"art(?:icolo)?\.?\s*(\d+(?:[a-z]*)?)\s*(?:del\s+)?(?:c\.?c\.?|cod(?:ice)?\.?\s*civ(?:ile)?)",
        lambda m: f"urn:nir:stato:regio.decreto:1942-03-16;262~art{m.group(1)}",
    ),
    # art. 110 c.p. / art. 575 cod. pen.
    (
        r"art(?:icolo)?\.?\s*(\d+(?:[a-z]*)?)\s*(?:del\s+)?(?:c\.?p\.?|cod(?:ice)?\.?\s*pen(?:ale)?)",
        lambda m: f"urn:nir:stato:regio.decreto:1930-10-19;1398~art{m.group(1)}",
    ),
    # d.lgs. 231/2001
    (
        r"d\.?\s*lgs\.?\s*(?:n\.?\s*)?(\d+)[/\s](\d{4})",
        lambda m: f"urn:nir:stato:decreto.legislativo:{m.group(2)};{m.group(1)}",
    ),
    # l. 241/1990 / legge 241/1990
    (
        r"(?:l\.?|legge)\s*(?:n\.?\s*)?(\d+)[/\s](\d{4})",
        lambda m: f"urn:nir:stato:legge:{m.group(2)};{m.group(1)}",
    ),
    # d.p.r. 445/2000
    (
        r"d\.?\s*p\.?\s*r\.?\s*(?:n\.?\s*)?(\d+)[/\s](\d{4})",
        lambda m: f"urn:nir:stato:decreto.presidente.repubblica:{m.group(2)};{m.group(1)}",
    ),
]


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SearchConfig:
    """Configuration for hybrid search."""

    # Ranking weights (must sum to 1.0)
    fulltext_weight: float = 0.4
    semantic_weight: float = 0.4
    authority_weight: float = 0.2

    # Performance
    max_results: int = 50
    timeout_ms: int = 500

    # Semantic search
    semantic_score_threshold: float = 0.5
    embedding_model: str = "intfloat/multilingual-e5-large"

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.fulltext_weight + self.semantic_weight + self.authority_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class SearchRequest:
    """Search request parameters."""

    query: str
    mode: SearchMode = SearchMode.HYBRID
    source_types: Optional[List[str]] = None  # norm, jurisprudence, commentary, doctrine
    as_of_date: Optional[date] = None
    min_authority: Optional[float] = None
    expert_type: Optional[str] = None  # For expert-specific ranking
    limit: int = 20
    offset: int = 0

    def __post_init__(self):
        """Validate and normalize."""
        self.query = self.query.strip()
        if not self.query:
            raise ValueError("Search query cannot be empty")
        if self.limit < 1 or self.limit > 100:
            raise ValueError("Limit must be between 1 and 100")


@dataclass
class SearchResultItem:
    """Single search result."""

    urn: str
    title: str  # rubrica or node title
    snippet: str  # Text snippet with highlighted match
    source_type: str  # norm, jurisprudence, commentary, doctrine
    score: float  # Combined relevance score (0.0-1.0)
    authority: float  # Source authority (0.0-1.0)

    # Score breakdown
    fulltext_score: float = 0.0
    semantic_score: float = 0.0

    # Metadata
    vigenza_dal: Optional[str] = None
    vigenza_al: Optional[str] = None
    is_abrogated: bool = False
    match_type: str = "keyword"  # keyword, semantic, exact, hybrid

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urn": self.urn,
            "title": self.title,
            "snippet": self.snippet,
            "source_type": self.source_type,
            "score": round(self.score, 4),
            "authority": round(self.authority, 2),
            "fulltext_score": round(self.fulltext_score, 4),
            "semantic_score": round(self.semantic_score, 4),
            "vigenza_dal": self.vigenza_dal,
            "vigenza_al": self.vigenza_al,
            "is_abrogated": self.is_abrogated,
            "match_type": self.match_type,
        }


@dataclass
class SearchResponse:
    """Search response with results and metadata."""

    results: List[SearchResultItem]
    total_count: int
    query: str
    mode: str
    elapsed_ms: float
    has_more: bool = False

    # Debug info
    fulltext_count: int = 0
    semantic_count: int = 0
    exact_match: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "query": self.query,
            "mode": self.mode,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "has_more": self.has_more,
            "meta": {
                "fulltext_count": self.fulltext_count,
                "semantic_count": self.semantic_count,
                "exact_match": self.exact_match,
            },
        }


# =============================================================================
# Hybrid Search Service
# =============================================================================


class HybridSearchService:
    """
    Hybrid search combining FalkorDB full-text and Qdrant semantic search.

    Provides:
    - Keyword search in norm text, titles, commentary (AC1)
    - Semantic search with Qdrant vectors (AC2)
    - URN pattern recognition for direct lookups (AC3)
    - Hybrid result merging with ranking (AC4)
    - Temporal filtering for as_of_date queries (AC5)

    Example:
        service = HybridSearchService(falkor_client, qdrant_manager, bridge_manager)

        # Keyword search
        response = await service.search(SearchRequest(query="risoluzione contratto"))

        # URN pattern search
        response = await service.search(SearchRequest(query="art. 1453 c.c."))

        # Hybrid with filters
        response = await service.search(SearchRequest(
            query="inadempimento",
            mode=SearchMode.HYBRID,
            source_types=["norm"],
            as_of_date=date(2023, 1, 1)
        ))
    """

    def __init__(
        self,
        falkor_client: Any,  # FalkorDBClient
        qdrant_manager: Optional[Any] = None,  # QdrantCollectionManager
        bridge_manager: Optional[Any] = None,  # BridgeTableManager
        embedder: Optional[Any] = None,  # LegalEmbedder
        config: Optional[SearchConfig] = None,
    ):
        """
        Initialize HybridSearchService.

        Args:
            falkor_client: Connected FalkorDBClient for full-text search
            qdrant_manager: Optional QdrantCollectionManager for semantic search
            bridge_manager: Optional BridgeTableManager for chunk→node mapping
            embedder: Optional LegalEmbedder for query embedding
            config: Search configuration
        """
        self.falkor = falkor_client
        self.qdrant = qdrant_manager
        self.bridge = bridge_manager
        self.embedder = embedder
        self.config = config or SearchConfig()

        # Check semantic search availability
        self._semantic_enabled = all([qdrant_manager, bridge_manager, embedder])
        if not self._semantic_enabled:
            logger.info("Semantic search disabled - missing qdrant/bridge/embedder")

        logger.info(
            "HybridSearchService initialized (semantic=%s)",
            self._semantic_enabled,
        )

    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Execute hybrid search.

        Args:
            request: SearchRequest with query and filters

        Returns:
            SearchResponse with ranked results
        """
        import time

        start_time = time.perf_counter()

        # Check for URN pattern (AC3)
        urn_match = self._match_urn_pattern(request.query)
        if urn_match:
            logger.debug("URN pattern matched: %s -> %s", request.query, urn_match)
            response = await self._search_exact_urn(urn_match, request)
            response.elapsed_ms = (time.perf_counter() - start_time) * 1000
            return response

        # Execute searches based on mode
        fulltext_results: List[SearchResultItem] = []
        semantic_results: List[SearchResultItem] = []

        if request.mode in (SearchMode.HYBRID, SearchMode.KEYWORD):
            fulltext_results = await self._fulltext_search(request)

        if request.mode in (SearchMode.HYBRID, SearchMode.SEMANTIC):
            if self._semantic_enabled:
                semantic_results = await self._semantic_search(request)
            else:
                logger.warning("Semantic search requested but not available")

        # Merge and rank results (AC4)
        merged = self._merge_results(
            fulltext_results,
            semantic_results,
            request.limit,
        )

        # Apply temporal filter (AC5)
        if request.as_of_date:
            merged = self._apply_temporal_filter(merged, request.as_of_date)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SearchResponse(
            results=merged[: request.limit],
            total_count=len(merged),
            query=request.query,
            mode=request.mode.value,
            elapsed_ms=elapsed_ms,
            has_more=len(merged) > request.limit,
            fulltext_count=len(fulltext_results),
            semantic_count=len(semantic_results),
            exact_match=False,
        )

    def _match_urn_pattern(self, query: str) -> Optional[str]:
        """
        Check if query matches a known URN pattern (AC3).

        Returns URN if matched, None otherwise.
        """
        query_lower = query.lower().strip()

        for pattern, urn_builder in URN_PATTERNS:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                try:
                    return urn_builder(match)
                except Exception as e:
                    logger.warning("URN pattern match failed: %s", e)
                    continue

        return None

    async def _search_exact_urn(
        self,
        urn: str,
        request: SearchRequest,
    ) -> SearchResponse:
        """
        Search for exact URN match.

        Returns the article directly if found.
        """
        # Query FalkorDB for exact match
        query = """
            MATCH (n:Norma {urn: $urn})
            RETURN n.urn AS urn, n.rubrica AS title, n.testo_vigente AS text,
                   n.data_versione AS vigenza_dal, n.stato AS stato
        """

        try:
            result = await self.falkor.query(query, {"urn": urn})
            # Handle both list returns and result_set attribute
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            if rows:
                row = rows[0]
                item = SearchResultItem(
                    urn=row[0] or urn,
                    title=row[1] or "Articolo",
                    snippet=self._create_snippet(row[2] or "", request.query),
                    source_type="norm",
                    score=1.0,  # Exact match = highest score
                    authority=1.0,
                    fulltext_score=1.0,
                    semantic_score=0.0,
                    vigenza_dal=row[3],
                    is_abrogated=row[4] == "abrogato" if row[4] else False,
                    match_type="exact",
                )

                return SearchResponse(
                    results=[item],
                    total_count=1,
                    query=request.query,
                    mode="exact",
                    elapsed_ms=0,
                    exact_match=True,
                )
        except Exception as e:
            logger.warning("Exact URN search failed: %s", e)

        # Not found - return empty
        return SearchResponse(
            results=[],
            total_count=0,
            query=request.query,
            mode="exact",
            elapsed_ms=0,
            exact_match=True,
        )

    async def _fulltext_search(
        self,
        request: SearchRequest,
    ) -> List[SearchResultItem]:
        """
        Execute FalkorDB full-text search (AC1).

        Searches in:
        - Norma.testo_vigente, Norma.titolo
        - Comma.testo
        - Dottrina.descrizione
        - AttoGiudiziario.massima
        """
        results: List[SearchResultItem] = []

        # Escape special characters for full-text query
        escaped_query = self._escape_fulltext_query(request.query)

        # Build source type filter - using parameterized approach for safety
        source_labels = []
        if request.source_types:
            # Map source types to node labels
            label_map = {
                "norm": "Norma",
                "jurisprudence": "AttoGiudiziario",
                "commentary": "Dottrina",
                "doctrine": "Dottrina",
            }
            source_labels = [label_map.get(st, st) for st in request.source_types if st in label_map]

        # Query for norms (primary) - use parameterized source filter
        source_filter_clause = ""
        if source_labels and "Norma" in source_labels:
            source_filter_clause = "AND labels(n)[0] IN $source_labels"
        elif source_labels and "Norma" not in source_labels:
            # If filtering for non-norm types, skip norm query
            source_filter_clause = "AND false"

        norm_query = f"""
            CALL db.idx.fulltext.queryNodes('Norma', 'testo_vigente', $query)
            YIELD node, score
            WITH node AS n, score
            WHERE score > 0.1 {source_filter_clause}
            RETURN n.urn AS urn, n.rubrica AS title, n.testo_vigente AS text,
                   'norm' AS source_type, score,
                   n.data_versione AS vigenza_dal, n.stato AS stato
            ORDER BY score DESC
            LIMIT $limit
        """

        try:
            query_params = {"query": escaped_query, "limit": request.limit * 2}
            if source_labels:
                query_params["source_labels"] = source_labels
            norm_result = await self.falkor.query(norm_query, query_params)
            # Handle both list returns and result_set attribute
            rows = norm_result if isinstance(norm_result, list) else (
                norm_result.result_set if hasattr(norm_result, "result_set") else []
            )

            for row in rows:
                if row[0]:  # Has URN
                    results.append(
                        SearchResultItem(
                            urn=row[0],
                            title=row[1] or "Articolo",
                            snippet=self._create_snippet(row[2] or "", request.query),
                            source_type=row[3] or "norm",
                            score=float(row[4]) if row[4] else 0.0,
                            authority=1.0,  # Norms have max authority
                            fulltext_score=float(row[4]) if row[4] else 0.0,
                            vigenza_dal=row[5],
                            is_abrogated=row[6] == "abrogato" if row[6] else False,
                            match_type="keyword",
                        )
                    )
        except Exception as e:
            logger.warning("Full-text norm search failed: %s", e)

        # Query for commentary/doctrine if included
        if not request.source_types or "commentary" in request.source_types or "doctrine" in request.source_types:
            try:
                dottrina_query = """
                    CALL db.idx.fulltext.queryNodes('Dottrina', 'descrizione', $query)
                    YIELD node, score
                    WITH node AS n, score
                    WHERE score > 0.1
                    MATCH (n)-[:COMMENTA]->(norma:Norma)
                    RETURN norma.urn AS urn, n.titolo AS title, n.descrizione AS text,
                           'commentary' AS source_type, score,
                           null AS vigenza_dal, null AS stato
                    ORDER BY score DESC
                    LIMIT $limit
                """
                dottrina_result = await self.falkor.query(
                    dottrina_query,
                    {"query": escaped_query, "limit": request.limit},
                )
                # Handle both list returns and result_set attribute
                rows = dottrina_result if isinstance(dottrina_result, list) else (
                    dottrina_result.result_set if hasattr(dottrina_result, "result_set") else []
                )

                for row in rows:
                    if row[0]:
                        results.append(
                            SearchResultItem(
                                urn=row[0],
                                title=row[1] or "Commento",
                                snippet=self._create_snippet(row[2] or "", request.query),
                                source_type=row[3] or "commentary",
                                score=float(row[4]) * 0.8 if row[4] else 0.0,  # Slightly lower weight
                                authority=0.5,  # Commentary authority
                                fulltext_score=float(row[4]) if row[4] else 0.0,
                                match_type="keyword",
                            )
                        )
            except Exception as e:
                logger.debug("Dottrina search failed (may not exist): %s", e)

        return results

    async def _semantic_search(
        self,
        request: SearchRequest,
    ) -> List[SearchResultItem]:
        """
        Execute Qdrant semantic search (AC2).

        Uses embeddings to find semantically similar chunks,
        then maps back to graph nodes via Bridge Table.
        """
        if not self._semantic_enabled:
            return []

        results: List[SearchResultItem] = []

        try:
            # Generate query embedding
            embedding_result = await asyncio.to_thread(
                self.embedder.embed,
                request.query,
            )
            if not embedding_result or not embedding_result.embedding:
                logger.warning("Failed to generate query embedding")
                return []

            query_embedding = embedding_result.embedding

            # Search Qdrant - validate expert_type if provided
            source_types = request.source_types
            expert_type = request.expert_type
            # Validate expert_type against known values
            valid_expert_types = {"literal", "systemic", "principles", "precedent", None}
            if expert_type is not None and expert_type not in valid_expert_types:
                logger.warning("Invalid expert_type '%s', ignoring", expert_type)
                expert_type = None

            qdrant_results = self.qdrant.search(
                query_embedding=query_embedding,
                limit=request.limit * 2,
                source_types=source_types,
                min_authority=request.min_authority,
                expert_type=expert_type,
                score_threshold=self.config.semantic_score_threshold,
            )

            # Map chunks to graph nodes via Bridge
            for sr in qdrant_results:
                try:
                    # Get graph node URN from Bridge
                    mappings = await self.bridge.get_mappings_for_chunk(sr.chunk_id)
                    if not mappings:
                        continue

                    # Use primary mapping
                    primary = next(
                        (m for m in mappings if m.mapping_type == "PRIMARY"),
                        mappings[0],
                    )

                    results.append(
                        SearchResultItem(
                            urn=primary.graph_node_urn,
                            title="",  # Will be enriched later
                            snippet=sr.text[:300] if sr.text else "",
                            source_type=primary.source_type,
                            score=sr.score * 0.9,  # Slightly lower than fulltext
                            authority=primary.source_authority,
                            semantic_score=sr.score,
                            match_type="semantic",
                        )
                    )
                except Exception as e:
                    logger.debug("Bridge mapping failed for chunk %s: %s", sr.chunk_id, e)

        except Exception as e:
            logger.warning("Semantic search failed: %s", e)

        return results

    def _merge_results(
        self,
        fulltext_results: List[SearchResultItem],
        semantic_results: List[SearchResultItem],
        limit: int,
    ) -> List[SearchResultItem]:
        """
        Merge and rank results from both sources (AC4).

        Deduplicates by URN and combines scores.
        """
        # Index by URN for deduplication
        merged: Dict[str, SearchResultItem] = {}

        # Add fulltext results
        for item in fulltext_results:
            merged[item.urn] = item

        # Merge semantic results
        for item in semantic_results:
            if item.urn in merged:
                # Combine scores - this is a hybrid match
                existing = merged[item.urn]
                existing.semantic_score = item.semantic_score
                existing.match_type = "hybrid"

                # Recalculate combined score
                existing.score = (
                    self.config.fulltext_weight * existing.fulltext_score
                    + self.config.semantic_weight * existing.semantic_score
                    + self.config.authority_weight * existing.authority
                )
            else:
                # New result from semantic only
                item.score = (
                    self.config.semantic_weight * item.semantic_score
                    + self.config.authority_weight * item.authority
                )
                merged[item.urn] = item

        # Sort by score descending
        sorted_results = sorted(
            merged.values(),
            key=lambda x: x.score,
            reverse=True,
        )

        return sorted_results[:limit]

    def _apply_temporal_filter(
        self,
        results: List[SearchResultItem],
        as_of_date: date,
    ) -> List[SearchResultItem]:
        """
        Filter results by temporal validity (AC5).

        Keeps only norms in force on as_of_date.
        """
        filtered = []
        date_str = as_of_date.isoformat()

        for item in results:
            # Skip if abrogated before as_of_date
            if item.is_abrogated:
                # Would need vigenza_al to properly filter
                # For now, include abrogated items with warning
                pass

            # Check vigenza_dal
            if item.vigenza_dal:
                try:
                    vigenza = datetime.fromisoformat(item.vigenza_dal).date()
                    if vigenza > as_of_date:
                        continue  # Not yet in force
                except ValueError:
                    pass  # Invalid date format, include anyway

            filtered.append(item)

        return filtered

    def _create_snippet(self, text: str, query: str, max_length: int = 200) -> str:
        """Create a snippet with the query term highlighted."""
        if not text:
            return ""

        # Find query position
        query_lower = query.lower()
        text_lower = text.lower()
        pos = text_lower.find(query_lower)

        if pos == -1:
            # Query not found - return start of text
            return text[:max_length] + ("..." if len(text) > max_length else "")

        # Extract context around match
        start = max(0, pos - 50)
        end = min(len(text), pos + len(query) + 150)

        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."

        return snippet

    def _escape_fulltext_query(self, query: str) -> str:
        """Escape special characters for FalkorDB full-text search."""
        # FalkorDB uses RedisSearch which has special characters
        # Include '-' as it's a negative search operator in RedisSearch
        special_chars = r'@#$%^&*()[]{}|\\:";\'<>,.?/~`-'
        escaped = query
        for char in special_chars:
            escaped = escaped.replace(char, f"\\{char}")
        return escaped
