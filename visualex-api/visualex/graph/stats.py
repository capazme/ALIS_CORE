"""
Knowledge Graph Statistics Service
==================================

Provides statistics and coverage metrics for the Knowledge Graph.

Supports:
- Node counts by type (Norma, Articolo, Comma, etc.)
- Edge counts by relationship type
- Coverage metrics (Brocardi enrichment, jurisprudence links)
- Scraping timestamps per source
- Cached responses with TTL

Reference: Story 3-8: KG Coverage Dashboard
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Any, Optional

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient

__all__ = [
    "KGStats",
    "NodeTypeCount",
    "EdgeTypeCount",
    "CoverageMetrics",
    "SourceStatus",
    "KGStatsService",
]

logger = logging.getLogger(__name__)


@dataclass
class NodeTypeCount:
    """Count of nodes by type."""

    node_type: str
    count: int
    label: str  # Italian label for display

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.node_type,
            "count": self.count,
            "label": self.label,
        }


@dataclass
class EdgeTypeCount:
    """Count of edges by type."""

    edge_type: str
    count: int
    label: str  # Italian label for display

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.edge_type,
            "count": self.count,
            "label": self.label,
        }


@dataclass
class CoverageMetrics:
    """Coverage and enrichment metrics."""

    total_articles: int = 0
    articles_with_brocardi: int = 0
    articles_with_jurisprudence: int = 0
    libro_iv_coverage_percent: float = 0.0  # Libro IV C.C. coverage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        brocardi_percent = (
            (self.articles_with_brocardi / self.total_articles * 100)
            if self.total_articles > 0
            else 0.0
        )
        jurisprudence_percent = (
            (self.articles_with_jurisprudence / self.total_articles * 100)
            if self.total_articles > 0
            else 0.0
        )

        return {
            "total_articles": self.total_articles,
            "articles_with_brocardi": self.articles_with_brocardi,
            "articles_with_jurisprudence": self.articles_with_jurisprudence,
            "brocardi_coverage_percent": round(brocardi_percent, 1),
            "jurisprudence_coverage_percent": round(jurisprudence_percent, 1),
            "libro_iv_coverage_percent": round(self.libro_iv_coverage_percent, 1),
        }


@dataclass
class SourceStatus:
    """Status of a scraping source."""

    source_name: str
    last_scrape: Optional[datetime] = None
    items_scraped: int = 0
    status: str = "unknown"  # "ok", "stale", "error", "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source": self.source_name,
            "last_scrape": self.last_scrape.isoformat() if self.last_scrape else None,
            "items_scraped": self.items_scraped,
            "status": self.status,
        }


@dataclass
class KGStats:
    """Complete Knowledge Graph statistics."""

    generated_at: datetime
    node_counts: List[NodeTypeCount] = field(default_factory=list)
    edge_counts: List[EdgeTypeCount] = field(default_factory=list)
    coverage: Optional[CoverageMetrics] = None
    sources: List[SourceStatus] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "node_counts": [n.to_dict() for n in self.node_counts],
            "edge_counts": [e.to_dict() for e in self.edge_counts],
            "coverage": self.coverage.to_dict() if self.coverage else None,
            "sources": [s.to_dict() for s in self.sources],
        }


class KGStatsService:
    """
    Service for retrieving Knowledge Graph statistics.

    Provides:
    - Node counts by type (AC1)
    - Edge counts by relationship type (AC1)
    - Coverage metrics (AC2)
    - Source status and last scrape timestamps (AC1)
    - Cached responses with TTL (AC4)

    Example:
        service = KGStatsService(falkor_client)

        # Get all stats
        stats = await service.get_stats()

        # Get just node counts
        counts = await service.get_node_counts()
    """

    # Italian labels for node types
    NODE_LABELS = {
        "Norma": "Norme",
        "Articolo": "Articoli",
        "Comma": "Commi",
        "AttoGiudiziario": "Atti Giudiziari",
        "Dottrina": "Dottrina",
        "Versione": "Versioni",
        "Concetto": "Concetti",
    }

    # Italian labels for edge types
    EDGE_LABELS = {
        "cita": "Citazioni",
        "contiene": "Contenimento",
        "parte_di": "Appartenenza",
        "sostituisce": "Sostituzioni",
        "abroga_totalmente": "Abrogazioni Totali",
        "abroga_parzialmente": "Abrogazioni Parziali",
        "integra": "Integrazioni",
        "deroga_a": "Deroghe",
        "interpreta": "Interpretazioni",
        "ha_versione": "Versioni",
    }

    # Cache
    _cache: Optional[KGStats] = None
    _cache_time: Optional[datetime] = None
    _cache_ttl_seconds: int = 3600  # 1 hour

    def __init__(self, client: "FalkorDBClient"):
        """
        Initialize KGStatsService.

        Args:
            client: Connected FalkorDBClient instance
        """
        self.client = client
        logger.info("KGStatsService initialized")

    async def get_stats(self, use_cache: bool = True) -> KGStats:
        """
        Get complete Knowledge Graph statistics.

        Args:
            use_cache: Whether to use cached response if available

        Returns:
            KGStats with all metrics
        """
        # Check cache
        if use_cache and self._is_cache_valid():
            logger.debug("Returning cached stats")
            return self._cache

        # Fetch fresh stats
        import asyncio

        node_counts_task = self.get_node_counts()
        edge_counts_task = self.get_edge_counts()
        coverage_task = self.get_coverage_metrics()
        sources_task = self.get_source_status()

        results = await asyncio.gather(
            node_counts_task,
            edge_counts_task,
            coverage_task,
            sources_task,
            return_exceptions=True,
        )

        node_counts = results[0] if not isinstance(results[0], Exception) else []
        edge_counts = results[1] if not isinstance(results[1], Exception) else []
        coverage = results[2] if not isinstance(results[2], Exception) else None
        sources = results[3] if not isinstance(results[3], Exception) else []

        total_nodes = sum(n.count for n in node_counts)
        total_edges = sum(e.count for e in edge_counts)

        stats = KGStats(
            generated_at=datetime.now(),
            node_counts=node_counts,
            edge_counts=edge_counts,
            coverage=coverage,
            sources=sources,
            total_nodes=total_nodes,
            total_edges=total_edges,
        )

        # Update cache
        self._cache = stats
        self._cache_time = datetime.now()

        return stats

    async def get_node_counts(self) -> List[NodeTypeCount]:
        """
        Get count of nodes by type.

        Returns:
            List of NodeTypeCount for each node label
        """
        query = """
            CALL db.labels() YIELD label
            CALL {
                WITH label
                MATCH (n)
                WHERE label IN labels(n)
                RETURN count(n) AS cnt
            }
            RETURN label, cnt
            ORDER BY cnt DESC
        """

        counts = []

        try:
            result = await self.client.query(query, {})

            for record in result:
                label = record.get("label", "Unknown")
                count = record.get("cnt", 0)

                counts.append(NodeTypeCount(
                    node_type=label,
                    count=count,
                    label=self.NODE_LABELS.get(label, label),
                ))

        except Exception as e:
            logger.warning("Node count query failed: %s", e)
            # Fallback: query known types directly
            counts = await self._get_node_counts_fallback()

        return counts

    async def _get_node_counts_fallback(self) -> List[NodeTypeCount]:
        """Fallback method for node counts."""
        known_types = ["Norma", "Articolo", "Comma", "AttoGiudiziario", "Dottrina", "Versione"]
        counts = []

        for node_type in known_types:
            try:
                result = await self.client.query(
                    f"MATCH (n:{node_type}) RETURN count(n) AS cnt",
                    {}
                )
                if result:
                    count = result[0].get("cnt", 0)
                    counts.append(NodeTypeCount(
                        node_type=node_type,
                        count=count,
                        label=self.NODE_LABELS.get(node_type, node_type),
                    ))
            except Exception:
                pass

        return counts

    async def get_edge_counts(self) -> List[EdgeTypeCount]:
        """
        Get count of edges by relationship type.

        Returns:
            List of EdgeTypeCount for each relationship type
        """
        query = """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL {
                WITH relationshipType
                MATCH ()-[r]->()
                WHERE type(r) = relationshipType
                RETURN count(r) AS cnt
            }
            RETURN relationshipType, cnt
            ORDER BY cnt DESC
        """

        counts = []

        try:
            result = await self.client.query(query, {})

            for record in result:
                rel_type = record.get("relationshipType", "unknown")
                count = record.get("cnt", 0)

                counts.append(EdgeTypeCount(
                    edge_type=rel_type,
                    count=count,
                    label=self.EDGE_LABELS.get(rel_type, rel_type.replace("_", " ").title()),
                ))

        except Exception as e:
            logger.warning("Edge count query failed: %s", e)
            # Fallback: query known types directly
            counts = await self._get_edge_counts_fallback()

        return counts

    async def _get_edge_counts_fallback(self) -> List[EdgeTypeCount]:
        """Fallback method for edge counts."""
        known_types = ["cita", "contiene", "sostituisce", "abroga_totalmente", "interpreta"]
        counts = []

        for edge_type in known_types:
            try:
                result = await self.client.query(
                    f"MATCH ()-[r:{edge_type}]->() RETURN count(r) AS cnt",
                    {}
                )
                if result:
                    count = result[0].get("cnt", 0)
                    counts.append(EdgeTypeCount(
                        edge_type=edge_type,
                        count=count,
                        label=self.EDGE_LABELS.get(edge_type, edge_type),
                    ))
            except Exception:
                pass

        return counts

    async def get_coverage_metrics(self) -> CoverageMetrics:
        """
        Get coverage and enrichment metrics.

        Returns:
            CoverageMetrics with coverage percentages
        """
        metrics = CoverageMetrics()

        try:
            # Total articles
            total_result = await self.client.query(
                "MATCH (a:Norma) WHERE a.urn CONTAINS '~art' RETURN count(a) AS cnt",
                {}
            )
            if total_result:
                metrics.total_articles = total_result[0].get("cnt", 0)

            # Articles with Brocardi enrichment (has commentary/spiegazione)
            brocardi_result = await self.client.query(
                """
                MATCH (a:Norma)
                WHERE a.urn CONTAINS '~art'
                  AND (a.spiegazione IS NOT NULL OR a.ratio IS NOT NULL)
                RETURN count(a) AS cnt
                """,
                {}
            )
            if brocardi_result:
                metrics.articles_with_brocardi = brocardi_result[0].get("cnt", 0)

            # Articles with jurisprudence links
            jurisprudence_result = await self.client.query(
                """
                MATCH (a:Norma)<-[:interpreta]-(j:AttoGiudiziario)
                WHERE a.urn CONTAINS '~art'
                RETURN count(DISTINCT a) AS cnt
                """,
                {}
            )
            if jurisprudence_result:
                metrics.articles_with_jurisprudence = jurisprudence_result[0].get("cnt", 0)

            # Libro IV coverage (arts 1321-2059 Codice Civile)
            # Total articles in Libro IV
            libro_iv_total = 2059 - 1321 + 1  # ~739 articles

            libro_iv_result = await self.client.query(
                """
                MATCH (a:Norma)
                WHERE a.urn STARTS WITH 'urn:nir:stato:regio.decreto:1942-03-16;262~art'
                RETURN count(a) AS cnt
                """,
                {}
            )
            if libro_iv_result:
                libro_iv_count = libro_iv_result[0].get("cnt", 0)
                # This is approximate - actual coverage should filter by article range
                metrics.libro_iv_coverage_percent = min(
                    (libro_iv_count / libro_iv_total * 100) if libro_iv_total > 0 else 0,
                    100.0
                )

        except Exception as e:
            logger.warning("Coverage metrics query failed: %s", e)

        return metrics

    async def get_source_status(self) -> List[SourceStatus]:
        """
        Get status of scraping sources.

        Returns:
            List of SourceStatus for each source
        """
        sources = []

        try:
            # Check Normattiva norms
            normattiva_result = await self.client.query(
                """
                MATCH (n:Norma)
                WHERE n.fonte = 'normattiva' OR n.urn STARTS WITH 'urn:nir'
                RETURN count(n) AS cnt,
                       max(n.data_ultima_modifica) AS last_update
                """,
                {}
            )

            if normattiva_result:
                record = normattiva_result[0]
                last_update_str = record.get("last_update")
                last_update = self._parse_datetime(last_update_str) if last_update_str else None

                sources.append(SourceStatus(
                    source_name="Normattiva",
                    last_scrape=last_update,
                    items_scraped=record.get("cnt", 0),
                    status="ok" if last_update else "unknown",
                ))

            # Check Brocardi content
            brocardi_result = await self.client.query(
                """
                MATCH (n:Norma)
                WHERE n.spiegazione IS NOT NULL OR n.ratio IS NOT NULL
                RETURN count(n) AS cnt
                """,
                {}
            )

            if brocardi_result:
                sources.append(SourceStatus(
                    source_name="Brocardi",
                    items_scraped=brocardi_result[0].get("cnt", 0),
                    status="ok" if brocardi_result[0].get("cnt", 0) > 0 else "unknown",
                ))

            # Check EUR-Lex content
            eurlex_result = await self.client.query(
                """
                MATCH (n:Norma)
                WHERE n.fonte = 'eurlex' OR n.urn CONTAINS ':unione.europea:'
                RETURN count(n) AS cnt
                """,
                {}
            )

            if eurlex_result:
                sources.append(SourceStatus(
                    source_name="EUR-Lex",
                    items_scraped=eurlex_result[0].get("cnt", 0),
                    status="ok" if eurlex_result[0].get("cnt", 0) > 0 else "unknown",
                ))

        except Exception as e:
            logger.warning("Source status query failed: %s", e)

        return sources

    def invalidate_cache(self) -> None:
        """Invalidate the stats cache."""
        self._cache = None
        self._cache_time = None
        logger.debug("Stats cache invalidated")

    def _is_cache_valid(self) -> bool:
        """Check if cache is still valid."""
        if self._cache is None or self._cache_time is None:
            return False

        elapsed = (datetime.now() - self._cache_time).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                try:
                    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    return None
        return None
