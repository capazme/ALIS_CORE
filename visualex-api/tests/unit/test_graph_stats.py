"""
Tests for KG Statistics Service
===============================

Tests cover:
- AC1: Node and edge counts
- AC2: Coverage metrics
- AC3: Source status
- AC4: Caching with TTL
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock

from visualex.graph.stats import (
    KGStats,
    NodeTypeCount,
    EdgeTypeCount,
    CoverageMetrics,
    SourceStatus,
    KGStatsService,
)


# =============================================================================
# Data Class Tests
# =============================================================================


class TestNodeTypeCount:
    """Tests for NodeTypeCount dataclass."""

    def test_creation(self):
        """Test basic creation."""
        count = NodeTypeCount(
            node_type="Norma",
            count=1500,
            label="Norme",
        )

        assert count.node_type == "Norma"
        assert count.count == 1500
        assert count.label == "Norme"

    def test_to_dict(self):
        """Test serialization."""
        count = NodeTypeCount(node_type="Articolo", count=500, label="Articoli")
        d = count.to_dict()

        assert d["type"] == "Articolo"
        assert d["count"] == 500
        assert d["label"] == "Articoli"


class TestEdgeTypeCount:
    """Tests for EdgeTypeCount dataclass."""

    def test_creation(self):
        """Test basic creation."""
        count = EdgeTypeCount(
            edge_type="cita",
            count=3000,
            label="Citazioni",
        )

        assert count.edge_type == "cita"
        assert count.count == 3000

    def test_to_dict(self):
        """Test serialization."""
        count = EdgeTypeCount(edge_type="sostituisce", count=100, label="Sostituzioni")
        d = count.to_dict()

        assert d["type"] == "sostituisce"
        assert d["count"] == 100


class TestCoverageMetrics:
    """Tests for CoverageMetrics dataclass."""

    def test_creation(self):
        """Test basic creation."""
        metrics = CoverageMetrics(
            total_articles=1000,
            articles_with_brocardi=800,
            articles_with_jurisprudence=300,
            libro_iv_coverage_percent=75.5,
        )

        assert metrics.total_articles == 1000
        assert metrics.articles_with_brocardi == 800

    def test_to_dict_with_percentages(self):
        """Test serialization calculates percentages."""
        metrics = CoverageMetrics(
            total_articles=1000,
            articles_with_brocardi=800,
            articles_with_jurisprudence=300,
            libro_iv_coverage_percent=75.5,
        )

        d = metrics.to_dict()

        assert d["total_articles"] == 1000
        assert d["brocardi_coverage_percent"] == 80.0  # 800/1000 * 100
        assert d["jurisprudence_coverage_percent"] == 30.0  # 300/1000 * 100
        assert d["libro_iv_coverage_percent"] == 75.5

    def test_to_dict_zero_articles(self):
        """Test serialization with zero articles."""
        metrics = CoverageMetrics(total_articles=0)
        d = metrics.to_dict()

        assert d["brocardi_coverage_percent"] == 0.0
        assert d["jurisprudence_coverage_percent"] == 0.0


class TestSourceStatus:
    """Tests for SourceStatus dataclass."""

    def test_creation(self):
        """Test basic creation."""
        status = SourceStatus(
            source_name="Normattiva",
            last_scrape=datetime(2024, 1, 15, 10, 30),
            items_scraped=5000,
            status="ok",
        )

        assert status.source_name == "Normattiva"
        assert status.items_scraped == 5000
        assert status.status == "ok"

    def test_to_dict(self):
        """Test serialization."""
        status = SourceStatus(
            source_name="Brocardi",
            last_scrape=datetime(2024, 1, 15),
            items_scraped=800,
            status="ok",
        )

        d = status.to_dict()

        assert d["source"] == "Brocardi"
        assert d["items_scraped"] == 800
        assert d["last_scrape"] is not None

    def test_to_dict_no_scrape(self):
        """Test serialization with no last scrape."""
        status = SourceStatus(source_name="EUR-Lex", status="unknown")
        d = status.to_dict()

        assert d["last_scrape"] is None
        assert d["status"] == "unknown"


class TestKGStats:
    """Tests for KGStats dataclass."""

    def test_creation(self):
        """Test basic creation."""
        stats = KGStats(
            generated_at=datetime.now(),
            total_nodes=10000,
            total_edges=25000,
        )

        assert stats.total_nodes == 10000
        assert stats.total_edges == 25000

    def test_to_dict_complete(self):
        """Test full serialization."""
        stats = KGStats(
            generated_at=datetime(2024, 1, 15, 10, 30),
            node_counts=[NodeTypeCount("Norma", 1000, "Norme")],
            edge_counts=[EdgeTypeCount("cita", 5000, "Citazioni")],
            coverage=CoverageMetrics(total_articles=500),
            sources=[SourceStatus("Normattiva", status="ok")],
            total_nodes=1000,
            total_edges=5000,
        )

        d = stats.to_dict()

        assert d["total_nodes"] == 1000
        assert d["total_edges"] == 5000
        assert len(d["node_counts"]) == 1
        assert len(d["edge_counts"]) == 1
        assert d["coverage"] is not None
        assert len(d["sources"]) == 1


# =============================================================================
# KGStatsService Tests
# =============================================================================


class TestKGStatsService:
    """Tests for KGStatsService."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = KGStatsService(self.mock_client)
        # Clear cache between tests
        self.service.invalidate_cache()

    @pytest.mark.asyncio
    async def test_get_node_counts(self):
        """Test getting node counts (AC1)."""
        self.mock_client.query.return_value = [
            {"label": "Norma", "cnt": 1500},
            {"label": "Articolo", "cnt": 500},
            {"label": "Comma", "cnt": 2000},
        ]

        counts = await self.service.get_node_counts()

        assert len(counts) == 3
        assert counts[0].node_type == "Norma"
        assert counts[0].count == 1500
        assert counts[0].label == "Norme"

    @pytest.mark.asyncio
    async def test_get_node_counts_fallback(self):
        """Test fallback when main query fails."""
        # First call fails, fallback queries succeed
        self.mock_client.query.side_effect = [
            Exception("DB error"),
            [{"cnt": 100}],  # Norma
            [{"cnt": 50}],   # Articolo
            [{"cnt": 200}],  # Comma
            [{"cnt": 10}],   # AttoGiudiziario
            [{"cnt": 5}],    # Dottrina
            [{"cnt": 20}],   # Versione
        ]

        counts = await self.service.get_node_counts()

        # Should get counts from fallback queries
        assert len(counts) > 0

    @pytest.mark.asyncio
    async def test_get_edge_counts(self):
        """Test getting edge counts (AC1)."""
        self.mock_client.query.return_value = [
            {"relationshipType": "cita", "cnt": 5000},
            {"relationshipType": "contiene", "cnt": 3000},
            {"relationshipType": "interpreta", "cnt": 500},
        ]

        counts = await self.service.get_edge_counts()

        assert len(counts) == 3
        assert counts[0].edge_type == "cita"
        assert counts[0].count == 5000
        assert counts[0].label == "Citazioni"

    @pytest.mark.asyncio
    async def test_get_coverage_metrics(self):
        """Test getting coverage metrics (AC2)."""
        self.mock_client.query.side_effect = [
            [{"cnt": 1000}],  # Total articles
            [{"cnt": 800}],   # With Brocardi
            [{"cnt": 300}],   # With jurisprudence
            [{"cnt": 500}],   # Libro IV
        ]

        metrics = await self.service.get_coverage_metrics()

        assert metrics.total_articles == 1000
        assert metrics.articles_with_brocardi == 800
        assert metrics.articles_with_jurisprudence == 300

    @pytest.mark.asyncio
    async def test_get_source_status(self):
        """Test getting source status (AC1)."""
        self.mock_client.query.side_effect = [
            [{"cnt": 5000, "last_update": "2024-01-15T10:30:00"}],  # Normattiva
            [{"cnt": 800}],  # Brocardi
            [{"cnt": 100}],  # EUR-Lex
        ]

        sources = await self.service.get_source_status()

        assert len(sources) == 3
        source_names = {s.source_name for s in sources}
        assert "Normattiva" in source_names
        assert "Brocardi" in source_names
        assert "EUR-Lex" in source_names

    @pytest.mark.asyncio
    async def test_get_stats_complete(self):
        """Test getting complete stats."""
        # Mock all queries
        self.mock_client.query.side_effect = [
            # Node counts
            [{"label": "Norma", "cnt": 1000}],
            # Edge counts
            [{"relationshipType": "cita", "cnt": 2000}],
            # Coverage - total articles
            [{"cnt": 500}],
            # Coverage - brocardi
            [{"cnt": 400}],
            # Coverage - jurisprudence
            [{"cnt": 150}],
            # Coverage - libro IV
            [{"cnt": 300}],
            # Sources - Normattiva
            [{"cnt": 1000, "last_update": None}],
            # Sources - Brocardi
            [{"cnt": 400}],
            # Sources - EUR-Lex
            [{"cnt": 50}],
        ]

        stats = await self.service.get_stats(use_cache=False)

        assert stats.total_nodes > 0
        assert stats.total_edges > 0
        assert len(stats.node_counts) > 0
        assert len(stats.edge_counts) > 0
        assert stats.coverage is not None

    @pytest.mark.asyncio
    async def test_caching(self):
        """Test cache is used (AC4)."""
        self.mock_client.query.return_value = [{"label": "Norma", "cnt": 100}]

        # First call - hits DB
        await self.service.get_stats(use_cache=True)
        first_call_count = self.mock_client.query.call_count

        # Second call - should use cache
        await self.service.get_stats(use_cache=True)
        second_call_count = self.mock_client.query.call_count

        assert second_call_count == first_call_count  # No new queries

    @pytest.mark.asyncio
    async def test_cache_invalidation(self):
        """Test cache can be invalidated."""
        self.mock_client.query.return_value = [{"label": "Norma", "cnt": 100}]

        # First call
        await self.service.get_stats(use_cache=True)

        # Invalidate cache
        self.service.invalidate_cache()

        # Should hit DB again
        await self.service.get_stats(use_cache=True)

        # Query should have been called again
        assert self.mock_client.query.call_count > 4  # Multiple queries

    @pytest.mark.asyncio
    async def test_cache_bypass(self):
        """Test cache bypass with use_cache=False."""
        self.mock_client.query.return_value = [{"label": "Norma", "cnt": 100}]

        # First call
        await self.service.get_stats(use_cache=True)
        first_count = self.mock_client.query.call_count

        # Force fresh fetch
        await self.service.get_stats(use_cache=False)

        assert self.mock_client.query.call_count > first_count

    @pytest.mark.asyncio
    async def test_query_failure_handled(self):
        """Test graceful handling of query failures."""
        self.mock_client.query.side_effect = Exception("DB error")

        stats = await self.service.get_stats(use_cache=False)

        # Should return stats object even if queries fail
        assert stats is not None
        assert stats.total_nodes == 0
        assert stats.total_edges == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestStatsServiceIntegration:
    """Integration-style tests."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = KGStatsService(self.mock_client)

    def test_node_labels_italian(self):
        """Test node labels are in Italian."""
        assert KGStatsService.NODE_LABELS["Norma"] == "Norme"
        assert KGStatsService.NODE_LABELS["Articolo"] == "Articoli"
        assert KGStatsService.NODE_LABELS["AttoGiudiziario"] == "Atti Giudiziari"

    def test_edge_labels_italian(self):
        """Test edge labels are in Italian."""
        assert KGStatsService.EDGE_LABELS["cita"] == "Citazioni"
        assert KGStatsService.EDGE_LABELS["interpreta"] == "Interpretazioni"

    def test_cache_ttl_one_hour(self):
        """Test cache TTL is 1 hour (AC4)."""
        assert KGStatsService.CACHE_TTL_SECONDS == 3600

    def test_cache_validity_check(self):
        """Test cache validity logic."""
        self.service._cache = KGStats(generated_at=datetime.now())
        self.service._cache_time = datetime.now() - timedelta(minutes=30)

        # Should be valid (30 min < 1 hour)
        assert self.service._is_cache_valid() is True

        # Expire cache
        self.service._cache_time = datetime.now() - timedelta(hours=2)

        # Should be invalid (2 hours > 1 hour)
        assert self.service._is_cache_valid() is False
