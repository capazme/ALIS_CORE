"""
Tests for Hybrid Search Service
===============================

Tests cover:
- AC1: Keyword search in norm text, titles, commentary
- AC2: Semantic search with Qdrant (mocked)
- AC3: URN pattern recognition
- AC4: Hybrid result merging with ranking
- AC5: Temporal filtering
- AC6: Performance (<500ms)
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, AsyncMock, patch

from visualex.graph.search import (
    SearchConfig,
    SearchRequest,
    SearchResultItem,
    SearchResponse,
    HybridSearchService,
    SearchMode,
    URN_PATTERNS,
)


# =============================================================================
# SearchConfig Tests
# =============================================================================


class TestSearchConfig:
    """Tests for SearchConfig dataclass."""

    def test_default_values(self):
        """Test default configuration."""
        config = SearchConfig()

        assert config.fulltext_weight == 0.4
        assert config.semantic_weight == 0.4
        assert config.authority_weight == 0.2
        assert config.max_results == 50

    def test_weights_must_sum_to_one(self):
        """Test weight validation."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            SearchConfig(
                fulltext_weight=0.5,
                semantic_weight=0.5,
                authority_weight=0.5,
            )

    def test_custom_weights(self):
        """Test custom weight configuration."""
        config = SearchConfig(
            fulltext_weight=0.6,
            semantic_weight=0.3,
            authority_weight=0.1,
        )
        assert config.fulltext_weight == 0.6


# =============================================================================
# SearchRequest Tests
# =============================================================================


class TestSearchRequest:
    """Tests for SearchRequest dataclass."""

    def test_basic_request(self):
        """Test basic search request."""
        request = SearchRequest(query="risoluzione contratto")

        assert request.query == "risoluzione contratto"
        assert request.mode == SearchMode.HYBRID
        assert request.limit == 20

    def test_empty_query_rejected(self):
        """Test empty query is rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            SearchRequest(query="")

        with pytest.raises(ValueError, match="cannot be empty"):
            SearchRequest(query="   ")

    def test_limit_validation(self):
        """Test limit bounds."""
        with pytest.raises(ValueError, match="between 1 and 100"):
            SearchRequest(query="test", limit=0)

        with pytest.raises(ValueError, match="between 1 and 100"):
            SearchRequest(query="test", limit=101)

    def test_full_request(self):
        """Test request with all options."""
        request = SearchRequest(
            query="inadempimento",
            mode=SearchMode.KEYWORD,
            source_types=["norm", "commentary"],
            as_of_date=date(2023, 1, 1),
            min_authority=0.5,
            expert_type="literal",
            limit=50,
            offset=10,
        )

        assert request.mode == SearchMode.KEYWORD
        assert request.source_types == ["norm", "commentary"]
        assert request.as_of_date == date(2023, 1, 1)


# =============================================================================
# SearchResultItem Tests
# =============================================================================


class TestSearchResultItem:
    """Tests for SearchResultItem dataclass."""

    def test_creation(self):
        """Test result item creation."""
        item = SearchResultItem(
            urn="urn:nir:stato:legge:2020;178~art1",
            title="Art. 1 - Disposizioni generali",
            snippet="Il contratto si risolve...",
            source_type="norm",
            score=0.85,
            authority=1.0,
        )

        assert item.urn == "urn:nir:stato:legge:2020;178~art1"
        assert item.score == 0.85
        assert item.match_type == "keyword"

    def test_to_dict(self):
        """Test serialization."""
        item = SearchResultItem(
            urn="urn:test",
            title="Test",
            snippet="...",
            source_type="norm",
            score=0.8567,
            authority=1.0,
            fulltext_score=0.9,
            semantic_score=0.7,
            vigenza_dal="2020-01-01",
        )

        d = item.to_dict()

        assert d["urn"] == "urn:test"
        assert d["score"] == 0.8567
        assert d["fulltext_score"] == 0.9
        assert d["vigenza_dal"] == "2020-01-01"


# =============================================================================
# URN Pattern Matching Tests (AC3)
# =============================================================================


class TestURNPatternMatching:
    """Tests for URN pattern recognition (AC3)."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock(return_value=MagicMock(result_set=[]))
        self.service = HybridSearchService(self.mock_client)

    def test_art_codice_civile(self):
        """Test 'art. 1453 c.c.' pattern."""
        urn = self.service._match_urn_pattern("art. 1453 c.c.")
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

    def test_articolo_codice_civile(self):
        """Test 'articolo 2043 del codice civile' pattern."""
        urn = self.service._match_urn_pattern("articolo 2043 del codice civile")
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art2043"

    def test_art_codice_penale(self):
        """Test 'art. 575 c.p.' pattern."""
        urn = self.service._match_urn_pattern("art. 575 c.p.")
        assert urn == "urn:nir:stato:regio.decreto:1930-10-19;1398~art575"

    def test_decreto_legislativo(self):
        """Test 'd.lgs. 231/2001' pattern."""
        urn = self.service._match_urn_pattern("d.lgs. 231/2001")
        assert urn == "urn:nir:stato:decreto.legislativo:2001;231"

    def test_legge(self):
        """Test 'l. 241/1990' pattern."""
        urn = self.service._match_urn_pattern("l. 241/1990")
        assert urn == "urn:nir:stato:legge:1990;241"

    def test_legge_full(self):
        """Test 'legge 241/1990' pattern."""
        urn = self.service._match_urn_pattern("legge 241/1990")
        assert urn == "urn:nir:stato:legge:1990;241"

    def test_dpr(self):
        """Test 'd.p.r. 445/2000' pattern."""
        urn = self.service._match_urn_pattern("d.p.r. 445/2000")
        assert urn == "urn:nir:stato:decreto.presidente.repubblica:2000;445"

    def test_art_with_bis(self):
        """Test 'art. 16bis c.c.' pattern."""
        urn = self.service._match_urn_pattern("art. 16bis c.c.")
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art16bis"

    def test_no_match(self):
        """Test non-matching query."""
        urn = self.service._match_urn_pattern("risoluzione del contratto")
        assert urn is None

    def test_case_insensitive(self):
        """Test case insensitivity."""
        urn = self.service._match_urn_pattern("ART. 1453 C.C.")
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"


# =============================================================================
# HybridSearchService Tests
# =============================================================================


class TestHybridSearchService:
    """Tests for HybridSearchService."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = HybridSearchService(self.mock_client)

    @pytest.mark.asyncio
    async def test_exact_urn_search(self):
        """Test exact URN pattern search (AC3)."""
        # Mock successful norm lookup
        self.mock_client.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    [
                        "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
                        "Risolubilita' del contratto per inadempimento",
                        "Nei contratti con prestazioni corrispettive...",
                        "2020-01-01",
                        "vigente",
                    ]
                ]
            )
        )

        request = SearchRequest(query="art. 1453 c.c.")
        response = await self.service.search(request)

        assert response.exact_match is True
        assert len(response.results) == 1
        assert response.results[0].urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        assert response.results[0].score == 1.0
        assert response.results[0].match_type == "exact"

    @pytest.mark.asyncio
    async def test_keyword_search(self):
        """Test keyword search mode (AC1)."""
        # Mock fulltext search
        self.mock_client.query = AsyncMock(
            return_value=MagicMock(
                result_set=[
                    [
                        "urn:nir:stato:legge:2020;178~art1",
                        "Disposizioni generali",
                        "Testo con risoluzione del contratto...",
                        "norm",
                        0.85,
                        "2020-01-01",
                        "vigente",
                    ],
                    [
                        "urn:nir:stato:legge:2019;145~art10",
                        "Risoluzione anticipata",
                        "La risoluzione anticipata del contratto...",
                        "norm",
                        0.72,
                        "2019-06-15",
                        "vigente",
                    ],
                ]
            )
        )

        request = SearchRequest(query="risoluzione contratto", mode=SearchMode.KEYWORD)
        response = await self.service.search(request)

        assert response.mode == "keyword"
        assert len(response.results) >= 1
        # Results should be sorted by score
        if len(response.results) > 1:
            assert response.results[0].score >= response.results[1].score

    @pytest.mark.asyncio
    async def test_search_with_source_type_filter(self):
        """Test filtering by source type."""
        self.mock_client.query = AsyncMock(
            return_value=MagicMock(result_set=[])
        )

        request = SearchRequest(
            query="test",
            source_types=["norm"],
        )
        response = await self.service.search(request)

        assert response is not None
        # Verify query was called
        self.mock_client.query.assert_called()

    @pytest.mark.asyncio
    async def test_temporal_filter(self):
        """Test temporal filtering (AC5)."""
        # Create results with different dates
        item_2020 = SearchResultItem(
            urn="urn:2020",
            title="Article 2020",
            snippet="...",
            source_type="norm",
            score=0.9,
            authority=1.0,
            vigenza_dal="2020-01-01",
        )
        item_2025 = SearchResultItem(
            urn="urn:2025",
            title="Article 2025",
            snippet="...",
            source_type="norm",
            score=0.8,
            authority=1.0,
            vigenza_dal="2025-01-01",
        )

        results = [item_2020, item_2025]

        # Filter for date in 2023
        filtered = self.service._apply_temporal_filter(results, date(2023, 6, 1))

        # Should only include 2020 article (in force)
        # 2025 article not yet in force
        assert len(filtered) == 1
        assert filtered[0].urn == "urn:2020"

    def test_result_merging(self):
        """Test hybrid result merging (AC4)."""
        fulltext_results = [
            SearchResultItem(
                urn="urn:1",
                title="Title 1",
                snippet="...",
                source_type="norm",
                score=0.9,
                authority=1.0,
                fulltext_score=0.9,
            ),
            SearchResultItem(
                urn="urn:2",
                title="Title 2",
                snippet="...",
                source_type="norm",
                score=0.7,
                authority=1.0,
                fulltext_score=0.7,
            ),
        ]

        semantic_results = [
            SearchResultItem(
                urn="urn:1",  # Same as fulltext
                title="Title 1",
                snippet="...",
                source_type="norm",
                score=0.8,
                authority=1.0,
                semantic_score=0.8,
            ),
            SearchResultItem(
                urn="urn:3",  # New from semantic
                title="Title 3",
                snippet="...",
                source_type="norm",
                score=0.75,
                authority=0.8,
                semantic_score=0.75,
            ),
        ]

        merged = self.service._merge_results(fulltext_results, semantic_results, limit=10)

        # Should have 3 unique URNs
        urns = [r.urn for r in merged]
        assert len(urns) == 3
        assert "urn:1" in urns
        assert "urn:2" in urns
        assert "urn:3" in urns

        # urn:1 should be hybrid with combined score
        urn1_result = next(r for r in merged if r.urn == "urn:1")
        assert urn1_result.match_type == "hybrid"
        assert urn1_result.fulltext_score == 0.9
        assert urn1_result.semantic_score == 0.8

    def test_snippet_creation(self):
        """Test snippet creation with highlight."""
        text = "Il contratto si risolve per inadempimento quando una delle parti non adempie."
        snippet = self.service._create_snippet(text, "risolve")

        assert "risolve" in snippet
        assert len(snippet) <= 250  # max_length + ellipsis

    def test_snippet_query_not_found(self):
        """Test snippet when query not in text."""
        text = "Testo senza la parola cercata."
        snippet = self.service._create_snippet(text, "nonexistent")

        # Should return start of text
        assert snippet.startswith("Testo")

    def test_fulltext_query_escaping(self):
        """Test special character escaping."""
        escaped = self.service._escape_fulltext_query("art. 1453 (c.c.)")

        # Should escape special chars
        assert "\\." in escaped
        assert "\\(" in escaped
        assert "\\)" in escaped


# =============================================================================
# SearchResponse Tests
# =============================================================================


class TestSearchResponse:
    """Tests for SearchResponse dataclass."""

    def test_response_creation(self):
        """Test response creation."""
        results = [
            SearchResultItem(
                urn="urn:test",
                title="Test",
                snippet="...",
                source_type="norm",
                score=0.9,
                authority=1.0,
            )
        ]

        response = SearchResponse(
            results=results,
            total_count=1,
            query="test",
            mode="hybrid",
            elapsed_ms=125.5,
        )

        assert response.total_count == 1
        assert response.elapsed_ms == 125.5

    def test_to_dict(self):
        """Test response serialization."""
        results = [
            SearchResultItem(
                urn="urn:test",
                title="Test",
                snippet="...",
                source_type="norm",
                score=0.9,
                authority=1.0,
            )
        ]

        response = SearchResponse(
            results=results,
            total_count=1,
            query="test",
            mode="hybrid",
            elapsed_ms=125.567,
            fulltext_count=1,
            semantic_count=0,
        )

        d = response.to_dict()

        assert d["total_count"] == 1
        assert d["elapsed_ms"] == 125.57  # Rounded
        assert d["meta"]["fulltext_count"] == 1
        assert len(d["results"]) == 1


# =============================================================================
# Integration Tests (with mocked services)
# =============================================================================


class TestSearchIntegration:
    """Integration-style tests with mocked services."""

    def setup_method(self):
        """Create service with all mocked dependencies."""
        self.mock_falkor = MagicMock()
        self.mock_falkor.query = AsyncMock(return_value=MagicMock(result_set=[]))

        self.mock_qdrant = MagicMock()
        self.mock_qdrant.search = MagicMock(return_value=[])

        self.mock_bridge = MagicMock()
        self.mock_bridge.get_mappings_for_chunk = AsyncMock(return_value=[])

        self.mock_embedder = MagicMock()
        self.mock_embedder.embed = MagicMock(
            return_value=MagicMock(embedding=[0.1] * 1024)
        )

    @pytest.mark.asyncio
    async def test_hybrid_mode_calls_both_searches(self):
        """Test that hybrid mode queries both FalkorDB and Qdrant."""
        service = HybridSearchService(
            falkor_client=self.mock_falkor,
            qdrant_manager=self.mock_qdrant,
            bridge_manager=self.mock_bridge,
            embedder=self.mock_embedder,
        )

        request = SearchRequest(query="inadempimento", mode=SearchMode.HYBRID)
        await service.search(request)

        # Both should be called
        self.mock_falkor.query.assert_called()
        self.mock_qdrant.search.assert_called()

    @pytest.mark.asyncio
    async def test_keyword_mode_skips_semantic(self):
        """Test that keyword mode skips Qdrant."""
        service = HybridSearchService(
            falkor_client=self.mock_falkor,
            qdrant_manager=self.mock_qdrant,
            bridge_manager=self.mock_bridge,
            embedder=self.mock_embedder,
        )

        request = SearchRequest(query="test", mode=SearchMode.KEYWORD)
        await service.search(request)

        # Only FalkorDB should be called
        self.mock_falkor.query.assert_called()
        self.mock_qdrant.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_semantic_disabled_when_missing_deps(self):
        """Test semantic search disabled when dependencies missing."""
        service = HybridSearchService(
            falkor_client=self.mock_falkor,
            # No qdrant, bridge, embedder
        )

        assert service._semantic_enabled is False

        request = SearchRequest(query="test", mode=SearchMode.SEMANTIC)
        response = await service.search(request)

        # Should complete but with no semantic results
        assert response.semantic_count == 0

    @pytest.mark.asyncio
    async def test_performance_under_500ms(self):
        """Test search completes under 500ms (AC6)."""
        import time

        service = HybridSearchService(self.mock_falkor)

        request = SearchRequest(query="test")

        start = time.perf_counter()
        response = await service.search(request)
        elapsed = (time.perf_counter() - start) * 1000

        # Should complete quickly with mocked dependencies
        assert elapsed < 500
        assert response.elapsed_ms < 500
