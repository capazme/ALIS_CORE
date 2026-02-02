"""
Tests for Cross-Reference Service
=================================

Tests cover:
- AC1: Outgoing references (norms this article cites)
- AC2: Incoming references (norms citing this article)
- AC3: Modifying legislation
- AC4: Jurisprudence
- AC5: Grouped response
- AC6: Pagination
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from visualex.graph.neighbors import (
    ReferenceType,
    CrossReference,
    CrossReferenceGroup,
    CrossReferenceResponse,
    CrossReferenceService,
)


# =============================================================================
# CrossReference Tests
# =============================================================================


class TestCrossReference:
    """Tests for CrossReference dataclass."""

    def test_creation(self):
        """Test basic creation."""
        ref = CrossReference(
            urn="urn:nir:stato:legge:2020;178~art1",
            title="Art. 1 - Disposizioni generali",
            relationship="CITA",
            source_type="norm",
        )

        assert ref.urn == "urn:nir:stato:legge:2020;178~art1"
        assert ref.relationship == "CITA"
        assert ref.source_type == "norm"

    def test_with_metadata(self):
        """Test creation with optional metadata."""
        ref = CrossReference(
            urn="urn:test",
            title="Test",
            relationship="MODIFICA",
            source_type="norm",
            snippet="...modifica l'articolo...",
            date="2023-01-15",
            authority="Parlamento",
        )

        assert ref.snippet == "...modifica l'articolo..."
        assert ref.date == "2023-01-15"
        assert ref.authority == "Parlamento"

    def test_to_dict(self):
        """Test serialization."""
        ref = CrossReference(
            urn="urn:test",
            title="Test",
            relationship="CITA",
            source_type="norm",
            snippet="context",
        )

        d = ref.to_dict()

        assert d["urn"] == "urn:test"
        assert d["snippet"] == "context"
        assert "date" not in d  # Optional field not set

    def test_to_dict_excludes_none(self):
        """Test that None values are excluded from dict."""
        ref = CrossReference(
            urn="urn:test",
            title="Test",
            relationship="CITA",
            source_type="norm",
        )

        d = ref.to_dict()

        assert "snippet" not in d
        assert "date" not in d
        assert "authority" not in d


# =============================================================================
# CrossReferenceGroup Tests
# =============================================================================


class TestCrossReferenceGroup:
    """Tests for CrossReferenceGroup dataclass."""

    def test_creation(self):
        """Test group creation."""
        refs = [
            CrossReference(
                urn=f"urn:test:{i}",
                title=f"Test {i}",
                relationship="CITA",
                source_type="norm",
            )
            for i in range(3)
        ]

        group = CrossReferenceGroup(
            reference_type=ReferenceType.OUTGOING,
            label="Riferimenti in uscita",
            references=refs,
            total_count=10,
            has_more=True,
        )

        assert group.reference_type == ReferenceType.OUTGOING
        assert len(group.references) == 3
        assert group.total_count == 10
        assert group.has_more is True

    def test_to_dict(self):
        """Test group serialization."""
        group = CrossReferenceGroup(
            reference_type=ReferenceType.INCOMING,
            label="Riferimenti in entrata",
            references=[
                CrossReference(
                    urn="urn:test",
                    title="Test",
                    relationship="CITA",
                    source_type="norm",
                )
            ],
            total_count=1,
        )

        d = group.to_dict()

        assert d["type"] == "incoming"
        assert d["label"] == "Riferimenti in entrata"
        assert len(d["references"]) == 1
        assert d["total_count"] == 1


# =============================================================================
# CrossReferenceResponse Tests
# =============================================================================


class TestCrossReferenceResponse:
    """Tests for CrossReferenceResponse dataclass."""

    def test_creation(self):
        """Test response creation."""
        groups = [
            CrossReferenceGroup(
                reference_type=ReferenceType.OUTGOING,
                label="Out",
                references=[],
                total_count=5,
            ),
            CrossReferenceGroup(
                reference_type=ReferenceType.INCOMING,
                label="In",
                references=[],
                total_count=3,
            ),
        ]

        response = CrossReferenceResponse(
            urn="urn:test:article",
            groups=groups,
            total_references=8,
        )

        assert response.urn == "urn:test:article"
        assert len(response.groups) == 2
        assert response.total_references == 8

    def test_to_dict(self):
        """Test response serialization."""
        response = CrossReferenceResponse(
            urn="urn:test",
            groups=[],
            total_references=0,
        )

        d = response.to_dict()

        assert d["urn"] == "urn:test"
        assert d["groups"] == []
        assert d["total_references"] == 0


# =============================================================================
# ReferenceType Tests
# =============================================================================


class TestReferenceType:
    """Tests for ReferenceType enum."""

    def test_values(self):
        """Test enum values."""
        assert ReferenceType.OUTGOING.value == "outgoing"
        assert ReferenceType.INCOMING.value == "incoming"
        assert ReferenceType.MODIFIED_BY.value == "modified_by"
        assert ReferenceType.JURISPRUDENCE.value == "jurisprudence"


# =============================================================================
# CrossReferenceService Tests
# =============================================================================


class TestCrossReferenceService:
    """Tests for CrossReferenceService."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = CrossReferenceService(self.mock_client)

    @pytest.mark.asyncio
    async def test_get_outgoing_references(self):
        """Test outgoing references query (AC1)."""
        # Mock count query
        count_result = MagicMock()
        count_result.result_set = [[5]]

        # Mock references query
        refs_result = MagicMock()
        refs_result.result_set = [
            ["urn:nir:stato:legge:2020;1~art1", "Art. 1", "CITA", "contesto"],
            ["urn:nir:stato:legge:2020;1~art2", "Art. 2", "CITA", None],
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_outgoing_references(
            urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )

        assert group.reference_type == ReferenceType.OUTGOING
        assert group.total_count == 5
        assert len(group.references) == 2
        assert group.references[0].urn == "urn:nir:stato:legge:2020;1~art1"
        assert group.references[0].relationship == "CITA"

    @pytest.mark.asyncio
    async def test_get_incoming_references(self):
        """Test incoming references query (AC2)."""
        count_result = MagicMock()
        count_result.result_set = [[3]]

        refs_result = MagicMock()
        refs_result.result_set = [
            ["urn:citing:1", "Citing Article 1", "CITA", None],
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_incoming_references(urn="urn:test")

        assert group.reference_type == ReferenceType.INCOMING
        assert group.total_count == 3
        assert len(group.references) == 1
        assert group.references[0].urn == "urn:citing:1"

    @pytest.mark.asyncio
    async def test_get_modifications(self):
        """Test modifications query (AC3)."""
        count_result = MagicMock()
        count_result.result_set = [[2]]

        refs_result = MagicMock()
        refs_result.result_set = [
            ["urn:modifying:1", "Legge modificante", "MODIFICA", "2023-01-01", "modifica comma 1"],
            ["urn:modifying:2", "D.Lgs. abrogante", "ABROGA", "2024-06-15", None],
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_modifications(urn="urn:test")

        assert group.reference_type == ReferenceType.MODIFIED_BY
        assert group.total_count == 2
        assert len(group.references) == 2
        assert group.references[0].relationship == "MODIFICA"
        assert group.references[0].date == "2023-01-01"
        assert group.references[1].relationship == "ABROGA"

    @pytest.mark.asyncio
    async def test_get_jurisprudence(self):
        """Test jurisprudence query (AC4)."""
        count_result = MagicMock()
        count_result.result_set = [[4]]

        refs_result = MagicMock()
        refs_result.result_set = [
            [
                "massima_123",
                "La Corte stabilisce che il contratto si risolve automaticamente...",
                "INTERPRETA",
                "Cassazione Civile",
                "2023-05-20",
                "La Corte stabilisce che il contratto si risolve automaticamente...",
            ],
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_jurisprudence(urn="urn:test")

        assert group.reference_type == ReferenceType.JURISPRUDENCE
        assert group.total_count == 4
        assert len(group.references) == 1
        assert group.references[0].source_type == "jurisprudence"
        assert group.references[0].authority == "Cassazione Civile"

    @pytest.mark.asyncio
    async def test_get_cross_references_grouped(self):
        """Test grouped response (AC5)."""
        # Mock all queries
        empty_count = MagicMock()
        empty_count.result_set = [[0]]

        empty_refs = MagicMock()
        empty_refs.result_set = []

        outgoing_count = MagicMock()
        outgoing_count.result_set = [[2]]

        outgoing_refs = MagicMock()
        outgoing_refs.result_set = [
            ["urn:out:1", "Out 1", "CITA", None],
            ["urn:out:2", "Out 2", "CITA", None],
        ]

        # Set up side effects for parallel queries
        # Each query method calls count then refs
        self.mock_client.query = AsyncMock(
            side_effect=[
                outgoing_count, outgoing_refs,  # outgoing
                empty_count, empty_refs,  # incoming
                empty_count, empty_refs,  # modifications
                empty_count, empty_refs,  # jurisprudence
            ]
        )

        response = await self.service.get_cross_references(urn="urn:test")

        assert response.urn == "urn:test"
        # Only non-empty groups should be included
        non_empty_groups = [g for g in response.groups if g.total_count > 0]
        assert len(non_empty_groups) >= 1
        assert response.total_references >= 2

    @pytest.mark.asyncio
    async def test_pagination(self):
        """Test pagination support (AC6)."""
        count_result = MagicMock()
        count_result.result_set = [[50]]  # 50 total

        refs_result = MagicMock()
        refs_result.result_set = [
            [f"urn:ref:{i}", f"Ref {i}", "CITA", None]
            for i in range(10)  # Return 10
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_outgoing_references(
            urn="urn:test",
            limit=10,
            offset=20,
        )

        assert group.total_count == 50
        assert len(group.references) == 10
        assert group.has_more is True  # 50 > 20 + 10

        # Verify query was called with pagination params
        calls = self.mock_client.query.call_args_list
        assert calls[1][0][1]["limit"] == 10
        assert calls[1][0][1]["offset"] == 20

    @pytest.mark.asyncio
    async def test_has_more_false_when_complete(self):
        """Test has_more is False when all results returned."""
        count_result = MagicMock()
        count_result.result_set = [[3]]

        refs_result = MagicMock()
        refs_result.result_set = [
            ["urn:1", "Ref 1", "CITA", None],
            ["urn:2", "Ref 2", "CITA", None],
            ["urn:3", "Ref 3", "CITA", None],
        ]

        self.mock_client.query = AsyncMock(side_effect=[count_result, refs_result])

        group = await self.service.get_outgoing_references(
            urn="urn:test",
            limit=10,
            offset=0,
        )

        assert group.total_count == 3
        assert len(group.references) == 3
        assert group.has_more is False  # 3 <= 0 + 3

    @pytest.mark.asyncio
    async def test_query_failure_handled(self):
        """Test graceful handling of query failures."""
        self.mock_client.query = AsyncMock(side_effect=Exception("DB error"))

        group = await self.service.get_outgoing_references(urn="urn:test")

        # Should return empty group, not raise
        assert group.total_count == 0
        assert len(group.references) == 0

    @pytest.mark.asyncio
    async def test_get_all_neighbors(self):
        """Test getting all neighbors for graph visualization."""
        result = MagicMock()
        result.result_set = [
            ["urn:neighbor:1", "Neighbor 1", "CITA", "Norma"],
            ["massima_1", "Massima", "INTERPRETA", "AttoGiudiziario"],
            ["dottrina_1", "Commento", "COMMENTA", "Dottrina"],
        ]

        self.mock_client.query = AsyncMock(return_value=result)

        refs = await self.service.get_all_neighbors(urn="urn:test")

        assert len(refs) == 3
        assert refs[0].source_type == "norm"
        assert refs[1].source_type == "jurisprudence"
        assert refs[2].source_type == "commentary"


# =============================================================================
# Integration Tests
# =============================================================================


class TestCrossReferenceIntegration:
    """Integration-style tests."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = CrossReferenceService(self.mock_client)

    @pytest.mark.asyncio
    async def test_labels_italian(self):
        """Test that labels are in Italian."""
        assert CrossReferenceService.LABELS[ReferenceType.OUTGOING] == "Riferimenti in uscita"
        assert CrossReferenceService.LABELS[ReferenceType.INCOMING] == "Riferimenti in entrata"
        assert CrossReferenceService.LABELS[ReferenceType.MODIFIED_BY] == "Modificato da"
        assert CrossReferenceService.LABELS[ReferenceType.JURISPRUDENCE] == "Giurisprudenza"

    def test_modification_edges_complete(self):
        """Test all modification edge types are included."""
        from visualex.graph.schema import EdgeType

        modification_edges = CrossReferenceService.MODIFICATION_EDGES

        assert EdgeType.SOSTITUISCE in modification_edges
        assert EdgeType.ABROGA_TOTALMENTE in modification_edges
        assert EdgeType.ABROGA_PARZIALMENTE in modification_edges
        assert EdgeType.INTEGRA in modification_edges
        assert EdgeType.SOSPENDE in modification_edges
        assert EdgeType.DEROGA_A in modification_edges
        assert EdgeType.PROROGA in modification_edges
        assert EdgeType.INSERISCE in modification_edges
