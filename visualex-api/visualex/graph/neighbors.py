"""
Cross-Reference Service
=======================

Retrieves graph neighbors (cross-references) for legal norms.

Provides grouped relationships:
- Outgoing references (norms this article cites)
- Incoming references (norms citing this article)
- Modifications (legislation modifying this article)
- Jurisprudence (case law interpreting this article)

Reference: Story 3-5: Cross-References Panel
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Any, Optional

from visualex.graph.schema import EdgeType, NodeType

__all__ = [
    "ReferenceType",
    "CrossReference",
    "CrossReferenceGroup",
    "CrossReferenceResponse",
    "CrossReferenceService",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class ReferenceType(str, Enum):
    """Type of cross-reference."""

    OUTGOING = "outgoing"  # Norms this article cites
    INCOMING = "incoming"  # Norms citing this article
    MODIFIED_BY = "modified_by"  # Legislation modifying this article
    JURISPRUDENCE = "jurisprudence"  # Case law interpreting this article


@dataclass
class CrossReference:
    """A single cross-reference."""

    urn: str  # URN or node_id of the referenced item
    title: str  # Rubrica or title
    relationship: str  # Edge type (CITA, MODIFICA, INTERPRETA, etc.)
    source_type: str  # norm, jurisprudence, etc.

    # Optional metadata
    snippet: Optional[str] = None  # Context snippet
    date: Optional[str] = None  # Date of modification/judgment
    authority: Optional[str] = None  # Court/issuing authority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "urn": self.urn,
            "title": self.title,
            "relationship": self.relationship,
            "source_type": self.source_type,
        }
        if self.snippet:
            result["snippet"] = self.snippet
        if self.date:
            result["date"] = self.date
        if self.authority:
            result["authority"] = self.authority
        return result


@dataclass
class CrossReferenceGroup:
    """Grouped cross-references of one type."""

    reference_type: ReferenceType
    label: str  # Display label (e.g., "Riferimenti in uscita")
    references: List[CrossReference]
    total_count: int
    has_more: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.reference_type.value,
            "label": self.label,
            "references": [r.to_dict() for r in self.references],
            "total_count": self.total_count,
            "has_more": self.has_more,
        }


@dataclass
class CrossReferenceResponse:
    """Complete cross-reference response for an article."""

    urn: str
    groups: List[CrossReferenceGroup]
    total_references: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urn": self.urn,
            "groups": [g.to_dict() for g in self.groups],
            "total_references": self.total_references,
        }


# =============================================================================
# Cross-Reference Service
# =============================================================================


class CrossReferenceService:
    """
    Service for retrieving cross-references (graph neighbors).

    Provides:
    - Outgoing references (AC1): norms this article cites
    - Incoming references (AC2): norms citing this article
    - Modifications (AC3): legislation modifying this article
    - Jurisprudence (AC4): case law interpreting this article
    - Grouped response (AC5): results grouped by type
    - Pagination (AC6): offset/limit support

    Example:
        service = CrossReferenceService(falkor_client)

        # Get all cross-references
        response = await service.get_cross_references(
            urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )

        # Get specific type with pagination
        outgoing = await service.get_outgoing_references(
            urn="urn:...",
            limit=20,
            offset=0
        )
    """

    # Edge types for each reference category
    MODIFICATION_EDGES = [
        EdgeType.SOSTITUISCE,
        EdgeType.ABROGA_TOTALMENTE,
        EdgeType.ABROGA_PARZIALMENTE,
        EdgeType.INTEGRA,
        EdgeType.SOSPENDE,
        EdgeType.DEROGA_A,
        EdgeType.PROROGA,
        EdgeType.INSERISCE,
    ]

    # Labels for display
    LABELS = {
        ReferenceType.OUTGOING: "Riferimenti in uscita",
        ReferenceType.INCOMING: "Riferimenti in entrata",
        ReferenceType.MODIFIED_BY: "Modificato da",
        ReferenceType.JURISPRUDENCE: "Giurisprudenza",
    }

    def __init__(self, client: Any):
        """
        Initialize CrossReferenceService.

        Args:
            client: Connected FalkorDBClient instance
        """
        self.client = client
        logger.info("CrossReferenceService initialized")

    async def get_cross_references(
        self,
        urn: str,
        limit_per_group: int = 10,
    ) -> CrossReferenceResponse:
        """
        Get all cross-references for an article (AC5).

        Args:
            urn: Article URN
            limit_per_group: Max references per group

        Returns:
            CrossReferenceResponse with all groups
        """
        groups = []
        total = 0

        # Fetch all groups in parallel
        import asyncio

        results = await asyncio.gather(
            self.get_outgoing_references(urn, limit=limit_per_group),
            self.get_incoming_references(urn, limit=limit_per_group),
            self.get_modifications(urn, limit=limit_per_group),
            self.get_jurisprudence(urn, limit=limit_per_group),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, Exception):
                logger.warning("Cross-reference query failed: %s", result)
                continue
            if result and result.total_count > 0:
                groups.append(result)
                total += result.total_count

        return CrossReferenceResponse(
            urn=urn,
            groups=groups,
            total_references=total,
        )

    async def get_outgoing_references(
        self,
        urn: str,
        limit: int = 20,
        offset: int = 0,
    ) -> CrossReferenceGroup:
        """
        Get norms this article cites (AC1).

        Args:
            urn: Article URN
            limit: Max results
            offset: Pagination offset

        Returns:
            CrossReferenceGroup with outgoing references
        """
        # Query for outgoing cita edges (lowercase to match schema)
        query = """
            MATCH (n:Norma {urn: $urn})-[r:cita]->(m:Norma)
            RETURN m.urn AS target_urn, m.rubrica AS title, type(r) AS rel_type,
                   r.contesto AS snippet
            ORDER BY m.urn
            SKIP $offset
            LIMIT $limit
        """

        count_query = """
            MATCH (n:Norma {urn: $urn})-[:cita]->(m:Norma)
            RETURN count(m) AS cnt
        """

        references = []
        total_count = 0

        try:
            # Get count - handle both list and result_set returns
            count_result = await self.client.query(count_query, {"urn": urn})
            count_rows = count_result if isinstance(count_result, list) else (
                count_result.result_set if hasattr(count_result, "result_set") else []
            )
            if count_rows:
                total_count = count_rows[0][0] or 0

            # Get references
            result = await self.client.query(
                query, {"urn": urn, "limit": limit, "offset": offset}
            )
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            for row in rows or []:
                if row[0]:  # Has target URN
                    references.append(
                        CrossReference(
                            urn=row[0],
                            title=row[1] or "Articolo",
                            relationship=row[2] or "cita",
                            source_type="norm",
                            snippet=row[3],
                        )
                    )

        except Exception as e:
            logger.warning("Outgoing references query failed: %s", e)

        return CrossReferenceGroup(
            reference_type=ReferenceType.OUTGOING,
            label=self.LABELS[ReferenceType.OUTGOING],
            references=references,
            total_count=total_count,
            has_more=total_count > offset + len(references),
        )

    async def get_incoming_references(
        self,
        urn: str,
        limit: int = 20,
        offset: int = 0,
    ) -> CrossReferenceGroup:
        """
        Get norms citing this article (AC2).

        Args:
            urn: Article URN
            limit: Max results
            offset: Pagination offset

        Returns:
            CrossReferenceGroup with incoming references
        """
        # Query for incoming cita edges (lowercase to match schema)
        query = """
            MATCH (m:Norma)-[r:cita]->(n:Norma {urn: $urn})
            RETURN m.urn AS source_urn, m.rubrica AS title, type(r) AS rel_type,
                   r.contesto AS snippet
            ORDER BY m.urn
            SKIP $offset
            LIMIT $limit
        """

        count_query = """
            MATCH (m:Norma)-[:cita]->(n:Norma {urn: $urn})
            RETURN count(m) AS cnt
        """

        references = []
        total_count = 0

        try:
            # Handle both list and result_set returns
            count_result = await self.client.query(count_query, {"urn": urn})
            count_rows = count_result if isinstance(count_result, list) else (
                count_result.result_set if hasattr(count_result, "result_set") else []
            )
            if count_rows:
                total_count = count_rows[0][0] or 0

            result = await self.client.query(
                query, {"urn": urn, "limit": limit, "offset": offset}
            )
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            for row in rows or []:
                if row[0]:
                    references.append(
                        CrossReference(
                            urn=row[0],
                            title=row[1] or "Articolo",
                            relationship=row[2] or "cita",
                            source_type="norm",
                            snippet=row[3],
                        )
                    )

        except Exception as e:
            logger.warning("Incoming references query failed: %s", e)

        return CrossReferenceGroup(
            reference_type=ReferenceType.INCOMING,
            label=self.LABELS[ReferenceType.INCOMING],
            references=references,
            total_count=total_count,
            has_more=total_count > offset + len(references),
        )

    async def get_modifications(
        self,
        urn: str,
        limit: int = 20,
        offset: int = 0,
    ) -> CrossReferenceGroup:
        """
        Get legislation modifying this article (AC3).

        Args:
            urn: Article URN
            limit: Max results
            offset: Pagination offset

        Returns:
            CrossReferenceGroup with modifying legislation
        """
        # Build edge type list for query
        edge_types = "|".join(e.value for e in self.MODIFICATION_EDGES)

        query = f"""
            MATCH (m:Norma)-[r:{edge_types}]->(n:Norma {{urn: $urn}})
            RETURN m.urn AS source_urn, m.rubrica AS title, type(r) AS rel_type,
                   r.data_efficacia AS date, r.descrizione AS snippet
            ORDER BY r.data_efficacia DESC, m.urn
            SKIP $offset
            LIMIT $limit
        """

        count_query = f"""
            MATCH (m:Norma)-[:{edge_types}]->(n:Norma {{urn: $urn}})
            RETURN count(m) AS cnt
        """

        references = []
        total_count = 0

        try:
            # Handle both list and result_set returns
            count_result = await self.client.query(count_query, {"urn": urn})
            count_rows = count_result if isinstance(count_result, list) else (
                count_result.result_set if hasattr(count_result, "result_set") else []
            )
            if count_rows:
                total_count = count_rows[0][0] or 0

            result = await self.client.query(
                query, {"urn": urn, "limit": limit, "offset": offset}
            )
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            for row in rows or []:
                if row[0]:
                    references.append(
                        CrossReference(
                            urn=row[0],
                            title=row[1] or "Norma",
                            relationship=row[2] or "modifica",
                            source_type="norm",
                            date=row[3],
                            snippet=row[4],
                        )
                    )

        except Exception as e:
            logger.warning("Modifications query failed: %s", e)

        return CrossReferenceGroup(
            reference_type=ReferenceType.MODIFIED_BY,
            label=self.LABELS[ReferenceType.MODIFIED_BY],
            references=references,
            total_count=total_count,
            has_more=total_count > offset + len(references),
        )

    async def get_jurisprudence(
        self,
        urn: str,
        limit: int = 20,
        offset: int = 0,
    ) -> CrossReferenceGroup:
        """
        Get case law interpreting this article (AC4).

        Args:
            urn: Article URN
            limit: Max results
            offset: Pagination offset

        Returns:
            CrossReferenceGroup with jurisprudence
        """
        # Query for jurisprudence (lowercase to match schema)
        query = """
            MATCH (j:AttoGiudiziario)-[r:interpreta]->(n:Norma {urn: $urn})
            RETURN j.node_id AS node_id, j.massima AS title, type(r) AS rel_type,
                   j.organo_emittente AS authority, j.data AS date,
                   j.massima AS snippet
            ORDER BY j.data DESC
            SKIP $offset
            LIMIT $limit
        """

        count_query = """
            MATCH (j:AttoGiudiziario)-[:interpreta]->(n:Norma {urn: $urn})
            RETURN count(j) AS cnt
        """

        references = []
        total_count = 0

        try:
            # Handle both list and result_set returns
            count_result = await self.client.query(count_query, {"urn": urn})
            count_rows = count_result if isinstance(count_result, list) else (
                count_result.result_set if hasattr(count_result, "result_set") else []
            )
            if count_rows:
                total_count = count_rows[0][0] or 0

            result = await self.client.query(
                query, {"urn": urn, "limit": limit, "offset": offset}
            )
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            for row in rows or []:
                if row[0]:
                    # Truncate massima for title
                    title = row[1][:100] + "..." if row[1] and len(row[1]) > 100 else (row[1] or "Massima")

                    references.append(
                        CrossReference(
                            urn=row[0],  # node_id for jurisprudence
                            title=title,
                            relationship=row[2] or "interpreta",
                            source_type="jurisprudence",
                            authority=row[3],
                            date=row[4],
                            snippet=row[5][:200] if row[5] else None,
                        )
                    )

        except Exception as e:
            logger.warning("Jurisprudence query failed: %s", e)

        return CrossReferenceGroup(
            reference_type=ReferenceType.JURISPRUDENCE,
            label=self.LABELS[ReferenceType.JURISPRUDENCE],
            references=references,
            total_count=total_count,
            has_more=total_count > offset + len(references),
        )

    async def get_all_neighbors(
        self,
        urn: str,
        limit: int = 50,
    ) -> List[CrossReference]:
        """
        Get all graph neighbors regardless of type.

        Useful for graph visualization.

        Args:
            urn: Article URN
            limit: Max total results

        Returns:
            List of all CrossReference items
        """
        query = """
            MATCH (n:Norma {urn: $urn})-[r]-(m)
            RETURN m.urn AS target_urn, m.rubrica AS title, type(r) AS rel_type,
                   labels(m)[0] AS node_type
            LIMIT $limit
        """

        references = []

        try:
            result = await self.client.query(query, {"urn": urn, "limit": limit})
            # Handle both list and result_set returns
            rows = result if isinstance(result, list) else (
                result.result_set if hasattr(result, "result_set") else []
            )

            for row in rows or []:
                if row[0]:
                    node_type = row[3] or "Norma"
                    source_type = "norm"
                    if node_type == "AttoGiudiziario":
                        source_type = "jurisprudence"
                    elif node_type == "Dottrina":
                        source_type = "commentary"

                    references.append(
                        CrossReference(
                            urn=row[0],
                            title=row[1] or node_type,
                            relationship=row[2] or "related",
                            source_type=source_type,
                        )
                    )

        except Exception as e:
            logger.warning("All neighbors query failed: %s", e)

        return references
