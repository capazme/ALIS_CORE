"""
Temporal Versioning Module
==========================

Query norms as they existed at a specific date for multivigenza support.

Italian legal texts undergo frequent modifications. This module enables:
- Point-in-time queries (as_of_date)
- Version timeline retrieval
- Abrogation status detection
- Historical version navigation
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Dict, List, Any, Optional, Union

# Schema types available if needed for future extensions
# from visualex.graph.schema import NodeType, EdgeType

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient

__all__ = [
    "TemporalQuery",
    "VersionedNorm",
    "VersionTimeline",
    "VersionDiff",
    "DiffSegment",
    "NormStatus",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


class NormStatus:
    """Status constants for norms."""
    VIGENTE = "vigente"
    ABROGATO = "abrogato"
    SOSPESO = "sospeso"
    NON_ANCORA_IN_VIGORE = "non_ancora_in_vigore"


@dataclass
class VersionedNorm:
    """A norm at a specific point in time."""

    urn: str
    testo_vigente: str
    vigenza_dal: Optional[date] = None
    vigenza_al: Optional[date] = None
    is_current: bool = False
    is_abrogato: bool = False
    newer_version_exists: bool = False
    status: str = NormStatus.VIGENTE
    titolo: Optional[str] = None
    rubrica: Optional[str] = None
    tipo_modifica: Optional[str] = None
    # Modifying legislation info
    modifying_norm_urn: Optional[str] = None
    modifying_norm_title: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "urn": self.urn,
            "testo_vigente": self.testo_vigente,
            "vigenza_dal": self.vigenza_dal.isoformat() if self.vigenza_dal else None,
            "vigenza_al": self.vigenza_al.isoformat() if self.vigenza_al else None,
            "is_current": self.is_current,
            "is_abrogato": self.is_abrogato,
            "newer_version_exists": self.newer_version_exists,
            "status": self.status,
            "titolo": self.titolo,
            "rubrica": self.rubrica,
            "tipo_modifica": self.tipo_modifica,
        }
        if self.modifying_norm_urn:
            result["modifying_norm_urn"] = self.modifying_norm_urn
        if self.modifying_norm_title:
            result["modifying_norm_title"] = self.modifying_norm_title
        return result


@dataclass
class DiffSegment:
    """A segment in a version diff."""

    text: str
    change_type: str  # "unchanged", "added", "removed"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "change_type": self.change_type,
        }


@dataclass
class VersionDiff:
    """Diff between two versions of a norm."""

    urn: str
    version_a_date: Optional[date]
    version_b_date: Optional[date]
    segments: List[DiffSegment] = field(default_factory=list)
    additions_count: int = 0
    deletions_count: int = 0
    unchanged_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urn": self.urn,
            "version_a_date": self.version_a_date.isoformat() if self.version_a_date else None,
            "version_b_date": self.version_b_date.isoformat() if self.version_b_date else None,
            "segments": [s.to_dict() for s in self.segments],
            "additions_count": self.additions_count,
            "deletions_count": self.deletions_count,
            "unchanged_count": self.unchanged_count,
        }


@dataclass
class VersionTimeline:
    """Complete version history of a norm."""

    urn: str
    versions: List[VersionedNorm] = field(default_factory=list)
    current_version: Optional[VersionedNorm] = None
    abrogation_date: Optional[date] = None
    total_versions: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "urn": self.urn,
            "versions": [v.to_dict() for v in self.versions],
            "current_version": self.current_version.to_dict() if self.current_version else None,
            "abrogation_date": self.abrogation_date.isoformat() if self.abrogation_date else None,
            "total_versions": self.total_versions,
        }


# =============================================================================
# Temporal Query Class
# =============================================================================


class TemporalQuery:
    """
    Query norms at specific points in time.

    Supports:
    - Point-in-time queries (as_of_date parameter)
    - Version timeline retrieval
    - Abrogation status detection
    - Historical version navigation
    """

    def __init__(self, client: "FalkorDBClient"):
        """
        Initialize temporal query handler.

        Args:
            client: Connected FalkorDBClient instance
        """
        self.client = client

    async def get_norm_at_date(
        self,
        urn: str,
        as_of_date: Optional[Union[date, str]] = None,
    ) -> Optional[VersionedNorm]:
        """
        Get a norm as it existed at a specific date (AC1, AC2).

        Args:
            urn: The norm URN
            as_of_date: The date to query (None = current version)

        Returns:
            VersionedNorm with the version in force at that date, or None
        """
        # Parse date if string
        query_date = self._parse_date(as_of_date) if as_of_date else None

        # First, get the norm node
        norm_result = await self.client.query(
            """
            MATCH (n:Norma {urn: $urn})
            OPTIONAL MATCH (n)-[:ha_versione]->(v:Versione)
            OPTIONAL MATCH (abrog)-[:abroga_totalmente]->(n)
            RETURN n,
                   collect(v) as versions,
                   abrog.data_versione as abrogation_date
            """,
            {"urn": urn}
        )

        if not norm_result:
            logger.debug("Norm not found: %s", urn)
            return None

        record = norm_result[0]
        norm_data = self._extract_node_properties(record.get("n"))
        versions = record.get("versions", [])
        abrogation_date = record.get("abrogation_date")

        if not norm_data:
            return None

        # Determine abrogation status
        is_abrogato = abrogation_date is not None or norm_data.get("stato") == NormStatus.ABROGATO

        # If no specific date requested, return current version
        if not query_date:
            return self._build_current_version(norm_data, is_abrogato)

        # Find version valid at the query date
        return await self._find_version_at_date(
            urn, norm_data, versions, query_date, is_abrogato
        )

    async def get_version_timeline(self, urn: str) -> VersionTimeline:
        """
        Get complete version history of a norm (AC4).

        Args:
            urn: The norm URN

        Returns:
            VersionTimeline with all versions ordered chronologically
        """
        result = await self.client.query(
            """
            MATCH (n:Norma {urn: $urn})
            OPTIONAL MATCH (n)-[:ha_versione]->(v:Versione)
            OPTIONAL MATCH (abrog)-[:abroga_totalmente]->(n)
            RETURN n,
                   collect(v) as versions,
                   abrog.data_versione as abrogation_date
            """,
            {"urn": urn}
        )
        # Note: versions are sorted in Python below (line ~253) since
        # ORDER BY after collect() doesn't order collection elements

        if not result:
            return VersionTimeline(urn=urn)

        record = result[0]
        norm_data = self._extract_node_properties(record.get("n"))
        raw_versions = record.get("versions", [])
        abrogation_date_str = record.get("abrogation_date")

        abrogation_date = self._parse_date(abrogation_date_str) if abrogation_date_str else None

        versions: List[VersionedNorm] = []
        current_version: Optional[VersionedNorm] = None

        # Process each version
        for raw_v in raw_versions:
            v_props = self._extract_node_properties(raw_v)
            if not v_props:
                continue

            vigenza_dal = self._parse_date(v_props.get("data_inizio_validita"))
            vigenza_al = self._parse_date(v_props.get("data_fine_validita"))

            version = VersionedNorm(
                urn=urn,
                testo_vigente=v_props.get("testo_completo", ""),
                vigenza_dal=vigenza_dal,
                vigenza_al=vigenza_al,
                is_current=(vigenza_al is None),
                is_abrogato=False,
                newer_version_exists=(vigenza_al is not None),
                tipo_modifica=v_props.get("descrizione_modifiche"),
            )
            versions.append(version)

            if vigenza_al is None:
                current_version = version

        # If no versions in Versione nodes, create from norm itself
        if not versions and norm_data:
            current_version = self._build_current_version(
                norm_data,
                abrogation_date is not None,
            )
            versions = [current_version] if current_version else []

        # Sort by start date
        versions.sort(key=lambda v: v.vigenza_dal or date.min)

        # Mark abrogation on current version if applicable
        if current_version and abrogation_date:
            current_version.is_abrogato = True
            current_version.status = NormStatus.ABROGATO

        return VersionTimeline(
            urn=urn,
            versions=versions,
            current_version=current_version,
            abrogation_date=abrogation_date,
            total_versions=len(versions),
        )

    async def is_abrogated(self, urn: str) -> Optional[bool]:
        """
        Check if a norm has been abrogated (AC3).

        Args:
            urn: The norm URN

        Returns:
            True if abrogated, False if not abrogated, None if norm not found
        """
        result = await self.client.query(
            """
            MATCH (n:Norma {urn: $urn})
            OPTIONAL MATCH (abrog)-[:abroga_totalmente]->(n)
            RETURN n.stato as stato, abrog IS NOT NULL as has_abrogation
            """,
            {"urn": urn}
        )

        if not result:
            logger.debug("Norm not found for abrogation check: %s", urn)
            return None

        record = result[0]
        stato = record.get("stato")
        has_abrogation = record.get("has_abrogation", False)

        return stato == NormStatus.ABROGATO or has_abrogation

    async def get_abrogation_info(self, urn: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed abrogation information for a norm.

        Args:
            urn: The norm URN

        Returns:
            Dict with abrogation details or None if not abrogated
        """
        result = await self.client.query(
            """
            MATCH (abrog)-[r:abroga_totalmente]->(n:Norma {urn: $urn})
            RETURN abrog.urn as abrogating_norm,
                   abrog.titolo as abrogating_title,
                   r.data_efficacia as effective_date
            """,
            {"urn": urn}
        )

        if not result:
            return None

        record = result[0]
        return {
            "is_abrogated": True,
            "abrogating_norm_urn": record.get("abrogating_norm"),
            "abrogating_norm_title": record.get("abrogating_title"),
            "effective_date": record.get("effective_date"),
        }

    async def get_versions_in_range(
        self,
        urn: str,
        from_date: Union[date, str],
        to_date: Union[date, str],
    ) -> List[VersionedNorm]:
        """
        Get all versions that were valid within a date range.

        Args:
            urn: The norm URN
            from_date: Start of range
            to_date: End of range

        Returns:
            List of versions valid during the range
        """
        start = self._parse_date(from_date)
        end = self._parse_date(to_date)

        if not start or not end:
            return []

        result = await self.client.query(
            """
            MATCH (n:Norma {urn: $urn})-[:ha_versione]->(v:Versione)
            WHERE v.data_inizio_validita <= $to_date
              AND (v.data_fine_validita IS NULL OR v.data_fine_validita >= $from_date)
            RETURN v
            ORDER BY v.data_inizio_validita ASC
            """,
            {
                "urn": urn,
                "from_date": start.isoformat(),
                "to_date": end.isoformat(),
            }
        )

        versions = []
        for record in result:
            v_props = self._extract_node_properties(record.get("v"))
            if not v_props:
                continue

            vigenza_dal = self._parse_date(v_props.get("data_inizio_validita"))
            vigenza_al = self._parse_date(v_props.get("data_fine_validita"))

            version = VersionedNorm(
                urn=urn,
                testo_vigente=v_props.get("testo_completo", ""),
                vigenza_dal=vigenza_dal,
                vigenza_al=vigenza_al,
                is_current=(vigenza_al is None),
                tipo_modifica=v_props.get("descrizione_modifiche"),
            )
            versions.append(version)

        return versions

    async def compare_versions(
        self,
        urn: str,
        date_a: Union[date, str],
        date_b: Union[date, str],
    ) -> Optional[VersionDiff]:
        """
        Compare two versions of a norm and produce a diff.

        Uses word-level diff for legal precision.

        Args:
            urn: The norm URN
            date_a: Date for first version (older)
            date_b: Date for second version (newer)

        Returns:
            VersionDiff with segments showing changes, or None if versions not found
        """
        import asyncio

        parsed_date_a = self._parse_date(date_a)
        parsed_date_b = self._parse_date(date_b)

        if not parsed_date_a or not parsed_date_b:
            return None

        # Get both versions in parallel for better performance
        version_a, version_b = await asyncio.gather(
            self.get_norm_at_date(urn, parsed_date_a),
            self.get_norm_at_date(urn, parsed_date_b),
        )

        if not version_a or not version_b:
            return None

        # Compute word-level diff
        segments, additions, deletions, unchanged = self._compute_word_diff(
            version_a.testo_vigente or "",
            version_b.testo_vigente or "",
        )

        return VersionDiff(
            urn=urn,
            version_a_date=parsed_date_a,
            version_b_date=parsed_date_b,
            segments=segments,
            additions_count=additions,
            deletions_count=deletions,
            unchanged_count=unchanged,
        )

    def _compute_word_diff(
        self,
        text_a: str,
        text_b: str,
    ) -> tuple[List[DiffSegment], int, int, int]:
        """
        Compute word-level diff between two texts.

        Returns:
            Tuple of (segments, additions_count, deletions_count, unchanged_count)
        """
        import difflib

        # Split into words while preserving whitespace
        words_a = text_a.split()
        words_b = text_b.split()

        segments: List[DiffSegment] = []
        additions = 0
        deletions = 0
        unchanged = 0

        # Use SequenceMatcher for word-level diff
        matcher = difflib.SequenceMatcher(None, words_a, words_b)

        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                text = " ".join(words_a[i1:i2])
                if text:
                    segments.append(DiffSegment(text=text, change_type="unchanged"))
                    unchanged += i2 - i1
            elif tag == "replace":
                # Old text removed
                old_text = " ".join(words_a[i1:i2])
                if old_text:
                    segments.append(DiffSegment(text=old_text, change_type="removed"))
                    deletions += i2 - i1
                # New text added
                new_text = " ".join(words_b[j1:j2])
                if new_text:
                    segments.append(DiffSegment(text=new_text, change_type="added"))
                    additions += j2 - j1
            elif tag == "delete":
                text = " ".join(words_a[i1:i2])
                if text:
                    segments.append(DiffSegment(text=text, change_type="removed"))
                    deletions += i2 - i1
            elif tag == "insert":
                text = " ".join(words_b[j1:j2])
                if text:
                    segments.append(DiffSegment(text=text, change_type="added"))
                    additions += j2 - j1

        return segments, additions, deletions, unchanged

    # ==========================================================================
    # Private Helper Methods
    # ==========================================================================

    def _parse_date(self, value: Optional[Union[date, str]]) -> Optional[date]:
        """Parse a date from string or return as-is if already a date."""
        if value is None:
            return None
        if isinstance(value, date):
            return value
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, str):
            try:
                return datetime.strptime(value, "%Y-%m-%d").date()
            except ValueError:
                try:
                    return datetime.strptime(value, "%d/%m/%Y").date()
                except ValueError:
                    logger.warning("Could not parse date: %s", value)
                    return None
        return None

    def _extract_node_properties(
        self,
        node: Optional[Any],
    ) -> Optional[Dict[str, Any]]:
        """Extract properties from a FalkorDB node result."""
        if node is None:
            return None

        # Handle different result formats
        if isinstance(node, dict):
            if "properties" in node:
                return node["properties"]
            return node

        # FalkorDB Node object
        if hasattr(node, "properties"):
            return dict(node.properties)

        return None

    def _build_current_version(
        self,
        norm_data: Dict[str, Any],
        is_abrogato: bool,
    ) -> VersionedNorm:
        """Build a VersionedNorm for the current state."""
        vigenza_dal = self._parse_date(norm_data.get("data_entrata_in_vigore"))
        if not vigenza_dal:
            vigenza_dal = self._parse_date(norm_data.get("data_versione"))

        return VersionedNorm(
            urn=norm_data.get("urn", ""),
            testo_vigente=norm_data.get("testo_vigente", ""),
            vigenza_dal=vigenza_dal,
            vigenza_al=None,
            is_current=True,
            is_abrogato=is_abrogato,
            newer_version_exists=False,
            status=NormStatus.ABROGATO if is_abrogato else NormStatus.VIGENTE,
            titolo=norm_data.get("titolo"),
            rubrica=norm_data.get("rubrica"),
        )

    async def _find_version_at_date(
        self,
        urn: str,
        norm_data: Dict[str, Any],
        versions: List[Any],
        query_date: date,
        is_abrogato: bool,
    ) -> Optional[VersionedNorm]:
        """Find the version that was valid at a specific date."""
        # First, check Versione nodes
        for raw_v in versions:
            v_props = self._extract_node_properties(raw_v)
            if not v_props:
                continue

            vigenza_dal = self._parse_date(v_props.get("data_inizio_validita"))
            vigenza_al = self._parse_date(v_props.get("data_fine_validita"))

            # Check if this version was valid at query_date
            # Note: vigenza_al is inclusive (Italian multivigenza "dal X al Y" includes Y)
            if vigenza_dal and vigenza_dal <= query_date:
                if vigenza_al is None or vigenza_al >= query_date:
                    # Found the matching version
                    newer_exists = vigenza_al is not None

                    return VersionedNorm(
                        urn=urn,
                        testo_vigente=v_props.get("testo_completo", ""),
                        vigenza_dal=vigenza_dal,
                        vigenza_al=vigenza_al,
                        is_current=(vigenza_al is None),
                        is_abrogato=is_abrogato and (vigenza_al is None),
                        newer_version_exists=newer_exists,
                        status=NormStatus.VIGENTE,
                        tipo_modifica=v_props.get("descrizione_modifiche"),
                    )

        # No version found in Versione nodes, fall back to norm data
        # Check if norm was in force at query_date
        norm_vigenza_dal = self._parse_date(norm_data.get("data_entrata_in_vigore"))
        if not norm_vigenza_dal:
            norm_vigenza_dal = self._parse_date(norm_data.get("data_versione"))

        if norm_vigenza_dal and norm_vigenza_dal <= query_date:
            return VersionedNorm(
                urn=urn,
                testo_vigente=norm_data.get("testo_vigente", ""),
                vigenza_dal=norm_vigenza_dal,
                vigenza_al=None,
                is_current=True,
                is_abrogato=is_abrogato,
                newer_version_exists=False,
                status=NormStatus.ABROGATO if is_abrogato else NormStatus.VIGENTE,
                titolo=norm_data.get("titolo"),
                rubrica=norm_data.get("rubrica"),
            )

        # Norm was not yet in force at query_date
        return None
