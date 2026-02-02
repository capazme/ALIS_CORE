"""
Modification Alert Service
==========================

Detects recent modifications to legal norms and provides alerts.

Supports:
- Recent modification detection (within N days)
- Modification details (date, modifying legislation)
- Newer version availability check
- Batch checking for multiple URNs (Dossier support)

Reference: Story 3-7: Modification Detection Alert
"""

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Dict, List, Any, Optional

if TYPE_CHECKING:
    from visualex.graph.client import FalkorDBClient

__all__ = [
    "ModificationAlert",
    "ModificationAlertService",
    "AlertType",
]

logger = logging.getLogger(__name__)


class AlertType:
    """Types of modification alerts."""

    RECENT_MODIFICATION = "recent_modification"
    NEWER_VERSION = "newer_version"
    ABROGATED = "abrogated"


@dataclass
class ModificationAlert:
    """Alert about a norm modification."""

    urn: str
    alert_type: str
    message: str
    modification_date: Optional[date] = None
    modifying_norm_urn: Optional[str] = None
    modifying_norm_title: Optional[str] = None
    days_ago: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "urn": self.urn,
            "alert_type": self.alert_type,
            "message": self.message,
        }
        if self.modification_date:
            result["modification_date"] = self.modification_date.isoformat()
        if self.modifying_norm_urn:
            result["modifying_norm_urn"] = self.modifying_norm_urn
        if self.modifying_norm_title:
            result["modifying_norm_title"] = self.modifying_norm_title
        if self.days_ago is not None:
            result["days_ago"] = self.days_ago
        return result


class ModificationAlertService:
    """
    Service for detecting and alerting on norm modifications.

    Provides:
    - Recent modification detection (AC1)
    - Newer version availability check (AC3)
    - Batch checking for Dossier support (AC2)

    Example:
        service = ModificationAlertService(falkor_client)

        # Check single norm
        alert = await service.check_recent_modification(
            urn="urn:nir:stato:legge:2020;178~art5",
            days=30
        )

        # Check if newer version exists
        alert = await service.check_newer_version(
            urn="urn:...",
            as_of_date="2023-01-01"
        )

        # Batch check for Dossier
        alerts = await service.check_batch([urn1, urn2, ...], days=30)
    """

    # Default alert window in days
    DEFAULT_ALERT_DAYS = 30

    # Italian message templates
    MESSAGES = {
        AlertType.RECENT_MODIFICATION: (
            "Questo articolo è stato modificato il {date} da {norm}"
        ),
        AlertType.NEWER_VERSION: "Versione più recente disponibile",
        AlertType.ABROGATED: "Questo articolo è stato abrogato il {date}",
    }

    def __init__(self, client: "FalkorDBClient"):
        """
        Initialize ModificationAlertService.

        Args:
            client: Connected FalkorDBClient instance
        """
        self.client = client
        logger.info("ModificationAlertService initialized")

    async def check_recent_modification(
        self,
        urn: str,
        days: int = DEFAULT_ALERT_DAYS,
    ) -> Optional[ModificationAlert]:
        """
        Check if a norm was modified recently (AC1).

        Args:
            urn: The norm URN
            days: Number of days to consider "recent"

        Returns:
            ModificationAlert if recently modified, None otherwise
        """
        cutoff_date = date.today() - timedelta(days=days)

        # Query for recent modifications
        query = """
            MATCH (m)-[r:sostituisce|abroga_totalmente|abroga_parzialmente|integra|deroga_a]->(n:Norma {urn: $urn})
            WHERE r.data_efficacia >= $cutoff_date
            RETURN m.urn AS modifying_urn,
                   m.titolo AS modifying_title,
                   type(r) AS modification_type,
                   r.data_efficacia AS modification_date
            ORDER BY r.data_efficacia DESC
            LIMIT 1
        """

        try:
            result = await self.client.query(
                query,
                {"urn": urn, "cutoff_date": cutoff_date.isoformat()}
            )

            if not result:
                return None

            record = result[0]
            mod_date_str = record.get("modification_date")
            mod_date = self._parse_date(mod_date_str) if mod_date_str else None

            if not mod_date:
                return None

            days_ago = (date.today() - mod_date).days
            mod_title = record.get("modifying_title") or record.get("modifying_urn", "norma")

            message = self.MESSAGES[AlertType.RECENT_MODIFICATION].format(
                date=mod_date.strftime("%d/%m/%Y"),
                norm=mod_title,
            )

            return ModificationAlert(
                urn=urn,
                alert_type=AlertType.RECENT_MODIFICATION,
                message=message,
                modification_date=mod_date,
                modifying_norm_urn=record.get("modifying_urn"),
                modifying_norm_title=record.get("modifying_title"),
                days_ago=days_ago,
            )

        except Exception as e:
            logger.warning("Recent modification check failed for %s: %s", urn, e)
            return None

    async def check_newer_version(
        self,
        urn: str,
        as_of_date: date,
    ) -> Optional[ModificationAlert]:
        """
        Check if a newer version exists than the queried date (AC3).

        Args:
            urn: The norm URN
            as_of_date: The date being queried

        Returns:
            ModificationAlert if newer version exists, None otherwise
        """
        query = """
            MATCH (n:Norma {urn: $urn})
            OPTIONAL MATCH (n)-[:ha_versione]->(v:Versione)
            WHERE v.data_inizio_validita > $as_of_date
            RETURN count(v) > 0 AS has_newer,
                   max(v.data_inizio_validita) AS latest_version_date
        """

        try:
            result = await self.client.query(
                query,
                {"urn": urn, "as_of_date": as_of_date.isoformat()}
            )

            if not result:
                return None

            record = result[0]
            if not record.get("has_newer"):
                # Also check if norm's data_versione is newer
                norm_query = """
                    MATCH (n:Norma {urn: $urn})
                    WHERE n.data_versione > $as_of_date
                    RETURN n.data_versione AS latest_date
                """
                norm_result = await self.client.query(
                    norm_query,
                    {"urn": urn, "as_of_date": as_of_date.isoformat()}
                )

                if not norm_result:
                    return None

                return ModificationAlert(
                    urn=urn,
                    alert_type=AlertType.NEWER_VERSION,
                    message=self.MESSAGES[AlertType.NEWER_VERSION],
                )

            return ModificationAlert(
                urn=urn,
                alert_type=AlertType.NEWER_VERSION,
                message=self.MESSAGES[AlertType.NEWER_VERSION],
            )

        except Exception as e:
            logger.warning("Newer version check failed for %s: %s", urn, e)
            return None

    async def check_abrogation(
        self,
        urn: str,
    ) -> Optional[ModificationAlert]:
        """
        Check if a norm has been abrogated.

        Args:
            urn: The norm URN

        Returns:
            ModificationAlert if abrogated, None otherwise
        """
        query = """
            MATCH (m)-[r:abroga_totalmente]->(n:Norma {urn: $urn})
            RETURN m.urn AS abrogating_urn,
                   m.titolo AS abrogating_title,
                   r.data_efficacia AS abrogation_date
        """

        try:
            result = await self.client.query(query, {"urn": urn})

            if not result:
                return None

            record = result[0]
            abrog_date_str = record.get("abrogation_date")
            abrog_date = self._parse_date(abrog_date_str) if abrog_date_str else None

            message = self.MESSAGES[AlertType.ABROGATED].format(
                date=abrog_date.strftime("%d/%m/%Y") if abrog_date else "data ignota"
            )

            return ModificationAlert(
                urn=urn,
                alert_type=AlertType.ABROGATED,
                message=message,
                modification_date=abrog_date,
                modifying_norm_urn=record.get("abrogating_urn"),
                modifying_norm_title=record.get("abrogating_title"),
            )

        except Exception as e:
            logger.warning("Abrogation check failed for %s: %s", urn, e)
            return None

    async def check_batch(
        self,
        urns: List[str],
        days: int = DEFAULT_ALERT_DAYS,
    ) -> List[ModificationAlert]:
        """
        Check multiple URNs for recent modifications (AC2 - Dossier support).

        Args:
            urns: List of URNs to check
            days: Number of days to consider "recent"

        Returns:
            List of ModificationAlert for norms that were recently modified
        """
        if not urns:
            return []

        cutoff_date = date.today() - timedelta(days=days)

        query = """
            MATCH (m)-[r:sostituisce|abroga_totalmente|abroga_parzialmente|integra|deroga_a]->(n:Norma)
            WHERE n.urn IN $urns AND r.data_efficacia >= $cutoff_date
            RETURN n.urn AS urn,
                   m.urn AS modifying_urn,
                   m.titolo AS modifying_title,
                   type(r) AS modification_type,
                   r.data_efficacia AS modification_date
            ORDER BY r.data_efficacia DESC
        """

        alerts = []

        try:
            result = await self.client.query(
                query,
                {"urns": urns, "cutoff_date": cutoff_date.isoformat()}
            )

            # Group by URN, take most recent modification for each
            seen_urns = set()
            for record in result:
                urn = record.get("urn")
                if urn in seen_urns:
                    continue
                seen_urns.add(urn)

                mod_date_str = record.get("modification_date")
                mod_date = self._parse_date(mod_date_str) if mod_date_str else None
                days_ago = (date.today() - mod_date).days if mod_date else None
                mod_title = record.get("modifying_title") or record.get("modifying_urn", "norma")

                message = self.MESSAGES[AlertType.RECENT_MODIFICATION].format(
                    date=mod_date.strftime("%d/%m/%Y") if mod_date else "data ignota",
                    norm=mod_title,
                )

                alerts.append(ModificationAlert(
                    urn=urn,
                    alert_type=AlertType.RECENT_MODIFICATION,
                    message=message,
                    modification_date=mod_date,
                    modifying_norm_urn=record.get("modifying_urn"),
                    modifying_norm_title=record.get("modifying_title"),
                    days_ago=days_ago,
                ))

        except Exception as e:
            logger.warning("Batch modification check failed: %s", e)

        return alerts

    async def get_all_alerts(
        self,
        urn: str,
        as_of_date: Optional[date] = None,
        days: int = DEFAULT_ALERT_DAYS,
    ) -> List[ModificationAlert]:
        """
        Get all applicable alerts for a norm.

        Args:
            urn: The norm URN
            as_of_date: If querying historical version, check for newer versions
            days: Number of days for recent modification check

        Returns:
            List of all applicable alerts
        """
        import asyncio

        tasks = [
            self.check_recent_modification(urn, days),
            self.check_abrogation(urn),
        ]

        if as_of_date:
            tasks.append(self.check_newer_version(urn, as_of_date))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        alerts = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Alert check failed: %s", result)
                continue
            if result:
                alerts.append(result)

        return alerts

    def _parse_date(self, value: Any) -> Optional[date]:
        """Parse a date from various formats."""
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
                    return None
        return None
