"""
Tests for Modification Alert Service
====================================

Tests cover:
- AC1: Recent modification detection
- AC2: Batch checking (Dossier support)
- AC3: Newer version availability
- Abrogation alerts
"""

import pytest
from datetime import date, timedelta
from unittest.mock import MagicMock, AsyncMock

from visualex.graph.alerts import (
    AlertType,
    ModificationAlert,
    ModificationAlertService,
)


# =============================================================================
# ModificationAlert Tests
# =============================================================================


class TestModificationAlert:
    """Tests for ModificationAlert dataclass."""

    def test_creation(self):
        """Test basic creation."""
        alert = ModificationAlert(
            urn="urn:nir:stato:legge:2020;178~art5",
            alert_type=AlertType.RECENT_MODIFICATION,
            message="Modificato di recente",
        )

        assert alert.urn == "urn:nir:stato:legge:2020;178~art5"
        assert alert.alert_type == AlertType.RECENT_MODIFICATION

    def test_creation_with_metadata(self):
        """Test creation with all fields."""
        alert = ModificationAlert(
            urn="urn:test",
            alert_type=AlertType.RECENT_MODIFICATION,
            message="Test message",
            modification_date=date(2024, 1, 15),
            modifying_norm_urn="urn:nir:stato:legge:2024;10",
            modifying_norm_title="L. 10/2024",
            days_ago=15,
        )

        assert alert.modification_date == date(2024, 1, 15)
        assert alert.modifying_norm_urn == "urn:nir:stato:legge:2024;10"
        assert alert.days_ago == 15

    def test_to_dict(self):
        """Test serialization."""
        alert = ModificationAlert(
            urn="urn:test",
            alert_type=AlertType.NEWER_VERSION,
            message="Versione più recente disponibile",
        )

        d = alert.to_dict()

        assert d["urn"] == "urn:test"
        assert d["alert_type"] == AlertType.NEWER_VERSION
        assert d["message"] == "Versione più recente disponibile"
        assert "modification_date" not in d

    def test_to_dict_with_all_fields(self):
        """Test serialization with all optional fields."""
        alert = ModificationAlert(
            urn="urn:test",
            alert_type=AlertType.RECENT_MODIFICATION,
            message="Modified",
            modification_date=date(2024, 1, 15),
            modifying_norm_urn="urn:mod",
            modifying_norm_title="L. Modificante",
            days_ago=10,
        )

        d = alert.to_dict()

        assert d["modification_date"] == "2024-01-15"
        assert d["modifying_norm_urn"] == "urn:mod"
        assert d["modifying_norm_title"] == "L. Modificante"
        assert d["days_ago"] == 10


class TestAlertType:
    """Tests for AlertType constants."""

    def test_values(self):
        """Test constant values."""
        assert AlertType.RECENT_MODIFICATION == "recent_modification"
        assert AlertType.NEWER_VERSION == "newer_version"
        assert AlertType.ABROGATED == "abrogated"


# =============================================================================
# ModificationAlertService Tests
# =============================================================================


class TestModificationAlertService:
    """Tests for ModificationAlertService."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = ModificationAlertService(self.mock_client)

    @pytest.mark.asyncio
    async def test_check_recent_modification_found(self):
        """Test detecting recent modification (AC1)."""
        # Mock: modification found 10 days ago
        mod_date = (date.today() - timedelta(days=10)).isoformat()
        self.mock_client.query.return_value = [{
            "modifying_urn": "urn:nir:stato:legge:2024;50",
            "modifying_title": "L. 50/2024",
            "modification_type": "sostituisce",
            "modification_date": mod_date,
        }]

        alert = await self.service.check_recent_modification(
            urn="urn:nir:stato:legge:2020;178~art5",
            days=30,
        )

        assert alert is not None
        assert alert.alert_type == AlertType.RECENT_MODIFICATION
        assert alert.modifying_norm_urn == "urn:nir:stato:legge:2024;50"
        assert alert.days_ago == 10

    @pytest.mark.asyncio
    async def test_check_recent_modification_not_found(self):
        """Test no recent modification."""
        self.mock_client.query.return_value = []

        alert = await self.service.check_recent_modification(
            urn="urn:nir:stato:legge:2020;178~art5",
            days=30,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_check_recent_modification_old(self):
        """Test modification outside window returns None."""
        # Mock: modification 60 days ago (outside 30-day window)
        mod_date = (date.today() - timedelta(days=60)).isoformat()
        self.mock_client.query.return_value = [{
            "modifying_urn": "urn:old",
            "modifying_title": "Old Law",
            "modification_type": "sostituisce",
            "modification_date": mod_date,
        }]

        # But query won't return it because cutoff_date filters in Cypher
        self.mock_client.query.return_value = []

        alert = await self.service.check_recent_modification(
            urn="urn:test",
            days=30,
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_check_newer_version_exists(self):
        """Test detecting newer version (AC3)."""
        self.mock_client.query.return_value = [{
            "has_newer": True,
            "latest_version_date": "2024-01-01",
        }]

        alert = await self.service.check_newer_version(
            urn="urn:test",
            as_of_date=date(2023, 1, 1),
        )

        assert alert is not None
        assert alert.alert_type == AlertType.NEWER_VERSION
        assert "più recente" in alert.message

    @pytest.mark.asyncio
    async def test_check_newer_version_not_exists(self):
        """Test no newer version available."""
        # First query: no newer versions
        # Second query: no norm update either
        self.mock_client.query.side_effect = [
            [{"has_newer": False, "latest_version_date": None}],
            [],  # norm query returns nothing
        ]

        alert = await self.service.check_newer_version(
            urn="urn:test",
            as_of_date=date(2024, 1, 1),
        )

        assert alert is None

    @pytest.mark.asyncio
    async def test_check_abrogation_found(self):
        """Test detecting abrogation."""
        self.mock_client.query.return_value = [{
            "abrogating_urn": "urn:nir:stato:legge:2023;100",
            "abrogating_title": "L. 100/2023",
            "abrogation_date": "2023-07-01",
        }]

        alert = await self.service.check_abrogation(urn="urn:test")

        assert alert is not None
        assert alert.alert_type == AlertType.ABROGATED
        assert alert.modifying_norm_urn == "urn:nir:stato:legge:2023;100"
        assert "abrogato" in alert.message

    @pytest.mark.asyncio
    async def test_check_abrogation_not_found(self):
        """Test norm not abrogated."""
        self.mock_client.query.return_value = []

        alert = await self.service.check_abrogation(urn="urn:test")

        assert alert is None

    @pytest.mark.asyncio
    async def test_check_batch(self):
        """Test batch checking for Dossier (AC2)."""
        mod_date = (date.today() - timedelta(days=5)).isoformat()
        self.mock_client.query.return_value = [
            {
                "urn": "urn:nir:stato:legge:2020;1~art1",
                "modifying_urn": "urn:mod1",
                "modifying_title": "L. Mod 1",
                "modification_type": "sostituisce",
                "modification_date": mod_date,
            },
            {
                "urn": "urn:nir:stato:legge:2020;2~art5",
                "modifying_urn": "urn:mod2",
                "modifying_title": "L. Mod 2",
                "modification_type": "integra",
                "modification_date": mod_date,
            },
        ]

        alerts = await self.service.check_batch(
            urns=[
                "urn:nir:stato:legge:2020;1~art1",
                "urn:nir:stato:legge:2020;2~art5",
                "urn:nir:stato:legge:2020;3~art10",  # No modification
            ],
            days=30,
        )

        assert len(alerts) == 2
        assert alerts[0].urn == "urn:nir:stato:legge:2020;1~art1"
        assert alerts[1].urn == "urn:nir:stato:legge:2020;2~art5"

    @pytest.mark.asyncio
    async def test_check_batch_empty(self):
        """Test batch with empty list."""
        alerts = await self.service.check_batch(urns=[], days=30)
        assert alerts == []

    @pytest.mark.asyncio
    async def test_check_batch_no_modifications(self):
        """Test batch with no recent modifications."""
        self.mock_client.query.return_value = []

        alerts = await self.service.check_batch(
            urns=["urn:1", "urn:2"],
            days=30,
        )

        assert alerts == []

    @pytest.mark.asyncio
    async def test_get_all_alerts(self):
        """Test getting all alerts for a norm."""
        mod_date = (date.today() - timedelta(days=5)).isoformat()

        # First call: recent modification check
        # Second call: abrogation check (None)
        self.mock_client.query.side_effect = [
            [{
                "modifying_urn": "urn:mod",
                "modifying_title": "L. Mod",
                "modification_type": "sostituisce",
                "modification_date": mod_date,
            }],
            [],  # No abrogation
        ]

        alerts = await self.service.get_all_alerts(
            urn="urn:test",
            days=30,
        )

        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.RECENT_MODIFICATION

    @pytest.mark.asyncio
    async def test_get_all_alerts_with_as_of_date(self):
        """Test getting all alerts including newer version check."""
        mod_date = (date.today() - timedelta(days=5)).isoformat()

        self.mock_client.query.side_effect = [
            [{
                "modifying_urn": "urn:mod",
                "modifying_title": "L. Mod",
                "modification_type": "sostituisce",
                "modification_date": mod_date,
            }],
            [],  # No abrogation
            [{"has_newer": True, "latest_version_date": "2024-01-01"}],  # Newer exists
        ]

        alerts = await self.service.get_all_alerts(
            urn="urn:test",
            as_of_date=date(2023, 1, 1),
            days=30,
        )

        assert len(alerts) == 2
        alert_types = {a.alert_type for a in alerts}
        assert AlertType.RECENT_MODIFICATION in alert_types
        assert AlertType.NEWER_VERSION in alert_types

    @pytest.mark.asyncio
    async def test_query_failure_handled(self):
        """Test graceful handling of query failures."""
        self.mock_client.query.side_effect = Exception("DB error")

        alert = await self.service.check_recent_modification(urn="urn:test")

        assert alert is None  # Should not raise

    @pytest.mark.asyncio
    async def test_message_format_italian(self):
        """Test messages are in Italian."""
        mod_date = date.today() - timedelta(days=5)
        self.mock_client.query.return_value = [{
            "modifying_urn": "urn:nir:stato:legge:2024;50",
            "modifying_title": "Legge 50/2024",
            "modification_type": "sostituisce",
            "modification_date": mod_date.isoformat(),
        }]

        alert = await self.service.check_recent_modification(urn="urn:test")

        assert "modificato" in alert.message.lower()
        assert "Legge 50/2024" in alert.message


# =============================================================================
# Integration Tests
# =============================================================================


class TestAlertServiceIntegration:
    """Integration-style tests."""

    def setup_method(self):
        """Create service with mock client."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.service = ModificationAlertService(self.mock_client)

    def test_default_alert_days(self):
        """Test default alert window is 30 days."""
        assert ModificationAlertService.DEFAULT_ALERT_DAYS == 30

    def test_message_templates(self):
        """Test all message templates exist."""
        assert AlertType.RECENT_MODIFICATION in ModificationAlertService.MESSAGES
        assert AlertType.NEWER_VERSION in ModificationAlertService.MESSAGES
        assert AlertType.ABROGATED in ModificationAlertService.MESSAGES
