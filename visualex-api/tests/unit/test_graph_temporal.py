"""
Tests for Graph Temporal Versioning Module
==========================================

Tests for:
- TemporalQuery: Point-in-time queries
- VersionTimeline: Version history retrieval
- Abrogation detection
"""

import pytest
from datetime import date
from unittest.mock import MagicMock, AsyncMock

from visualex.graph.temporal import (
    TemporalQuery,
    VersionedNorm,
    VersionTimeline,
    VersionDiff,
    DiffSegment,
    NormStatus,
)


# =============================================================================
# TemporalQuery Tests
# =============================================================================


class TestTemporalQuery:
    """Test suite for TemporalQuery class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.query = TemporalQuery(self.mock_client)

    # --- AC1: Query with as_of_date ---

    @pytest.mark.asyncio
    async def test_get_norm_at_date_current(self):
        """Test getting current version when no date specified (AC1)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
                    "testo_vigente": "Il contratto...",
                    "data_entrata_in_vigore": "1942-04-21",
                    "titolo": "Della risoluzione del contratto",
                }
            },
            "versions": [],
            "abrogation_date": None,
        }]

        result = await self.query.get_norm_at_date(
            "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )

        assert result is not None
        assert result.urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        assert result.is_current is True
        assert result.is_abrogato is False
        assert result.status == NormStatus.VIGENTE

    @pytest.mark.asyncio
    async def test_get_norm_at_date_with_version_nodes(self):
        """Test getting version at specific date from Versione nodes (AC1)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020-12-30;178~art5",
                    "testo_vigente": "Current text",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-01-01",
                        "data_fine_validita": "2021-06-30",
                        "testo_completo": "Old text version 1",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2021-07-01",
                        "data_fine_validita": None,
                        "testo_completo": "Current text",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        # Query for date in first version
        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2020-12-30;178~art5",
            as_of_date="2020-06-15"
        )

        assert result is not None
        assert result.testo_vigente == "Old text version 1"
        assert result.is_current is False
        assert result.newer_version_exists is True
        assert result.vigenza_dal == date(2020, 1, 1)
        assert result.vigenza_al == date(2021, 6, 30)

    @pytest.mark.asyncio
    async def test_get_norm_at_exact_end_date(self):
        """Test getting version on exact end date (boundary condition) (AC1)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020-12-30;178~art5",
                    "testo_vigente": "Current text",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-01-01",
                        "data_fine_validita": "2021-06-30",
                        "testo_completo": "Old text version 1",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2021-07-01",
                        "data_fine_validita": None,
                        "testo_completo": "Current text",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        # Query for exact end date of first version - should still return it
        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2020-12-30;178~art5",
            as_of_date=date(2021, 6, 30)  # Exact end date
        )

        assert result is not None
        assert result.testo_vigente == "Old text version 1"
        assert result.vigenza_al == date(2021, 6, 30)
        # Italian multivigenza: "dal X al Y" includes Y

    # --- AC2: Pre-modification Version Retrieval ---

    @pytest.mark.asyncio
    async def test_get_pre_modification_version(self):
        """Test getting pre-modification version (AC2)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020-12-30;178~art10",
                    "testo_vigente": "Modified text after 2024-03-01",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-12-30",
                        "data_fine_validita": "2024-02-28",
                        "testo_completo": "Original text before modification",
                        "descrizione_modifiche": "Original version",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2024-03-01",
                        "data_fine_validita": None,
                        "testo_completo": "Modified text after 2024-03-01",
                        "descrizione_modifiche": "Modificato da L. 50/2024",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        # Query for date before modification (2024-02-15)
        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2020-12-30;178~art10",
            as_of_date=date(2024, 2, 15)
        )

        assert result is not None
        assert result.testo_vigente == "Original text before modification"
        assert result.newer_version_exists is True
        assert result.is_current is False

    @pytest.mark.asyncio
    async def test_get_post_modification_version(self):
        """Test getting current version after modification (AC2)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020-12-30;178~art10",
                    "testo_vigente": "Modified text",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-12-30",
                        "data_fine_validita": "2024-02-28",
                        "testo_completo": "Original text",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2024-03-01",
                        "data_fine_validita": None,
                        "testo_completo": "Modified text",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        # Query for date after modification (2024-03-15)
        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2020-12-30;178~art10",
            as_of_date=date(2024, 3, 15)
        )

        assert result is not None
        assert result.testo_vigente == "Modified text"
        assert result.is_current is True
        assert result.newer_version_exists is False

    # --- AC3: Abrogated Article Handling ---

    @pytest.mark.asyncio
    async def test_get_abrogated_norm_current(self):
        """Test getting abrogated norm without date (AC3)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2000-01-01;10~art5",
                    "testo_vigente": "This article was abrogated",
                    "stato": "abrogato",
                }
            },
            "versions": [],
            "abrogation_date": "2023-01-01",
        }]

        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2000-01-01;10~art5"
        )

        assert result is not None
        assert result.is_abrogato is True
        assert result.status == NormStatus.ABROGATO

    @pytest.mark.asyncio
    async def test_is_abrogated(self):
        """Test abrogation check (AC3)."""
        self.mock_client.query.return_value = [{
            "stato": "abrogato",
            "has_abrogation": True,
        }]

        is_abrog = await self.query.is_abrogated(
            "urn:nir:stato:legge:2000-01-01;10~art5"
        )

        assert is_abrog is True

    @pytest.mark.asyncio
    async def test_is_not_abrogated(self):
        """Test non-abrogated norm check (AC3)."""
        self.mock_client.query.return_value = [{
            "stato": "vigente",
            "has_abrogation": False,
        }]

        is_abrog = await self.query.is_abrogated(
            "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        )

        assert is_abrog is False

    @pytest.mark.asyncio
    async def test_is_abrogated_norm_not_found(self):
        """Test abrogation check returns None for non-existent norm (AC3)."""
        self.mock_client.query.return_value = []

        is_abrog = await self.query.is_abrogated(
            "urn:nir:stato:legge:9999-01-01;999"
        )

        assert is_abrog is None  # None means norm not found, not "not abrogated"

    @pytest.mark.asyncio
    async def test_get_abrogation_info(self):
        """Test getting detailed abrogation info (AC3)."""
        self.mock_client.query.return_value = [{
            "abrogating_norm": "urn:nir:stato:legge:2023-01-01;5",
            "abrogating_title": "Legge 5/2023",
            "effective_date": "2023-07-01",
        }]

        info = await self.query.get_abrogation_info(
            "urn:nir:stato:legge:2000-01-01;10~art5"
        )

        assert info is not None
        assert info["is_abrogated"] is True
        assert info["abrogating_norm_urn"] == "urn:nir:stato:legge:2023-01-01;5"
        assert info["effective_date"] == "2023-07-01"

    # --- AC4: Version Timeline ---

    @pytest.mark.asyncio
    async def test_get_version_timeline(self):
        """Test getting complete version timeline (AC4)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020-12-30;178~art5",
                    "testo_vigente": "Current text v3",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-12-30",
                        "data_fine_validita": "2021-06-30",
                        "testo_completo": "Text v1",
                        "descrizione_modifiche": "Original",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2021-07-01",
                        "data_fine_validita": "2023-12-31",
                        "testo_completo": "Text v2",
                        "descrizione_modifiche": "First amendment",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2024-01-01",
                        "data_fine_validita": None,
                        "testo_completo": "Current text v3",
                        "descrizione_modifiche": "Second amendment",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        timeline = await self.query.get_version_timeline(
            "urn:nir:stato:legge:2020-12-30;178~art5"
        )

        assert timeline.urn == "urn:nir:stato:legge:2020-12-30;178~art5"
        assert timeline.total_versions == 3
        assert len(timeline.versions) == 3

        # Check chronological order
        assert timeline.versions[0].vigenza_dal == date(2020, 12, 30)
        assert timeline.versions[1].vigenza_dal == date(2021, 7, 1)
        assert timeline.versions[2].vigenza_dal == date(2024, 1, 1)

        # Check current version
        assert timeline.current_version is not None
        assert timeline.current_version.is_current is True
        assert timeline.current_version.testo_vigente == "Current text v3"

        # Verify earlier versions marked as not current
        assert timeline.versions[0].is_current is False
        assert timeline.versions[0].newer_version_exists is True

    @pytest.mark.asyncio
    async def test_get_version_timeline_with_abrogation(self):
        """Test timeline for abrogated norm (AC4)."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2000-01-01;10~art5",
                    "testo_vigente": "Final text before abrogation",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2000-01-01",
                        "data_fine_validita": None,
                        "testo_completo": "Final text before abrogation",
                    }
                },
            ],
            "abrogation_date": "2023-01-01",
        }]

        timeline = await self.query.get_version_timeline(
            "urn:nir:stato:legge:2000-01-01;10~art5"
        )

        assert timeline.abrogation_date == date(2023, 1, 1)
        assert timeline.current_version is not None
        assert timeline.current_version.is_abrogato is True
        assert timeline.current_version.status == NormStatus.ABROGATO

    @pytest.mark.asyncio
    async def test_get_version_timeline_empty(self):
        """Test timeline for norm not found."""
        self.mock_client.query.return_value = []

        timeline = await self.query.get_version_timeline(
            "urn:nir:stato:legge:9999-01-01;999"
        )

        assert timeline.urn == "urn:nir:stato:legge:9999-01-01;999"
        assert timeline.total_versions == 0
        assert timeline.current_version is None

    # --- Edge Cases ---

    @pytest.mark.asyncio
    async def test_get_norm_not_found(self):
        """Test query for non-existent norm."""
        self.mock_client.query.return_value = []

        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:9999-01-01;999"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_get_norm_before_in_force(self):
        """Test query for date before norm came into force."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2024-01-01;1~art1",
                    "testo_vigente": "New law text",
                    "data_entrata_in_vigore": "2024-01-01",
                }
            },
            "versions": [],
            "abrogation_date": None,
        }]

        # Query for date before law came into force
        result = await self.query.get_norm_at_date(
            "urn:nir:stato:legge:2024-01-01;1~art1",
            as_of_date=date(2023, 12, 15)
        )

        # Should return None as law wasn't in force yet
        assert result is None

    @pytest.mark.asyncio
    async def test_get_versions_in_range(self):
        """Test getting versions within date range."""
        self.mock_client.query.return_value = [
            {
                "v": {
                    "properties": {
                        "data_inizio_validita": "2021-01-01",
                        "data_fine_validita": "2021-12-31",
                        "testo_completo": "Version in 2021",
                    }
                }
            },
            {
                "v": {
                    "properties": {
                        "data_inizio_validita": "2022-01-01",
                        "data_fine_validita": "2022-12-31",
                        "testo_completo": "Version in 2022",
                    }
                }
            },
        ]

        versions = await self.query.get_versions_in_range(
            "urn:nir:stato:legge:2020-12-30;178~art5",
            from_date=date(2021, 6, 1),
            to_date=date(2022, 6, 1),
        )

        assert len(versions) == 2


# =============================================================================
# Data Structure Tests
# =============================================================================


class TestVersionedNorm:
    """Test suite for VersionedNorm dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        version = VersionedNorm(
            urn="urn:nir:stato:legge:2020-12-30;178~art5",
            testo_vigente="Test text",
            vigenza_dal=date(2020, 12, 30),
            vigenza_al=date(2021, 6, 30),
            is_current=False,
            is_abrogato=False,
            newer_version_exists=True,
            status=NormStatus.VIGENTE,
        )

        d = version.to_dict()

        assert d["urn"] == "urn:nir:stato:legge:2020-12-30;178~art5"
        assert d["vigenza_dal"] == "2020-12-30"
        assert d["vigenza_al"] == "2021-06-30"
        assert d["is_current"] is False
        assert d["newer_version_exists"] is True

    def test_to_dict_null_dates(self):
        """Test to_dict with null dates."""
        version = VersionedNorm(
            urn="test",
            testo_vigente="Test",
            is_current=True,
        )

        d = version.to_dict()

        assert d["vigenza_dal"] is None
        assert d["vigenza_al"] is None


class TestVersionTimeline:
    """Test suite for VersionTimeline dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        timeline = VersionTimeline(
            urn="urn:nir:stato:legge:2020-12-30;178~art5",
            versions=[
                VersionedNorm(
                    urn="urn:nir:stato:legge:2020-12-30;178~art5",
                    testo_vigente="V1",
                    vigenza_dal=date(2020, 12, 30),
                    is_current=True,
                )
            ],
            total_versions=1,
        )

        d = timeline.to_dict()

        assert d["urn"] == "urn:nir:stato:legge:2020-12-30;178~art5"
        assert d["total_versions"] == 1
        assert len(d["versions"]) == 1


class TestNormStatus:
    """Test suite for NormStatus constants."""

    def test_status_values(self):
        """Test status constant values."""
        assert NormStatus.VIGENTE == "vigente"
        assert NormStatus.ABROGATO == "abrogato"
        assert NormStatus.SOSPESO == "sospeso"
        assert NormStatus.NON_ANCORA_IN_VIGORE == "non_ancora_in_vigore"


# =============================================================================
# Version Diff Tests (Story 3-6)
# =============================================================================


class TestDiffSegment:
    """Test suite for DiffSegment dataclass."""

    def test_creation(self):
        """Test DiffSegment creation."""
        segment = DiffSegment(text="test text", change_type="unchanged")
        assert segment.text == "test text"
        assert segment.change_type == "unchanged"

    def test_to_dict(self):
        """Test DiffSegment to_dict."""
        segment = DiffSegment(text="new text", change_type="added")
        d = segment.to_dict()
        assert d["text"] == "new text"
        assert d["change_type"] == "added"


class TestVersionDiff:
    """Test suite for VersionDiff dataclass."""

    def test_creation(self):
        """Test VersionDiff creation."""
        segments = [
            DiffSegment(text="unchanged", change_type="unchanged"),
            DiffSegment(text="old", change_type="removed"),
            DiffSegment(text="new", change_type="added"),
        ]
        diff = VersionDiff(
            urn="urn:nir:stato:legge:2020;1~art1",
            version_a_date=date(2020, 1, 1),
            version_b_date=date(2021, 1, 1),
            segments=segments,
            additions_count=1,
            deletions_count=1,
            unchanged_count=1,
        )
        assert diff.urn == "urn:nir:stato:legge:2020;1~art1"
        assert len(diff.segments) == 3
        assert diff.additions_count == 1
        assert diff.deletions_count == 1

    def test_to_dict(self):
        """Test VersionDiff to_dict."""
        diff = VersionDiff(
            urn="urn:test",
            version_a_date=date(2020, 1, 1),
            version_b_date=date(2021, 1, 1),
            segments=[DiffSegment(text="test", change_type="unchanged")],
            unchanged_count=1,
        )
        d = diff.to_dict()
        assert d["urn"] == "urn:test"
        assert d["version_a_date"] == "2020-01-01"
        assert d["version_b_date"] == "2021-01-01"
        assert len(d["segments"]) == 1


class TestVersionComparison:
    """Test suite for version comparison functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.query = AsyncMock()
        self.query = TemporalQuery(self.mock_client)

    @pytest.mark.asyncio
    async def test_compare_versions_basic(self):
        """Test basic version comparison."""
        # Mock responses for both versions
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:nir:stato:legge:2020;1~art1",
                    "testo_vigente": "Original text",
                    "data_entrata_in_vigore": "2020-01-01",
                }
            },
            "versions": [
                {
                    "properties": {
                        "data_inizio_validita": "2020-01-01",
                        "data_fine_validita": "2020-12-31",
                        "testo_completo": "The contract is valid for one year.",
                    }
                },
                {
                    "properties": {
                        "data_inizio_validita": "2021-01-01",
                        "data_fine_validita": None,
                        "testo_completo": "The contract is valid for two years.",
                    }
                },
            ],
            "abrogation_date": None,
        }]

        diff = await self.query.compare_versions(
            urn="urn:nir:stato:legge:2020;1~art1",
            date_a="2020-06-01",
            date_b="2021-06-01",
        )

        assert diff is not None
        assert diff.urn == "urn:nir:stato:legge:2020;1~art1"
        assert diff.version_a_date == date(2020, 6, 1)
        assert diff.version_b_date == date(2021, 6, 1)
        # Should have detected the change from "one" to "two"
        assert diff.additions_count > 0 or diff.deletions_count > 0

    @pytest.mark.asyncio
    async def test_compare_versions_no_changes(self):
        """Test comparing identical versions."""
        self.mock_client.query.return_value = [{
            "n": {
                "properties": {
                    "urn": "urn:test",
                    "testo_vigente": "Same text",
                    "data_entrata_in_vigore": "2020-01-01",
                }
            },
            "versions": [],
            "abrogation_date": None,
        }]

        diff = await self.query.compare_versions(
            urn="urn:test",
            date_a="2020-06-01",
            date_b="2020-12-01",
        )

        assert diff is not None
        assert diff.additions_count == 0
        assert diff.deletions_count == 0
        assert diff.unchanged_count > 0

    @pytest.mark.asyncio
    async def test_compare_versions_not_found(self):
        """Test comparison when norm not found."""
        self.mock_client.query.return_value = []

        diff = await self.query.compare_versions(
            urn="urn:nonexistent",
            date_a="2020-01-01",
            date_b="2021-01-01",
        )

        assert diff is None

    def test_compute_word_diff_additions(self):
        """Test word diff with additions."""
        text_a = "The contract is valid."
        text_b = "The new contract is valid today."

        segments, additions, deletions, unchanged = self.query._compute_word_diff(
            text_a, text_b
        )

        assert additions > 0  # "new" and "today" added
        assert unchanged > 0  # "The", "contract", "is", "valid." unchanged

    def test_compute_word_diff_deletions(self):
        """Test word diff with deletions."""
        text_a = "The old contract is very valid."
        text_b = "The contract is valid."

        segments, additions, deletions, unchanged = self.query._compute_word_diff(
            text_a, text_b
        )

        assert deletions > 0  # "old" and "very" removed

    def test_compute_word_diff_replace(self):
        """Test word diff with replacements."""
        text_a = "Article one establishes the rules."
        text_b = "Article two establishes the guidelines."

        segments, additions, deletions, unchanged = self.query._compute_word_diff(
            text_a, text_b
        )

        assert deletions > 0  # "one" and "rules" removed
        assert additions > 0  # "two" and "guidelines" added


class TestVersionedNormModifyingInfo:
    """Test modifying norm info in VersionedNorm."""

    def test_to_dict_with_modifying_norm(self):
        """Test to_dict includes modifying norm info when present."""
        version = VersionedNorm(
            urn="urn:nir:stato:legge:2020;1~art1",
            testo_vigente="Modified text",
            vigenza_dal=date(2021, 1, 1),
            is_current=True,
            modifying_norm_urn="urn:nir:stato:legge:2020;50",
            modifying_norm_title="Legge 50/2020",
        )

        d = version.to_dict()

        assert d["modifying_norm_urn"] == "urn:nir:stato:legge:2020;50"
        assert d["modifying_norm_title"] == "Legge 50/2020"

    def test_to_dict_without_modifying_norm(self):
        """Test to_dict excludes modifying norm info when not present."""
        version = VersionedNorm(
            urn="urn:test",
            testo_vigente="Original",
            is_current=True,
        )

        d = version.to_dict()

        assert "modifying_norm_urn" not in d
        assert "modifying_norm_title" not in d
