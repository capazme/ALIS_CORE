"""
Tests for URN generation functionality.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestURNGeneration:
    """Tests for generate_urn function."""

    def test_generate_urn_codice_civile(self):
        """Test URN generation for Codice Civile."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice civile",
            article="1453",
        )

        assert urn is not None
        assert "regio.decreto:1942-03-16;262" in urn
        assert "art1453" in urn

    def test_generate_urn_codice_penale(self):
        """Test URN generation for Codice Penale."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice penale",
            article="575",
        )

        assert urn is not None
        assert "regio.decreto:1930-10-19;1398" in urn
        assert "art575" in urn

    def test_generate_urn_legge_with_date(self):
        """Test URN generation for a law with full date."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="legge",
            date="2020-01-15",
            act_number="123",
            article="1",
        )

        assert urn is not None
        assert "legge:2020-01-15;123" in urn
        assert "art1" in urn

    def test_generate_urn_with_article_extension(self):
        """Test URN generation with article bis/ter extension."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice civile",
            article="2929-bis",
        )

        assert urn is not None
        assert "art2929bis" in urn

    def test_generate_urn_with_version(self):
        """Test URN generation with version info."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice civile",
            article="1453",
            version="vigente",
            version_date="2023-06-01",
        )

        assert urn is not None
        assert "!vig=2023-06-01" in urn

    def test_generate_urn_originale_version(self):
        """Test URN generation with originale version."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice civile",
            article="1453",
            version="originale",
        )

        assert urn is not None
        assert "@originale" in urn

    def test_generate_urn_with_annex(self):
        """Test URN generation with annex specified."""
        from visualex.utils.urngenerator import generate_urn

        urn = generate_urn(
            act_type="codice penale",
            article="1",
            annex="1",
        )

        assert urn is not None
        # Annex should be in URN


class TestAppendFunctions:
    """Tests for URN append helper functions."""

    def test_append_article_info_simple(self):
        """Test appending simple article info."""
        from visualex.utils.urngenerator import append_article_info

        result = append_article_info("base:urn", "42", None)

        assert result == "base:urn~art42"

    def test_append_article_info_with_extension(self):
        """Test appending article with extension."""
        from visualex.utils.urngenerator import append_article_info

        result = append_article_info("base:urn", "42-bis", None)

        assert result == "base:urn~art42bis"

    def test_append_article_info_strips_prefix(self):
        """Test that 'art.' prefix is stripped from article number."""
        from visualex.utils.urngenerator import append_article_info

        result = append_article_info("base:urn", "art. 42", None)

        assert result == "base:urn~art42"

    def test_append_version_info_vigente(self):
        """Test appending vigente version info."""
        from visualex.utils.urngenerator import append_version_info

        result = append_version_info("base:urn", "vigente", "2023-01-01")

        assert result == "base:urn!vig=2023-01-01"

    def test_append_version_info_originale(self):
        """Test appending originale version info."""
        from visualex.utils.urngenerator import append_version_info

        result = append_version_info("base:urn", "originale", None)

        assert result == "base:urn@originale"

    def test_append_version_info_none(self):
        """Test that no version info is appended when version is None."""
        from visualex.utils.urngenerator import append_version_info

        result = append_version_info("base:urn", None, None)

        assert result == "base:urn"


class TestURNToFilename:
    """Tests for URN to filename conversion."""

    def test_urn_to_filename_with_number(self):
        """Test filename generation from URN with number."""
        from visualex.utils.urngenerator import urn_to_filename

        urn = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:2020-01-15;123~art1"
        filename = urn_to_filename(urn)

        assert filename == "123_2020.pdf"

    def test_urn_to_filename_codice(self):
        """Test filename generation from codice URN."""
        from visualex.utils.urngenerator import urn_to_filename

        urn = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile"
        filename = urn_to_filename(urn)

        assert filename == "Codice.civile.pdf"


class TestCompleteDateOrParse:
    """Tests for date parsing and completion."""

    def test_complete_date_or_parse_full_date(self):
        """Test parsing full date."""
        from visualex.utils.urngenerator import complete_date_or_parse

        result = complete_date_or_parse("2020-01-15", "legge", "123")

        assert result == "2020-01-15"

    def test_complete_date_or_parse_year_only(self):
        """Test parsing year-only date (defaults to Jan 1st)."""
        from visualex.utils.urngenerator import complete_date_or_parse

        result = complete_date_or_parse("2020", "legge", "123")

        assert result == "2020-01-01"

    def test_complete_date_or_parse_none(self):
        """Test parsing None date."""
        from visualex.utils.urngenerator import complete_date_or_parse

        result = complete_date_or_parse(None, "codice civile", None)

        assert result is None

    def test_complete_date_or_parse_empty(self):
        """Test parsing empty date."""
        from visualex.utils.urngenerator import complete_date_or_parse

        result = complete_date_or_parse("", "codice civile", None)

        assert result is None
