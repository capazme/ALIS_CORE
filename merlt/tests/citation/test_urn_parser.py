"""
Tests for URN Parser
====================

Tests for parsing Italian legal URNs into structured components.
"""

import pytest

from merlt.citation.urn_parser import (
    ParsedURN,
    parse_urn,
    get_codice_abbreviation,
    get_act_type_abbreviation,
    format_italian_date,
    CODICE_ABBREVIATIONS,
    ACT_TYPE_ABBREVIATIONS,
)


class TestParseURN:
    """Tests for the parse_urn function."""

    def test_parse_codice_civile_article(self):
        """Test parsing a Codice Civile article URN."""
        urn = "urn:nir:stato:codice.civile:1942;art1453"
        result = parse_urn(urn)

        assert result.parsed_successfully
        assert result.authority == "stato"
        assert result.act_type == "codice.civile"
        assert result.year == "1942"
        assert result.article == "1453"
        assert result.is_codice
        assert result.codice_abbrev == "c.c."

    def test_parse_codice_penale_article(self):
        """Test parsing a Codice Penale article URN."""
        urn = "urn:nir:stato:codice.penale:1930;art575"
        result = parse_urn(urn)

        assert result.parsed_successfully
        assert result.act_type == "codice.penale"
        assert result.article == "575"
        assert result.is_codice
        assert result.codice_abbrev == "c.p."

    def test_parse_legge_with_full_date(self):
        """Test parsing a law with full date."""
        urn = "urn:nir:stato:legge:1990-08-07;241~art1"
        result = parse_urn(urn)

        assert result.parsed_successfully
        assert result.authority == "stato"
        assert result.act_type == "legge"
        assert result.date == "1990-08-07"
        assert result.year == "1990"
        assert result.act_number == "241"
        assert result.article == "1"
        assert not result.is_codice

    def test_parse_decreto_legislativo(self):
        """Test parsing a decreto legislativo."""
        urn = "urn:nir:stato:decreto.legislativo:2003-06-30;196~art1"
        result = parse_urn(urn)

        assert result.parsed_successfully
        assert result.act_type == "decreto.legislativo"
        assert result.date == "2003-06-30"
        assert result.act_number == "196"
        assert result.article == "1"

    def test_parse_costituzione(self):
        """Test parsing a Constitution article."""
        urn = "urn:nir:stato:costituzione;art24"
        result = parse_urn(urn)

        assert result.authority == "stato"
        assert result.act_type == "costituzione"

    def test_parse_article_with_bis(self):
        """Test parsing an article with 'bis' suffix."""
        urn = "urn:nir:stato:codice.civile:1942;art1453-bis"
        result = parse_urn(urn)

        assert result.article == "1453-bis"

    def test_parse_article_with_comma(self):
        """Test parsing an article with comma specification."""
        urn = "urn:nir:stato:codice.civile:1942;art1453~com1"
        result = parse_urn(urn)

        assert result.article == "1453"
        assert result.comma == "1"

    def test_parse_normattiva_url(self):
        """Test parsing a Normattiva URL."""
        url = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:codice.civile:1942;art1453"
        result = parse_urn(url)

        assert result.parsed_successfully
        assert result.act_type == "codice.civile"
        assert result.article == "1453"

    def test_parse_invalid_urn_returns_raw(self):
        """Test that invalid URNs still return a ParsedURN with raw value."""
        urn = "invalid:urn:format"
        result = parse_urn(urn)

        assert result.raw_urn == urn
        # May or may not parse successfully depending on format

    def test_parse_empty_urn(self):
        """Test parsing an empty URN."""
        result = parse_urn("")
        assert result.raw_urn == ""

    def test_parse_codice_procedura_civile(self):
        """Test parsing Codice di Procedura Civile."""
        urn = "urn:nir:stato:codice.di.procedura.civile:1940;art100"
        result = parse_urn(urn)

        assert result.is_codice
        assert result.article == "100"


class TestGetCodiceAbbreviation:
    """Tests for the get_codice_abbreviation function."""

    def test_codice_civile(self):
        """Test abbreviation for Codice Civile."""
        assert get_codice_abbreviation("codice.civile") == "c.c."

    def test_codice_penale(self):
        """Test abbreviation for Codice Penale."""
        assert get_codice_abbreviation("codice.penale") == "c.p."

    def test_codice_procedura_civile(self):
        """Test abbreviation for Codice di Procedura Civile."""
        assert get_codice_abbreviation("codice.di.procedura.civile") == "c.p.c."

    def test_codice_procedura_penale(self):
        """Test abbreviation for Codice di Procedura Penale."""
        assert get_codice_abbreviation("codice.di.procedura.penale") == "c.p.p."

    def test_costituzione(self):
        """Test abbreviation for Costituzione."""
        assert get_codice_abbreviation("costituzione") == "Cost."

    def test_normalized_form(self):
        """Test abbreviation with normalized form (spaces instead of dots)."""
        assert get_codice_abbreviation("codice civile") == "c.c."

    def test_unknown_codice(self):
        """Test that unknown codes return None."""
        assert get_codice_abbreviation("unknown.code") is None


class TestGetActTypeAbbreviation:
    """Tests for the get_act_type_abbreviation function."""

    def test_legge(self):
        """Test abbreviation for legge."""
        assert get_act_type_abbreviation("legge") == "L."

    def test_decreto_legislativo(self):
        """Test abbreviation for decreto legislativo."""
        assert get_act_type_abbreviation("decreto.legislativo") == "D.lgs."

    def test_decreto_legge(self):
        """Test abbreviation for decreto legge."""
        assert get_act_type_abbreviation("decreto.legge") == "D.L."

    def test_dpr(self):
        """Test abbreviation for DPR."""
        assert get_act_type_abbreviation("decreto.del.presidente.della.repubblica") == "D.P.R."

    def test_regio_decreto(self):
        """Test abbreviation for regio decreto."""
        assert get_act_type_abbreviation("regio.decreto") == "R.D."

    def test_unknown_act_type(self):
        """Test that unknown act types return None."""
        assert get_act_type_abbreviation("unknown.type") is None


class TestFormatItalianDate:
    """Tests for the format_italian_date function."""

    def test_full_date(self):
        """Test formatting a full date."""
        assert format_italian_date("1990-08-07") == "7 agosto 1990"

    def test_first_day_of_month(self):
        """Test formatting the first day of a month."""
        assert format_italian_date("2020-01-01") == "1 gennaio 2020"

    def test_december_date(self):
        """Test formatting a December date."""
        assert format_italian_date("2020-12-30") == "30 dicembre 2020"

    def test_march_date(self):
        """Test formatting a March date."""
        assert format_italian_date("1942-03-16") == "16 marzo 1942"

    def test_invalid_date_returns_original(self):
        """Test that invalid dates return the original string."""
        assert format_italian_date("invalid") == "invalid"
        assert format_italian_date("1990") == "1990"


class TestParsedURNAttributes:
    """Tests for ParsedURN dataclass attributes."""

    def test_default_values(self):
        """Test default values of ParsedURN."""
        parsed = ParsedURN(raw_urn="test")

        assert parsed.raw_urn == "test"
        assert parsed.authority == ""
        assert parsed.act_type == ""
        assert parsed.date is None
        assert parsed.act_number is None
        assert parsed.article is None
        assert parsed.comma is None
        assert parsed.is_codice is False
        assert parsed.codice_abbrev is None
        assert parsed.codice_full_name is None
        assert parsed.year is None
        assert parsed.parsed_successfully is False
        assert parsed.parse_errors == []

    def test_full_initialization(self):
        """Test full initialization of ParsedURN."""
        parsed = ParsedURN(
            raw_urn="urn:nir:stato:codice.civile:1942;art1453",
            authority="stato",
            act_type="codice.civile",
            year="1942",
            article="1453",
            is_codice=True,
            codice_abbrev="c.c.",
            codice_full_name="Codice Civile",
            parsed_successfully=True,
        )

        assert parsed.authority == "stato"
        assert parsed.is_codice
        assert parsed.codice_abbrev == "c.c."


class TestEdgeCases:
    """Tests for edge cases and unusual URN formats."""

    def test_urn_with_multiple_tildes(self):
        """Test URN with multiple tilde separators."""
        urn = "urn:nir:stato:legge:2020-12-30;178~art1~com2"
        result = parse_urn(urn)
        # Should handle gracefully
        assert result.act_type == "legge"

    def test_urn_with_only_year(self):
        """Test URN with only year (no full date)."""
        urn = "urn:nir:stato:codice.civile:1942;art1"
        result = parse_urn(urn)

        assert result.year == "1942"
        assert result.date is None

    def test_urn_without_article(self):
        """Test URN without article specification."""
        urn = "urn:nir:stato:legge:2020-12-30;178"
        result = parse_urn(urn)

        assert result.act_number == "178"
        assert result.article is None

    def test_regional_authority(self):
        """Test URN with regional authority."""
        urn = "urn:nir:regione.lombardia:legge:2020-01-01;1~art1"
        result = parse_urn(urn)

        assert result.authority == "regione.lombardia"


class TestFormatItalianDateEdgeCases:
    """Tests for format_italian_date edge cases."""

    def test_month_zero_returns_original(self):
        """Test that month 0 returns original string."""
        assert format_italian_date("2020-00-15") == "2020-00-15"

    def test_month_13_returns_original(self):
        """Test that month 13 returns original string."""
        assert format_italian_date("2020-13-15") == "2020-13-15"

    def test_month_negative_returns_original(self):
        """Test that negative month returns original string."""
        assert format_italian_date("2020--1-15") == "2020--1-15"

    def test_non_numeric_parts(self):
        """Test that non-numeric date parts return original."""
        assert format_italian_date("abcd-ef-gh") == "abcd-ef-gh"

    def test_empty_date_parts(self):
        """Test date with too few parts."""
        assert format_italian_date("2020-01") == "2020-01"
