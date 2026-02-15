"""
Tests for Citation Formatter
============================

Tests for the CitationFormatter service and format generators.
"""

import json
import pytest

from merlt.citation.formatter import CitationFormatter, CitationFormat, FormattedCitation
from merlt.citation.formats.italian_legal import ItalianLegalFormat, FormattedSource
from merlt.citation.formats.bibtex import BibTeXFormat, BibTeXEntry
from merlt.citation.formats.plain_text import PlainTextFormat
from merlt.citation.formats.json_format import JSONFormat


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def formatter():
    """Create a CitationFormatter instance."""
    return CitationFormatter(alis_version="MERL-T v1.0-test")


@pytest.fixture
def sample_sources():
    """Sample sources for testing."""
    return [
        {
            "article_urn": "urn:nir:stato:codice.civile:1942;art1453",
            "expert": "literal",
            "relevance": 0.95,
            "title": "Risolubilità del contratto per inadempimento",
        },
        {
            "article_urn": "urn:nir:stato:legge:1990-08-07;241~art1",
            "expert": "systemic",
            "relevance": 0.85,
            "title": "Norme in materia di procedimento amministrativo",
        },
        {
            "article_urn": "urn:nir:stato:decreto.legislativo:2003-06-30;196~art1",
            "expert": "principles",
            "relevance": 0.80,
        },
    ]


@pytest.fixture
def giurisprudenza_source():
    """Sample giurisprudenza source."""
    return {
        "article_urn": "",
        "expert": "precedent",
        "relevance": 0.90,
        "type": "giurisprudenza",
        "title": "Cass. Civ., Sez. II, 15/03/2023, n. 1234",
    }


# =============================================================================
# CITATION FORMATTER TESTS
# =============================================================================


class TestCitationFormatter:
    """Tests for the CitationFormatter service."""

    def test_init_default_version(self):
        """Test default ALIS version."""
        fmt = CitationFormatter()
        assert fmt.alis_version == "MERL-T v1.0"

    def test_init_custom_version(self):
        """Test custom ALIS version."""
        fmt = CitationFormatter(alis_version="v2.0")
        assert fmt.alis_version == "v2.0"

    def test_format_sources_italian_legal(self, formatter, sample_sources):
        """Test formatting in Italian legal style."""
        result = formatter.format_sources(
            sample_sources,
            CitationFormat.ITALIAN_LEGAL,
            query_summary="Risoluzione contratto",
        )

        assert "FONTI GIURIDICHE" in result
        assert "Art. 1453 c.c." in result
        assert "L. 7 agosto 1990, n. 241" in result
        assert "ALIS" in result
        assert "Risoluzione contratto" in result

    def test_format_sources_bibtex(self, formatter, sample_sources):
        """Test formatting in BibTeX style."""
        result = formatter.format_sources(
            sample_sources,
            CitationFormat.BIBTEX,
        )

        assert "@legislation{" in result
        assert "title = {" in result
        assert "ALIS Citation Export" in result

    def test_format_sources_plain_text(self, formatter, sample_sources):
        """Test formatting in plain text style."""
        result = formatter.format_sources(
            sample_sources,
            CitationFormat.PLAIN_TEXT,
        )

        assert "Fonti giuridiche consultate" in result
        assert "1." in result
        assert "Art. 1453 c.c." in result

    def test_format_sources_json(self, formatter, sample_sources):
        """Test formatting in JSON style."""
        result = formatter.format_sources(
            sample_sources,
            CitationFormat.JSON,
        )

        # Should be valid JSON
        data = json.loads(result)
        assert "citations" in data
        assert "metadata" in data
        assert "summary" in data
        assert len(data["citations"]) == 3

    def test_format_sources_string_format(self, formatter, sample_sources):
        """Test that string format names work."""
        result = formatter.format_sources(
            sample_sources,
            "italian_legal",
        )
        assert "FONTI GIURIDICHE" in result

    def test_format_sources_invalid_format(self, formatter, sample_sources):
        """Test that invalid format raises error."""
        with pytest.raises(ValueError):
            formatter.format_sources(sample_sources, "invalid_format")

    def test_format_single(self, formatter):
        """Test formatting a single source."""
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = formatter.format_single(source, CitationFormat.ITALIAN_LEGAL)

        assert isinstance(result, FormattedCitation)
        assert "Art. 1453 c.c." in result.text
        assert result.format == CitationFormat.ITALIAN_LEGAL

    def test_format_single_bibtex(self, formatter):
        """Test formatting a single source in BibTeX."""
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = formatter.format_single(source, CitationFormat.BIBTEX)

        assert "@legislation{" in result.text
        assert "Art. 1453" in result.text

    def test_get_file_extension(self, formatter):
        """Test getting file extensions."""
        assert formatter.get_file_extension(CitationFormat.ITALIAN_LEGAL) == "txt"
        assert formatter.get_file_extension(CitationFormat.BIBTEX) == "bib"
        assert formatter.get_file_extension(CitationFormat.PLAIN_TEXT) == "txt"
        assert formatter.get_file_extension(CitationFormat.JSON) == "json"

    def test_get_media_type(self, formatter):
        """Test getting media types."""
        assert "text/plain" in formatter.get_media_type(CitationFormat.ITALIAN_LEGAL)
        assert "bibtex" in formatter.get_media_type(CitationFormat.BIBTEX)
        assert "json" in formatter.get_media_type(CitationFormat.JSON)

    def test_list_formats(self):
        """Test listing available formats."""
        formats = CitationFormatter.list_formats()

        assert len(formats) == 4
        names = [f["name"] for f in formats]
        assert "italian_legal" in names
        assert "bibtex" in names
        assert "plain_text" in names
        assert "json" in names

    def test_format_without_attribution(self, formatter, sample_sources):
        """Test formatting without ALIS attribution."""
        result = formatter.format_sources(
            sample_sources,
            CitationFormat.ITALIAN_LEGAL,
            include_attribution=False,
        )

        assert "Elaborazione a cura di ALIS" not in result


# =============================================================================
# ITALIAN LEGAL FORMAT TESTS
# =============================================================================


class TestItalianLegalFormat:
    """Tests for the ItalianLegalFormat generator."""

    def test_format_codice_article(self):
        """Test formatting a code article."""
        fmt = ItalianLegalFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = fmt.format_source(source)

        assert result.text == "Art. 1453 c.c."
        assert result.category == "norme"

    def test_format_legge(self):
        """Test formatting a law."""
        fmt = ItalianLegalFormat()
        source = {"article_urn": "urn:nir:stato:legge:1990-08-07;241~art1"}
        result = fmt.format_source(source)

        assert "L." in result.text
        assert "7 agosto 1990" in result.text
        assert "n. 241" in result.text
        assert "art. 1" in result.text

    def test_format_giurisprudenza(self):
        """Test formatting case law."""
        fmt = ItalianLegalFormat()
        source = {
            "expert": "precedent",
            "type": "giurisprudenza",
            "title": "Cass. Civ., Sez. II, 15/03/2023, n. 1234",
        }
        result = fmt.format_source(source)

        assert result.category == "giurisprudenza"
        assert "Cass. Civ." in result.text

    def test_format_all_with_sections(self, sample_sources, giurisprudenza_source):
        """Test that format_all creates sections."""
        fmt = ItalianLegalFormat()
        sources = sample_sources + [giurisprudenza_source]
        result = fmt.format_all(sources, query_summary="Test query")

        assert "NORME" in result
        assert "GIURISPRUDENZA" in result
        assert "Test query" in result

    def test_format_all_deduplicates(self):
        """Test that format_all deduplicates citations."""
        fmt = ItalianLegalFormat()
        sources = [
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},  # duplicate
        ]
        result = fmt.format_all(sources)

        # Should only appear once
        assert result.count("Art. 1453 c.c.") == 1


# =============================================================================
# BIBTEX FORMAT TESTS
# =============================================================================


class TestBibTeXFormat:
    """Tests for the BibTeXFormat generator."""

    def test_format_codice_entry(self):
        """Test formatting a code entry."""
        fmt = BibTeXFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        entry = fmt.format_source(source)

        assert entry.entry_type == "legislation"
        assert "1453" in entry.cite_key
        assert "cc" in entry.cite_key
        assert "Art. 1453" in entry.fields["title"]

    def test_format_legge_entry(self):
        """Test formatting a law entry."""
        fmt = BibTeXFormat()
        source = {"article_urn": "urn:nir:stato:legge:1990-08-07;241~art1"}
        entry = fmt.format_source(source)

        assert entry.entry_type == "legislation"
        assert "1990" in entry.fields.get("year", "")

    def test_cite_key_uniqueness(self):
        """Test that cite keys are unique."""
        fmt = BibTeXFormat()
        sources = [
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
        ]

        entry1 = fmt.format_source(sources[0])
        entry2 = fmt.format_source(sources[1])

        assert entry1.cite_key != entry2.cite_key

    def test_escape_special_characters(self):
        """Test that special characters are escaped."""
        fmt = BibTeXFormat()
        source = {
            "article_urn": "urn:nir:stato:codice.civile:1942;art1453",
            "title": "Test & Title with % special # chars",
        }
        entry = fmt.format_source(source)

        # Check escaping in note field
        if "note" in entry.fields:
            assert "&" not in entry.fields["note"] or "\\&" in entry.fields["note"]

    def test_format_all_creates_valid_bibtex(self, sample_sources):
        """Test that format_all creates valid BibTeX."""
        fmt = BibTeXFormat()
        result = fmt.format_all(sample_sources)

        # Should have BibTeX structure
        assert "@legislation{" in result
        assert "title = {" in result
        assert "}" in result


# =============================================================================
# PLAIN TEXT FORMAT TESTS
# =============================================================================


class TestPlainTextFormat:
    """Tests for the PlainTextFormat generator."""

    def test_format_numbered_list(self, sample_sources):
        """Test that output is a numbered list."""
        fmt = PlainTextFormat()
        result = fmt.format_all(sample_sources)

        assert "1." in result
        assert "2." in result
        assert "3." in result

    def test_format_deduplicates(self):
        """Test that duplicates are removed."""
        fmt = PlainTextFormat()
        sources = [
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
        ]
        result = fmt.format_all(sources)

        assert result.count("Art. 1453 c.c.") == 1


# =============================================================================
# JSON FORMAT TESTS
# =============================================================================


class TestJSONFormat:
    """Tests for the JSONFormat generator."""

    def test_format_valid_json(self, sample_sources):
        """Test that output is valid JSON."""
        fmt = JSONFormat()
        result = fmt.format_all(sample_sources)

        data = json.loads(result)
        assert isinstance(data, dict)

    def test_format_has_required_fields(self, sample_sources):
        """Test that JSON has required fields."""
        fmt = JSONFormat()
        result = fmt.format_all(sample_sources)
        data = json.loads(result)

        assert "metadata" in data
        assert "citations" in data
        assert "summary" in data

    def test_format_citation_structure(self):
        """Test structure of individual citations."""
        fmt = JSONFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = fmt.format_source(source)

        assert "raw_urn" in result
        assert "formatted" in result
        assert "components" in result
        assert "source_metadata" in result

    def test_format_summary_counts(self, sample_sources):
        """Test that summary has correct counts."""
        fmt = JSONFormat()
        result = fmt.format_all(sample_sources)
        data = json.loads(result)

        assert data["summary"]["total_citations"] == 3
        assert "by_category" in data["summary"]
        assert "by_expert" in data["summary"]

    def test_format_all_dict(self, sample_sources):
        """Test format_all_dict returns dict."""
        fmt = JSONFormat()
        result = fmt.format_all_dict(sample_sources)

        assert isinstance(result, dict)
        assert "citations" in result

    def test_format_deduplicates_by_urn(self):
        """Test that JSON format deduplicates by URN."""
        fmt = JSONFormat()
        sources = [
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
            {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"},
        ]
        result = fmt.format_all(sources)
        data = json.loads(result)

        assert len(data["citations"]) == 1


# =============================================================================
# UTF-8 AND ITALIAN CHARACTER TESTS
# =============================================================================


class TestUTF8Support:
    """Tests for UTF-8 and Italian character support."""

    def test_italian_characters_preserved(self, formatter):
        """Test that Italian characters are preserved."""
        sources = [
            {
                "article_urn": "urn:nir:stato:legge:1990-08-07;241",
                "title": "Legge sulla trasparenza amministrativa - attività",
            }
        ]
        result = formatter.format_sources(sources, CitationFormat.ITALIAN_LEGAL)

        # Should contain the month name with accent
        assert "agosto" in result

    def test_json_preserves_unicode(self, formatter):
        """Test that JSON format preserves Unicode characters."""
        sources = [
            {
                "article_urn": "urn:nir:stato:codice.civile:1942;art1453",
                "title": "Risolubilità per inadempimento - obbligatorietà",
            }
        ]
        result = formatter.format_sources(sources, CitationFormat.JSON)

        # Should use ensure_ascii=False
        assert "à" in result or "Risolubilità" in result


# =============================================================================
# ERROR PATH TESTS
# =============================================================================


class TestErrorPaths:
    """Tests for error handling and edge cases."""

    def test_format_source_without_urn(self, formatter):
        """Test formatting a source with no URN."""
        source = {"title": "Some title", "expert": "literal"}
        result = formatter.format_single(source, CitationFormat.ITALIAN_LEGAL)

        assert isinstance(result, FormattedCitation)
        assert "Some title" in result.text

    def test_format_empty_sources_list(self, formatter):
        """Test formatting an empty sources list."""
        result = formatter.format_sources([], CitationFormat.ITALIAN_LEGAL)
        assert "FONTI GIURIDICHE" in result

    def test_format_source_with_malformed_urn(self, formatter):
        """Test formatting a source with a malformed URN."""
        source = {"article_urn": "not-a-valid-urn"}
        result = formatter.format_single(source, CitationFormat.ITALIAN_LEGAL)
        assert isinstance(result, FormattedCitation)

    def test_bibtex_format_source_text(self):
        """Test BibTeX format_source_text returns string."""
        fmt = BibTeXFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = fmt.format_source_text(source)
        assert isinstance(result, str)
        assert "@legislation{" in result

    def test_json_format_source_text(self):
        """Test JSON format_source_text returns string."""
        fmt = JSONFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = fmt.format_source_text(source)
        assert isinstance(result, str)
        data = json.loads(result)
        assert "raw_urn" in data

    def test_plain_text_format_source_text(self):
        """Test PlainText format_source_text returns string."""
        fmt = PlainTextFormat()
        source = {"article_urn": "urn:nir:stato:codice.civile:1942;art1453"}
        result = fmt.format_source_text(source)
        assert isinstance(result, str)
        assert "Art. 1453 c.c." in result
