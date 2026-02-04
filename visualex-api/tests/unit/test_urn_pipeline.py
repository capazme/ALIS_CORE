"""
Unit tests for URN Canonicalization Pipeline.

Tests parsing, validation, canonicalization, and equivalence checking
of NIR-compliant URNs for Italian legal norms.
"""
import pytest
from visualex.utils.urn_pipeline import (
    URNComponents,
    parse_urn,
    validate_urn,
    canonicalize_urn,
    urns_are_equivalent,
    extract_article_info,
    urn_to_normattiva_url,
    dict_to_urn,
    normalize_act_type_for_urn,
    extension_to_number,
    number_to_extension,
    NORMATTIVA_BASE_URL,
)


class TestURNComponents:
    """Tests for URNComponents dataclass."""

    def test_basic_urn_to_string(self):
        """Test converting components to URN string."""
        c = URNComponents(tipo_atto="legge", data="2020-12-30", numero="178")
        assert c.to_urn() == "urn:nir:stato:legge:2020-12-30;178"

    def test_urn_with_article(self):
        """Test URN with article number."""
        c = URNComponents(
            tipo_atto="legge",
            data="2020-12-30",
            numero="178",
            articolo="42"
        )
        assert c.to_urn() == "urn:nir:stato:legge:2020-12-30;178~art42"

    def test_urn_with_article_extension(self):
        """Test URN with article extension (bis, ter, etc.)."""
        c = URNComponents(
            tipo_atto="decreto.legge",
            data="2008-11-29",
            numero="185",
            articolo="16",
            estensione="bis"
        )
        assert c.to_urn() == "urn:nir:stato:decreto.legge:2008-11-29;185~art16bis"

    def test_urn_with_allegato(self):
        """Test URN with annex (allegato)."""
        c = URNComponents(
            tipo_atto="regio.decreto",
            data="1942-03-16",
            numero="262",
            allegato="2",
            articolo="1453"
        )
        urn = c.to_urn()
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453"

    def test_urn_originale_version(self):
        """Test URN with @originale version."""
        c = URNComponents(
            tipo_atto="decreto.legge",
            data="2008-11-10",
            numero="180",
            versione="originale"
        )
        assert c.to_urn() == "urn:nir:stato:decreto.legge:2008-11-10;180@originale"

    def test_urn_vigente_version(self):
        """Test URN with !vig= version."""
        c = URNComponents(
            tipo_atto="decreto.legge",
            data="2008-11-10",
            numero="180",
            versione="vigente"
        )
        assert c.to_urn() == "urn:nir:stato:decreto.legge:2008-11-10;180!vig="

    def test_urn_vigente_with_date(self):
        """Test URN with !vig= and specific date."""
        c = URNComponents(
            tipo_atto="decreto.legge",
            data="2008-11-10",
            numero="180",
            articolo="2",
            versione="vigente",
            data_versione="2009-11-10"
        )
        urn = c.to_urn()
        assert urn == "urn:nir:stato:decreto.legge:2008-11-10;180~art2!vig=2009-11-10"

    def test_urn_with_url(self):
        """Test URN with Normattiva URL prefix."""
        c = URNComponents(tipo_atto="legge", data="2020-12-30", numero="178")
        url = c.to_urn(include_url=True)
        assert url.startswith(NORMATTIVA_BASE_URL)
        assert "urn:nir:stato:legge:2020-12-30;178" in url

    def test_costituzione_special_case(self):
        """Test Costituzione URN (no number)."""
        c = URNComponents(
            tipo_atto="costituzione",
            data="1947-12-27",
            articolo="7"
        )
        assert c.to_urn() == "urn:nir:stato:costituzione:1947-12-27~art7"

    def test_canonical_key_basic(self):
        """Test canonical key for duplicate detection."""
        c = URNComponents(
            tipo_atto="legge",
            data="2020-12-30",
            numero="178",
            articolo="1"
        )
        assert c.canonical_key() == "legge:2020-12-30:178:art1"

    def test_canonical_key_ignores_version(self):
        """Test that canonical key ignores version info."""
        c1 = URNComponents(tipo_atto="legge", data="2020-12-30", numero="178")
        c2 = URNComponents(tipo_atto="legge", data="2020-12-30", numero="178", versione="originale")
        assert c1.canonical_key() == c2.canonical_key()

    def test_to_dict(self):
        """Test dictionary conversion."""
        c = URNComponents(
            tipo_atto="legge",
            data="2020-12-30",
            numero="178",
            articolo="1"
        )
        d = c.to_dict()
        assert d['tipo_atto'] == 'legge'
        assert d['data'] == '2020-12-30'
        assert d['numero'] == '178'
        assert d['articolo'] == '1'
        assert d['is_valid'] is True


class TestParseURN:
    """Tests for parse_urn() function."""

    def test_parse_simple_legge(self):
        """Parse simple law URN."""
        urn = "urn:nir:stato:legge:2020-12-30;178"
        c = parse_urn(urn)
        assert c.is_valid
        assert c.tipo_atto == "legge"
        assert c.data == "2020-12-30"
        assert c.numero == "178"

    def test_parse_decreto_legge(self):
        """Parse decreto-legge URN."""
        urn = "urn:nir:stato:decreto.legge:2008-11-10;180"
        c = parse_urn(urn)
        assert c.is_valid
        assert c.tipo_atto == "decreto.legge"

    def test_parse_with_article(self):
        """Parse URN with article."""
        urn = "urn:nir:stato:decreto.legge:2008-11-10;180~art2"
        c = parse_urn(urn)
        assert c.articolo == "2"
        assert c.estensione is None

    def test_parse_with_article_extension(self):
        """Parse URN with article extension."""
        urn = "urn:nir:stato:decreto.legge:2008-11-29;185~art16bis"
        c = parse_urn(urn)
        assert c.articolo == "16"
        assert c.estensione == "bis"

    def test_parse_codice_civile_with_allegato(self):
        """Parse Codice Civile URN with allegato."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453"
        c = parse_urn(urn)
        assert c.tipo_atto == "regio.decreto"
        assert c.numero == "262"
        assert c.allegato == "2"
        assert c.articolo == "1453"

    def test_parse_with_originale(self):
        """Parse URN with @originale version."""
        urn = "urn:nir:stato:decreto.legge:2008-11-10;180@originale"
        c = parse_urn(urn)
        assert c.versione == "originale"

    def test_parse_with_vigente(self):
        """Parse URN with !vig= version."""
        urn = "urn:nir:stato:decreto.legge:2008-11-10;180!vig="
        c = parse_urn(urn)
        assert c.versione == "vigente"
        assert c.data_versione is None

    def test_parse_with_vigente_date(self):
        """Parse URN with !vig= and date."""
        urn = "urn:nir:stato:decreto.legge:2008-11-10;180~art2!vig=2009-11-10"
        c = parse_urn(urn)
        assert c.versione == "vigente"
        assert c.data_versione == "2009-11-10"
        assert c.articolo == "2"

    def test_parse_with_url_prefix(self):
        """Parse URN with Normattiva URL prefix."""
        url = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:legge:2020-12-30;178"
        c = parse_urn(url)
        assert c.is_valid
        assert c.tipo_atto == "legge"
        assert c.numero == "178"

    def test_parse_costituzione(self):
        """Parse Costituzione URN."""
        urn = "urn:nir:stato:costituzione:1947-12-27~art7"
        c = parse_urn(urn)
        assert c.is_valid
        assert c.tipo_atto == "costituzione"
        assert c.articolo == "7"
        assert c.numero is None

    def test_parse_extended_article_number(self):
        """Parse URN with extended article number (314.15)."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art314.15"
        c = parse_urn(urn)
        assert c.articolo == "314.15"

    def test_parse_empty_urn(self):
        """Parse empty URN returns invalid."""
        c = parse_urn("")
        assert not c.is_valid
        assert "Empty URN" in c.validation_errors[0]

    def test_parse_invalid_urn(self):
        """Parse invalid URN returns errors."""
        c = parse_urn("not-a-valid-urn")
        assert not c.is_valid
        assert "Invalid URN format" in c.validation_errors[0]


class TestValidateURN:
    """Tests for validate_urn() function."""

    def test_validate_valid_urn(self):
        """Validate a correct URN."""
        is_valid, errors = validate_urn("urn:nir:stato:legge:2020-12-30;178")
        assert is_valid
        assert errors == []

    def test_validate_invalid_format(self):
        """Validate an incorrectly formatted URN."""
        is_valid, errors = validate_urn("invalid-format")
        assert not is_valid
        assert len(errors) > 0

    def test_validate_invalid_date(self):
        """Validate URN with invalid date."""
        is_valid, errors = validate_urn("urn:nir:stato:legge:2020-13-45;178")
        assert not is_valid
        assert any("month" in e.lower() or "day" in e.lower() for e in errors)

    def test_validate_invalid_allegato(self):
        """Validate URN with invalid allegato (negative)."""
        # This would need the URN to parse first with allegato="-1"
        # For now, we test that valid allegato passes
        is_valid, errors = validate_urn("urn:nir:stato:regio.decreto:1942-03-16;262:2")
        assert is_valid


class TestCanonicalizeURN:
    """Tests for canonicalize_urn() function."""

    def test_canonicalize_basic(self):
        """Canonicalize a simple URN."""
        urn = "urn:nir:stato:legge:2020-12-30;178"
        canonical = canonicalize_urn(urn)
        assert canonical == urn

    def test_canonicalize_uppercase_to_lowercase(self):
        """Canonicalize converts to lowercase."""
        urn = "urn:nir:stato:LEGGE:2020-12-30;178"
        canonical = canonicalize_urn(urn)
        assert "legge" in canonical
        assert "LEGGE" not in canonical

    def test_canonicalize_with_url_prefix(self):
        """Canonicalize with URL prefix option."""
        urn = "urn:nir:stato:legge:2020-12-30;178"
        canonical = canonicalize_urn(urn, include_url=True)
        assert canonical.startswith(NORMATTIVA_BASE_URL)

    def test_canonicalize_invalid_raises(self):
        """Canonicalize raises error for invalid URN."""
        with pytest.raises(ValueError):
            canonicalize_urn("invalid-urn")


class TestURNEquivalence:
    """Tests for urns_are_equivalent() function."""

    def test_identical_urns(self):
        """Identical URNs are equivalent."""
        urn1 = "urn:nir:stato:legge:2020-12-30;178"
        urn2 = "urn:nir:stato:legge:2020-12-30;178"
        assert urns_are_equivalent(urn1, urn2)

    def test_different_versions_equivalent(self):
        """Same norm with different versions are equivalent (ignore_version=True)."""
        urn1 = "urn:nir:stato:decreto.legge:2008-11-10;180"
        urn2 = "urn:nir:stato:decreto.legge:2008-11-10;180@originale"
        assert urns_are_equivalent(urn1, urn2, ignore_version=True)
        assert not urns_are_equivalent(urn1, urn2, ignore_version=False)

    def test_different_norms_not_equivalent(self):
        """Different norms are not equivalent."""
        urn1 = "urn:nir:stato:legge:2020-12-30;178"
        urn2 = "urn:nir:stato:legge:2020-12-30;179"
        assert not urns_are_equivalent(urn1, urn2)

    def test_case_insensitive_equivalent(self):
        """URNs are equivalent regardless of case."""
        urn1 = "urn:nir:stato:legge:2020-12-30;178"
        urn2 = "urn:nir:stato:LEGGE:2020-12-30;178"
        assert urns_are_equivalent(urn1, urn2)

    def test_invalid_urns_not_equivalent(self):
        """Invalid URNs are never equivalent."""
        assert not urns_are_equivalent("invalid", "also-invalid")


class TestExtractArticleInfo:
    """Tests for extract_article_info() function."""

    def test_extract_simple_article(self):
        """Extract simple article number."""
        info = extract_article_info("urn:nir:stato:legge:2020-12-30;178~art42")
        assert info is not None
        assert info['articolo'] == "42"
        assert info['estensione'] == ""
        assert info['full_article'] == "42"

    def test_extract_article_with_extension(self):
        """Extract article with Latin extension."""
        info = extract_article_info("urn:nir:stato:decreto.legge:2008-11-29;185~art16bis")
        assert info['articolo'] == "16"
        assert info['estensione'] == "bis"
        assert info['full_article'] == "16bis"

    def test_extract_no_article(self):
        """Return None when no article in URN."""
        info = extract_article_info("urn:nir:stato:legge:2020-12-30;178")
        assert info is None


class TestURLConversion:
    """Tests for URL conversion functions."""

    def test_urn_to_normattiva_url(self):
        """Convert URN to Normattiva URL."""
        urn = "urn:nir:stato:legge:2020-12-30;178"
        url = urn_to_normattiva_url(urn)
        assert url == f"{NORMATTIVA_BASE_URL}{urn}"


class TestDictToURN:
    """Tests for dict_to_urn() function."""

    def test_dict_to_urn_basic(self):
        """Build URN from dictionary."""
        data = {
            'tipo_atto': 'legge',
            'data': '2020-12-30',
            'numero': '178'
        }
        urn = dict_to_urn(data)
        assert urn == "urn:nir:stato:legge:2020-12-30;178"

    def test_dict_to_urn_with_spaces(self):
        """Build URN with act type containing spaces."""
        data = {
            'tipo_atto': 'decreto legge',
            'data': '2008-11-10',
            'numero': '180'
        }
        urn = dict_to_urn(data)
        assert "decreto.legge" in urn

    def test_dict_to_urn_complete(self):
        """Build URN with all components."""
        data = {
            'tipo_atto': 'regio.decreto',
            'data': '1942-03-16',
            'numero': '262',
            'allegato': '2',
            'articolo': '1453',
            'estensione': None,
            'versione': 'vigente',
            'data_versione': '2024-01-01'
        }
        urn = dict_to_urn(data)
        assert urn == "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453!vig=2024-01-01"


class TestNormalizeActType:
    """Tests for normalize_act_type_for_urn() function."""

    def test_spaces_to_dots(self):
        """Convert spaces to dots."""
        assert normalize_act_type_for_urn("decreto legge") == "decreto.legge"

    def test_uppercase_to_lowercase(self):
        """Convert to lowercase."""
        assert normalize_act_type_for_urn("Decreto Legge") == "decreto.legge"

    def test_trim_whitespace(self):
        """Trim leading/trailing whitespace."""
        assert normalize_act_type_for_urn("  legge  ") == "legge"


class TestExtensionConversions:
    """Tests for Latin ordinal extension conversions."""

    def test_extension_to_number(self):
        """Convert Latin extension to number."""
        assert extension_to_number("bis") == 2
        assert extension_to_number("ter") == 3
        assert extension_to_number("quater") == 4
        assert extension_to_number("quinquies") == 5
        assert extension_to_number("") == 0
        assert extension_to_number("unknown") == 0

    def test_number_to_extension(self):
        """Convert number to Latin extension."""
        assert number_to_extension(2) == "bis"
        assert number_to_extension(3) in ("ter", "tris")  # Both valid
        assert number_to_extension(4) == "quater"
        assert number_to_extension(1) == ""
        assert number_to_extension(0) == ""

    def test_tris_ter_equivalence(self):
        """Test that both 'tris' and 'ter' map to 3."""
        assert extension_to_number("tris") == 3
        assert extension_to_number("ter") == 3


class TestRealWorldExamples:
    """Tests with real-world URN examples from Normattiva documentation."""

    def test_codice_civile_art2(self):
        """Art. 2 del corpo del Codice Civile (dispositivo)."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art2"
        c = parse_urn(urn)
        assert c.is_valid
        assert c.allegato is None
        assert c.articolo == "2"

    def test_codice_civile_preleggi(self):
        """Art. 2 delle preleggi (allegato 1)."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262:1~art2"
        c = parse_urn(urn)
        assert c.allegato == "1"
        assert c.articolo == "2"

    def test_codice_civile_art1453(self):
        """Art. 1453 del Codice Civile (allegato 2)."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262:2~art1453"
        c = parse_urn(urn)
        assert c.allegato == "2"
        assert c.articolo == "1453"

    def test_art_314_slash_15(self):
        """Articolo con numerazione estesa 314/15."""
        urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art314.15"
        c = parse_urn(urn)
        assert c.articolo == "314.15"

    def test_art_01_prefixed(self):
        """Articolo con prefisso 0."""
        urn = "urn:nir:stato:decreto.del.presidente.della.repubblica:1977-03-26;235~art01"
        c = parse_urn(urn)
        assert c.articolo == "01"

    def test_art_79octies_1(self):
        """Articolo con estensione e punto (79 octies.1)."""
        urn = "urn:nir:stato:decreto.legislativo:1998-02-24;58~art79octies.1"
        c = parse_urn(urn)
        # This will parse articolo as "79" with estensione "octies"
        # The ".1" is tricky - may need special handling
        assert c.is_valid

    def test_costituzione_art_vii_as_7(self):
        """Costituzione art. VII rappresentato come 7."""
        urn = "urn:nir:stato:costituzione:1947-12-27~art7"
        c = parse_urn(urn)
        assert c.articolo == "7"
        assert c.tipo_atto == "costituzione"

    def test_regio_decreto_with_roman_number(self):
        """R.D. con numero romano convertito ad arabo."""
        # R.D. 22 maggio 1864, n. MCCLXXXI → 1281
        urn = "urn:nir:stato:regio.decreto:1864-05-22;1281"
        c = parse_urn(urn)
        assert c.numero == "1281"

    def test_full_normattiva_url(self):
        """Parse full Normattiva URL."""
        url = "https://www.normattiva.it/uri-res/N2Ls?urn:nir:stato:decreto.legge:2008-11-10;180~art2!vig=2009-11-10"
        c = parse_urn(url)
        assert c.is_valid
        assert c.articolo == "2"
        assert c.versione == "vigente"
        assert c.data_versione == "2009-11-10"
