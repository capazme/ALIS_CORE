"""
Tests for text operations and normalization.
"""
import pytest


class TestNormalizeActType:
    """Tests for normalize_act_type function."""

    def test_normalize_codice_civile(self):
        """Test normalization of 'codice civile'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("codice civile")

        assert result.lower() == "codice civile"

    def test_normalize_codice_penale(self):
        """Test normalization of 'codice penale'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("codice penale")

        assert result.lower() == "codice penale"

    def test_normalize_legge(self):
        """Test normalization of 'legge'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("legge")

        assert "legge" in result.lower()

    def test_normalize_decreto_legge(self):
        """Test normalization of 'decreto legge'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("decreto legge")

        assert "decreto" in result.lower() and "legge" in result.lower()

    def test_normalize_decreto_legislativo(self):
        """Test normalization of 'decreto legislativo'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("decreto legislativo")

        assert "decreto" in result.lower() and "legislativo" in result.lower()

    def test_normalize_regio_decreto(self):
        """Test normalization of 'regio decreto'."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("regio decreto")

        assert "regio" in result.lower() and "decreto" in result.lower()

    def test_normalize_case_insensitive(self):
        """Test that normalization is case insensitive."""
        from visualex.utils.text_op import normalize_act_type

        result1 = normalize_act_type("CODICE CIVILE")
        result2 = normalize_act_type("codice civile")
        result3 = normalize_act_type("Codice Civile")

        assert result1.lower() == result2.lower() == result3.lower()

    def test_normalize_with_search_flag(self):
        """Test normalization with search=True."""
        from visualex.utils.text_op import normalize_act_type

        result = normalize_act_type("codice civile", search=True)

        # With search=True, should return display format
        assert result is not None


class TestParseDate:
    """Tests for parse_date function."""

    def test_parse_date_iso_format(self):
        """Test parsing ISO format date."""
        from visualex.utils.text_op import parse_date

        result = parse_date("2020-01-15")

        assert result == "2020-01-15"

    def test_parse_date_italian_format(self):
        """Test parsing Italian format date."""
        from visualex.utils.text_op import parse_date

        result = parse_date("15 gennaio 2020")

        assert result == "2020-01-15"

    def test_parse_date_slash_format(self):
        """Test parsing slash format date."""
        from visualex.utils.text_op import parse_date

        result = parse_date("15/01/2020")

        assert result == "2020-01-15"

    def test_parse_date_various_months(self):
        """Test parsing dates with various Italian month names."""
        from visualex.utils.text_op import parse_date

        dates = [
            ("1 febbraio 2020", "2020-02-01"),
            ("15 marzo 2020", "2020-03-15"),
            ("10 aprile 2020", "2020-04-10"),
            ("20 maggio 2020", "2020-05-20"),
            ("5 giugno 2020", "2020-06-05"),
            ("30 luglio 2020", "2020-07-30"),
            ("15 agosto 2020", "2020-08-15"),
            ("1 settembre 2020", "2020-09-01"),
            ("10 ottobre 2020", "2020-10-10"),
            ("25 novembre 2020", "2020-11-25"),
            ("31 dicembre 2020", "2020-12-31"),
        ]

        for input_date, expected in dates:
            result = parse_date(input_date)
            assert result == expected, f"Failed for {input_date}"


class TestEstraiDataDaDenominazione:
    """Tests for estrai_data_da_denominazione function."""

    def test_extract_date_from_legge(self):
        """Test extracting date from law denomination."""
        from visualex.utils.text_op import estrai_data_da_denominazione

        text = "LEGGE 15 gennaio 2020, n. 123"
        result = estrai_data_da_denominazione(text)

        assert "2020" in result
        assert "01" in result or "gennaio" in result.lower()

    def test_extract_date_from_decreto(self):
        """Test extracting date from decreto denomination."""
        from visualex.utils.text_op import estrai_data_da_denominazione

        text = "DECRETO LEGISLATIVO 30 dicembre 2019, n. 162"
        result = estrai_data_da_denominazione(text)

        assert "2019" in result
        assert "12" in result or "dicembre" in result.lower()
