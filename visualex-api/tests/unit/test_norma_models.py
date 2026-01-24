"""
Tests for Norma and NormaVisitata models.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestNormaModel:
    """Tests for Norma dataclass."""

    def test_norma_creation_basic(self):
        """Test basic Norma creation."""
        from visualex.models.norma import Norma

        norma = Norma(
            tipo_atto="legge",
            data="2020-01-15",
            numero_atto="123",
        )

        assert norma.tipo_atto == "legge"
        assert norma.data == "2020-01-15"
        assert norma.numero_atto == "123"

    def test_norma_creation_codice_civile(self):
        """Test Norma creation for Codice Civile."""
        from visualex.models.norma import Norma

        norma = Norma(tipo_atto="codice civile")

        assert norma.tipo_atto == "codice civile"
        assert norma.data is None
        assert norma.numero_atto is None

    def test_norma_invalid_tipo_atto(self):
        """Test that empty tipo_atto raises ValueError."""
        from visualex.models.norma import Norma

        with pytest.raises(ValueError, match="tipo_atto must be a non-empty string"):
            Norma(tipo_atto="")

    def test_norma_invalid_date_format(self):
        """Test that invalid date format raises ValueError."""
        from visualex.models.norma import Norma

        with pytest.raises(ValueError, match="Invalid date format"):
            Norma(tipo_atto="legge", data="15-01-2020")  # Wrong format

    def test_norma_valid_year_only_date(self):
        """Test that year-only date is valid."""
        from visualex.models.norma import Norma

        norma = Norma(tipo_atto="legge", data="2020", numero_atto="123")

        assert norma.data == "2020"

    def test_norma_str_representation(self):
        """Test Norma string representation."""
        from visualex.models.norma import Norma

        norma = Norma(
            tipo_atto="legge",
            data="2020-01-15",
            numero_atto="123",
        )

        str_repr = str(norma)

        assert "Legge" in str_repr or "legge" in str_repr.lower()
        assert "123" in str_repr

    def test_norma_to_dict(self):
        """Test Norma to_dict conversion."""
        from visualex.models.norma import Norma

        norma = Norma(
            tipo_atto="legge",
            data="2020-01-15",
            numero_atto="123",
        )

        result = norma.to_dict()

        assert "tipo_atto" in result
        assert "data" in result
        assert "numero_atto" in result
        assert result["numero_atto"] == "123"

    def test_norma_url_property(self):
        """Test Norma URL property generation."""
        from visualex.models.norma import Norma

        norma = Norma(tipo_atto="codice civile")

        url = norma.url

        assert url is not None
        assert "normattiva.it" in url


class TestNormaVisitataModel:
    """Tests for NormaVisitata dataclass."""

    def test_norma_visitata_creation(self):
        """Test basic NormaVisitata creation."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(
            norma=norma,
            numero_articolo="1453",
        )

        assert nv.norma == norma
        assert nv.numero_articolo == "1453"

    def test_norma_visitata_with_version(self):
        """Test NormaVisitata with version info."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(
            norma=norma,
            numero_articolo="1453",
            versione="vigente",
            data_versione="2023-06-01",
        )

        assert nv.versione == "vigente"
        assert nv.data_versione == "2023-06-01"

    def test_norma_visitata_urn_property(self):
        """Test NormaVisitata URN property generation."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(
            norma=norma,
            numero_articolo="1453",
        )

        urn = nv.urn

        assert urn is not None
        assert "art1453" in urn

    def test_norma_visitata_equality(self):
        """Test NormaVisitata equality comparison."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv1 = NormaVisitata(norma=norma, numero_articolo="1453")
        nv2 = NormaVisitata(norma=norma, numero_articolo="1453")

        assert nv1 == nv2

    def test_norma_visitata_inequality(self):
        """Test NormaVisitata inequality comparison."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv1 = NormaVisitata(norma=norma, numero_articolo="1453")
        nv2 = NormaVisitata(norma=norma, numero_articolo="1454")

        assert nv1 != nv2

    def test_norma_visitata_hash(self):
        """Test NormaVisitata is hashable for use in sets."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv1 = NormaVisitata(norma=norma, numero_articolo="1453")
        nv2 = NormaVisitata(norma=norma, numero_articolo="1453")

        # Should be able to use in sets
        s = {nv1, nv2}
        assert len(s) == 1  # Same hash, same article

    def test_norma_visitata_str_representation(self):
        """Test NormaVisitata string representation."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(norma=norma, numero_articolo="1453")

        str_repr = str(nv)

        assert "1453" in str_repr

    def test_norma_visitata_to_dict(self):
        """Test NormaVisitata to_dict conversion."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice civile")
        nv = NormaVisitata(
            norma=norma,
            numero_articolo="1453",
            allegato="1",
        )

        result = nv.to_dict()

        assert "numero_articolo" in result
        assert result["numero_articolo"] == "1453"
        assert "allegato" in result
        assert result["allegato"] == "1"
        assert "urn" in result

    def test_norma_visitata_from_dict(self):
        """Test NormaVisitata from_dict factory."""
        from visualex.models.norma import NormaVisitata

        data = {
            "tipo_atto": "codice civile",
            "numero_articolo": "1453",
        }

        nv = NormaVisitata.from_dict(data)

        assert nv.numero_articolo == "1453"
        assert nv.norma.tipo_atto == "codice civile"

    def test_norma_visitata_with_allegato(self):
        """Test NormaVisitata with allegato."""
        from visualex.models.norma import Norma, NormaVisitata

        norma = Norma(tipo_atto="codice penale")
        nv = NormaVisitata(
            norma=norma,
            numero_articolo="575",
            allegato="1",
        )

        assert nv.allegato == "1"
