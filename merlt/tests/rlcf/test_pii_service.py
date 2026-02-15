"""Tests for PII masking service."""
import pytest
from merlt.rlcf.pii_service import PIIMaskingService


@pytest.fixture
def svc():
    return PIIMaskingService()


class TestPIIMasking:
    """Test individual PII pattern detection."""

    def test_mask_codice_fiscale(self, svc):
        text = "Il CF dell'utente è RSSMRA85M01H501Z e va protetto"
        result = svc.mask_text(text)
        assert "[CF]" in result
        assert "RSSMRA85M01H501Z" not in result

    def test_mask_email(self, svc):
        text = "Contattare mario.rossi@email.it per info"
        result = svc.mask_text(text)
        assert "[EMAIL]" in result
        assert "mario.rossi@email.it" not in result

    def test_mask_date_slash(self, svc):
        text = "Nato il 15/03/1985 a Roma"
        result = svc.mask_text(text)
        assert "[DATA]" in result
        assert "15/03/1985" not in result

    def test_mask_date_dash(self, svc):
        text = "Scadenza 01-12-2025"
        result = svc.mask_text(text)
        assert "[DATA]" in result
        assert "01-12-2025" not in result

    def test_mask_phone_with_prefix(self, svc):
        text = "Chiamare +39 333 123 4567"
        result = svc.mask_text(text)
        assert "[TELEFONO]" in result
        assert "333" not in result

    def test_mask_phone_local(self, svc):
        text = "Tel. 06 1234567"
        result = svc.mask_text(text)
        assert "[TELEFONO]" in result

    def test_no_pii_unchanged(self, svc):
        text = "L'articolo 1453 del codice civile disciplina la risoluzione"
        result = svc.mask_text(text)
        assert result == text

    def test_none_input(self, svc):
        assert svc.mask_text(None) is None

    def test_empty_string(self, svc):
        assert svc.mask_text("") == ""

    def test_multiple_pii(self, svc):
        text = "CF: RSSMRA85M01H501Z, email: test@example.com, nato 01/01/1990"
        result = svc.mask_text(text)
        assert "[CF]" in result
        assert "[EMAIL]" in result
        assert "[DATA]" in result


class TestConsentLevel:
    """Test consent-level-aware behavior."""

    def test_anonymous_no_store(self, svc):
        assert svc.should_store_text("anonymous") is False

    def test_basic_stores(self, svc):
        assert svc.should_store_text("basic") is True

    def test_full_stores(self, svc):
        assert svc.should_store_text("full") is True
