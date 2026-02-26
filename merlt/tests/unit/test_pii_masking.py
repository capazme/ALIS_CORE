"""
[P0] Unit tests for PIIMaskingService.

Tests Italian PII pattern detection and masking:
- Codice Fiscale (16-char alphanumeric)
- Email addresses
- Phone numbers (Italian formats)
- Dates of birth (dd/mm/yyyy variants)
"""

import pytest

from merlt.rlcf.pii_service import PIIMaskingService


@pytest.fixture
def svc():
    return PIIMaskingService()


# --- Codice Fiscale ---

class TestCodiceFiscale:
    def test_p0_pii_masks_codice_fiscale(self, svc):
        text = "Il codice fiscale e' RSSMRA85T10A562S"
        result = svc.mask_text(text)
        assert "[CF]" in result
        assert "RSSMRA85T10A562S" not in result

    def test_p0_pii_masks_codice_fiscale_lowercase(self, svc):
        text = "CF: rssmra85t10a562s"
        result = svc.mask_text(text)
        assert "[CF]" in result
        assert "rssmra85t10a562s" not in result

    def test_p0_pii_masks_multiple_cf(self, svc):
        text = "RSSMRA85T10A562S e BNCLRA90A01H501X"
        result = svc.mask_text(text)
        assert result.count("[CF]") == 2


# --- Email ---

class TestEmail:
    def test_p0_pii_masks_email(self, svc):
        text = "Contattare mario.rossi@email.it per info"
        result = svc.mask_text(text)
        assert "[EMAIL]" in result
        assert "mario.rossi@email.it" not in result

    def test_p0_pii_masks_email_complex(self, svc):
        text = "user+tag@subdomain.example.com"
        result = svc.mask_text(text)
        assert "[EMAIL]" in result

    def test_p0_pii_masks_multiple_emails(self, svc):
        text = "a@b.it e c@d.com"
        result = svc.mask_text(text)
        assert result.count("[EMAIL]") == 2


# --- Phone numbers ---

class TestPhone:
    def test_p0_pii_masks_phone_with_prefix(self, svc):
        text = "Telefono: +39 333 123 4567"
        result = svc.mask_text(text)
        assert "[TELEFONO]" in result
        assert "333" not in result

    def test_p0_pii_masks_phone_landline(self, svc):
        text = "Ufficio: 06 123 4567"
        result = svc.mask_text(text)
        assert "[TELEFONO]" in result

    def test_p0_pii_masks_phone_mobile(self, svc):
        text = "Cell: 345 678 9012"
        result = svc.mask_text(text)
        assert "[TELEFONO]" in result


# --- Dates ---

class TestDate:
    def test_p0_pii_masks_date_slash(self, svc):
        text = "nato il 15/03/1985"
        result = svc.mask_text(text)
        assert "[DATA]" in result
        assert "15/03/1985" not in result

    def test_p0_pii_masks_date_dash(self, svc):
        text = "data: 01-12-2000"
        result = svc.mask_text(text)
        assert "[DATA]" in result

    def test_p0_pii_masks_date_dot(self, svc):
        text = "data: 01.12.2000"
        result = svc.mask_text(text)
        assert "[DATA]" in result

    def test_p0_pii_masks_date_short_year(self, svc):
        text = "nato il 5/3/85"
        result = svc.mask_text(text)
        assert "[DATA]" in result


# --- Edge cases ---

class TestEdgeCases:
    def test_p0_pii_no_masking_clean_text(self, svc):
        text = "Articolo 1321 del codice civile"
        result = svc.mask_text(text)
        assert result == text

    def test_p0_pii_empty_string(self, svc):
        assert svc.mask_text("") == ""

    def test_p0_pii_none_input(self, svc):
        assert svc.mask_text(None) is None

    def test_p0_pii_multiple_types(self, svc):
        text = "RSSMRA85T10A562S email: a@b.it tel: +39 333 123 4567 nato 01/01/1990"
        result = svc.mask_text(text)
        assert "[CF]" in result
        assert "[EMAIL]" in result
        assert "[TELEFONO]" in result
        assert "[DATA]" in result

    def test_p0_pii_legal_text_untouched(self, svc):
        text = (
            "L'art. 2043 c.c. prevede che qualunque fatto doloso o colposo "
            "che cagiona ad altri un danno ingiusto obbliga colui che ha "
            "commesso il fatto a risarcire il danno."
        )
        result = svc.mask_text(text)
        assert result == text


# --- Consent level ---

class TestConsentLevel:
    def test_p0_should_store_text_basic(self, svc):
        assert svc.should_store_text("basic") is True

    def test_p0_should_store_text_full(self, svc):
        assert svc.should_store_text("full") is True

    def test_p0_should_not_store_text_anonymous(self, svc):
        assert svc.should_store_text("anonymous") is False
