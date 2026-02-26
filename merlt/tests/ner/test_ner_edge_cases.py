"""
Edge case tests for NER (Legal Named Entity Recognition).

Tests per situazioni limite del modello NER:
- Citazioni sovrapposte e ambigue
- Forme abbreviate non standard
- Riferimenti bis/ter/quater Unicode
- Input vuoto, whitespace, testo senza citazioni
- Testi molto lunghi
- Piu' codici nella stessa frase
"""

import pytest

from merlt.ner import LegalNERModel, CitationMatch


@pytest.fixture(scope="module")
def model():
    """Fixture: modello NER (singola istanza per modulo, caricamento lento)."""
    try:
        return LegalNERModel()
    except (RuntimeError, ImportError):
        pytest.skip("spaCy model not available")


# ============================================================================
# Citation Pattern Edge Cases
# ============================================================================

class TestCitationPatterns:
    """Tests per pattern di citazione problematici."""

    def test_overlapping_article_range(self, model):
        """Citazioni con range: 'artt. 1453-1455'."""
        text = "Gli artt. 1453-1455 del codice civile disciplinano la risoluzione."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)

    def test_abbreviated_forms(self, model):
        """Forme abbreviate non standard."""
        text = "Ai sensi dell'art. 52 c.p., la legittima difesa esclude la punibilita'."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)

    def test_comma_reference(self, model):
        """Riferimento a comma specifico."""
        text = "Il comma 3 dell'articolo 1453 prevede un'eccezione."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)

    def test_articolo_unico(self, model):
        """Riferimento a 'articolo unico'."""
        text = "L'articolo unico della legge 123/2020 stabilisce che."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)


# ============================================================================
# Bis/Ter/Quater Unicode
# ============================================================================

class TestBisTerQuater:
    """Tests per articoli bis, ter, quater."""

    def test_extract_article_bis(self, model):
        """Estrazione articolo con -bis."""
        result = model._extract_article_number("art. 52-bis")
        assert result == "52-bis"

    def test_extract_article_ter(self, model):
        """Estrazione articolo con -ter."""
        result = model._extract_article_number("articolo 3-ter")
        assert result == "3-ter"

    def test_extract_article_quater(self, model):
        """Estrazione articolo con -quater."""
        result = model._extract_article_number("art. 3-quater")
        assert result == "3-quater"

    def test_bis_in_full_text(self, model):
        """Citazione -bis in testo completo."""
        text = "Ai sensi dell'art. 1-bis del D.Lgs. 50/2016."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)


# ============================================================================
# Invalid Input
# ============================================================================

class TestInvalidInput:
    """Tests per input invalido o vuoto."""

    def test_empty_input(self, model):
        """Input vuoto."""
        citations = model.extract_citations("")
        assert isinstance(citations, list)
        assert len(citations) == 0

    def test_whitespace_only_input(self, model):
        """Input solo whitespace."""
        citations = model.extract_citations("   \n\t  \n  ")
        assert isinstance(citations, list)
        assert len(citations) == 0

    def test_text_with_no_citations(self, model):
        """Testo senza alcuna citazione giuridica."""
        text = "Il tempo oggi e' bello e la temperatura e' mite."
        citations = model.extract_citations(text)
        assert isinstance(citations, list)
        # May or may not find entities depending on model, but should not crash

    def test_invalid_urn_resolution(self, model):
        """Risoluzione con contesto invalido non causa errore."""
        text = "L'art. 100 prevede una disciplina particolare."
        context = {
            "tipo_atto": "",
            "estremi": "Documento sconosciuto senza tipo",
        }
        citations = model.extract_citations(text, context_norma=context)
        assert isinstance(citations, list)


# ============================================================================
# Multiple Codici
# ============================================================================

class TestMultipleCodici:
    """Tests per piu' codici nella stessa frase."""

    def test_multiple_codici_same_sentence(self, model):
        """Piu' codici nella stessa frase."""
        text = (
            "L'art. 1453 del codice civile si applica unitamente "
            "all'art. 640 del codice penale."
        )
        citations = model.extract_citations(text)
        assert isinstance(citations, list)

    def test_normalize_codice_civile_variants(self, model):
        """Normalizzazione varianti codice civile."""
        assert model._normalize_codice("c.c.") == "codice civile"
        assert model._normalize_codice("cod. civ.") == "codice civile"
        assert model._normalize_codice("codice civile") == "codice civile"

    def test_normalize_codice_penale_variants(self, model):
        """Normalizzazione varianti codice penale."""
        assert model._normalize_codice("c.p.") == "codice penale"
        assert model._normalize_codice("cod. pen.") == "codice penale"


# ============================================================================
# Long Input
# ============================================================================

class TestLongInput:
    """Tests per input molto lunghi."""

    def test_long_text_10k_chars(self, model):
        """Testo di 10K caratteri non causa timeout o errore."""
        # Build a long text with some legal content
        base = "L'art. 1 del codice civile prevede che le norme si applicano secondo criteri interpretativi. "
        long_text = base * 120  # ~10K chars
        assert len(long_text) > 10000

        citations = model.extract_citations(long_text)
        assert isinstance(citations, list)


# ============================================================================
# CitationMatch Dataclass
# ============================================================================

class TestCitationMatchEdgeCases:
    """Tests per CitationMatch in condizioni limite."""

    def test_citation_match_default_confidence(self):
        """CitationMatch ha confidence default 1.0."""
        cm = CitationMatch(text="art. 1", label="ARTICOLO", start=0, end=6)
        assert cm.confidence == 1.0
        assert cm.resolved_urn is None
        assert cm.metadata == {}

    def test_citation_match_repr_without_urn(self):
        """Repr senza URN."""
        cm = CitationMatch(text="art. 1", label="ARTICOLO", start=0, end=6)
        r = repr(cm)
        assert "art. 1" in r
        assert "ARTICOLO" in r
        assert "urn=" not in r

    def test_citation_match_repr_with_urn(self):
        """Repr con URN."""
        cm = CitationMatch(
            text="art. 1",
            label="ARTICOLO",
            start=0,
            end=6,
            resolved_urn="urn:nir:stato:cc~art1",
        )
        r = repr(cm)
        assert "urn=" in r

    def test_extract_article_number_no_match(self, model):
        """_extract_article_number con testo senza articolo."""
        assert model._extract_article_number("nessun numero qui") == ""
        assert model._extract_article_number("") == ""

    def test_extract_act_type_costituzione(self, model):
        """Estrae tipo atto Costituzione da estremi."""
        estremi = "Costituzione della Repubblica Italiana"
        assert model._extract_act_type_from_estremi(estremi) == "costituzione"

    def test_extract_act_type_unknown(self, model):
        """Tipo atto sconosciuto restituisce stringa vuota."""
        estremi = "Regolamento UE 123/2020"
        assert model._extract_act_type_from_estremi(estremi) == ""
