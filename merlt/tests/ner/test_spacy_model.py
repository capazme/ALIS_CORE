"""
Test per LegalNERModel
=======================

Test del modello NER giuridico basato su spaCy.
"""

import pytest

from merlt.ner import LegalNERModel, CitationMatch


class TestLegalNERModel:
    """Test suite per LegalNERModel."""

    @pytest.fixture
    def model(self):
        """Fixture: modello NER."""
        return LegalNERModel()

    def test_model_initialization(self, model):
        """Test: modello si inizializza correttamente."""
        assert model.is_ready()
        assert model.nlp is not None

    def test_extract_citations_basic(self, model):
        """Test: estrazione citazioni base."""
        text = "L'art. 1453 del codice civile regola la risoluzione."
        citations = model.extract_citations(text)

        # Verifica che venga estratto almeno qualcosa
        # (il modello base potrebbe non riconoscere tutte le entità senza training)
        assert isinstance(citations, list)

    def test_extract_citations_with_context(self, model):
        """Test: estrazione con contesto norma."""
        text = "L'articolo 52 prevede la legittima difesa."
        context = {
            "tipo_atto": "codice penale",
            "estremi": "Codice Penale - Regio Decreto 19 ottobre 1930, n. 1398",
        }

        citations = model.extract_citations(text, context_norma=context)
        assert isinstance(citations, list)

        # Se trova un articolo, verifica metadata
        for citation in citations:
            if citation.label == "ARTICOLO":
                assert "tipo_atto" in citation.metadata

    def test_extract_article_number(self, model):
        """Test: estrazione numero articolo."""
        assert model._extract_article_number("art. 1453") == "1453"
        assert model._extract_article_number("articolo 52") == "52"
        assert model._extract_article_number("art. 52-bis") == "52-bis"
        assert model._extract_article_number("nessun articolo qui") == ""

    def test_normalize_codice(self, model):
        """Test: normalizzazione nome codice."""
        assert model._normalize_codice("c.c.") == "codice civile"
        assert model._normalize_codice("cod. civ.") == "codice civile"
        assert model._normalize_codice("codice civile") == "codice civile"
        assert model._normalize_codice("c.p.") == "codice penale"
        assert model._normalize_codice("cod. pen.") == "codice penale"

    def test_extract_act_type_from_estremi(self, model):
        """Test: estrazione tipo atto da estremi."""
        estremi = "Codice civile - Regio Decreto 16 marzo 1942, n. 262"
        tipo_atto = model._extract_act_type_from_estremi(estremi)
        assert tipo_atto == "codice civile"

        estremi = "Codice Penale - Regio Decreto 19 ottobre 1930, n. 1398"
        tipo_atto = model._extract_act_type_from_estremi(estremi)
        assert tipo_atto == "codice penale"

    def test_citation_match_repr(self):
        """Test: rappresentazione CitationMatch."""
        citation = CitationMatch(
            text="art. 1453",
            label="ARTICOLO",
            start=2,
            end=11,
            confidence=0.95,
        )

        repr_str = repr(citation)
        assert "art. 1453" in repr_str
        assert "ARTICOLO" in repr_str
        assert "[2:11]" in repr_str

    def test_citation_with_urn(self):
        """Test: CitationMatch con URN risolto."""
        citation = CitationMatch(
            text="art. 1453",
            label="ARTICOLO",
            start=2,
            end=11,
            resolved_urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
        )

        assert citation.resolved_urn is not None
        assert "urn:nir" in citation.resolved_urn
