"""
Test per Data Converter
========================

Test conversione feedback → formato spaCy.
"""

import pytest

from merlt.ner.data_converter import (
    feedback_to_spacy_format,
    validate_annotations,
    _remove_overlapping_entities,
)


class TestDataConverter:
    """Test suite per data converter."""

    def test_feedback_to_spacy_format_basic(self):
        """Test: conversione base feedback."""
        feedback = {
            "text": "L'art. 1453 del codice civile regola la risoluzione.",
            "citations": [
                {"start": 2, "end": 11, "label": "ARTICOLO"},
                {"start": 16, "end": 29, "label": "CODICE"},
            ],
        }

        text, annotations = feedback_to_spacy_format(feedback)

        assert text == feedback["text"]
        assert "entities" in annotations
        assert len(annotations["entities"]) == 2
        assert annotations["entities"][0] == (2, 11, "ARTICOLO")
        assert annotations["entities"][1] == (16, 29, "CODICE")

    def test_feedback_with_corrections(self):
        """Test: conversione con correzioni utente."""
        feedback = {
            "text": "L'art. 1453 del codice civile",
            "citations": [
                {"id": "c1", "start": 2, "end": 11, "label": "ARTICOLO"},
                {"id": "c2", "start": 16, "end": 29, "label": "WRONG_LABEL"},
            ],
            "user_corrections": {
                "modify": {
                    "c2": {"start": 16, "end": 29, "label": "CODICE"}
                }
            },
        }

        text, annotations = feedback_to_spacy_format(feedback)

        # Verifica che la label sia stata corretta
        assert annotations["entities"][1][2] == "CODICE"

    def test_feedback_with_invalid_citation(self):
        """Test: skip citazioni invalide."""
        feedback = {
            "text": "Testo di esempio",
            "citations": [
                {"start": 0, "end": 5, "label": "ARTICOLO"},  # OK
                {"start": 10, "end": 5, "label": "CODICE"},  # Invalid: end < start
                {"start": -1, "end": 3, "label": "LEGGE"},  # Invalid: start < 0
            ],
        }

        text, annotations = feedback_to_spacy_format(feedback)

        # Solo la prima citazione valida
        assert len(annotations["entities"]) == 1
        assert annotations["entities"][0] == (0, 5, "ARTICOLO")

    def test_remove_overlapping_entities(self):
        """Test: rimozione overlap."""
        entities = [
            (0, 10, "ARTICOLO"),  # Più lunga
            (5, 8, "COMMA"),  # Overlap, più corta → rimossa
            (15, 20, "CODICE"),  # No overlap
        ]

        cleaned = _remove_overlapping_entities(entities)

        assert len(cleaned) == 2
        assert (0, 10, "ARTICOLO") in cleaned
        assert (15, 20, "CODICE") in cleaned
        assert (5, 8, "COMMA") not in cleaned

    def test_validate_annotations_valid(self):
        """Test: validazione annotations valide."""
        text = "L'art. 1453 del codice civile"
        entities = [(2, 11, "ARTICOLO"), (16, 29, "CODICE")]

        assert validate_annotations(text, entities) is True

    def test_validate_annotations_invalid_bounds(self):
        """Test: validazione bounds invalidi."""
        text = "Testo breve"
        entities = [(0, 100, "ARTICOLO")]  # end > text length

        assert validate_annotations(text, entities) is False

    def test_validate_annotations_overlapping(self):
        """Test: validazione con overlap."""
        text = "L'art. 1453 del codice civile"
        entities = [
            (2, 11, "ARTICOLO"),
            (5, 15, "CODICE"),  # Overlap
        ]

        assert validate_annotations(text, entities) is False

    def test_validate_annotations_empty_span(self):
        """Test: validazione span vuoto."""
        text = "Testo    con spazi"
        entities = [(5, 9, "ARTICOLO")]  # Solo spazi

        # Nota: validate_annotations controlla che strip() non sia vuoto
        # In questo caso "    " → "" quindi invalid
        assert validate_annotations(text, entities) is False
