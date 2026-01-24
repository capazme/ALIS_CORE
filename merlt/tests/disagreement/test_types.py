"""
Test Disagreement Types
=======================

Test per la tassonomia del disagreement (types.py).
"""

import pytest
from datetime import datetime

from merlt.disagreement.types import (
    # Enums
    DisagreementType,
    DisagreementLevel,
    # Dataclasses
    DisagreementAnalysis,
    DisagreementExplanation,
    DisagreementSample,
    ExpertResponseData,
    ExpertPairConflict,
    TokenAttribution,
    AnnotationCandidate,
    Annotation,
    # Constants
    TYPE_LEVEL_FREQUENCY,
    EXPERT_NAMES,
    EXPERT_PAIRS,
)


class TestDisagreementType:
    """Test per DisagreementType enum."""

    def test_all_types_defined(self):
        """Verifica che tutti i 6 tipi siano definiti."""
        assert len(DisagreementType) == 6
        assert DisagreementType.ANTINOMY.value == "ANT"
        assert DisagreementType.INTERPRETIVE_GAP.value == "LAC"
        assert DisagreementType.METHODOLOGICAL.value == "MET"
        assert DisagreementType.OVERRULING.value == "OVR"
        assert DisagreementType.HIERARCHICAL.value == "GER"
        assert DisagreementType.SPECIALIZATION.value == "SPE"

    def test_label_property(self):
        """Verifica label leggibili."""
        assert DisagreementType.ANTINOMY.label == "Antinomia"
        assert DisagreementType.METHODOLOGICAL.label == "Divergenza Metodologica"
        assert DisagreementType.OVERRULING.label == "Overruling"

    def test_description_property(self):
        """Verifica descrizioni estese."""
        desc = DisagreementType.ANTINOMY.description
        assert "norme incompatibili" in desc
        assert "lex posterior" in desc or "criteri" in desc

        desc = DisagreementType.OVERRULING.description
        assert "precedente" in desc.lower() or "superato" in desc.lower()

    def test_resolution_criteria_property(self):
        """Verifica criteri di risoluzione."""
        criteria = DisagreementType.ANTINOMY.resolution_criteria
        assert isinstance(criteria, list)
        assert len(criteria) > 0
        assert "lex posterior" in criteria[0] or "lex specialis" in criteria[1]

        criteria = DisagreementType.HIERARCHICAL.resolution_criteria
        assert "lex superior" in criteria

    def test_string_value(self):
        """Verifica che sia str enum."""
        assert isinstance(DisagreementType.ANTINOMY, str)
        assert DisagreementType.ANTINOMY == "ANT"


class TestDisagreementLevel:
    """Test per DisagreementLevel enum."""

    def test_all_levels_defined(self):
        """Verifica che tutti i 4 livelli siano definiti."""
        assert len(DisagreementLevel) == 4
        assert DisagreementLevel.SEMANTIC.value == "SEM"
        assert DisagreementLevel.SYSTEMIC.value == "SIS"
        assert DisagreementLevel.TELEOLOGICAL.value == "TEL"
        assert DisagreementLevel.APPLICATIVE.value == "APP"

    def test_label_property(self):
        """Verifica label leggibili."""
        assert DisagreementLevel.SEMANTIC.label == "Semantico"
        assert DisagreementLevel.TELEOLOGICAL.label == "Teleologico"

    def test_expert_mapping_property(self):
        """Verifica mapping a expert MERL-T."""
        assert DisagreementLevel.SEMANTIC.expert_mapping == "LiteralExpert"
        assert DisagreementLevel.SYSTEMIC.expert_mapping == "SystemicExpert"
        assert DisagreementLevel.TELEOLOGICAL.expert_mapping == "PrinciplesExpert"
        assert DisagreementLevel.APPLICATIVE.expert_mapping == "PrecedentExpert"

    def test_preleggi_reference_property(self):
        """Verifica riferimenti normativi Preleggi."""
        ref = DisagreementLevel.SEMANTIC.preleggi_reference
        assert "Art. 12" in ref
        assert "significato proprio" in ref

        ref = DisagreementLevel.APPLICATIVE.preleggi_reference
        assert "casi simili" in ref


class TestExpertResponseData:
    """Test per ExpertResponseData dataclass."""

    def test_creation(self):
        """Test creazione base."""
        data = ExpertResponseData(
            expert_type="literal",
            interpretation="Il testo afferma che...",
            confidence=0.85,
            sources_cited=["urn:norma:cc:art1453"],
            reasoning_pattern="literal",
        )
        assert data.expert_type == "literal"
        assert data.confidence == 0.85
        assert len(data.sources_cited) == 1

    def test_to_dict(self):
        """Test serializzazione."""
        data = ExpertResponseData(
            expert_type="systemic",
            interpretation="Nel sistema normativo...",
            confidence=0.7,
        )
        d = data.to_dict()
        assert d["expert_type"] == "systemic"
        assert d["confidence"] == 0.7
        assert d["sources_cited"] == []

    def test_from_dict(self):
        """Test deserializzazione."""
        d = {
            "expert_type": "principles",
            "interpretation": "La ratio legis...",
            "confidence": 0.9,
            "sources_cited": ["cost:art3"],
            "reasoning_pattern": "teleological",
        }
        data = ExpertResponseData.from_dict(d)
        assert data.expert_type == "principles"
        assert data.reasoning_pattern == "teleological"


class TestDisagreementSample:
    """Test per DisagreementSample dataclass."""

    def test_creation_minimal(self):
        """Test creazione con campi minimi."""
        sample = DisagreementSample(
            sample_id="test_001",
            query="Cos'e' la buona fede?",
            expert_responses={},
        )
        assert sample.sample_id == "test_001"
        assert sample.has_disagreement is None
        assert sample.source == "unknown"

    def test_creation_full(self):
        """Test creazione con tutti i campi."""
        sample = DisagreementSample(
            sample_id="test_002",
            query="Il venditore puo' recedere?",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="No, art. 1372 c.c.",
                    confidence=0.9,
                ),
                "principles": ExpertResponseData(
                    expert_type="principles",
                    interpretation="Si, buona fede art. 1375",
                    confidence=0.8,
                ),
            },
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.72,
            resolvability=0.45,
            conflicting_pairs=[("literal", "principles")],
            source="rlcf",
            legal_domain="civile",
        )
        assert sample.has_disagreement is True
        assert sample.disagreement_type == DisagreementType.METHODOLOGICAL
        assert len(sample.expert_responses) == 2

    def test_is_labeled_property(self):
        """Test property is_labeled."""
        sample_unlabeled = DisagreementSample(
            sample_id="u1",
            query="test",
            expert_responses={},
        )
        assert sample_unlabeled.is_labeled is False

        sample_labeled = DisagreementSample(
            sample_id="l1",
            query="test",
            expert_responses={},
            has_disagreement=True,
        )
        assert sample_labeled.is_labeled is True

    def test_is_fully_labeled_property(self):
        """Test property is_fully_labeled."""
        # Negative case (no disagreement) - fully labeled
        sample_neg = DisagreementSample(
            sample_id="n1",
            query="test",
            expert_responses={},
            has_disagreement=False,
        )
        assert sample_neg.is_fully_labeled is True

        # Positive but incomplete
        sample_partial = DisagreementSample(
            sample_id="p1",
            query="test",
            expert_responses={},
            has_disagreement=True,
            disagreement_type=DisagreementType.ANTINOMY,
            # Missing level and intensity
        )
        assert sample_partial.is_fully_labeled is False

        # Positive and complete
        sample_full = DisagreementSample(
            sample_id="f1",
            query="test",
            expert_responses={},
            has_disagreement=True,
            disagreement_type=DisagreementType.ANTINOMY,
            disagreement_level=DisagreementLevel.SYSTEMIC,
            intensity=0.8,
        )
        assert sample_full.is_fully_labeled is True

    def test_to_dict_from_dict_roundtrip(self):
        """Test roundtrip serialization."""
        original = DisagreementSample(
            sample_id="rt_001",
            query="Test roundtrip",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="Test",
                    confidence=0.5,
                ),
            },
            has_disagreement=True,
            disagreement_type=DisagreementType.INTERPRETIVE_GAP,
            disagreement_level=DisagreementLevel.SEMANTIC,
            intensity=0.6,
            source="rlcf",
        )

        d = original.to_dict()
        restored = DisagreementSample.from_dict(d)

        assert restored.sample_id == original.sample_id
        assert restored.query == original.query
        assert restored.has_disagreement == original.has_disagreement
        assert restored.disagreement_type == original.disagreement_type
        assert restored.disagreement_level == original.disagreement_level
        assert restored.intensity == original.intensity
        assert len(restored.expert_responses) == len(original.expert_responses)


class TestDisagreementAnalysis:
    """Test per DisagreementAnalysis dataclass."""

    def test_creation_no_disagreement(self):
        """Test creazione senza disagreement."""
        analysis = DisagreementAnalysis(
            has_disagreement=False,
            confidence=0.95,
        )
        assert analysis.has_disagreement is False
        assert analysis.disagreement_type is None
        assert analysis.intensity == 0.0

    def test_creation_with_disagreement(self):
        """Test creazione con disagreement."""
        analysis = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.72,
            resolvability=0.45,
            confidence=0.88,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="literal",
                    expert_b="principles",
                    conflict_score=0.8,
                    contention_point="interpretazione recesso",
                ),
            ],
        )
        assert analysis.has_disagreement is True
        assert analysis.disagreement_type == DisagreementType.METHODOLOGICAL
        assert len(analysis.conflicting_pairs) == 1

    def test_synthesis_mode_property_convergent(self):
        """Test synthesis_mode per casi convergent."""
        # No disagreement -> convergent
        analysis = DisagreementAnalysis(has_disagreement=False)
        assert analysis.synthesis_mode == "convergent"

        # Low intensity, high resolvability -> convergent
        analysis = DisagreementAnalysis(
            has_disagreement=True,
            intensity=0.3,
            resolvability=0.8,
        )
        assert analysis.synthesis_mode == "convergent"

    def test_synthesis_mode_property_divergent(self):
        """Test synthesis_mode per casi divergent."""
        analysis = DisagreementAnalysis(
            has_disagreement=True,
            intensity=0.8,
            resolvability=0.3,
        )
        assert analysis.synthesis_mode == "divergent"

    def test_to_dict_from_dict_roundtrip(self):
        """Test roundtrip serialization."""
        original = DisagreementAnalysis(
            has_disagreement=True,
            disagreement_type=DisagreementType.OVERRULING,
            disagreement_level=DisagreementLevel.APPLICATIVE,
            intensity=0.9,
            resolvability=0.8,
            confidence=0.85,
            conflicting_pairs=[
                ExpertPairConflict(
                    expert_a="precedent",
                    expert_b="literal",
                    conflict_score=0.9,
                ),
            ],
        )

        d = original.to_dict()
        restored = DisagreementAnalysis.from_dict(d)

        assert restored.has_disagreement == original.has_disagreement
        assert restored.disagreement_type == original.disagreement_type
        assert restored.intensity == original.intensity
        assert len(restored.conflicting_pairs) == len(original.conflicting_pairs)


class TestDisagreementExplanation:
    """Test per DisagreementExplanation dataclass."""

    def test_creation(self):
        """Test creazione."""
        expl = DisagreementExplanation(
            natural_explanation=(
                "Il disagreement e' di tipo METODOLOGICO: LiteralExpert "
                "applica interpretazione letterale, PrinciplesExpert teleologica."
            ),
            key_tokens=["art. 1372", "buona fede", "recesso"],
            resolution_suggestions=[
                "Applicare interpretazione sistematica",
                "Considerare ratio legis",
            ],
        )
        assert "METODOLOGICO" in expl.natural_explanation
        assert len(expl.key_tokens) == 3
        assert len(expl.resolution_suggestions) == 2

    def test_to_dict(self):
        """Test serializzazione."""
        expl = DisagreementExplanation(
            natural_explanation="Test explanation",
            key_tokens=["token1", "token2"],
            expert_pair_scores={
                ("literal", "principles"): 0.8,
                ("systemic", "precedent"): 0.3,
            },
        )
        d = expl.to_dict()

        assert d["natural_explanation"] == "Test explanation"
        assert len(d["key_tokens"]) == 2
        # Pair keys converted to strings
        assert "literal__vs__principles" in d["expert_pair_scores"]
        assert d["expert_pair_scores"]["literal__vs__principles"] == 0.8


class TestConstants:
    """Test per costanti del modulo."""

    def test_expert_names(self):
        """Test EXPERT_NAMES."""
        assert len(EXPERT_NAMES) == 4
        assert "literal" in EXPERT_NAMES
        assert "systemic" in EXPERT_NAMES
        assert "principles" in EXPERT_NAMES
        assert "precedent" in EXPERT_NAMES

    def test_expert_pairs(self):
        """Test EXPERT_PAIRS."""
        # 4 expert -> 6 coppie (4 choose 2)
        assert len(EXPERT_PAIRS) == 6

        # Verifica alcune coppie specifiche
        assert ("literal", "systemic") in EXPERT_PAIRS
        assert ("principles", "precedent") in EXPERT_PAIRS

    def test_type_level_frequency(self):
        """Test matrice TYPE_LEVEL_FREQUENCY."""
        # Verifica struttura
        assert len(TYPE_LEVEL_FREQUENCY) == 6  # 6 tipi

        for dtype in DisagreementType:
            assert dtype in TYPE_LEVEL_FREQUENCY
            levels = TYPE_LEVEL_FREQUENCY[dtype]
            assert len(levels) == 4  # 4 livelli

            for dlevel in DisagreementLevel:
                assert dlevel in levels
                freq = levels[dlevel]
                assert freq in [0, 1, 2]  # raro, comune, molto comune

        # Verifica alcuni valori noti dalla spec
        assert TYPE_LEVEL_FREQUENCY[DisagreementType.OVERRULING][DisagreementLevel.APPLICATIVE] == 2
        assert TYPE_LEVEL_FREQUENCY[DisagreementType.INTERPRETIVE_GAP][DisagreementLevel.SEMANTIC] == 2


class TestAnnotation:
    """Test per Annotation dataclass."""

    def test_creation(self):
        """Test creazione annotazione."""
        ann = Annotation(
            sample_id="s001",
            annotator_id="annotator_1",
            has_disagreement=True,
            disagreement_type=DisagreementType.ANTINOMY,
            disagreement_level=DisagreementLevel.SYSTEMIC,
            intensity=0.8,
            resolvability=0.4,
            explanation="Le due norme sono in conflitto diretto.",
            time_spent_seconds=120,
        )
        assert ann.sample_id == "s001"
        assert ann.has_disagreement is True
        assert ann.time_spent_seconds == 120

    def test_to_dict(self):
        """Test serializzazione."""
        ann = Annotation(
            sample_id="s002",
            annotator_id="a1",
            has_disagreement=False,
        )
        d = ann.to_dict()

        assert d["sample_id"] == "s002"
        assert d["has_disagreement"] is False
        assert d["disagreement_type"] is None
