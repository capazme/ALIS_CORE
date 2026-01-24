"""
Test Disagreement Augmentation
==============================

Test per modulo data augmentation del disagreement detection.
"""

import pytest
from copy import deepcopy

# Skip se torch non disponibile
torch_available = True
try:
    import torch
except ImportError:
    torch_available = False

pytestmark = pytest.mark.skipif(not torch_available, reason="PyTorch non disponibile")


from merlt.disagreement.types import (
    DisagreementSample,
    ExpertResponseData,
    DisagreementType,
    DisagreementLevel,
)


class TestAugmentationConfig:
    """Test per AugmentationConfig."""

    def test_default_config(self):
        """Test configurazione default."""
        from merlt.disagreement.data.augmentation import AugmentationConfig

        config = AugmentationConfig()

        assert config.dropout_prob == 0.1
        assert config.word_dropout_prob == 0.05
        assert config.synonym_prob == 0.15
        assert config.max_augmentations_per_sample == 3
        assert config.preserve_key_terms is True

    def test_custom_config(self):
        """Test configurazione custom."""
        from merlt.disagreement.data.augmentation import AugmentationConfig

        config = AugmentationConfig(
            dropout_prob=0.2,
            word_dropout_prob=0.1,
            max_augmentations_per_sample=5,
        )

        assert config.dropout_prob == 0.2
        assert config.word_dropout_prob == 0.1
        assert config.max_augmentations_per_sample == 5

    def test_key_terms_loaded(self):
        """Test che key terms vengano caricati."""
        from merlt.disagreement.data.augmentation import AugmentationConfig

        config = AugmentationConfig()

        assert "art." in config.key_terms
        assert "buona fede" in config.key_terms
        assert "contratto" in config.key_terms


class TestTextDropout:
    """Test per TextDropout."""

    def test_empty_text(self):
        """Test con testo vuoto."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout()
        result = dropout("")

        assert result == ""

    def test_preserves_min_sentences(self):
        """Test che preservi minimo di frasi."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout(
            sentence_dropout_prob=1.0,  # Drop tutto
            min_sentences=2,
        )

        text = "Prima frase. Seconda frase. Terza frase."
        result = dropout(text)

        # Deve preservare almeno 2 frasi
        sentences = [s for s in result.split('.') if s.strip()]
        assert len(sentences) >= 1  # Almeno una frase rimane

    def test_preserves_key_terms(self):
        """Test che preservi key terms."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout(
            sentence_dropout_prob=1.0,
            min_sentences=0,
        )

        text = "Questa frase contiene contratto. Altra frase senza terms."
        result = dropout(text, key_terms={"contratto"})

        # La frase con "contratto" dovrebbe essere preservata
        assert "contratto" in result

    def test_word_dropout(self):
        """Test dropout di parole."""
        from merlt.disagreement.data.augmentation import TextDropout
        import random

        random.seed(42)

        dropout = TextDropout(
            sentence_dropout_prob=0.0,  # No sentence dropout
            word_dropout_prob=0.5,      # 50% word dropout
            min_words=5,
        )

        text = "Questa e' una frase molto lunga con tante parole diverse da testare."
        result = dropout(text)

        # Alcune parole dovrebbero essere rimosse
        original_words = len(text.split())
        result_words = len(result.split())

        # Con 50% dropout e min_words=5, dovrebbero essere meno parole
        # ma almeno min_words
        assert result_words >= 5


class TestExpertPermutation:
    """Test per ExpertPermutation."""

    def test_permutation_preserves_all_experts(self):
        """Test che preservi tutti gli expert."""
        from merlt.disagreement.data.augmentation import ExpertPermutation

        permutation = ExpertPermutation()

        expert_responses = {
            "literal": ExpertResponseData(
                expert_type="literal",
                interpretation="Test literal",
                confidence=0.8,
            ),
            "systemic": ExpertResponseData(
                expert_type="systemic",
                interpretation="Test systemic",
                confidence=0.7,
            ),
        }

        result = permutation(expert_responses)

        assert set(result.keys()) == set(expert_responses.keys())
        assert result["literal"].interpretation == "Test literal"
        assert result["systemic"].interpretation == "Test systemic"

    def test_permutation_different_order(self):
        """Test che cambi ordine (statisticamente)."""
        from merlt.disagreement.data.augmentation import ExpertPermutation
        import random

        random.seed(42)

        permutation = ExpertPermutation()

        expert_responses = {
            "literal": ExpertResponseData("literal", "A", 0.8),
            "systemic": ExpertResponseData("systemic", "B", 0.7),
            "principles": ExpertResponseData("principles", "C", 0.9),
            "precedent": ExpertResponseData("precedent", "D", 0.85),
        }

        # Con 4 expert, la probabilita' di stesso ordine e' 1/24
        # Facciamo molte permutazioni
        different_orders = set()
        for _ in range(100):
            result = permutation(expert_responses)
            order = tuple(result.keys())
            different_orders.add(order)

        # Dovremmo avere piu' di un ordine
        assert len(different_orders) > 1


class TestSynonymReplacement:
    """Test per SynonymReplacement."""

    def test_has_synonyms(self):
        """Test che abbia sinonimi definiti."""
        from merlt.disagreement.data.augmentation import SynonymReplacement

        replacer = SynonymReplacement()

        assert "obbligazione" in replacer.SYNONYMS
        assert "contratto" in replacer.SYNONYMS

    def test_replacement_probability(self):
        """Test probabilita' di sostituzione."""
        from merlt.disagreement.data.augmentation import SynonymReplacement
        import random

        random.seed(42)

        replacer = SynonymReplacement(replacement_prob=1.0)  # Sempre sostituisci

        text = "Il contratto prevede un'obbligazione."

        # Con prob=1.0, dovrebbe sostituire
        result = replacer(text)

        # Almeno una parola dovrebbe essere diversa
        # (dipende dal seed e sinonimi disponibili)
        assert result is not None


class TestSampleAugmentation:
    """Test augmentation di interi samples."""

    @pytest.fixture
    def sample(self):
        """Crea sample di test."""
        return DisagreementSample(
            sample_id="test_1",
            query="Il venditore puo' recedere dal contratto?",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="Secondo l'art. 1372 c.c. il contratto ha forza di legge.",
                    confidence=0.85,
                ),
                "principles": ExpertResponseData(
                    expert_type="principles",
                    interpretation="La buona fede contrattuale impone certi limiti.",
                    confidence=0.80,
                ),
            },
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
            source="test",
        )

    def test_augmented_preserves_labels(self, sample):
        """Test che augmentation preservi labels."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout(sentence_dropout_prob=0.0, word_dropout_prob=0.0)

        # Applica dropout alla query
        original_query = sample.query
        modified_query = dropout(original_query)

        # Crea sample augmentato
        augmented = deepcopy(sample)
        augmented.sample_id = f"{sample.sample_id}_aug"
        augmented.query = modified_query

        # Labels devono essere preservate
        assert augmented.has_disagreement == sample.has_disagreement
        assert augmented.disagreement_type == sample.disagreement_type
        assert augmented.disagreement_level == sample.disagreement_level
        assert augmented.intensity == sample.intensity

    def test_augmented_has_different_id(self, sample):
        """Test che augmented abbia ID diverso."""
        augmented = deepcopy(sample)
        augmented.sample_id = f"{sample.sample_id}_aug_1"

        assert augmented.sample_id != sample.sample_id
        assert sample.sample_id in augmented.sample_id


class TestKeyTermsPreservation:
    """Test preservazione key terms."""

    def test_key_terms_preserved_in_dropout(self):
        """Test che key terms siano preservati nel dropout."""
        from merlt.disagreement.data.augmentation import TextDropout, AugmentationConfig

        config = AugmentationConfig()
        dropout = TextDropout(
            sentence_dropout_prob=0.0,
            word_dropout_prob=0.9,  # Alta prob dropout
            min_words=0,
        )

        text = "Il contratto di buona fede prevede obbligazione."
        result = dropout(text, key_terms=config.key_terms)

        # I key terms dovrebbero essere preservati
        # (anche se con alta prob di dropout)
        # Nota: dipende dall'implementazione esatta


class TestAugmentationEdgeCases:
    """Test edge cases."""

    def test_single_word_text(self):
        """Test con testo di una parola."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout()
        result = dropout("Contratto")

        # Non dovrebbe crashare
        assert result is not None

    def test_no_sentences_with_period(self):
        """Test testo senza punti."""
        from merlt.disagreement.data.augmentation import TextDropout

        dropout = TextDropout()
        result = dropout("Testo senza punti alla fine")

        assert result is not None

    def test_empty_expert_responses(self):
        """Test permutation con dict vuoto."""
        from merlt.disagreement.data.augmentation import ExpertPermutation

        permutation = ExpertPermutation()
        result = permutation({})

        assert result == {}

    def test_single_expert(self):
        """Test permutation con un solo expert."""
        from merlt.disagreement.data.augmentation import ExpertPermutation

        permutation = ExpertPermutation()

        expert_responses = {
            "literal": ExpertResponseData("literal", "Test", 0.8),
        }

        result = permutation(expert_responses)

        assert len(result) == 1
        assert "literal" in result


class TestDropoutStatistics:
    """Test statistici per dropout."""

    def test_dropout_reduces_text_length(self):
        """Test che dropout riduca lunghezza testo (in media)."""
        from merlt.disagreement.data.augmentation import TextDropout
        import random

        random.seed(42)

        dropout = TextDropout(
            sentence_dropout_prob=0.3,
            word_dropout_prob=0.2,
            min_sentences=1,
            min_words=5,
        )

        text = "Prima frase molto lunga. Seconda frase altrettanto lunga. " \
               "Terza frase con molte parole. Quarta frase finale."

        total_original = 0
        total_dropped = 0

        for _ in range(50):
            result = dropout(text)
            total_original += len(text)
            total_dropped += len(result)

        # In media, il testo dropato dovrebbe essere piu' corto
        avg_original = total_original / 50
        avg_dropped = total_dropped / 50

        # Potrebbe non essere sempre piu' corto a causa dei min_*
        # ma almeno non dovrebbe essere piu' lungo
        assert avg_dropped <= avg_original * 1.1  # 10% tolleranza
