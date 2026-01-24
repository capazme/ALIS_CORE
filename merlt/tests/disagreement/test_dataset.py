"""
Test DisagreementDataset
=========================

Test per PyTorch Dataset del modulo disagreement.
"""

import pytest
from unittest.mock import MagicMock

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


class TestLabelEncodings:
    """Test per mappings TYPE_TO_IDX, LEVEL_TO_IDX."""

    def test_type_to_idx_completeness(self):
        """Test che tutti i tipi siano mappati."""
        from merlt.disagreement.data.dataset import TYPE_TO_IDX, IDX_TO_TYPE

        for dtype in DisagreementType:
            assert dtype in TYPE_TO_IDX
            idx = TYPE_TO_IDX[dtype]
            assert IDX_TO_TYPE[idx] == dtype

    def test_level_to_idx_completeness(self):
        """Test che tutti i livelli siano mappati."""
        from merlt.disagreement.data.dataset import LEVEL_TO_IDX, IDX_TO_LEVEL

        for level in DisagreementLevel:
            assert level in LEVEL_TO_IDX
            idx = LEVEL_TO_IDX[level]
            assert IDX_TO_LEVEL[idx] == level

    def test_expert_to_idx(self):
        """Test mapping expert names."""
        from merlt.disagreement.data.dataset import EXPERT_TO_IDX

        assert "literal" in EXPERT_TO_IDX
        assert "systemic" in EXPERT_TO_IDX
        assert "principles" in EXPERT_TO_IDX
        assert "precedent" in EXPERT_TO_IDX


class TestDisagreementDataset:
    """Test per DisagreementDataset."""

    @pytest.fixture
    def sample_data(self):
        """Crea samples di test."""
        samples = []

        # Sample con disagreement (has_disagreement != None -> is_labeled = True)
        samples.append(DisagreementSample(
            sample_id="test_1",
            query="Il venditore puo' recedere?",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="No, secondo art. 1372 c.c.",
                    confidence=0.85,
                ),
                "principles": ExpertResponseData(
                    expert_type="principles",
                    interpretation="Si, secondo buona fede.",
                    confidence=0.80,
                ),
            },
            has_disagreement=True,
            disagreement_type=DisagreementType.METHODOLOGICAL,
            disagreement_level=DisagreementLevel.TELEOLOGICAL,
            intensity=0.7,
            resolvability=0.4,
            conflicting_pairs=[("literal", "principles")],
            source="test",
        ))

        # Sample senza disagreement (has_disagreement = False -> is_labeled = True)
        samples.append(DisagreementSample(
            sample_id="test_2",
            query="Cos'e' la legittima difesa?",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="E' definita dall'art. 52 c.p.",
                    confidence=0.90,
                ),
            },
            has_disagreement=False,
            source="test",
        ))

        # Sample non labeled (has_disagreement = None -> is_labeled = False)
        samples.append(DisagreementSample(
            sample_id="test_3",
            query="Query non labellata",
            expert_responses={},
            has_disagreement=None,  # None significa non labeled
            source="test",
        ))

        return samples

    def test_initialization(self, sample_data):
        """Test inizializzazione."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)

        # Solo labeled samples
        assert len(dataset) == 2

    def test_getitem_structure(self, sample_data):
        """Test struttura item."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)
        item = dataset[0]

        assert "sample_id" in item
        assert "query_text" in item
        assert "expert_texts" in item
        assert "binary_target" in item
        assert "type_target" in item
        assert "level_target" in item
        assert "intensity_target" in item
        assert "resolvability_target" in item

    def test_expert_texts_order(self, sample_data):
        """Test che expert_texts sia ordinato per EXPERT_NAMES."""
        from merlt.disagreement.data.dataset import DisagreementDataset, EXPERT_TO_IDX

        dataset = DisagreementDataset(sample_data)
        item = dataset[0]

        # 4 expert (literal, systemic, principles, precedent)
        assert len(item["expert_texts"]) == 4

    def test_binary_target(self, sample_data):
        """Test binary target."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)

        item_dis = dataset[0]  # Con disagreement
        item_no = dataset[1]   # Senza disagreement

        assert item_dis["binary_target"] == 1
        assert item_no["binary_target"] == 0

    def test_type_target_encoding(self, sample_data):
        """Test encoding type target."""
        from merlt.disagreement.data.dataset import DisagreementDataset, TYPE_TO_IDX

        dataset = DisagreementDataset(sample_data)
        item = dataset[0]

        expected_idx = TYPE_TO_IDX[DisagreementType.METHODOLOGICAL]
        assert item["type_target"] == expected_idx

    def test_type_target_missing(self, sample_data):
        """Test type target quando mancante."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)
        item = dataset[1]  # Sample senza disagreement type

        assert item["type_target"] == -1

    def test_conflicting_pairs_encoding(self, sample_data):
        """Test encoding conflicting pairs."""
        from merlt.disagreement.data.dataset import DisagreementDataset, EXPERT_TO_IDX

        dataset = DisagreementDataset(sample_data)
        item = dataset[0]

        # Coppia (literal, principles)
        pairs = item["conflicting_pairs"]
        assert len(pairs) >= 1

        literal_idx = EXPERT_TO_IDX["literal"]
        principles_idx = EXPERT_TO_IDX["principles"]
        assert (literal_idx, principles_idx) in pairs or (principles_idx, literal_idx) in pairs

    def test_collate_fn_batching(self, sample_data):
        """Test collate_fn."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)

        batch = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch)

        assert collated["binary_target"].shape == (2,)
        assert collated["type_target"].shape == (2,)
        assert len(collated["sample_ids"]) == 2

    def test_collate_fn_masks(self, sample_data):
        """Test maschere in collate_fn."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)

        batch = [dataset[0], dataset[1]]
        collated = dataset.collate_fn(batch)

        # Mask dovrebbe essere True solo per sample 0 (type_target >= 0)
        assert "type_mask" in collated
        assert collated["type_mask"][0] == True   # Ha type
        assert collated["type_mask"][1] == False  # Non ha type

    def test_get_class_weights(self, sample_data):
        """Test calcolo class weights."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)
        weights = dataset.get_class_weights()

        assert "binary" in weights
        assert "type" in weights
        assert "level" in weights

        # Binary: 1 positivo, 1 negativo
        assert len(weights["binary"]) == 2

    def test_split(self, sample_data):
        """Test split dataset."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        # Aggiungi piu' samples per test split
        more_samples = sample_data[:2] * 5  # 10 samples labeled

        dataset = DisagreementDataset(more_samples)

        train, val, test = dataset.split(train_ratio=0.6, val_ratio=0.2, seed=42)

        assert len(train) == 6
        assert len(val) == 2
        assert len(test) == 2

    def test_get_stats(self, sample_data):
        """Test statistiche dataset."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(sample_data)
        stats = dataset.get_stats()

        assert stats["total_samples"] == 3
        assert stats["labeled_samples"] == 2
        assert stats["unlabeled_samples"] == 1

        assert "binary_distribution" in stats
        assert stats["binary_distribution"]["has_disagreement"] == 1
        assert stats["binary_distribution"]["no_disagreement"] == 1


class TestDisagreementDatasetWithTokenizer:
    """Test con tokenizer."""

    @pytest.fixture
    def mock_tokenizer(self):
        """Crea mock tokenizer."""
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 128)),
            "attention_mask": torch.ones(1, 128),
        }
        return tokenizer

    @pytest.fixture
    def sample_with_dis(self):
        """Sample singolo con disagreement."""
        return DisagreementSample(
            sample_id="test",
            query="Test query",
            expert_responses={
                "literal": ExpertResponseData(
                    expert_type="literal",
                    interpretation="Test interpretation",
                    confidence=0.85,
                ),
            },
            has_disagreement=True,
            source="test",
        )

    def test_tokenization(self, mock_tokenizer, sample_with_dis):
        """Test che tokenizer venga chiamato."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        dataset = DisagreementDataset(
            [sample_with_dis],
            tokenizer=mock_tokenizer,
            max_length=128,
        )

        item = dataset[0]

        assert "expert_input_ids" in item
        assert "expert_attention_mask" in item
        assert item["expert_input_ids"].shape == (4, 128)

    def test_embedding_fn(self, sample_with_dis):
        """Test con embedding function."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        def mock_embedding(text):
            return torch.randn(768)

        dataset = DisagreementDataset(
            [sample_with_dis],
            embedding_fn=mock_embedding,
        )

        item = dataset[0]

        assert "expert_embeddings" in item
        assert item["expert_embeddings"].shape == (4, 768)

    def test_embedding_caching(self, sample_with_dis):
        """Test caching embeddings."""
        from merlt.disagreement.data.dataset import DisagreementDataset

        call_count = {"count": 0}

        def mock_embedding(text):
            call_count["count"] += 1
            return torch.randn(768)

        dataset = DisagreementDataset(
            [sample_with_dis],
            embedding_fn=mock_embedding,
            cache_embeddings=True,
        )

        # Prima chiamata
        _ = dataset[0]
        count_after_first = call_count["count"]

        # Seconda chiamata (dovrebbe usare cache)
        _ = dataset[0]
        count_after_second = call_count["count"]

        assert count_after_second == count_after_first  # Cache hit


class TestStreamingDisagreementDataset:
    """Test per StreamingDisagreementDataset."""

    @pytest.fixture
    def jsonl_file(self, tmp_path):
        """Crea file JSONL di test."""
        import json

        file_path = tmp_path / "test_data.jsonl"

        samples = [
            {
                "sample_id": "s1",
                "query": "Query 1",
                "expert_responses": {
                    "literal": {
                        "expert_type": "literal",
                        "interpretation": "Test 1",
                        "confidence": 0.85,
                    }
                },
                "has_disagreement": True,
                "is_labeled": True,
                "source": "test",
            },
            {
                "sample_id": "s2",
                "query": "Query 2",
                "expert_responses": {},
                "has_disagreement": False,
                "is_labeled": True,
                "source": "test",
            },
        ]

        with open(file_path, "w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")

        return str(file_path)

    def test_len(self, jsonl_file):
        """Test conteggio righe."""
        from merlt.disagreement.data.dataset import StreamingDisagreementDataset

        dataset = StreamingDisagreementDataset(jsonl_file)

        assert len(dataset) == 2

    def test_iteration(self, jsonl_file):
        """Test iterazione."""
        from merlt.disagreement.data.dataset import StreamingDisagreementDataset

        dataset = StreamingDisagreementDataset(jsonl_file)

        items = list(dataset)

        # Solo labeled samples con content
        assert len(items) >= 1
        assert "sample_id" in items[0]
