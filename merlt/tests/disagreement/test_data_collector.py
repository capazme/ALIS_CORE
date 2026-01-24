"""
Test Disagreement Data Collector
================================

Test per il pipeline di raccolta dati (data/collector.py).
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock
import json

from merlt.disagreement.data.collector import (
    DisagreementDataCollector,
    RLCFSource,
    OverrulingSource,
    SyntheticSource,
    CollectionStats,
)
from merlt.disagreement.types import (
    DisagreementType,
    DisagreementLevel,
    DisagreementSample,
    ExpertResponseData,
)


class TestRLCFSource:
    """Test per RLCFSource."""

    @pytest.fixture
    def mock_rlcf_db(self):
        """Mock del database RLCF."""
        db = AsyncMock()
        return db

    def test_source_properties(self):
        """Verifica proprieta' della fonte."""
        source = RLCFSource(rlcf_db=None)
        assert source.name == "rlcf"
        assert source.quality_tier == "silver"

    @pytest.mark.asyncio
    async def test_collect_no_db(self):
        """Test collect senza database configurato."""
        source = RLCFSource(rlcf_db=None)

        samples = []
        async for sample in source.collect():
            samples.append(sample)

        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_collect_with_results(self, mock_rlcf_db):
        """Test collect con risultati."""
        # Setup mock response
        mock_rlcf_db.execute.return_value = [
            {
                "response_id": "r1",
                "query": "Cos'e' la buona fede?",
                "legal_domain": "civile",
                "aggregation_metadata": json.dumps({
                    "expert_interpretations": {
                        "LiteralExpert": {
                            "interpretation": "La buona fede e' definita...",
                            "confidence": 0.8,
                            "sources": ["urn:cc:art1175"],
                        },
                        "PrinciplesExpert": {
                            "interpretation": "Principio di correttezza...",
                            "confidence": 0.85,
                            "sources": ["urn:cc:art1375"],
                        },
                    },
                    "disagreement_score": 0.4,
                    "contention_points": ["significato polisemico"],
                    "conflicting_pairs": [["literal", "principles"]],
                }),
                "created_at": datetime.now(),
            },
        ]

        source = RLCFSource(
            rlcf_db=mock_rlcf_db,
            disagreement_threshold=0.3,
        )

        samples = []
        async for sample in source.collect(limit=10):
            samples.append(sample)

        assert len(samples) == 1
        assert samples[0].sample_id == "rlcf_r1"
        assert samples[0].has_disagreement is True
        assert len(samples[0].expert_responses) == 2

    def test_infer_type_from_contention_antinomy(self):
        """Test inferenza tipo ANTINOMY."""
        source = RLCFSource(rlcf_db=None)

        dtype = source._infer_type_from_contention([
            "contrasto tra le due norme",
            "posizioni incompatibili",
        ])
        assert dtype == DisagreementType.ANTINOMY

    def test_infer_type_from_contention_methodological(self):
        """Test inferenza tipo METHODOLOGICAL."""
        source = RLCFSource(rlcf_db=None)

        dtype = source._infer_type_from_contention([
            "interpretazione letterale vs teleologica",
        ])
        assert dtype == DisagreementType.METHODOLOGICAL

    def test_infer_type_from_contention_overruling(self):
        """Test inferenza tipo OVERRULING."""
        source = RLCFSource(rlcf_db=None)

        dtype = source._infer_type_from_contention([
            "precedente superato dalla Cassazione",
        ])
        assert dtype == DisagreementType.OVERRULING

    def test_infer_level_from_pairs(self):
        """Test inferenza livello dalle coppie."""
        source = RLCFSource(rlcf_db=None)

        # Literal piu' frequente -> SEMANTIC
        level = source._infer_level_from_pairs([
            ["literal", "systemic"],
            ["literal", "principles"],
        ])
        assert level == DisagreementLevel.SEMANTIC

        # Precedent piu' frequente -> APPLICATIVE
        level = source._infer_level_from_pairs([
            ["precedent", "systemic"],
            ["precedent", "principles"],
        ])
        assert level == DisagreementLevel.APPLICATIVE

    @pytest.mark.asyncio
    async def test_count_no_db(self):
        """Test count senza database."""
        source = RLCFSource(rlcf_db=None)
        count = await source.count()
        assert count == 0


class TestOverrulingSource:
    """Test per OverrulingSource."""

    @pytest.fixture
    def mock_graph_db(self):
        """Mock del database grafo."""
        db = AsyncMock()
        return db

    def test_source_properties(self):
        """Verifica proprieta' della fonte."""
        source = OverrulingSource(graph_db=None)
        assert source.name == "overruling"
        assert source.quality_tier == "gold"

    @pytest.mark.asyncio
    async def test_collect_no_db(self):
        """Test collect senza database configurato."""
        source = OverrulingSource(graph_db=None)

        samples = []
        async for sample in source.collect():
            samples.append(sample)

        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_collect_with_results(self, mock_graph_db):
        """Test collect con relazioni overruling."""
        mock_graph_db.query.return_value = [
            {
                "new_urn": "urn:cass:2020:1234",
                "new_massima": "Il nuovo orientamento afferma...",
                "new_data": "2020-01-15",
                "old_urn": "urn:cass:2015:5678",
                "old_massima": "Il precedente orientamento...",
                "old_data": "2015-06-20",
                "motivazione": "Revirement giurisprudenziale",
            },
        ]

        source = OverrulingSource(graph_db=mock_graph_db)

        samples = []
        async for sample in source.collect():
            samples.append(sample)

        assert len(samples) == 1
        sample = samples[0]

        assert sample.has_disagreement is True
        assert sample.disagreement_type == DisagreementType.OVERRULING
        assert sample.disagreement_level == DisagreementLevel.APPLICATIVE
        assert sample.intensity == 1.0  # Overruling sempre massima intensita'
        assert sample.source == "overruling"

    @pytest.mark.asyncio
    async def test_count_no_db(self):
        """Test count senza database."""
        source = OverrulingSource(graph_db=None)
        count = await source.count()
        assert count == 0


class TestSyntheticSource:
    """Test per SyntheticSource."""

    def test_source_properties(self):
        """Verifica proprieta' della fonte."""
        source = SyntheticSource()
        assert source.name == "synthetic"
        assert source.quality_tier == "bronze"

    @pytest.mark.asyncio
    async def test_collect_not_implemented(self):
        """Test che collect restituisce vuoto (non implementato)."""
        source = SyntheticSource()

        samples = []
        async for sample in source.collect():
            samples.append(sample)

        assert len(samples) == 0

    @pytest.mark.asyncio
    async def test_generate_for_type_not_implemented(self):
        """Test generate_for_type placeholder."""
        source = SyntheticSource()

        samples = await source.generate_for_type(
            DisagreementType.ANTINOMY,
            count=5,
        )
        assert len(samples) == 0


class TestDisagreementDataCollector:
    """Test per DisagreementDataCollector."""

    @pytest.fixture
    def mock_sources(self):
        """Mock delle fonti dati."""
        rlcf = AsyncMock(spec=RLCFSource)
        rlcf.name = "rlcf"
        rlcf.quality_tier = "silver"

        async def rlcf_collect(*args, **kwargs):
            yield DisagreementSample(
                sample_id="rlcf_1",
                query="Test RLCF",
                expert_responses={},
                has_disagreement=True,
                disagreement_type=DisagreementType.METHODOLOGICAL,
                source="rlcf",
            )

        rlcf.collect = rlcf_collect

        overruling = AsyncMock(spec=OverrulingSource)
        overruling.name = "overruling"
        overruling.quality_tier = "gold"

        async def ovr_collect(*args, **kwargs):
            yield DisagreementSample(
                sample_id="ovr_1",
                query="Test Overruling",
                expert_responses={},
                has_disagreement=True,
                disagreement_type=DisagreementType.OVERRULING,
                source="overruling",
            )

        overruling.collect = ovr_collect

        return [rlcf, overruling]

    def test_init_with_sources(self, mock_sources):
        """Test init con fonti custom."""
        collector = DisagreementDataCollector(sources=mock_sources)
        assert len(collector.sources) == 2

    def test_init_with_databases(self):
        """Test init con database references."""
        collector = DisagreementDataCollector(
            rlcf_db=MagicMock(),
            graph_db=MagicMock(),
        )
        assert len(collector.sources) == 2

    @pytest.mark.asyncio
    async def test_collect_all(self, mock_sources):
        """Test raccolta da tutte le fonti."""
        collector = DisagreementDataCollector(sources=mock_sources)

        samples = await collector.collect_all()

        assert len(samples) == 2
        assert samples[0].sample_id == "rlcf_1"
        assert samples[1].sample_id == "ovr_1"

    @pytest.mark.asyncio
    async def test_collect_all_with_limit(self, mock_sources):
        """Test raccolta con limite."""
        collector = DisagreementDataCollector(sources=mock_sources)

        samples = await collector.collect_all(limit=1)

        assert len(samples) == 1

    @pytest.mark.asyncio
    async def test_collect_all_filter_sources(self, mock_sources):
        """Test raccolta filtrando fonti."""
        collector = DisagreementDataCollector(sources=mock_sources)

        samples = await collector.collect_all(sources=["rlcf"])

        # Solo samples da RLCF
        assert all(s.source == "rlcf" for s in samples)

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_sources):
        """Test statistiche raccolta."""
        collector = DisagreementDataCollector(sources=mock_sources)
        await collector.collect_all()

        stats = collector.get_stats()

        assert stats is not None
        assert stats.total_samples == 2
        assert "rlcf" in stats.by_source
        assert "overruling" in stats.by_source
        assert stats.by_quality.get("silver", 0) >= 1
        assert stats.by_quality.get("gold", 0) >= 1

    @pytest.mark.asyncio
    async def test_export_import_jsonl(self, mock_sources, tmp_path):
        """Test export/import JSONL."""
        collector = DisagreementDataCollector(sources=mock_sources)
        samples = await collector.collect_all()

        # Export
        export_path = str(tmp_path / "samples.jsonl")
        await collector.export_to_jsonl(export_path)

        # Verifica file creato
        with open(export_path, "r") as f:
            lines = f.readlines()
        assert len(lines) == 2

        # Import
        imported = await collector.import_from_jsonl(export_path)
        assert len(imported) == 2
        assert imported[0].sample_id == samples[0].sample_id


class TestCollectionStats:
    """Test per CollectionStats dataclass."""

    def test_creation(self):
        """Test creazione stats."""
        stats = CollectionStats(
            total_samples=100,
            by_source={"rlcf": 70, "overruling": 30},
            by_type={"MET": 40, "OVR": 30, "LAC": 30},
            by_quality={"silver": 70, "gold": 30},
            collection_time_seconds=5.2,
        )
        assert stats.total_samples == 100
        assert stats.by_source["rlcf"] == 70
        assert stats.collection_time_seconds == 5.2
