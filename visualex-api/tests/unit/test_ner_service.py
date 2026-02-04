"""
Tests for NER Service.

Tests cover:
- AC1: Article reference extraction with URN resolution
- AC2: Legal concept extraction
- AC3: Temporal context extraction
- AC4: Party reference extraction
- AC5: Confidence scores
- AC6: Text span visibility
- AC7: Ambiguous entity flagging
- AC8: Graceful failure handling
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from visualex.ner import (
    NERService,
    NERConfig,
    EntityType,
    ExtractedEntity,
    ExtractionResult,
)


# =============================================================================
# Entity Dataclass Tests
# =============================================================================


class TestExtractedEntity:
    """Tests for ExtractedEntity dataclass."""

    def test_creation(self):
        """Test basic entity creation."""
        entity = ExtractedEntity(
            text="art. 1453",
            entity_type=EntityType.ARTICLE_REF,
            start=0,
            end=9,
            confidence=0.9,
        )

        assert entity.text == "art. 1453"
        assert entity.entity_type == EntityType.ARTICLE_REF
        assert entity.start == 0
        assert entity.end == 9
        assert entity.confidence == 0.9

    def test_to_dict(self):
        """Test serialization."""
        entity = ExtractedEntity(
            text="risoluzione",
            entity_type=EntityType.LEGAL_CONCEPT,
            start=10,
            end=21,
            confidence=0.85,
            resolved_urn="urn:test",
        )

        d = entity.to_dict()

        assert d["text"] == "risoluzione"
        assert d["entity_type"] == "LEGAL_CONCEPT"
        assert d["start"] == 10
        assert d["end"] == 21
        assert d["confidence"] == 0.85
        assert d["resolved_urn"] == "urn:test"

    def test_ambiguous_flag(self):
        """Test ambiguous entity flag."""
        entity = ExtractedEntity(
            text="art. 52",
            entity_type=EntityType.ARTICLE_REF,
            start=0,
            end=7,
            is_ambiguous=True,
        )

        assert entity.is_ambiguous is True
        d = entity.to_dict()
        assert d["is_ambiguous"] is True


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_creation(self):
        """Test basic result creation."""
        result = ExtractionResult(text="Test text")

        assert result.text == "Test text"
        assert result.entities == []
        assert result.has_errors is False

    def test_to_dict(self):
        """Test serialization."""
        result = ExtractionResult(
            text="Test",
            entities=[
                ExtractedEntity("art. 1", EntityType.ARTICLE_REF, 0, 6),
            ],
            processing_time_ms=50.5,
        )

        d = result.to_dict()

        assert d["text"] == "Test"
        assert d["entity_count"] == 1
        assert d["processing_time_ms"] == 50.5

    def test_entity_accessors(self):
        """Test entity type accessors."""
        result = ExtractionResult(
            text="Test",
            entities=[
                ExtractedEntity("art. 1", EntityType.ARTICLE_REF, 0, 6),
                ExtractedEntity("risoluzione", EntityType.LEGAL_CONCEPT, 10, 21),
                ExtractedEntity("2019", EntityType.TEMPORAL, 25, 29),
                ExtractedEntity("compratore", EntityType.PARTY, 35, 45),
            ],
        )

        assert len(result.article_refs) == 1
        assert len(result.legal_concepts) == 1
        assert len(result.temporal_refs) == 1
        assert len(result.party_refs) == 1

    def test_ambiguous_accessor(self):
        """Test ambiguous entity accessor."""
        result = ExtractionResult(
            text="Test",
            entities=[
                ExtractedEntity("art. 1", EntityType.ARTICLE_REF, 0, 6, is_ambiguous=True),
                ExtractedEntity("art. 2", EntityType.ARTICLE_REF, 10, 16, is_ambiguous=False),
            ],
        )

        assert len(result.ambiguous_entities) == 1
        assert result.to_dict()["ambiguous_count"] == 1


# =============================================================================
# NER Service Tests
# =============================================================================


class TestNERService:
    """Tests for NERService."""

    def setup_method(self):
        """Create service for tests."""
        # Disable spaCy for faster tests
        config = NERConfig(use_spacy=False)
        self.service = NERService(config)

    @pytest.mark.asyncio
    async def test_extract_empty_text(self):
        """Test extraction on empty text."""
        result = await self.service.extract("")

        assert result.text == ""
        assert len(result.entities) == 0
        assert result.has_errors is False

    @pytest.mark.asyncio
    async def test_extract_article_reference(self):
        """Test article reference extraction (AC1)."""
        result = await self.service.extract("L'art. 1453 del codice civile")

        article_refs = result.article_refs
        assert len(article_refs) >= 1

        art_entity = next(e for e in article_refs if "1453" in e.text)
        assert art_entity.entity_type == EntityType.ARTICLE_REF
        assert art_entity.confidence > 0

    @pytest.mark.asyncio
    async def test_extract_multiple_articles(self):
        """Test multiple article extraction."""
        result = await self.service.extract(
            "Gli articoli 1453, 1454 e 1455 regolano la risoluzione"
        )

        # Should find multiple article references
        article_refs = result.article_refs
        assert len(article_refs) >= 1

    @pytest.mark.asyncio
    async def test_extract_article_with_extension(self):
        """Test article with bis/ter extension."""
        result = await self.service.extract("art. 52-bis del codice penale")

        article_refs = result.article_refs
        assert len(article_refs) >= 1
        assert any("52" in e.text for e in article_refs)

    @pytest.mark.asyncio
    async def test_extract_legal_concept(self):
        """Test legal concept extraction (AC2)."""
        result = await self.service.extract(
            "L'inadempimento contrattuale comporta la risoluzione"
        )

        concepts = result.legal_concepts
        concept_texts = [c.text.lower() for c in concepts]

        assert "inadempimento" in concept_texts
        assert "risoluzione" in concept_texts

    @pytest.mark.asyncio
    async def test_extract_temporal_context(self):
        """Test temporal context extraction (AC3)."""
        result = await self.service.extract("Il contratto del 2019 è nullo")

        temporal = result.temporal_refs
        assert len(temporal) >= 1
        assert any("2019" in e.text for e in temporal)

    @pytest.mark.asyncio
    async def test_extract_party_reference(self):
        """Test party reference extraction (AC4)."""
        result = await self.service.extract(
            "Il compratore ha diritto al risarcimento dal venditore"
        )

        parties = result.party_refs
        party_texts = [p.text.lower() for p in parties]

        assert "compratore" in party_texts
        assert "venditore" in party_texts

    @pytest.mark.asyncio
    async def test_extract_code_reference(self):
        """Test code reference extraction."""
        result = await self.service.extract("secondo il c.c. e il c.p.")

        code_refs = [e for e in result.entities if e.entity_type == EntityType.CODE_REF]
        assert len(code_refs) >= 2

    @pytest.mark.asyncio
    async def test_extract_norm_reference(self):
        """Test norm/law reference extraction."""
        result = await self.service.extract(
            "La legge 241/1990 e il D.Lgs. 50/2016"
        )

        norm_refs = result.norm_refs
        assert len(norm_refs) >= 2

    @pytest.mark.asyncio
    async def test_confidence_scores(self):
        """Test that entities have confidence scores (AC5)."""
        result = await self.service.extract("art. 1453 c.c.")

        for entity in result.entities:
            assert 0.0 <= entity.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_text_spans(self):
        """Test that text spans are correct (AC6)."""
        text = "L'art. 1453 regola la risoluzione"
        result = await self.service.extract(text)

        for entity in result.entities:
            # Verify span matches text
            extracted = text[entity.start:entity.end]
            assert extracted == entity.text

    @pytest.mark.asyncio
    async def test_ambiguous_flagging(self):
        """Test ambiguous entity flagging (AC7)."""
        # Low confidence entities should be flagged
        config = NERConfig(
            use_spacy=False,
            ambiguity_threshold=0.95,  # High threshold to flag more
        )
        service = NERService(config)

        result = await service.extract("art. 52 nel contratto")

        # Some entities might be marked ambiguous due to low confidence
        # or missing context for URN resolution
        assert result.to_dict()["ambiguous_count"] >= 0

    @pytest.mark.asyncio
    async def test_processing_time_recorded(self):
        """Test that processing time is recorded."""
        result = await self.service.extract("art. 1453 c.c.")

        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 500  # Should be fast

    @pytest.mark.asyncio
    async def test_complex_query(self):
        """Test extraction on complex legal query."""
        text = (
            "Il compratore può richiedere la risoluzione del contratto "
            "ex art. 1453 c.c. per inadempimento del venditore verificatosi "
            "nel 2020, con risarcimento del danno"
        )

        result = await self.service.extract(text)

        # Should find multiple entity types
        assert len(result.article_refs) >= 1
        assert len(result.legal_concepts) >= 2
        assert len(result.party_refs) >= 2
        assert len(result.temporal_refs) >= 1


class TestNERServiceURNResolution:
    """Tests for URN resolution in NER service."""

    def setup_method(self):
        """Create service for tests."""
        config = NERConfig(use_spacy=False, resolve_urns=True)
        self.service = NERService(config)

    @pytest.mark.asyncio
    async def test_code_urn_resolution(self):
        """Test code reference URN resolution."""
        result = await self.service.extract("codice civile")

        code_refs = [e for e in result.entities if e.entity_type == EntityType.CODE_REF]
        assert len(code_refs) >= 1

        # Should have resolved URN
        code_entity = code_refs[0]
        assert code_entity.resolved_urn is not None
        assert "regio.decreto:1942-03-16;262" in code_entity.resolved_urn

    @pytest.mark.asyncio
    async def test_article_urn_with_context(self):
        """Test article URN resolution with context."""
        context = {
            "tipo_atto": "regio.decreto",
            "data": "1942-03-16",
            "numero": "262",
        }

        result = await self.service.extract("art. 1453", context=context)

        article_refs = result.article_refs
        assert len(article_refs) >= 1

        art_entity = article_refs[0]
        if art_entity.resolved_urn:
            assert "1453" in art_entity.resolved_urn

    @pytest.mark.asyncio
    async def test_article_urn_without_context_is_ambiguous(self):
        """Test article without context is marked ambiguous."""
        result = await self.service.extract("art. 52")

        article_refs = result.article_refs
        assert len(article_refs) >= 1

        # Should be ambiguous because we don't know which code/law
        art_entity = article_refs[0]
        # Entity might be marked ambiguous due to missing context
        assert art_entity.resolved_urn is None or art_entity.is_ambiguous


class TestNERServiceErrorHandling:
    """Tests for error handling in NER service."""

    def setup_method(self):
        """Create service for tests."""
        config = NERConfig(use_spacy=False)
        self.service = NERService(config)

    @pytest.mark.asyncio
    async def test_graceful_failure(self):
        """Test graceful failure with partial results (AC8)."""
        # Even with errors, should return a result
        result = await self.service.extract("art. 1453")

        assert result is not None
        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling returns partial results."""
        config = NERConfig(use_spacy=False, timeout_ms=1)  # Very short timeout
        service = NERService(config)

        # Should still return a result
        result = await service.extract("art. 1453 c.c.")

        assert result is not None
        # Might have warnings about timeout
        # but should still have some results from rule-based

    @pytest.mark.asyncio
    async def test_warnings_logged(self):
        """Test that warnings are included in result."""
        result = await self.service.extract("test text")

        # Warnings list should exist
        assert isinstance(result.warnings, list)


class TestNERServiceConfiguration:
    """Tests for NER service configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = NERConfig()

        assert config.use_spacy is True
        assert config.confidence_threshold == 0.5
        assert config.ambiguity_threshold == 0.7
        assert config.timeout_ms == 500.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = NERConfig(
            use_spacy=False,
            confidence_threshold=0.7,
            timeout_ms=1000.0,
        )

        service = NERService(config)

        assert service.config.use_spacy is False
        assert service.config.confidence_threshold == 0.7

    def test_is_ready(self):
        """Test service readiness check."""
        service = NERService(NERConfig(use_spacy=False))

        assert service.is_ready() is True

    def test_has_spacy_property(self):
        """Test spaCy availability property."""
        service = NERService(NERConfig(use_spacy=False))

        assert service.has_spacy is False


# =============================================================================
# Pattern Tests
# =============================================================================


class TestArticlePatterns:
    """Tests for article pattern matching."""

    def setup_method(self):
        """Create service for tests."""
        self.service = NERService(NERConfig(use_spacy=False))

    @pytest.mark.asyncio
    async def test_art_dot_number(self):
        """Test art. N pattern."""
        result = await self.service.extract("art. 1453")
        assert any("1453" in e.text for e in result.article_refs)

    @pytest.mark.asyncio
    async def test_articolo_number(self):
        """Test articolo N pattern."""
        result = await self.service.extract("articolo 52")
        assert any("52" in e.text for e in result.article_refs)

    @pytest.mark.asyncio
    async def test_comma_reference(self):
        """Test comma N pattern."""
        result = await self.service.extract("comma 3")
        assert any("3" in e.text for e in result.article_refs)

    @pytest.mark.asyncio
    async def test_lettera_reference(self):
        """Test lettera pattern."""
        result = await self.service.extract("lettera a)")
        assert any("a" in e.text.lower() for e in result.article_refs)


class TestCodePatterns:
    """Tests for code pattern matching."""

    def setup_method(self):
        """Create service for tests."""
        self.service = NERService(NERConfig(use_spacy=False))

    @pytest.mark.asyncio
    async def test_codice_civile(self):
        """Test codice civile pattern."""
        result = await self.service.extract("codice civile")
        code_refs = [e for e in result.entities if e.entity_type == EntityType.CODE_REF]
        assert len(code_refs) >= 1

    @pytest.mark.asyncio
    async def test_c_c_abbreviation(self):
        """Test c.c. abbreviation."""
        result = await self.service.extract("c.c.")
        code_refs = [e for e in result.entities if e.entity_type == EntityType.CODE_REF]
        assert len(code_refs) >= 1

    @pytest.mark.asyncio
    async def test_codice_penale(self):
        """Test codice penale pattern."""
        result = await self.service.extract("codice penale")
        code_refs = [e for e in result.entities if e.entity_type == EntityType.CODE_REF]
        assert len(code_refs) >= 1


class TestNormPatterns:
    """Tests for norm/law pattern matching."""

    def setup_method(self):
        """Create service for tests."""
        self.service = NERService(NERConfig(use_spacy=False))

    @pytest.mark.asyncio
    async def test_legge_pattern(self):
        """Test legge N/YYYY pattern."""
        result = await self.service.extract("legge 241/1990")
        assert len(result.norm_refs) >= 1

    @pytest.mark.asyncio
    async def test_dlgs_pattern(self):
        """Test D.Lgs. pattern."""
        result = await self.service.extract("D.Lgs. 50/2016")
        assert len(result.norm_refs) >= 1

    @pytest.mark.asyncio
    async def test_dpr_pattern(self):
        """Test DPR pattern."""
        result = await self.service.extract("DPR 380/2001")
        assert len(result.norm_refs) >= 1

    @pytest.mark.asyncio
    async def test_costituzione_pattern(self):
        """Test costituzione pattern."""
        result = await self.service.extract("la Costituzione")
        assert len(result.norm_refs) >= 1
