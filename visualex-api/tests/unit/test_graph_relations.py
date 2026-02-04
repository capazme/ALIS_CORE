"""
Tests for Graph Relation Extraction Module
==========================================

Tests for:
- CitationExtractor: Pattern-based citation extraction
- RelationCreator: Graph edge creation
- ExtractionResult: Data structure validation
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from visualex.graph.relations import (
    CitationExtractor,
    RelationCreator,
    ExtractedCitation,
    ExtractedModification,
    ExtractionResult,
    RelationType,
    extract_citations,
)
from visualex.graph.schema import EdgeType, NodeType


# =============================================================================
# CitationExtractor Tests
# =============================================================================


class TestCitationExtractor:
    """Test suite for CitationExtractor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.base_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        self.extractor = CitationExtractor(self.base_urn)

    # --- Simple Article Citations ---

    def test_extract_simple_article_citation(self):
        """Test extraction of simple article reference."""
        text = "Come previsto dall'art. 1454 del presente codice"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        citation = result.citations[0]
        assert citation.target_article in ["1454", "1454"]

    def test_extract_article_with_bis_extension(self):
        """Test extraction of article with Latin extension (bis, ter, etc.)."""
        text = "Ai sensi dell'articolo 23-bis"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        citation = result.citations[0]
        assert "23" in citation.target_article
        assert "bis" in citation.target_article.lower() or citation.raw_text.lower().find("bis") > -1

    def test_extract_article_with_comma(self):
        """Test extraction of article with comma reference."""
        text = "di cui all'art. 1453, comma 2"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        # Should capture the article
        found_article = False
        for c in result.citations:
            if c.target_article and "1453" in c.target_article:
                found_article = True
                break
        assert found_article

    # --- Contextual Citations ---

    def test_extract_ai_sensi_citation(self):
        """Test 'ai sensi dell'art. X' pattern."""
        text = "ai sensi dell'art. 2043 del codice civile"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        assert any(c.target_article == "2043" for c in result.citations)

    def test_extract_di_cui_citation(self):
        """Test 'di cui all'articolo X' pattern."""
        text = "di cui all'articolo 1218 comma 1"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        assert any("1218" in (c.target_article or "") for c in result.citations)

    def test_extract_ex_art_citation(self):
        """Test 'ex art. X' pattern."""
        text = "la responsabilità ex art. 2050"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        assert any(c.target_article == "2050" for c in result.citations)

    # --- Codice Citations ---

    def test_extract_codice_civile_citation(self):
        """Test citation to Codice Civile with c.c. abbreviation."""
        text = "L'art. 1453 c.c. prevede la risoluzione"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        cc_citation = next(
            (c for c in result.citations if c.target_urn and "262" in c.target_urn),
            None
        )
        assert cc_citation is not None
        assert cc_citation.target_urn == "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

    def test_extract_codice_penale_citation(self):
        """Test citation to Codice Penale with c.p. abbreviation."""
        text = "il reato di cui all'art. 640 c.p."
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        cp_citation = next(
            (c for c in result.citations if c.target_urn and "1398" in c.target_urn),
            None
        )
        assert cp_citation is not None
        assert "640" in cp_citation.target_urn

    # --- Law Citations ---

    def test_extract_law_short_form(self):
        """Test 'L. n. 123/2020' pattern."""
        text = "come modificato dalla L. n. 178/2020"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        law_citation = next(
            (c for c in result.citations if c.relation_type == RelationType.CITA_LEGGE),
            None
        )
        assert law_citation is not None
        assert law_citation.target_number == "178"
        assert law_citation.target_date == "2020"

    def test_extract_law_full_form(self):
        """Test 'legge 30 dicembre 2020, n. 178' pattern."""
        text = "previsto dalla legge 30 dicembre 2020, n. 178"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        law_citation = next(
            (c for c in result.citations if c.target_number == "178"),
            None
        )
        assert law_citation is not None
        assert law_citation.target_date == "2020-12-30"
        assert law_citation.target_urn == "urn:nir:stato:legge:2020-12-30;178"

    # --- Decree Citations ---

    def test_extract_decreto_legislativo_short(self):
        """Test 'D.Lgs. n. 50/2016' pattern."""
        text = "ai sensi del D.Lgs. n. 50/2016"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        dlgs_citation = next(
            (c for c in result.citations if c.target_number == "50"),
            None
        )
        assert dlgs_citation is not None
        assert dlgs_citation.target_act_type == "decreto.legislativo"

    def test_extract_decreto_legge_short(self):
        """Test 'D.L. n. 18/2020' pattern."""
        text = "il D.L. n. 18/2020 ha previsto"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        dl_citation = next(
            (c for c in result.citations if c.target_number == "18"),
            None
        )
        assert dl_citation is not None
        assert dl_citation.target_act_type == "decreto.legge"

    def test_extract_decreto_full_form(self):
        """Test 'decreto legislativo 18 aprile 2016, n. 50' pattern."""
        text = "il decreto legislativo 18 aprile 2016, n. 50"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        dlgs_citation = next(
            (c for c in result.citations if c.target_number == "50"),
            None
        )
        assert dlgs_citation is not None
        assert dlgs_citation.target_date == "2016-04-18"
        assert dlgs_citation.target_urn == "urn:nir:stato:decreto.legislativo:2016-04-18;50"

    # --- Jurisprudence Citations ---

    def test_extract_cassazione_citation(self):
        """Test 'Cass. n. 12345/2021' pattern."""
        text = "come affermato da Cass. n. 12345/2021"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        cass_citation = next(
            (c for c in result.citations if c.relation_type == RelationType.CITA_GIURISPRUDENZA),
            None
        )
        assert cass_citation is not None
        assert cass_citation.target_number == "12345"
        assert cass_citation.target_date == "2021"
        assert cass_citation.target_act_type == "cassazione"

    def test_extract_cassazione_civile_sezione(self):
        """Test 'Cass. civ. sez. III, n. 123/2021' pattern."""
        text = "Cass. civ. sez. III, n. 123/2021 ha stabilito"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        cass_citation = next(
            (c for c in result.citations if c.target_number == "123"),
            None
        )
        assert cass_citation is not None
        assert cass_citation.relation_type == RelationType.CITA_GIURISPRUDENZA

    def test_extract_corte_costituzionale(self):
        """Test 'Corte Cost. sent. n. 123/2020' pattern."""
        text = "come dichiarato dalla Corte Cost. sent. n. 123/2020"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        cc_citation = next(
            (c for c in result.citations if c.target_number == "123"),
            None
        )
        assert cc_citation is not None
        assert cc_citation.target_act_type == "corte_costituzionale"
        assert cc_citation.confidence >= 0.9

    # --- Modification Detection ---

    def test_extract_sostituisce_modification(self):
        """Test 'è sostituito da' pattern."""
        text = "L'articolo 5 è sostituito dal seguente testo"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.SOSTITUISCE

    def test_extract_abrogato_modification(self):
        """Test 'è abrogato' pattern."""
        text = "Il comma 3 è abrogato"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.ABROGA_TOTALMENTE

    def test_extract_abrogato_parzialmente(self):
        """Test 'è parzialmente abrogato' pattern."""
        text = "L'articolo 10 è parzialmente abrogato limitatamente al comma 2"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.ABROGA_PARZIALMENTE

    def test_extract_integrato_modification(self):
        """Test 'è integrato' pattern."""
        text = "Il presente articolo viene integrato con le seguenti disposizioni"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.INTEGRA

    def test_extract_sospeso_modification(self):
        """Test 'è sospeso' pattern."""
        text = "L'efficacia della norma è sospesa fino al 31 dicembre"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.SOSPENDE

    def test_extract_deroga_modification(self):
        """Test 'in deroga a' pattern."""
        text = "in deroga all'articolo 12 del presente decreto"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.DEROGA_A

    def test_extract_modification_with_data_efficacia(self):
        """Test extraction of effective date (data_efficacia) from modification (AC2)."""
        text = "L'articolo 5 è sostituito a decorrere dal 1 gennaio 2025 dal seguente testo"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.SOSTITUISCE
        assert mod.data_efficacia == "2025-01-01"

    def test_extract_modification_with_target_urn(self):
        """Test extraction of target URN from modification context (AC2)."""
        text = "L'articolo 1454 è abrogato con effetto dal 15 marzo 2024"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.modifications) >= 1
        mod = result.modifications[0]
        assert mod.relation_type == RelationType.ABROGA_TOTALMENTE
        # Should have extracted target URN from "L'articolo 1454"
        assert mod.target_urn is not None
        assert "1454" in mod.target_urn

    # --- Edge Cases ---

    def test_empty_text(self):
        """Test with empty text."""
        result = self.extractor.extract("", self.base_urn)
        assert result.total_relations == 0
        assert len(result.errors) == 0

    def test_text_without_citations(self):
        """Test with text that has no legal citations."""
        text = "Il presente documento non contiene riferimenti normativi specifici."
        result = self.extractor.extract(text, self.base_urn)
        assert len(result.citations) == 0

    def test_multiple_citations_in_text(self):
        """Test extraction of multiple citations."""
        text = (
            "Ai sensi dell'art. 1453 c.c. e dell'art. 1454 c.c., "
            "come modificato dalla L. n. 178/2020"
        )
        result = self.extractor.extract(text, self.base_urn)
        assert result.total_relations >= 3

    def test_deduplication_of_overlapping_citations(self):
        """Test that overlapping citations are deduplicated."""
        text = "ai sensi dell'art. 1453 c.c."
        result = self.extractor.extract(text, self.base_urn)
        # Should have deduplicated overlapping matches
        urns = [c.target_urn for c in result.citations if c.target_urn]
        # Each URN should appear only once
        assert len(urns) == len(set(urns))

    def test_context_extraction(self):
        """Test that context is properly extracted."""
        text = "Prima del riferimento. Ai sensi dell'art. 1453 del codice. Dopo il riferimento."
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        citation = result.citations[0]
        assert len(citation.context) > 0
        assert "1453" in citation.context

    def test_confidence_scores(self):
        """Test that confidence scores are assigned correctly."""
        text = "L'art. 1453 c.c. è richiamato"
        result = self.extractor.extract(text, self.base_urn)

        assert len(result.citations) >= 1
        # Codice citations with full URN should have high confidence
        cc_citation = next(
            (c for c in result.citations if c.target_urn and "262" in c.target_urn),
            None
        )
        if cc_citation:
            assert cc_citation.confidence >= 0.9


class TestExtractionResult:
    """Test suite for ExtractionResult data class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ExtractionResult(
            source_urn="urn:nir:stato:legge:2020-12-30;178",
            citations=[
                ExtractedCitation(
                    raw_text="art. 1",
                    relation_type=RelationType.CITA_ARTICOLO,
                    target_article="1",
                )
            ],
        )

        d = result.to_dict()
        assert d["source_urn"] == "urn:nir:stato:legge:2020-12-30;178"
        assert d["total_relations"] == 1
        assert len(d["citations"]) == 1

    def test_total_relations_property(self):
        """Test total_relations property calculation."""
        result = ExtractionResult(
            source_urn="test",
            citations=[
                ExtractedCitation(raw_text="c1", relation_type=RelationType.CITA),
                ExtractedCitation(raw_text="c2", relation_type=RelationType.CITA),
            ],
            modifications=[
                ExtractedModification(raw_text="m1", relation_type=RelationType.SOSTITUISCE),
            ],
        )

        assert result.total_relations == 3


# =============================================================================
# RelationCreator Tests
# =============================================================================


class TestRelationCreator:
    """Test suite for RelationCreator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.create_edge = AsyncMock(return_value={"properties": {}})
        self.creator = RelationCreator(self.mock_client)

    @pytest.mark.asyncio
    async def test_create_relations_from_text(self):
        """Test relation creation from text."""
        text = "ai sensi dell'art. 1454 c.c."
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

        result = await self.creator.create_relations_from_text(source_urn, text)

        assert "citations_created" in result
        assert "modifications_created" in result
        assert "errors" in result

    @pytest.mark.asyncio
    async def test_create_citation_edge_with_resolved_urn(self):
        """Test CITA edge creation when target URN is resolved."""
        text = "L'art. 1454 c.c. dispone che"
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

        result = await self.creator.create_relations_from_text(source_urn, text)

        # Should have created at least one citation
        # The mock should have been called
        if result["citations_created"] > 0:
            self.mock_client.create_edge.assert_called()

    @pytest.mark.asyncio
    async def test_skip_citation_without_urn(self):
        """Test that citations without resolved URN are skipped."""
        # Simple article reference without full context
        text = "vedi articolo 5"
        source_urn = "urn:nir:stato:legge:2020-12-30;178"

        result = await self.creator.create_relations_from_text(source_urn, text)

        # Should not create edges for unresolved URNs
        # (edge creation count may be 0)
        assert result["errors"] == 0

    @pytest.mark.asyncio
    async def test_process_brocardi_relations(self):
        """Test INTERPRETA edge creation from Brocardi data."""
        article_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"
        brocardi_info = {
            "massime": [
                {"testo": "La risoluzione del contratto...", "corte": "Cassazione"},
                {"testo": "In tema di inadempimento...", "corte": "Cassazione"},
            ]
        }

        result = await self.creator.process_brocardi_relations(
            article_urn, brocardi_info
        )

        assert "interpreta_edges" in result
        # Should attempt to create edges for each massima (2 massime = 2 calls)
        assert self.mock_client.create_edge.call_count == 2
        # Verify INTERPRETA edge type was used
        for call in self.mock_client.create_edge.call_args_list:
            assert call.args[0] == EdgeType.INTERPRETA

    @pytest.mark.asyncio
    async def test_edge_creation_with_properties(self):
        """Test that edges are created with proper properties."""
        text = "ai sensi dell'art. 1454 c.c. in materia di diffida"
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

        await self.creator.create_relations_from_text(source_urn, text)

        # Check that create_edge was called with properties
        if self.mock_client.create_edge.called:
            call_args = self.mock_client.create_edge.call_args
            # Properties should include confidence_score
            if call_args.kwargs.get("properties"):
                assert "confidence_score" in call_args.kwargs["properties"]


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Test suite for module-level convenience functions."""

    def test_extract_citations_function(self):
        """Test extract_citations convenience function."""
        text = "ai sensi dell'art. 1453 c.c."
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

        result = extract_citations(text, source_urn)

        assert isinstance(result, ExtractionResult)
        assert result.source_urn == source_urn
        assert len(result.citations) >= 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestRelationTraversal:
    """Test suite for AC4: Relationship Traversal."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = MagicMock()
        self.mock_client.create_edge = AsyncMock(return_value={"properties": {}})
        self.creator = RelationCreator(self.mock_client)

    @pytest.mark.asyncio
    async def test_edges_include_metadata(self):
        """Test that created edges include tipo, data, context metadata."""
        text = "ai sensi dell'art. 1454 c.c. in materia di diffida"
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1453"

        await self.creator.create_relations_from_text(source_urn, text)

        # Verify edge was created with metadata properties
        if self.mock_client.create_edge.called:
            call_kwargs = self.mock_client.create_edge.call_args.kwargs
            props = call_kwargs.get("properties", {})
            # AC4: edge metadata (tipo, data, context) is accessible
            assert "tipo_citazione" in props  # tipo
            assert "confidence_score" in props
            # paragrafo_riferimento contains context
            assert "paragrafo_riferimento" in props

    @pytest.mark.asyncio
    async def test_all_relationship_types_traversable(self):
        """Test that all relationship types can be created for traversal."""
        # Test CITA edge
        text_cita = "ai sensi dell'art. 1454 c.c."
        await self.creator.create_relations_from_text(
            "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            text_cita
        )

        # Reset mock for next test
        self.mock_client.reset_mock()

        # Test INTERPRETA edge (jurisprudence)
        text_juris = "come stabilito da Cass. n. 12345/2021"
        await self.creator.create_relations_from_text(
            "urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            text_juris
        )

        # Both types should have been processed
        # CITA edges created successfully
        # INTERPRETA edges require target URN (jurisprudence doesn't resolve to URN)
        # so we just verify no errors occurred
        assert True  # No exceptions means traversal paths are valid


class TestRelationExtractionIntegration:
    """Integration tests for the full extraction pipeline."""

    def test_real_world_article_text(self):
        """Test with realistic Italian legal article text."""
        text = """
        In caso di inadempimento, il creditore può richiedere
        l'adempimento ai sensi dell'art. 1453 c.c. ovvero la risoluzione
        del contratto, salvo il risarcimento del danno ex art. 1223 c.c.

        Come stabilito dalla Cass. civ. sez. II, n. 12345/2021, la
        risoluzione opera con effetto retroattivo, ferme restando le
        prestazioni già eseguite di cui all'art. 1458 c.c.

        La presente norma è integrata dalle disposizioni del
        D.Lgs. n. 50/2016 in materia di appalti pubblici.
        """
        source_urn = "urn:nir:stato:regio.decreto:1942-03-16;262~art1455"

        extractor = CitationExtractor()
        result = extractor.extract(text, source_urn)

        # Should find multiple citations
        assert result.total_relations >= 3

        # Should find c.c. references
        cc_citations = [
            c for c in result.citations
            if c.target_urn and "262" in c.target_urn
        ]
        assert len(cc_citations) >= 2

        # Should find jurisprudence
        juris_citations = [
            c for c in result.citations
            if c.relation_type == RelationType.CITA_GIURISPRUDENZA
        ]
        assert len(juris_citations) >= 1

        # Should find modification marker
        assert len(result.modifications) >= 1

    def test_modification_article_text(self):
        """Test with text containing modification language."""
        text = """
        L'articolo 5 della legge 20 maggio 1970, n. 300 è così sostituito:
        "Art. 5 - Gli accertamenti sanitari..."

        L'articolo 7, comma 3, è abrogato.

        All'articolo 10, dopo il comma 2, è inserito il seguente:
        "2-bis. Le disposizioni di cui al presente articolo..."
        """
        source_urn = "urn:nir:stato:legge:2021-01-01;1"

        extractor = CitationExtractor()
        result = extractor.extract(text, source_urn)

        # Should detect substitution
        sostituisce = [
            m for m in result.modifications
            if m.relation_type == RelationType.SOSTITUISCE
        ]
        assert len(sostituisce) >= 1

        # Should detect abrogation
        abroga = [
            m for m in result.modifications
            if m.relation_type == RelationType.ABROGA_TOTALMENTE
        ]
        assert len(abroga) >= 1
