"""
Tests for Graph Chunking Module
================================

Tests for:
- SourceType enum
- ChunkResult dataclass
- LegalChunker with source-type strategies
"""

import pytest
from datetime import datetime

from visualex.graph.chunking import (
    SourceType,
    ChunkResult,
    ChunkBatchResult,
    LegalChunker,
    ChunkingConfig,
    SOURCE_AUTHORITY,
)


# =============================================================================
# SourceType Tests
# =============================================================================


class TestSourceType:
    """Test suite for SourceType enum."""

    def test_source_type_values(self):
        """Test all source type values are defined."""
        assert SourceType.NORM.value == "norm"
        assert SourceType.JURISPRUDENCE.value == "jurisprudence"
        assert SourceType.COMMENTARY.value == "commentary"
        assert SourceType.DOCTRINE.value == "doctrine"

    def test_source_type_from_string(self):
        """Test creating source type from string."""
        assert SourceType("norm") == SourceType.NORM
        assert SourceType("commentary") == SourceType.COMMENTARY

    def test_source_authority_defaults(self):
        """Test default authority scores."""
        assert SOURCE_AUTHORITY[SourceType.NORM] == 1.0
        assert SOURCE_AUTHORITY[SourceType.JURISPRUDENCE] == 0.8
        assert SOURCE_AUTHORITY[SourceType.COMMENTARY] == 0.5
        assert SOURCE_AUTHORITY[SourceType.DOCTRINE] == 0.4


# =============================================================================
# ChunkResult Tests
# =============================================================================


class TestChunkResult:
    """Test suite for ChunkResult dataclass."""

    def test_chunk_result_creation(self):
        """Test creating a ChunkResult."""
        chunk = ChunkResult(
            chunk_id="test-123",
            text="Art. 1453 c.c. - La risoluzione del contratto...",
            source_type=SourceType.NORM,
            source_urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1453",
            source_authority=1.0,
            chunk_position=0,
            token_count=50,
        )

        assert chunk.chunk_id == "test-123"
        assert chunk.source_type == SourceType.NORM
        assert chunk.source_authority == 1.0
        assert chunk.token_count == 50

    def test_chunk_result_to_dict(self):
        """Test ChunkResult serialization."""
        chunk = ChunkResult(
            chunk_id="test-456",
            text="Test text",
            source_type=SourceType.COMMENTARY,
            source_urn="urn:test",
            source_authority=0.5,
            chunk_position=1,
            token_count=30,
            metadata={"section": "Spiegazione"},
        )

        d = chunk.to_dict()

        assert d["chunk_id"] == "test-456"
        assert d["source_type"] == "commentary"
        assert d["metadata"]["section"] == "Spiegazione"
        assert "created_at" in d

    def test_chunk_result_default_values(self):
        """Test ChunkResult default values."""
        chunk = ChunkResult(
            chunk_id="test",
            text="Text",
            source_type=SourceType.NORM,
            source_urn="urn:test",
            source_authority=1.0,
            chunk_position=0,
            token_count=10,
        )

        assert chunk.parent_article_urn is None
        assert chunk.metadata == {}
        assert chunk.created_at is not None


# =============================================================================
# LegalChunker Tests
# =============================================================================


class TestLegalChunker:
    """Test suite for LegalChunker."""

    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = LegalChunker()

    # --- AC1: Norm Chunking ---

    def test_chunk_norm_with_commi(self):
        """Test chunking norm text with numbered commi (AC1)."""
        text = """1. Il contratto può essere risolto per inadempimento.
2. La risoluzione opera di diritto quando il termine è essenziale.
3. La parte non inadempiente può recedere dal contratto."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:nir:stato:regio.decreto:1942-03-16;262~art1454",
            source_type=SourceType.NORM,
        )

        assert len(chunks) == 3
        assert chunks[0].chunk_position == 0
        assert "1." in chunks[0].text
        assert chunks[0].source_type == SourceType.NORM
        assert chunks[0].source_authority == 1.0

    def test_chunk_norm_respects_comma_structure(self):
        """Test that comma structure is preserved (AC1)."""
        text = """1. Primo comma con testo abbastanza lungo per essere significativo.
2. Secondo comma con altro contenuto importante per il test."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test:norm",
            source_type=SourceType.NORM,
        )

        assert len(chunks) == 2
        assert chunks[0].text.startswith("1.")
        assert chunks[1].text.startswith("2.")

    def test_chunk_norm_single_paragraph(self):
        """Test norm without numbered commi."""
        text = "Questo articolo non ha commi numerati ma contiene testo legale importante."

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test:norm:simple",
            source_type=SourceType.NORM,
        )

        assert len(chunks) >= 1
        assert chunks[0].source_type == SourceType.NORM

    # --- AC2: Commentary Chunking ---

    def test_chunk_commentary_with_massima(self):
        """Test chunking commentary with massima preserved (AC2)."""
        text = """Massima: La risoluzione del contratto per inadempimento richiede
la prova del nesso causale.

La dottrina dominante ritiene che...

Massima: Il termine essenziale si desume dalla natura dell'obbligazione."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:brocardi:art1453",
            source_type=SourceType.COMMENTARY,
        )

        # Should have at least the massime as separate chunks
        assert len(chunks) >= 2
        # Check that massime are preserved
        massima_chunks = [c for c in chunks if "Massima:" in c.text]
        assert len(massima_chunks) >= 1

    def test_chunk_commentary_paragraphs(self):
        """Test paragraph-level chunking for commentary."""
        text = """Primo paragrafo con spiegazione dettagliata.

Secondo paragrafo con ulteriori dettagli.

Terzo paragrafo di approfondimento."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:brocardi:commentary",
            source_type=SourceType.COMMENTARY,
        )

        # Should respect paragraph boundaries
        assert len(chunks) >= 1
        assert chunks[0].source_type == SourceType.COMMENTARY
        assert chunks[0].source_authority == 0.5

    # --- AC3: Source Type Metadata ---

    def test_chunk_includes_metadata(self):
        """Test that chunks include required metadata (AC3)."""
        text = "Testo normativo di esempio per il test."

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test:metadata",
            source_type=SourceType.NORM,
            extra_metadata={"libro": "IV", "titolo": "II"},
        )

        assert len(chunks) == 1
        chunk = chunks[0]

        # Check required metadata
        assert chunk.source_type == SourceType.NORM
        assert chunk.source_urn == "urn:test:metadata"
        assert chunk.source_authority == 1.0
        assert chunk.chunk_position == 0
        assert chunk.token_count > 0

        # Check extra metadata
        assert chunk.metadata["libro"] == "IV"
        assert chunk.metadata["titolo"] == "II"

    def test_chunk_authority_override(self):
        """Test overriding default authority score."""
        chunks = self.chunker.chunk_document(
            text="Test text",
            source_urn="urn:test",
            source_type=SourceType.COMMENTARY,
            source_authority=0.9,  # Override default 0.5
        )

        assert chunks[0].source_authority == 0.9

    def test_chunk_parent_article_urn(self):
        """Test parent article URN is set correctly."""
        chunks = self.chunker.chunk_document(
            text="Comma text",
            source_urn="urn:test:art1453-com1",
            source_type=SourceType.NORM,
            parent_article_urn="urn:test:art1453",
        )

        assert chunks[0].parent_article_urn == "urn:test:art1453"

    # --- AC4: Batch Processing ---

    def test_batch_chunking(self):
        """Test batch processing of documents (AC4)."""
        documents = [
            {
                "text": "1. Primo comma. 2. Secondo comma.",
                "source_urn": "urn:doc1",
                "source_type": SourceType.NORM,
            },
            {
                "text": "Commento al codice civile.",
                "source_urn": "urn:doc2",
                "source_type": SourceType.COMMENTARY,
            },
            {
                "text": "Sentenza della Cassazione.",
                "source_urn": "urn:doc3",
                "source_type": "jurisprudence",  # Test string conversion
            },
        ]

        result = self.chunker.chunk_batch(documents)

        assert result.total_documents == 3
        assert result.total_chunks >= 3
        assert len(result.failed_documents) == 0
        assert result.duration_seconds >= 0

    def test_batch_chunking_with_empty_text(self):
        """Test batch processing handles empty text gracefully (AC4)."""
        documents = [
            {
                "text": "Valid document text.",
                "source_urn": "urn:valid",
                "source_type": SourceType.NORM,
            },
            {
                "text": "",  # Empty produces no chunks but doesn't fail
                "source_urn": "urn:empty",
                "source_type": SourceType.NORM,
            },
        ]

        result = self.chunker.chunk_batch(documents)

        assert result.total_documents == 2
        # Empty text produces no chunks but doesn't fail
        assert len(result.failed_documents) == 0

    def test_batch_chunking_with_invalid_source_type(self):
        """Test batch processing isolates failures per document (AC4)."""
        documents = [
            {
                "text": "Valid document text.",
                "source_urn": "urn:valid",
                "source_type": SourceType.NORM,
            },
            {
                "text": "Another valid document.",
                "source_urn": "urn:invalid-type",
                "source_type": "not_a_valid_type",  # Invalid - will raise ValueError
            },
            {
                "text": "Third document is valid.",
                "source_urn": "urn:valid2",
                "source_type": SourceType.COMMENTARY,
            },
        ]

        result = self.chunker.chunk_batch(documents)

        # Should process 3 documents, with 1 failure isolated
        assert result.total_documents == 3
        assert len(result.failed_documents) == 1
        assert result.failed_documents[0]["source_urn"] == "urn:invalid-type"
        assert "not_a_valid_type" in result.failed_documents[0]["error"]
        # Valid documents still produced chunks
        assert result.total_chunks >= 2

    def test_batch_result_to_dict(self):
        """Test ChunkBatchResult serialization."""
        result = ChunkBatchResult(
            total_documents=10,
            total_chunks=50,
            duration_seconds=2.5,
        )

        d = result.to_dict()

        assert d["total_documents"] == 10
        assert d["total_chunks"] == 50
        assert d["failed_count"] == 0
        assert d["duration_seconds"] == 2.5

    # --- Edge Cases ---

    def test_chunk_empty_text(self):
        """Test handling of empty text."""
        chunks = self.chunker.chunk_document(
            text="",
            source_urn="urn:empty",
            source_type=SourceType.NORM,
        )

        assert chunks == []

    def test_chunk_whitespace_only(self):
        """Test handling of whitespace-only text."""
        chunks = self.chunker.chunk_document(
            text="   \n\n   \t   ",
            source_urn="urn:whitespace",
            source_type=SourceType.NORM,
        )

        assert chunks == []

    def test_chunk_very_long_text(self):
        """Test chunking of very long text with overlap."""
        # Create text with ~1000 words
        long_text = " ".join(["parola"] * 1000)

        chunks = self.chunker.chunk_document(
            text=long_text,
            source_urn="urn:long",
            source_type=SourceType.NORM,
        )

        # Should split into multiple chunks
        assert len(chunks) > 1
        # Check overlap exists (some words should appear in consecutive chunks)
        total_tokens = sum(c.token_count for c in chunks)
        # Due to overlap, total tokens should exceed original
        original_tokens = int(1000 * 1.3)  # 1000 words * 1.3 tokens/word
        assert total_tokens >= original_tokens

    def test_chunk_id_uniqueness(self):
        """Test that chunk IDs are unique."""
        text = "1. Primo comma. 2. Secondo comma. 3. Terzo comma."

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test",
            source_type=SourceType.NORM,
        )

        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_jurisprudence_chunking(self):
        """Test jurisprudence source type chunking."""
        text = """La Corte di Cassazione, Sezione II Civile, con sentenza n. 12345/2024,
ha stabilito il seguente principio di diritto:

"L'inadempimento contrattuale deve essere valutato in base alla gravità
dell'obbligazione violata."

In applicazione di tale principio, il ricorso è stato accolto."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:cassazione:sent:12345:2024",
            source_type=SourceType.JURISPRUDENCE,
        )

        assert len(chunks) >= 1
        assert chunks[0].source_type == SourceType.JURISPRUDENCE
        assert chunks[0].source_authority == 0.8

    def test_doctrine_chunking(self):
        """Test doctrine source type chunking."""
        text = """La teoria generale del contratto, come elaborata dalla dottrina,
individua quattro elementi essenziali: accordo, causa, oggetto e forma.

In particolare, l'accordo rappresenta l'incontro delle volontà delle parti."""

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:doctrine:test",
            source_type=SourceType.DOCTRINE,
        )

        assert len(chunks) >= 1
        assert chunks[0].source_type == SourceType.DOCTRINE
        assert chunks[0].source_authority == 0.4

    def test_token_estimation(self):
        """Test token count estimation."""
        # 10 words * 1.3 = 13 tokens (approx)
        text = "uno due tre quattro cinque sei sette otto nove dieci"

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test",
            source_type=SourceType.NORM,
        )

        assert chunks[0].token_count == 13  # 10 * 1.3 = 13

    def test_chunk_position_sequential(self):
        """Test chunk positions are sequential."""
        text = "1. Primo. 2. Secondo. 3. Terzo. 4. Quarto."

        chunks = self.chunker.chunk_document(
            text=text,
            source_urn="urn:test",
            source_type=SourceType.NORM,
        )

        positions = [c.chunk_position for c in chunks]
        assert positions == list(range(len(chunks)))
